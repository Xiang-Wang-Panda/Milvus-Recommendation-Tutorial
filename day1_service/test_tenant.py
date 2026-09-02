"""租户隔离的越权测试。跑法：先起服务，再 python test_tenant.py [base_url]

这不是功能测试，是安全边界测试。partition key 只管数据分布，不管谁能看谁——
少拼一句 tenant_id，Milvus 会一声不吭地把别人的数据返回给你。所以这里每一条断言
都要求"确实拿不到"，而不是"这次恰好没查到"：先在 A 里写一条只有 A 有的数据，
再用 B 的身份去要它。
"""
import sys
import uuid

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
A, B = f"tenant-a-{uuid.uuid4().hex[:6]}", f"tenant-b-{uuid.uuid4().hex[:6]}"
MARK = f"zzqqxx{uuid.uuid4().hex[:8]}"  # 不可能撞上真数据的词，BM25 一搜就该只有它
ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- 实际 ' + str(detail)[:120]}")


def call(method, path, tenant, **kw):
    return requests.request(method, BASE + path, headers={"X-Tenant-Id": tenant}, timeout=120, **kw)


def search(tenant, q, **extra):
    r = call("post", "/search", tenant, json={"q": q, "top_k": 10, "strong": True, **extra})
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    print(f"A={A}  B={B}  哨兵词={MARK}\n")

    call("post", "/items", A, json=[{"id": f"{A}-x1", "text": f"{MARK} secret movie",
                                     "category": "Secret", "year": 2024}]).raise_for_status()
    call("post", "/users", A, json=[{"user_id": f"{A}-u1", "history": [f"{A}-x1"],
                                     "prefer": "Secret"}]).raise_for_status()
    call("post", "/items", B, json=[{"id": f"{B}-y1", "text": "ordinary movie",
                                     "category": "Plain", "year": 2024}]).raise_for_status()

    print("1. 数据确实写进去了（A 自己拿得到，不然后面的 PASS 都是假的）")
    check("A 搜哨兵词命中自己的数据", any(h["id"] == f"{A}-x1" for h in search(A, MARK)))
    # 这一条专门防空过：A 的用户是"写完 item 立刻回查"那条路径建出来的，一致性没配对
    # 就会被 store.add_users 静默跳过（实测集群 Bounded 下只有 2/10 读得到刚写的行）。
    # 用户根本不存在的话，下面"B 读 A 的用户 → 404"是因为它不存在才过的，测不到隔离。
    r = call("get", f"/users/{A}-u1", A)
    check("A 自己读得到自己的用户（否则下面那条 404 是空过）", r.status_code == 200, r.status_code)

    print("2. B 拿不到 A 的东西")
    check("B 搜同一个哨兵词 → 空", search(B, MARK) == [], search(B, MARK))
    check("B 向量搜 → 不含 A 的 id",
          all(not h["id"].startswith(A) for h in search(B, MARK, mode="vector")))
    check("B 带 filter 绕 → 不含 A 的 id",
          all(not h["id"].startswith(A)
              for h in search(B, "movie", filter='category == "Secret"')))
    r = call("get", f"/users/{A}-u1", B)
    check("B 直接按 id 读 A 的用户 → 404", r.status_code == 404, r.status_code)
    r = call("post", "/recommend", B, json={"item_ids": [f"{A}-x1"], "top_k": 5})
    check("B 拿 A 的 item_id 做无状态推荐 → 400（在 B 里这个 id 不存在）",
          r.status_code == 400, r.status_code)
    check("B 的 sample 里没有 A 的用户",
          all(not u["user_id"].startswith(A)
              for u in call("get", "/users/sample?n=100", B).json()))
    check("B 的 count 不含 A 的数据",
          call("get", "/health", B).json()["items"] == 1,
          call("get", "/health", B).json()["items"])

    print("3. 构造出来的租户名进不去 filter 表达式")
    for bad in ['a" or id != "', "a' or 1==1", "a and tenant_id != \"\"", "", "x" * 65]:
        r = call("post", "/search", bad, json={"q": MARK, "top_k": 10})
        check(f"tenant_id={bad[:24]!r} → 400", r.status_code == 400, r.status_code)

    print(f"\n{ok} passed, {fail} failed")
    sys.exit(1 if fail else 0)
