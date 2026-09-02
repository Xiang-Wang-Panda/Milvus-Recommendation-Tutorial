"""内容一路的基线核对:证明这个库的内容召回路确实是 0.091 那个口径。

为什么不直接跑 eval.py:eval.py 每个用户走一次 Milvus 往返,1500 个用户要几十分钟。
而 FLAT 是精确检索,所以一次性把全部物品向量取出来在 numpy 里算全量打分,
结果和过 Milvus 完全等价(实测对到小数点后三位),几十秒就出来。

它同时核对四件事——少任何一件,内容一路就不是 0.091 那个口径:

    1. dense 的 metric 是 IP(不是 COSINE)
    2. 物品向量的模长 == log(1+pop)^α          ← 先验真的乘进去了
    3. 查询向量是"最近 HISTORY_K 项各自归一化后的均值"
    4. 热度榜是无泄漏那份(建表时剔掉了每人历史的最后一项)

用法(MILVUS_URI 指向哪个库就核对哪个服务):
    MILVUS_URI=../day1_service/milvus.db python base_check.py
"""
import argparse
import csv
import logging
import math
import os
import pathlib
import random
from pathlib import Path

logging.disable(logging.ERROR)

import numpy as np  # noqa: E402

import popularity  # noqa: E402
import store  # noqa: E402

DATA = Path(__file__).parent / "data"
PRODUCTS = os.getenv("PRODUCTS_CSV", "products_loo.csv")
USERS = os.getenv("USERS_CSV", "users_loo.csv")
TENANT = os.getenv("TENANT", "default")
K, SEED, NEG = 10, 42, -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1500)
    a = ap.parse_args()

    store._load("items")
    # Day 0 的 schema 没有 tenant_id（租户是 Day 1 才加的），所以这里不能无脑拼 scope()。
    # 核对脚本要能对着三个服务的库跑，不然"Day 0 也是这个口径"只是个说法。
    has_tenant = "tenant_id" in store._fields("items")
    where = (lambda f: store.scope(TENANT, f)) if has_tenant else (lambda f: f)
    items = [r["id"] for r in csv.DictReader(open(DATA / PRODUCTS, encoding="utf-8"))]
    idx = {i: k for k, i in enumerate(items)}
    dim = store._dim("items")

    print(f"库 {store.URI}")
    try:
        sig = store._stored_signature()
    except Exception:
        sig = None                       # Day 0 的指纹在库旁边的文件里，不在 properties
    if sig is None:
        f = pathlib.Path(store.URI + ".embedding")
        sig = f.read_text(encoding="utf-8").strip() if f.exists() else "（没有指纹记录）"
    print(f"  指纹 {sig}   租户字段 {'有' if has_tenant else '无（Day 0 的 schema）'}")
    print(f"  metric {popularity.METRIC}   POP_ALPHA {popularity.ALPHA:g}   "
          f"HISTORY_K {store.HISTORY_K}   热度榜 {popularity.POP_FILE.name}")

    # ---- 取全部向量 ----
    V = np.zeros((len(items), dim), dtype=np.float32)
    got = 0
    for s in range(0, len(items), 500):
        part = items[s:s + 500]
        for r in store.client.query("items", filter=where(store._in(part)),
                                    output_fields=["id", "dense"], limit=len(part)):
            V[idx[r["id"]]] = r["dense"]
            got += 1
    assert got == len(items), f"只取到 {got}/{len(items)} 个向量——库和 csv 对不上"

    # ---- 核对 2:模长就是先验权重 ----
    norms = np.linalg.norm(V, axis=1)
    want = np.array([popularity.weight(i) for i in items], dtype=np.float32)
    dev = float(np.abs(norms - want).max())
    print(f"  模长核对:最大偏差 {dev:.2e}  "
          f"(模长范围 {norms.min():.3f}~{norms.max():.3f})"
          f"  {'✓' if dev < 1e-3 else '✗ 先验没乘进去'}")

    U = V / (norms[:, None] + 1e-9)      # 单位方向，查询侧用它

    # ---- 取样，和 eval.py 同一套（同一个种子、同一个门槛）----
    rows = [r for r in csv.DictReader(open(DATA / USERS, encoding="utf-8"))
            if len([i for i in r["history"].split("|") if i]) >= 5]
    random.Random(SEED).shuffle(rows)
    rows = rows[:a.users] if a.users else rows

    Q, T, M = [], [], []
    for r in rows:
        h = [i for i in r["history"].split("|") if i]
        target, seen = h[-1], h[:-1]
        recent = list(dict.fromkeys(seen))[-store.HISTORY_K:] if store.HISTORY_K \
            else list(dict.fromkeys(seen))
        ri = [idx[i] for i in recent if i in idx]
        if not ri or target not in idx:
            continue
        q = U[ri].mean(axis=0)           # 核对 3：各自归一化后再平均
        Q.append(q / (np.linalg.norm(q) + 1e-9))
        T.append(idx[target])
        m = np.zeros(len(items), dtype=bool)
        for i in seen:
            if i in idx:
                m[idx[i]] = True
        M.append(m)
    Q, T, M = np.asarray(Q, np.float32), np.asarray(T), np.asarray(M)
    n = len(T)

    P = np.array([popularity.table().get(i, 0) for i in items], dtype=np.float32)
    rank_pop = (-P).argsort().argsort() + 1

    def run(name, S):
        S = np.where(M, NEG, S)                       # 看过的不能再推
        top = np.argpartition(-S, K, axis=1)[:, :K]
        o = np.take_along_axis(S, top, 1).argsort(1)[:, ::-1]
        top = np.take_along_axis(top, o, 1)
        hit = (top == T[:, None])
        hr = float(hit.any(1).mean())
        pos = np.where(hit.any(1), hit.argmax(1) + 1, 0)
        nd = float(np.where(pos > 0, 1.0 / np.log2(np.maximum(pos, 1) + 1), 0.0).mean())
        se = math.sqrt(max(hr * (1 - hr), 0) / n)
        print(f"  {name:<26} HR@{K} {hr:.3f} ±{se:.3f}   NDCG@{K} {nd:.3f}"
              f"   召回的流行度排名中位数 {int(np.median(rank_pop[top]))}/{len(items)}")
        return hr

    print(f"\nn={n} 个用户\n=== 对照组 ===")
    hot = np.tile(np.log1p(P), (n, 1))
    base = run("推最热门", hot)
    rnd = np.random.default_rng(SEED).random((n, len(items))).astype(np.float32)
    run("随机推", rnd)

    print("=== 内容一路 ===")
    cos = Q @ U.T                                    # 纯语义，α 的作用被剥掉
    run("纯内容（α=0 等效）", cos)
    hr = run("内容 + 流行度先验", Q @ V.T)            # 核对 1：IP = w(i) × cos

    print(f"\n=== 判定 ===")
    print(f"  基线 {base:.3f} → 内容一路 {hr:.3f}"
          f"（{hr / base:.2f}x，相对提升 {(hr / base - 1) * 100:+.0f}%）")
    print(f"  {'✓' if abs(hr - 0.091) < 0.012 else '⚠'} 目标口径 0.091，实测 {hr:.3f}")


if __name__ == "__main__":
    main()
