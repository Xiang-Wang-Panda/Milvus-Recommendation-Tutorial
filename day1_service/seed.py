"""灌数据 + 冒烟测试。先起服务，再跑 python seed.py [base_url] [用户数上限]。

和 Day 0 的唯一区别：每个请求带 X-Tenant-Id（TENANT 环境变量，默认 default）。
灌完自动跑一次 compact——BM25 的 idf 按 segment 统计，不合并的话刚灌的批次排序不可比。
"""
import csv
import os
import sys
import time
from pathlib import Path

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MAX_USERS = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = 全灌
TENANT = os.getenv("TENANT", "default")
DATA = Path(__file__).parent / "data"
# 换数据集用：Day 2 那份 products_loo.csv / users_loo.csv 数据用全，
# 是正文 0.091 那个口径。默认还是 Day 0 的简化版。
PRODUCTS = os.getenv("PRODUCTS_CSV", "products.csv")
USERS = os.getenv("USERS_CSV", "users.csv")
H = {"X-Tenant-Id": TENANT}


def post(path, data):
    r = requests.post(BASE + path, json=data, headers=H, timeout=600)
    r.raise_for_status()
    return r.json()


def rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def batch(path, data, size=500):
    t = time.time()
    for i in range(0, len(data), size):
        post(path, data[i:i + size])
        print(f"\r  {path} {min(i + size, len(data))}/{len(data)}", end="", flush=True)
    print(f"\r  {path} <- {len(data)} 条，{time.time() - t:.1f}s")


def items():
    out = []
    for r in rows(PRODUCTS):
        # year 仍然转 int，但 Day 1 就算忘了转也不会静默出错：year 是显式 INT64 字段，
        # "1999" 会被 pydantic 转成 1999（实测过，'year >= 1999' 照样筛得到），
        # "abc" 直接 422。Day 0 那个"存成字符串导致数值过滤失效"的坑犯不了了
        out.append({"id": r["id"], "text": r["text"], "category": r["category"],
                    **({"year": int(r["year"])} if r["year"] else {})})
    return out


def users():
    out = [{"user_id": r["user_id"], "history": r["history"].split("|"), "prefer": r["prefer"]}
           for r in rows(USERS)]
    return out[:MAX_USERS] if MAX_USERS else out


def show(title, hits, fmt):
    print(f"\n[{title}]")
    for h in hits:
        print("  " + fmt(h))


if __name__ == "__main__":
    print(f"灌数据到租户 {TENANT}（商品塔要先灌完，用户兴趣向量是从商品向量算的）")
    batch("/items", items())
    batch("/users", users(), size=200)
    print("  compact:", post("/admin/compact", None))
    print(" ", requests.get(BASE + "/health", headers=H, timeout=60).json())

    show("搜索 BM25: star wars", post("/search", {"q": "star wars", "top_k": 5}),
         lambda h: f"{h['score']:6.2f}  {h['text']}")
    show("搜索 vector: space adventure",
         post("/search", {"q": "space adventure", "top_k": 5, "mode": "vector"}),
         lambda h: f"{h['score']:6.3f}  {h['text']}")
    show("搜索 BM25 + filter: comedy 且 year>=1995（现在走 INVERTED 标量索引）",
         post("/search", {"q": "love", "top_k": 5, "filter": 'category == "Comedy" and year >= 1995'}),
         lambda h: f"{h['score']:6.2f}  {h['text']}")

    uid = users()[0]["user_id"]
    show(f"推荐 {uid}（已排除历史）", post("/recommend", {"user_id": uid, "top_k": 5}),
         lambda h: f"{h['score']:6.3f}  {h['text']}")
    show(f"相似用户 {uid}",
         requests.post(f"{BASE}/similar-users?user_id={uid}&top_k=5", headers=H, timeout=120).json(),
         lambda h: f"{h['user_id']:>8}  {h['prefer']:<12} {h['score']:.3f}")

    p = requests.get(f"{BASE}/users/{uid}", headers=H, timeout=60).json()
    show(f"画像 {uid}（prefer={p['prefer']}）", p["history"][:5], lambda h: h["text"])
