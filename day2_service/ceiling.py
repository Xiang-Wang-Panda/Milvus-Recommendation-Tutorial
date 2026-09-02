"""召回天花板:精排只能重排召回给的那 N 个,所以有硬上限。

    精排后的 HR@10  ≤  召回的 HR@N

**为精排花第一次 API 调用之前,先跑这个。**如果 HR@50 和 HR@10 差不多,
那就是召回压根没把答案捞进来,再聪明的精排也变不出来,直接省下这笔钱。

反过来不成立:天花板高只说明"有空间",不说明"模型能拿到"。它是个单向指标。

用法:
    python ceiling.py --users 1500 --mode multi
"""
import argparse
import logging
import math
import os

logging.disable(logging.ERROR)

import eval as ev  # noqa: E402  复用取样和指标，评测口径必须只有一份
import store  # noqa: E402

TENANT = os.getenv("TENANT", "default")
DEPTHS = [10, 20, 50, 100, 200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--mode", default="multi", choices=["content", "cf", "multi"])
    a = ap.parse_args()

    store._load("items")
    rows = ev.users(a.users)
    maxd = max(DEPTHS)
    hit = {d: 0 for d in DEPTHS}
    n = 0

    for r in rows:
        h = [i for i in r["history"].split("|") if i]
        target, seen = h[-1], h[:-1]
        vec = store.query_vector(TENANT, seen) if a.mode != "cf" else None
        hits = store.recommend(TENANT, vec, maxd, seen, mode=a.mode, history=seen)
        ids = [x["id"] for x in hits]
        n += 1
        if target in ids:
            pos = ids.index(target) + 1
            for d in DEPTHS:
                if pos <= d:
                    hit[d] += 1

    se = lambda p: math.sqrt(max(p * (1 - p), 0) / n) if n else 0
    print(f"召回天花板  mode={a.mode}  n={n}\n")
    base = hit[10] / n
    for d in DEPTHS:
        p = hit[d] / n
        print(f"  HR@{d:<4} {p:.3f} ±{se(p):.3f}"
              + (f"   可捞回空间 {p - base:+.3f}" if d > 10 else "   ← 现在的分数"))

    top50 = hit[50] / n
    print(f"\n  精排如果取 top-50,上限是 {top50:.3f},当前 {base:.3f},账面空间 {top50 - base:+.3f}。")
    if top50 - base < 0.02:
        print("  空间太小,别花这笔钱——问题在召回,不在排序。")
    else:
        print("  空间够大,可以花钱试。但记住这只是上限,不是预期收益。")


if __name__ == "__main__":
    main()
