"""量 LLM 精排能把 HR@10 拉高多少。

**三条纪律,少一条数字就不能信:**

  1. 模型看不到答案。它只看到用户历史(去掉留一藏起的那一项)和候选列表,
     正确答案和其他 N-1 个长得一样,没有任何标记。
  2. 候选顺序打乱后再给模型。否则它可能只是跟着召回顺序抄,而召回顺序本身
     已经是个不错的排序——那样量出来的是"抄得准不准",不是"排得准不准"。
  3. 模型漏掉的候选按召回原序接在后面,保证集合不变,只比顺序。

**检验要用配对的。**两个方法跑的是同一批用户,所以不能用两样本 z 检验
(那会高估方差)。HR@10 用 McNemar:只看"一个中了另一个没中"的那些用户,
所以它能直接告诉你精排救回了几个、弄丢了几个。NDCG@10 用配对自助法给区间。

结果缓存到 json,重跑不重复付费。改了提示词记得删缓存。

用法:
    RERANK=1 python rerank_eval.py --users 200 --mode multi --topn 50
"""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

logging.disable(logging.ERROR)

import eval as ev  # noqa: E402  复用取样和指标，评测口径只有一份
import rerank  # noqa: E402
import store  # noqa: E402

TENANT = os.getenv("TENANT", "default")
K = 10
CACHE = Path(__file__).parent / "data" / "rerank_cache.json"
# 每 1M token 的价格，用来估这次实验花了多少
PRICE = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (3.0, 15.0),
         "claude-haiku-4-5": (1.0, 5.0)}


def load_cache():
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def mcnemar(a, b):
    """配对的 HR 检验。a/b 是等长的 0/1 列表。

    返回 (b 救回的个数, b 弄丢的个数, p)。只有这两个不一致的格子进检验——
    两边都中或都不中的用户不含区分两个方法的信息。
    """
    win = sum(1 for x, y in zip(a, b) if not x and y)     # 精排救回来的
    lose = sum(1 for x, y in zip(a, b) if x and not y)    # 精排弄丢的
    n = win + lose
    if n == 0:
        return 0, 0, 1.0
    # 精确二项检验（双侧），n 小的时候比卡方近似靠得住
    tail = sum(math.comb(n, k) for k in range(0, min(win, lose) + 1)) / (2 ** n)
    return win, lose, min(1.0, 2 * tail)


def boot_ci(a, b, rounds=5000, seed=42):
    """配对自助法给 NDCG 差值的 95% 区间。重采样的是用户,不是单个指标值。"""
    rnd = random.Random(seed)
    n = len(a)
    diffs = []
    idx = range(n)
    for _ in range(rounds):
        pick = [rnd.randrange(n) for _ in idx]
        diffs.append(sum(b[i] - a[i] for i in pick) / n)
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--mode", default="multi", choices=["content", "cf", "multi"])
    ap.add_argument("--topn", type=int, default=50, help="精排多少个候选")
    a = ap.parse_args()

    if not rerank.ENABLED:
        raise SystemExit("要先打开 RERANK=1（并设好 ANTHROPIC_API_KEY）")

    store._load("items")
    rows = ev.users(a.users)
    cache = load_cache()
    key_pre = f"{rerank.MODEL}|{a.mode}|{a.topn}|"

    print(f"精排评测  模型 {rerank.MODEL}  召回路 {a.mode}  top-{a.topn}  n={len(rows)}")
    print(f"  缓存 {CACHE.name}：已有 {len(cache)} 条\n")

    fresh = [r for r in rows if key_pre + r["user_id"] not in cache]
    if fresh:
        print(f"  {len(fresh)} 个用户要新调模型，先预热缓存 …")
        rerank.warm_up()

    base_hit, rr_hit, base_nd, rr_nd = [], [], [], []
    usage = {"in": 0, "out": 0, "cache_write": 0, "cache_read": 0}
    rnd = random.Random(ev.SEED)

    for n_done, r in enumerate(rows, 1):
        h = [i for i in r["history"].split("|") if i]
        target, seen = h[-1], h[:-1]
        vec = store.query_vector(TENANT, seen) if a.mode != "cf" else None
        hits = store.recommend(TENANT, vec, a.topn, seen, mode=a.mode, history=seen)
        if not hits:
            continue
        cands = [{"id": x["id"], "text": x["text"]} for x in hits]

        key = key_pre + r["user_id"]
        if key in cache:
            order = cache[key]
        else:
            # 纪律 2：顺序打乱后再给模型
            shuffled = cands[:]
            rnd.shuffle(shuffled)
            # 纪律 1：历史里已经去掉了 target，候选里 target 和其他 N-1 个长得一样
            hist_texts = [x["text"] for x in store.get_items(TENANT, seen[-store.HISTORY_K:])]
            ranked, u = rerank.rerank(hist_texts, shuffled, K)
            if u is None:
                continue                       # 降级了，这个用户不计入
            order = [x["id"] for x in ranked]
            cache[key] = order
            for k in usage:
                usage[k] += u[k]
            if n_done % 10 == 0:
                CACHE.write_text(json.dumps(cache), encoding="utf-8")
                print(f"\r  {n_done}/{len(rows)}", end="", flush=True)

        def score(ids):
            top = ids[:K]
            if target not in top:
                return 0, 0.0
            return 1, 1.0 / math.log2(top.index(target) + 2)

        b_h, b_n = score([x["id"] for x in cands])
        r_h, r_n = score(order)
        base_hit.append(b_h); base_nd.append(b_n)
        rr_hit.append(r_h); rr_nd.append(r_n)

    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    n = len(base_hit)
    if not n:
        raise SystemExit("一个用户都没跑成功（全部降级了？检查 ANTHROPIC_API_KEY）")

    hr0, hr1 = sum(base_hit) / n, sum(rr_hit) / n
    nd0, nd1 = sum(base_nd) / n, sum(rr_nd) / n
    win, lose, pv = mcnemar(base_hit, rr_hit)
    lo, hi = boot_ci(base_nd, rr_nd)

    print(f"\r{'':30}\r")
    print(f"  召回原序(top-{a.topn} 的前 10)   HR@{K} {hr0:.3f} ±{ev.se(hr0, n):.3f}   "
          f"NDCG@{K} {nd0:.3f}")
    print(f"  LLM 精排后                  HR@{K} {hr1:.3f} ±{ev.se(hr1, n):.3f}   "
          f"NDCG@{K} {nd1:.3f}")
    print(f"\n=== 配对检验(n={n})===")
    print(f"  McNemar：精排救回 {win} 个、弄丢 {lose} 个，p={pv:.3f}"
          f" → {'分得开' if pv < 0.05 else '分不开'}")
    print(f"  NDCG 差值 {nd1 - nd0:+.4f}，95% 配对自助区间 [{lo:+.4f}, {hi:+.4f}]"
          f" → {'不含 0' if lo * hi > 0 else '含 0，分不开'}")

    if any(usage.values()):
        pin, pout = PRICE.get(rerank.MODEL, (5.0, 25.0))
        cost = (usage["in"] + usage["cache_write"] * 1.25 + usage["cache_read"] * 0.1) / 1e6 * pin \
            + usage["out"] / 1e6 * pout
        print(f"\n  这次新调用：输入 {usage['in']:,} / 输出 {usage['out']:,} token"
              f"，缓存写 {usage['cache_write']:,} 读 {usage['cache_read']:,}")
        print(f"  约 ${cost:.2f}（缓存已经算进去了）")
    else:
        print("\n  全部命中本地缓存，这次没花钱。")


if __name__ == "__main__":
    main()
