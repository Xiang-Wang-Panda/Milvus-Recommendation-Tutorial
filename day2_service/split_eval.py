"""三种离线评测口径,同一套向量各评一遍 —— 因为"藏哪一项"决定了你在量什么。

    A 标准留一法      藏每人时间序的最后一项
                     和论文可比(SASRec / BERT4Rec 都用这个),但在 ML-1M 上
                     那一项和前几项是同一次打分会话(间隔中位数 17 秒),
                     所以它量的是"补全这一批",不是"预测下次"

    B 跨会话留一法    藏"距上一项 >= SESSION_GAP"的项,即新会话的第一项
                     更接近首页要回答的问题

    C 全局时间切分    某个时刻 T 之前的全部数据建索引,T 之后的交互当正样本
                     **工业界的离线口径就是这个**:1..N 天训练,N+1 天测试。
                     它是唯一一个"索引里绝不可能有未来信息"的口径 ——
                     A 和 B 都是按用户切的,共现表里塞满了别人的未来。

三个都报,因为它们各自回答不同的问题;只报一个就会有人拿它去承诺线上。

用法:
    python split_eval.py            # 三个口径都跑
    python split_eval.py --only C
"""
import argparse
import csv
import logging
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

logging.disable(logging.ERROR)

import numpy as np  # noqa: E402

import store  # noqa: E402

DATA = Path(__file__).parent / "data"
PRODUCTS = os.getenv("PRODUCTS_CSV", "products_loo.csv")
TENANT = os.getenv("TENANT", "default")
SESSION_GAP = int(os.getenv("SESSION_GAP", "3600"))   # 隔多久算新会话
CF_TOPK = int(os.getenv("CF_TOPK", "50"))
CF_MAX_HIST = int(os.getenv("CF_MAX_HIST", "50"))
QK = int(os.getenv("HISTORY_K", "5"))
KS = (10, 50, 200)
SEED = 42


def load():
    """带时间戳的完整交互。留一法那份 csv 丢了时间信息,这里回原始文件拿。"""
    prod = {r["id"]: r for r in csv.DictReader(open(DATA / PRODUCTS, encoding="utf-8"))}
    seqs = defaultdict(list)
    for line in open(DATA / "ml-1m" / "ratings.dat", encoding="latin-1"):
        u, m, _, t = line.strip().split("::")
        if "m" + m in prod:
            seqs["u" + u].append((int(t), "m" + m))
    for u in seqs:
        seqs[u].sort()
    return prod, seqs


def vectors(items):
    """一次取完全部物品向量。FLAT 是精确检索,numpy 复算和过 Milvus 等价。"""
    store._load("items")
    idx = {i: k for k, i in enumerate(items)}
    V = np.zeros((len(items), store._dim("items")), dtype=np.float32)
    got = 0
    for s in range(0, len(items), 500):
        part = items[s:s + 500]
        for r in store.client.query("items", filter=store.scope(TENANT, store._in(part)),
                                    output_fields=["id", "dense"], limit=len(part)):
            V[idx[r["id"]]] = r["dense"]
            got += 1
    assert got == len(items), f"只取到 {got}/{len(items)} 个向量"
    return V, V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9), idx


def build_cf(contrib, min_gap=0):
    """contrib: {user: [(t, item)]} —— **只喂允许被索引看见的那部分交互**。

    min_gap:只统计时间差 >= min_gap 秒的物品对。这个参数要和预测视野对齐 ——
    不过滤的话表里塞满"同一次打分被一起录入"的边,它们对"下次看什么"没有预测力,
    还会把长间隔的边从 top-K 名额里挤出去。实测跨会话口径下:
        不过滤 0.064(打不赢推热门)  →  剔掉 <1 小时的边 0.116(1.71x)
    """
    cp, ci = defaultdict(Counter), Counter()
    for seq in contrib.values():
        sn = set()
        seq = [(t, m) for t, m in seq if not (m in sn or sn.add(m))][-CF_MAX_HIST:]
        if len(seq) < 2:
            continue
        ci.update(m for _, m in seq)
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                (ta, x), (tb, y) = seq[a], seq[b]
                if min_gap and abs(tb - ta) < min_gap:
                    continue
                cp[x][y] += 1
                cp[y][x] += 1
    out = {}
    for i, nb in cp.items():
        sc = {j: n / math.sqrt(ci[i] * ci[j]) for j, n in nb.items()}
        out[i] = dict(sorted(sc.items(), key=lambda kv: -kv[1])[:CF_TOPK])
    return out


def build_pop(contrib):
    c = Counter()
    for seq in contrib.values():
        c.update(m for _, m in seq)
    return c


def evaluate(cases, items, idx, V, U, CF, pop, label):
    """cases: [(history, positives)]。positives 是集合 —— 口径 C 有多个正样本。"""
    N = len(items)
    logP = np.array([math.log1p(pop.get(i, 0)) for i in items], dtype=np.float32)
    hot_order = [items[j] for j in np.argsort(-logP)]
    rnd = np.random.default_rng(SEED)
    maxk = max(KS)
    acc = {m: {k: 0 for k in KS} for m in ("随机推", "推最热门", "内容一路", "协同过滤", "融合")}
    n = 0
    for hist, pos in cases:
        seen = set(hist)
        q = hist[-QK:]
        ri = [idx[i] for i in q if i in idx]
        if not ri or not pos:
            continue
        n += 1
        # 内容一路
        qv = U[ri].mean(0)
        qv /= np.linalg.norm(qv) + 1e-9
        sc = V @ qv
        for i in seen:
            if i in idx:
                sc[idx[i]] = -1e9
        ctop = [items[j] for j in np.argsort(-sc)[:maxk]]
        # 协同过滤
        cs = Counter()
        for x in q:
            for j, w in CF.get(x, {}).items():
                cs[j] += w
        for x in seen:
            cs.pop(x, None)
        ftop = [j for j, _ in cs.most_common(maxk)]
        # 融合：等权 RRF，和 store.py 里 RRFRanker(60) 一致
        s = Counter()
        for lst, w in ((ctop, 1.0), (ftop, 1.0)):
            for r, x in enumerate(lst, 1):
                s[x] += w / (60 + r)
        btop = [x for x, _ in s.most_common(maxk)]

        lists = {"随机推": list(rnd.choice([i for i in items if i not in seen],
                                        min(maxk, N - len(seen)), replace=False)),
                 "推最热门": [i for i in hot_order if i not in seen][:maxk],
                 "内容一路": ctop, "协同过滤": ftop, "融合": btop}
        for m, lst in lists.items():
            for k in KS:
                if pos & set(lst[:k]):
                    acc[m][k] += 1

    print(f"\n=== {label} ===  测试点 {n}")
    print(f"  {'':10}" + "".join(f"{'HR@'+str(k):>12}" for k in KS))
    base = acc["推最热门"][10] / n if n else 0
    for m in ("随机推", "推最热门", "内容一路", "协同过滤", "融合"):
        row = "".join(f"{acc[m][k]/n:>12.3f}" for k in KS)
        r = acc[m][10] / n
        se = math.sqrt(max(r * (1 - r), 0) / n) if n else 0
        mark = "" if m in ("随机推", "推最热门") else f"   {r/base:>4.2f}x 基线" if base else ""
        print(f"  {m:<10}{row}   ±{se:.3f}{mark}")
    return acc, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="ABC 里挑，如 --only C")
    ap.add_argument("--cf-min-gap", type=int, default=None,
                    help="共现表只统计间隔 >= 这么多秒的对。默认：口径 A 不过滤，B/C 用 SESSION_GAP")
    a = ap.parse_args()

    prod, seqs = load()
    items = list(prod)
    V, U, idx = vectors(items)
    print(f"物品 {len(items)}   用户 {len(seqs)}   "
          f"交互 {sum(len(v) for v in seqs.values())}")
    print(f"指纹 {store._stored_signature()}   HISTORY_K={QK}   "
          f"新会话阈值 {SESSION_GAP}s")

    run = (a.only or "ABC").upper()

    # ---------- A 标准留一法 ----------
    if "A" in run:
        contrib = {u: v[:-1] for u, v in seqs.items() if len(v) >= 2}   # 剔掉每人最后一项
        gap = a.cf_min_gap if a.cf_min_gap is not None else 0
        CF, pop = build_cf(contrib, gap), build_pop(contrib)
        cases = []
        for u, v in seqs.items():
            if len(v) < QK + 1:
                continue
            h = list(dict.fromkeys(m for _, m in v[:-1]))
            if len(h) >= QK:
                cases.append((h, {v[-1][1]}))
        evaluate(cases, items, idx, V, U, CF, pop,
                 f"A 标准留一法（藏最后一项，共现表{'不过滤' if not gap else f'剔<{gap}s'}）")

    # ---------- B 跨会话留一法 ----------
    if "B" in run:
        contrib = {u: v[:-1] for u, v in seqs.items() if len(v) >= 2}
        gap = a.cf_min_gap if a.cf_min_gap is not None else SESSION_GAP
        CF, pop = build_cf(contrib, gap), build_pop(contrib)
        cases = []
        for u, v in seqs.items():
            pts = [k for k in range(QK, len(v)) if v[k][0] - v[k - 1][0] >= SESSION_GAP]
            for k in pts[-10:]:                       # 每人最多 10 个点，别让重度用户主导
                h = list(dict.fromkeys(m for _, m in v[:k]))
                if len(h) >= QK:
                    cases.append((h, {v[k][1]}))
        random.Random(SEED).shuffle(cases)
        evaluate(cases[:20000], items, idx, V, U, CF, pop,
                 f"B 跨会话留一法（藏新会话首项，共现表剔<{gap}s 的边）")

    # ---------- C 全局时间切分 ----------
    if "C" in run:
        alltimes = sorted(t for v in seqs.values() for t, _ in v)
        T = alltimes[int(len(alltimes) * 0.9)]         # 最后 10% 的时间当测试期
        contrib = {u: [(t, m) for t, m in v if t < T] for u, v in seqs.items()}
        contrib = {u: v for u, v in contrib.items() if v}
        gap = a.cf_min_gap if a.cf_min_gap is not None else SESSION_GAP
        CF, pop = build_cf(contrib, gap), build_pop(contrib)
        cases = []
        for u, v in seqs.items():
            past = [m for t, m in v if t < T]
            future = {m for t, m in v if t >= T}
            h = list(dict.fromkeys(past))
            future -= set(h)                          # 已经看过的不算新正样本
            if len(h) >= QK and future:
                cases.append((h, future))
        import datetime
        print(f"\n  切分点 T = {datetime.datetime.fromtimestamp(T, datetime.UTC):%Y-%m-%d}"
              f"（T 之前 {sum(len(v) for v in contrib.values())} 条交互建索引）")
        print(f"  有测试正样本的用户 {len(cases)}，"
              f"人均正样本 {sum(len(p) for _, p in cases)/max(len(cases),1):.1f} 个")
        evaluate(cases, items, idx, V, U, CF, pop,
                 f"C 全局时间切分（T 之后当正样本，共现表剔<{gap}s 的边）")
        print("\n  注:C 是多正样本，HR@k 的含义是"
              "「top-k 里至少命中一个他后来真的看了的」，所以数值天然比 A/B 高。"
              "\n  它和 A/B 不可直接比大小，但它是三个里唯一"
              "「索引里不含任何未来信息」的口径。")


if __name__ == "__main__":
    main()
