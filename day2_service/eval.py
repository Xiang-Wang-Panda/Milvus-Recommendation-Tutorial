"""留一法评测。**这一步必须放第一位:不做完,后面所有优化都是盲调。**

召回质量看不出来——你只能看到首页那 10 条,看不到它漏掉了什么。

标注不用人工标:用户历史本身就是标注,藏起最后一项就行。

    用户 u783 的历史(按时间排):
      … Pete's Dragon → Fox and the Hound → American Tail → Space Jam
                                                                ↑ 藏起来
    用前面的历史算查询向量 → 推 10 个 → 看 Space Jam 在不在里面

**基线取"推最热门"**,不取"我上一版的做法"。一个跑不赢无脑推热门的推荐系统,
不管内部多复杂,都没有提供个性化价值;而拿上一版当分母,分母越差涨幅越好看。

三个子命令:
    rec     留一法评测召回质量(主战场)
    search  拿物品标题当查询做已知条目检索,对比 bm25 / vector / hybrid
    group   同一份推荐开/不开 group_by,相关性和多样性一起打印

不调 store.init():评测脚本不该有删数据的能力。新进程连已存在的库时集合是
released 的,所以自己 load 一次。

**所有召回都走 store.recommend_batch()。**逐用户查是 1500 次往返,实测一轮要半小时;
批量之后是几十次往返,一轮几十秒。评测慢到不敢跑,就等于没有评测。
"""
import argparse
import csv
import logging
import math
import os
import random
from collections import Counter
from pathlib import Path

logging.disable(logging.ERROR)   # pymilvus 的 load 提示刷屏

import cooccur  # noqa: E402
import popularity  # noqa: E402
import store  # noqa: E402

DATA = Path(__file__).parent / "data"
PRODUCTS = os.getenv("PRODUCTS_CSV", "products_loo.csv")
USERS = os.getenv("USERS_CSV", "users_loo.csv")
TENANT = os.getenv("TENANT", "default")
K = 10
SEED = 42


# ---------- 取样 ----------
def users(n, min_hist=5):
    rows = [r for r in csv.DictReader(open(DATA / USERS, encoding="utf-8"))
            if len([i for i in r["history"].split("|") if i]) >= min_hist]
    random.Random(SEED).shuffle(rows)
    return rows[:n] if n else rows


def items():
    return list(csv.DictReader(open(DATA / PRODUCTS, encoding="utf-8")))


# ---------- 指标 ----------
def se(p, n):
    """标准误。n 翻 4 倍误差才减半——所以别指望多跑一点样本就能把 0.01 的差分开。"""
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0


class Metric:
    def __init__(self):
        self.n = self.hit = 0
        self.ndcg = 0.0

    def add(self, ranked, target):
        self.n += 1
        if target in ranked:
            pos = ranked.index(target) + 1
            self.hit += 1
            # NDCG@10：命中在第 1 位得 1.0，第 10 位得 0.29。
            # HR 只问"中没中"，NDCG 还问"排得够不够靠前"。
            self.ndcg += 1.0 / math.log2(pos + 1)

    @property
    def hr(self):
        return self.hit / self.n if self.n else 0.0

    def line(self, name):
        return (f"  {name:<28} HR@{K} {self.hr:.3f} ±{se(self.hr, self.n):.3f}   "
                f"NDCG@{K} {self.ndcg / self.n if self.n else 0:.3f}   n={self.n}")


def two_prop_z(p1, p2, n):
    """两个比例的 z 检验。用未四舍五入的命中数算,不然 0.045 和 0.0454 会得出不同结论。"""
    x1, x2 = round(p1 * n), round(p2 * n)
    p = (x1 + x2) / (2 * n)
    s = math.sqrt(p * (1 - p) * 2 / n) if 0 < p < 1 else 0.0
    if not s:
        return 0.0, 1.0
    z = (p1 - p2) / s
    return z, math.erfc(abs(z) / math.sqrt(2))


# ---------- rec ----------
def cmd_rec(args):
    store._load("items")
    rows = users(args.users)
    pool = items()
    all_ids = [r["id"] for r in pool]
    print(f"留一法评测:{len(rows)} 个用户,{len(all_ids)} 个候选物品")
    print(f"  embedding {store._signature()}  index {store.INDEX_TYPE}  "
          f"metric {popularity.METRIC}  HISTORY_K {store.HISTORY_K}")
    print(f"  {cooccur.signature()}\n")

    # ---- 两个对照组，必须打印在最前面 ----
    pop = popularity.table()
    hot = [i for i, _ in sorted(pop.items(), key=lambda kv: -kv[1]) if i in set(all_ids)]
    rnd = random.Random(SEED)

    m_hot, m_rnd = Metric(), Metric()
    for r in rows:
        h = [i for i in r["history"].split("|") if i]
        target, seen = h[-1], set(h[:-1])
        m_hot.add([i for i in hot if i not in seen][:K], target)
        m_rnd.add(rnd.sample([i for i in all_ids if i not in seen], K), target)

    print("=== 对照组(基线取推最热门)===")
    print(m_hot.line("对照组:推最热门"))
    print(m_rnd.line("对照组:随机推"))
    base = m_hot.hr

    # ---- 各条召回路 ----
    configs = [("内容一路", dict(mode="content")),
               ("协同过滤一路", dict(mode="cf")),
               ("内容 + 协同过滤(RRF)", dict(mode="multi"))]
    if args.only:
        configs = [c for c in configs if args.only in c[0]]

    # 一次把所有用户的历史和答案摊平，查询向量批量算（物品向量去重后一次取完）
    hist = [[i for i in r["history"].split("|") if i][:-1] for r in rows]
    target = [[i for i in r["history"].split("|") if i][-1] for r in rows]
    vecs = store.query_vectors(TENANT, hist)

    print("\n=== 召回路 ===")
    results = []
    for name, kw in configs:
        batch = store.recommend_batch(TENANT, list(zip(vecs, hist)), K, **kw)
        m = Metric()
        for hits, t in zip(batch, target):
            m.add([x["id"] for x in hits], t)
        print(m.line(name))
        results.append((name, m))

    # ---- 结尾自己判定 ----
    if results:
        best_name, best = max(results, key=lambda t: t[1].hr)
        gap = se(best.hr, best.n) + se(base, m_hot.n)
        z, pv = two_prop_z(best.hr, base, best.n)
        print(f"\n=== 判定 ===")
        print(f"  最好的是「{best_name}」{best.hr:.3f},基线 {base:.3f}")
        if base >= 0.999:
            print("  基线已经满分:这个评测集饱和了,换个更难的查询集再比。")
        elif best.hr - base <= gap:
            print(f"  差 {best.hr - base:+.3f} 没超过噪声门槛 {gap:.3f}(两个标准误之和)"
                  f" —— 这一档分不开,别拿它选型。")
        else:
            print(f"  差 {best.hr - base:+.3f} 超过噪声门槛 {gap:.3f},"
                  f"z={z:+.2f} p={pv:.4f} —— 分得开。")


# ---------- search ----------
def cmd_search(args):
    store._load("items")
    pool = items()
    random.Random(SEED).shuffle(pool)
    pool = pool[:args.users] if args.users else pool
    print(f"已知条目检索:{len(pool)} 个物品,拿标题当查询\n")
    for mode in ("bm25", "vector", "hybrid"):
        m = Metric()
        for s in range(0, len(pool), 128):        # 搜索侧同样批量
            part = pool[s:s + 128]
            for hits, r in zip(store.search_batch(TENANT, [x["text"] for x in part],
                                                  K, mode=mode), part):
                m.add([h["id"] for h in hits], r["id"])
        print(m.line(mode))
    print("\n  注:这是 BM25 的主场(查询和文档字面几乎一致),三种模式都接近满分。"
          "\n  这个评测集是饱和的,量不出 hybrid 在困难查询上的收益。")


# ---------- group ----------
def cmd_group(args):
    store._load("items")
    rows = users(args.users)
    cat = {r["id"]: r["category"] for r in items()}
    year = {r["id"]: int(r["year"]) for r in items() if r["year"]}
    print(f"分组去重:{len(rows)} 个用户,mode=multi\n")

    hist = [[i for i in r["history"].split("|") if i][:-1] for r in rows]
    target_of = [[i for i in r["history"].split("|") if i][-1] for r in rows]
    vecs = store.query_vectors(TENANT, hist)

    for label, kw in (("不分组", {}), ("按 category 分组", dict(group_by="category"))):
        batch = store.recommend_batch(TENANT, list(zip(vecs, hist)), K, mode="multi", **kw)
        m = Metric()
        ncat, dom, yrs, span = [], [], [], []
        for hits, target in zip(batch, target_of):
            ids = [x["id"] for x in hits]
            m.add(ids, target)
            if not ids:
                continue
            ncat.append(len({cat.get(i) for i in ids}))
            # "系列霸屏"：top-10 里标题前两个词相同的条目最多重复几次
            pre = Counter(" ".join(x["text"].split()[:2]) for x in hits)
            dom.append(pre.most_common(1)[0][1])
            ys = [year[i] for i in ids if i in year]
            if ys:
                yrs.append(sum(ys) / len(ys))
                span.append(max(ys) - min(ys))
        avg = lambda v: sum(v) / len(v) if v else 0
        print(m.line(label))
        print(f"      不同 category 数 {avg(ncat):.2f}   系列霸屏 {avg(dom):.2f}   "
              f"平均年份 {avg(yrs):.1f}   年份跨度 {avg(span):.1f}")

    print("\n  注:HR@10 只有一个正确答案,这个指标结构性地惩罚多样性——"
          "\n  任何把 top-10 让给不同类型的动作都会挤掉更可能命中的候选。"
          "\n  用它衡量分组是用错了尺子,要评判得上线做 A/B。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rec", "search", "group"])
    ap.add_argument("--users", type=int, default=500, help="0 = 全部")
    ap.add_argument("--only", default="", help="按名字子串只跑某个配置")
    a = ap.parse_args()
    {"rec": cmd_rec, "search": cmd_search, "group": cmd_group}[a.cmd](a)
