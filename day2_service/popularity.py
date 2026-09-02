"""流行度先验:把 P(item) 烘进向量模长,用 Milvus 的 IP 代替 COSINE。

内容一路只估了 P(u|i)(文本匹配度),把 P(i) 当成均匀分布——默认 3416 部电影
被看的概率都一样。实测这个假设错得很厉害:内容路 top-10 的流行度排名中位数在
1500/3416 附近,一直在长尾里捞;而正确答案的中位数在 553/3416。
它找对了类型(儿童动画),找错了档位(只有几十人看过)。

**怎么在 Milvus 里做,而不是在应用层做后处理:**

    存进去的向量 = 单位向量 × w(i)     w(i) = log(1+pop(i))^α
    查询向量     = 单位向量
    IP 内积      = w(i) × cos(i, q)   ← 正好是要的打分

所以只要把 dense 字段的 metric 从 COSINE 换成 IP,先验就变成了向量模长的一部分。
hybrid_search、group_by、filter 全部照常工作,没有应用层重排,也不用多一次往返。

α 的取法:实测 0.25–0.35 是一段平台(不是尖点,所以不敏感)。α=0 退回纯 COSINE。

**评测口径要剔掉留一藏的那一项**(默认就是这样),否则热度榜见过答案。
生产环境用 --full 建全量榜:线上没有"测试集"这回事。
"""
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
POP_FILE = Path(os.getenv("POP_FILE", DATA / "popularity_noleak.json"))
ALPHA = float(os.getenv("POP_ALPHA", "0.25"))   # 0 = 关闭(dense 退回 COSINE)
CAP = float(os.getenv("POP_CAP", "0")) or None  # 先把 pop 截到上限再取 log,压住超级热门

ENABLED = ALPHA > 0
METRIC = "IP" if ENABLED else "COSINE"

_table = None


def table():
    global _table
    if _table is None:
        _table = json.loads(POP_FILE.read_text(encoding="utf-8")) if POP_FILE.exists() else {}
    return _table


def weight(item_id):
    """向量要乘的模长。

    没有热度数据的物品(新物品、或者热度榜还没建)权重给 1.0,不被压也不被抬。
    这一行不是洁癖:log1p(0)**α == 0,不特判的话新物品会被乘成零向量,
    然后 Milvus 要么拒收要么让它永远召不回来,而且一声不吭。
    """
    if not ENABLED:
        return 1.0
    c = float(table().get(str(item_id), 0))
    if c <= 0:
        return 1.0
    if CAP:
        c = min(c, CAP)
    return math.log1p(c) ** ALPHA


def scale(vec, item_id):
    w = weight(item_id)
    return vec if w == 1.0 else [x * w for x in vec]


def signature():
    """α 要进 embedding 指纹:同一个模型、不同 α 是两套向量,混在一个集合里会静默变差,
    而且 metric 也不一样(IP vs COSINE)。表的大小也进去——换了热度榜就是换了向量。"""
    if not ENABLED:
        return ""
    cap = f":cap{int(CAP)}" if CAP else ""
    return f"+pop:a{ALPHA:g}{cap}:{len(table())}"


def build(users_csv, out=POP_FILE, drop_last=True):
    """每部电影被多少个用户看过。

    drop_last:剔掉每个用户历史的最后一项——留一法藏的正是那一项。
    实测影响很小(热度榜是几千人的合计,单个用户进不进去不影响排名),但这是免费的。
    """
    c = Counter()
    with open(users_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = [i for i in row["history"].split("|") if i]
            c.update(h[:-1] if drop_last else h)
    out.write_text(json.dumps(c), encoding="utf-8")
    v = sorted(c.values())
    print(f"{out.name}: {len(c)} 个物品有热度数据")
    print(f"  最热 {v[-1]} 人、中位数 {v[len(v) // 2]} 人、最冷 {v[0]} 人")
    print(f"  {'(已剔掉每人最后一项:无泄漏,评测用)' if drop_last else '(全量:生产用)'}")
    return c


if __name__ == "__main__":
    import sys

    full = "--full" in sys.argv
    users = DATA / os.getenv("USERS_CSV", "users_loo.csv")
    build(users, out=DATA / ("popularity.json" if full else "popularity_noleak.json"),
          drop_last=not full)
