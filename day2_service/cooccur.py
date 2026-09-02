"""物品共现:内容向量编码不了"看过《教父》的人也爱看《好家伙》"这类信息。

两部片子文本不像,但同一批人会看。这个信号只在用户行为里,而且不用训练模型,
数一遍就行:两个物品被同一个人看过就有关系,被同一批人看过的次数越多关系越强。

    权重 w(i,j) = c(i,j) / sqrt(c(i) * c(j))

那个归一化分母不是可选项。不除的话热门物品和谁都共现,会霸占所有人的推荐。

**怎么存进 Milvus:**打分公式本身就是一个内积——

    物品 i 的 cf 向量 = {j: w(i,j)}     j 是和 i 共现过的物品
    查询向量          = {h: 1.0}        h 是这个用户最近 K 项历史
    两者点积          = Σ w(i,h)        这个用户的历史和物品 i 的关联强度总和

所以"按 CF 分取 top-K"和"向量检索"是同一件事,一个向量字段就能承载。

所以用 SPARSE_FLOAT_VECTOR + SPARSE_INVERTED_INDEX:查询只有 5 个非零维,
物品各有 50 个,倒排索引正是为这种形状设计的,而且没有维度上限——
百万物品的目录一样装得下。

**一条实测限制:**Milvus Lite 3.2.0 的稀疏字段 metric_type=IP 没有严格生效,
打分带了 BM25 的 tf 饱和和长度归一(合成用例:内积同为 3.0 的两个向量
返回 0.9738 和 0.6416)。端到端的代价很小——HR@10 0.181 vs 精确内积 0.188,
在一个标准误内,因为 BM25 在共现权重上近似单调。官方文档写的是稀疏向量只支持 IP,
所以这大概率是 Lite 的实现落后,集群上要自己验一次。

物品 id 到稀疏向量下标用 crc32 取模 2^31,不维护映射表:下标空间足够大撞不上,
而且无状态——换机器、重建库都不用同步一份 id→index 的表。

**--holdout-last**:建表时把每人历史的最后一项一起剔掉,生成评测专用的无泄漏表。
不剔的话,留一法藏的正是那一项,而共现表在统计的时候已经见过它了。
"""
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zlib import crc32

DATA = Path(__file__).parent / "data"
USERS = os.getenv("USERS_CSV", "users_loo.csv")
PRODUCTS = os.getenv("PRODUCTS_CSV", "products_loo.csv")
CF_FILE = Path(os.getenv("CF_FILE", DATA / "cooccur_loo_noleak.json"))
TOPK = int(os.getenv("CF_TOPK", "50"))            # 每个物品保留多少个共现邻居
MAX_HIST = int(os.getenv("CF_MAX_HIST", "50"))    # 见下面 build() 的注释
# 只统计时间差 >= 这么多秒的物品对。0 = 不过滤。
# MovieLens 的评分是批量录入的（被藏那一项距前一项的时间差中位数只有 17 秒），
# 表里大量边其实来自"同一次填表"。如果产品位置是"隔天再来的首页推荐"，
# 这些边帮不上忙，设成 3600 更合适；"继续浏览"这类当次会话内的位置就不用过滤。
MIN_GAP = int(os.getenv("CF_MIN_GAP", "0"))

_table = None


def index_of(item_id):
    """物品 id → 稀疏向量下标。2^31 是 Milvus 稀疏向量下标的上限。"""
    return crc32(str(item_id).encode()) % (2 ** 31)


def table():
    global _table
    if _table is None:
        _table = json.loads(CF_FILE.read_text(encoding="utf-8")) if CF_FILE.exists() else {}
    return _table


def item_vector(item_id):
    """物品侧的 cf 向量:{邻居下标: 权重}。

    没有共现数据的物品(新物品,或者表还没建)放它自己,不写空向量。
    实测 Milvus 接受空稀疏向量,但它在 IP 下得分恒为 0、行为不好预测;
    放它自己的话至少写入和检索都是确定的——召不回别人,冷启动交给语义那一路兜。
    """
    nb = table().get(str(item_id))
    if not nb:
        return {index_of(item_id): 1.0}
    return {index_of(j): float(w) for j, w in nb.items()}


def query_vector(history):
    """查询侧的 cf 向量:用户看过的每个物品那一维给 1.0,和物品侧点积正好是 Σ w(i,h)。

    **传进来的应该是最近 K 项,不是全部历史。**这一路和内容路共用 store.HISTORY_K,
    切片在 store.recommend 里做。实测差别很大(500 个用户):

        CF 查询用最近 5 项    HR@10 0.194
        CF 查询用最近 20 项   HR@10 0.142
        CF 查询用全部历史     HR@10 0.090

    原因是 Σ 会把"这个人这些年看过的一切"加在一起,而和什么都共现的正是全局热门物品
    ——历史越长,这一路就越退化成一个更贵的热门推荐。

    去重也是必须的:同一个物品出现两次会让它的权重翻倍。
    """
    idx = {}
    for h in dict.fromkeys(i for i in history if i):
        idx[index_of(h)] = 1.0
    return idx or {0: 1.0}


def signature():
    """cf 表进不进指纹?不进。

    共现只影响 cf 那个字段的值,不影响 dense 向量的语义空间,换一份表不需要重灌
    整个集合(虽然要重灌 cf 字段)。这里留一个函数是为了 verify_seed.py 能打印它。
    """
    gap = f":gap{MIN_GAP}" if MIN_GAP else ""
    return f"cf:{CF_FILE.name}:{len(table())}:top{TOPK}{gap}"


def _sequences(drop_last=True):
    """每个用户的历史。MIN_GAP > 0 时要带时间戳,所以回原始 ratings.dat 取。

    users_loo.csv 只存了 id 序列,时间信息在预处理时就丢了。不带过滤时读 csv 更快,
    带过滤时才去读原始文件——一次多花几秒,换来"能按预测视野裁边"这个能力。
    """
    keep = {r["id"] for r in csv.DictReader(open(DATA / PRODUCTS, encoding="utf-8"))}
    if not MIN_GAP:
        for row in csv.DictReader(open(DATA / USERS, encoding="utf-8")):
            h = [i for i in row["history"].split("|") if i in keep]
            yield [(0, m) for m in (h[:-1] if drop_last else h)]
        return
    from collections import defaultdict as _dd
    seqs = _dd(list)
    for line in open(DATA / "ml-1m" / "ratings.dat", encoding="latin-1"):
        u, m, _, t = line.strip().split("::")
        if "m" + m in keep:
            seqs["u" + u].append((int(t), "m" + m))
    for u in seqs:
        v = sorted(seqs[u])
        yield v[:-1] if drop_last else v


def build(drop_last=True):
    """从历史算共现。

    MAX_HIST:全量协议下有人历史长到 2277 项,两两配对就是 260 万对——一个重度用户
    就能把整张表填满,而这种"什么都看过"的用户贡献的共现信号本身最杂。
    实测截到最近 50 项比截到 200 项还好(HR@10 0.193 vs 0.178),建表也快得多。

    MIN_GAP:只统计时间差 >= MIN_GAP 秒的物品对,过滤掉"同一次填表"产生的边。
    """
    c_pair, c_item = defaultdict(Counter), Counter()
    users = 0
    for seq in _sequences(drop_last):
        sn = set()
        seq = [(t, m) for t, m in seq if not (m in sn or sn.add(m))][-MAX_HIST:]
        if len(seq) < 2:
            continue
        users += 1
        c_item.update(m for _, m in seq)
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                (ta, x), (tb, y) = seq[a], seq[b]
                if MIN_GAP and abs(tb - ta) < MIN_GAP:
                    continue
                c_pair[x][y] += 1
                c_pair[y][x] += 1

    raw = sum(len(v) for v in c_pair.values())
    out, edges = {}, 0
    for i, nb in c_pair.items():
        # w(i,j) = c(i,j)/sqrt(c(i)*c(j))：不除的话热门物品和谁都共现，会霸占所有人的推荐
        scored = {j: n / math.sqrt(c_item[i] * c_item[j]) for j, n in nb.items()}
        top = sorted(scored.items(), key=lambda kv: -kv[1])[:TOPK]
        out[i] = {j: round(w, 6) for j, w in top}
        edges += len(top)

    CF_FILE.write_text(json.dumps(out), encoding="utf-8")
    print(f"{CF_FILE.name}: {len(out)} 个物品,{edges} 条边(每个物品最多留 {TOPK} 个邻居)")
    print(f"  共现对总数 {raw},被裁掉 {raw - edges} 条")
    print(f"  {users} 个用户参与统计"
          f"{'(已剔掉每人最后一项:无泄漏,评测用)' if drop_last else '(全量:生产用)'}"
          f"{f',只统计时间差 >= {MIN_GAP}s 的物品对' if MIN_GAP else ''}")
    return out


if __name__ == "__main__":
    drop = "--holdout-last" in sys.argv
    CF_FILE = Path(os.getenv("CF_FILE", DATA / (
        "cooccur_loo_noleak.json" if drop else "cooccur_loo.json")))
    build(drop_last=drop)
