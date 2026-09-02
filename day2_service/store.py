"""Milvus 存储层（Day 2）：items（商品塔）+ users（用户塔）。

Day 1 的四件事照旧：连集群、索引可调、租户隔离、一致性可选。
Day 2 只动召回层，业务函数签名只多了 mode / group_by 两个可选参数：

  1. items 多一个 cf 字段（SPARSE_FLOAT_VECTOR），存物品共现
  2. mode=multi / hybrid 走 Milvus 原生 hybrid_search + RRFRanker，融合在服务端做
  3. dense 的 metric 由 popularity 决定：α>0 时先验烘进向量模长，metric 换成 IP
  4. group_by / group_size 透传成 Milvus 的分组去重

八个接口的语义一个没变，加的都是召回层的开关。
"""
import json
import os
import re

# 必须在 import pymilvus 之前取走：pymilvus 自己也读 MILVUS_URI，且只认 http(s)://，
# 留在环境里会让 ./milvus.db 这种本地路径在 import 阶段就报 Illegal uri。
# 上了集群这个冲突反而消失了（值本来就是 http 地址），但本地跑还得留着这一行。
URI = os.environ.pop("MILVUS_URI", "./milvus.db")

from pymilvus import DataType, Function, FunctionType, MilvusClient  # noqa: E402
from pymilvus.exceptions import DataNotMatchException  # noqa: E402

import cooccur  # noqa: E402
import embedding  # noqa: E402
import popularity  # noqa: E402

CLUSTER = URI.startswith("http")


def _credentials():
    """本地文件不认证；http(s) 才带凭据。
    Zilliz Cloud 发的是一整串 token；自建集群一般是 user/password（默认 root/Milvus）。
    MILVUS_DB 是库级隔离用的——一租户一 database 时每个实例连自己那个库。"""
    if not CLUSTER:
        return {}
    out = {}
    if os.getenv("MILVUS_TOKEN"):
        out["token"] = os.environ["MILVUS_TOKEN"]
    elif os.getenv("MILVUS_USER"):
        out["user"] = os.environ["MILVUS_USER"]
        out["password"] = os.getenv("MILVUS_PASSWORD", "")
    if os.getenv("MILVUS_DB"):
        out["db_name"] = os.environ["MILVUS_DB"]
    return out


client = MilvusClient(uri=URI, **_credentials())

ANALYZER = os.getenv("TEXT_ANALYZER", "standard")  # chinese / english / standard
CONSISTENCY = os.getenv("MILVUS_CONSISTENCY", "Bounded")  # Strong/Bounded/Session/Eventually
REPLICAS = int(os.getenv("MILVUS_REPLICAS", "1"))
PARTITIONS = int(os.getenv("MILVUS_TENANT_PARTITIONS", "16"))  # partition key 的桶数
INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "AUTOINDEX")  # AUTOINDEX/HNSW/IVF_FLAT/FLAT

# Day 2 的召回层旋钮
HISTORY_K = int(os.getenv("HISTORY_K", "5"))          # 查询向量只用最近 K 项历史，0 = 全部
RRF_K = int(os.getenv("RRF_K", "60"))                 # RRF 的 k，调它收益远小于多加一路召回
RECALL_FANOUT = int(os.getenv("RECALL_FANOUT", "4"))  # 每一路召 top_k × FANOUT 再融合
# 单次召回的 limit 上限。全量协议下一个人的历史可能有两千多项，limit 得够大才能排除完
# 还剩 top_k，但 Milvus 对 limit 有硬上限（16384），顶到就报 "limit out of range"。
MAX_RECALL = int(os.getenv("MAX_RECALL", "4096"))

# 建索引和查询是两组独立旋钮：M/efConstruction/nlist 建完就固定了，ef/nprobe 每次查询可调。
# 具体填多少别拍脑袋，跑 bench.py index 看自己数据上的延迟-召回曲线。
_BUILD = {"HNSW": {"M": 16, "efConstruction": 200}, "IVF_FLAT": {"nlist": 1024}}
_QUERY = {"HNSW": {"ef": 64}, "IVF_FLAT": {"nprobe": 16}}
INDEX_PARAMS = {**_BUILD.get(INDEX_TYPE, {}), **json.loads(os.getenv("MILVUS_INDEX_PARAMS") or "{}")}
SEARCH_PARAMS = {**_QUERY.get(INDEX_TYPE, {}), **json.loads(os.getenv("MILVUS_SEARCH_PARAMS") or "{}")}

# BM25 的旋钮：k1 是词频饱和（一个词出现 8 次值不值 8 倍），b 是长度归一（长文档该罚多少）。
# SPARSE_INVERTED_INDEX 没有 HNSW 那种延迟-召回取舍，稀疏索引这边能调的就是这两个 + analyzer。
# 空着就用 Milvus 默认（k1=1.2, b=0.75）。填法：MILVUS_BM25_PARAMS='{"bm25_k1":1.5,"bm25_b":0.5}'
#
# **别用 describe_index 验证它生效了。** 实测集群会把你传的任何键原样存下来再回显给你，
# 包括 bogus_param——回显不代表它被读了。唯一的判据是分数变不变，实测（3 个文档，查 "star"）：
#   {} 默认              doc1 0.2771  doc2(长) 0.0778  doc3 0.2191
#   {"bogus_param": 1}   完全一致 —— 假参数确实被忽略
#   k1=0.1 b=0.0         0.1451 / 0.1335 / 0.1335 —— b=0 不做长度归一，长文追平短文
#   k1=3.0 b=1.0         0.4963 / 0.0577 / 0.4705 —— 长度惩罚拉满，长文被压到最后
BM25_PARAMS = json.loads(os.getenv("MILVUS_BM25_PARAMS") or "{}")

# 跟下面 init() 建的 schema 对齐：字段对不上就重建那一个集合
ITEMS_FIELDS = {"id", "tenant_id", "text", "category", "year", "sparse", "dense", "cf"}
USERS_FIELDS = {"user_id", "tenant_id", "dense", "history", "prefer"}

PROP_EMBED = "app.embedding.signature"  # 指纹存在 items 的 collection properties 里


def _signature():
    """向量的完整指纹。**α 必须进去**：同一个模型、不同 α 是两套向量，
    维度还一样，所以维度检查发现不了；混在一个集合里只会静默变差。
    而且 α>0 时 metric 是 IP、α=0 时是 COSINE，索引本身也不一样。"""
    return embedding.signature() + popularity.signature() + cooccur.signature()

# Milvus Lite 只内置 standard 一种 analyzer type，中英文各自拼出来：
# chinese 走 jieba tokenizer（需 pip install jieba），english 是 standard + 小写/停用词/词干
ANALYZER_PARAMS = {
    "standard": {"type": "standard"},
    "chinese": {"tokenizer": "jieba"},
    "english": {"tokenizer": "standard",
                "filter": ["lowercase",
                           {"type": "stop", "stop_words": ["_english_"]},
                           {"type": "stemmer", "language": "english"}]},
}

_TENANT = re.compile(r"^[A-Za-z0-9_.:@-]{1,64}$")


def scope(tenant, filter=""):
    """把 tenant_id 拼进 filter。**这是安全边界，不是功能。**

    partition key 保证的是数据怎么分布，不是谁能看谁——忘了拼这一句，A 租户就能搜到
    B 租户的数据，而 Milvus 一声不吭地照常返回。所以所有读路径统一从这里出 filter，
    没有第二个地方允许自己拼 tenant_id。

    白名单也不是洁癖：tenant 是外部传进来的字符串，直接进 filter 表达式就是注入面
    （一个引号就能把 `tenant_id == "x"` 闭合掉，后面接 `or id != ""` 全库就出来了）。
    """
    if not _TENANT.match(tenant or ""):
        raise ValueError(f"非法 tenant_id {tenant!r}：只允许 [A-Za-z0-9_.:@-]，1-64 位")
    own = f'tenant_id == "{tenant}"'
    return f"{own} and ({filter})" if filter else own


def init():
    dim = embedding.dim()
    if client.has_collection("items") and _embedding_changed(dim):
        # 向量本身变了，两个塔的数据都作废（要重新跑 seed.py）
        client.drop_collection("items")
        if client.has_collection("users"):
            client.drop_collection("users")
    else:
        # 字段增删过（Day 1 给两个集合都加了 tenant_id，items 还多了 category/year）。
        # Milvus 不接受 schema 外的字段，写入会直接失败，所以对不上就地重建那一个——
        # 但别连坐：items 的向量还好着的时候没必要把 users 也删了。
        for name, want in (("items", ITEMS_FIELDS), ("users", USERS_FIELDS)):
            if client.has_collection(name) and _fields(name) != want:
                print(f"{name} schema 变了：{sorted(_fields(name))} -> {sorted(want)}，重建 {name}")
                client.drop_collection(name)

    if not client.has_collection("items"):
        # enable_dynamic_field 留着：低频/不稳定的字段还是原样收下，只是 category/year
        # 这两个高频过滤字段提成了显式字段（见下面 add_index 的注释）
        s = client.create_schema(enable_dynamic_field=True, num_partitions=PARTITIONS)
        s.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        s.add_field("tenant_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
        s.add_field("text", DataType.VARCHAR, max_length=4096,
                    enable_analyzer=True,
                    analyzer_params=ANALYZER_PARAMS.get(ANALYZER, {"type": ANALYZER}))
        # nullable：products.csv 里有一批电影没年份。不给 nullable 的话这些行写不进去，
        # 补个 0 又会污染 year >= 1995 这种过滤（0 也是个数）。null 参与不了比较，正好。
        s.add_field("category", DataType.VARCHAR, max_length=64, nullable=True)
        s.add_field("year", DataType.INT64, nullable=True)
        s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        s.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
        # Day 2：物品共现。和上面那个 sparse 共用同一套稀疏向量机制——查询只有 5 个
        # 非零维、物品各有 50 个，倒排索引正是为这种形状设计的，而且没有维度上限。
        # 注意 Milvus Lite 3.2.0 上稀疏字段的 metric_type=IP 没有严格生效（打分带了
        # BM25 的 tf 饱和），端到端代价 HR@10 0.181 vs 精确内积 0.188，在噪声内。
        s.add_field("cf", DataType.SPARSE_FLOAT_VECTOR)
        # sparse 不自己算：Milvus 原生 BM25 Function 从 text 生成，写入/查询两端都走它
        s.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                input_field_names=["text"], output_field_names=["sparse"]))
        idx = client.prepare_index_params()
        idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                      params=BM25_PARAMS)
        # metric 由 popularity 决定：α>0 时先验烘进了向量模长，必须用 IP 才能把它算进去
        # （COSINE 会把模长归一化掉，先验就白加了）。α=0 时退回 COSINE。
        idx.add_index(field_name="dense", index_type=INDEX_TYPE, metric_type=popularity.METRIC,
                      params=INDEX_PARAMS)
        idx.add_index(field_name="cf", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        # 标量索引：category 是等值过滤，year 是范围过滤，两个都用 INVERTED。
        # year 看着更像 STL_SORT 的活，但 milvus-lite 3.2 只实现了 INVERTED
        # （填 STL_SORT 会报 "scalar index_type 'STL_SORT' is not implemented"），
        # 而 INVERTED 本身也支持范围，所以两边统一成 INVERTED，本地和集群同一套代码。
        idx.add_index(field_name="category", index_type="INVERTED")
        idx.add_index(field_name="year", index_type="INVERTED")
        _create("items", s, idx)
        _remember(_signature())

    if not client.has_collection("users"):
        s = client.create_schema(num_partitions=PARTITIONS)
        s.add_field("user_id", DataType.VARCHAR, is_primary=True, max_length=64)
        s.add_field("tenant_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
        s.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)  # 兴趣向量：历史物品向量的均值
        # 全量协议下一个人的历史能到两千多项，8192 装不下，写入会静默截断
        s.add_field("history", DataType.VARCHAR, max_length=65535)
        s.add_field("prefer", DataType.VARCHAR, max_length=128)
        idx = client.prepare_index_params()
        # 必须和 items 用同一个 metric：mean() 里物品向量已经带了流行度先验的模长，
        # users 的兴趣向量是它们的均值，查询走 _dense_params() 也是 popularity.METRIC——
        # 索引建成 COSINE 的话，α>0 时这里会跟 items 错开，real 集群直接报
        # "metric type not match"（Milvus Lite 不校验，本地测不出这个坑）。
        idx.add_index(field_name="dense", index_type=INDEX_TYPE, metric_type=popularity.METRIC,
                      params=INDEX_PARAMS)
        _create("users", s, idx)

    _load("items")
    _load("users")
    if not _stored_signature():
        # Day 0 迁上来的库没有这条 property（指纹当时在旁边的文件里）。补一次，
        # 这一次没法判断向量是不是真的对得上，但下一次换模型就能发现了。
        _remember(_signature())


def _load(name):
    """新进程连上已存在的库时集合是 released 状态，不显式 load 第一次 query 就报
    "call load() before search/get/query"。Day 0 加这一行是为了不报错；到了集群上它是个
    内存管理动作：load 把数据从对象存储拉进 query node 的内存，replica_number 决定拉几份
    （多副本分担读、也做故障转移，代价是内存翻倍）。单机 standalone 填 >1 没有意义。"""
    client.load_collection(name, **({"replica_number": REPLICAS} if REPLICAS > 1 else {}))


def _create(name, schema, idx):
    """milvus-lite 建集合是先 os.makedirs(exist_ok=False) 再写 schema.json，中间被打断就会留下
    一个目录：has_collection 只认 schema.json 所以看不见它，makedirs 又因为目录在而永远失败，
    服务再也起不来。这里把那个 FileExistsError 翻译成能直接照做的一句话。"""
    try:
        client.create_collection(name, schema=schema, index_params=idx)
    except Exception as e:
        if "File exists" not in str(e):
            raise
        raise RuntimeError(
            f"集合 {name} 只建了一半（上次启动建到一半被中断，缺 schema.json）。"
            f"删掉这个目录再启动：rm -rf {URI}/collections/{name}") from e


def _remember(sig):
    """embedding 指纹写进 items 的 collection properties。

    Day 0 是写在库旁边的 milvus.db.embedding 文件里，那在单进程单文件下没问题；
    连集群之后每个服务实例各写一份本地文件，两个实例配了不同 EMBED_MODEL 就会互相
    触发重建——数据没了都不知道是谁干的。存进 Milvus 才是单一事实源。
    实测 milvus-lite 3.2.0：properties 在进程重开后仍在（描述字段会丢，properties 不会）。"""
    try:
        client.alter_collection_properties("items", {PROP_EMBED: sig})
    except Exception as e:  # 集群若拒绝自定义 property key，降级成"记不住"，不要拖垮启动
        print(f"指纹写不进 collection properties（{e}）：换 embedding 模型时不会自动重建，"
              f"需要手动 drop items")


def _stored_signature():
    return (client.describe_collection("items").get("properties") or {}).get(PROP_EMBED)


def _embedding_changed(dim):
    """维度对不上就必须重建。维度一样但换了模型时，靠 properties 里的指纹兜一层；
    指纹不在（Day 0 迁上来的老库）就认为没变——宁可不重建，也不要误删数据。"""
    if _dim("items") != dim:
        print(f"dense 维度 {_dim('items')} -> {dim}，重建集合")
        return True
    was = _stored_signature()
    if was and was != _signature():
        print(f"embedding 变了：{was} -> {_signature()}，重建集合")
        return True
    return False


def _fields(name):
    return {f["name"] for f in client.describe_collection(name)["fields"]}


def _dim(name):
    return next((f["params"]["dim"] for f in client.describe_collection(name)["fields"]
                 if f["name"] == "dense"), None)  # 老库没有 dense 字段时返回 None


# ---------- 写 ----------
def add_items(tenant, items):
    """dense 乘上流行度权重、cf 从共现表取，两件事都在写入侧做完。

    这样查询侧什么都不用改：IP 内积天然等于 w(i) × cos(i,q)，
    协同过滤那一路就是一次普通的稀疏向量检索。没有应用层重排，没有多一次往返。
    """
    vecs = embedding.encode([i["text"] for i in items])
    rows = [dict(i, tenant_id=tenant,
                 dense=popularity.scale(v, i["id"]),
                 cf=cooccur.item_vector(i["id"]))
            for i, v in zip(items, vecs)]
    try:
        client.upsert("items", rows)  # sparse 由 BM25 生成
    except DataNotMatchException as e:
        # Day 0 踩过 year 存成字符串导致 'year >= 2000' 静默失效。现在 year 是显式 INT64：
        # 走 HTTP 进来的先被 pydantic 收拾（"1999"→1999，"abc"→422），绕过 pydantic 直接
        # 调这个函数的由 schema 兜底。两层都不会让一个字符串安静地躺在 year 里。
        raise ValueError(f"字段类型和 schema 对不上（category 要字符串、year 要整数）：{e}") from e
    return len(rows)


def add_users(tenant, users):
    """用户塔依赖商品向量：history 里的物品必须已经灌进同一个租户的 items，否则跳过这个用户。
    也就是说没有历史的新用户在这里建不出来——冷启动要另外一套做法。

    strong=True 不是保守，是必须的：这条路径**读的正是刚写进去的 items**
    （seed.py 先灌商品再灌用户，只隔几秒）。Bounded 查的是一个稍旧的时间戳，商品还没进
    可查视图，`vec_of` 就是空的——于是每个用户都被"跳过"，用户塔一条也建不出来，而接口
    返回 200、skipped 里躺着 6038 个 id，看着像数据有问题而不像一致性有问题。
    实测这条路径（集群，写完立刻读同一个 id）：Bounded 2/10，Strong 10/10。
    Day 0 在进程内没有这回事，这是上集群后第一个会咬人的地方。
    """
    vec_of = _vec_map(tenant, [i for u in users for i in u["history"]], strong=True)
    rows, skipped = [], []
    for u in users:
        # 只用最近 HISTORY_K 项。全量协议下平均历史 165 项，全部取均值出来的向量太散
        # ——它描述的是"这个人这些年都看过什么"，而要预测的是"他下一个看什么"。
        hist = u["history"][-HISTORY_K:] if HISTORY_K else u["history"]
        vecs = [vec_of[i] for i in hist if i in vec_of]
        if vecs:
            rows.append({"user_id": u["user_id"], "tenant_id": tenant, "dense": mean(vecs),
                         "history": "|".join(u["history"]), "prefer": u.get("prefer", "")})
        else:
            skipped.append(u["user_id"])
    if rows:
        client.upsert("users", rows)
    return len(rows), skipped


# ---------- 读 ----------
def _read(strong=False):
    """写完立刻回查的路径传 strong=True。

    Day 0 的"加一部电影，下一秒就搜到"是 Milvus Lite 的进程内行为，没有一致性这回事。
    集群默认 Bounded：查询用的是一个稍旧的时间戳，刚 upsert 的行可能还没进可查视图。

    **别把它当"偶发"。** 实测 Zilliz Cloud，60 次「写入 + 立刻搜」：
      Bounded  命中 20/60，p50 79.2ms  p99 163ms
      Strong   命中 60/60，p50 84.4ms  p99 271ms
    也就是默认一致性下 2/3 查不到，而不是偶尔查不到。Strong 的代价也不在平均值上
    （p50 只 +6%），全在尾部（p99 +66%）——这才是"别整个服务开 Strong"的真正理由：
    SLO 是按 p99 定的。
    """
    return {"consistency_level": "Strong" if strong else CONSISTENCY}


def _dense_params(metric=None):
    return {"metric_type": metric or popularity.METRIC,
            **({"params": SEARCH_PARAMS} if SEARCH_PARAMS else {})}


def _group(group_by, group_size):
    """分组去重透传成 Milvus 的三个参数。

    strict_group_size=True 是"每组必须给够 group_size 条"，给不够就少返回；
    False 是"尽力凑"。做多样性用 True，语义更好预测。
    """
    if not group_by:
        return {}
    return {"group_by_field": group_by, "group_size": group_size,
            "strict_group_size": True}


def _limit(n):
    """Milvus 对单次召回的 limit 有硬上限，顶到会报 limit out of range。"""
    return max(1, min(n, MAX_RECALL))


def search(tenant, q, top_k=10, mode="bm25", filter="", strong=False,
           group_by=None, group_size=1):
    """mode: bm25 走 sparse / vector 走 dense / hybrid 两路 RRF 融合。"""
    if mode == "hybrid":
        return _hybrid(tenant, [
            (_ann("sparse", [q], {"metric_type": "BM25"}, _limit(top_k * RECALL_FANOUT),
                  scope(tenant, filter))),
            (_ann("dense", embedding.encode([q]), _dense_params(),
                  _limit(top_k * RECALL_FANOUT), scope(tenant, filter))),
        ], top_k, group_by, group_size, strong)

    if mode == "bm25":
        data, field, params = [q], "sparse", {"metric_type": "BM25"}
    else:
        data, field, params = embedding.encode([q]), "dense", _dense_params()
    res = client.search("items", data=data, anns_field=field, limit=_limit(top_k),
                        filter=scope(tenant, filter), output_fields=["id", "text"],
                        search_params=params, **_group(group_by, group_size), **_read(strong))
    return _hits(res, params["metric_type"])


def search_batch(tenant, qs, top_k=10, mode="bm25", filter="", chunk=128):
    """一次往返搜一批查询。和 recommend_batch 同理，评测走这条。"""
    expr = scope(tenant, filter)
    out = []
    for s in range(0, len(qs), chunk):
        part = qs[s:s + chunk]
        if mode == "hybrid":
            reqs = [_ann("sparse", part, {"metric_type": "BM25"}, _limit(top_k), expr),
                    _ann("dense", embedding.encode(part), _dense_params(), _limit(top_k), expr)]
            from pymilvus import RRFRanker
            res = client.hybrid_search("items", reqs, ranker=RRFRanker(RRF_K),
                                       limit=_limit(top_k), output_fields=["id", "text"],
                                       **_read())
            fix = float
        elif mode == "bm25":
            res = client.search("items", data=part, anns_field="sparse", limit=_limit(top_k),
                                filter=expr, output_fields=["id", "text"],
                                search_params={"metric_type": "BM25"}, **_read())
            fix = abs
        else:
            res = client.search("items", data=embedding.encode(part), anns_field="dense",
                                limit=_limit(top_k), filter=expr,
                                output_fields=["id", "text"],
                                search_params=_dense_params(), **_read())
            fix = float
        out += [[dict(h["entity"], score=fix(h["distance"])) for h in row] for row in res]
    return out


def query_vector(tenant, history):
    """历史 → 稠密查询向量。只用最近 HISTORY_K 项，和用户塔的算法保持一致。

    评测走的是这条路而不是用户塔里存好的向量：留一法要藏起最后一项，
    而塔里那个向量是拿全量历史（含答案）算的。
    """
    hist = list(history)[-HISTORY_K:] if HISTORY_K else list(history)
    vecs = item_vectors(tenant, hist)
    return mean(vecs) if vecs else None


def recommend(tenant, vec, top_k, exclude, filter="", mode="content",
              history=None, group_by=None, group_size=1):
    """mode: content 只用 dense / cf 只用共现 / multi 两路 RRF 融合。

    三种都能单独调，是为了能分别评测——只提供融合后的那一个，就量不出每一路
    各自贡献了多少，也没法在某一路挂掉的时候判断该不该降级。

    多召一批再剔掉历史：limit = top_k + len(exclude)，免得排除完不够 top_k。
    """
    exclude = [i for i in (exclude or []) if i]
    # exclude 用全量历史（看过的不能再推），但查询向量只用最近 HISTORY_K 项。
    # 两者不是一回事：前者是业务约束，后者是"预测下一个"这件事该看多远。
    hist = [i for i in (history or exclude) if i]
    hist = list(dict.fromkeys(hist))[-HISTORY_K:] if HISTORY_K else list(dict.fromkeys(hist))
    need = _limit(top_k + len(exclude))

    if mode == "multi":
        reqs = []
        if vec is not None:
            reqs.append(_ann("dense", [vec], _dense_params(),
                             _limit(top_k * RECALL_FANOUT + len(exclude)), scope(tenant, filter)))
        if hist:
            reqs.append(_ann("cf", [cooccur.query_vector(hist)], {"metric_type": "IP"},
                             _limit(top_k * RECALL_FANOUT + len(exclude)), scope(tenant, filter)))
        if not reqs:
            return []
        # 某一路没召回就只用另一路，不用写特判——新物品没有共现数据、
        # 冷启动用户没有历史，两种情况都在这里自然降级。
        hits = _hybrid(tenant, reqs, need, group_by, group_size)
    elif mode == "cf":
        if not hist:
            return []
        res = client.search("items", data=[cooccur.query_vector(hist)], anns_field="cf",
                            limit=need, filter=scope(tenant, filter),
                            output_fields=["id", "text"], search_params={"metric_type": "IP"},
                            **_group(group_by, group_size), **_read())
        hits = _hits(res, "IP")
    else:
        if vec is None:
            return []
        res = client.search("items", data=[vec], anns_field="dense", limit=need,
                            filter=scope(tenant, filter), output_fields=["id", "text"],
                            search_params=_dense_params(),
                            **_group(group_by, group_size), **_read())
        hits = _hits(res)

    seen = set(exclude)
    return [h for h in hits if h["id"] not in seen][:top_k]


def recommend_batch(tenant, queries, top_k, mode="content", filter="",
                    group_by=None, group_size=1, chunk=128):
    """一次往返算一批用户。queries: [(vec, history)]，返回和它一一对应的 [[hit, ...]]。

    **评测必须走这条路。**逐用户调 recommend() 是 1500 次往返，实测跑一轮要半小时；
    Milvus 的 search / hybrid_search 本来就接受一次传多条查询向量，
    批量之后同一轮是几十次往返。

    两个坑：

      1. filter 是**请求级**的，做不到一条查询向量一个 filter。所以"排除看过的"
         只能在 Python 侧做，limit 得按这一批里最长的历史留余量。
      2. 于是要**先按历史长度分桶**。全量协议下有人历史 2277 项，不分桶的话这一个人
         会把整批的 limit 顶到 2287，1500 条查询乘上去就是几百万行结果。

    分桶之后每桶的 limit 只服务真正需要它的那些用户，数据量降一个量级。
    """
    need = [top_k + len(h) for _, h in queries]
    tiers = [64, 128, 256, 512, 1024, 2048, MAX_RECALL]
    out = [None] * len(queries)

    buckets = {}
    for i, nd in enumerate(need):
        t = next((t for t in tiers if nd <= t), MAX_RECALL)
        buckets.setdefault(t, []).append(i)

    for tier, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), chunk):
            part = idxs[s:s + chunk]
            rows = _batch_call(tenant, [queries[i] for i in part], _limit(tier),
                               mode, filter, group_by, group_size)
            for i, hits in zip(part, rows):
                seen = set(queries[i][1])
                out[i] = [h for h in hits if h["id"] not in seen][:top_k]
    return out


def _recent(h):
    """排除列表用全量历史，查询向量只用最近 HISTORY_K 项 —— 两者不是一回事。

    这一刀漏了的后果实测很大：cf 那一路用全量历史查是 HR@10 0.075，
    切到最近 5 项是 0.19。Σ 会把"这个人这些年看过的一切"加在一起，
    而和什么都共现的正是全局热门物品，历史越长这一路越退化成热门推荐。
    """
    h = list(dict.fromkeys(i for i in h if i))
    return h[-HISTORY_K:] if HISTORY_K else h


def _batch_call(tenant, part, limit, mode, filter, group_by, group_size):
    expr = scope(tenant, filter)
    grp = _group(group_by, group_size)
    if mode == "multi":
        reqs = [
            _ann("dense", [v for v, _ in part], _dense_params(), limit, expr),
            _ann("cf", [cooccur.query_vector(_recent(h)) for _, h in part],
                 {"metric_type": "IP"}, limit, expr),
        ]
        from pymilvus import RRFRanker
        res = client.hybrid_search("items", reqs, ranker=RRFRanker(RRF_K), limit=limit,
                                   output_fields=["id", "text"], **grp, **_read())
        metric = "RRF"
    elif mode == "cf":
        res = client.search("items", data=[cooccur.query_vector(_recent(h)) for _, h in part],
                            anns_field="cf", limit=limit, filter=expr,
                            output_fields=["id", "text"],
                            search_params={"metric_type": "IP"}, **grp, **_read())
        metric = "IP"
    else:
        res = client.search("items", data=[v for v, _ in part], anns_field="dense",
                            limit=limit, filter=expr, output_fields=["id", "text"],
                            search_params=_dense_params(), **grp, **_read())
        metric = None
    fix = abs if metric == "BM25" else float
    return [[dict(h["entity"], score=fix(h["distance"])) for h in row] for row in res]


def query_vectors(tenant, histories):
    """一批历史 → 一批查询向量。所有物品向量一次取完，均值在 Python 侧算。

    逐用户调 query_vector() 又是 1500 次往返，而这些历史里的物品大量重复
    （热门电影出现在几百个用户的最近 5 项里），去重后一次 _vec_map 就够了。
    """
    recent = []
    for h in histories:
        h = list(dict.fromkeys(i for i in h if i))
        recent.append(h[-HISTORY_K:] if HISTORY_K else h)
    vec_of = _vec_map(tenant, [i for r in recent for i in r])
    out = []
    for r in recent:
        vs = [vec_of[i] for i in r if i in vec_of]
        out.append(mean(vs) if vs else None)
    return out


def _ann(field, data, param, limit, expr):
    from pymilvus import AnnSearchRequest
    return AnnSearchRequest(data=data, anns_field=field, param=param, limit=limit, expr=expr)


def _hybrid(tenant, reqs, top_k, group_by=None, group_size=1, strong=False):
    """一次请求发多路检索，融合在服务端完成。

    别在 Python 里查两路再自己合并：多一次往返，而且两路的分数量纲不一样
    （BM25 在 milvus-lite 里是负的、余弦在 -1..1、共现权重又是另一个量级），
    没法直接相加。RRF 把分数扔掉只留名次，量纲问题就没了：

        score = Σ 1/(k + 这一路里的排名)

    所以它奖励"多路共识"而不是"某一路的极端高分"——两路都排中上的候选，
    会赢过"一路第一、另一路很后"的候选。这正是多路召回想要的效果。
    """
    from pymilvus import RRFRanker
    res = client.hybrid_search("items", reqs, ranker=RRFRanker(RRF_K), limit=_limit(top_k),
                               output_fields=["id", "text"],
                               **_group(group_by, group_size), **_read(strong))
    return _hits(res, "RRF")


def similar_users(tenant, vec, top_k, exclude):
    res = client.search("users", data=[vec], anns_field="dense", limit=top_k + 1,
                        filter=scope(tenant), output_fields=["user_id", "prefer"],
                        search_params=_dense_params(), **_read())
    return [h for h in _hits(res) if h["user_id"] != exclude][:top_k]  # 自己肯定排第一，去掉


def _vec_map(tenant, ids, strong=False, chunk=500):
    """一批 id 换 {id: dense}。去重 + 分批，一次往返拿一批。

    Day 0 是一个用户一次 query——进程内调用，几十微秒，6038 个用户也就一眨眼，所以
    "循环里查库"这种写法在本地根本不显形。上集群之后同一段代码是 6038 次网络往返：
    **实测灌全量用户 707s，其中商品塔只花了 2.5s。** 而这些 id 里绝大部分是重复的
    （热门电影出现在几百个用户的历史里），去重后 6038 个用户的历史一共才 3000 多个物品。

    chunk=500 是给表达式长度留余量：`id in [...]` 是一条字符串塞进 filter，Milvus 对
    表达式长度有上限，id 再长一点就顶到了。分批也让 limit 不会撞上 query 的返回上限。
    """
    out = {}
    ids = list(dict.fromkeys(i for i in ids if i))  # 去重且保序
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        rows = client.query("items", filter=scope(tenant, _in(part)),
                            output_fields=["id", "dense"], limit=len(part), **_read(strong))
        out.update({r["id"]: r["dense"] for r in rows})
    return out


def item_vectors(tenant, ids):
    vec_of = _vec_map(tenant, ids)
    return [vec_of[i] for i in ids if i in vec_of]


def get_items(tenant, ids):
    if not ids:
        return []
    return client.query("items", filter=scope(tenant, _in(ids)), output_fields=["id", "text"],
                        limit=len(ids), **_read())


def get_user(tenant, user_id):
    rows = client.query("users", filter=scope(tenant, f'user_id == "{_escape(user_id)}"'),
                        output_fields=["user_id", "dense", "history", "prefer"], limit=1, **_read())
    return rows[0] if rows else None


def sample_users(tenant, n):
    return client.query("users", filter=scope(tenant), output_fields=["user_id", "prefer"],
                        limit=n, **_read())


def count(name, tenant):
    return client.query(name, filter=scope(tenant), output_fields=["count(*)"],
                        **_read())[0]["count(*)"]


# ---------- 运维 ----------
def compact(name="items"):
    """把小 segment 合并。Day 0 那个"刚写进去的行用类型词搜不到"的坑就出在这：
    BM25 的 idf 是按 segment 统计的，新写入的行独占一个新 segment，"Adventure" 这种词
    在它那儿 idf≈0，会被老数据压到后面。合并之后统计量拉平，排序才是全局可比的。

    四种部署四种行为，都是实测：
      milvus-lite            同步做完，返回 job id 0
      自建 standalone        返回真的 job id，后台异步做
      Zilliz Cloud 专属集群  返回 job id -1 —— 请求收下了，但不给你一个能查的 job，
                             合并由平台自己排。**不报错，也不代表它做了。**
      Zilliz Cloud serverless PERMISSION_DENIED —— 直接不开放这个权限

    也就是说托管服务上这条优化不由你控制：既不能确认它跑了，也不能等它跑完。
    所以别把它放在"灌完数据必须成功"的关键路径上，降级成一条说明。
    """
    try:
        return {"collection": name, "job_id": client.compact(name)}
    except Exception as e:
        if "PrivilegeCompaction" not in str(e) and "permission deny" not in str(e):
            raise
        return {"collection": name, "skipped": "托管服务不开放手动 compaction，由平台自己调度"}


def stats():
    """给 /health 和告警用：集合是不是还 loaded，是判断"它挂没挂"最直接的一个信号。"""
    out = {"uri": URI, "cluster": CLUSTER, "consistency": CONSISTENCY,
           "index": {"type": INDEX_TYPE, "build": INDEX_PARAMS, "search": SEARCH_PARAMS,
                     "bm25": BM25_PARAMS or "默认 k1=1.2 b=0.75", "dense_metric": popularity.METRIC},
           "recall": {"history_k": HISTORY_K, "rrf_k": RRF_K, "fanout": RECALL_FANOUT,
                      "pop_alpha": popularity.ALPHA, "cf": cooccur.signature()},
           "replicas": REPLICAS}
    for name in ("items", "users"):
        try:
            out[f"{name}_loaded"] = str(client.get_load_state(name).get("state"))
        except Exception as e:
            out[f"{name}_loaded"] = f"unknown: {e}"
    return out


# ---------- 小工具 ----------
def mean(vecs):
    """先把每个向量各自归一化，再平均。

    α>0 之后物品向量的模长不再是 1（热门物品模长更大），直接平均的话热门物品会
    主导查询向量的方向——先验会被算两次：一次在查询向量里，一次在被检索的物品上。
    查询侧要的是纯方向，先验只该作用在被检索的那一侧。"""
    unit = [embedding.normalize(v) for v in vecs]
    return embedding.normalize([sum(col) / len(unit) for col in zip(*unit)])


def _hits(res, metric=None):
    # BM25 取绝对值：Milvus Lite 的 BM25 分数是负的（越相关越负），集群版是正的，两边返回顺序
    # 都已经是最相关在前，abs 只是让 score 统一成"越大越相关"。COSINE 的负分有意义，不能动。
    fix = abs if metric == "BM25" else float
    return [dict(h["entity"], score=fix(h["distance"])) for h in res[0]]


def _escape(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _in(ids):
    return "id in [%s]" % ", ".join('"%s"' % _escape(i) for i in ids)
