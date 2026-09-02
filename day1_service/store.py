"""Milvus 存储层（Day 1）：items（商品塔）+ users（用户塔）。

相对 Day 0 的四件事：连集群（token 认证）、索引可调、租户隔离、一致性可选。
业务函数的签名只多了一个 tenant，其余原样——Day 0 把 Milvus 调用都收在这一层，
就是为了让"上生产"这一步不外溢到 main.py 和前端。
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

import embedding  # noqa: E402

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
ITEMS_FIELDS = {"id", "tenant_id", "text", "category", "year", "sparse", "dense"}
USERS_FIELDS = {"user_id", "tenant_id", "dense", "history", "prefer"}

PROP_EMBED = "app.embedding.signature"  # 指纹存在 items 的 collection properties 里


def _signature():
    return embedding.signature()

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
        # sparse 不自己算：Milvus 原生 BM25 Function 从 text 生成，写入/查询两端都走它
        s.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                input_field_names=["text"], output_field_names=["sparse"]))
        idx = client.prepare_index_params()
        idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                      params=BM25_PARAMS)
        idx.add_index(field_name="dense", index_type=INDEX_TYPE, metric_type="COSINE",
                      params=INDEX_PARAMS)
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
        s.add_field("history", DataType.VARCHAR, max_length=65535)
        s.add_field("prefer", DataType.VARCHAR, max_length=128)
        idx = client.prepare_index_params()
        idx.add_index(field_name="dense", index_type=INDEX_TYPE, metric_type="COSINE",
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
    vecs = embedding.encode([i["text"] for i in items])
    rows = [dict(i, tenant_id=tenant, dense=v) for i, v in zip(items, vecs)]
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
        vecs = [vec_of[i] for i in u["history"] if i in vec_of]
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
    return {"metric_type": metric or "COSINE",
            **({"params": SEARCH_PARAMS} if SEARCH_PARAMS else {})}


def search(tenant, q, top_k=10, mode="bm25", filter="", strong=False):
    if mode == "bm25":
        data, field, params = [q], "sparse", {"metric_type": "BM25"}
    else:
        data, field, params = embedding.encode([q]), "dense", _dense_params()
    res = client.search("items", data=data, anns_field=field, limit=top_k,
                        filter=scope(tenant, filter), output_fields=["id", "text"],
                        search_params=params, **_read(strong))
    return _hits(res, params["metric_type"])


def recommend(tenant, vec, top_k, exclude, filter=""):
    """多召一批再剔掉历史，免得排除完不够 top_k。"""
    res = client.search("items", data=[vec], anns_field="dense", limit=top_k + len(exclude),
                        filter=scope(tenant, filter), output_fields=["id", "text"],
                        search_params=_dense_params(), **_read())
    return [h for h in _hits(res) if h["id"] not in set(exclude)][:top_k]


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
                     "bm25": BM25_PARAMS or "默认 k1=1.2 b=0.75",
                     "dense_metric": "COSINE"},
           "replicas": REPLICAS}
    for name in ("items", "users"):
        try:
            out[f"{name}_loaded"] = str(client.get_load_state(name).get("state"))
        except Exception as e:
            out[f"{name}_loaded"] = f"unknown: {e}"
    return out


# ---------- 小工具 ----------
def mean(vecs):
    return embedding.normalize([sum(col) / len(vecs) for col in zip(*vecs)])


def _hits(res, metric="COSINE"):
    # BM25 取绝对值：Milvus Lite 的 BM25 分数是负的（越相关越负），集群版是正的，两边返回顺序
    # 都已经是最相关在前，abs 只是让 score 统一成"越大越相关"。COSINE 的负分有意义，不能动。
    fix = abs if metric == "BM25" else float
    return [dict(h["entity"], score=fix(h["distance"])) for h in res[0]]


def _escape(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _in(ids):
    return "id in [%s]" % ", ".join('"%s"' % _escape(i) for i in ids)
