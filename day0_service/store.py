"""Milvus 存储层：items（商品塔）+ users（用户塔）。改 MILVUS_URI 即可从本地文件切到集群。"""
import os
from pathlib import Path

# 必须在 import pymilvus 之前取走：pymilvus 自己也读 MILVUS_URI，且只认 http(s)://，
# 留在环境里会让 ./milvus.db 这种本地路径在 import 阶段就报 Illegal uri。
URI = os.environ.pop("MILVUS_URI", "./milvus.db")

from pymilvus import DataType, Function, FunctionType, MilvusClient  # noqa: E402

import embedding  # noqa: E402

client = MilvusClient(uri=URI)
ANALYZER = os.getenv("TEXT_ANALYZER", "standard")  # chinese / english / standard

USERS_FIELDS = {"user_id", "dense", "history", "prefer"}  # 跟下面 init() 里建的 schema 对齐

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


def init():
    dim = embedding.dim()
    if client.has_collection("items") and _embedding_changed(dim):
        # 向量本身变了，两个塔的数据都作废（要重新跑 seed.py）
        client.drop_collection("items")
        if client.has_collection("users"):
            client.drop_collection("users")
    elif client.has_collection("users") and _fields("users") != USERS_FIELDS:
        # users 的字段增删过（Milvus 不接受 schema 外的字段，写入会直接失败）。
        # 但 items 的向量还是好的，别连坐——只重建 users。
        print(f"users schema 变了：{sorted(_fields('users'))} -> {sorted(USERS_FIELDS)}，"
              f"只重建 users，items 保留")
        client.drop_collection("users")

    if not client.has_collection("items"):
        s = client.create_schema(enable_dynamic_field=True)  # category/year 等标量原样存、能过滤
        s.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        s.add_field("text", DataType.VARCHAR, max_length=4096,
                    enable_analyzer=True,
                    analyzer_params=ANALYZER_PARAMS.get(ANALYZER, {"type": ANALYZER}))
        s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        s.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
        # sparse 不自己算：Milvus 原生 BM25 Function 从 text 生成，写入/查询两端都走它
        s.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                input_field_names=["text"], output_field_names=["sparse"]))
        idx = client.prepare_index_params()
        idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        idx.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        _create("items", s, idx)

    if not client.has_collection("users"):
        s = client.create_schema()
        s.add_field("user_id", DataType.VARCHAR, is_primary=True, max_length=64)
        s.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)  # 兴趣向量：历史物品向量的均值
        s.add_field("history", DataType.VARCHAR, max_length=8192)
        s.add_field("prefer", DataType.VARCHAR, max_length=128)
        idx = client.prepare_index_params()
        idx.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        _create("users", s, idx)

    # 新进程连上已存在的库时集合是 released 状态，不显式 load 的话第一次 query 就报
    # "call load() before search/get/query"。新建的集合已经是 loaded，load 一次也无害。
    client.load_collection("items")
    client.load_collection("users")
    _mark_file().write_text(embedding.signature(), encoding="utf-8")


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


def _embedding_changed(dim):
    """维度对不上就必须重建。维度一样但换了模型时，靠旁路文件里的指纹兜一层；
    文件不在（老库、或集群部署）就认为没变——宁可不重建，也不要误删数据。"""
    if _dim("items") != dim:
        print(f"dense 维度 {_dim('items')} -> {dim}，重建集合")
        return True
    f = _mark_file()
    was = f.read_text(encoding="utf-8").strip() if f.exists() else None
    if was and was != embedding.signature():
        print(f"embedding 变了：{was} -> {embedding.signature()}，重建集合")
        return True
    return False


def _mark_file():
    # 跟着库放：本地文件就是 ./milvus.db.embedding；连集群时落在当前目录
    return Path(f"{URI}.embedding" if not URI.startswith("http") else ".embedding")


def _fields(name):
    return {f["name"] for f in client.describe_collection(name)["fields"]}


def _dim(name):
    return next((f["params"]["dim"] for f in client.describe_collection(name)["fields"]
                 if f["name"] == "dense"), None)  # 老库没有 dense 字段时返回 None


# ---------- 写 ----------
def add_items(items):
    vecs = embedding.encode([i["text"] for i in items])
    client.upsert("items", [dict(i, dense=v) for i, v in zip(items, vecs)])  # sparse 由 BM25 生成
    return len(items)


def add_users(users):
    """用户塔依赖商品向量：history 里的物品必须已经灌进 items，否则这个用户会被跳过。
    也就是说没有历史的新用户在这里建不出来——冷启动要另外一套做法，不在 Day 0 范围内。"""
    rows, skipped = [], []
    for u in users:
        vecs = item_vectors(u["history"])
        if vecs:
            rows.append({"user_id": u["user_id"], "dense": mean(vecs),
                         "history": "|".join(u["history"]), "prefer": u.get("prefer", "")})
        else:
            skipped.append(u["user_id"])
    if rows:
        client.upsert("users", rows)
    return len(rows), skipped


# ---------- 读 ----------
def search(q, top_k=10, mode="bm25", filter=""):
    if mode == "bm25":
        data, field, metric = [q], "sparse", "BM25"
    else:
        data, field, metric = embedding.encode([q]), "dense", "COSINE"
    res = client.search("items", data=data, anns_field=field, limit=top_k, filter=filter,
                        output_fields=["id", "text"], search_params={"metric_type": metric})
    return _hits(res, metric)


def recommend(vec, top_k, exclude, filter=""):
    """多召一批再剔掉历史，免得排除完不够 top_k。"""
    res = client.search("items", data=[vec], anns_field="dense", limit=top_k + len(exclude),
                        filter=filter, output_fields=["id", "text"],
                        search_params={"metric_type": "COSINE"})
    return [h for h in _hits(res) if h["id"] not in set(exclude)][:top_k]


def similar_users(vec, top_k, exclude):
    res = client.search("users", data=[vec], anns_field="dense", limit=top_k + 1,
                        output_fields=["user_id", "prefer"],
                        search_params={"metric_type": "COSINE"})
    return [h for h in _hits(res) if h["user_id"] != exclude][:top_k]  # 自己肯定排第一，去掉


def item_vectors(ids):
    if not ids:
        return []
    rows = client.query("items", filter=_in(ids), output_fields=["dense"], limit=len(ids))
    return [r["dense"] for r in rows]


def get_items(ids):
    if not ids:
        return []
    return client.query("items", filter=_in(ids), output_fields=["id", "text"], limit=len(ids))


def get_user(user_id):
    rows = client.query("users", filter=f'user_id == "{user_id}"',
                        output_fields=["user_id", "dense", "history", "prefer"], limit=1)
    return rows[0] if rows else None


def sample_users(n):
    return client.query("users", filter="", output_fields=["user_id", "prefer"], limit=n)


def count(name):
    return client.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"]


# ---------- 小工具 ----------
def mean(vecs):
    return embedding.normalize([sum(col) / len(vecs) for col in zip(*vecs)])


def _hits(res, metric="COSINE"):
    # BM25 取绝对值：Milvus Lite 的 BM25 分数是负的（越相关越负），集群版是正的，两边返回顺序
    # 都已经是最相关在前，abs 只是让 score 统一成"越大越相关"。COSINE 的负分有意义，不能动。
    fix = abs if metric == "BM25" else float
    return [dict(h["entity"], score=fix(h["distance"])) for h in res[0]]


def _in(ids):
    return "id in [%s]" % ", ".join('"%s"' % i for i in ids)
