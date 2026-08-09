"""文本转向量。默认 hash 后端（零下载、零依赖），换真模型只要动 encode/dim 两个分支。"""
import hashlib
import math
import os
import re

# 512 是故意的：常见小模型（bge-small / MiniLM 等）都是 384 维，撞上了的话切后端时
# store.init() 的维度检查发现不了，两种向量会混在一个集合里，检索结果乱且不报错。
HASH_DIM = 512
BACKEND = os.getenv("EMBED_BACKEND", "hash")  # hash | st
MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# 英文/数字按词切，中文按字切——不然 jieba 那套中文语料在 hash 后端里会全是零向量
_TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")
_model = None


def _st():
    """本地 sentence-transformers 模型，第一次调用才加载（也才下权重）。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL)
    return _model


def _hash_vec(text):
    """random projection 的穷人版：token 定位到某一维，符号位决定加还是减，最后归一化。"""
    v = [0.0] * HASH_DIM
    for tok in _TOKEN.findall(text.lower()):
        h = hashlib.md5(tok.encode()).digest()
        v[int.from_bytes(h[:4], "big") % HASH_DIM] += 1.0 if h[4] % 2 else -1.0
    return normalize(v)


def normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def encode(texts):
    if BACKEND == "st":
        return [v.tolist() for v in _st().encode(texts, normalize_embeddings=True)]
    return [_hash_vec(t) for t in texts]


def dim():
    if BACKEND != "st":
        return HASH_DIM
    m = _st()
    # sentence-transformers 5.x 改名了，老名字还在但会打 FutureWarning
    f = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
    return f()


def signature():
    """人看的标识，也用来判断要不要重建（见 store._check_rebuild）。"""
    return f"st:{MODEL}:{dim()}" if BACKEND == "st" else f"hash:{dim()}"
