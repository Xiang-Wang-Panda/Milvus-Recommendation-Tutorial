from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import embedding
import store

app = FastAPI(title="Day 0 搜索 / 推荐")
store.init()


class Item(BaseModel):
    model_config = {"extra": "allow"}  # category/year 等额外字段原样收下，进 dynamic field
    id: str
    text: str


class User(BaseModel):
    user_id: str
    history: List[str] = []  # 物品 id，必须已经灌进 items（兴趣向量是它们的均值）
    prefer: str = ""


class Search(BaseModel):
    q: str
    top_k: int = 10
    mode: str = "bm25"  # bm25 走 sparse / vector 走 dense
    filter: str = ""  # 如 'category == "movie" and year >= 2000'


class Rec(BaseModel):
    user_id: Optional[str] = None  # 有 user_id 用存好的兴趣向量
    item_ids: Optional[List[str]] = None  # 没有就拿这批物品现算均值（无状态推荐）
    top_k: int = 10
    filter: str = ""


@app.get("/health")
def health():
    return {"analyzer": store.ANALYZER, "embed": embedding.signature(),
            "items": store.count("items"), "users": store.count("users")}


@app.post("/items")
def add_items(items: List[Item]):
    return {"upserted": store.add_items([i.model_dump() for i in items])}


@app.post("/search")
def search(q: Search):
    return store.search(q.q, q.top_k, q.mode, q.filter)


@app.post("/users")
def add_users(users: List[User]):
    upserted, skipped = store.add_users([u.model_dump() for u in users])
    return {"upserted": upserted, "skipped": skipped}  # skipped = history 全是未知物品


@app.post("/recommend")
def recommend(r: Rec):
    if r.user_id:
        u = store.get_user(r.user_id) or _404(r.user_id)
        return store.recommend(u["dense"], r.top_k, u["history"].split("|"), r.filter)
    vecs = store.item_vectors(r.item_ids or [])
    if not vecs:
        raise HTTPException(400, "need user_id or known item_ids")
    return store.recommend(store.mean(vecs), r.top_k, r.item_ids, r.filter)


@app.post("/similar-users")
def similar_users(user_id: str, top_k: int = 5):
    u = store.get_user(user_id) or _404(user_id)
    return store.similar_users(u["dense"], top_k, user_id)


@app.get("/users/sample")  # 必须排在 /users/{user_id} 前面，否则 sample 会被当成 user_id
def sample_users(n: int = 30):
    return store.sample_users(n)


@app.get("/users/{user_id}")
def profile(user_id: str):
    u = store.get_user(user_id) or _404(user_id)
    return {"prefer": u["prefer"],
            "history": store.get_items([i for i in u["history"].split("|") if i])}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


def _404(uid):
    raise HTTPException(404, f"user {uid} not found")
