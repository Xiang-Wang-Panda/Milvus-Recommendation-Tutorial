"""Day 2 服务层。相对 Day 1 只多了召回层的开关：mode 和 group_by。

八个接口的语义一个没变，所以前端和调用方不用改。
"""
import os
import time
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

import embedding
import store

DEFAULT_TENANT = os.getenv("DEFAULT_TENANT", "default")

app = FastAPI(title="Day 2 搜索 / 推荐（多路召回）")
store.init()

# 服务侧埋点。桶按"搜索接口该有的样子"排：10ms 以内算好，超过 1s 基本等于坏了。
# 分位数不要在这里算——Prometheus 侧用 histogram_quantile 才能跨实例聚合。
LAT = Histogram("app_request_seconds", "请求延迟", ["route"],
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))
REQS = Counter("app_requests_total", "请求数", ["route", "status"])


@app.middleware("http")
async def observe(request: Request, call_next):
    t = time.perf_counter()
    resp = await call_next(request)
    # 用匹配到的路由模板而不是 request.url.path：/users/u1、/users/u2 会撑爆标签基数。
    # route 是路由完成后才写进 scope 的，所以这一行必须放在 call_next 之后。
    route = getattr(request.scope.get("route"), "path", "unmatched")
    LAT.labels(route).observe(time.perf_counter() - t)
    REQS.labels(route, resp.status_code).inc()
    return resp


def tenant(x_tenant_id: Annotated[str, Header()] = DEFAULT_TENANT) -> str:
    """租户从请求头来。这里只取值，合法性由 store.scope() 统一把关——
    校验散在两个地方，迟早有一条读路径漏掉。"""
    return x_tenant_id


T = Annotated[str, Depends(tenant)]


class Item(BaseModel):
    model_config = {"extra": "allow"}  # 除 category/year 外的额外字段仍进 dynamic field
    id: str
    text: str
    category: Optional[str] = None
    year: Optional[int] = None  # 显式 int：字符串年份在这里就被 pydantic 挡住，不用等 Milvus 报


class User(BaseModel):
    user_id: str
    history: List[str] = []  # 物品 id，必须已经灌进同租户的 items
    prefer: str = ""


class Search(BaseModel):
    q: str
    top_k: int = 10
    mode: str = "bm25"  # bm25 走 sparse / vector 走 dense / hybrid 两路 RRF 融合
    filter: str = ""  # 如 'category == "Comedy" and year >= 1995'
    strong: bool = False  # 写完立刻回查的场景打开，代价是延迟
    group_by: Optional[str] = None  # 分组去重字段。搜索默认不开——搜片名时就是要看全系列
    group_size: int = 1


class Rec(BaseModel):
    user_id: Optional[str] = None  # 有 user_id 用存好的兴趣向量
    item_ids: Optional[List[str]] = None  # 没有就拿这批物品现算均值（无状态推荐）
    top_k: int = 10
    filter: str = ""
    mode: str = "multi"  # content 只用内容 / cf 只用共现 / multi 两路 RRF 融合
    group_by: Optional[str] = None  # 首页推荐一般开 category，详情页看情况
    group_size: int = 1


@app.exception_handler(ValueError)
def bad_request(request: Request, e: ValueError):
    # store 层的非法 tenant_id、schema 类型不匹配都走这里，别让它变成 500
    return Response(content=str(e), status_code=400)


@app.get("/health")
def health(t: T):
    return {"tenant": t, "analyzer": store.ANALYZER, "embed": store._signature(),
            "items": store.count("items", t), "users": store.count("users", t), **store.stats()}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/items")
def add_items(items: List[Item], t: T):
    return {"upserted": store.add_items(t, [i.model_dump(exclude_none=True) for i in items])}


@app.post("/search")
def search(q: Search, t: T):
    return store.search(t, q.q, q.top_k, q.mode, q.filter, q.strong,
                        q.group_by, q.group_size)


@app.post("/users")
def add_users(users: List[User], t: T):
    upserted, skipped = store.add_users(t, [u.model_dump() for u in users])
    return {"upserted": upserted, "skipped": skipped}  # skipped = history 全是未知物品


@app.post("/recommend")
def recommend(r: Rec, t: T):
    if r.user_id:
        u = store.get_user(t, r.user_id) or _404(r.user_id)
        hist = [i for i in u["history"].split("|") if i]
        # 塔里存的 dense 已经是"最近 HISTORY_K 项的均值"，直接用；
        # 共现那一路要的是完整历史（用户看过的每个物品都是一个查询维度）。
        return store.recommend(t, u["dense"], r.top_k, hist, r.filter,
                               r.mode, hist, r.group_by, r.group_size)
    ids = r.item_ids or []
    vecs = store.item_vectors(t, ids)
    if not vecs:
        raise HTTPException(400, "need user_id or known item_ids")
    return store.recommend(t, store.mean(vecs), r.top_k, ids, r.filter,
                           r.mode, ids, r.group_by, r.group_size)


@app.post("/similar-users")
def similar_users(user_id: str, t: T, top_k: int = 5):
    u = store.get_user(t, user_id) or _404(user_id)
    return store.similar_users(t, u["dense"], top_k, user_id)


@app.post("/admin/compact")
def compact(t: T, name: str = "items"):
    """手动合并 segment。BM25 的 idf 按 segment 算，灌完一批数据跑一次，排序才全局可比。
    注意这是集合级动作，不分租户——放 /admin 下面提醒一句：这不该开给业务调用方。
    托管服务可能不给这个权限，那时候返回的是 skipped 而不是报错(见 store.compact)。"""
    return store.compact(name)


@app.get("/users/sample")  # 必须排在 /users/{user_id} 前面，否则 sample 会被当成 user_id
def sample_users(t: T, n: int = 30):
    return store.sample_users(t, n)


@app.get("/users/{user_id}")
def profile(user_id: str, t: T):
    u = store.get_user(t, user_id) or _404(user_id)
    return {"prefer": u["prefer"],
            "history": store.get_items(t, [i for i in u["history"].split("|") if i])}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


def _404(uid):
    # 跨租户查一个存在但不属于你的 user_id，走的也是这里：对调用方来说"不存在"和
    # "不是你的"必须长得一样，否则这个接口就成了探测别人租户里有谁的工具。
    raise HTTPException(404, f"user {uid} not found")
