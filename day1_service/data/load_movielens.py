"""下载 MovieLens-1M，转成 products.csv / users.csv。"""
import csv
import io
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

HERE = Path(__file__).parent
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
YEAR = re.compile(r"\((\d{4})\)\s*$")  # 标题结尾的 (1995)


def read(name):
    # ml-1m 是 :: 分隔 + latin-1 编码
    for line in open(HERE / "ml-1m" / name, encoding="latin-1"):
        if line.strip():
            yield line.strip().split("::")


def write(name, fields, rows):
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} 行")


if not (HERE / "ml-1m/movies.dat").exists():
    print(f"下载 {URL} ...")
    z = requests.get(URL, timeout=600)
    z.raise_for_status()
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(HERE)

# ---------- 商品：一部电影一行 ----------
genres = {}
movies = []
for mid, title, g in read("movies.dat"):
    genres[mid] = g.split("|")
    m = YEAR.search(title)
    movies.append({"id": f"m{mid}",
                   "text": f"{title} {g.replace('|', ' ')}",  # 标题 + 类型词，喂给 BM25 和 embedding
                   "category": genres[mid][0],  # 主类型 = 第一个
                   "year": m.group(1) if m else ""})
write("products.csv", ["id", "text", "category", "year"], movies)

# ---------- 用户：评分 >=4 视为喜欢，按时间取最近 50 部 ----------
liked = defaultdict(list)
for uid, mid, rating, ts in read("ratings.dat"):
    if int(rating) >= 4:
        liked[uid].append((int(ts), mid))

users = []
for uid, seen in liked.items():
    recent = [m for _, m in sorted(seen)[-50:]]
    top = Counter(g for m in recent for g in genres[m]).most_common(1)
    users.append({"user_id": f"u{uid}",
                  "history": "|".join(f"m{m}" for m in recent),
                  "prefer": top[0][0] if top else ""})
write("users.csv", ["user_id", "history", "prefer"], users)
