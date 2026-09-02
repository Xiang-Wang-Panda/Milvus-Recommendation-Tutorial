"""重做 MovieLens-1M 留一法数据集,这次数据用全。

Day 0 那份 users.csv 为了跑通链路做了两处简化:只留评分 >=4 的,每人只留最近 50 部。
跑 demo 没问题,但拿它量分数会骗自己——砍掉的既有负反馈信号也有长期兴趣,
而且留一法藏的那一项本来就在时间序列末尾,历史被截断之后预测下一项的难度都变了。

这里三处改回来:
  1. 全部评分都算一次交互(隐式反馈),不按评分过滤
  2. history 按 timestamp 排、不截断
  3. 交互数 < 5 的物品剔掉,剔完交互数 < 5 的用户也剔掉

产出 products_loo.csv / users_loo.csv,和 Day 0/1 那两个文件并存,靠环境变量选:
    PRODUCTS_CSV=products_loo.csv USERS_CSV=users_loo.csv
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
YEAR = re.compile(r"\((\d{4})\)\s*$")  # 标题结尾的 (1995)
MIN_ITEM = 5   # 交互数低于这个的物品剔掉
MIN_USER = 5   # 剔完物品之后历史短于这个的用户剔掉


def read(name):
    for line in open(HERE / "ml-1m" / name, encoding="latin-1"):
        if line.strip():
            yield line.strip().split("::")


def write(name, fields, rows):
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} 行")


genres, titles = {}, {}
for mid, title, g in read("movies.dat"):
    genres[mid] = g.split("|")
    titles[mid] = (title, g)

# ---------- 交互:全部评分都算,不按分数过滤 ----------
inter = []
for uid, mid, rating, ts in read("ratings.dat"):
    inter.append((uid, mid, int(ts)))
print(f"原始:{len(inter)} 条交互,{len({u for u, _, _ in inter})} 个用户,{len(titles)} 部电影")

# ---------- 过滤:交互数太少的物品和用户都剔掉 ----------
item_cnt = Counter(m for _, m, _ in inter)
keep_items = {m for m, c in item_cnt.items() if c >= MIN_ITEM}
inter = [(u, m, t) for u, m, t in inter if m in keep_items]

by_user = defaultdict(list)
for u, m, t in inter:
    by_user[u].append((t, m))
by_user = {u: v for u, v in by_user.items() if len(v) >= MIN_USER}
inter_n = sum(len(v) for v in by_user.values())
print(f"过滤(物品交互数 >= {MIN_ITEM}):保留 {inter_n} 条交互,"
      f"{len(by_user)} 个用户,{len(keep_items)} 个物品")

# ---------- 商品 ----------
movies = []
for mid in sorted(keep_items, key=int):
    title, g = titles[mid]
    m = YEAR.search(title)
    movies.append({"id": f"m{mid}",
                   "text": f"{title} {g.replace('|', ' ')}",
                   "category": genres[mid][0],
                   "year": m.group(1) if m else ""})
write("products_loo.csv", ["id", "text", "category", "year"], movies)

# ---------- 用户:完整时间序列 ----------
users, lens = [], []
for uid in sorted(by_user, key=int):
    seq = [m for _, m in sorted(by_user[uid])]     # 按 timestamp 排,不截断
    lens.append(len(seq))
    top = Counter(g for m in seq for g in genres[m]).most_common(1)
    users.append({"user_id": f"u{uid}",
                  "history": "|".join(f"m{m}" for m in seq),
                  "prefer": top[0][0] if top else ""})
write("users_loo.csv", ["user_id", "history", "prefer"], users)

lens.sort()
print(f"每人历史长度:中位数 {lens[len(lens) // 2]},均值 {sum(lens) // len(lens)},最长 {lens[-1]}")
