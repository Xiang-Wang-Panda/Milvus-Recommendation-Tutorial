"""Day 1 的量尺。三个子命令，都直连 Milvus（不走 HTTP），用 bench_ 前缀的临时集合，
不碰服务在用的 items/users。

  python bench.py index        索引选型：FLAT 当 ground truth，扫 HNSW / IVF_FLAT 的旋钮
  python bench.py filter       标量过滤：dynamic field vs 显式字段+INVERTED，三档选择率
  python bench.py consistency  写完立刻读：Bounded vs Strong 的命中率和延迟

公共参数：--limit N 只灌 N 条（先跑通再跑全量）、--queries N 查询条数、--keep 跑完不删集合、
--scale N 把语料扰动复制放大 N 倍（只有 index 用得上：3883 条量不出索引的差别）。

连哪儿由 MILVUS_URI 决定，和服务一致。本地 Milvus Lite 上量出来的绝对值没有参考价值
（没有网络、没有 query node），要的是集群上的数——但曲线的形状在两边是一样的。
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import embedding
import store  # 复用它的 client：同一个 URI、同一套认证

client = store.client
DATA = Path(__file__).parent / "data"
TOPK = 10

# 索引选型的候选。M/efConstruction 建完就固定；ef/nprobe 每次查询可调，所以一次建、多次扫。
CANDIDATES = [
    ("HNSW", {"M": 8, "efConstruction": 100}, [{"ef": 16}, {"ef": 32}, {"ef": 64}, {"ef": 128}]),
    ("HNSW", {"M": 16, "efConstruction": 200}, [{"ef": 16}, {"ef": 32}, {"ef": 64}, {"ef": 128}]),
    ("HNSW", {"M": 32, "efConstruction": 400}, [{"ef": 32}, {"ef": 64}, {"ef": 128}]),
    ("IVF_FLAT", {"nlist": 64}, [{"nprobe": 1}, {"nprobe": 4}, {"nprobe": 16}]),
    ("IVF_FLAT", {"nlist": 256}, [{"nprobe": 4}, {"nprobe": 16}, {"nprobe": 64}]),
    ("AUTOINDEX", {}, [{}]),  # Day 0 用的就是它，放进来看看它到底调到了哪一档
]


# ---------- 数据 ----------
def rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def items(limit):
    out = [{"id": r["id"], "text": r["text"], "category": r["category"] or None,
            "year": int(r["year"]) if r["year"] else None} for r in rows("products.csv")]
    return out[:limit] if limit else out


def inflate(rows_, factor):
    """把语料放大 factor 倍：复制真实向量再挪几个维度，**不是生成随机向量**。

    为什么需要放大：3883 条在集群上量不出索引的差别（实测 recall 全是 0.99+，p50 全挤在
    网络地板上）。HNSW 的图里只有几千个点，怎么走都能走到真正的最近邻——索引选型这件事
    在 Day 0 那个数据量上根本不存在，得先有量。

    为什么不能用随机向量填充：随机向量在高维里彼此近似正交，会在真实语料周围铺一层
    "谁都不像谁"的背景噪声，近似索引轻松跳过它们，recall 虚高。扰动复制保留原语料的
    聚簇结构——真实数据里就是有一堆彼此很像的邻居，那才是索引会走错的地方。

    LCG 而不是 random：同一个 factor 每次跑出同一份语料，两次 bench 的数才可比。
    """
    if factor <= 1:
        return rows_
    out, dim, seed = list(rows_), len(rows_[0]["dense"]), 12345

    def nxt():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return seed

    for k in range(1, factor):
        for r in rows_:
            v = list(r["dense"])
            for _ in range(8):  # 只动 8 个维度：够把它挪开，又不至于把它踢出原来的簇
                v[nxt() % dim] += (nxt() / 0x7FFFFFFF - 0.5) * 0.3
            out.append({"id": f"{r['id']}#{k}", "dense": embedding.normalize(v)})
    return out


def queries(n, vec_of):
    """查询向量必须是真实分布。随机向量在高维里彼此近似正交，量出来的召回虚高——
    Day 0 的 hash 后端已经演示过：哈希向量之间基本正交，recommend 的分数全是 0.0。
    这里用真用户的兴趣向量（历史物品向量的均值），就是线上真正会打进来的那种查询。"""
    out = []
    for r in rows("users.csv"):
        vecs = [vec_of[i] for i in r["history"].split("|") if i in vec_of]
        if vecs:
            out.append(store.mean(vecs))
        if len(out) >= n:
            break
    if not out:
        sys.exit("users.csv 里没有一个用户的历史命中当前 items（--limit 太小？）")
    return out


# ---------- 建集合 ----------
def drop(name):
    if client.has_collection(name):
        client.drop_collection(name)


def build(name, data, dim, index_type, params, scalar=None, dynamic=False):
    """scalar=True 建显式 category/year + INVERTED；dynamic=True 让它们落进 dynamic field。"""
    from pymilvus import DataType

    drop(name)
    s = client.create_schema(enable_dynamic_field=dynamic)
    s.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    s.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    if scalar:
        s.add_field("category", DataType.VARCHAR, max_length=64, nullable=True)
        s.add_field("year", DataType.INT64, nullable=True)
    idx = client.prepare_index_params()
    idx.add_index(field_name="dense", index_type=index_type, metric_type="COSINE", params=params)
    if scalar:
        idx.add_index(field_name="category", index_type="INVERTED")
        idx.add_index(field_name="year", index_type="INVERTED")
    t = time.perf_counter()
    client.create_collection(name, schema=s, index_params=idx)
    for i in range(0, len(data), 1000):
        client.insert(name, data[i:i + 1000])
    client.flush(name)
    client.load_collection(name)
    wait_index(name)
    return time.perf_counter() - t


def wait_index(name, timeout=900):
    """等索引真的建完。

    **不等就量，量到的是"索引建到哪一步了"，不是索引质量。** 实测 Zilliz Cloud serverless：
    load_collection 返回时 describe_index 还是 state=InProgress / indexed_rows=0 /
    pending=3883，约 2.4s 后才 Finished。索引没建好的时候查询走暴力扫，recall 是完美的
    1.000——所以第一次跑出来的表里 ef 越大 recall 反而越低，每组第一档都是 1.000。
    milvus-lite 是同步建的，这个循环在本地会立刻返回，两边同一套代码。
    """
    deadline = time.time() + timeout
    for field in client.list_indexes(name):
        while True:
            d = client.describe_index(name, field)
            if d.get("state") == "Finished" and not d.get("pending_index_rows"):
                break
            if time.time() > deadline:
                raise TimeoutError(f"{name}.{field} 索引 {timeout}s 还没建完：{d.get('state')}")
            time.sleep(1)


def run(name, qs, params, filter="", strong=True):
    """返回 (每次查询的耗时, 每次查询的 id 列表)。strong 是为了量准——Bounded 下
    有可能查到一个还没长全的视图，那量的就不是索引了。"""
    lat, hits = [], []
    for q in qs[:1]:  # 预热一次再计时：第一次查询要拉连接、填缓存，算进去会污染 p50
        client.search(name, data=[q], anns_field="dense", limit=TOPK, filter=filter,
                      output_fields=["id"], search_params={"metric_type": "COSINE"},
                      consistency_level="Strong")
    for q in qs:
        t = time.perf_counter()
        r = client.search(name, data=[q], anns_field="dense", limit=TOPK, filter=filter,
                          output_fields=["id"],
                          search_params={"metric_type": "COSINE", **({"params": params} if params else {})},
                          consistency_level="Strong" if strong else store.CONSISTENCY)
        lat.append((time.perf_counter() - t) * 1000)
        hits.append([h["entity"]["id"] for h in r[0]])
    return lat, hits


def rtt(dim):
    """查询链路的地板：在一个只有 1 行的集合上做 search，算力可以忽略，剩下的全是网络。

    别拿 describe_collection 当探针——实测它 592ms，比同一时刻 3883 条的真实 search
    (299ms) 还慢一倍，因为它走的是元数据/协调节点那条路，不是查询路径。量错基线比不量更糟：
    会得出"所有索引都比网络还快"这种结论。
    """
    build("bench_rtt", [{"id": "x", "dense": [0.0] * (dim - 1) + [1.0]}], dim, "FLAT", {})
    lat = []
    for _ in range(11):
        t = time.perf_counter()
        client.search("bench_rtt", data=[[0.0] * (dim - 1) + [1.0]], anns_field="dense", limit=1,
                      output_fields=["id"], search_params={"metric_type": "COSINE"},
                      consistency_level="Strong")
        lat.append((time.perf_counter() - t) * 1000)
    drop("bench_rtt")
    return statistics.median(lat[1:])  # 丢掉第一次：拉连接的开销不算地板


def p(xs, q):
    return sorted(xs)[min(len(xs) - 1, int(len(xs) * q))]


def line(lat):
    return f"{statistics.median(lat):7.2f} {p(lat, .95):7.2f} {p(lat, .99):7.2f}"


# ---------- index ----------
def cmd_index(a):
    if not store.CLUSTER:
        # 实测过的，不是猜的：milvus-lite 3.2.0 上 M=4 和 M=64 建出来的索引 recall 一模一样
        # （5000 条 × 64 维随机向量，都是 0.6267），ef=10 和 ef=500 也一模一样；
        # IVF_FLAT 填 nlist=64 和 256，日志里都打印 "clustering to 128 centroids"，
        # 而 nprobe=1 就能拿到 recall=1.0——它压根没在做 IVF。describe_index 也不回参数。
        # 也就是说：这一步的数只有连集群才有意义，本地跑只能验证脚本本身没写错。
        print("!! 当前连的是 Milvus Lite（本地文件），它会把索引参数吃掉：M/efConstruction/"
              "nlist/ef/nprobe 都不生效。\n"
              "!! 下面这张表只能证明 bench 跑得通，不能用来选型。选型要连 standalone 或集群：\n"
              "!!   docker compose up -d && MILVUS_URI=http://localhost:19530 python bench.py index\n")
    data = items(a.limit)
    dim = embedding.dim()
    vecs = embedding.encode([d["text"] for d in data])
    qs = queries(a.queries, {d["id"]: v for d, v in zip(data, vecs)})
    rows_ = inflate([{"id": d["id"], "dense": v} for d, v in zip(data, vecs)], a.scale)
    print(f"{len(rows_)} 条 × {dim} 维"
          f"{f'（{len(data)} 条真实语料 × {a.scale} 倍扰动复制）' if a.scale > 1 else ''}，"
          f"{len(qs)} 个真实用户兴趣向量当查询，top_k={TOPK}\n")

    # ground truth 必须用 FLAT 暴力搜。拿另一个近似索引当基准，量出来的是"两个近似算法
    # 有多像"，不是召回率——两个都漏掉的那部分永远不会出现在分母里。
    bt = build("bench_gt", rows_, dim, "FLAT", {})
    floor = rtt(dim) if store.CLUSTER else 0.0
    if store.CLUSTER:
        print(f"查询链路地板 {floor:.1f}ms（1 行集合上的 search，几乎全是网络）"
              f"—— 下面所有延迟都含这一份，配置之间的差值才是索引的差别\n")
    gt_lat, gt = run("bench_gt", qs, {})
    print(f"{'索引':<10}{'建索引参数':<28}{'查询参数':<16}{'建索引s':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'recall@10':>11}")
    print(f"{'FLAT':<10}{'-':<28}{'-':<16}{bt:8.1f}{line(gt_lat)}{1.0:11.3f}   ← ground truth")

    table = []
    for index_type, params, sweeps in CANDIDATES:
        name = "bench_idx"
        try:
            bt = build(name, rows_, dim, index_type, params)
        except Exception as e:
            print(f"{index_type:<10}{str(params):<28}建索引失败：{str(e)[:60]}")
            continue
        for sp in sweeps:
            lat, got = run(name, qs, sp)
            recall = statistics.mean(len(set(g) & set(t)) / len(t) for g, t in zip(got, gt))
            print(f"{index_type:<10}{str(params):<28}{str(sp):<16}{bt:8.1f}{line(lat)}{recall:11.3f}")
            table.append((statistics.median(lat), recall, index_type))
        if not a.keep:
            drop(name)
    if not a.keep:
        drop("bench_gt")
    verdict(table, floor, len(rows_), a.scale)


def verdict(table, floor, n, scale):
    """**这张表能不能用来选型，得由数据自己回答，而且要按两根轴分开问。**

    索引选型是延迟-召回的取舍，所以有两根轴，它们各自可能分得开也可能分不开：

      延迟轴：p50 的极差要高过网络地板的抖动，不然"挑 p99 最低的"挑的是运气
      召回轴：recall 的极差要够大，不然任何索引都能找到真正的最近邻

    关键是**这两根轴不是同时活过来的**，所以不能只判一句"分不开"。实测：3883 条时两根
    都是死的（recall 极差 0.007、p50 极差 1.6ms 对着 74ms 的地板）；放大到 116490 条，
    召回轴活了、延迟轴还是死的（recall 0.733→0.913，p50 全在 78-87ms 里）。后者是**能**
    选型的——所有配置延迟一样，召回差 18 个百分点，按召回选就行；笼统判一句"分不开"
    就把这个结论丢了，而它恰好是最有用的那个。

    所以下面四个分支是四种不同的建议，不是一个判断的四种措辞。
    """
    if not table:
        return
    lat = [t[0] for t in table]
    rec = [t[1] for t in table]
    d_lat = max(lat) - min(lat)
    d_rec = max(rec) - min(rec)
    # 地板的 1/4 当抖动阈值：地板 70ms 时要求 p50 极差 >17ms 才算真差别
    lat_live = d_lat >= max(floor * 0.25, 3.0)
    rec_live = d_rec >= 0.02
    print(f"\n判定：延迟轴 p50 极差 {d_lat:.1f}ms vs 网络地板 {floor:.1f}ms → "
          f"{'分得开' if lat_live else '**分不开**'}"
          f"\n      召回轴 recall 极差 {d_rec:.3f} → {'分得开' if rec_live else '**分不开**'}")

    if not lat_live and not rec_live:
        print(f"      两根轴都是死的 → **这一档别选型，AUTOINDEX 别动**，"
              f"省下的时间去调召回。"
              f"\n      {n} 条太少：向量计算是微秒级，全被一次网络往返盖住；"
              f"图里点太少，怎么走都走得到最近邻。"
              f"\n      要量出曲线：--scale N 放大语料（当前 {scale}）。")
    elif rec_live and not lat_live:
        best = max(table, key=lambda t: t[1])
        auto = [t[1] for t in table if t[2] == "AUTOINDEX"]  # 按名字找，别靠它排在最后
        print(f"      **只有召回轴活着 → 按召回选，延迟不要钱。** 这一档所有配置的 p50 都在"
              f" {min(lat):.0f}-{max(lat):.0f}ms 之间，"
              f"\n      差别在网络抖动里；而召回从 {min(rec):.3f} 到 {max(rec):.3f}，"
              f"差 {d_rec * 100:.0f} 个百分点——**同样的延迟，白拿的召回**，"
              f"\n      挑 recall 最高那一档：{best[2]} recall={best[1]:.3f}。"
              + (f"\n      而 AUTOINDEX 这一轮 {max(auto):.3f}，"
                 f"{'垫底' if max(auto) == min(rec) else '排在后面'}——"
                 f"「让 Milvus 自己挑」在这个量级上开始要付代价了。" if auto else ""))
    elif lat_live and not rec_live:
        print(f"      只有延迟轴活着 → 召回都一样，直接挑 p99 最低的那一档。")
    else:
        print(f"      两根轴都活着 → 这才是教科书里那条延迟-召回曲线：先划一条召回底线"
              f"（比如 0.95），再在够线的里面挑 p99 最低的。"
              f"\n      HNSW 的 ef 是查询期旋钮——同一个索引可以按接口给不同的 ef，"
              f"搜索给高的、推荐给低的。")

    if scale > 1:
        # 放大语料是拿真实向量扰动复制出来的，每个原始向量周围有 scale-1 个近邻克隆。
        # top-10 的 ground truth 于是变成"同一批克隆里挑 10 个"，这是个近似平局，
        # ANN 换几个同分的邻居就掉 recall——所以绝对值偏低，别当成真实语料上的召回。
        print(f"      **--scale {scale} 的 recall 绝对值偏低,别直接引用**："
              f"每个原始向量有 {scale - 1} 个扰动克隆，"
              f"\n      top-{TOPK} 的正确答案里全是彼此近似平局的克隆，ANN 换几个同分邻居"
              f"就掉召回。能用的是配置之间的**相对高低**，不是这个数本身。"
              f"\n      要真实的召回绝对值，得有 {n} 条真正互不相同的数据。")


# ---------- filter ----------
def cmd_filter(a):
    data = items(a.limit)
    dim = embedding.dim()
    vecs = embedding.encode([d["text"] for d in data])
    qs = queries(a.queries, {d["id"]: v for d, v in zip(data, vecs)})
    floor = rtt(dim) if store.CLUSTER else 0.0

    # 按 year 的分位数取阈值往 1% / 10% / 50% 三档凑，然后**回头数一遍真实命中率**。
    # 不能直接把目标值当选择率写上去：电影年份重复得厉害（几百部都是 1996），
    # 分位数一撞上并列值就跳过一大片，1% 那档实际可能是 8%。表里报的是数出来的那个数。
    years = sorted(d["year"] for d in data if d["year"] is not None)
    n = len(data)
    cases = []
    for target in (0.01, 0.10, 0.50):
        y = years[min(len(years) - 1, int(len(years) * (1 - target)))]
        expr = f"year >= {y}"
        # 两档撞上同一个年份就只留一份：并列值多的时候（--limit 800 下 1% 和 10% 都落到
        # 1996）会塌成同一个条件，表里出两行一模一样的过滤条件、两组不一样的延迟，
        # 读者只会以为脚本写错了。少一行比多一行假对比好。
        if any(c[0] == expr for c in cases):
            continue
        cases.append((expr, sum(d["year"] is not None and d["year"] >= y for d in data) / n))

    dyn = [{"id": d["id"], "dense": v, "category": d["category"], "year": d["year"]}
           for d, v in zip(data, vecs)]  # 无显式字段 → 这两个键落进 dynamic field
    exp = dyn
    b1 = build("bench_dyn", dyn, dim, "HNSW", {"M": 16, "efConstruction": 200}, dynamic=True)
    b2 = build("bench_exp", exp, dim, "HNSW", {"M": 16, "efConstruction": 200},
               scalar=True, dynamic=True)
    print(f"{len(dyn)} 条，{len(qs)} 个查询，HNSW(M=16)，建索引 dynamic {b1:.1f}s / 显式+INVERTED {b2:.1f}s")
    if store.CLUSTER:
        print(f"查询链路地板 {floor:.1f}ms —— 两列都含这一份")
    print()
    print(f"{'过滤条件':<20}{'实际命中':>8}{'':>3}{'dynamic p50':>12}{'p95':>8}{'p99':>8}"
          f"{'  |':>3}{'INVERTED p50':>13}{'p95':>8}{'p99':>8}")
    gaps = []
    for f, r in cases + [("（不过滤）", 1.0)]:
        expr = "" if f.startswith("（") else f
        l1, _ = run("bench_dyn", qs, {"ef": 64}, expr)
        l2, _ = run("bench_exp", qs, {"ef": 64}, expr)
        print(f"{f:<20}{r:7.0%}{'':>3}{line(l1):>28}{'  |':>3}{line(l2):>29}")
        if expr:
            gaps.append(statistics.median(l1) - statistics.median(l2))
    if not a.keep:
        drop("bench_dyn"), drop("bench_exp")

    # 同一条纪律：两列分不开的时候必须自己说出来，不能照着噪声下结论。
    # 这里的分不开和 index 那边不是一个原因——标量过滤的代价是真的存在（Milvus Lite 上
    # 量到稳定 3 倍），只是它是毫秒级的，被几十毫秒的网络盖住了。
    if store.CLUSTER and max(gaps, default=0) < max(floor * 0.15, 3.0):
        print(f"\n判定：**两列的差别小于网络抖动（最大 {max(gaps, default=0):.1f}ms vs 地板 "
              f"{floor:.1f}ms），这张表分不开。**"
              f"\n      不是 dynamic field 不要钱——它的代价是每行解一次 JSON，毫秒级，"
              f"在本地 Milvus Lite 上量到稳定 3 倍（见 README）；"
              f"\n      是远程集群上这几毫秒被一次网络往返盖住了。反过来说：**这一档数据量下，"
              f"为了过滤性能去改 schema 不划算**，"
              f"\n      提成显式字段的理由是类型约束（year 不会再是字符串），不是延迟。"
              f"\n      要量延迟差别：本地 Lite 上跑这条命令（没网络），或者上到十万量级。")
    else:
        print("\n看法：选择率越低（过滤掉的越多），HNSW 越容易在图里走不到足够的候选，"
              "\n      Milvus 会退化成带过滤的暴力扫——这时候延迟由标量过滤的代价主导，标量索引才显出差别。")


# ---------- consistency ----------
def cmd_consistency(a):
    """整篇最值得跑的一条：同一段"写入 + 立刻搜"，两种一致性下的成功率差多少。

    在 Milvus Lite 上两边都会 100%（进程内，写完就可见），这条只有连集群才有意义——
    差异不在代码里，在部署形态里，这正是 Day 1 要说的事。
    """
    dim = embedding.dim()
    build("bench_cons", [{"id": "seed", "dense": embedding.encode(["seed"])[0]}], dim,
          "HNSW", {"M": 16, "efConstruction": 200})
    print(f"{'一致性':<12}{'写完立刻查到':>14}{'查询 p50':>10}{'p95':>9}{'p99':>9}")
    for level in ("Bounded", "Strong"):
        lat, found = [], 0
        for i in range(a.queries):
            key = f"cons-{level}-{i}"
            vec = embedding.encode([key])[0]
            client.upsert("bench_cons", [{"id": key, "dense": vec}])
            t = time.perf_counter()
            r = client.search("bench_cons", data=[vec], anns_field="dense", limit=1,
                              output_fields=["id"], search_params={"metric_type": "COSINE"},
                              consistency_level=level)
            lat.append((time.perf_counter() - t) * 1000)
            found += any(h["entity"]["id"] == key for h in r[0])
        print(f"{level:<12}{found}/{a.queries:<11}{line(lat)}")
    if not a.keep:
        drop("bench_cons")
    print("\n看法：Bounded 查的是一个稍旧的时间戳，写完立刻读会偶发落空；Strong 等数据可见再返回。"
          "\n      别整个服务开 Strong——只给「写完要回查」的那条路径开（/search 的 strong=true）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["index", "filter", "consistency"])
    ap.add_argument("--limit", type=int, default=0, help="只灌前 N 条（0=全量）")
    ap.add_argument("--queries", type=int, default=200, help="查询条数")
    ap.add_argument("--scale", type=int, default=1,
                    help="把语料扰动复制放大 N 倍（index 子命令用）。"
                         "3883 条量不出索引差别，要 --scale 25 以上才分得开")
    ap.add_argument("--keep", action="store_true", help="跑完保留 bench_ 集合")
    a = ap.parse_args()
    print(f"连接 {store.URI}（{'集群' if store.CLUSTER else '本地文件'}）· "
          f"embedding {embedding.signature()}\n")
    {"index": cmd_index, "filter": cmd_filter, "consistency": cmd_consistency}[a.cmd](a)
