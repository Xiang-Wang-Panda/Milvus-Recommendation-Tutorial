# Day 1:上生产、扛住量

Day 0 的那套代码搬上集群。**八个业务接口算什么全没变**,`embedding.py` 一行没动;
变的是怎么连、索引怎么建、数据属于谁、读到的是哪个时刻的数据。

`main.py` 那 86 行改动分两类:八个接口各多收一个租户、各多一个埋点(机械改动,不是业务逻辑);
另外**新增了两个接口** `/metrics` 和 `/admin/compact`——它们不是业务,是运维面,
但确实是新的口子,别说成"接口没动"。

| 文件 | 相对 Day 0 | 代码行 diff |
| --- | --- | --- |
| `store.py` | 连集群认证、索引可调、`tenant_id` partition key、`category`/`year` 提成显式字段、一致性可选、批量取向量 | 136 → 227 行,改动 195 |
| `main.py` | 每个请求带 `X-Tenant-Id`、`/metrics`、`/admin/compact` | 67 → 97 行,改动 86 |
| `embedding.py` | 无改动 | 0 |
| `seed.py` | 带租户头、灌完自动 compact | 29 |
| `bench.py` | 新增。索引选型 / 过滤对比 / 一致性,三个子命令 | — |
| `test_tenant.py` | 新增。越权查询测试(安全边界,不是功能测试) | — |
| `docker-compose.yml` | 新增。Milvus standalone + Attu + Prometheus/Grafana | — |

行数是剔掉注释和 docstring 之后的纯代码。**195 行里 Step 1「换连接串上集群」只占 14 行**
(`_credentials()` 12 行 + `CLUSTER` 判断 + `MilvusClient` 那一行),其余 181 行是 Step 2-5
新加的功能——租户、索引旋钮、一致性、显式字段。这两个数别混着说:**移植到集群几乎免费,
贵的是上生产要补的那些能力。**

## 跑起来

```bash
pip install -r requirements.txt

# A. 还是本地文件(Day 0 那样,零依赖)
uvicorn main:app --reload
python seed.py

# B. 起个 standalone 集群,只换连接串
docker compose up -d                       # 等 90s,healthcheck 转绿
export MILVUS_URI=http://localhost:19530
uvicorn main:app                           # 同一份代码
python seed.py                             # 同一份 seed
open http://localhost:8001                 # Attu:看 collection / 索引进度 / segment

# C. Zilliz Cloud(下面所有集群数字都在这上面量的:gcp-asia-southeast1 专属集群)
export MILVUS_URI=https://xxx.zillizcloud.com MILVUS_TOKEN=xxx
uvicorn main:app
python seed.py
```

三条路只有 `MILVUS_URI`(和认证)不同。`store.py` 里判断的地方只有一处:
`URI.startswith("http")`。

实测在 Zilliz Cloud 上一把过:`init()` 建两个集合 + 索引 10.2s,`seed.py` 灌
3883 商品 + 6038 用户,八个接口(BM25 / 向量 / filter / 推荐 / 相似用户 / 画像 / count / compact)
结果全对,`test_tenant.py` 14 条断言全过。

**灌数据的时间是这么分的**,这个比例本身就是 Day 1 的第一课:

| | 改之前 | 改之后 |
| --- | --- | --- |
| items 3883 条 | 2.5s | 2.6s |
| users 6038 条 | **707.4s** | **46.5s** |
| `seed.py` 全程 | 11 分 51 秒 | **50 秒** |

差别不在 Milvus,在 `add_users` 是"循环里查库":Day 0 一个用户一次 `query`,
进程内几十微秒,6038 次也就一眨眼——**上集群之后同一段代码是 6038 次网络往返**。
改成一次拿一批(`_vec_map`,去重 + 500 一批),往返数从"用户数"降到"去重后的物品数 / 500"。

两个细节:去重是关键的一半(6038 个用户的历史去重后只有 3000 多个物品,热门电影出现在
几百个人的历史里);另一半是 items 那 2.5s 早就说明了问题——**同样是灌 3883 条,
写是一次批量、读是几千次单条**,两个数差 280 倍,瓶颈从来不在 Milvus 那边。
这类代码在本地永远不显形,因为本地没有往返这回事。

## Day 1 新增的环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `MILVUS_TOKEN` | — | 集群认证。Zilliz Cloud 给的一整串 token |
| `MILVUS_USER` / `MILVUS_PASSWORD` | — | 自建集群的账号密码(默认 `root`/`Milvus`) |
| `MILVUS_DB` | — | 连哪个 database。库级隔离用 |
| `MILVUS_CONSISTENCY` | `Bounded` | `Strong` / `Bounded` / `Session` / `Eventually` |
| `MILVUS_INDEX_TYPE` | `AUTOINDEX` | `HNSW` / `IVF_FLAT` / `FLAT` |
| `MILVUS_INDEX_PARAMS` | 按类型给默认 | JSON,如 `{"M":16,"efConstruction":200}` |
| `MILVUS_SEARCH_PARAMS` | 按类型给默认 | JSON,如 `{"ef":64}` |
| `MILVUS_BM25_PARAMS` | 空(Milvus 默认 k1=1.2 b=0.75) | JSON,如 `{"bm25_k1":1.5,"bm25_b":0.5}` |
| `MILVUS_REPLICAS` | `1` | `load_collection` 的副本数,>1 才传 |
| `MILVUS_TENANT_PARTITIONS` | `16` | partition key 的桶数 |
| `DEFAULT_TENANT` | `default` | 请求没带 `X-Tenant-Id` 时用它 |

改索引类型在这份代码里要重建集合 + 重灌(`MILVUS_INDEX_TYPE=HNSW` 之后 `drop` 掉 items 再
`seed` 一遍),但**这是本实现的限制,不是 Milvus 的限制**——`init()` 把索引定义放在
`create_collection` 里了,所以换索引就得换集合。

Milvus 本身允许原地换,实测(50 条数据,AUTOINDEX → HNSW):

```python
client.release_collection("items")
client.drop_index("items", "dense")
client.create_index("items", idx)      # HNSW, M=16, efConstruction=200
client.load_collection("items")
# count(*) 仍是 50，查询结果不变 —— 数据一行没动
```

这个区别很要紧:**索引选型是可逆决定,可以灌完数据再慢慢调**,不该被算进"灌数据前必须
定下来"的那一类。要在服务里支持热换索引,把索引创建从 `init()` 里拆出来单独一个
管理接口就行。

---

## 五步各自在哪

### Step 1:换连接串上集群

`_credentials()` 一个函数,十行。本地路径返回空字典,http(s) 才带凭据。

两个跟着变的东西:

- **embedding 指纹从旁路文件搬进 collection properties**。Day 0 写在 `milvus.db.embedding`
  里,单进程单文件没问题;多实例部署时每个实例各写一份本地文件,两个实例配了不同
  `EMBED_MODEL` 会互相触发重建。现在存在 items 的 property 上,单一事实源。
  **两边都实测过了**:milvus-lite 3.2.0 上 properties 进程重开后仍在(description 会丢,
  properties 不会);Zilliz Cloud 上自定义 key `app.embedding.signature` 写得进、读得回
  (`init()` 之后 `describe_collection` 拿到 `hash:512`)。所以本地和集群同一套代码,
  这条不需要按部署分叉。
- **`MILVUS_URI` 和 pymilvus 自己 config 撞名的坑消失了**。Day 0 必须在 `import pymilvus`
  之前 `os.environ.pop`,因为 pymilvus 在 import 阶段读这个变量且只认 `http[s]://`。
  上了集群值本来就是 http 地址,冲突自己没了——但那行 pop 还得留着,不然本地模式又会挂。

### Step 2:索引调优

```bash
python bench.py index --queries 200
```

FLAT 暴力搜当 ground truth,扫 HNSW(三组 M/efConstruction × 四档 ef)和
IVF_FLAT(两组 nlist × 三档 nprobe),每组出 p50/p95/p99、recall@10、建索引耗时。
查询向量是从 `users.csv` 采的真实用户兴趣向量,不是随机向量——随机向量在高维里
彼此近似正交,召回率会虚高(Day 0 的 hash 后端已经演示过:哈希向量基本正交,
`recommend` 的分数全是 0.0)。

> **这一步在 Milvus Lite 上做不了,必须连集群。** 实测 milvus-lite 3.2.0 会把索引参数吃掉:
> 5000 条 × 64 维随机向量,HNSW `M=4` 和 `M=64` 建出来 recall 都是 0.6267,`ef=10` 和
> `ef=500` 也一模一样;`IVF_FLAT` 填 `nlist=64` 和 `nlist=256`,日志里都打印
> `clustering to 128 centroids`,而 `nprobe=1` 就能拿到 `recall=1.0`——它根本没在做 IVF。
> `describe_index` 也不返回参数。本地跑这个命令只能验证脚本没写错,`bench.py` 会先打一行警告。

**连了集群之后又撞上第二堵墙:Day 0 的数据量根本量不出索引的差别。** 实测 3883 条 × 512 维:

```
查询链路地板 74.1ms(1 行集合上的 search,几乎全是网络)

索引        建索引参数                       查询参数          p50     p95     p99  recall@10
FLAT      -                           -             73.10   84.56   92.29      1.000  ← ground truth
HNSW      {'M': 8,  'efC': 100}       {'ef': 16}    78.91  138.84  151.37      0.991
HNSW      {'M': 8,  'efC': 100}       {'ef': 128}   78.92   83.98   92.09      0.998
HNSW      {'M': 16, 'efC': 200}       {'ef': 32}    79.55  345.53  459.98      0.998
HNSW      {'M': 32, 'efC': 400}       {'ef': 128}   76.37   87.06  111.51      0.998
IVF_FLAT  {'nlist': 64}               {'nprobe': 1} 78.58   89.83  111.35      0.993
IVF_FLAT  {'nlist': 256}              {'nprobe': 64}79.34   91.69  108.74      0.998
AUTOINDEX {}                          {}            79.97  110.97  198.13      0.995
```

两件事同时发生,让这张表选不出东西:

- **recall 全是 0.99+,极差 0.007。** 几千个点的 HNSW 图,怎么走都能走到真正的最近邻——
  召回率这个旋钮在这个数据量下是不存在的。
- **p50 全挤在 74.5-80.0ms(极差 5.5ms),而网络地板是 74.1ms。** 3883 条的向量计算是微秒级,一次查询里
  95% 的时间是网络往返。配置之间的 p99 差别(92ms 到 460ms)是抖动,不是索引——
  照着它挑,挑的是运气。

所以 `bench.py` 现在会**自己判定这张表能不能用来选型**,不能就直接说出来:

```
判定:**这一档量不出索引的差别,别拿它选型。**
      recall 极差 0.007(3883 条太少,任何索引都能找到真正的最近邻)
      p50 极差 5.5ms vs 网络地板 74.1ms(延迟差别全被网络吃掉,剩下的是抖动)
      这一档的正确结论就是 **AUTOINDEX 别动**,省下的时间去调召回。
      要真的量出曲线:--scale N 把语料放大,到十万量级配置之间才分得开。
```

**一张选不出东西的表必须自己说出来**,否则读者会照着噪声调参数,还以为调对了。
这也是 Day 0 选 AUTOINDEX 的事后追认:在你的数据长到十万量级之前,索引选型不是你的问题。

要真的量出那条曲线,得先有量:

```bash
python bench.py index --scale 30 --queries 100   # 3883 × 30 = 116k 条
```

`--scale` 是**扰动复制真实向量**,不是生成随机向量:随机向量在高维里彼此近似正交,会在
真实语料周围铺一层"谁都不像谁"的背景噪声,近似索引轻松跳过它们,recall 虚高
(Day 0 的 hash 后端已经演示过这个现象)。扰动复制保留原语料的聚簇结构——真实数据里
就是有一堆彼此很像的邻居,那才是索引会走错的地方。用 LCG 而不是 `random`,同一个
`--scale` 每次跑出同一份语料,两次 bench 的数才可比。

**放大到 116490 条,一根轴活了,另一根还是死的:**

```
查询链路地板 70.6ms

索引        建索引参数                 查询参数           建索引s    p50     p99  recall@10
FLAT      -                       -                92.0  76.71  120.28      1.000  ← ground truth
HNSW      {'M': 8,  'efC': 100}   {'ef': 16}       66.3  78.31  102.06      0.846
HNSW      {'M': 8,  'efC': 100}   {'ef': 128}      66.3  79.35  255.28      0.881
HNSW      {'M': 16, 'efC': 200}   {'ef': 16}       65.7  81.25  147.86      0.842
HNSW      {'M': 16, 'efC': 200}   {'ef': 128}      65.7  82.34  167.28      0.869
HNSW      {'M': 32, 'efC': 400}   {'ef': 64}       71.0  85.84  184.46      0.913  ← 最高
HNSW      {'M': 32, 'efC': 400}   {'ef': 128}      71.0  83.25  150.62      0.913  ← 同分但 p99 更低
IVF_FLAT  {'nlist': 64}           {'nprobe': 16}   75.9  84.12  110.21      0.904
IVF_FLAT  {'nlist': 256}          {'nprobe': 4}    78.3  82.05  322.71      0.759
IVF_FLAT  {'nlist': 256}          {'nprobe': 64}   78.3  80.95  130.13      0.869
AUTOINDEX {}                      {}               94.0  80.38  302.83      0.733  ← 垫底
```

```
判定:延迟轴 p50 极差 8.9ms vs 网络地板 70.6ms → **分不开**
      召回轴 recall 极差 0.180 → 分得开
      **只有召回轴活着 → 按召回选,延迟不要钱。**
      挑 recall 最高那一档:HNSW recall=0.913。
      而 AUTOINDEX 这一轮 0.733,垫底——「让 Milvus 自己挑」在这个量级上开始要付代价了。
```

**这就是大纲要的那句"哪一档是你要的",而且答案跟直觉不一样:**

- 延迟轴还是死的(p50 全在 78-87ms,极差 8.9ms 对着 70.6ms 的地板)。所以这一档
  **不存在"延迟-召回取舍"——召回是白拿的**。教科书上那条曲线要等延迟轴也活过来才出现,
  而在远程集群上它可能一直不活,因为地板太高。
- 召回轴活了,而且 **AUTOINDEX 垫底(0.733)**。3883 条那一档的结论是"AUTOINDEX 别动",
  116490 条这一档反过来了:同样的延迟,`HNSW M=32/efC=400` 白给 18 个百分点的召回。
  **"让 Milvus 自己挑"是有保质期的。**
- `M=32/ef=64` 和 `ef=128` 召回同分(0.913),但 `ef=128` 的 p99 更低(150 vs 184)。
  同分的时候看尾巴。

> **`--scale` 的 recall 绝对值偏低,别直接引用。** 每个原始向量有 29 个扰动克隆,top-10 的
> 正确答案里全是彼此近似平局的克隆,ANN 换几个同分邻居就掉召回——所以 0.913 不代表真实
> 语料上的 91%。**能用的是配置之间的相对高低,不是这个数本身。** 要真实的绝对值,
> 得有 11 万条真正互不相同的数据。`bench.py` 会把这句警告打在表下面。

`compact` 那件事在 `/admin/compact`:BM25 的 idf 按 segment 统计,新写入的行独占一个新
segment,常见词在它那儿 idf≈0 会被老数据压下去——Day 0 前端那条"回查用标题别带类型词"
的注释就是在绕这个。灌完数据 compact 一次,统计量拉平,排序才全局可比。`seed.py` 现在
自动跑一次。

**但托管服务上这件事你控制不了**,四种部署四种行为(都是实测):

| 部署 | `client.compact()` 返回 |
| --- | --- |
| milvus-lite | 同步做完,job id `0` |
| 自建 standalone | 真的 job id,后台异步 |
| Zilliz Cloud 专属集群 | job id **`-1`** —— 请求收下了,但不给你一个能查的 job |
| Zilliz Cloud serverless | `PERMISSION_DENIED` |

`-1` 这个返回值最坑:**不报错,也不代表它做了**,你既不能确认它跑了也不能等它跑完。
所以别把它放在"灌完数据必须成功"的关键路径上,降级成一条说明。

#### BM25 的旋钮:`k1` / `b` 确实能调,但验证方法反直觉

稀疏索引这边没有 HNSW 那种延迟-召回取舍,能调的是 `k1`(词频饱和:一个词出现 8 次值不值
8 倍)和 `b`(长度归一:长文档该罚多少)。**Milvus 确实暴露它们**,填在 sparse 索引的
`params` 里,环境变量 `MILVUS_BM25_PARAMS='{"bm25_k1":1.5,"bm25_b":0.5}'`。

但**别用 `describe_index` 验证它生效了**。实测集群会把你传的任何键原样存下来再回显,
包括 `bogus_param`:

| 索引参数 | doc1(`star`×8,短) | doc2(长文) | doc3(短) |
| --- | --- | --- | --- |
| `{}` 默认 | 0.2771 | 0.0778 | 0.2191 |
| `{"bogus_param": 1}` | 0.2771 | 0.0778 | 0.2191 |
| `{"bm25_k1":0.1,"bm25_b":0.0}` | 0.1451 | **0.1335** | 0.1335 |
| `{"bm25_k1":3.0,"bm25_b":1.0}` | 0.4963 | **0.0577** | 0.4705 |

`bogus_param` 那行和默认**一模一样**——回显不代表被读。真正的判据是分数变不变:
`b=0.0` 不做长度归一,长文档 doc2 从 0.0778 涨到 0.1335 **追平了短文档**;`b=1.0` 把它
压到 0.0577。这是 `b` 在起作用的直接证据。

**方法论比结论重要:** milvus-lite 上这四组全部"接受"且 `describe_index` 什么参数都不回,
集群上四组全部"接受"且什么参数都回——两种部署下"接受"和"回显"都没有判别力。
**能判别的只有行为。**

### Step 3:标量过滤扛住量

`category`(VARCHAR, nullable)和 `year`(INT64, nullable)从 dynamic field 提成显式字段,
各建一个 INVERTED 索引。dynamic field 留着,给低频/不稳定的字段。

#### 先纠一个前提:**dynamic field 里的 key 是能建索引的**

这一步原来的理由是"dynamic field 建不了标量索引"。**实测不对**——Milvus 2.6 支持给
JSON 路径建索引,dynamic field 就是一个叫 `$meta` 的 JSON 字段:

```python
idx.add_index(field_name="$meta", index_type="INVERTED", index_name="ix_year",
              params={"json_path": '$meta["year"]', "json_cast_type": "double"})
# 集群上建成了：list_indexes() -> ['dense', 'ix_year']
```

`json_cast_type` 是必填的,漏了报 `json index must specify cast type`。有意思的是
`field_name="year"` 直接写 key 名也走同一条路(报的是同一个缺 cast type 的错),
也就是说 Milvus 认得出你指的是 dynamic field 里的 key。

**所以"提成显式字段"的理由要重写。** 剩下的理由还有三条,而且都比"建不了索引"更硬:

1. **类型约束**。JSON 里的 `"1999"` 和 `1999` 是两个东西,`json_cast_type` 只在索引侧
   做转换,写入侧不拦你。显式 INT64 字段会在写入时就拒绝——Day 0 那个坑靠的是这个,
   不是索引。
2. **不用为每个 key 声明一次索引 + cast type**。schema 里写字段类型一次,索引跟着走。
3. **能 nullable**。JSON 里"没有这个 key"和"这个 key 是 null"要靠表达式区分,
   显式 nullable 字段的 null 语义是明确的(不参与比较)。

延迟不在这三条里面——见下面集群实测,这个数据量下延迟差别量不出来。

**`year` 用 INVERTED 不用 STL_SORT**:milvus-lite 3.2 只实现了 INVERTED
(填 STL_SORT 直接报 `scalar index_type 'STL_SORT' is not implemented`),而 INVERTED
本身支持范围查询,所以两边统一,本地和集群同一套代码。

`nullable=True` 不是可选项:`products.csv` 里有一批电影没年份,不给 nullable 这些行写不进去;
补个 0 又会污染 `year >= 1995` 这种过滤(0 也是个数)。null 参与不了比较,正好。

实测(3883 条,40 个查询,HNSW,**Milvus Lite**——这一档不涉及向量索引参数,所以本地数有参考意义):

```
过滤条件            实际命中   dynamic p50  p95     p99   |  INVERTED p50  p95    p99
year >= 2000          4%         6.87    33.44  102.69  |      2.25     2.47    2.51
year >= 1999         11%         6.81     7.12    7.29  |      2.40     2.73    2.73
year >= 1994         52%         7.04     7.40    8.32  |      2.48     2.75    2.80
(不过滤)             100%         2.54     2.97    2.97  |      2.50     2.67    2.68
```

三件事:不过滤时两边完全一样(2.5ms),差别百分之百出在过滤路径上;带过滤时 dynamic field
稳定慢 3 倍——值在 JSON 里,每行都要解析;选择率最低的那档(4%)dynamic 的 p99 炸到 102ms,
因为过滤掉太多之后向量索引走不到足够候选,退化成带过滤的扫描,这时候单行过滤的代价被放大。

**同一条命令连集群跑,这个 3 倍消失了:**

```
查询链路地板 75.2ms —— 两列都含这一份

过滤条件         实际命中  dynamic p50   p95     p99  | INVERTED p50   p95     p99
year >= 2000        4%        80.21  143.90  413.67  |       79.28   85.23  131.58
year >= 1999       11%        79.45   85.15  152.68  |       79.74  105.80  126.10
year >= 1994       52%        79.79   88.43   89.81  |       77.22  112.20  301.88
(不过滤)          100%        79.28   86.43   89.92  |       77.34   96.90  105.39

判定:**两列的差别小于网络抖动(最大 2.6ms vs 地板 75.2ms),这张表分不开。**
```

**这是 Day 1 最反直觉的一处:两种部署,各自只量得准一半。**

| | 向量索引参数(M/ef/nlist) | 标量过滤代价(dynamic vs INVERTED) |
| --- | --- | --- |
| Milvus Lite | 量不了(参数被吃掉) | **量得准**(没网络,毫秒级差别看得见) |
| 远程集群 | 量得准 | 量不了(几毫秒被 75ms 往返盖住) |

所以这两张表都得留着,一张是本地的一张是集群的,谁也替代不了谁。而且推论很实际:
**在这个数据量下,为了过滤性能去改 schema 不划算。** 把 `category`/`year` 提成显式字段的
真正理由是**类型约束**(`year` 不会再是字符串),不是延迟——延迟那部分要等数据长到十万量级
或者查询走内网才回来。

`year` 存成字符串导致数值过滤静默失效那个 Day 0 的坑,现在犯不了。实测三种输入:

| 传进来的 year | 结果 |
| --- | --- |
| `1999` | 正常 |
| `"1999"` | pydantic 转成 `1999` 存进去,`year >= 1999` 照样筛得到(**不是报错,是纠正**) |
| `"abc"` | 422 |

绕过 pydantic 直接调 `store.add_items` 的话由 Milvus schema 兜底(`DataNotMatchException`,
翻译成 400)。两层都不会让一个字符串安静地躺在 `year` 里——这是上生产白捡的好处。

```bash
python bench.py filter --queries 200
```

### Step 4:多租户隔离

`tenant_id` 在两个集合上都是 `is_partition_key=True`,桶数 `MILVUS_TENANT_PARTITIONS`。

**桶数不是租户数上限,这两个经常被搞混。** 租户是 hash 进桶的,多个租户共享一个桶,
所以租户数本身没有上限——受限的是桶数。实测 Zilliz Cloud:`num_partitions` 填 1024、
4096、8192、100000 全部**建成 + 写入 + load + 带 `tenant_id` 过滤的查询都通**。
也就是说桶数在这条链路上没有实际天花板。

那桶数该填多少?判据不是租户数,是**倾斜**:桶开多了元数据和小 segment 变多,开少了
一个大租户会把它那个桶撑得比别的大几个数量级,而查询是按桶裁剪的——大租户的邻居
跟着一起慢。默认 16 是给"几十到几百个体量相近的租户"的;有一个租户明显比别人大,
那不是加桶能解决的,是该给它单独一个 collection(见下面的升级信号)。

**partition key 不是权限系统。** 它保证的是数据怎么分布,不是谁能看谁——忘了在 filter 里
注入 `tenant_id`,A 租户就能搜到 B 租户的数据,而 Milvus 一声不吭地照常返回。所以所有读路径
统一走 `store.scope()` 出 filter,没有第二个地方允许自己拼 `tenant_id`。

`scope()` 里的白名单也不是洁癖:tenant 是外部字符串,直接进 filter 表达式就是注入面——
一个引号就能把 `tenant_id == "x"` 闭合掉,后面接 `or id != ""` 全库就出来了。

```bash
python test_tenant.py          # 14 条断言,全过才算隔离住了(集群上实测 14/14)
```

测的是"确实拿不到",不是"恰好没查到":先在 A 里写一条带哨兵词的数据,再用 B 的身份用
BM25、向量、filter、按 id 直读、无状态推荐、sample、count 七条路去要它,外加五个构造的
租户名(引号闭合、超长、空)必须被 400 挡掉。

**第 14 条断言是专门防空过的,它的来历值得说。** 原来只验证了 A 的 item 写进去了,没验证
A 的 **user** 建出来了——而 A 的 user 恰恰走"写完 item 立刻回查"那条路径。集群 Bounded 下
这条路径只有 2/10 读得到刚写的行,用户会被静默跳过;于是"B 读 A 的用户 → 404"变成
**因为这个用户根本不存在而 PASS**,一条安全断言测了个空气,而测试报告一片绿。
安全测试的每一条 PASS 都得先证明"要保护的东西真的在那儿"。

什么时候从 partition key 升级:

| 粒度 | 机制 | 适合 | 代价 |
| --- | --- | --- | --- |
| 行级 | `partition_key_field` | 租户多、每租户数据少 | 共享索引和资源,大租户会挤小租户 |
| 分区级 | 显式 partition + `partition_names` | 租户数中等,要按租户加载/卸载 | 分区数有上限 |
| 集合级 | 一租户一 collection | 租户少、数据量大、要各自的索引参数 | 每个 collection 有固定开销 |
| 库级 | database(`MILVUS_DB`) | 强隔离、独立权限 | 运维复杂度最高 |

信号很清楚:某个大租户需要跟别人不一样的索引参数、或者要能单独 load/release、
或者它的查询量已经在拖慢别人——这三条任意一条成立就该往上升一级。

### Step 5:一致性与可观测

**这一步直接对上 Day 0 的核心演示。** Day 0 卖点是"写进去下一秒就能被搜到",那是
Milvus Lite 的进程内行为,没有一致性这回事。集群默认 `Bounded`:查询用的是一个稍旧的
时间戳,刚 upsert 的行可能还没进可查视图,同一段 curl 在集群上会偶发失败。

服务默认 `Bounded`,只给"写完要回查"的那条路径开 Strong(`/search` 的 `strong: true`,
前端第 4 张卡片上是个勾选框)。不要整个服务开 Strong。

```bash
python bench.py consistency --queries 60
```

**集群实测(Zilliz Cloud,60 次「写入 + 立刻搜」):**

| 一致性 | 写完立刻查到 | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| Bounded(默认) | **20/60** | 79.22 | 99.24 | 163.00 |
| Strong | **60/60** | 84.39 | 249.71 | 270.54 |

对照:Milvus Lite 上两边都是 100%(进程内,写完就可见),Strong 的 p50 是 Bounded 的 2 倍多
(4.08 vs 1.84ms)。**同一段代码、同一句 curl,换个部署形态命中率从 100% 掉到 33%。**

两件事值得单独说:

- 大纲里写的是"Bounded 期望偶发查不到"。**实测不是偶发,是 2/3 查不到**——默认一致性下
  "写进去下一秒就能搜到"这个 Day 0 的核心演示基本是坏的,不是偶尔坏。措辞要改。
- Strong 的代价集中在尾部:p50 只 +6%(79→84ms),p99 +66%(163→271ms)。所以"别整个服务
  开 Strong"的理由不是平均慢,是**尾延迟被拉长**,而 SLO 是按 p99 定的。

**这条一致性不是只影响演示,它影响灌数据能不能成。** `store.add_users` 读的正是刚写进去的
items(先灌商品再灌用户,只隔几秒),Bounded 下读不到就把用户"跳过":接口返回 200、
`skipped` 里躺着 6038 个 id,看着像数据有问题,其实是一致性。所以这条写路径上的回查
硬编码 `strong=True`——单独实测这条路径:

```
Bounded(默认)  写完立刻读到 2/10
Strong        写完立刻读到 10/10
```

监控:

```bash
docker compose --profile obs up -d    # Prometheus:9090  Grafana:3000
curl localhost:8000/metrics           # 服务侧埋点
curl localhost:9091/metrics           # Milvus 自己的
```

服务侧的埋点用匹配到的路由模板打标签而不是 `request.url.path`——`/users/u1`、`/users/u2`
会把标签基数撑爆。分位数不在服务里算,Prometheus 侧 `histogram_quantile` 才能跨实例聚合。

**只看 Milvus 的指标不够**:用户感知到的慢可能全在 `embedding.encode` 那一步,跟 Milvus 无关。
两边都得抓。

上线前必须有告警的五个指标(表达式在 `monitoring/prometheus.yml` 底部):
查询 p99 延迟、查询错误率、collection 是否还 loaded、compaction 积压、节点内存水位。

---

## 大纲那六条"待核实"的答案

| # | 问题 | 答案 |
| --- | --- | --- |
| 1 | 集群版 collection properties / description 是否保留 | **保留**。自定义 key `app.embedding.signature` 写得进读得回,指纹就存这儿,本地和集群同一套代码 |
| 2 | dynamic field 里的 key 能否建索引 | **能**。`field_name="$meta"` + `params={"json_path": '$meta["year"]', "json_cast_type": "double"}`。原来"建不了索引"的说法是错的,Step 3 的理由要换成类型约束 |
| 3 | Milvus 是否暴露 BM25 的 `k1`/`b` | **暴露**,`bm25_k1`/`bm25_b`。但判据必须是分数变不变——集群会把 `bogus_param` 也存下来回显 |
| 4 | `client.compact()` 的调用方式和效果 | `client.compact(name)`。四种部署四种行为,Zilliz 专属集群返回 job id `-1`(收下了但不给你 job)。**对 BM25 idf 的实际影响没量到**——见下面已知限制 |
| 5 | partition key 的租户数上限 | **桶数不是租户数上限**。租户 hash 进桶,租户数无上限;桶数实测到 100000 仍能建 + 写 + load + 查。该关心的是倾斜不是数量 |
| 6 | `store.py` 在集群上的真实 diff | 纯代码 136 → 227 行,改动 195。**其中 Step 1「换连接串」只占 14 行** |

三条里有两条(#3、#5)是同一个教训:**"接受"和"回显"都不是证据,只有行为是。**
milvus-lite 和集群都会照单全收你传的任意参数;要知道它有没有生效,得看输出变不变。

## 已知限制

- **Milvus Lite 是单进程独占的**(一个 `LOCK` 文件)。服务在跑的时候 `bench.py` 连不上同一个库,
  要么先停服务,要么连集群。这也是 Day 1 存在的理由之一。
- **集群上量到的延迟里 95% 是网络往返**(地板 74-79ms,本机到 gcp-asia-southeast1)。
  所有绝对值只在这条链路上成立;换成同 VPC 内网,地板会掉一个数量级,标量过滤和索引参数
  的差别就重新显形了。**能跨部署比的是曲线形状和命中率,不是毫秒数。**
- `MILVUS_REPLICAS > 1` 在 standalone 上没有意义(只有一个 query node),没实测过多副本行为。
- `MILVUS_INDEX_TYPE` 换成 HNSW/IVF_FLAT 之后**没在服务里跑过全量**——`bench.py` 里验证过
  这些参数在集群上生效,但服务侧默认还是 AUTOINDEX(见 Step 2 的判定:这个数据量下别动它)。
- **`get_collection_stats` 的 `row_count` 在 Zilliz Cloud 上两个方向都不可信。**
  同一份数据(真实 3883 条),实测它先报 **3907**(upsert 是"删 + 插",没合并之前两份都算,
  灌两遍 users 时它报到 12078),平台自己跑完 compaction 之后又报 **0**。
  全程 `count(*)` 报的都是对的。**告警和容量判断只能用 `count(*)`,不能用 row_count**——
  拿它当"数据没了"的信号会半夜把你叫起来,拿它当"数据涨了"的信号会漏掉真的暴涨。
- **`compact()` 对 BM25 idf 的影响没量到。** 逻辑是清楚的(idf 按 segment 统计),
  Day 0 也观察到了症状,但在这个集群上没法验:`compact()` 返回 `-1`,既不知道它跑没跑
  也没法等它跑完,前后对比量到的差别归因不了。要验这条得在自建 standalone 上做
  (那里返回真的 job id,能等)。
- **`--scale` 放大出来的语料不是真实分布。** 每个原始向量有 N-1 个扰动克隆,top-10 的
  ground truth 里全是近似平局,recall 绝对值被压低。配置之间的相对高低可信,绝对值不可引用。
- **Prometheus / Grafana / Attu 这一套没在集群模式下跑过。** `docker-compose.yml` 里的
  Prometheus 抓的是 `milvus:9091`,那是本地 standalone 的地址;连 Zilliz Cloud 时
  Milvus 侧指标要走它自己的控制台,只有服务侧的 `:8000/metrics` 是通的(这个验过了,
  label 用的是路由模板不是路径)。
- 五个告警指标的阈值是从别处抄的经验值,**没有在真实负载上校准过**。
