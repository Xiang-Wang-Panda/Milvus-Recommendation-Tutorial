# Day 1:如何把 Day 0 的搜索推荐系统迁上 Milvus 集群

Day 0 结束时得到的是一个 `milvus.db` 文件、3883 部电影、6038 个用户、八个接口。链路通了,但它跑在笔记本上,数据在一个文件里,一个进程独占。

Day 1 只解决一件事:迁到能承载真实流量的地方。业务逻辑一行不改,改的是怎么连、数据属于谁、读到的是哪个时刻的数据。精度是 Day 2 的事。

写法和 Day 0 一样:一步一段可以直接粘的提示词,后面跟可以直接粘的验证命令。

区别是每一步都要先说清什么算过。Day 0 的失败会报出来 —— 服务起不来、搜不出东西;上集群之后大部分失败是静默的:80% 用户被跳过而接口返回 200、改了配置不报错也不生效、越权测试因为要保护的数据不存在而全绿。

五步是一条链:

```
Step 1  连上集群          不过则全断
Step 2  schema 定型       不过则 Step 3 无效（partition key 加不了，只能全量重来）
Step 3  灌得进            不过则 Step 4 没有数据可量
Step 4  判定要不要调       不过则 Step 5 的告警阈值没有依据
Step 5  上线闸门
```

顺序是被"可逆 / 不可逆"分出来的:不可逆的决定必须在灌数据之前做完(Step 2),可逆的决定必须在灌完之后做(Step 4)。Step 1-3 半天能跑完,Step 4-5 可以过几天再补。

---

## Step 1:连上集群

Milvus Lite 是个进程内的库,一个 `LOCK` 文件独占,两个进程打不开同一个库。上生产第一步是换成独立服务。三条路只有连接串不同:

```bash
# A. 还是本地文件（Day 0 那样）
uvicorn main:app

# B. 自建 standalone
docker compose up -d                    # 等 90s，healthcheck 转绿
export MILVUS_URI=http://localhost:19530
open http://localhost:8001              # Attu：看 collection / 索引进度 / segment

# C. 托管服务（本文所有集群数字都在这上面测的）
export MILVUS_URI=https://<your-instance>.zillizcloud.com:443
export MILVUS_TOKEN=<your-token>
```

提示词:

```text
现在要能连真正的 Milvus 集群了，不能只支持本地文件那种。
两种情况尽量共用一套代码，别写成两条分支。
另外那个 embedding 指纹现在是存在库旁边的一个文件里，
我估计多开几个实例的话这样存会有问题，你看看要不要挪个地方。
```

验证。两件事:同一份代码连得上,以及 `store.py` 里"换连接串"的 diff 接近零。

```bash
export MILVUS_URI=https://<your-instance>.zillizcloud.com:443
export MILVUS_TOKEN=<your-token>
uvicorn main:app                       # 终端 A

curl -s localhost:8000/health | jq '{uri, cluster, items_loaded}'
# {"uri":"https://...","cluster":true,"items_loaded":"Loaded"}
```

两个字段最值得看。`cluster` 是 `true`,说明走的是集群路径而不是本地文件。`items_loaded` 是 `Loaded`,说明数据已经从对象存储拉进了 query node 的内存 —— 没 load 的集合查不了,而这件事在 Milvus Lite 上是隐式的。

如果换个地址要改一堆代码,Day 0 选 Milvus Lite 的理由就不成立。

实测 Day 1 全程改了 281 行,而其中"换连接串上集群"只占 14 行,剩下 267 行是上生产要补的能力:租户、一致性、批量、埋点。这两个数不要混着说 —— 迁到集群几乎免费,贵的是补能力。

指纹换个存法不是洁癖:Day 0 存在库旁边的文件里,多实例部署时各写一份,两个实例配了不同 `EMBED_MODEL` 就会互相触发重建。

---

## Step 2:schema 定型

这一步只定"灌完就锁死"的那几项,其余留到 Step 4。差别很大:改可逆的是 `drop_index` + `create_index`,几秒钟,数据一行不动;改不可逆的是 drop collection + 全量重灌。

必须现在定的只有四件:

| 决定 | 不做的后果(实测) |
| --- | --- |
| `tenant_id` 当 partition key | 报错 `not support to add partition key field`,只能全量重来 |
| 桶数 `num_partitions` | `alter_collection_properties` 不报错也不生效,`describe` 回显还是旧值 |
| 文本的 `analyzer_params` | 报错 `analyzer_params does not allow update in collection field param` |
| `dense` 维度 | 向量按维度存储,无法修改 |

提示词:

```text
接下来要做多租户了，items 和 users 都得加个 tenant_id。
我看 Milvus 有个 partition key 的机制好像是为这种场景设计的，你研究一下要不要用。
上集群之后好像不少字段改不动了，帮我理一份清单：这个 schema 里
哪些改动事后还能做、哪些必须现在就定死，每条都拿实测证据说话——
报错信息，或者"改了但不生效"这种坑，别凭印象写。
```

第二行那个"不报错也不生效"最值得警惕:**不能用 `describe` 的回显验证配置生效了。**

验证。改一个不可逆的字段,看它报不报错:

```bash
J='Content-Type: application/json'
TEXT_ANALYZER=chinese uvicorn main:app         # 终端 A，重建集合

curl -s localhost:8000/items -H "$J" -d '[{"id":"zh1","text":"深圳的天气很好"}]'
curl -s localhost:8000/search -H "$J" -d '{"q":"天气","top_k":3}'
# 命中 —— analyzer 配对了
```

光改 `TEXT_ANALYZER` 环境变量重启没用——analyzer 是烘进这个字段的 schema 里的,不是运行时读的,重启并不会碰它,查询行为原封不动。真要看"改不了"得直接冲着已建好的 collection 去改:

```python
from pymilvus import MilvusClient
c = MilvusClient(uri="http://<host>:19530", token="<token>")
c.alter_collection_field("items", field_name="text",
                         field_params={"analyzer_params": {"type": "standard"}})
# pymilvus.exceptions.MilvusException: (code=1100, message=analyzer_params does not
# allow update in collection field param: invalid parameter)
```

这条在 Milvus Lite 上量不出来——`alter_collection_field` 那个 RPC 本身就没实现(`UNIMPLEMENTED`),报错性质完全不同。真正这条 `analyzer_params does not allow update` 只在连真实集群(自建或 Zilliz Cloud)时才拿得到,本文这条是在 Zilliz Cloud 上实测的。

---

## Step 3:灌得进

Day 0 有两个隐含假设,上集群之后都不成立:

| | Day 0 默认成立 | 集群上的实际情况 | 实测 |
| --- | --- | --- | --- |
| 往返 | 调用免费,循环里查库没问题 | 每次调用约 70ms 下限 | items 2.5s vs users 707.4s |
| 一致性 | 写完立刻读得到 | 默认读的是稍旧的快照 | 写完立刻读 2/10 |

提示词:

```text
seed.py 灌数据在集群上巨慢，帮我看看是不是循环里一条条查库了，
能合并就合并、能去重就去重，改完告诉我网络往返次数改了多少。
另外我记得 Milvus 有个一致性级别的配置，读接口是不是该配一下？
默认应该用弱一点的那档，但是有的地方写完得立刻读到，
你先找出这套代码里哪些路径是这种情况，再说说几档级别各自的代价。
```

验证:

```bash
time python seed.py                            # 全程计时，改前 11 分 51 秒
python bench.py consistency                    # 写入 + 立刻搜，量命中率和延迟
```

**往返。**同一份数据,写 8 次批量调用 2.5s,读 6038 次单条查询 707.4s。而这些 id 大部分是重复的 —— 6038 个用户的历史合起来 239752 个 id,去重后只有 3386 个物品。去重 + 分批之后往返从 6038 次降到 7 次:

| | 改之前 | 改之后 |
| --- | --- | --- |
| users 6038 条 | 707.4s | 46.5s |
| `seed.py` 全程 | 11 分 51 秒 | 50 秒 |

**一致性。**集群默认 `Bounded`,查询用的是一个稍旧的时间戳。实测 60 次"写入 + 立刻搜":

| 一致性 | 写完立刻查到 | p50 | p99 |
| --- | --- | --- | --- |
| Bounded(默认) | 20/60 | 79.2ms | 163ms |
| Strong | 60/60 | 84.4ms | 271ms |

默认一致性下是 2/3 查不到,不是偶尔查不到。Strong 的代价不在平均值(p50 只 +6%)而全在尾部(p99 +66%)—— 这才是"别整个服务开 Strong"的理由,SLO 按 p99 定。

所以按路径区分而不是按服务区分:默认 `Bounded`,只给"写完要回查"的那几条路硬编码 `Strong`。本项目里就是 `add_users`,它读的正是刚写入的 items。不开 Strong 的话每个用户都被跳过,而接口返回 200、`skipped` 里躺着 6038 个 id —— 看着像数据有问题,而不像一致性有问题。

顺带一条:`get_collection_stats` 的 `row_count` 先报多、再报 0，而数据一直完好。要数行数用 `count(*)`。

---

## Step 4:判定要不要调

这一步的产出不是"最优参数",而是"这张表能不能用来选型"。

提示词:

```text
写个 bench.py 帮我看看索引参数该怎么选，顺便把标量过滤的两种存法也比一下，
再看看写完立刻搜的命中率怎么样。用 FLAT 当基准答案，
查询集拿真实用户的兴趣向量，别用随机向量凑数。
每次跑完先自己判断一下这个数能不能信——差别有没有明显大过误差范围，
分不开的话就别硬下结论。最好再量一下最基础的网络延迟垫底是多少。
```

验证:

```bash
python bench.py index         # 扫 HNSW / IVF_FLAT，FLAT 当 ground truth
python bench.py filter        # dynamic field vs 显式字段 + INVERTED
python bench.py index --scale 200000    # 把数据扰动复制到 20 万条再扫一遍
```

两个约束不能省,否则量出来的是别的东西。ground truth 用 FLAT 暴力搜;查询集用真实用户兴趣向量,不要用随机向量 —— 随机向量在高维空间里彼此近似等距,近似索引在上面的表现和真实分布无关。

结果是这一档两根轴都分不开:

```
查询链路地板 74.1ms（1 行集合上的 search，几乎全是网络）
recall 极差 0.007  —— 数据太少，任何索引都能找到真正的最近邻
p50   极差 5.5ms   —— 全被 74.1ms 的网络吃掉，剩下的是抖动
```

所以这一档不该在索引参数上下任何结论,保持 `AUTOINDEX`。数据扰动复制到 20 万条之后召回轴才分得开(极差 0.180),延迟轴仍然分不开。

还有一件必须说清:这一步在 Milvus Lite 上做不了。milvus-lite 3.2.0 会忽略索引参数。而远程集群上标量过滤的 2.6ms 差别,又被 75ms 的往返完全掩盖:

| | 向量索引参数 | 标量过滤代价 |
| --- | --- | --- |
| Milvus Lite | 测不了(参数被忽略) | 测得准(没有网络,3 倍差别可见) |
| 远程集群 | 测得准 | 测不了(2.6ms 被 75ms 往返掩盖) |

两个环境各能测一半,这是必须接受的事实,不是能绕开的配置问题。

另一个反直觉的:别用 `describe_index` 验证参数生效了 —— 集群会把你传的任何键原样回显,包括 `bogus_param`。唯一的判据是分数变不变,三个文档查 `star`:

| 索引参数 | doc1(`star`×8,短) | doc2(长) | doc3(短) |
| --- | --- | --- | --- |
| `{}` 默认 | 0.2771 | 0.0778 | 0.2191 |
| `{"bogus_param": 1}` | 0.2771 | 0.0778 | 0.2191 |
| `{"bm25_k1":3.0,"bm25_b":1.0}` | 0.4963 | 0.0577 | 0.4705 |

第二行和第一行完全一致,假参数被忽略了;第三行变了,真参数生效了。索引选型是可逆的,量不出来就先不动。

---

## Step 5:上线闸门

两件事没做完就不该上线:证明租户之间拿不到对方的数据,以及有告警。

提示词:

```text
上线前得证明一下租户之间是不是真的互相看不到数据。
所有读写接口都得带 tenant_id，而且不能只是"传了就行"，得强制生效。
写个测试，别测出来"恰好没查到"这种假阳性——
你自己想办法证明"要保护的东西确实在那儿、而且确实拿不到"。
顺带把 tenant_id 传进奇怪字符串的情况也测一下。
```

验证:

```bash
python test_tenant.py                          # 七条越权路径 + 三种构造租户名
curl -s localhost:8000/metrics | grep app_request_seconds_bucket | head -3
```

最后一行是关键。**越权测试最常见的失败模式是要保护的数据根本没写进去,于是七条路都"拿不到",报告全绿。**所以每条断言前先用 A 的身份把那条数据查出来,证明它在。

隔离粒度有四档,本项目选行级:

| 粒度 | 机制 | 适合 | 代价 |
| --- | --- | --- | --- |
| 行级 | `partition_key_field` | 租户多、每租户数据少 | 共享索引,大租户影响小租户 |
| 分区级 | 显式 partition | 要按租户 load / release | 分区数有上限 |
| 集合级 | 一租户一 collection | 租户少、数据量大 | 每个 collection 各占资源 |
| 库级 | database | 强隔离、独立权限 | 运维最复杂 |

要注意 partition key 保证的是**数据怎么分布,不是谁能看谁**。忘了拼那句 filter,A 租户就能搜到 B 的数据,Milvus 一声不吭照常返回。所以所有读路径统一从一个函数出 filter。tenant 名还要走白名单 —— 它是外部传进来的字符串,一个引号就能把 `tenant_id == "x"` 闭合掉。

提示词:

```text
上线前还得把监控补上，每个接口至少要能看到延迟和错误率。
打标签的时候如果直接用 URL 路径会不会有问题？用户一多路径就一堆。
自建集群顺便配一下 Prometheus + Grafana，再列一下上线前必须要有告警的指标，
以及哪些东西在托管服务上根本拿不到、得想别的办法。
```

标签用路由模板不是风格问题:用 `request.url.path` 的话 `/users/u1`、`/users/u2` 各占一个标签值,几千个用户就把基数撑爆。

托管服务上有两个指标拿不到:

| 指标 | 自建 | 托管 |
| --- | --- | --- |
| 查询 p99 / 错误率 | 有 | 有(服务侧埋点) |
| collection 是否还 loaded | 有 | 有(`get_load_state`) |
| compaction 积压 | 有 | 没有 `:9091`,而且 compaction 不由你控制 |
| 节点内存水位 | 有 | 拿不到 |

`compact()` 顺带说一下,Day 0 留了个坑。BM25 的 idf 按 segment 统计,新写入的行独占一个新 segment,类型词在那儿 idf≈0,会被老数据压到后面 —— 所以灌完要合并一次。

但这个动作在四种部署下有四种行为:

| 部署 | `client.compact()` 返回 |
| --- | --- |
| milvus-lite | 同步完成,job id `0` |
| 自建 standalone | 真实 job id,后台异步 |
| 托管专属集群 | job id `-1` —— 请求收下了,但不给你能查的 job |
| 托管 serverless | `PERMISSION_DENIED` |

托管服务上这条优化不由你控制:既不能确认它跑了,也不能等它跑完。所以别把它放在关键路径上,降级成一条说明。

---

## 跑起来之后

四条规矩,都是这一篇撞出来的:

1. **上集群之后大部分失败是静默的。**接口返回 200 不等于做成了,所以每一步先定"什么算过"。
2. **不可逆的决定必须在灌数据之前做完。**partition key、analyzer、向量维度、桶数,错了要全量重灌。
3. **量不出来就不许下结论。**先看极差有没有高过噪声地板;这一档两根轴都分不开,那就保持 `AUTOINDEX`。
4. **别用回显验证配置生效。**`describe_index` 会把 `bogus_param` 原样返回给你,唯一的判据是行为变了没有。

Day 2 在这套东西上给推荐加一路召回。

本文所有集群数字都在一个托管实例上跑过一遍:pymilvus 3.0.1、milvus-lite 3.2.0(本地那一档)。延迟数字带网络,所以只在同一张表内部比较,跨表比没有意义 —— 查询链路地板 70~75ms 那一条就是为此量的。
