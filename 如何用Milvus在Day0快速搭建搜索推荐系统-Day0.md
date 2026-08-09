# 用 Milvus 在 Day 0 搭一套搜索 + 推荐

很多产品都要"能搜、能推",但不是每个团队都有搜索推荐的人。这篇给这类工程师:一台笔记本,零部署、零 GPU、零训练、零 API Key,把搜索、推荐、相似用户三件事跑起来。

代码不用自己写。你把要什么说清楚,让 Claude Code 写、跑、修,你验收。所以下面每一步都是一段可以直接粘的提示词,后面跟几行可以直接粘的验证命令 —— 每一步跑完,你能在终端里看见它确实通了,再往下走。数据用公开的 MovieLens-1M,换成自己的业务数据是一样的。

---

## Day 0 / Day 1 / Day 2

- **Day 0(本篇)** —— 本机跑通。Milvus Lite 就是一个本地文件,零部署;嵌入先用哈希,零下载;加一个单文件前端能当面演示。目标是把链路和接口定下来,不追指标。
- **Day 1 —— 上生产、扛住量。** `MILVUS_URI` 从本地文件换成 Milvus 集群或 Zilliz Cloud 的地址,建 HNSW/IVF 索引调延迟和召回,补上标量过滤、分区、多租户隔离和监控。存储层代码不动,换连接串就行。
- **Day 2 —— 搜得更准、推得更丰富。** 升级 embedding;推荐从纯内容改成多路召回(内容 + 协同过滤,用 RRF 融合);分组去重保多样性;top-K 交给 LLM 精排。这些都加在召回层上,业务代码基本不重写。

再往上追精度,是训练式和生成式推荐的范围(SASRec、LightGCN、TIGER、HSTU、OneRec),不在这个系列里。

## 架构

搜索、推荐、相似用户看着是三个功能,底下是同一件事:**把查询、用户、物品都变成向量,在向量空间里找最近邻。**区别只是"拿谁的向量,去查哪张表"。

所以整套系统只有三个组件:

- **嵌入模型** —— 文本变向量。起步用哈希凑数,后面换成真模型。
- **Milvus** —— 存向量,找最近邻。顺便存标量字段用来过滤,顺便算 BM25。
- **FastAPI** —— 八个读写接口,加一个单文件前端。

两张表,一边写一边读:

```
写
  物品 text ──► POST /items ──► 商品塔 items    每条物品:稠密向量 + BM25 稀疏向量 + 标量字段
  用户历史 ──► POST /users ──► 用户塔 users    每个用户:一条兴趣向量(历史物品向量求均值)

读
  /search         查询向量 ──► 商品塔      关键词走 BM25,语义走稠密向量
  /recommend      用户塔取向量 ──► 商品塔   排除已经看过的
  /similar-users  用户塔 ──► 用户塔        user→user 最近邻
```

用户塔的向量是历史物品向量的均值,所以它和商品塔天然在同一个空间里,不需要训练。也因此,商品要先灌完,用户塔才能建。

## Day 0 的五步

每步只加一件事:

1. BM25 搜索 —— 不需要任何模型
2. 加向量和用户塔 —— 推荐、相似用户
3. 灌真实数据
4. 换成真 embedding 模型 —— 搜索从关键词变语义
5. 单文件前端

全部跑完是 705 行代码、一个本地库文件。

准备:Python 3.10+(`milvus-lite` 要求),一个空目录,装好 [Claude Code](https://claude.com/claude-code) 后在目录里敲 `claude`。提示词分步给,跑通一步再给下一步,好排错。

---

## Step 1:BM25 搜索

先只做关键词搜索。这一步不涉及向量,也不下载任何模型。

选型只有两个:**Milvus Lite**,`pip` 装完就是一个本地文件,零部署,API 和 Milvus 集群一模一样,以后上量只换连接串;**FastAPI**,自带 `/docs`,几十行出一套带校验的接口。

提示词:

```text
用 FastAPI + Milvus Lite（pymilvus 的 MilvusClient）做一个关键词搜索服务。

store.py：集合 items，字段 id(VARCHAR 主键)、text(VARCHAR，开 analyzer)、
sparse(SPARSE_FLOAT_VECTOR)；开 dynamic field，让 category、year 这类字段原样存、能过滤。
sparse 不要自己算，用 Milvus 原生 BM25 Function 从 text 生成，索引用 SPARSE_INVERTED_INDEX。
analyzer 用环境变量 TEXT_ANALYZER 选（chinese=jieba / english / standard），
连接串用 MILVUS_URI，默认本地文件 ./milvus.db。

main.py：POST /items 批量 upsert、POST /search {q, top_k, filter}、GET /health。
顺带写 requirements.txt。
```

验证。三件事:写得进、搜得出、过滤生效。

```bash
pip install -r requirements.txt
TEXT_ANALYZER=english uvicorn main:app     # 终端 A，服务在 127.0.0.1:8000

# 终端 B
J='Content-Type: application/json'     # 后面几步的验证命令都会用到它

curl -s localhost:8000/items -H "$J" -d '[
  {"id":"1","text":"Interstellar space travel through a wormhole","category":"Sci-Fi","year":2014},
  {"id":"2","text":"Toy Story toys come alive","category":"Animation","year":1995}]'
# {"upserted":2}

curl -s localhost:8000/search -H "$J" -d '{"q":"space","top_k":2}'
# 只回 id=1，score 0.66 —— 这是 BM25 分,不是相似度

curl -s localhost:8000/search -H "$J" -d '{"q":"toys","top_k":2,"filter":"year > 2000"}'
# []  —— Toy Story 是 1995,被 year 过滤掉了
```

第三条最值得看:`year` 从没在 schema 里声明过,靠 dynamic field 原样存下来,直接就能当过滤条件用。

BM25 的稀疏向量由 Milvus 的 `Function` 从 `text` 自动生成,写入侧只管给文本。要连集群时把 `MILVUS_URI` 改成 `http://<host>:19530`,这层代码一行不动——这是选 Milvus Lite 的唯一理由。

---

## Step 2:加向量和用户塔

按上面的架构补齐:商品塔加一个稠密向量字段,再建一个用户塔,推荐和相似用户就都有了。

嵌入这步先不下模型,用哈希凑数:把词散列到 512 维再归一化,装完即跑,先把链路打通。真模型放到 Step 4。

维度这里故意挑了 512:常见的小模型(`bge-small`、`MiniLM`)都是 384 维,如果哈希也用 384,Step 4 换模型时"维度变了就重建"这个兜底就不会触发,两种向量会混在一个集合里,检索结果乱掉还不报错。

提示词:

```text
加向量能力。

embedding.py：encode(texts) 和 dim()。先只做 hash 后端——把词散列到 512 维再归一化，
不下任何模型。留好换模型的位置。

items 加一个 dense 字段（AUTOINDEX + COSINE），/search 加 mode 参数：bm25 走 sparse，
vector 走 dense。

新集合 users：user_id 主键、dense、history、prefer。
POST /users：兴趣向量 = 历史物品 dense 向量求均值，预算好存进去。
POST /recommend：给 user_id 就用存好的向量搜商品塔并排除历史，给 item_ids 就现算均值。
POST /similar-users：在 users 里做 user→user 最近邻。
GET /users/{id} 返回画像，GET /users/sample 给前端用。

用户塔依赖商品向量，要先灌完商品再建。
```

验证。Step 1 建的表没有 `dense` 字段,先删库再起:

```bash
rm -rf milvus.db
TEXT_ANALYZER=english uvicorn main:app     # 终端 A

# 终端 B
J='Content-Type: application/json'

curl -s localhost:8000/items -H "$J" -d '[
  {"id":"1","text":"Interstellar space travel through a wormhole"},
  {"id":"2","text":"Toy Story toys come alive"},
  {"id":"3","text":"Star Wars rebels fight the empire in space"},
  {"id":"4","text":"Finding Nemo animated fish adventure"}]'

curl -s localhost:8000/users -H "$J" -d '[
  {"user_id":"u1","history":["1","3"],"prefer":"Sci-Fi"},
  {"user_id":"u2","history":["2","4"],"prefer":"Animation"}]'
# {"upserted":2,"skipped":[]}   —— 用户塔建起来了；skipped 是 history 里一个物品都不认识的用户

curl -s localhost:8000/search -H "$J" -d '{"q":"space","mode":"vector","top_k":2}'
# 走 dense 了：id=1、id=3

curl -s localhost:8000/recommend -H "$J" -d '{"user_id":"u1","top_k":2}'
# 剩下 id=2 和 id=4 —— u1 看过的 1 和 3 被排掉了，这是这一步唯一要确认的事
#（两条分数都是 0.0，谁在前面不固定，别对着顺序较真）

curl -s localhost:8000/recommend -H "$J" -d '{"item_ids":["3"],"top_k":2}'
# 不用建用户也能推：现算这批物品的均值

curl -s -X POST "localhost:8000/similar-users?user_id=u1&top_k=1"
# [{"user_id":"u2",...}]
```

这一步别看分数。哈希向量之间基本正交,`recommend` 的 score 会是 0.0,`similar-users` 也是 —— 现在只验链路通不通、历史有没有被排掉。分数要到 Step 4 换了真模型才有意义。

(如果生成的 `/similar-users` 收的是 JSON body 而不是 query 参数,把最后一条换成 `-H "$J" -d '{"user_id":"u1","top_k":1}'`。)

---

## Step 3:灌真实数据

自造的随机数据看不出推荐好坏,换公开数据集。

提示词:

```text
data/load_movielens.py：下载 MovieLens-1M，转成两个 csv。
products.csv：一部电影一行，id=m<MovieID>，text=标题+类型词，category=主类型，year=年份。
users.csv：user_id=u<UserID>，history=评分>=4 的电影按时间取最近 50 部用 | 连接，
prefer=历史里最多的类型。

seed.py：读 csv，分批 POST /items，再 POST /users，
最后跑一次搜索、推荐、相似用户，把结果打出来当冒烟测试。
```

验证:

```bash
rm -rf milvus.db     # milvus-lite 3.x 下它是个目录，-r 不能省
python data/load_movielens.py                  # 3883 部电影 / 6038 个用户
TEXT_ANALYZER=english uvicorn main:app     # 终端 A
python seed.py                                 # 终端 B：灌数据 + 冒烟测试

curl -s localhost:8000/health
# {"analyzer":"english","embed":"hash:512","items":3883,"users":6038}   —— 两张表都灌满了

curl -s localhost:8000/search -H "$J" -d '{"q":"star wars","top_k":2}' \
  | jq -r '.[]|"\(.score|.*100|round/100)  \(.text)"'

# 7.83  Star Wars: Episode IV - A New Hope (1977) ...
# 7.83  Star Wars: Episode I - The Phantom Menace (1999) ...
```

seed.py 最后那次冒烟测试的输出大致是这样(hash 后端):

```text
[搜索] BM25 'Star Wars'
  7.83  Star Wars: Episode IV - A New Hope (1977) Action Adventure Fantasy Sci-Fi
  7.83  Star Wars: Episode I - The Phantom Menace (1999) Action Adventure Fantasy Sci-Fi

[推荐] u1 —— 排除历史
  0.657  Babe (1995) Children's Comedy Drama
  0.629  Wide Awake (1998) Children's Comedy Drama

[相似用户] u1
  u4767 (Drama, 0.908) · u413 (Drama, 0.897) · u5887 (Drama, 0.896)
```

BM25 那两行在同一份数据上可以稳定复现。推荐和相似用户是哈希向量算出来的,只是词面相似,别当质量看——这正是下一步要换掉的东西。

---

## Step 4:换成真模型

哈希向量只会字面匹配。搜"想找烧脑的太空片"这种不含片名的说法,它什么也搜不出来。换成真 embedding 模型,搜索才从关键词变成语义。

模型跑在本地,不联网、不花钱,代价是要下一次模型。

提示词:

```text
embedding.py 加一个 st 后端：sentence-transformers 本地模型，
默认 BAAI/bge-small-en-v1.5，用 EMBED_BACKEND=st 切换，EMBED_MODEL 指定模型。
维度跟 hash 不一样时自动重建集合。
```

验证:

```bash
pip install sentence-transformers
EMBED_BACKEND=st TEXT_ANALYZER=english uvicorn main:app     # 终端 A
# 启动时会打印：dense 维度 512 -> 384，重建集合   —— 不用手动删库
python seed.py                                                  # 终端 B：用模型重新编码一遍

# 用一句不含任何片名的话搜
curl -s localhost:8000/search -H "$J" \
  -d '{"q":"mind-bending movie about space and time","mode":"vector","top_k":5}' \
  | jq -r '.[]|"\(.score|.*1000|round/1000)  \(.text)"'
```

五条结果,没有一条包含查询里的词:

```text
0.727  Destination Moon (1950) Sci-Fi
0.723  Timecop (1994) Action Sci-Fi
0.713  Black Hole, The (1979) Sci-Fi
0.709  Time Masters (1982) Animation Sci-Fi
0.707  2001: A Space Odyssey (1968) Drama Mystery Sci-Fi Thriller
```

推荐也跟着变准。同一个 u1,哈希向量推的是词面像的,模型向量推的是真的同类:

```bash
curl -s localhost:8000/recommend -H "$J" -d '{"user_id":"u1","top_k":2}' \
  | jq -r '.[]|"\(.score|.*1000|round/1000)  \(.text)"'
# 0.87   Parent Trap, The (1998) Children's Drama
# 0.868  Jungle Book, The (1967) Animation Children's Comedy Musical
```

重建是**删数据**:切后端、换模型、改维度,都要重新跑一遍 `seed.py`。

还有一个只靠维度兜不住的情况:两个模型维度一样(比如 `bge-small-en-v1.5` 和 `MiniLM-L6-v2` 都是 384 维),换 `EMBED_MODEL` 时维度没变,检查就不会触发,你以为换了模型,实际搜的还是旧模型的向量。想接住这种,得另外记一个"这批向量是谁算的"的指纹——比如往库边上写一个小文件存 `st:<模型名>:<维度>`,启动时比一下。

---

## Step 5:一个单文件前端

Day 0 最好的演示是当着人写一条,立刻搜出来。

提示词:

```text
写 static/index.html，单文件，vanilla JS，无框架无构建，后端 GET / 托管。四个面板：
搜索（选 bm25/vector）、推荐（选用户 → 先显示画像再显示推荐）、相似用户、
加电影（写入后立刻搜到它）。样式简单能截图就行。
```

验证:打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/),一个写面板 + 三个读面板跑一遍。同样的事用命令行也能验,写完立刻读得到:

```bash
curl -s -o /dev/null -w '%{http_code} %{content_type}\n' localhost:8000/
# 200 text/html; charset=utf-8   —— 前端被后端托管着

# 写一部数据集里没有的电影，立刻搜得到
curl -s localhost:8000/items -H "$J" \
  -d '[{"id":"m9999","text":"Interstellar (2014) Sci-Fi Drama"}]'
curl -s localhost:8000/search -H "$J" -d '{"q":"Interstellar","top_k":1}' \
  | jq -r '.[]|"\(.score|.*100|round/100)  \(.text)"'
# 8.44  Interstellar (2014) Sci-Fi Drama   —— 第一

# 刚写的这部电影也立刻进了推荐池：拿它自己当种子，现算向量去召回同类
curl -s localhost:8000/recommend -H "$J" -d '{"item_ids":["m9999"],"top_k":3}' \
  | jq -r '.[]|"\(.score|.*1000|round/1000)  \(.text)"'
# 0.814  Close Encounters of the Third Kind (1977) Drama Sci-Fi
# 0.805  Delta of Venus (1994) Drama
# 0.793  Conceiving Ada (1997) Drama Sci-Fi
```

没有重建索引,没有重新训练,写进去下一秒就能被搜到、被推出来。

---

## 跑起来之后

到这里,MovieLens-1M 全量在一台笔记本上跑着:3883 部电影、6038 个用户、八个接口、一个演示前端,存储是目录下一个 `milvus.db` 文件。Day 0 要的就是这个——链路通了,接口定了,能当面演示。

再往下是开头说的 Day 1 和 Day 2:`MILVUS_URI` 换成集群地址上量,以及在召回层上加多路召回和精排。

---

## 附:接口一览

服务起来后 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 是 FastAPI 自动生成的接口文档,能直接点着试。下面是全部接口:

| 接口 | 作用 |
| --- | --- |
| `GET  /health` | analyzer、embedding 指纹(`hash:512` / `st:<模型>:<维度>`)、两张表各有多少条 |
| `POST /items` | 批量 upsert 物品。`id` `text` 必填,其他字段(`category`、`year`…)原样存、能过滤 |
| `POST /search` | `{q, top_k, mode, filter}`。`mode=bm25` 走关键词,`mode=vector` 走语义 |
| `POST /users` | 批量 upsert 用户。兴趣向量 = `history` 里各物品向量求均值,写入时算好。`history` 里一个物品都不认识的用户会被跳过,返回 `skipped` |
| `POST /recommend` | 给 `user_id` 用存好的向量,给 `item_ids` 现算均值;两种都排除历史 |
| `POST /similar-users` | `?user_id=&top_k=`,在用户塔里做 user→user 最近邻 |
| `GET  /users/{id}` | 用户画像:`prefer` + 历史物品明细 |
| `GET  /users/sample` | `?n=`,随便取 n 个用户,前端下拉框用 |
| `GET  /` | 单文件前端 |

每个接口一条命令,贴完就能对着看返回:

```bash
J='Content-Type: application/json'                                          # JSON 头
curl -s localhost:8000/health
curl -s localhost:8000/items  -H "$J" -d '[{"id":"x1","text":"a movie about space"}]'
curl -s localhost:8000/search -H "$J" -d '{"q":"space","mode":"vector","top_k":3}' | jq -r '.[].text'
curl -s localhost:8000/users  -H "$J" -d '[{"user_id":"ux","history":["m260","m1196"],"prefer":"Sci-Fi"}]'
curl -s localhost:8000/recommend -H "$J" -d '{"user_id":"ux","top_k":3}'    | jq -r '.[].text'
curl -s -X POST "localhost:8000/similar-users?user_id=ux&top_k=3"           | jq -c '.[]'
curl -s localhost:8000/users/ux                                             | jq -c '.'
curl -s "localhost:8000/users/sample?n=3"                                   | jq -c '.'
```

四个环境变量决定全部形态,代码不改:`MILVUS_URI`(本地文件 / 集群地址)、`TEXT_ANALYZER`(chinese / english / standard)、`EMBED_BACKEND`(hash / st)、`EMBED_MODEL`。

本文所有命令和输出都在一个全新目录里从零跑过一遍:macOS 15.7.4、Python 3.13.7、pymilvus 3.0.1、milvus-lite 3.2.0、sentence-transformers 5.7.0。`pymilvus>=2.4` 现在会装到 3.0.x,上面的写法在这个版本下验证通过。验证命令只用到 `curl`(系统自带)和 `jq`(只负责把 JSON 排版好看,不装也能跑,输出是一行原始 JSON)。
