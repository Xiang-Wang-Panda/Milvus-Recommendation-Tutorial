"""LLM 精排:把召回给的 N 个候选交给 Claude 重排出 top-K。

它能做召回做不到的事:向量距离不理解否定和条件。"想找烧脑的太空片但不要恐怖的"
会把恐怖太空片一起召回,因为"太空"的语义太强,而模型能读懂"不要恐怖的"。

**但这一路在无查询的推荐上实测是空转**(见 rerank_eval.py 的结论)。
它的主场是查询意图复杂的搜索。开关默认关着。

四条写法上的纪律:

  1. 结构化输出(messages.parse + pydantic),别返回自由文本再正则解析。
  2. 评分标准放 system 打 cache_control,而且**要先预热**:缓存只有在第一条响应
     开始返回之后才可读,直接开并发的话首批会各自写一次缓存(按 1.25 倍单价计费)。
  3. 只让模型返回 top-K,不要重排全部 N 个——输出 token 是成本大头。
  4. max_tokens 给足:Claude Opus 5 默认开思考,而 max_tokens 是思考和输出的
     总额,配小了会被思考吃掉然后截断。

降级是硬要求:没凭据、超时、限流一律返回召回原序,不能让搜索接口 500。
模型漏掉的候选按召回原序接在后面——精排只该改顺序,不该改集合。
"""
import os

ENABLED = os.getenv("RERANK", "") not in ("", "0", "false")
MODEL = os.getenv("RERANK_MODEL", "claude-opus-5")
MAX_TOKENS = int(os.getenv("RERANK_MAX_TOKENS", "4096"))

# 放 system 里打缓存的那一段:每次请求只有候选列表在变,这一段一个字不动。
# 至少要几百 token 才值得缓存(Opus 5 的缓存最小单位是 512 token)。
RUBRIC = """你在给一个电影推荐系统做重排序。

输入是这个用户最近看过的片子，以及一批候选片子。你的任务是把候选按
"这个用户接下来最可能看哪一部"从高到低排序，返回最可能的前 10 个 id。

判断依据，按重要性排：

1. 题材和类型的延续性。看儿童动画的人接下来大概率还看儿童动画，
   看黑帮片的人接下来大概率还看犯罪剧情片。这是最强的信号。
2. 系列和导演的关联。看了《教父》大概率会看《教父 2》，
   看了一部昆汀大概率会看另一部昆汀。
3. 年代的接近程度。看九十年代片子的人和只看八十年代经典的人是两类人。
4. 名气。同样贴合的两部片子里，被更多人看过的那部更可能被选中——
   但别让名气盖过前三条，把一部谁都看过但类型完全不搭的片子排到前面是错的。

注意：
- 只从给定的候选里选，不要编造任何 id。
- 用户已经看过的片子不会出现在候选里，不用考虑去重。
- 候选的顺序是随机打乱的，顺序本身不含信息，别跟着它抄。
- 只返回 10 个 id，按可能性从高到低。"""


def _client():
    import anthropic
    return anthropic.Anthropic()


def rerank(history_texts, candidates, top_k=10):
    """candidates: [{"id":..., "text":...}]，返回重排后的同一批 candidates。

    返回值第二项是用量信息（token 数、缓存命中），给评测脚本算成本用；
    降级的时候是 None。
    """
    if not ENABLED or not candidates:
        return candidates, None
    try:
        return _rerank(history_texts, candidates, top_k)
    except Exception as e:
        # 没凭据、超时、限流、返回格式不对——全部降级成召回原序。
        # 精排是加分项，不该成为搜索接口的单点故障。
        print(f"精排降级（返回召回原序）：{type(e).__name__}: {str(e)[:120]}")
        return candidates, None


def _rerank(history_texts, candidates, top_k):
    from pydantic import BaseModel

    class Ranked(BaseModel):
        item_ids: list[str]      # 只要 top-K，按可能性降序

    hist = "\n".join(f"- {t}" for t in history_texts)
    cands = "\n".join(f"{c['id']}\t{c['text']}" for c in candidates)

    resp = _client().messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,       # 思考 + 输出的总额，配小了会被思考吃掉然后截断
        output_format=Ranked,
        system=[{                    # 评分标准缓存住，每次只变候选
            "type": "text", "text": RUBRIC,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user",
                   "content": f"用户最近看过：\n{hist}\n\n候选（id 和标题，顺序已打乱）：\n{cands}"}],
    )
    picked = resp.parsed_output.item_ids[:top_k]

    # 模型漏掉的候选按召回原序接在后面：保证集合不变，只比顺序。
    # 不这么做的话，"模型忘了返回某一项" 会被算成 "召回没召到"，两件事混在一起。
    by_id = {c["id"]: c for c in candidates}
    out = [by_id[i] for i in dict.fromkeys(picked) if i in by_id]
    out += [c for c in candidates if c["id"] not in {x["id"] for x in out}]

    u = resp.usage
    return out, {"in": u.input_tokens, "out": u.output_tokens,
                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}


def warm_up():
    """先单发一条把缓存写进去，再扇出。

    不预热的话首批并发请求会各自写一次缓存，按 1.25 倍单价计费，
    而且 cache_read_input_tokens 一直是 0，看着像"缓存没生效"。
    """
    if not ENABLED:
        return None
    try:
        _, usage = _rerank(["Toy Story (1995) Animation Children's Comedy"],
                           [{"id": "m1", "text": "Toy Story (1995) Animation"}], 10)
        return usage
    except Exception as e:
        print(f"预热失败（不影响后续，只是首批会各写一次缓存）：{type(e).__name__}")
        return None
