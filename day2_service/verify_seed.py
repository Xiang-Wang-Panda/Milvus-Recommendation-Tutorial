"""灌完数据之后,证明库里躺的确实是你以为的那批向量。

这类失败是**静默的**:集合没重建、向量还是上一次的、服务没起来但脚本没中止、
查询侧和文档侧不在一个空间里——全都不报错,只是分数变差,而变差多少你看不出来。

指纹机制能挡住"忘了重灌",但指纹只是"我声明我是谁",不是"我确实是谁"。
所以这里做最后一道验证:随机抽几条物品,把它**应该被编码的那段文本**重新编码一遍,
和库里存的向量算余弦。≈1.000 才算对上。

注意 α>0 时库里的向量模长不是 1(先验乘进去了),所以要先归一化再算余弦——
比的是方向,模长由 popularity.weight() 单独核对。

用法(和服务用同一套环境变量):
    EMBED_BACKEND=st EMBED_MODEL=... POP_ALPHA=0.25 python verify_seed.py
"""
import csv
import logging
import math
import os
import random
import sys
from pathlib import Path

logging.disable(logging.ERROR)

import cooccur  # noqa: E402
import embedding  # noqa: E402
import popularity  # noqa: E402
import store  # noqa: E402

DATA = Path(__file__).parent / "data"
PRODUCTS = os.getenv("PRODUCTS_CSV", "products_loo.csv")
TENANT = os.getenv("TENANT", "default")
N = int(os.getenv("VERIFY_N", "8"))


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def main():
    want_sig, got_sig = store._signature(), store._stored_signature()
    dim_want, dim_got = embedding.dim(), store._dim("items")

    print(f"指纹  应该是 {want_sig!r}")
    print(f"      库里是 {got_sig!r}")
    print(f"维度  应该是 {dim_want}   库里是 {dim_got}")
    print(f"先验  POP_ALPHA={popularity.ALPHA:g}  metric={popularity.METRIC}"
          f"  热度榜 {len(popularity.table())} 条")
    print(f"共现  {cooccur.signature()}")

    bad = []
    if got_sig != want_sig:
        bad.append(f"指纹不一致:{got_sig} != {want_sig}")
    if dim_got != dim_want:
        bad.append(f"维度不一致:{dim_got} != {dim_want}")

    store._load("items")
    n_items = store.count("items", TENANT)
    rows = list(csv.DictReader(open(DATA / PRODUCTS, encoding="utf-8")))
    print(f"条数  库里 {n_items}，csv 里 {len(rows)}")
    if n_items != len(rows):
        bad.append(f"条数不一致:{n_items} != {len(rows)}")

    # ---- 核心验证:抽样重编码,和库里的向量比余弦 ----
    sample = random.Random(0).sample(rows, min(N, len(rows)))
    ids = [r["id"] for r in sample]
    stored = {r["id"]: r for r in store.client.query(
        "items", filter=store.scope(TENANT, store._in(ids)),
        output_fields=["id", "dense", "cf"], limit=len(ids))}

    fresh = embedding.encode([r["text"] for r in sample])

    print(f"\n抽样重编码 {len(sample)} 条，和库里向量算余弦(应该 ≈1.000):")
    worst_cos, worst_w = 1.0, 0.0
    for r, v in zip(sample, fresh):
        row = stored.get(r["id"])
        if row is None:
            bad.append(f"{r['id']} 在库里查不到")
            continue
        sv = row["dense"]
        cos = sum(a * b for a, b in zip(unit(sv), unit(v)))
        # 模长应该正好是 popularity.weight()
        norm = math.sqrt(sum(x * x for x in sv))
        w_want = popularity.weight(r["id"])
        dw = abs(norm - w_want)
        worst_cos, worst_w = min(worst_cos, cos), max(worst_w, dw)
        ncf = len(row["cf"]) if row.get("cf") is not None else 0
        flag = "✓" if cos > 0.999 and dw < 1e-3 else "✗"
        print(f"  {flag} cos {cos:.4f}  模长 {norm:.4f}(该是 {w_want:.4f})  "
              f"cf {ncf:>2} 维  {r['text'][:38]}")
    if worst_cos <= 0.999:
        bad.append(f"最差余弦只有 {worst_cos:.4f}：库里的向量不是这段文本编出来的")
    if worst_w >= 1e-3:
        bad.append(f"模长最大偏差 {worst_w:.4f}：流行度先验没正确乘进去")

    # ---- 打分函数自检:一个错误的 metric 能跑完全程不报错,还给出合理的数 ----
    # 这一项是补上的。第一版 cf 用 SPARSE_FLOAT_VECTOR + metric_type=IP，
    # 而 Milvus Lite 实际按 BM25 打分，整轮评测跑完都没报错，分数还只差 4%。
    probe = _metric_probe()
    print(f"\n打分函数自检(稠密 IP 必须等于手算内积):{probe or '通过'}")
    if probe:
        bad.append(probe)

    print()
    if bad:
        for b in bad:
            print(f"✗ {b}")
        sys.exit("验证失败：库里的向量和你以为的对不上，量出来的分数没有意义。")
    print(f"✓ 全部对上(最差余弦 {worst_cos:.4f}，模长最大偏差 {worst_w:.1e})。")


def _metric_probe():
    """用三个已知内积的合成向量验一次:Milvus 返回的必须就是内积。

    不验的话,换错字段类型或 metric 会静默地把打分函数换掉,而所有命令都正常退出。
    """
    from pymilvus import DataType, MilvusClient
    tmp = Path("/tmp/metric_probe.db")
    c = MilvusClient(uri=str(tmp))
    name = "probe"
    try:
        if c.has_collection(name):
            c.drop_collection(name)
        sch = c.create_schema()
        sch.add_field("id", DataType.VARCHAR, is_primary=True, max_length=8)
        sch.add_field("v", DataType.FLOAT_VECTOR, dim=4)
        ix = c.prepare_index_params()
        ix.add_index(field_name="v", index_type="FLAT", metric_type="IP")
        c.create_collection(name, schema=sch, index_params=ix)
        docs = {"a": [1.0, 2.0, 0.0, 0.0], "b": [3.0, 0.0, 0.0, 0.0],
                "c": [0.0, 1.0, 5.0, 0.0]}
        c.upsert(name, [{"id": k, "v": v} for k, v in docs.items()])
        c.flush(name)
        c.load_collection(name)
        q = [1.0, 1.0, 0.0, 0.0]
        got = {h["entity"]["id"]: h["distance"] for h in
               c.search(name, data=[q], anns_field="v", limit=3,
                        output_fields=["id"], search_params={"metric_type": "IP"})[0]}
        worst = 0.0
        for k, v in docs.items():
            want = sum(a * b for a, b in zip(q, v))
            worst = max(worst, abs(got.get(k, 1e9) - want))
        c.drop_collection(name)
        if worst > 1e-4:
            return (f"稠密 IP 返回的不是内积(最大偏差 {worst:.4f})"
                    f"——打分函数被静默换掉了,量出来的分数没有意义")
    except Exception as e:
        return f"自检本身跑失败:{type(e).__name__}: {str(e)[:80]}"
    return ""


if __name__ == "__main__":
    main()
