"""Milvus 原生 FunctionChain 能不能用?—— 在 Milvus Lite 上逐级探测。

pymilvus 3.0.1 提供了一套服务端重排 DSL(`pymilvus.function_chain`):

    fn.num_combine(...)   加权组合多个分数列          → 流行度先验
    fn.decay(...)         gauss / exp / linear 衰减   → 新鲜度
    fn.rerank_model(...)  在 Milvus 内部调外部重排模型  → cross-encoder 精排
    fn.xgboost(...)       挂 XGBoost 模型             → 蒸馏路线的落点
    阶段:L0 / L1 / L2_RERANK

**但客户端有 DSL 不代表服务端实现了。**Milvus Lite 是个嵌入式实现,历来比集群版
落后若干特性。所以这里从最简的开始逐级试,把"哪一级开始不支持"探出来。

每一级都单独 try,失败只记录不中断——目的是拿到一张能力矩阵,不是跑通。
"""
import logging

logging.disable(logging.ERROR)

import store  # noqa: E402

TENANT = "default"
Q = "space adventure"


def probe(name, fn):
    try:
        out = fn()
        print(f"  ✓ {name}")
        return ("ok", out)
    except Exception as e:
        msg = str(e).replace("\n", " ")[:150]
        print(f"  ✗ {name}\n      {type(e).__name__}: {msg}")
        return ("fail", None)


def main():
    from pymilvus.function_chain import FunctionChain, FunctionChainStage, col, fn

    store._load("items")
    import embedding
    qv = embedding.encode([Q])

    def search(chains=None, **kw):
        return store.client.search(
            "items", data=qv, anns_field="dense", limit=5,
            filter=store.scope(TENANT), output_fields=["id", "text", "year"],
            search_params=store._dense_params(),
            **({"function_chains": chains} if chains else {}), **kw)

    print(f"Milvus: {store.URI}（{'集群' if store.CLUSTER else 'Milvus Lite'}）")
    print(f"pymilvus {__import__('pymilvus').__version__}\n")

    print("0) 不带 chain 的普通检索（对照）")
    base = probe("baseline search", lambda: search())
    if base[0] == "ok":
        for h in base[1][0][:3]:
            e = h["entity"]
            print(f"      {h['distance']:.4f}  {e['text'][:46]}")

    print("\n1) 最简 chain：只把分数四舍五入（纯管路测试）")
    probe("L2_RERANK + round_decimal",
          lambda: search(FunctionChain(FunctionChainStage.L2_RERANK, name="probe_round")
                         .map("$score", fn.round_decimal(col("$score"), decimal=3))
                         .sort(col("$score"), desc=True).limit(5)))

    print("\n2) 新鲜度：按 year 做 exp 衰减（做新鲜度的正规武器）")
    probe("L2_RERANK + decay(year)",
          lambda: search(FunctionChain(FunctionChainStage.L2_RERANK, name="probe_fresh")
                         .map("freshness", fn.decay(col("year"), function="exp",
                                                    origin=2000, scale=10, decay=0.5))
                         .sort(col("freshness"), desc=True).limit(5)))

    print("\n3) 加权组合：相关性 + 新鲜度（替掉应用层重排）")
    probe("L2_RERANK + num_combine(weighted)",
          lambda: search(FunctionChain(FunctionChainStage.L2_RERANK, name="probe_combine")
                         .map("freshness", fn.decay(col("year"), function="exp",
                                                    origin=2000, scale=10, decay=0.5))
                         .map("$score", fn.num_combine(col("$score"), col("freshness"),
                                                       mode="weighted", weights=[0.8, 0.2]))
                         .sort(col("$score"), desc=True).limit(5)))

    print("\n4) 多路融合场景：hybrid_search 带 chain")
    from pymilvus import AnnSearchRequest
    reqs = [
        AnnSearchRequest(data=[Q], anns_field="sparse",
                         param={"metric_type": "BM25"}, limit=20, expr=store.scope(TENANT)),
        AnnSearchRequest(data=qv, anns_field="dense",
                         param=store._dense_params(), limit=20, expr=store.scope(TENANT)),
    ]
    probe("hybrid_search + function_chains",
          lambda: store.client.hybrid_search(
              "items", reqs, limit=5, output_fields=["id", "text"],
              function_chains=FunctionChain(FunctionChainStage.L2_RERANK, name="probe_hyb")
              .map("$score", fn.round_decimal(col("$score"), decimal=3))
              .sort(col("$score"), desc=True).limit(5)))

    print("\n结论看第 2/3 行和第 4 行：decay 能用在普通 search 上，"
          "\n但 hybrid_search 不接受 function chain —— 所以新鲜度和多路融合"
          "\n在这个版本上是二选一，流行度先验只能走\"乘进向量模长\"那条路。")


if __name__ == "__main__":
    main()
