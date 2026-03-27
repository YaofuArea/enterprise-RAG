"""
retrieval.py — 混合检索 + 重排序

学习重点：
  1. RRF（倒数排名融合）：如何把向量分数和 BM25 分数合并
  2. 为什么用排名而非分数来融合
  3. Cross-Encoder Reranker：和 Bi-Encoder 的本质区别
  4. MMR（最大边际相关性）：让检索结果多样化，避免同质化
  5. 多 Query 结果合并：去重 + 重排
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 修复 Mac 上 FAISS + PyTorch 的 OpenMP 冲突
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder

from ingestion import Chunk
from indexing import VectorIndex, BM25Index
from query_enhancement import EnhancedQuery


# ─────────────────────────────────────────
# 检索结果数据结构
# ─────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk: Chunk
    vector_score: float = 0.0       # 向量相似度得分（余弦，[-1,1]）
    bm25_score: float = 0.0         # BM25 词频得分（无界，仅用于排名）
    vector_rank: int = 999          # 在向量检索结果中的排名（1=最好）
    bm25_rank: int = 999            # 在 BM25 结果中的排名
    rrf_score: float = 0.0          # RRF 融合分数
    rerank_score: float = 0.0       # Cross-Encoder 重排分数
    final_score: float = 0.0        # 最终排序分数


# ─────────────────────────────────────────
# 1. 混合检索 + RRF 融合
# ─────────────────────────────────────────

class HybridRetriever:
    """
    混合检索：向量检索（Dense）+ BM25（Sparse）→ RRF 融合。

    RRF（Reciprocal Rank Fusion）原理：

    核心公式：
      RRF_score(d) = Σ  1 / (k + rank_i(d))

      d     = 文档
      rank_i = 文档 d 在第 i 个检索系统中的排名（1-based）
      k     = 平滑常数，通常取 60（经验值，防止排名第1的文档分数过高）

    例子（k=60）：
      文档A：向量排名第1，BM25排名第3
        RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

      文档B：向量排名第5，BM25排名第1
        RRF = 1/(60+5) + 1/(60+1) = 0.01538 + 0.01639 = 0.03177

      文档A 略高于 文档B，因为在向量检索中排名更靠前。

    为什么用排名而非分数来融合：

    问题：向量分数（0.7）和 BM25 分数（8.5）完全不在同一个量纲。
      - 向量分数是余弦相似度，值域 [-1,1]，差异通常很小（0.5~0.9）
      - BM25 分数是对数概率加权求和，值域 [0, ∞)，不同 query 的分数差异很大

    直接把两个分数加权相加（alpha×vec + beta×bm25）需要手动调权重，
    而且对不同 query 效果不稳定。

    用排名（1,2,3...）代替分数：
      - 排名是序数，天然可比
      - 不需要归一化
      - k=60 的平滑让排名差距不会过于悬殊
      - 在一个系统里没出现的文档，排名默认为极大值（如 999）

    k=60 的来源：
      2009 年 Cormack 等人的论文实验，在大量数据集上测试，60 是最优经验值。
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        embedding_model: SentenceTransformer,
        rrf_k: int = 60,
    ):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.model = embedding_model
        self.k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: int = 20,    # 向量检索多取一些，RRF 后再筛
        bm25_top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        单个 query 的混合检索。
        """
        # 1. 向量检索
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)[0]
        vec_results = self.vector_index.search(query_emb, top_k=vector_top_k)

        # 2. BM25 检索
        bm25_results = self.bm25_index.search(query, top_k=bm25_top_k)

        # 3. RRF 融合
        return self._rrf_merge(vec_results, bm25_results, top_k)

    def search_multi_query(
        self,
        enhanced: EnhancedQuery,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        多 Query 检索：对 EnhancedQuery 中的所有 query 分别检索，结果合并。

        合并策略：
          1. 每个 query 独立检索，得到各自的 RRF 排序结果
          2. 跨 query 合并时，同一个 chunk 可能被多个 query 命中
          3. 被多个 query 命中的 chunk，取所有命中中 RRF 分数最高的
          4. 最后统一排序

        为什么取最高而不是求和：
          求和会让被多个改写版本命中的 chunk 分数虚高，
          但这些 chunk 可能只是关键词重叠，不一定是最相关的。
          取最高分能保留真正相关的 chunk，同时不惩罚只被一个 query 命中的 chunk。
        """
        all_queries = enhanced.all_queries()
        print(f"  [HybridRetriever] 检索 {len(all_queries)} 个 query 版本")

        # 每个 chunk 的最高分（用 content_hash 作为去重 key）
        chunk_best: dict[str, RetrievalResult] = {}

        for q in all_queries:
            results = self.search(q, top_k=top_k * 2)
            for res in results:
                chunk_key = res.chunk.metadata.get("content_hash", res.chunk.content[:50])
                if chunk_key not in chunk_best or res.rrf_score > chunk_best[chunk_key].rrf_score:
                    chunk_best[chunk_key] = res

        # 按 RRF 分数排序，取 top_k
        merged = sorted(chunk_best.values(), key=lambda r: r.rrf_score, reverse=True)
        return merged[:top_k]

    def _rrf_merge(
        self,
        vec_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """RRF 核心逻辑"""

        # 建立 chunk_key → RetrievalResult 的映射
        results: dict[str, RetrievalResult] = {}

        def get_key(chunk: Chunk) -> str:
            return chunk.metadata.get("content_hash", chunk.content[:50])

        # 处理向量检索结果，记录排名和分数
        for rank, (chunk, score) in enumerate(vec_results, start=1):
            key = get_key(chunk)
            if key not in results:
                results[key] = RetrievalResult(chunk=chunk)
            results[key].vector_score = score
            results[key].vector_rank = rank

        # 处理 BM25 结果，记录排名和分数
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            key = get_key(chunk)
            if key not in results:
                results[key] = RetrievalResult(chunk=chunk)
            results[key].bm25_score = score
            results[key].bm25_rank = rank

        # 计算 RRF 分数
        for res in results.values():
            rrf_vec = 1.0 / (self.k + res.vector_rank)   # 向量检索贡献
            rrf_bm25 = 1.0 / (self.k + res.bm25_rank)    # BM25 贡献
            res.rrf_score = rrf_vec + rrf_bm25
            res.final_score = res.rrf_score

        # 按 RRF 分数排序
        sorted_results = sorted(results.values(), key=lambda r: r.rrf_score, reverse=True)
        return sorted_results[:top_k]


# ─────────────────────────────────────────
# 2. Cross-Encoder Reranker（精排）
# ─────────────────────────────────────────

class Reranker:
    """
    Cross-Encoder 精排器。

    Bi-Encoder（向量检索）vs Cross-Encoder（重排序）：

    Bi-Encoder（检索阶段）：
      query  → encoder → embedding_q
      doc    → encoder → embedding_d
      相似度 = cosine(embedding_q, embedding_d)

      特点：query 和 doc 独立编码，doc 可以离线预计算存储。
      优点：O(1) 检索（向量索引），可扩展到亿级文档
      缺点：没有 query-doc 的交叉注意力，精度上限有限

    Cross-Encoder（重排序阶段）：
      [query, doc] → encoder → 相关性分数

      特点：query 和 doc 拼接在一起，模型看到完整的 pair。
      优点：query 和 doc 之间有全局注意力，精度大幅高于 Bi-Encoder
      缺点：不能预计算，每个 (query, doc) pair 都要现算，速度 O(N)
      → 只适合对少量候选（top-20 ~ top-50）做精排，不能用于全量检索

    为什么两阶段：
      第一阶段（Bi-Encoder）：从海量文档中快速粗筛 top-K（快，精度70%）
      第二阶段（Cross-Encoder）：对 top-K 精排，输出最终结果（慢但精度高）

    模型选择：
      BAAI/bge-reranker-v2-m3     → 多语言，中文效果好，推荐
      BAAI/bge-reranker-large     → 纯中英，精度更高
      cross-encoder/ms-marco-MiniLM-L6-v2 → 英文，速度极快
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        print(f"[Reranker] 加载 Cross-Encoder 模型: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        # max_length=512：query+doc 拼接后的最大 token 数
        # 超出部分会被截断，所以 chunk 不能太大

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        对粗检索结果做精排。

        输入：query + 粗检索的 top-K 结果（通常 K=20）
        输出：精排后的 top-k 结果（通常 k=5）
        """
        if not results:
            return []

        # 构造 (query, doc) pair 列表
        pairs = [(query, res.chunk.content) for res in results]

        # Cross-Encoder 批量打分
        # 输出：float 数组，值域通常 [-10, 10]（logit），不是概率
        scores = self.model.predict(pairs, show_progress_bar=False)

        # 更新分数
        for res, score in zip(results, scores):
            res.rerank_score = float(score)
            res.final_score = float(score)  # 最终分数改用 rerank 分数

        # 按 rerank 分数重新排序
        reranked = sorted(results, key=lambda r: r.rerank_score, reverse=True)
        return reranked[:top_k]


# ─────────────────────────────────────────
# 3. MMR（最大边际相关性）
# ─────────────────────────────────────────

class MMRSelector:
    """
    MMR（Maximal Marginal Relevance）：在相关性和多样性之间取平衡。

    问题：
      如果用户的文档里有 10 个 chunk 都在讲"年假申请"，
      Top-5 检索结果可能全是相似内容，信息量重复，浪费 context window。

    MMR 公式：
      MMR = argmax [ λ × sim(q, d_i) - (1-λ) × max sim(d_i, d_j) ]
                   ↑ 相关性                   ↑ 与已选 chunk 的相似度（惩罚项）

      λ=1.0 → 纯相关性排序（等同于普通检索）
      λ=0.0 → 纯多样性（最大化差异）
      λ=0.5 → 平衡（常用值）

    贪心选择过程：
      1. 第一个 chunk 选相关性最高的
      2. 后续每个 chunk 选"相关性高 AND 与已选集合差异大"的
      3. 重复直到选满 top_k 个

    适用场景：
      - 文档有大量相似内容（同一主题的多个段落）
      - 希望覆盖多个角度，而非反复强调同一点
      - context window 有限时（减少冗余信息）
    """

    def __init__(self, embedding_model: SentenceTransformer, lambda_param: float = 0.5):
        self.model = embedding_model
        self.lambda_param = lambda_param

    def select(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if len(results) <= top_k:
            return results

        # 计算 query embedding 和所有 chunk 的 embedding
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        chunk_texts = [r.chunk.content for r in results]
        chunk_embs = self.model.encode(
            chunk_texts, normalize_embeddings=True, show_progress_bar=False
        )

        # 计算每个 chunk 与 query 的相似度
        query_sims = np.dot(chunk_embs, query_emb)  # shape: (N,)

        selected_indices = []
        remaining_indices = list(range(len(results)))

        for _ in range(top_k):
            if not remaining_indices:
                break

            if not selected_indices:
                # 第一个：直接选相关性最高的
                best_idx = remaining_indices[np.argmax([query_sims[i] for i in remaining_indices])]
            else:
                # 后续：MMR 分数
                best_score = float("-inf")
                best_idx = remaining_indices[0]

                for idx in remaining_indices:
                    # 相关性项
                    relevance = query_sims[idx]

                    # 与已选集合的最大相似度（惩罚项）
                    selected_embs = chunk_embs[selected_indices]
                    max_redundancy = float(np.max(np.dot(selected_embs, chunk_embs[idx])))

                    # MMR 分数
                    mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_redundancy

                    if mmr > best_score:
                        best_score = mmr
                        best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [results[i] for i in selected_indices]


# ─────────────────────────────────────────
# 完整检索管道
# ─────────────────────────────────────────

class RetrievalPipeline:
    """
    组合所有检索策略的完整管道。

    流程：
    用户 query
        ↓ Query 增强（HyDE / Multi-Query）
    多个 query 版本
        ↓ 混合检索（向量 + BM25 + RRF），每个 query 都检索
    粗检索候选（top-20）
        ↓ Cross-Encoder 精排
    精排结果（top-10）
        ↓ MMR 去重
    最终结果（top-5）
        ↓ 送入 LLM 生成
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        embedding_model: SentenceTransformer,
        use_reranker: bool = True,
        use_mmr: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.hybrid = HybridRetriever(vector_index, bm25_index, embedding_model)
        self.mmr = MMRSelector(embedding_model) if use_mmr else None

        if use_reranker:
            self.reranker = Reranker(reranker_model)
        else:
            self.reranker = None

    def retrieve(
        self,
        enhanced: EnhancedQuery,
        final_top_k: int = 5,
        retrieval_top_k: int = 20,   # 粗检索数量（给 reranker 的候选集）
    ) -> list[RetrievalResult]:
        """完整检索流程"""

        # 1. 混合检索（多 query 版本）
        candidates = self.hybrid.search_multi_query(enhanced, top_k=retrieval_top_k)
        print(f"  [检索] 粗检索候选: {len(candidates)} 个")

        # 2. HyDE：如果有假设答案，用它的 embedding 额外补充一批候选
        if enhanced.hyde_document:
            hyde_enhanced = EnhancedQuery(
                original=enhanced.hyde_document,  # 直接把假设答案当 query
            )
            # 只做向量检索（HyDE 的优势是语义，BM25 对假设答案没意义）
            hyde_emb = self.hybrid.model.encode(
                [enhanced.hyde_document], normalize_embeddings=True
            ).astype(np.float32)[0]
            hyde_results = self.hybrid.vector_index.search(hyde_emb, top_k=10)

            # 把 HyDE 额外找到的 chunk 合并进候选集（去重）
            existing_keys = {
                r.chunk.metadata.get("content_hash", r.chunk.content[:50])
                for r in candidates
            }
            for chunk, score in hyde_results:
                key = chunk.metadata.get("content_hash", chunk.content[:50])
                if key not in existing_keys:
                    new_res = RetrievalResult(chunk=chunk, vector_score=score, rrf_score=score * 0.8)
                    candidates.append(new_res)
                    existing_keys.add(key)

            print(f"  [HyDE] 补充后候选: {len(candidates)} 个")

        # 3. Cross-Encoder 精排
        effective_query = enhanced.resolved_query or enhanced.original
        if self.reranker and candidates:
            candidates = self.reranker.rerank(effective_query, candidates, top_k=retrieval_top_k // 2)
            print(f"  [Reranker] 精排后: {len(candidates)} 个")

        # 4. MMR 去重
        if self.mmr and candidates:
            candidates = self.mmr.select(effective_query, candidates, top_k=final_top_k)
            print(f"  [MMR] 去重后: {len(candidates)} 个")
        else:
            candidates = candidates[:final_top_k]

        return candidates


# ─────────────────────────────────────────
# 演示运行
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    console = Console()

    console.print("\n[bold cyan]═══ 检索管道演示 ═══[/bold cyan]\n")

    # 加载已构建好的索引
    from sentence_transformers import SentenceTransformer
    from indexing import IndexBuilder

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    builder = IndexBuilder.load(model, "./index")

    # 构建检索管道（不用 reranker 加快演示速度，生产环境建议开）
    pipeline = RetrievalPipeline(
        vector_index=builder.vector_index,
        bm25_index=builder.bm25_index,
        embedding_model=model,
        use_reranker=False,  # 演示时关闭，避免下载 reranker 模型
        use_mmr=True,
    )

    # 演示查询
    test_cases = [
        ("年假申请需要提前几天", ["年假申请多长时间", "请假审批流程", "OA系统申请假期"]),
        ("密码需要满足什么要求", ["密码复杂度规则", "账号安全设置"]),
        ("采购申请超过多少金额需要总经理审批", ["大额采购审批流程", "采购金额审批层级"]),
    ]

    for original_query, extra_queries in test_cases:
        console.print(f"\n[yellow]查询：{original_query}[/yellow]")

        # 构造 EnhancedQuery（演示模式，不调 LLM）
        enhanced = EnhancedQuery(
            original=original_query,
            multi_queries=extra_queries,
        )

        results = pipeline.retrieve(enhanced, final_top_k=3, retrieval_top_k=10)

        table = Table(show_lines=True)
        table.add_column("排名", width=4)
        table.add_column("RRF分", width=8)
        table.add_column("向量分", width=8)
        table.add_column("BM25排名", width=8)
        table.add_column("内容（前70字）", width=55)
        table.add_column("来源", style="green", width=15)

        for i, res in enumerate(results, 1):
            table.add_row(
                str(i),
                f"{res.rrf_score:.4f}",
                f"{res.vector_score:.3f}",
                str(res.bm25_rank) if res.bm25_rank < 999 else "-",
                res.chunk.content[:70].replace("\n", " "),
                res.chunk.metadata.get("source", ""),
            )
        console.print(table)

    console.print("\n[green]✓ 检索管道演示完成，下一步：cache.py[/green]")
