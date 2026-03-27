"""
pipeline.py — 完整 RAG 流程组装

学习重点：
  1. 所有模块如何串联成一个完整流程
  2. 配置驱动：用 dataclass 管理所有可调参数
  3. 两条路径：缓存命中（快）vs 完整 RAG（慢）
  4. 错误处理：每个环节都可能失败，如何优雅降级
  5. 可观测性：每一步耗时、命中情况的日志
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
from dataclasses import dataclass, field
from typing import Optional, Iterator
from sentence_transformers import SentenceTransformer

from ingestion import IngestionPipeline, Chunk
from indexing import IndexBuilder, VectorIndex, BM25Index
from raptor import RaptorBuilder, RaptorRetriever, make_mock_summarizer
from query_enhancement import QueryEnhancer, MockQueryEnhancer, EnhancedQuery, LLMClient
from retrieval import RetrievalPipeline, RetrievalResult
from cache import SemanticCache
from generation import Generator, PromptBuilder, TokenCounter
from conversation import ConversationHistory, Turn, SessionManager


# ─────────────────────────────────────────
# 配置
# ─────────────────────────────────────────

@dataclass
class RAGConfig:
    """
    所有可调参数集中在一个配置类里。

    好处：
      1. 不用翻代码找参数，一处改动全局生效
      2. 方便做 A/B 测试（两套 config，对比效果）
      3. 可以序列化到 YAML/JSON，由运维人员调整
    """
    # 模型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "claude-sonnet-4-6"

    # 摄入
    chunk_size: int = 500
    chunk_overlap: int = 50

    # 检索
    retrieval_top_k: int = 20     # 粗检索数量
    final_top_k: int = 5          # 最终返回给 LLM 的 chunk 数
    use_reranker: bool = False     # 是否用 Cross-Encoder 精排（需要额外下载模型）
    use_mmr: bool = True           # 是否用 MMR 去重
    use_raptor: bool = True        # 是否启用 RAPTOR 层级召回

    # Query 增强
    use_hyde: bool = False         # 需要 LLM，演示模式默认关
    use_multi_query: bool = False  # 需要 LLM，演示模式默认关
    use_step_back: bool = False

    # 缓存
    use_cache: bool = True
    cache_threshold: float = 0.88  # 用比之前更宽松的阈值（bge-small 的相似度上限较低）
    cache_ttl: int = 3600

    # 生成
    max_context_tokens: int = 3000
    min_relevance_score: float = 0.2

    # 路径
    data_dir: str = "./data"
    index_dir: str = "./index"


# ─────────────────────────────────────────
# 完整 RAG 管道
# ─────────────────────────────────────────

class RAGPipeline:
    """
    把所有模块组装成一个完整的 RAG 系统。

    两条执行路径：

    路径 A（缓存命中，~10ms）：
      query → embed → 语义缓存查找 → 命中 → 直接返回

    路径 B（完整 RAG，~2~5s）：
      query → 指代消解 → Query增强 → 混合检索 → Rerank → MMR
            → Prompt构建 → LLM生成 → 写缓存 → 返回

    RAPTOR 的位置：
      RAPTOR 是索引层的一部分，不是单独的检索路径。
      检索时，RAPTOR 的节点（摘要层 + 叶子层）都在向量索引里，
      和普通 chunk 一起参与 hybrid search。
      宏观问题自然命中摘要节点，细节问题命中叶子节点。
    """

    def __init__(self, config: RAGConfig, api_key: str = None):
        self.config = config
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._initialized = False

        # 检查是否有有效 LLM key
        self.has_llm = self.api_key.startswith("sk-ant-")

    def initialize(self, force_reindex: bool = False):
        """
        初始化所有组件。
        分开 __init__ 和 initialize 的原因：
          允许先创建对象，再决定是否需要重建索引，
          不会在 import 时就触发耗时操作。
        """
        t0 = time.time()
        print("\n[RAGPipeline] 初始化...")

        # 1. Embedding 模型（所有组件共用一个，避免重复加载）
        print(f"  加载 Embedding 模型: {self.config.embedding_model}")
        self.embed_model = SentenceTransformer(self.config.embedding_model)

        # 2. 索引（优先从磁盘加载，避免重复计算）
        index_exists = os.path.exists(f"{self.config.index_dir}/vector/faiss.index")

        if force_reindex or not index_exists:
            print("  构建索引（首次运行或强制重建）...")
            self._build_index()
        else:
            print("  从磁盘加载已有索引...")
            self.index_builder = IndexBuilder.load(
                self.embed_model, self.config.index_dir
            )

        # 3. RAPTOR（层级召回）
        # 注意：RAPTOR 树总是在进程内重建，不从磁盘 pickle 加载。
        # 原因：pickle 保存时绑定的是保存时的模块路径（如 __main__ 或 raptor），
        # 在不同入口加载时类名解析会失败（AttributeError）。
        # Mock 摘要重建一次只需 ~2s，成本可以接受。
        # 生产环境若需要跨进程复用，改用 JSON/MessagePack 序列化树结构。
        if self.config.use_raptor:
            print("  构建 RAPTOR 树...")
            self._build_raptor()

        # 4. 检索管道
        self.retrieval_pipeline = RetrievalPipeline(
            vector_index=self.index_builder.vector_index,
            bm25_index=self.index_builder.bm25_index,
            embedding_model=self.embed_model,
            use_reranker=self.config.use_reranker,
            use_mmr=self.config.use_mmr,
        )

        # 5. Query 增强器
        if self.has_llm and (self.config.use_hyde or self.config.use_multi_query):
            llm_client = LLMClient(self.api_key)
            self.query_enhancer = QueryEnhancer(llm_client)
        else:
            self.query_enhancer = MockQueryEnhancer()

        # 6. 语义缓存
        if self.config.use_cache:
            # 同 RAPTOR，缓存总是从空开始（避免 pickle 跨模块类名解析问题）
            # 生产环境用 Redis Stack 持久化，不依赖 pickle
            self.cache = SemanticCache(
                embedding_model=self.embed_model,
                similarity_threshold=self.config.cache_threshold,
                ttl_seconds=self.config.cache_ttl,
            )
        else:
            self.cache = None

        # 7. 生成器
        if self.has_llm:
            self.generator = Generator(self.api_key, self.config.llm_model)
        else:
            self.generator = None

        # 8. 会话管理
        self.session_manager = SessionManager()

        self._initialized = True
        print(f"[RAGPipeline] 初始化完成 ({time.time()-t0:.1f}s)\n")

    def _build_index(self):
        """构建并保存索引"""
        ingest = IngestionPipeline(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = ingest.ingest_directory(self.config.data_dir)

        self.index_builder = IndexBuilder(self.embed_model)
        self.index_builder.build(chunks)
        self.index_builder.save(self.config.index_dir)

    def _build_raptor(self):
        """构建并保存 RAPTOR 树"""
        ingest = IngestionPipeline(chunk_size=400, chunk_overlap=40)
        chunks = ingest.ingest_directory(self.config.data_dir)

        summarizer = make_mock_summarizer()
        raptor_builder = RaptorBuilder(
            self.embed_model, summarizer, max_levels=2, min_nodes_to_cluster=4
        )
        tree = raptor_builder.build(chunks)
        self.raptor_retriever = RaptorRetriever(tree, self.embed_model)
        self.raptor_retriever.save(f"{self.config.index_dir}/raptor")

    def ask(
        self,
        question: str,
        session_id: str = "default",
    ) -> dict:
        """
        单次问答，完整流程。
        返回标准格式的响应字典。
        """
        assert self._initialized, "请先调用 pipeline.initialize()"
        t_start = time.time()

        print(f"\n{'='*50}")
        print(f"[RAG] 问题: {question}")

        # ── 获取会话历史
        history = self.session_manager.get_or_create(session_id)
        history_messages = history.get_messages_for_prompt()
        recent_turns = history.get_recent_turns(3)

        # ── 路径 A：缓存检查
        if self.cache:
            cached = self.cache.get(question)
            if cached:
                t_total = time.time() - t_start
                print(f"[RAG] 缓存命中，耗时 {t_total*1000:.0f}ms")

                turn = Turn(
                    question=question,
                    answer=cached.answer,
                    sources=cached.sources,
                    from_cache=True,
                )
                history.add_turn(turn)
                self.session_manager.save(history)

                return {
                    "answer": cached.answer,
                    "sources": cached.sources,
                    "from_cache": True,
                    "latency_ms": int(t_total * 1000),
                }

        # ── 路径 B：完整 RAG

        # Step 1: 指代消解 + Query 增强
        t1 = time.time()
        coref_history = [{"role": "user" if i%2==0 else "assistant", "content": t.question if i%2==0 else t.answer}
                         for i, t in enumerate(recent_turns) for _ in range(2)]

        if self.has_llm and hasattr(self.query_enhancer, 'coref'):
            resolved_q = self.query_enhancer.coref.resolve(question, coref_history)
        else:
            resolved_q = question

        enhanced = EnhancedQuery(
            original=question,
            resolved_query=resolved_q if resolved_q != question else None,
        )

        # 加入 Mock 改写（演示模式）
        if not self.has_llm:
            enhanced.multi_queries = []

        print(f"  Step1 Query增强: {time.time()-t1:.2f}s")

        # Step 2: 混合检索
        t2 = time.time()
        results = self.retrieval_pipeline.retrieve(
            enhanced,
            final_top_k=self.config.final_top_k,
            retrieval_top_k=self.config.retrieval_top_k,
        )
        print(f"  Step2 检索: {time.time()-t2:.2f}s, {len(results)} 个结果")

        # Step 3: 生成答案
        t3 = time.time()
        if self.generator and results:
            try:
                response = self.generator.generate(
                    resolved_q,
                    results,
                    conversation_history=history_messages,
                    min_relevance_score=self.config.min_relevance_score,
                )
            except Exception as e:
                # LLM 调用失败（auth错误、网络超时等）→ 降级到 mock 生成
                print(f"  [Warning] LLM 调用失败({type(e).__name__})，降级到 Mock 生成")
                response = self._mock_generate(resolved_q, results)
        else:
            # 没有 LLM 时，Mock 生成（展示检索到了什么）
            response = self._mock_generate(resolved_q, results)

        print(f"  Step3 生成: {time.time()-t3:.2f}s")

        # Step 4: 写入缓存
        if self.cache and response.get("answer"):
            self.cache.set(question, response["answer"], response.get("sources", []))

        # Step 5: 更新会话历史
        t_total = time.time() - t_start
        turn = Turn(
            question=question,
            answer=response["answer"],
            sources=response.get("sources", []),
            token_count=response.get("usage", {}).get("input_tokens", 0),
        )
        history.add_turn(turn)
        self.session_manager.save(history)

        response["latency_ms"] = int(t_total * 1000)
        response["from_cache"] = False
        print(f"[RAG] 完成，总耗时 {t_total*1000:.0f}ms")
        return response

    def ask_stream(
        self,
        question: str,
        session_id: str = "default",
    ) -> Iterator[str]:
        """
        流式问答。先检查缓存，缓存命中直接返回；否则流式生成。
        """
        assert self._initialized, "请先调用 pipeline.initialize()"

        history = self.session_manager.get_or_create(session_id)

        # 缓存检查
        if self.cache:
            cached = self.cache.get(question)
            if cached:
                yield cached.answer
                return

        # 完整流程（简化版，不做 query 增强）
        enhanced = EnhancedQuery(original=question)
        results = self.retrieval_pipeline.retrieve(enhanced, final_top_k=self.config.final_top_k)

        if not self.generator:
            yield self._mock_generate(question, results)["answer"]
            return

        full_answer = ""
        for chunk in self.generator.generate_stream(question, results):
            full_answer += chunk
            yield chunk

        # 流式完成后写缓存
        if self.cache and full_answer:
            sources = list({r.chunk.metadata.get("source", "") for r in results})
            self.cache.set(question, full_answer, sources)

    def _mock_generate(self, question: str, results: list[RetrievalResult]) -> dict:
        """
        无 LLM 时的 Mock 生成：直接拼接检索到的 chunk 内容。
        这是调试时很有用的模式：可以验证检索质量，不依赖 LLM。
        """
        if not results:
            return {"answer": "未检索到相关内容。", "sources": [], "confidence": "low"}

        # 取最相关的 chunk 内容
        top_result = results[0]
        answer = (
            f"[Mock 模式，无 LLM] 根据检索结果，最相关内容如下：\n\n"
            f"{top_result.chunk.content[:300]}...\n\n"
            f"[来源: {top_result.chunk.metadata.get('source', '未知')}]"
        )
        sources = list({r.chunk.metadata.get("source", "") for r in results})
        return {"answer": answer, "sources": sources, "confidence": "medium"}

    def ingest_file(self, file_path: str):
        """动态添加新文档（不需要重建全部索引）"""
        assert self._initialized
        ingest = IngestionPipeline(self.config.chunk_size, self.config.chunk_overlap)
        new_chunks = ingest.ingest_file(file_path)

        if new_chunks:
            import numpy as np
            from indexing import embed_chunks
            embeddings = embed_chunks(self.embed_model, new_chunks)
            self.index_builder.vector_index.add(new_chunks, embeddings)
            # BM25 需要重建（不支持增量添加）
            all_chunks = self.index_builder.bm25_index.chunks + new_chunks
            self.index_builder.bm25_index.build(all_chunks)

            # 使对应缓存失效
            if self.cache:
                self.cache.invalidate_by_source(os.path.basename(file_path))

            print(f"[RAGPipeline] 已添加 {len(new_chunks)} 个新 chunk")


# ─────────────────────────────────────────
# 演示运行
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    console.print("\n[bold cyan]═══ 完整 RAG 流程演示 ═══[/bold cyan]\n")

    config = RAGConfig(
        use_raptor=True,
        use_cache=True,
        cache_threshold=0.88,
        use_reranker=False,
        use_hyde=False,
        use_multi_query=False,
    )

    pipeline = RAGPipeline(config)
    pipeline.initialize()

    test_session = "demo_session"

    questions = [
        "年假需要提前几天申请？",
        "密码需要满足哪些复杂度要求？",
        "采购超过五万块需要谁审批？",
        "年假需要提前几天申请？",   # 第二次问同一个问题，测试缓存
    ]

    for q in questions:
        result = pipeline.ask(q, session_id=test_session)
        console.print(Panel(
            f"[bold]问：[/bold]{q}\n\n"
            f"[bold]答：[/bold]{result['answer'][:300]}\n\n"
            f"[dim]来源: {result['sources']} | "
            f"缓存: {'是' if result['from_cache'] else '否'} | "
            f"耗时: {result['latency_ms']}ms[/dim]",
            border_style="green" if result["from_cache"] else "blue",
        ))

    # 缓存统计
    if pipeline.cache:
        console.print(f"\n[bold]缓存统计：[/bold] {pipeline.cache.stats}")

    console.print("\n[green]✓ 完整 RAG 流程演示完成，下一步：server.py[/green]")
