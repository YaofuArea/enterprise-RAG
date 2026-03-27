"""
generation.py — RAG 生成层

学习重点：
  1. Prompt 工程：防幻觉、引用要求、拒答指令
  2. Lost-in-the-Middle 问题：上下文排列顺序影响答案质量
  3. Context Window 管理：Token 计数，防止超出限制
  4. 引用溯源：让 LLM 在答案中标注来源
  5. 流式输出（SSE）：生产环境必须有的用户体验
  6. 置信度评估：何时应该拒答
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import tiktoken
from typing import Generator, Iterator
import anthropic

from retrieval import RetrievalResult


# ─────────────────────────────────────────
# Token 计数工具
# ─────────────────────────────────────────

class TokenCounter:
    """
    Token 计数器，防止 Prompt 超出模型 Context Window。

    为什么需要 Token 计数：
      Claude 的 Context Window 有上限（claude-sonnet-4-6 是 200K token）。
      但实际使用中，超长 prompt 会：
        1. 成本飙升（按 token 计费）
        2. 响应变慢（更长的 prompt 推理更慢）
        3. Lost-in-the-Middle 问题加重

      最佳实践：控制检索内容在 2000~4000 token 以内，
      给系统 prompt (~500) 和生成 (~1000) 留空间。

    tiktoken 说明：
      这是 OpenAI 的 tokenizer，和 Anthropic 的不完全一样。
      Claude 的实际 token 数会略有差异，但误差通常 < 5%，够用。
      如果需要精确计数，用 anthropic.count_tokens()（需要 API 调用，慢）。
    """
    def __init__(self, model: str = "cl100k_base"):
        # cl100k_base = GPT-4 用的 tokenizer，对中文估算也足够准
        try:
            self.enc = tiktoken.get_encoding(model)
        except Exception:
            self.enc = None

    def count(self, text: str) -> int:
        if self.enc is None:
            # 没有 tiktoken 时的粗估：中文约 1.5 字符/token，英文约 4 字符/token
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)
        return len(self.enc.encode(text))


# ─────────────────────────────────────────
# Prompt 构建器
# ─────────────────────────────────────────

class PromptBuilder:
    """
    构建 RAG 的 Prompt，这是生成质量的核心。

    关键设计决策：

    1. 角色设定（System Prompt）
       明确告诉模型它是"企业知识库助手"，而不是通用 AI。
       这能减少模型用训练数据回答（可能不适用于当前企业）的概率。

    2. 防幻觉指令
       明确要求"只使用提供的文档内容回答"，
       当文档中没有相关信息时，明确说"根据现有文档无法回答"，
       而不是凭空编造。

    3. 引用格式要求
       指定引用格式（如 [来源: hr_policy.txt]），
       方便用户追溯原始文档验证答案。

    4. Lost-in-the-Middle 问题
       论文："Lost in the Middle: How Language Models Use Long Contexts" (Stanford 2023)

       发现：LLM 对长上下文的注意力分布是 U 形的：
         - 开头的内容：注意力高
         - 结尾的内容：注意力高
         - 中间的内容：注意力显著下降（被"遗忘"）

       解决方案：
         把最相关的 chunk 放在最前和最后，不相关的放中间。
         对于 top-5 结果：排列顺序应是 [1, 3, 5, 4, 2]（相关性降序再升序）

    5. Context 长度控制
       超出 max_context_tokens 时，截断最不相关的 chunk，
       确保 prompt 不超长。
    """

    SYSTEM_PROMPT = """你是一个专业的企业内部知识库助手。

你的职责：
- 严格基于提供的文档内容回答问题，不要使用文档以外的信息
- 如果文档中没有相关信息，明确说"根据现有文档，暂无此问题的记录"
- 回答要准确、简洁，关键数字和条件必须精确引用
- 在回答末尾注明信息来源，格式：[来源: 文件名]

禁止行为：
- 不得编造文档中没有的信息
- 不得根据常识或训练数据回答（除非用户明确允许）
- 不得在没有依据的情况下给出建议"""

    def __init__(
        self,
        max_context_tokens: int = 3000,
        token_counter: TokenCounter = None,
    ):
        self.max_context_tokens = max_context_tokens
        self.counter = token_counter or TokenCounter()

    def build(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
        conversation_history: list[dict] = None,
    ) -> tuple[str, list[dict]]:
        """
        构建 Prompt，返回 (system_prompt, messages)。

        messages 格式符合 Anthropic Messages API。
        """
        # 1. 构建上下文文档块
        context_text, used_sources = self._build_context(retrieval_results)

        # 2. 构建用户消息（包含问题和上下文）
        user_content = f"""请根据以下文档内容回答问题。

=== 参考文档 ===
{context_text}
=== 参考文档结束 ===

问题：{question}

要求：
- 只使用上方文档中的信息
- 引用具体数字和规定时，请精确引用原文
- 在回答末尾标注来源：[来源: 文件名]"""

        # 3. 组装 messages（包含历史对话）
        messages = []
        if conversation_history:
            # 加入历史对话（最多保留最近 6 轮，防止 token 过多）
            for msg in conversation_history[-6:]:
                messages.append(msg)

        messages.append({"role": "user", "content": user_content})

        return self.SYSTEM_PROMPT, messages

    def _build_context(
        self,
        results: list[RetrievalResult],
    ) -> tuple[str, list[str]]:
        """
        把检索结果组装成上下文文本。

        Lost-in-the-Middle 排列策略：
          假设有 5 个 chunk，相关性排名 1~5
          普通排列：[1,2,3,4,5] → 中间的 3,4 注意力弱
          优化排列：[1,3,5,4,2] → 最相关的 1 在最前，2 在最后，都是高注意力区域

          实现：把偶数位置的（不那么相关的）放中间，奇数位置的放两端。
        """
        if not results:
            return "（未找到相关文档）", []

        # Lost-in-the-Middle 重排
        reordered = self._lost_in_middle_reorder(results)

        context_parts = []
        used_sources = []
        total_tokens = 0

        for i, res in enumerate(reordered):
            chunk = res.chunk
            source = chunk.metadata.get("source", "未知来源")
            section = chunk.metadata.get("section", "")

            # 构建单个文档块的标题
            header = f"[文档{i+1}] 来源：{source}"
            if section:
                header += f" | 章节：{section}"

            chunk_text = f"{header}\n{chunk.content}\n"
            chunk_tokens = self.counter.count(chunk_text)

            # Token 超出限制时停止添加
            if total_tokens + chunk_tokens > self.max_context_tokens:
                print(f"  [PromptBuilder] Token 限制({self.max_context_tokens})，"
                      f"已使用 {i} 个 chunk，跳过剩余 {len(reordered)-i} 个")
                break

            context_parts.append(chunk_text)
            used_sources.append(source)
            total_tokens += chunk_tokens

        print(f"  [PromptBuilder] 上下文 ~{total_tokens} tokens，{len(context_parts)} 个 chunk")
        return "\n---\n".join(context_parts), list(set(used_sources))

    def _lost_in_middle_reorder(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """
        Lost-in-the-Middle 排列：最相关的放首尾，次相关的放中间。

        原始顺序（按相关性降序）：[A, B, C, D, E]
        重排后：                  [A, C, E, D, B]
          A 在最前（高注意力）
          B 在最后（高注意力）
          C, D, E 在中间（低注意力）

        这样 A（最相关）和 B（第二相关）都在高注意力区域。
        """
        if len(results) <= 2:
            return results

        # 奇数位（0,2,4...）放前面，偶数位（1,3,5...）放后面（反转）
        front = results[::2]        # 索引 0,2,4... （相关性 1,3,5...）
        back = results[1::2][::-1]  # 索引 1,3,5... 反转（相关性 ...4,2）
        return front + back


# ─────────────────────────────────────────
# 生成器（流式 + 非流式）
# ─────────────────────────────────────────

class Generator:
    """
    调用 LLM 生成答案。支持流式和非流式两种模式。

    流式输出的价值：
      非流式：等待 3~8 秒后，一次性显示完整答案
      流式：  0.5 秒后开始显示第一个字，用户体验好很多

      在企业应用中，流式输出几乎是必须的（用户无法接受 5 秒白屏）。
      实现方式：Server-Sent Events (SSE) 或 WebSocket。

    置信度控制：
      当检索结果相关性很低时（所有 chunk 的分数都低于某个阈值），
      应该拒绝回答而不是用不相关内容瞎编。
      这被称为"幻觉防护（Hallucination Guard）"。
    """

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.prompt_builder = PromptBuilder()

    def generate(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
        conversation_history: list[dict] = None,
        min_relevance_score: float = 0.3,   # 低于此分数时拒答
    ) -> dict:
        """
        非流式生成，返回完整答案。
        """
        # 幻觉防护：检索结果太差时直接拒答
        if not retrieval_results:
            return {
                "answer": "根据现有知识库，暂无此问题的相关记录。请确认您的问题是否在企业文档涵盖范围内，或联系相关部门咨询。",
                "sources": [],
                "confidence": "low",
                "from_cache": False,
            }

        # 用向量相似度分数（0~1 之间）判断相关性，不用 RRF 分数（量纲不同）
        top_score = max(r.vector_score for r in retrieval_results)
        if top_score < min_relevance_score:
            return {
                "answer": f"根据现有文档，未能找到与您问题高度相关的内容（最高相关度: {top_score:.2f}）。建议联系 HR 或 IT 部门获取准确信息。",
                "sources": [],
                "confidence": "low",
                "from_cache": False,
            }

        system_prompt, messages = self.prompt_builder.build(
            question, retrieval_results, conversation_history
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
        )

        answer = response.content[0].text
        sources = list({
            r.chunk.metadata.get("source", "")
            for r in retrieval_results
            if r.chunk.metadata.get("source")
        })

        return {
            "answer": answer,
            "sources": sources,
            "confidence": "high" if top_score > 0.6 else "medium",
            "from_cache": False,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        }

    def generate_stream(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
        conversation_history: list[dict] = None,
    ) -> Iterator[str]:
        """
        流式生成，逐字返回。

        用法（FastAPI SSE）：
            @app.get("/ask")
            async def ask(question: str):
                async def event_stream():
                    for chunk in generator.generate_stream(question, results):
                        yield f"data: {json.dumps({'text': chunk})}\\n\\n"
                return StreamingResponse(event_stream(), media_type="text/event-stream")

        Anthropic streaming 原理：
          client.messages.stream() 返回一个上下文管理器，
          内部通过 HTTP chunked transfer 接收服务器推送的 SSE 事件。
          每个 text_delta 事件包含新增的文本片段。
        """
        if not retrieval_results:
            yield "根据现有知识库，暂无此问题的相关记录。"
            return

        system_prompt, messages = self.prompt_builder.build(
            question, retrieval_results, conversation_history
        )

        # 使用 stream 上下文管理器
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text_chunk in stream.text_stream:
                # text_stream 是一个生成器，每次 yield 一小段文本（通常 1~10 个字）
                yield text_chunk


# ─────────────────────────────────────────
# 演示运行（Mock 模式，不调 LLM）
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from ingestion import Chunk
    from retrieval import RetrievalResult

    console = Console()
    console.print("\n[bold cyan]═══ 生成层演示 ═══[/bold cyan]\n")

    # ── 演示 1：Lost-in-the-Middle 排列
    console.print("[bold]1. Lost-in-the-Middle 排列演示[/bold]")

    # 构造 5 个模拟检索结果
    mock_chunks = [
        Chunk(content=f"文档片段{i}：关于年假的第{i}条规定...", metadata={"source": "hr.txt"})
        for i in range(1, 6)
    ]
    mock_results = [
        RetrievalResult(chunk=c, final_score=1.0 - i*0.15)
        for i, c in enumerate(mock_chunks)
    ]

    builder = PromptBuilder()
    reordered = builder._lost_in_middle_reorder(mock_results)

    console.print("原始顺序（相关性 1→5）:")
    console.print("  " + " → ".join(f"片段{r.chunk.content[4]}" for r in mock_results))
    console.print("重排后（首尾高注意力区域放最相关）:")
    console.print("  " + " → ".join(f"片段{r.chunk.content[4]}" for r in reordered))
    console.print("  [dim]注：最相关的片段1在首，片段2在尾，中间是次相关内容[/dim]")

    # ── 演示 2：Token 计数
    console.print("\n[bold]2. Token 计数演示[/bold]")
    counter = TokenCounter()
    test_texts = [
        ("纯中文", "年假申请须提前3个工作日通过OA系统提交，经直属主管审批后生效。"),
        ("中英混合", "员工须开启 MFA (Multi-Factor Authentication) 进行身份验证。"),
        ("纯英文", "Employees must submit annual leave requests 3 working days in advance."),
    ]
    for label, text in test_texts:
        tokens = counter.count(text)
        console.print(f"  {label}({len(text)}字符): ~{tokens} tokens")

    # ── 演示 3：Prompt 结构
    console.print("\n[bold]3. Prompt 结构展示[/bold]")
    sample_results = mock_results[:3]
    system_prompt, messages = builder.build("年假有几天", sample_results)

    console.print(Panel(system_prompt, title="System Prompt", border_style="blue"))
    console.print(Panel(
        messages[-1]["content"][:500] + "...",
        title="User Message（前500字）",
        border_style="green"
    ))

    # ── 演示 4：真实 LLM 调用（需要有效 API key）
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_real = api_key.startswith("sk-ant-") and os.environ.get("USE_REAL_LLM") == "1"

    if use_real:
        console.print("\n[bold]4. 流式生成演示（真实 LLM）[/bold]")
        from indexing import IndexBuilder
        from sentence_transformers import SentenceTransformer
        from retrieval import RetrievalPipeline
        from query_enhancement import EnhancedQuery

        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        idx_builder = IndexBuilder.load(model, "./index")
        pipeline = RetrievalPipeline(
            idx_builder.vector_index, idx_builder.bm25_index,
            model, use_reranker=False, use_mmr=True
        )

        question = "年假申请需要提前几天，需要谁审批？"
        enhanced = EnhancedQuery(original=question, multi_queries=["年假申请流程", "请假审批"])
        results = pipeline.retrieve(enhanced, final_top_k=3)

        gen = Generator()
        console.print(f"\n[yellow]问题：{question}[/yellow]")
        console.print("答案（流式）：", end="")

        for chunk in gen.generate_stream(question, results):
            console.print(chunk, end="", highlight=False)
        console.print()
    else:
        console.print(f"\n[dim]跳过真实 LLM 调用（设置 USE_REAL_LLM=1 开启）[/dim]")

    console.print("\n[green]✓ 生成层演示完成，下一步：conversation.py[/green]")
