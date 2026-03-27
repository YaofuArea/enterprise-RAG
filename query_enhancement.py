"""
query_enhancement.py — Query 侧增强

学习重点：
  1. HyDE：为什么用"假设答案"去检索比用"问题"更准
  2. Multi-Query：多角度改写，覆盖不同表达方式
  3. Step-Back Prompting：从具体问题退到抽象原则
  4. 指代消解：多轮对话中"它/这个/上述"的还原
  5. 并发调用 LLM：多个改写任务同时发，节省时间
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import anthropic


# ─────────────────────────────────────────
# LLM 客户端封装
# ─────────────────────────────────────────

class LLMClient:
    """
    统一封装 Anthropic 调用，方便后续替换模型。

    模型选择策略：
      Query 增强任务 → claude-haiku（便宜、够用、快）
      最终答案生成   → claude-sonnet（能力强）

    为什么不都用 Sonnet：
      Query 增强调用频繁（每次问答都要调），成本敏感
      Haiku 对"改写问题"这类任务已经足够
      Sonnet 的优势在推理和长文本理解，不在改写
    """
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.fast_model = "claude-haiku-4-5-20251001"   # Query 增强用
        self.smart_model = "claude-sonnet-4-6"           # 最终生成用

    def complete(self, prompt: str, model: str = None, max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=model or self.fast_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    async def complete_async(self, prompt: str, model: str = None, max_tokens: int = 500) -> str:
        """
        异步版本：多个 query 增强任务可以并发执行。
        Multi-Query 生成 3 个改写版本，并发调用比串行快 3x。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.complete(prompt, model, max_tokens)
        )


# ─────────────────────────────────────────
# 增强结果数据结构
# ─────────────────────────────────────────

@dataclass
class EnhancedQuery:
    """Query 增强的输出，包含所有增强后的版本"""
    original: str                          # 用户原始问题
    hyde_document: Optional[str] = None    # HyDE 生成的假设答案
    multi_queries: list[str] = field(default_factory=list)   # Multi-Query 改写版本
    step_back_query: Optional[str] = None  # Step-Back 抽象化的问题
    resolved_query: Optional[str] = None   # 消解指代后的问题（多轮对话用）

    def all_queries(self) -> list[str]:
        """
        返回所有可用于检索的 query 版本。
        检索时对每个版本都检索一遍，结果取并集，最大化召回率。
        """
        queries = [self.original]
        if self.resolved_query and self.resolved_query != self.original:
            queries.append(self.resolved_query)
        queries.extend(self.multi_queries)
        if self.step_back_query:
            queries.append(self.step_back_query)
        # 去重（不同增强方法可能生成相同的 query）
        seen = set()
        return [q for q in queries if not (q in seen or seen.add(q))]


# ─────────────────────────────────────────
# 1. HyDE（假设性文档嵌入）
# ─────────────────────────────────────────

class HyDEEnhancer:
    """
    HyDE: Hypothetical Document Embeddings
    论文：Precise Zero-Shot Dense Retrieval without Relevance Labels (2022)

    核心洞察：
      问题（query）和答案（document）在 embedding 空间里的分布是不同的。
      用"问题的 embedding"去检索"答案的 embedding"存在语义鸿沟。

      例如：
        用户问：  "年假可以折现吗？"
        文档写着："当年未使用的年假逾期作废，不予折现补偿。"

      "折现吗？"和"不予折现补偿"的语义相似度，
      低于"年假不予折现" 和"不予折现补偿"的语义相似度。

    HyDE 的做法：
      1. 让 LLM 根据问题生成一段假设性的答案（即使 LLM 不确定内容）
      2. 用假设答案的 embedding 去检索，而不是问题的 embedding
      3. 假设答案的语言风格和文档更接近，embedding 距离更小

    为什么"假设答案不准确"也没关系：
      HyDE 不直接用假设答案作为最终回答，
      只是用它的 embedding 方向来"导航"向量空间。
      即使答案有错，embedding 的语义方向通常还是对的。

    效果：
      BEIR 基准测试上平均提升约 8-20% 的检索精度（取决于数据集）
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, query: str) -> str:
        """
        生成假设性文档。
        注意 prompt 设计：
          1. 要求生成"文档风格"而非"对话风格"（和真实文档的 embedding 更接近）
          2. 限制长度，太长的假设文档会稀释关键信息
          3. 明确说明"根据企业规章制度"，引导模型生成规范性文档风格
        """
        prompt = f"""请根据以下问题，生成一段企业内部文档风格的回答。
要求：
- 使用正式的文档语言（如"员工须...""应当...""不得..."）
- 包含具体的数字、时间、条件等细节（即使不确定，也要给出合理的示例值）
- 长度控制在100-150字
- 不要说"根据规定"，直接陈述内容

问题：{query}

文档内容："""
        return self.llm.complete(prompt, max_tokens=200)

    async def generate_async(self, query: str) -> str:
        prompt = f"""请根据以下问题，生成一段企业内部文档风格的回答。
要求：使用正式文档语言，包含具体细节，100-150字。

问题：{query}
文档内容："""
        return await self.llm.complete_async(prompt, max_tokens=200)


# ─────────────────────────────────────────
# 2. Multi-Query（多角度改写）
# ─────────────────────────────────────────

class MultiQueryEnhancer:
    """
    Multi-Query：把一个问题改写成多个不同表达，分别检索，结果取并集。

    解决的问题：
      用户的表达和文档的表达往往不一致。
      单一 query 的召回是单点，很容易 miss。
      多个角度的 query 覆盖更广，显著提升召回率。

    例子：
      原始问题："年假申请流程"
      改写1：  "请假需要走什么手续"
      改写2：  "OA系统如何提交假期申请"
      改写3：  "年假审批需要多长时间"

      原始问题只能命中"年假"相关的 chunk，
      改写1 可能额外命中"请假"相关的通用描述，
      改写2 可能命中 IT 系统操作手册里的 OA 相关内容，
      取并集后召回率更全面。

    注意事项：
      - 不要生成太多改写（3-5 个够了），太多会引入噪声
      - 改写要有实质性的多样性，不能只是换个说法
      - 结果合并时要去重，避免重复 chunk 占据 top-k 名额
    """

    def __init__(self, llm: LLMClient, n_queries: int = 3):
        self.llm = llm
        self.n = n_queries

    def generate(self, query: str) -> list[str]:
        prompt = f"""请将以下问题改写成 {self.n} 个不同角度的检索查询。

要求：
- 每个改写从不同角度表达同一个信息需求
- 使用不同的关键词和表达方式（同义词、上位词、下位词）
- 每行输出一个查询，不要编号，不要解释

原始问题：{query}

改写后的查询："""

        response = self.llm.complete(prompt, max_tokens=200)
        # 解析输出：按行分割，过滤空行
        queries = [
            line.strip().lstrip("•-·")  # 去掉可能的列表符号
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return queries[:self.n]  # 只取前 n 个

    async def generate_async(self, query: str) -> list[str]:
        """并发生成多个改写（如果需要多次调用）"""
        prompt = f"""将以下问题改写成 {self.n} 个不同角度的检索查询，每行一个：

{query}"""
        response = await self.llm.complete_async(prompt, max_tokens=200)
        queries = [l.strip() for l in response.split("\n") if l.strip() and len(l.strip()) > 5]
        return queries[:self.n]


# ─────────────────────────────────────────
# 3. Step-Back Prompting
# ─────────────────────────────────────────

class StepBackEnhancer:
    """
    Step-Back Prompting：把具体问题"退一步"变成更抽象的问题。

    论文：Take a Step Back: Evoking Reasoning via Abstraction in LLMs (Google 2023)

    原理：
      具体问题 → 可能只有一个非常精确的 chunk 能回答
      抽象问题 → 更多相关 chunk 都能部分回答，召回更多上下文

    例子：
      具体问题："我请了 5 天年假，今年还剩几天？"
      抽象问题："公司年假制度的规则是什么？"

      回答具体问题，需要先理解年假总天数规则，
      Step-Back 先检索规则，给 LLM 提供足够上下文，再回答具体问题。

    适用场景：
      - 需要背景知识才能回答的具体问题
      - 计算类问题（先查规则，再计算）
      - "为什么"类问题（需要检索原因和背景）

    不适用场景：
      - 本身就很抽象的问题（再退一步会太模糊）
      - 简单的查询类问题（"密码最少几位"直接检索就行）
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, query: str) -> str:
        prompt = f"""请将以下具体问题"退一步"，改写成一个更抽象、更通用的问题，
用于检索背景知识和通用规则。

例如：
  具体问题："我今年的年假余额怎么查？"
  抽象问题："公司年假管理制度的相关规定"

  具体问题："报销需要什么发票？"
  抽象问题："公司费用报销的规则和要求"

具体问题：{query}
抽象问题："""
        return self.llm.complete(prompt, max_tokens=100)

    async def generate_async(self, query: str) -> str:
        prompt = f"""将以下具体问题改写成更抽象的检索查询（一句话）：
{query}
抽象查询："""
        return await self.llm.complete_async(prompt, max_tokens=100)


# ─────────────────────────────────────────
# 4. 指代消解（多轮对话必需）
# ─────────────────────────────────────────

class CoreferenceResolver:
    """
    指代消解：把多轮对话中的代词/指代词还原为具体实体。

    问题场景：
      轮1：用户问"年假申请流程是什么？"
      轮2：用户问"那它需要主管审批吗？"

      "它"指的是"年假申请"，但如果直接把"那它需要主管审批吗"
      送去检索，向量库不知道"它"是什么，召回结果会很差。

    解决：
      结合对话历史，让 LLM 把"它"还原成"年假申请"，
      生成完整的独立问题："年假申请是否需要主管审批？"

    这一步在多轮对话中至关重要，但经常被忽略。
    没有指代消解的多轮 RAG，第 2 轮以后的召回质量会严重下降。
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def resolve(self, query: str, conversation_history: list[dict]) -> str:
        """
        conversation_history 格式：
        [
            {"role": "user", "content": "年假申请流程是什么？"},
            {"role": "assistant", "content": "年假申请须提前3个工作日..."},
            {"role": "user", "content": "那它需要主管审批吗？"},  ← 当前问题
        ]
        """
        if not conversation_history:
            return query

        # 检查是否包含指代词（简单规则过滤，避免每次都调 LLM）
        reference_words = ["它", "这个", "那个", "上述", "该", "此", "这些", "那些",
                           "其", "之前", "刚才", "前面"]
        needs_resolve = any(word in query for word in reference_words)

        if not needs_resolve:
            return query  # 没有指代词，直接返回原问题

        # 构建对话历史的文本表示
        history_text = ""
        for msg in conversation_history[-4:]:  # 只用最近 4 轮，避免 context 太长
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}：{msg['content']}\n"

        prompt = f"""根据对话历史，将最后一个问题中的指代词（它/这个/该/上述等）
替换为具体的实体，使问题可以独立理解。只输出改写后的问题，不要解释。

对话历史：
{history_text}
最后的问题：{query}

改写后的完整问题："""

        resolved = self.llm.complete(prompt, max_tokens=150)

        # 如果 LLM 返回的结果比原问题短很多，可能出错了，退回原始问题
        if len(resolved) < len(query) * 0.5:
            return query

        return resolved


# ─────────────────────────────────────────
# 组合：Query 增强器
# ─────────────────────────────────────────

class QueryEnhancer:
    """
    统一的 Query 增强器，组合所有增强策略。

    使用策略选择：
      不是每次都要用所有增强，要根据 query 特征选择：

      HyDE       → 知识密集型问题（需要背景知识的查询）
      Multi-Query → 几乎所有查询都适用（成本低，收益稳定）
      Step-Back  → 具体的、依赖背景知识的问题
      指代消解   → 多轮对话中必须做，单轮不需要

    性能优化：
      Multi-Query 和 HyDE 可以并发执行（都是独立的 LLM 调用）
      用 asyncio.gather 同时发起，总时间 ≈ 单次调用时间，而非 2x
    """

    def __init__(self, llm: LLMClient):
        self.hyde = HyDEEnhancer(llm)
        self.multi_query = MultiQueryEnhancer(llm, n_queries=3)
        self.step_back = StepBackEnhancer(llm)
        self.coref = CoreferenceResolver(llm)

    def enhance(
        self,
        query: str,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        use_step_back: bool = False,   # 默认关，有需要时开
        conversation_history: list[dict] = None,
    ) -> EnhancedQuery:
        """
        同步版本：串行执行所有增强策略。
        """
        result = EnhancedQuery(original=query)

        # 1. 指代消解（最先做，后续所有增强都用消解后的 query）
        if conversation_history:
            result.resolved_query = self.coref.resolve(query, conversation_history)
            effective_query = result.resolved_query
        else:
            effective_query = query

        # 2. HyDE
        if use_hyde:
            print("  [HyDE] 生成假设答案...")
            result.hyde_document = self.hyde.generate(effective_query)

        # 3. Multi-Query
        if use_multi_query:
            print("  [Multi-Query] 生成改写版本...")
            result.multi_queries = self.multi_query.generate(effective_query)

        # 4. Step-Back
        if use_step_back:
            print("  [Step-Back] 生成抽象查询...")
            result.step_back_query = self.step_back.generate(effective_query)

        return result

    async def enhance_async(
        self,
        query: str,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        use_step_back: bool = False,
        conversation_history: list[dict] = None,
    ) -> EnhancedQuery:
        """
        异步并发版本：HyDE、Multi-Query、Step-Back 同时发起，节省等待时间。

        串行时间：T_hyde + T_multiquery + T_stepback ≈ 3s
        并发时间：max(T_hyde, T_multiquery, T_stepback) ≈ 1s
        """
        result = EnhancedQuery(original=query)

        # 先做指代消解（其他增强依赖它的结果）
        if conversation_history:
            result.resolved_query = self.coref.resolve(query, conversation_history)
            effective_query = result.resolved_query
        else:
            effective_query = query

        # 并发执行其余增强
        tasks = {}
        if use_hyde:
            tasks["hyde"] = self.hyde.generate_async(effective_query)
        if use_multi_query:
            tasks["multi"] = self.multi_query.generate_async(effective_query)
        if use_step_back:
            tasks["step_back"] = self.step_back.generate_async(effective_query)

        if tasks:
            # asyncio.gather 并发执行所有任务
            keys = list(tasks.keys())
            values = await asyncio.gather(*tasks.values())
            results_map = dict(zip(keys, values))

            if "hyde" in results_map:
                result.hyde_document = results_map["hyde"]
            if "multi" in results_map:
                result.multi_queries = results_map["multi"]
            if "step_back" in results_map:
                result.step_back_query = results_map["step_back"]

        return result


# ─────────────────────────────────────────
# Mock 版本（不需要 API key，用于测试）
# ─────────────────────────────────────────

class MockQueryEnhancer:
    """
    不依赖 LLM 的 Mock 版本，用于测试和演示结构。
    在没有 API key 时展示增强结果的结构。
    """

    def enhance(self, query: str, **kwargs) -> EnhancedQuery:
        result = EnhancedQuery(original=query)

        result.hyde_document = (
            f'根据企业相关制度规定，关于"{query}"的内容如下：员工须按照'
            f"公司规章制度执行，具体流程需提前申请，经审批后方可执行。"
        )
        result.multi_queries = [
            f"{query}的具体流程",
            f"如何办理{query[:8]}相关手续",
            f"{query[:6]}的规定和要求",
        ]
        result.step_back_query = f"公司关于{query[:6]}的总体制度规定"

        return result


# ─────────────────────────────────────────
# 演示运行
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print("\n[bold cyan]═══ Query 增强演示 ═══[/bold cyan]\n")

    # 判断是否有可用的 API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_real = api_key.startswith("sk-ant-") and os.environ.get("USE_REAL_LLM") == "1"

    test_queries = [
        {
            "query": "年假申请流程是什么",
            "history": None,
            "desc": "单轮查询"
        },
        {
            "query": "那它需要主管审批吗",
            "history": [
                {"role": "user", "content": "年假申请流程是什么"},
                {"role": "assistant", "content": "年假申请须提前3个工作日通过OA系统提交，经直属主管审批后生效。"},
            ],
            "desc": "多轮对话（含指代词）"
        },
        {
            "query": "密码过期了怎么修改",
            "history": None,
            "desc": "IT 相关查询"
        },
    ]

    for case in test_queries:
        console.print(f"\n[bold yellow]场景：{case['desc']}[/bold yellow]")
        console.print(f"原始问题：[cyan]{case['query']}[/cyan]")

        if use_real:
            llm = LLMClient()
            enhancer = QueryEnhancer(llm)
            enhanced = enhancer.enhance(
                case["query"],
                use_hyde=True,
                use_multi_query=True,
                use_step_back=True,
                conversation_history=case["history"],
            )
        else:
            enhancer = MockQueryEnhancer()
            enhanced = enhancer.enhance(case["query"])
            if case["history"]:
                # 模拟指代消解效果
                enhanced.resolved_query = case["query"].replace("它", "年假申请").replace("那", "")

        # 展示结果
        if enhanced.resolved_query:
            console.print(f"  [green]→ 指代消解后：[/green]{enhanced.resolved_query}")

        if enhanced.hyde_document:
            console.print(Panel(
                enhanced.hyde_document[:200],
                title="[blue]HyDE 假设答案（用此 embedding 检索）[/blue]",
                border_style="blue",
            ))

        if enhanced.multi_queries:
            console.print("  [magenta]Multi-Query 改写：[/magenta]")
            for i, q in enumerate(enhanced.multi_queries, 1):
                console.print(f"    {i}. {q}")

        if enhanced.step_back_query:
            console.print(f"  [yellow]Step-Back 抽象查询：[/yellow]{enhanced.step_back_query}")

        console.print(f"  [dim]→ 共 {len(enhanced.all_queries())} 个检索 query（取并集）[/dim]")

    console.print(f"\n[dim]模式: {'真实 LLM' if use_real else 'Mock（设置 USE_REAL_LLM=1 使用真实 LLM）'}[/dim]")
    console.print("\n[green]✓ Query 增强演示完成，下一步：retrieval.py[/green]")
