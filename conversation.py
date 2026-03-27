"""
conversation.py — 多轮对话管理

学习重点：
  1. 对话历史的存储结构
  2. Token 预算管理：历史对话会占用 context window
  3. 指代消解与历史对话的配合（已在 query_enhancement.py 实现）
  4. 对话摘要压缩：历史太长时用 LLM 压缩
  5. 会话隔离：多用户场景下的 session 管理
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import json
from dataclasses import dataclass, field
from typing import Optional
from generation import TokenCounter


# ─────────────────────────────────────────
# 对话历史
# ─────────────────────────────────────────

@dataclass
class Turn:
    """单轮对话"""
    question: str
    answer: str
    sources: list[str]
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0       # 这一轮消耗的 token 数（用于统计成本）
    from_cache: bool = False   # 是否来自缓存（来自缓存的不消耗 LLM token）


class ConversationHistory:
    """
    对话历史管理器。

    核心问题：历史对话放多少进 Prompt？

    不放：每次都是单轮，用户体验差（"它"无法指代）
    全放：历史越来越长，token 超限，成本飙升

    解决策略（三档）：

    1. 滑动窗口（最简单）
       只保留最近 N 轮对话，老的直接丢弃。
       缺点：丢失早期重要信息。
       适用：对话长度可控，早期信息不重要的场景。

    2. 摘要压缩（平衡）
       当历史超过阈值时，用 LLM 把前几轮压缩成摘要，
       后续对话保留摘要 + 最近几轮原文。
       优点：压缩后还能保留关键信息
       缺点：需要额外的 LLM 调用，有延迟和成本

    3. 选择性保留（精细）
       LLM 判断哪些历史轮次与当前问题相关，只保留相关的。
       成本最高，但信息保留最精准。

    这里实现前两种，生产推荐策略 2。
    """

    def __init__(
        self,
        session_id: str,
        max_turns: int = 10,                 # 最多保留的对话轮数
        max_history_tokens: int = 2000,      # 历史对话的最大 token 数
        compression_threshold: int = 1500,   # 超过此 token 数时触发摘要压缩
    ):
        self.session_id = session_id
        self.max_turns = max_turns
        self.max_history_tokens = max_history_tokens
        self.compression_threshold = compression_threshold

        self.turns: list[Turn] = []
        self.compressed_summary: str = ""     # 被压缩的历史摘要
        self._counter = TokenCounter()

    def add_turn(self, turn: Turn):
        """添加一轮对话"""
        self.turns.append(turn)

        # 超过最大轮数时，丢弃最老的
        if len(self.turns) > self.max_turns:
            dropped = self.turns.pop(0)
            print(f"  [History] 超过 {self.max_turns} 轮，丢弃最早一轮: '{dropped.question[:20]}'")

    def get_messages_for_prompt(self) -> list[dict]:
        """
        返回适合放进 Prompt 的历史消息列表。
        格式符合 Anthropic Messages API: [{"role": "user/assistant", "content": "..."}]

        Token 管理：
          累计计算历史 token 数，超出预算时从最老的轮次开始丢弃，
          直到历史 token 总数在 max_history_tokens 以内。
        """
        messages = []

        # 如果有压缩摘要，加在最前面
        if self.compressed_summary:
            messages.append({
                "role": "user",
                "content": f"[对话摘要] 以下是之前对话的摘要：{self.compressed_summary}"
            })
            messages.append({
                "role": "assistant",
                "content": "我已了解之前的对话内容，请继续。"
            })

        # 从最新往最老遍历，累计 token 数
        # （优先保留最近的轮次，因为指代消解主要依赖最近几轮）
        recent_turns = []
        total_tokens = self._counter.count(self.compressed_summary)

        for turn in reversed(self.turns):
            turn_text = turn.question + turn.answer
            turn_tokens = self._counter.count(turn_text)

            if total_tokens + turn_tokens > self.max_history_tokens:
                print(f"  [History] 历史 token 超出 {self.max_history_tokens}，"
                      f"截断至最近 {len(recent_turns)} 轮")
                break

            recent_turns.insert(0, turn)
            total_tokens += turn_tokens

        # 把轮次转成 messages 格式
        for turn in recent_turns:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})

        return messages

    def get_recent_turns(self, n: int = 3) -> list[Turn]:
        """返回最近 n 轮对话（用于指代消解）"""
        return self.turns[-n:]

    def compress_with_llm(self, llm_client) -> str:
        """
        用 LLM 把历史对话压缩成摘要。

        触发时机：当前历史 token 数超过 compression_threshold 时调用。

        压缩策略：
          把前一半的历史轮次压缩成摘要，保留后一半原文。
          这样既减少了 token，又保留了最新轮次的完整上下文。

        压缩后效果：
          假设有 10 轮历史，每轮 200 token，总计 2000 token
          压缩前 5 轮（1000 token → 摘要约 150 token）
          保留后 5 轮原文（1000 token）
          总计: 150 + 1000 = 1150 token，压缩了约 40%
        """
        if len(self.turns) < 4:
            return ""  # 轮次太少，不需要压缩

        # 只压缩前一半轮次
        turns_to_compress = self.turns[:len(self.turns)//2]
        history_text = "\n".join([
            f"用户：{t.question}\n助手：{t.answer}"
            for t in turns_to_compress
        ])

        prompt = f"""请将以下对话历史压缩成一段简洁的摘要（100字以内），
保留关键信息（用户问了什么主题、获得了什么答案、有什么重要约定）：

{history_text}

摘要："""

        summary = llm_client.complete(prompt, max_tokens=200)
        self.compressed_summary = summary

        # 删除已压缩的轮次
        self.turns = self.turns[len(turns_to_compress):]
        print(f"  [History] 压缩了 {len(turns_to_compress)} 轮历史，摘要长度: {len(summary)}字")
        return summary

    @property
    def total_tokens_spent(self) -> int:
        """这个会话总共花费的 token 数（成本追踪）"""
        return sum(t.token_count for t in self.turns)

    def to_dict(self) -> dict:
        """序列化（存到 Redis / 数据库）"""
        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "question": t.question,
                    "answer": t.answer,
                    "sources": t.sources,
                    "timestamp": t.timestamp,
                    "from_cache": t.from_cache,
                }
                for t in self.turns
            ],
            "compressed_summary": self.compressed_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationHistory":
        obj = cls(session_id=data["session_id"])
        obj.compressed_summary = data.get("compressed_summary", "")
        for t in data.get("turns", []):
            obj.turns.append(Turn(
                question=t["question"],
                answer=t["answer"],
                sources=t["sources"],
                timestamp=t["timestamp"],
                from_cache=t.get("from_cache", False),
            ))
        return obj


# ─────────────────────────────────────────
# 会话管理器
# ─────────────────────────────────────────

class SessionManager:
    """
    多用户场景下的会话管理。

    生产架构：
      会话数据存 Redis（TTL 自动过期），不放内存。
      原因：
        1. 服务重启后会话不丢失
        2. 多实例共享同一个用户的会话
        3. Redis TTL 自动清理过期会话，不需要手动 GC

    这里用内存字典演示，结构和 Redis 实现完全对应，
    替换时只需要把 dict 操作换成 redis.get/set/setex。
    """

    def __init__(self, session_ttl_seconds: int = 1800):  # 30分钟无操作自动过期
        self._sessions: dict[str, ConversationHistory] = {}
        self._last_active: dict[str, float] = {}
        self.ttl = session_ttl_seconds

    def get_or_create(self, session_id: str) -> ConversationHistory:
        """获取或创建会话"""
        self._cleanup_expired()

        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationHistory(session_id)
            print(f"  [Session] 新建会话: {session_id}")
        else:
            print(f"  [Session] 恢复会话: {session_id} ({len(self._sessions[session_id].turns)} 轮历史)")

        self._last_active[session_id] = time.time()
        return self._sessions[session_id]

    def save(self, history: ConversationHistory):
        """保存会话（写回存储）"""
        self._sessions[history.session_id] = history
        self._last_active[history.session_id] = time.time()

    def _cleanup_expired(self):
        """清理过期会话（生产环境由 Redis TTL 自动处理）"""
        now = time.time()
        expired = [
            sid for sid, last in self._last_active.items()
            if now - last > self.ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            del self._last_active[sid]
            print(f"  [Session] 过期清理: {sid}")


# ─────────────────────────────────────────
# 演示运行
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    console = Console()

    console.print("\n[bold cyan]═══ 多轮对话演示 ═══[/bold cyan]\n")

    # 模拟一个完整的多轮对话会话
    session_mgr = SessionManager()
    history = session_mgr.get_or_create("user_001")

    mock_dialogue = [
        ("年假有几天？",
         "根据员工手册，工龄满1年不满10年享有5天年假，满10年不满20年享有10天，满20年及以上享有15天。[来源: hr_policy.txt]",
         ["hr_policy.txt"]),
        ("那它需要提前多久申请？",
         "年假申请须提前3个工作日通过OA系统提交，经直属主管审批后生效。[来源: hr_policy.txt]",
         ["hr_policy.txt"]),
        ("如果当年没用完怎么办？",
         "当年未使用的年假可顺延至次年3月31日前使用，逾期作废，不予折现补偿。[来源: hr_policy.txt]",
         ["hr_policy.txt"]),
        ("密码多久要改一次？",    # 话题切换
         "系统账号密码有效期为90天，到期须强制修改，且不得与近12次密码重复。[来源: security_policy.txt]",
         ["security_policy.txt"]),
        ("那密码要满足什么要求？",
         "密码长度不少于12位，须包含大写字母、小写字母、数字、特殊字符四类中至少三类。[来源: security_policy.txt]",
         ["security_policy.txt"]),
    ]

    for i, (q, a, sources) in enumerate(mock_dialogue):
        turn = Turn(question=q, answer=a, sources=sources, token_count=len(q+a)//3)
        history.add_turn(turn)
        console.print(f"  [轮{i+1}] 用户: {q}")
        console.print(f"         助手: {a[:60]}...")

    # 展示历史对话结构
    console.print("\n[bold]历史对话的 Messages 结构（送入 LLM 的格式）：[/bold]")
    messages = history.get_messages_for_prompt()

    table = Table(show_lines=True)
    table.add_column("role", width=10)
    table.add_column("content（前60字）", width=60)
    for msg in messages:
        table.add_row(msg["role"], msg["content"][:60])
    console.print(table)

    # 演示指代消解需要的历史
    console.print("\n[bold]指代消解示例：[/bold]")
    recent = history.get_recent_turns(3)
    console.print(f"用户问：'那它需要提前多久申请？'")
    console.print(f"最近 {len(recent)} 轮历史：")
    for t in recent:
        console.print(f"  Q: {t.question}")
        console.print(f"  A: {t.answer[:50]}...")
    console.print("→ 结合历史，'它' = '年假' → 消解后：'年假需要提前多久申请？'")

    # Token 统计
    console.print(f"\n[bold]会话统计：[/bold]")
    console.print(f"  对话轮数: {len(history.turns)}")
    console.print(f"  累计 token（估算）: {history.total_tokens_spent}")
    hist_text = " ".join(t.question + t.answer for t in history.turns)
    console.print(f"  历史占用 token: ~{TokenCounter().count(hist_text)}")

    session_mgr.save(history)
    console.print("\n[green]✓ 多轮对话演示完成，下一步：pipeline.py[/green]")
