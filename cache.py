"""
cache.py — 语义缓存

学习重点：
  1. 进程内语义缓存（单实例，够大多数场景）
  2. 分布式语义缓存设计（多实例的方案）
  3. 阈值的工程取舍：太高 = 没用，太低 = 误命中
  4. TTL 过期：知识库更新后如何让旧缓存失效
  5. FAISS 删除问题：IndexIDMap vs 定期重建
  6. 缓存统计：命中率监控（生产必备）
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import json
import pickle
import numpy as np
import faiss
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────
# 缓存条目
# ─────────────────────────────────────────

@dataclass
class CacheEntry:
    question: str          # 原始问题（用于展示，不用于匹配）
    answer: str            # 缓存的答案
    sources: list[str]     # 来源文件列表
    embedding: np.ndarray  # 问题的 embedding（用于相似度匹配）
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0     # 被命中的次数（用于 LFU 淘汰策略）
    entry_id: int = 0      # 在 FAISS 索引中的 ID


@dataclass
class CacheStats:
    """缓存统计，生产环境必须监控"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def __str__(self):
        return (f"命中率={self.hit_rate:.1%} "
                f"({self.cache_hits}/{self.total_requests}) "
                f"淘汰={self.evictions}")


# ─────────────────────────────────────────
# 进程内语义缓存（单实例）
# ─────────────────────────────────────────

class SemanticCache:
    """
    基于 FAISS 的进程内语义缓存。

    数据流：
      新问题 → embed → FAISS 查最近邻 → 相似度 > 阈值？
        是 → 返回缓存答案（hit）
        否 → 走完整 RAG → 把结果存入缓存（miss + write）

    内存布局：
      self.index    → FAISS IndexIDMap，存向量（支持按 ID 删除）
      self.entries  → dict[int, CacheEntry]，存元数据和答案
      两者通过 entry_id 对应，必须保持同步

    为什么 FAISS 不直接存答案：
      FAISS 只存浮点向量，不存任意数据。
      答案文本、来源等元数据必须在 Python 侧管理。

    阈值选择指南：
      0.95+  → 极严格，几乎只命中完全相同的问题，缓存价值低
      0.90~0.93 → 推荐范围，能命中同义改写，误命中率低
      0.85~0.89 → 宽松，命中率高但可能混淆相关问题
      < 0.85 → 危险，"年假怎么申请"可能命中"病假怎么申请"

      如何在你的数据上标定：
        收集一批历史问题，人工判断哪些应该命中哪些不应该，
        画 PR 曲线，找 F1 最高的阈值点。
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        similarity_threshold: float = 0.92,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        eviction_policy: str = "lfu",   # "lfu"（最少使用）或 "lru"（最近最少使用）
    ):
        self.model = embedding_model
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.eviction_policy = eviction_policy

        dim = embedding_model.get_sentence_embedding_dimension()

        # IndexIDMap 包装 IndexFlatIP，支持：
        #   add_with_ids(vectors, ids) → 用自定义 ID 添加
        #   remove_ids(ids)            → 按 ID 删除（普通 IndexFlatIP 不支持）
        # 这对缓存淘汰（删除旧条目）至关重要
        base_index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base_index)

        self.entries: dict[int, CacheEntry] = {}   # entry_id → CacheEntry
        self._next_id: int = 0                     # 自增 ID 生成器
        self.stats = CacheStats()

    def get(self, question: str) -> Optional[CacheEntry]:
        """
        查询缓存。
        返回 CacheEntry（命中）或 None（未命中）。
        """
        self.stats.total_requests += 1

        if self.index.ntotal == 0:
            self.stats.cache_misses += 1
            return None

        query_vec = self._embed(question)
        # 只查最相似的 1 个
        scores, ids = self.index.search(query_vec, k=1)
        sim = float(scores[0][0])
        entry_id = int(ids[0][0])

        if entry_id == -1:  # FAISS 没找到
            self.stats.cache_misses += 1
            return None

        entry = self.entries.get(entry_id)
        if entry is None:
            self.stats.cache_misses += 1
            return None

        # 检查 TTL 是否过期
        if time.time() - entry.timestamp > self.ttl:
            # 过期了，删除这个条目
            self._delete(entry_id)
            self.stats.cache_misses += 1
            return None

        # 检查相似度阈值
        if sim < self.threshold:
            self.stats.cache_misses += 1
            print(f"  [Cache Miss] 最高相似度 {sim:.4f} < 阈值 {self.threshold}")
            return None

        # 命中
        entry.hit_count += 1
        self.stats.cache_hits += 1
        print(f"  [Cache Hit] 相似度={sim:.4f} | 原问题='{entry.question[:30]}'")
        return entry

    def set(self, question: str, answer: str, sources: list[str]):
        """写入缓存条目"""
        # 容量检查：超过 max_size 先淘汰
        if len(self.entries) >= self.max_size:
            self._evict()

        vec = self._embed(question)
        entry_id = self._next_id
        self._next_id += 1

        entry = CacheEntry(
            question=question,
            answer=answer,
            sources=sources,
            embedding=vec,
            entry_id=entry_id,
        )

        # 同步写入 FAISS 和 Python dict
        self.index.add_with_ids(vec, np.array([entry_id], dtype=np.int64))
        self.entries[entry_id] = entry
        print(f"  [Cache Write] '{question[:30]}' | 当前缓存量: {len(self.entries)}")

    def invalidate_by_source(self, source: str):
        """
        按文档来源使缓存失效。
        知识库更新了某个文档时调用，删除所有引用该文档的缓存条目。
        比 invalidate_all 更精准，不影响其他文档的缓存。
        """
        to_delete = [
            eid for eid, entry in self.entries.items()
            if source in entry.sources
        ]
        for eid in to_delete:
            self._delete(eid)
        print(f"  [Cache Invalidate] 删除 {len(to_delete)} 条引用 '{source}' 的缓存")

    def invalidate_all(self):
        """清空所有缓存（知识库大规模更新时使用）"""
        self.index.reset()
        self.entries.clear()
        print("  [Cache Clear] 已清空所有缓存")

    def _embed(self, text: str) -> np.ndarray:
        """
        embed 并归一化，shape: (1, dim)。
        normalize_embeddings=True 保证内积 == 余弦相似度。
        """
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec.astype(np.float32)

    def _delete(self, entry_id: int):
        """从 FAISS 和 dict 中同步删除一个条目"""
        ids = np.array([entry_id], dtype=np.int64)
        n_removed = self.index.remove_ids(ids)
        if entry_id in self.entries:
            del self.entries[entry_id]
        return n_removed

    def _evict(self):
        """
        淘汰策略。

        LFU（Least Frequently Used）：删除命中次数最少的
          优点：保留热点缓存，冷门缓存被淘汰
          缺点：新写入的条目命中次数为 0，容易被立刻淘汰（频次偏差）
          适用：查询分布稳定，热点 query 明显的场景

        LRU（Least Recently Used）：删除最久没被访问的
          优点：对时间局部性友好，最近用过的大概率近期还会用
          缺点：对周期性访问不友好（上周热点本周不热，但会一直保留）
          适用：查询有明显时间局部性的场景

        这里实现 LFU，按 hit_count 升序排，删最少使用的那批。
        """
        if not self.entries:
            return

        # 按命中次数排序，删除命中最少的 10% 条目
        n_evict = max(1, len(self.entries) // 10)

        if self.eviction_policy == "lfu":
            sorted_ids = sorted(
                self.entries.keys(),
                key=lambda eid: self.entries[eid].hit_count
            )
        else:  # lru：按最后访问时间，这里用 timestamp 近似
            sorted_ids = sorted(
                self.entries.keys(),
                key=lambda eid: self.entries[eid].timestamp
            )

        for eid in sorted_ids[:n_evict]:
            self._delete(eid)
            self.stats.evictions += 1

        print(f"  [Cache Evict] 淘汰 {n_evict} 条（策略: {self.eviction_policy}）")

    def save(self, path: str):
        """持久化：服务重启后缓存不丢失"""
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/cache.index")
        with open(f"{path}/cache_entries.pkl", "wb") as f:
            pickle.dump({
                "entries": self.entries,
                "next_id": self._next_id,
                "stats": self.stats,
            }, f)
        print(f"[Cache] 已持久化 {len(self.entries)} 条缓存到 {path}/")

    @classmethod
    def load(cls, path: str, embedding_model: SentenceTransformer, **kwargs) -> "SemanticCache":
        obj = cls(embedding_model, **kwargs)
        try:
            obj.index = faiss.read_index(f"{path}/cache.index")
            with open(f"{path}/cache_entries.pkl", "rb") as f:
                data = pickle.load(f)
            obj.entries = data["entries"]
            obj._next_id = data["next_id"]
            obj.stats = data.get("stats", CacheStats())
            print(f"[Cache] 从 {path}/ 加载 {len(obj.entries)} 条缓存")
        except FileNotFoundError:
            print(f"[Cache] 未找到持久化文件，从空缓存开始")
        return obj


# ─────────────────────────────────────────
# 分布式语义缓存（多实例方案说明）
# ─────────────────────────────────────────

class DistributedSemanticCacheDesign:
    """
    这个类不运行，只是用来展示分布式缓存的设计思路。

    多实例部署时，进程内 FAISS 缓存无法共享，需要外置存储。

    方案一：Redis Stack（推荐，一站式）
      Redis Stack = Redis + RediSearch + RedisJSON
      内置向量索引，支持 KNN 搜索。

      写入：
        redis.hset(f"cache:{id}", {
            "question": question,
            "answer": answer,
            "sources": json.dumps(sources),
        })
        # 向量存为 float32 bytes
        redis.execute_command(
            "FT.ADD", "cache_idx", id, 1.0,
            "VECTOR", embedding.tobytes()
        )

      查询：
        results = redis.execute_command(
            "FT.SEARCH", "cache_idx",
            f"*=>[KNN 1 @embedding $vec AS score]",
            "PARAMS", 2, "vec", query_emb.tobytes(),
            "RETURN", 3, "question", "answer", "score",
            "SORTBY", "score",
            "DIALECT", 2
        )

    方案二：Qdrant / Weaviate 专用向量数据库
      用 metadata 字段区分缓存条目和普通文档：
        collection.upsert(
            collection_name="knowledge_base",
            points=[PointStruct(
                id=cache_id,
                vector=embedding,
                payload={
                    "type": "cache",           # ← 标记为缓存
                    "answer": answer,
                    "sources": sources,
                    "expires_at": ttl_timestamp,
                }
            )]
        )
      检索时加过滤：type = "cache" AND expires_at > now()

    方案三：妥协方案（FAISS + Redis）
      FAISS：负责向量检索（仍在进程内，不共享）
      Redis：负责存 ID → 答案的映射（共享）
      问题：FAISS 索引仍然不共享，冷启动时需要重建
      适合：缓存更新频率低（每天重建一次索引的场景）
    """
    pass


# ─────────────────────────────────────────
# 演示运行
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    console = Console()

    console.print("\n[bold cyan]═══ 语义缓存演示 ═══[/bold cyan]\n")

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    cache = SemanticCache(
        embedding_model=model,
        similarity_threshold=0.92,
        max_size=100,
        ttl_seconds=300,
    )

    # 模拟一批问答（模拟 RAG 返回的结果）
    mock_qa_pairs = [
        ("年假申请流程是什么",
         "年假申请须提前3个工作日通过OA系统提交，经直属主管审批后生效。当年未使用的年假可顺延至次年3月31日前，逾期作废。",
         ["hr_policy.txt"]),
        ("密码需要满足什么要求",
         "密码长度不少于12位，须包含大写字母、小写字母、数字、特殊字符中的至少三类。密码有效期90天，不得与近12次密码重复。",
         ["security_policy.txt"]),
        ("报销发票抬头填什么",
         "发票抬头须填写公司全称'XX科技有限公司'，税号为91110108XXXXXXXX，不符合规定的发票财务不予报销。",
         ["it_system_manual.txt"]),
    ]

    # 写入缓存
    console.print("[bold]1. 写入缓存[/bold]")
    for question, answer, sources in mock_qa_pairs:
        cache.set(question, answer, sources)

    # 测试命中和未命中
    console.print("\n[bold]2. 缓存命中测试[/bold]")
    test_queries = [
        ("年假怎么申请",       True,  "同义改写，应命中"),
        ("请假需要走什么流程",  True,  "语义相似，应命中"),
        ("密码要几位",          True,  "子话题，应命中"),
        ("病假怎么请",          False, "相关但不同主题，不应命中"),
        ("公司食堂几点开门",    False, "完全无关，不应命中"),
    ]

    table = Table(title="缓存命中测试", show_lines=True)
    table.add_column("查询", width=20)
    table.add_column("预期", width=6)
    table.add_column("实际", width=6)
    table.add_column("命中原问题", width=20)
    table.add_column("相似度判断")

    for query, expected_hit, desc in test_queries:
        result = cache.get(query)
        actual_hit = result is not None
        match = "✓" if actual_hit == expected_hit else "✗"
        table.add_row(
            query,
            "命中" if expected_hit else "未命中",
            ("命中" if actual_hit else "未命中") + f" {match}",
            result.question[:20] if result else "-",
            desc,
        )
    console.print(table)

    # 统计
    console.print(f"\n[bold]3. 缓存统计[/bold]")
    console.print(f"  {cache.stats}")

    # 阈值对比
    console.print("\n[bold]4. 阈值敏感性对比[/bold]")
    console.print("（相同问题 '请假流程' 在不同阈值下的命中情况）")

    query = "请假流程"
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)

    if cache.index.ntotal > 0:
        scores, ids = cache.index.search(q_emb, 3)
        threshold_table = Table(show_lines=True)
        threshold_table.add_column("阈值")
        threshold_table.add_column("最高相似度")
        threshold_table.add_column("结果")
        threshold_table.add_column("说明")

        for thresh in [0.99, 0.95, 0.92, 0.88, 0.80]:
            sim = float(scores[0][0])
            hit = sim >= thresh
            threshold_table.add_row(
                str(thresh),
                f"{sim:.4f}",
                "命中" if hit else "未命中",
                "过严" if thresh >= 0.95 else ("推荐" if 0.90 <= thresh <= 0.93 else "过松"),
            )
        console.print(threshold_table)

    # 持久化
    cache.save("./index/cache")
    console.print("\n[green]✓ 语义缓存演示完成，下一步：generation.py[/green]")
