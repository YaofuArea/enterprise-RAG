"""
raptor.py — RAPTOR 层级召回

论文：RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
      Stanford 2024 | https://arxiv.org/abs/2401.18059

学习重点：
  1. 为什么需要层级索引（宏观问题 vs 细节问题）
  2. GMM 聚类 vs K-Means：为什么 RAPTOR 选 GMM
  3. 递归摘要构建树形结构
  4. 检索时如何利用树（collapsed tree 策略）
  5. 递归深度如何控制（防止过度压缩）
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from ingestion import Chunk


# ─────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────

@dataclass
class TreeNode:
    """
    RAPTOR 树的节点。
    叶子节点 = 原始 chunk
    中间节点 = 一组 chunk 的摘要
    根节点   = 整个文档集合的摘要
    """
    text: str                            # 节点文本（原始 chunk 或摘要）
    level: int                           # 0 = 叶子，1 = 一级摘要，2 = 二级摘要...
    embedding: Optional[np.ndarray] = None
    children: list["TreeNode"] = field(default_factory=list)  # 子节点（叶子层为空）
    metadata: dict = field(default_factory=dict)
    node_id: str = ""

    def is_leaf(self) -> bool:
        return self.level == 0


@dataclass
class RaptorTree:
    """整棵 RAPTOR 树"""
    all_nodes: list[TreeNode] = field(default_factory=list)   # 所有层的节点（用于检索）
    root_nodes: list[TreeNode] = field(default_factory=list)  # 最顶层节点
    max_level: int = 0


# ─────────────────────────────────────────
# 核心：RAPTOR 构建器
# ─────────────────────────────────────────

class RaptorBuilder:
    """
    RAPTOR 树构建器。

    构建流程：
      Level 0（叶子）：原始 chunk
          ↓ GMM 聚类 + LLM 摘要
      Level 1：cluster 摘要
          ↓ 再次 GMM 聚类 + LLM 摘要
      Level 2：摘要的摘要
          ↓ 递归，直到节点数 < 阈值
      Root：整个语料的顶层摘要

    检索策略（Collapsed Tree）：
      不是从根节点逐层向下，而是把所有层的节点
      都放进同一个向量库，统一检索。
      宏观问题 → 高层摘要节点得分高
      细节问题 → 叶子节点得分高
      一次检索，自动命中合适的粒度。
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        llm_summarizer,                        # 调用 LLM 生成摘要的函数
        max_levels: int = 3,                   # 最大递归层数（防止无限递归）
        min_nodes_to_cluster: int = 4,         # 节点数少于此值时停止递归
        reduction_dim: int = 10,               # UMAP 降维目标维度（用于聚类前降维）
        max_clusters: int = 10,                # GMM 最大聚类数
    ):
        self.model = embedding_model
        self.summarizer = llm_summarizer
        self.max_levels = max_levels
        self.min_nodes = min_nodes_to_cluster
        self.reduction_dim = reduction_dim
        self.max_clusters = max_clusters

    def build(self, chunks: list[Chunk]) -> RaptorTree:
        """从原始 chunk 构建 RAPTOR 树"""
        tree = RaptorTree()

        print(f"\n[RAPTOR] 开始构建树，原始 chunk 数: {len(chunks)}")

        # Level 0：把原始 chunk 转成叶子节点
        leaf_nodes = []
        for i, chunk in enumerate(chunks):
            node = TreeNode(
                text=chunk.content,
                level=0,
                metadata=chunk.metadata,
                node_id=f"L0_{i}",
            )
            leaf_nodes.append(node)

        # 计算叶子节点的 embedding
        self._embed_nodes(leaf_nodes)
        tree.all_nodes.extend(leaf_nodes)

        # 递归向上构建
        current_level_nodes = leaf_nodes
        for level in range(1, self.max_levels + 1):
            if len(current_level_nodes) < self.min_nodes:
                print(f"[RAPTOR] Level {level}: 节点数 {len(current_level_nodes)} < "
                      f"阈值 {self.min_nodes}，停止递归")
                break

            print(f"\n[RAPTOR] 构建 Level {level}...")
            next_level_nodes = self._build_level(current_level_nodes, level)

            if not next_level_nodes:
                break

            tree.all_nodes.extend(next_level_nodes)
            tree.max_level = level
            current_level_nodes = next_level_nodes

        tree.root_nodes = current_level_nodes
        print(f"\n[RAPTOR] 树构建完成: {len(tree.all_nodes)} 个节点，"
              f"共 {tree.max_level + 1} 层")
        return tree

    def _build_level(
        self,
        nodes: list[TreeNode],
        level: int
    ) -> list[TreeNode]:
        """
        构建一层：聚类 → 为每个 cluster 生成摘要 → 生成新节点
        """
        # 1. 提取 embedding 矩阵
        embeddings = np.array([n.embedding for n in nodes])

        # 2. 确定聚类数（GMM 用 BIC 自动选最优 k）
        n_clusters = self._select_n_clusters(embeddings)
        print(f"  聚类数: {n_clusters}（BIC 自动选择）")

        # 3. GMM 聚类
        labels = self._gmm_cluster(embeddings, n_clusters)

        # 4. 按 cluster 分组，为每组生成摘要节点
        new_nodes = []
        for cluster_id in range(n_clusters):
            cluster_nodes = [n for n, l in zip(nodes, labels) if l == cluster_id]

            if not cluster_nodes:
                continue

            # 拼接同一个 cluster 的所有文本
            combined_text = "\n\n".join(n.text for n in cluster_nodes)

            # 调用 LLM 生成摘要
            summary = self.summarizer(combined_text)

            # 创建摘要节点
            summary_node = TreeNode(
                text=summary,
                level=level,
                children=cluster_nodes,
                node_id=f"L{level}_{cluster_id}",
                metadata={
                    "level": level,
                    "cluster_id": cluster_id,
                    "source_count": len(cluster_nodes),
                    # 聚合子节点的来源信息（用于引用溯源）
                    "sources": list({
                        n.metadata.get("source", "")
                        for n in cluster_nodes
                        if n.metadata.get("source")
                    }),
                }
            )
            new_nodes.append(summary_node)
            print(f"  Cluster {cluster_id}: {len(cluster_nodes)} 个节点 → 摘要({len(summary)}字)")

        # 5. 计算新节点的 embedding
        self._embed_nodes(new_nodes)
        return new_nodes

    def _select_n_clusters(self, embeddings: np.ndarray) -> int:
        """
        用 BIC（贝叶斯信息准则）自动选择最优聚类数。

        为什么用 GMM 而不是 K-Means：

        K-Means 的问题：
          1. 硬聚类：每个点只属于一个 cluster（0或1）
          2. 假设 cluster 是球形的，语义空间的 cluster 形状各异
          3. 需要预先指定 k

        GMM（高斯混合模型）的优势：
          1. 软聚类：每个点对每个 cluster 都有一个概率（0~1）
             → 一个 chunk 可以"部分属于"多个 cluster，边界更自然
          2. 允许 cluster 是椭圆形（协方差矩阵可以是任意形状）
          3. 用 BIC 自动选 k，不需要人工指定

        BIC 原理：
          BIC = k * ln(n) - 2 * ln(L̂)
          L̂ = 模型的最大似然估计
          k = 参数数量（惩罚项，防止 k 过大）
          BIC 越小越好（在拟合质量和模型复杂度之间平衡）

        UMAP 降维的原因：
          高维空间（512维）中，距离度量会失效（维度灾难）
          GMM 在高维下性能很差，需要先降维到 10~20 维
          这里用 PCA 代替 UMAP（UMAP 效果更好但需要额外安装）
        """
        # 先 PCA 降维（UMAP 效果更好，但 PCA 不需要额外安装）
        reduced = self._reduce_dimensions(embeddings)

        max_k = min(self.max_clusters, len(embeddings) - 1)
        if max_k < 2:
            return 1

        best_bic = float("inf")
        best_k = 2

        for k in range(2, max_k + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",  # full = 允许任意椭圆形 cluster
                    random_state=42,
                    max_iter=100,
                )
                gmm.fit(reduced)
                bic = gmm.bic(reduced)
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception:
                break

        return best_k

    def _gmm_cluster(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        GMM 聚类，返回每个节点的 cluster 标签。

        covariance_type 选项：
          "full"      → 每个 cluster 有独立的协方差矩阵，最灵活，计算最慢
          "tied"      → 所有 cluster 共享一个协方差矩阵，快，但假设形状相同
          "diag"      → 对角协方差，介于 full 和 tied 之间
          "spherical" → 球形，等价于 K-Means，最快最差

        RAPTOR 论文用 "full"，精度最高。
        """
        reduced = self._reduce_dimensions(embeddings)

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42,
        )
        gmm.fit(reduced)

        # predict 返回硬标签（概率最高的 cluster）
        # predict_proba 返回软标签（每个 cluster 的概率）
        # RAPTOR 原论文用软标签（一个节点可属于多个 cluster），这里简化为硬标签
        labels = gmm.predict(reduced)
        return labels

    def _reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
        """
        用 PCA 降维，解决高维空间的维度灾难问题。
        生产环境推荐用 UMAP（非线性降维，保留局部结构更好）：
          import umap
          reducer = umap.UMAP(n_components=10, metric='cosine')
          return reducer.fit_transform(embeddings)
        """
        from sklearn.decomposition import PCA
        n_comp = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
        if n_comp < 2:
            return embeddings
        pca = PCA(n_components=n_comp, random_state=42)
        return pca.fit_transform(embeddings)

    def _embed_nodes(self, nodes: list[TreeNode]):
        """批量计算节点 embedding"""
        texts = [n.text for n in nodes]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for node, emb in zip(nodes, embeddings):
            node.embedding = emb.astype(np.float32)


# ─────────────────────────────────────────
# RAPTOR 检索器
# ─────────────────────────────────────────

class RaptorRetriever:
    """
    基于 RAPTOR 树的检索器。

    检索策略：Collapsed Tree（论文推荐，效果最好）

    把所有层的节点都平铺到一个向量索引里，统一检索。
    不区分层级，让相似度得分自然决定用哪层的节点。

    优点：
      - 宏观问题（"这份文档讲了什么"）→ 高层摘要节点得分高
      - 细节问题（"第三条款的违约金"）→ 叶子节点得分高
      - 一个索引，自适应，不需要手动选层

    替代策略（Tree Traversal，论文也提到）：
      从根节点开始，逐层向下检索，直到叶子。
      优点：解释性强，可以追溯决策路径。
      缺点：层层依赖，上层错误会传播；速度慢。
    """

    def __init__(self, tree: RaptorTree, embedding_model: SentenceTransformer):
        self.tree = tree
        self.model = embedding_model
        self._build_flat_index()

    def _build_flat_index(self):
        """把所有层节点的 embedding 放入一个 FAISS 索引（Collapsed Tree）"""
        import faiss

        nodes = [n for n in self.tree.all_nodes if n.embedding is not None]
        if not nodes:
            self.flat_index = None
            self.indexed_nodes = []
            return

        dim = nodes[0].embedding.shape[0]
        self.flat_index = faiss.IndexFlatIP(dim)
        self.indexed_nodes = nodes

        embeddings = np.array([n.embedding for n in nodes]).astype(np.float32)
        self.flat_index.add(embeddings)
        print(f"[RaptorRetriever] Collapsed Tree 索引: {len(nodes)} 个节点（所有层）")

    def search(self, query: str, top_k: int = 5) -> list[tuple[TreeNode, float]]:
        """
        检索，返回 (节点, 相似度) 列表。
        结果会包含不同层级的节点，由相似度自然排序。
        """
        if self.flat_index is None:
            return []

        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.flat_index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            node = self.indexed_nodes[idx]
            results.append((node, float(score)))

        return results

    def search_with_level_info(self, query: str, top_k: int = 5) -> list[dict]:
        """
        检索并附带层级信息，便于调试和观察 RAPTOR 的行为。
        """
        raw = self.search(query, top_k)
        results = []
        for node, score in raw:
            results.append({
                "text": node.text,
                "level": node.level,
                "score": score,
                "is_leaf": node.is_leaf(),
                "sources": node.metadata.get("sources", [node.metadata.get("source", "")]),
                "node_id": node.node_id,
            })
        return results

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/raptor_tree.pkl", "wb") as f:
            pickle.dump(self.tree, f)
        print(f"[RAPTOR] 已保存到 {path}/")

    @classmethod
    def load(cls, path: str, embedding_model: SentenceTransformer) -> "RaptorRetriever":
        with open(f"{path}/raptor_tree.pkl", "rb") as f:
            tree = pickle.load(f)
        obj = cls(tree, embedding_model)
        print(f"[RAPTOR] 从 {path}/ 加载，共 {len(tree.all_nodes)} 个节点")
        return obj


# ─────────────────────────────────────────
# LLM 摘要函数（依赖 Anthropic）
# ─────────────────────────────────────────

def make_summarizer(api_key: str = None):
    """
    创建用于 RAPTOR 的摘要函数。
    摘要质量直接影响树的检索效果，prompt 很重要：
      1. 要求保留关键事实和数字（不能过度抽象）
      2. 要求输出简洁（摘要层的 chunk 不应比原文长）
      3. 限制输出长度（防止模型啰嗦）
    """
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)

    def summarize(text: str) -> str:
        """
        对一组 chunk 生成摘要。
        这是 RAPTOR 构建过程中最耗时的步骤（每个 cluster 都要调一次 LLM）。
        成本估算：假设 10 个 cluster，每次输入 1000 token，
        Claude Haiku 大约 $0.001 / 次，整棵树构建成本极低。
        """
        prompt = f"""请对以下企业文档内容生成一段简洁的中文摘要。

要求：
- 保留所有关键信息、数字、规则和流程步骤
- 不要遗漏重要的限制条件（如金额阈值、时间限制）
- 输出100-200字，不要超过200字
- 直接输出摘要内容，不要加"摘要："等前缀

文档内容：
{text[:3000]}"""  # 限制输入长度避免超过 context window

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # 摘要任务用 Haiku，便宜快速
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    return summarize


def make_mock_summarizer():
    """
    Mock 摘要函数，不调用 LLM，用于测试和演示。
    直接返回文本的前 150 字作为"摘要"。
    """
    def summarize(text: str) -> str:
        # 简单截取前 150 字作为 mock 摘要
        clean = text.replace("\n", " ").strip()
        return f"[摘要] {clean[:150]}..."
    return summarize


# ─────────────────────────────────────────
# 演示运行
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from sentence_transformers import SentenceTransformer

    from ingestion import IngestionPipeline

    console = Console()
    console.print("\n[bold cyan]═══ RAPTOR 层级召回演示 ═══[/bold cyan]\n")

    # 1. 摄入文档
    pipeline = IngestionPipeline(chunk_size=400, chunk_overlap=40)
    chunks = pipeline.ingest_directory("./data")

    # 2. 加载 embedding 模型
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # 3. 构建 RAPTOR 树（用 Mock 摘要，不需要 API key）
    # 设置 USE_REAL_LLM=1 且 ANTHROPIC_API_KEY 有效时才用真实 LLM
    use_real_llm = os.environ.get("USE_REAL_LLM") == "1"
    summarizer = make_summarizer() if use_real_llm else make_mock_summarizer()
    console.print(f"[dim]摘要模式: {'真实 LLM (Claude)' if use_real_llm else 'Mock（截取前150字）'}[/dim]")

    builder = RaptorBuilder(
        embedding_model=model,
        llm_summarizer=summarizer,
        max_levels=2,
        min_nodes_to_cluster=4,
    )
    tree = builder.build(chunks)

    # 打印树结构统计
    level_counts = {}
    for node in tree.all_nodes:
        level_counts[node.level] = level_counts.get(node.level, 0) + 1

    table = Table(title="RAPTOR 树结构")
    table.add_column("层级", style="cyan")
    table.add_column("节点数")
    table.add_column("说明")
    for level, count in sorted(level_counts.items()):
        desc = "原始 chunk（叶子）" if level == 0 else f"Level {level} 摘要节点"
        table.add_row(f"Level {level}", str(count), desc)
    console.print(table)

    # 4. 构建检索器
    retriever = RaptorRetriever(tree, model)

    # 5. 演示：对比宏观问题 vs 细节问题的命中层级
    console.print("\n[bold]检索演示：观察不同问题命中的层级[/bold]")

    queries = [
        ("宏观问题", "这些文档都涉及哪些主题？"),
        ("细节问题", "年假申请需要提前几天？"),
        ("细节问题", "密码需要包含哪些字符类型？"),
    ]

    for q_type, query in queries:
        results = retriever.search_with_level_info(query, top_k=3)
        console.print(f"\n[yellow][{q_type}] {query}[/yellow]")

        t = Table(show_lines=True)
        t.add_column("排名", width=4)
        t.add_column("层级", width=6)
        t.add_column("得分", width=6)
        t.add_column("内容（前80字）", width=60)

        for i, r in enumerate(results):
            level_label = f"L{r['level']}({'叶子' if r['is_leaf'] else '摘要'})"
            t.add_row(
                str(i+1),
                level_label,
                f"{r['score']:.3f}",
                r["text"][:80].replace("\n", " "),
            )
        console.print(t)

    # 6. 保存
    retriever.save("./index/raptor")
    console.print("\n[green]✓ RAPTOR 构建完成，下一步：query_enhancement.py[/green]")
