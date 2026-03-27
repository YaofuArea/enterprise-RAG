"""
02_indexing.py — 向量索引 + 稀疏索引

学习重点：
  1. Embedding 模型选型：维度、多语言、速度的取舍
  2. FAISS 索引类型：Flat / IVF / HNSW 的速度-精度权衡
  3. BM25 稀疏索引：为什么向量检索不够用
  4. 两种索引如何并存，为下一步混合检索做准备
  5. 索引持久化：序列化到磁盘，服务重启不用重建
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 修复 Mac 上 FAISS + PyTorch 的 OpenMP 冲突
import pickle
import json
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingestion import Chunk, IngestionPipeline


# ─────────────────────────────────────────
# Embedding 模型
# ─────────────────────────────────────────

def load_embedding_model(model_name: str = "BAAI/bge-small-zh-v1.5") -> SentenceTransformer:
    """
    模型选型说明：

    中文推荐：
      BAAI/bge-small-zh-v1.5  → 512维，轻量快速，中文效果好，首选
      BAAI/bge-large-zh-v1.5  → 1024维，精度更高，但慢 3x，显存占用大
      shibing624/text2vec-base-chinese → 768维，老牌模型，兼容性好

    多语言（中英混合文档）：
      BAAI/bge-m3             → 支持100+语言，1024维，效果最好但最慢
      intfloat/multilingual-e5-base → 768维，速度和效果平衡

    维度的影响：
      维度越高 → 表达能力越强，但 FAISS 存储和检索更慢
      512维 vs 1024维：检索速度差约 2x，精度差约 5%
      企业内部知识库用 512 维够了，不需要上 1024

    normalize_embeddings=True 的原因：
      归一化后向量长度为 1，余弦相似度 = 内积（点积）
      FAISS 的 IndexFlatIP（内积索引）就利用了这一点，
      比 L2 距离在语义相似度任务上效果更好
    """
    print(f"[Embedding] 加载模型: {model_name}（首次运行会自动下载）")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"[Embedding] 向量维度: {dim}")
    return model


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[Chunk],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    批量计算 chunk 的 embedding。

    batch_size 的选择：
      太小 → GPU/CPU 利用率低，速度慢
      太大 → 显存溢出（GPU）或内存压力（CPU）
      CPU 推理：batch_size=32~64 比较合适
      GPU 推理：batch_size=128~256

    为什么 normalize_embeddings=True：
      后续 FAISS 用内积（IndexFlatIP）做相似度，
      归一化后内积 == 余弦相似度，值域 [-1, 1]，1 表示完全相同
    """
    texts = [chunk.content for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # 必须归一化，否则 IndexFlatIP 结果不是余弦相似度
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)  # FAISS 只接受 float32


# ─────────────────────────────────────────
# FAISS 向量索引
# ─────────────────────────────────────────

class VectorIndex:
    """
    FAISS 向量索引封装。

    FAISS 索引类型选型（核心知识点）：

    ┌─────────────────┬──────────┬──────────┬────────────────────────────┐
    │ 索引类型         │ 精度     │ 速度     │ 适用场景                    │
    ├─────────────────┼──────────┼──────────┼────────────────────────────┤
    │ IndexFlatIP     │ 100%精确 │ 慢（O(N)）│ < 10万向量，要求精确结果    │
    │ IndexIVFFlat    │ ~95%     │ 快        │ 10万~1000万向量             │
    │ IndexHNSWFlat   │ ~99%     │ 很快      │ 任意规模，内存要求高        │
    │ IndexIVFPQ      │ ~90%     │ 极快      │ 亿级向量，内存有限          │
    └─────────────────┴──────────┴──────────┴────────────────────────────┘

    企业知识库通常 < 10万 chunk，用 IndexFlatIP 就够了（精确检索，不用近似）。
    如果文档量很大（百万级），改用 IndexHNSWFlat（无需训练，精度高，速度快）。

    为什么用 IndexFlatIP 而不是 IndexFlatL2：
      IP = 内积（Inner Product）
      归一化向量的内积 == 余弦相似度
      L2 = 欧氏距离，和语义相似度的相关性弱于余弦相似度

    IndexIDMap 的作用：
      默认 FAISS 用 0,1,2... 作为向量 ID，不支持自定义 ID 和删除。
      用 IndexIDMap 包装后，可以用自定义 int64 ID，也支持按 ID 删除向量。
      这对增量更新（删除旧向量、添加新向量）至关重要。
    """

    def __init__(self, dim: int, index_type: str = "flat"):
        self.dim = dim
        self.index_type = index_type
        self.index = self._build_index(dim, index_type)
        self.chunks: list[Chunk] = []      # 和 FAISS 向量位置一一对应
        self.id_to_pos: dict[int, int] = {}  # 自定义 ID → FAISS 位置映射

    def _build_index(self, dim: int, index_type: str) -> faiss.Index:
        if index_type == "flat":
            # 精确检索，小数据集首选
            base = faiss.IndexFlatIP(dim)
            return faiss.IndexIDMap(base)  # 包装以支持自定义 ID 和删除

        elif index_type == "hnsw":
            # HNSW（Hierarchical Navigable Small World）
            # M=32：每个节点的连接数，越大精度越高但内存越多（经验值16~64）
            # ef_construction=200：建图时的搜索范围，越大建图质量越好但越慢
            index = faiss.IndexHNSWFlat(dim, 32)  # M=32
            index.hnsw.efConstruction = 200
            # efSearch 在检索时设置，控制召回精度
            return index  # HNSW 不支持 IndexIDMap

        elif index_type == "ivf":
            # IVF（Inverted File Index）：把向量空间分成 nlist 个区域，
            # 检索时只搜索最近的几个区域（nprobe），大幅减少计算量
            # nlist = sqrt(N) 是经验公式，N 是向量总数
            # 需要先 train，然后 add
            quantizer = faiss.IndexFlatIP(dim)
            nlist = 100  # 分区数，实际使用时设为 sqrt(数据量)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            return faiss.IndexIDMap(index)

        else:
            raise ValueError(f"不支持的索引类型: {index_type}")

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        """添加向量到索引"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 和 embeddings 数量不一致")

        # IVF 索引需要先训练（建立聚类中心）
        if self.index_type == "ivf":
            inner = faiss.downcast_index(self.index.index)  # 取出被包装的索引
            if not inner.is_trained:
                print(f"[FAISS] 训练 IVF 索引（需要足够的数据量）...")
                inner.train(embeddings)

        # 生成自定义 ID（用当前长度作为起始 ID，保证唯一）
        start_id = len(self.chunks)
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)

        if self.index_type == "hnsw":
            # HNSW 不支持 IndexIDMap，直接 add（ID 就是顺序索引）
            self.index.add(embeddings)
        else:
            self.index.add_with_ids(embeddings, ids)

        self.chunks.extend(chunks)
        print(f"[VectorIndex] 添加 {len(chunks)} 个向量，当前总量: {len(self.chunks)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """
        向量检索。
        返回 (chunk, similarity_score) 列表，按相似度降序排列。
        """
        if len(self.chunks) == 0:
            return []

        # query_embedding shape: (dim,) → 需要 reshape 成 (1, dim)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if self.index_type == "hnsw":
            # HNSW 检索时可以调整 efSearch 影响精度和速度
            self.index.hnsw.efSearch = 50  # 越大越精准但越慢

        actual_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS 找不到时返回 -1
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    def delete(self, chunk_indices: list[int]):
        """
        删除指定 chunk（增量更新时使用）。
        只有 IndexIDMap 支持删除，HNSW 不支持。
        """
        if self.index_type == "hnsw":
            raise NotImplementedError("HNSW 索引不支持删除，需重建整个索引")

        ids = np.array(chunk_indices, dtype=np.int64)
        self.index.remove_ids(ids)
        # 注意：self.chunks 列表中的对应位置也要清除
        # 简单实现：标记为 None，检索时跳过
        for idx in chunk_indices:
            if idx < len(self.chunks):
                self.chunks[idx] = None
        print(f"[VectorIndex] 删除 {len(chunk_indices)} 个向量")

    def save(self, path: str):
        """
        持久化到磁盘。
        FAISS 索引用 faiss.write_index 序列化（二进制格式，高效）。
        chunks 元数据用 pickle 序列化。
        注意：两个文件必须配套保存/加载，版本要一致。
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(f"{path}/meta.json", "w") as f:
            json.dump({"dim": self.dim, "index_type": self.index_type}, f)
        print(f"[VectorIndex] 已保存到 {path}/")

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        with open(f"{path}/meta.json") as f:
            meta = json.load(f)
        obj = cls.__new__(cls)
        obj.dim = meta["dim"]
        obj.index_type = meta["index_type"]
        obj.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/chunks.pkl", "rb") as f:
            obj.chunks = pickle.load(f)
        obj.id_to_pos = {}
        print(f"[VectorIndex] 从 {path}/ 加载，共 {len(obj.chunks)} 个向量")
        return obj


# ─────────────────────────────────────────
# BM25 稀疏索引
# ─────────────────────────────────────────

class BM25Index:
    """
    BM25（Best Match 25）稀疏检索索引。

    为什么有了向量检索还需要 BM25：

    场景：用户问"SKU-20241101 的库存是多少"
    向量检索：把"SKU-20241101"编码成向量，找语义相似的文本 → 很可能找不到
    BM25：精确匹配关键词"SKU-20241101" → 直接命中

    向量检索擅长：语义相似（近义词、改写、模糊查询）
    BM25 擅长：精确匹配（专有名词、产品编号、人名、缩写）

    BM25 公式（Okapi BM25）：
      score(q,d) = Σ IDF(qi) × (tf(qi,d) × (k1+1)) / (tf(qi,d) + k1×(1-b+b×|d|/avgdl))

      关键参数：
        k1=1.5：词频饱和参数，控制词频对得分的影响上限（避免一个词出现100次就得满分）
        b=0.75：文档长度归一化参数，长文档中出现的词权重会被适当降低

    中文的特殊处理：
      BM25 需要把文本拆成词（tokenize）。
      英文按空格拆就行，中文需要分词。
      简化方案：按字符切（每个字作为一个 token）
      更好的方案：用 jieba 分词（需要额外安装）
      这里用按字符切的简化方案，效果对中文也可以接受。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]):
        """构建 BM25 索引"""
        self.chunks = chunks

        # Tokenize：中文按字符切，英文按空格切
        # 生产环境建议用 jieba.cut 分词，召回效果更好
        tokenized = [self._tokenize(chunk.content) for chunk in chunks]

        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        print(f"[BM25Index] 构建完成，共 {len(chunks)} 个文档")

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """
        BM25 检索。
        返回 (chunk, bm25_score) 列表，按分数降序排列。

        注意：BM25 分数是绝对值，不是 [0,1] 之间的归一化分数。
        不同查询的分数不可比较，但同一查询下的排序有意义。
        这也是为什么混合检索用排名（rank）而非分数（score）来融合。
        """
        if self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 获取 top_k 的下标
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 过滤掉完全不相关的（分数为0）
                results.append((self.chunks[idx], float(scores[idx])))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """
        简化版中文分词：按字符切分。
        更好的方案：
          import jieba
          return list(jieba.cut(text))
        jieba 分词的问题：专有名词（如产品型号）可能被切错
        可以向 jieba 用户词典添加专有名词解决这个问题。
        """
        # 过滤标点和空白，保留有意义的字符
        tokens = []
        for char in text:
            if char.strip() and char not in "，。！？；：、""''（）【】《》…—":
                tokens.append(char)
        return tokens

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        print(f"[BM25Index] 已保存到 {path}/")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(f"{path}/bm25.pkl", "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.bm25 = data["bm25"]
        obj.chunks = data["chunks"]
        print(f"[BM25Index] 从 {path}/ 加载，共 {len(obj.chunks)} 个文档")
        return obj


# ─────────────────────────────────────────
# 多粒度索引（Parent Document Retriever）
# ─────────────────────────────────────────

class MultiGranularityIndex:
    """
    多粒度索引：同一份文档，同时建"小 chunk"和"大 chunk"两个粒度的索引。

    核心思想（Parent Document Retriever 模式）：
      检索：用小 chunk（精确定位，语义集中）
      返回：换成对应的大 chunk（上下文完整，信息充分）

    为什么需要这个：

    问题一：chunk 太大 → 一个 chunk 混了多个主题，向量被"平均"，检索精度下降
    问题二：chunk 太小 → 上下文不够，LLM 生成时信息不完整

    解决：两全其美
      用 128~256 字符的小 chunk 做检索（精准定位）
      命中后返回对应的 512~1024 字符大 chunk（完整上下文）

    实现方式：
      小 chunk 的 metadata 里存 parent_id，指向对应的大 chunk
      检索到小 chunk 后，通过 parent_id 查大 chunk
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        child_chunk_size: int = 200,   # 用于检索的小 chunk
        parent_chunk_size: int = 600,  # 用于返回给 LLM 的大 chunk
    ):
        self.model = embedding_model
        self.child_size = child_chunk_size
        self.parent_size = parent_chunk_size

        # 两套索引
        dim = embedding_model.get_sentence_embedding_dimension()
        self.child_index = VectorIndex(dim=dim, index_type="flat")   # 小 chunk 向量索引
        self.parent_chunks: dict[str, Chunk] = {}                    # parent_id → 大 chunk

    def build(self, documents):
        """
        构建双粒度索引。
        注意：parent 和 child 都是从同一份原始文档切出来的，
        child 的粒度更细，parent 的粒度更粗。
        """
        from ingestion import RecursiveTextSplitter, Document

        parent_splitter = RecursiveTextSplitter(
            chunk_size=self.parent_size, chunk_overlap=50
        )
        child_splitter = RecursiveTextSplitter(
            chunk_size=self.child_size, chunk_overlap=20
        )

        all_child_chunks = []

        for doc in documents:
            # 先切大 chunk
            parent_chunks = parent_splitter.split(doc)

            for p_idx, parent in enumerate(parent_chunks):
                # 给每个大 chunk 生成唯一 ID
                parent_id = f"{doc.metadata.get('source', 'doc')}_{p_idx}"
                parent.metadata["parent_id"] = parent_id
                self.parent_chunks[parent_id] = parent

                # 对大 chunk 再切小 chunk
                tmp_doc = Document(
                    content=parent.content,
                    metadata={**parent.metadata, "parent_id": parent_id}
                )
                child_chunks = child_splitter.split(tmp_doc)

                # 每个小 chunk 记住自己的 parent_id
                for child in child_chunks:
                    child.metadata["parent_id"] = parent_id
                    all_child_chunks.append(child)

        # 只对小 chunk 建向量索引
        embeddings = embed_chunks(self.model, all_child_chunks, show_progress=True)
        self.child_index.add(all_child_chunks, embeddings)

        print(f"[MultiGranularityIndex] parent chunk: {len(self.parent_chunks)}, "
              f"child chunk: {len(all_child_chunks)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[Chunk]:
        """
        检索时：
        1. 用 query 在小 chunk 索引里找最相似的
        2. 通过 parent_id 换成对应的大 chunk 返回
        3. 对结果去重（多个小 chunk 可能指向同一个大 chunk）
        """
        # 搜多一点小 chunk，去重后还能保证 top_k 个大 chunk
        child_results = self.child_index.search(query_embedding, top_k * 3)

        seen_parent_ids = set()
        parent_results = []

        for child_chunk, score in child_results:
            parent_id = child_chunk.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent = self.parent_chunks.get(parent_id)
                if parent:
                    parent_results.append(parent)
                    if len(parent_results) >= top_k:
                        break

        return parent_results


# ─────────────────────────────────────────
# 完整索引构建器
# ─────────────────────────────────────────

class IndexBuilder:
    """
    把向量索引 + BM25 索引组合在一起，统一管理。
    为 05_retrieval.py 的混合检索做准备。
    """

    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        dim = embedding_model.get_sentence_embedding_dimension()
        self.vector_index = VectorIndex(dim=dim, index_type="flat")
        self.bm25_index = BM25Index()

    def build(self, chunks: list[Chunk]):
        """同时构建向量索引和 BM25 索引"""
        print(f"\n[IndexBuilder] 开始构建索引，共 {len(chunks)} 个 chunk")

        # 1. 计算 embedding
        print("[IndexBuilder] 计算向量 embedding...")
        embeddings = embed_chunks(self.model, chunks, show_progress=True)

        # 2. 构建向量索引
        self.vector_index.add(chunks, embeddings)

        # 3. 构建 BM25 索引（不需要 embedding，直接用文本）
        self.bm25_index.build(chunks)

        print("[IndexBuilder] 索引构建完成")

    def save(self, path: str = "./index"):
        self.vector_index.save(f"{path}/vector")
        self.bm25_index.save(f"{path}/bm25")
        print(f"[IndexBuilder] 全部索引已保存到 {path}/")

    @classmethod
    def load(cls, embedding_model: SentenceTransformer, path: str = "./index") -> "IndexBuilder":
        obj = cls.__new__(cls)
        obj.model = embedding_model
        obj.vector_index = VectorIndex.load(f"{path}/vector")
        obj.bm25_index = BM25Index.load(f"{path}/bm25")
        return obj


# ─────────────────────────────────────────
# 演示运行
# ─────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    console = Console()

    console.print("\n[bold cyan]═══ 索引构建演示 ═══[/bold cyan]\n")

    # 1. 摄入文档
    pipeline = IngestionPipeline(chunk_size=500, chunk_overlap=50)
    chunks = pipeline.ingest_directory("./data")

    # 2. 加载 embedding 模型
    model = load_embedding_model("BAAI/bge-small-zh-v1.5")

    # 3. 构建索引
    builder = IndexBuilder(model)
    builder.build(chunks)

    # 4. 保存索引
    builder.save("./index")

    # ── 演示：向量检索 vs BM25 检索
    console.print("\n[bold]检索对比演示[/bold]")
    test_queries = [
        "年假有几天",          # 语义查询
        "SKU-2024 库存",       # 关键词查询（BM25 擅长）
        "密码忘记了怎么办",    # 意图查询
    ]

    for query in test_queries:
        console.print(f"\n[yellow]查询：{query}[/yellow]")

        # 向量检索
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        vec_results = builder.vector_index.search(query_emb, top_k=2)

        # BM25 检索
        bm25_results = builder.bm25_index.search(query, top_k=2)

        table = Table(show_lines=True, title=f"查询: {query}")
        table.add_column("方式", style="cyan", width=8)
        table.add_column("得分", width=8)
        table.add_column("内容（前60字）", width=55)
        table.add_column("来源", style="green")

        for chunk, score in vec_results[:2]:
            table.add_row("向量", f"{score:.3f}", chunk.content[:60].replace('\n',' '), chunk.metadata.get("source",""))
        for chunk, score in bm25_results[:2]:
            table.add_row("BM25", f"{score:.2f}", chunk.content[:60].replace('\n',' '), chunk.metadata.get("source",""))

        console.print(table)

    console.print("\n[green]✓ 索引构建完成，下一步：03_raptor.py[/green]")
