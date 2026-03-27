# Enterprise RAG — 企业级检索增强生成系统

一个从零实现的企业级 RAG（Retrieval-Augmented Generation）知识库问答系统，代码注释密集，适合学习 RAG 全链路的工程实现。

## 涵盖的技术点

| 模块 | 技术 |
|------|------|
| 文档摄入 | 多格式解析（PDF/Word/TXT）、递归字符切割、语义切割、chunk overlap |
| 向量索引 | FAISS（Flat/IVF/HNSW 对比）、多粒度索引（Parent Document Retriever） |
| 稀疏索引 | BM25 Okapi，中文分词处理 |
| 层级召回 | **RAPTOR**：GMM 聚类 + 递归摘要 + Collapsed Tree 检索 |
| Query 增强 | **HyDE**、**Multi-Query**、**Step-Back Prompting**、指代消解 |
| 混合检索 | Dense + Sparse + **RRF 倒数排名融合** |
| 精排 | **Cross-Encoder Reranker**（Bi-Encoder vs Cross-Encoder 原理对比） |
| 多样性 | **MMR 最大边际相关性** |
| 语义缓存 | FAISS 相似度匹配 + LFU 淘汰 + TTL 过期 + 分布式方案设计 |
| 生成 | **Lost-in-the-Middle** 排列优化、Token 预算管理、流式输出（SSE） |
| 多轮对话 | 滑动窗口 + 摘要压缩、Session 管理 |
| API 服务 | FastAPI + SSE + 依赖注入 + 健康检查 |

## 项目结构

```
enterprise_rag/
├── data/                      # 示例知识库文档（4 份企业文档）
│   ├── hr_policy.txt          # HR 制度（年假、薪酬、考勤）
│   ├── it_system_manual.txt   # ERP 系统操作手册
│   ├── product_faq.txt        # 产品 FAQ
│   └── security_policy.txt    # 信息安全规范
│
├── ingestion.py               # 文档解析 + Chunking 策略
├── indexing.py                # FAISS 向量索引 + BM25 稀疏索引
├── raptor.py                  # RAPTOR 层级召回
├── query_enhancement.py       # HyDE / Multi-Query / Step-Back
├── retrieval.py               # 混合检索 + Reranker + MMR
├── cache.py                   # 语义缓存
├── generation.py              # Prompt 工程 + 流式生成
├── conversation.py            # 多轮对话管理
├── pipeline.py                # 完整流程组装
├── server.py                  # FastAPI 服务
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 按顺序运行各模块（每个都有独立演示）

```bash
export OMP_NUM_THREADS=1   # Mac 必须设置，修复 FAISS + PyTorch OpenMP 冲突

python ingestion.py        # 文档切割演示
python indexing.py         # 索引构建 + 向量/BM25 检索对比
python raptor.py           # RAPTOR 树构建 + 层级检索演示
python query_enhancement.py # Query 增强演示（Mock 模式，不需要 API key）
python retrieval.py        # 混合检索 + RRF + MMR 演示
python cache.py            # 语义缓存 + 阈值敏感性演示
python generation.py       # Prompt 结构 + Lost-in-the-Middle 演示
python conversation.py     # 多轮对话历史管理演示
python pipeline.py         # 完整 RAG 流程（Mock 模式）
```

### 3. 接入真实 LLM（可选）

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export USE_REAL_LLM=1

# 启动 API 服务
OMP_NUM_THREADS=1 uvicorn server:app --port 8000
```

访问 http://localhost:8000/docs 查看交互式 API 文档。

### 4. 调用示例

```bash
# 非流式问答
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "年假需要提前几天申请？", "session_id": "user_001"}'

# 上传新文档
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"

# 清空缓存
curl -X DELETE http://localhost:8000/cache
```

## 关键设计决策说明

### 为什么用 RRF 而不是加权融合

向量分数（余弦，0.5~0.9）和 BM25 分数（无界，0~20）量纲不同，直接加权需要手动调参且不稳定。RRF 用排名代替分数，天然可比，k=60 是经验最优值。

### 为什么 RAPTOR 不从磁盘加载

Python pickle 绑定保存时的模块路径，跨入口加载会出现 `AttributeError`。Mock 摘要重建一次约 2s，成本可接受。生产环境若需持久化，改用 JSON 序列化树结构。

### Mac 上的 OMP_NUM_THREADS=1

FAISS（用 OpenMP）和 PyTorch（也用 OpenMP）在 macOS 上有共享库冲突，会导致 Segfault。设置 `OMP_NUM_THREADS=1` 禁用 FAISS 的多线程即可解决。Linux 生产环境无此问题。

### 语义缓存阈值

`bge-small-zh` 模型对语义相近问题的相似度上限约 0.88~0.92，需要在真实数据上标定。不同 embedding 模型的阈值不能直接复用。

## 依赖说明

| 库 | 用途 |
|----|------|
| `sentence-transformers` | 本地 Embedding 模型，免费，无需 API |
| `faiss-cpu` | 向量索引（Meta 开源） |
| `rank-bm25` | BM25 稀疏检索 |
| `scikit-learn` | GMM 聚类（RAPTOR 使用） |
| `anthropic` | LLM 生成（可替换为 OpenAI） |
| `fastapi` + `uvicorn` | API 服务 |
| `PyMuPDF` | PDF 解析 |
| `tiktoken` | Token 计数 |

## License

MIT
