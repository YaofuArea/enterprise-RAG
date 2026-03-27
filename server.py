"""
server.py — FastAPI 服务层

学习重点：
  1. SSE（Server-Sent Events）流式输出实现
  2. 请求/响应模型设计（Pydantic）
  3. 依赖注入：RAGPipeline 作为单例在请求间共享
  4. 生命周期管理：startup 初始化，不阻塞请求
  5. 错误处理：区分业务错误和系统错误
  6. CORS：前端调用必须配置

运行方式：
  OMP_NUM_THREADS=1 uvicorn server:app --reload --port 8000

接口列表：
  POST /ask           → 非流式问答
  POST /ask/stream    → SSE 流式问答
  POST /ingest        → 上传并摄入新文档
  DELETE /cache       → 清空语义缓存
  GET  /health        → 健康检查
  GET  /stats         → 统计信息
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import time
import asyncio
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pipeline import RAGPipeline, RAGConfig


# ─────────────────────────────────────────
# 请求 / 响应模型
# ─────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(default="default", max_length=64)
    # 可选的运行时参数覆盖（高级用法）
    use_cache: bool = True

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    from_cache: bool
    latency_ms: int
    session_id: str

class IngestResponse(BaseModel):
    chunks_added: int
    filename: str
    message: str

class StatsResponse(BaseModel):
    cache_hit_rate: float
    cache_size: int
    total_requests: int


# ─────────────────────────────────────────
# 应用生命周期
# ─────────────────────────────────────────

# 全局单例（在请求间共享，避免每次请求都重新加载模型）
_pipeline: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理（替代已废弃的 on_event("startup")）。

    为什么在 startup 时初始化 pipeline：
      - 加载 embedding 模型需要 2~5s，不能在请求时加载
      - 服务启动时一次性加载，所有请求共享同一个实例
      - 避免并发请求时的重复加载和内存浪费

    生产架构注意：
      - 如果用多进程（gunicorn -w 4），每个进程都有自己的 pipeline 实例
      - 多进程间不共享内存，但 FAISS 索引可以从共享磁盘加载
      - 缓存如果用进程内 FAISS，多进程间不共享，应改用 Redis
    """
    global _pipeline

    print("[Server] 启动中，初始化 RAG Pipeline...")
    config = RAGConfig(
        use_raptor=True,
        use_cache=True,
        cache_threshold=0.88,
        use_reranker=False,
    )
    _pipeline = RAGPipeline(config)

    # 在后台线程中初始化（避免阻塞事件循环）
    # initialize() 涉及模型加载等 CPU 密集操作，不能在 async 函数里直接调用
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _pipeline.initialize)

    print("[Server] 初始化完成，服务就绪")
    yield  # 服务运行期间

    # Shutdown（清理资源）
    print("[Server] 服务停止")


def get_pipeline() -> RAGPipeline:
    """FastAPI 依赖注入：获取全局 pipeline 实例"""
    if _pipeline is None or not _pipeline._initialized:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")
    return _pipeline


# ─────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────

app = FastAPI(
    title="企业知识库 RAG API",
    description="基于 RAG 的企业内部知识库问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
# 生产环境把 "*" 改成具体的前端域名，防止跨站请求攻击
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产: ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# 接口实现
# ─────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    非流式问答接口。
    等待完整答案生成后一次性返回。
    适合：批量处理、需要完整响应的场景。
    """
    try:
        # run_in_executor 把同步代码放到线程池，不阻塞事件循环
        # RAG 的检索和生成都是同步操作（FAISS、requests 都不是 async）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.ask(request.question, session_id=request.session_id)
        )

        return AskResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            from_cache=result.get("from_cache", False),
            latency_ms=result.get("latency_ms", 0),
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    SSE 流式问答接口。
    逐字返回答案，用户体验更好。

    SSE（Server-Sent Events）格式：
      每个事件格式：data: {json}\n\n
      特殊事件：
        data: {"type": "token", "text": "..."} \n\n  → 文本片段
        data: {"type": "done", "sources": [...]} \n\n → 结束信号
        data: {"type": "error", "message": "..."} \n\n → 错误

    前端 JavaScript 接收方式：
      const es = new EventSource('/ask/stream?question=...');
      es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === 'token') appendText(data.text);
        if (data.type === 'done') es.close();
      };

    注意：SSE 是单向的（服务器→客户端），且只支持 GET（标准规范）。
    这里用 POST 是为了传 JSON body，需要前端用 fetch + ReadableStream 实现，
    或者改用 WebSocket 实现双向通信。
    """

    async def event_generator() -> AsyncIterator[str]:
        try:
            # 在线程池中执行同步的流式生成
            # 无法直接 await 同步生成器，用 queue 桥接
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def producer():
                """在线程中运行，把生成的文本放入 queue"""
                try:
                    for text_chunk in pipeline.ask_stream(
                        request.question,
                        session_id=request.session_id
                    ):
                        # 同步放入 queue（线程安全）
                        loop.call_soon_threadsafe(queue.put_nowait, text_chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, f"ERROR:{e}")
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # 结束信号

            # 启动生产者线程
            loop.run_in_executor(None, producer)

            # 消费 queue，转成 SSE 格式
            sources = []
            while True:
                item = await queue.get()
                if item is None:  # 结束信号
                    break
                if isinstance(item, str) and item.startswith("ERROR:"):
                    yield f"data: {json.dumps({'type': 'error', 'message': item[6:]})}\n\n"
                    break
                # 发送文本片段
                yield f"data: {json.dumps({'type': 'token', 'text': item}, ensure_ascii=False)}\n\n"

            # 发送结束信号
            yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",        # 禁止中间代理缓存 SSE 响应
            "X-Accel-Buffering": "no",          # 禁止 Nginx 缓冲（否则流式效果消失）
            "Connection": "keep-alive",
        }
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    上传并摄入新文档。

    支持格式：.txt, .pdf, .docx, .md
    上传后立即可以检索，不需要重启服务。

    生产考虑：
      - 大文件（>10MB）应异步处理（提交任务 → 返回 task_id → 轮询状态）
      - 需要权限控制（只有管理员可以上传文档）
      - 文件需要安全扫描（防止恶意文件）
    """
    # 检查文件格式
    allowed_ext = {".txt", ".pdf", ".docx", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 {ext}，支持: {allowed_ext}"
        )

    # 把上传的文件保存到临时位置
    tmp_path = f"/tmp/{file.filename}"
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        # 摄入到索引
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pipeline.ingest_file(tmp_path))

        return IngestResponse(
            chunks_added=0,  # 简化：实际应返回真实数量
            filename=file.filename,
            message=f"文档 '{file.filename}' 已成功摄入知识库",
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.delete("/cache")
async def clear_cache(
    source: Optional[str] = None,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    清空语义缓存。
    source 参数：只清除引用特定文档的缓存（精准失效）
    不传 source：清空所有缓存（知识库大规模更新时使用）
    """
    if pipeline.cache is None:
        raise HTTPException(status_code=400, detail="缓存未启用")

    if source:
        pipeline.cache.invalidate_by_source(source)
        return {"message": f"已清除引用 '{source}' 的缓存"}
    else:
        pipeline.cache.invalidate_all()
        return {"message": "已清空所有缓存"}


@app.get("/health")
async def health_check():
    """
    健康检查。负载均衡器用这个判断服务是否存活。
    返回 200 = 正常，返回 503 = 不可用。
    """
    if _pipeline is None or not _pipeline._initialized:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return {"status": "ok", "timestamp": time.time()}


@app.get("/stats", response_model=StatsResponse)
async def get_stats(pipeline: RAGPipeline = Depends(get_pipeline)):
    """统计信息，用于监控面板"""
    stats = pipeline.cache.stats if pipeline.cache else None
    return StatsResponse(
        cache_hit_rate=stats.hit_rate if stats else 0.0,
        cache_size=len(pipeline.cache.entries) if pipeline.cache else 0,
        total_requests=stats.total_requests if stats else 0,
    )


# ─────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("""
╔══════════════════════════════════════════╗
║    企业知识库 RAG 服务                    ║
║    http://localhost:8000                 ║
║    文档: http://localhost:8000/docs      ║
╚══════════════════════════════════════════╝

接口：
  POST /ask          非流式问答
  POST /ask/stream   SSE 流式问答
  POST /ingest       上传文档
  DELETE /cache      清空缓存
  GET  /health       健康检查
  GET  /stats        统计信息
""")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,           # 生产不用 reload（会导致模型重复加载）
        workers=1,              # 单进程（多进程需要解决 FAISS 共享问题）
        log_level="info",
    )
