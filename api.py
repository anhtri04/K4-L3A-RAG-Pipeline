"""FastAPI UI cho RAG Pipeline (thay thế/bổ sung Streamlit app.py).

Chạy:
    uvicorn api:app --reload --port 8000
Mở:
    http://localhost:8000/        -> Chat UI
    http://localhost:8000/docs    -> Swagger (document API tự sinh)
    http://localhost:8000/health  -> kiểm tra RAG index + LLM config

Lib dùng:
    - fastapi + uvicorn cho UI/API (document ở /docs, /redoc).
    - python-sdk `openai` trỏ về Opencode Zen để gọi model
      muse-spark-1.3-contributor (xem src/task10_generation.py).
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()

from src.task10_generation import (
    OPENCODE_DEFAULT_BASE_URL,
    OPENCODE_DEFAULT_MODEL,
    _normalize_base_url,
    generate_with_citation,
)
from src.task9_retrieval_pipeline import retrieve

app = FastAPI(
    title="K4 RAG Pipeline — Football Regulations",
    description=(
        "Chatbot RAG luật bóng đá (IFAB/FIFA/UEFA/AFC). "
        "Dùng python-sdk OpenAI-compatible gọi Opencode Zen "
        f"`{OPENCODE_DEFAULT_MODEL}`. Thử API ở /docs."
    ),
    version="0.1.0",
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="Câu hỏi của user")
    top_k: int = Field(default=5, ge=1, le=10, description="Số chunks retrieve")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><html lang="vi"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>K4 RAG — Football Regulations</title>
<style>body{font-family:system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px}
#log{border:1px solid #ddd;border-radius:12px;padding:12px;min-height:300px;margin:12px 0}
.msg{margin:10px 0}.u{color:#0b5}.a{white-space:pre-wrap}
.src{font-size:12px;color:#555;background:#f6f6f6;border-radius:8px;padding:8px;margin-top:6px}
.row{display:flex;gap:8px}input{flex:1;padding:10px;border-radius:8px;border:1px solid #ccc}
button{padding:10px 16px;border-radius:8px;border:0;background:#111;color:#fff;cursor:pointer}</style>
</head><body>
<h2>K4 RAG — Football Regulations</h2>
<p>Model: <code id="model"></code> · <a href="/docs">API docs</a></p>
<div id="log"></div>
<div class="row"><input id="q" placeholder="Hỏi luật bóng đá... (vd: luật việt vị IFAB 2026/27)"/>
<button onclick="send()">Gửi</button></div>
<script>
async function cfg(){const r=await fetch('/api/config');const j=await r.json();
document.getElementById('model').textContent=j.llm_model+' @ '+j.base_url;}
async function send(){const q=document.getElementById('q');const query=q.value.trim();if(!query)return;
q.value='';const log=document.getElementById('log');
log.innerHTML+=`<div class="msg"><b class="u">Bạn:</b> ${query}</div><div class="msg">⏳ ...</div>`;
log.scrollTop=log.scrollHeight;
const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({query,top_k:5})});
const j=await r.json();
log.lastChild.outerHTML=`<div class="msg"><b>Bot:</b> <span class="a">${j.answer}</span>
<div class="src">retrieval: ${j.retrieval_source} · sources: ${j.sources.map(s=>s.id+' ('+s.score.toFixed(3)+')').join(', ')||'none'}</div></div>`;}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
cfg();</script></body></html>"""


@app.get("/health")
def health():
    try:
        docs = retrieve("test", top_k=1)
        index_ok = True
    except Exception as exc:  # Chroma/embedding chưa index vẫn báo rõ
        docs = []
        index_ok = False
        index_error = str(exc)
    else:
        index_error = ""
    return {
        "status": "ok",
        "llm_provider": os.getenv("LLM_PROVIDER", "opencode"),
        "llm_model": os.getenv("LLM_MODEL", "") or OPENCODE_DEFAULT_MODEL,
        "base_url": _normalize_base_url(
            os.getenv("OPENCODE_BASE_URL", "") or OPENCODE_DEFAULT_BASE_URL
        ),
        "has_key": bool(os.getenv("OPENCODE_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")),
        "index_ok": index_ok,
        "index_error": index_error,
        "sample_hits": len(docs),
    }


@app.get("/api/config")
def api_config():
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "opencode"),
        "llm_model": os.getenv("LLM_MODEL", "") or OPENCODE_DEFAULT_MODEL,
        "base_url": _normalize_base_url(
            os.getenv("OPENCODE_BASE_URL", "") or OPENCODE_DEFAULT_BASE_URL
        ),
        "responses_endpoint": _normalize_base_url(
            os.getenv("OPENCODE_BASE_URL", "") or OPENCODE_DEFAULT_BASE_URL
        )
        + "/responses",
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"),
    }


@app.post("/api/search")
def api_search(req: SearchRequest):
    try:
        return {"query": req.query, "results": retrieve(req.query, top_k=req.top_k)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    try:
        return generate_with_citation(req.query, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
