"""
app_llamacpp.py — FastAPI proxy around llama-server.exe

Architecture:
  llama-server.exe  (llama.cpp built-in HTTP server, OpenAI-compatible API)
       ↑ subprocess
  FastAPI app       (our wrapper — handles CORS, our /chat and /chat/stream endpoints)
       ↑ HTTP
  Lovable UI / client.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import subprocess
import asyncio
import os
import sys
import time
import json
import platform
from typing import List, Optional

# ── Config ────────────────────────────────────────────────────────────────────
if platform.system() == "Windows":
    LLAMA_SERVER_BIN = os.getenv(
        "LLAMA_SERVER_BIN",
        "D:/llm-server/llama-bin/llama-server.exe"
    )
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "D:/llm-server/.cache/huggingface/hub/"
        "models--mradermacher--Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-i1-GGUF/"
        "snapshots/c49b734874219aa0609e0d46f04ef75f8d14fd59/"
        "Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated.i1-Q4_K_M.gguf"
    )
else:
    LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", "/app/llama-bin/llama-server")
    MODEL_PATH       = os.getenv("MODEL_PATH",        "/app/models/model.gguf")

N_GPU_LAYERS  = int(os.getenv("N_GPU_LAYERS",  "30"))   # tune until OOM then back off 2
N_CTX         = int(os.getenv("N_CTX",         "4096"))
N_THREADS     = int(os.getenv("N_THREADS",     str(os.cpu_count() // 2)))
LLAMA_PORT    = int(os.getenv("LLAMA_PORT",    "8080"))  # internal port for llama-server
API_PORT      = int(os.getenv("API_PORT",      "8000"))  # our FastAPI port

LLAMA_BASE    = f"http://127.0.0.1:{LLAMA_PORT}"

print("=" * 60)
print(f"  llama-server:  {LLAMA_SERVER_BIN}")
print(f"  Model:         {MODEL_PATH}")
print(f"  GPU layers:    {N_GPU_LAYERS}")
print(f"  Context:       {N_CTX}")
print(f"  CPU threads:   {N_THREADS}")
print(f"  llama port:    {LLAMA_PORT}")
print("=" * 60)

# ── Launch llama-server.exe as a subprocess ───────────────────────────────────
cmd = [
    LLAMA_SERVER_BIN,
    "--model",       MODEL_PATH,
    "--n-gpu-layers", str(N_GPU_LAYERS),
    "--ctx-size",    str(N_CTX),
    "--threads",     str(N_THREADS),
    "--port",        str(LLAMA_PORT),
    "--host",        "127.0.0.1",
    "--no-mmap",                        # safer on Windows
    "--log-disable",                    # less noise in our console
]

print("Starting llama-server.exe …")
llama_proc = subprocess.Popen(
    cmd,
    stdout=sys.stdout,   # pipe its output to our console
    stderr=sys.stderr,
)

# Wait until llama-server is ready to accept requests
print("Waiting for llama-server to become ready …")
for attempt in range(120):             # wait up to 2 minutes
    time.sleep(1)
    try:
        import urllib.request
        urllib.request.urlopen(f"{LLAMA_BASE}/health", timeout=2)
        print(f"llama-server ready after {attempt + 1}s ✓")
        break
    except Exception:
        pass
else:
    print("ERROR: llama-server did not start in time — check the logs above.")
    llama_proc.kill()
    sys.exit(1)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Local LLM API (llama-server)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared async HTTP client — reused across all requests
http_client = httpx.AsyncClient(base_url=LLAMA_BASE, timeout=300)

# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str       # "user" | "assistant" | "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    response: str
    metrics: dict

# ── Shutdown hook — kill llama-server when FastAPI stops ─────────────────────
@app.on_event("shutdown")
async def shutdown():
    print("Shutting down llama-server …")
    llama_proc.terminate()
    await http_client.aclose()

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        r = await http_client.get("/health")
        llama_health = r.json()
    except Exception as e:
        llama_health = {"error": str(e)}
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
        "llama_server": llama_health,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming: wait for full response, return JSON."""
    init_time = time.time()

    payload = {
        "messages": [m.dict() for m in req.messages],
        "max_tokens": req.max_new_tokens,
        "stream": False,
        "temperature": 0.7,
    }

    try:
        r = await http_client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response_text = data["choices"][0]["message"]["content"]
    total_time = time.time() - init_time
    prompt_tokens    = data.get("usage", {}).get("prompt_tokens", 0)
    completion_tokens = data.get("usage", {}).get("completion_tokens", 0)

    metrics = {
        "first_token_latency_s": None,   # not available in non-streaming mode
        "total_time_s":          round(total_time, 3),
        "prompt_tokens":         prompt_tokens,
        "total_tokens":          completion_tokens,
        "tokens_per_second":     round(completion_tokens / total_time, 2) if total_time > 0 else 0,
    }
    return ChatResponse(response=response_text, metrics=metrics)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming: tokens arrive as Server-Sent Events in real time."""
    payload = {
        "messages": [m.dict() for m in req.messages],
        "max_tokens": req.max_new_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    async def event_generator():
        init_time = time.time()
        first_token_time = None
        token_count = 0

        try:
            async with http_client.stream("POST", "/v1/chat/completions", json=payload) as r:
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

                    token = data["choices"][0].get("delta", {}).get("content", "")
                    if not token:
                        continue

                    if first_token_time is None:
                        first_token_time = time.time()
                    token_count += 1
                    yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'token': f'[Error: {e}]'})}\n\n"

        finally:
            end_time = time.time()
            total = end_time - init_time
            metrics = {
                "first_token_latency_s": round(first_token_time - init_time, 3) if first_token_time else None,
                "total_time_s":          round(total, 3),
                "total_tokens":          token_count,
                "tokens_per_second":     round(token_count / total, 2) if total > 0 else 0,
            }
            yield f"data: {json.dumps({'done': True, 'metrics': metrics})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )