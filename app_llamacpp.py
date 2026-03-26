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
import httpx
import subprocess
import asyncio
import os
import sys
import time
import json
import platform
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, AliasChoices, ConfigDict

logger = logging.getLogger(__name__)

# ── Memory config ─────────────────────────────────────────────────────────────
MAX_BUFFER_TOKENS   = 1200   # verbatim turns kept until this limit
SUMMARY_MAX_TOKENS  = 500    # max tokens in the rolling summary
CHARS_PER_TOKEN     = 3      # rough estimate

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
MEMORY_DIR    = Path(os.getenv("MEMORY_DIR", "./conversation_memory"))
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"  llama-server:  {LLAMA_SERVER_BIN}")
print(f"  Model:         {MODEL_PATH}")
print(f"  GPU layers:    {N_GPU_LAYERS}")
print(f"  Context:       {N_CTX}")
print(f"  CPU threads:   {N_THREADS}")
print(f"  llama port:    {LLAMA_PORT}")
print(f"  Memory dir:    {MEMORY_DIR.resolve()}")
print(f"  Default chat_id for memory (if omitted): {(os.getenv('DEFAULT_MEMORY_CHAT_ID') or 'default')!r}")
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
    model_config = ConfigDict(populate_by_name=True)

    # Backward-compatible path: caller can still pass full messages list.
    messages: Optional[List[Message]] = None
    # New path: send only latest user message and let server restore context.
    chat_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("chat_id", "chatId"),
    )
    system_prompt: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("system_prompt", "systemPrompt"),
    )
    user_message: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("user_message", "userMessage"),
    )
    max_new_tokens: Optional[int] = Field(
        default=512,
        validation_alias=AliasChoices("max_new_tokens", "maxNewTokens"),
    )

class ChatResponse(BaseModel):
    response: str
    metrics: dict

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)

def _count_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)

def _count_turns_tokens(turns: List[Dict[str, str]]) -> int:
    total = 0
    for t in turns:
        total += _count_tokens(t.get("user", ""))
        total += _count_tokens(t.get("assistant", ""))
    return total


async def _summarize_turns(turns: List[Dict[str, str]], existing_summary: str) -> str:
    """
    Summarize a list of turns into a rolling summary.
    Called only when verbatim buffer overflows — not on every turn.
    """
    transcript = "\n\n".join(
        f"User: {t.get('user','').strip()}\nAssistant: {t.get('assistant','').strip()[:600]}"
        for t in turns
    )
    prompt = f"""You are maintaining a conversation memory file.

Previous summary:
{existing_summary or "(none)"}

New turns to incorporate:
{transcript}

Write an updated memory document. Use this structure:

## Goal
What the user is trying to accomplish (2-4 sentences, be specific).

## Key Facts
All important specifics: names, IDs, VINs, file paths, versions, URLs, constraints.
Use bullet points. Include everything that might be needed later.

## Progress
What was tried. What worked. What was ruled out and why.
Be specific — include method names, error messages, outcomes.

## Current Status
1-2 sentences: where things stand and what the next step is.

Rules:
- Be informative and specific — both the AI and the human will read this
- Include ALL specific identifiers mentioned (VINs, IPs, paths, versions)
- Do not truncate or omit technical details
- Keep total length under 500 words
- If no progress yet, say so clearly

Write the memory document now:"""

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "stream": False,
        "temperature": 0.2,
    }
    try:
        r = await http_client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Summarization failed")
        # Fallback: simple concatenation
        lines = [existing_summary or ""]
        for t in turns:
            lines.append(f"- User: {t.get('user','')[:100]}")
        return "\n".join(lines)[:1500]

def _truncate_messages_to_fit(
    messages: List[Dict[str, str]], max_ctx: int, reserve: int = 512
) -> List[Dict[str, str]]:
    budget = max_ctx - reserve
    total = sum(_estimate_tokens(m.get("content", "")) for m in messages)
    if total <= budget:
        return messages

    print(f"[WARN] ~{total} tokens estimated, truncating to fit {budget} token budget")
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system  = [m for m in messages if m["role"] != "system"]
    last_user   = non_system[-1]
    history     = non_system[:-1]

    while history:
        total = sum(_estimate_tokens(m.get("content", "")) for m in system_msgs + history + [last_user])
        if total <= budget:
            break
        history.pop(0)

    return system_msgs + history + [last_user]

def _resolve_chat_id(req: ChatRequest) -> str:
    """
    Persist and load memory under this id. If the client omits chat_id (e.g. raw /chat with only
    messages), fall back to DEFAULT_MEMORY_CHAT_ID env or 'default' so curl/local tests still write files.
    """
    raw = (req.chat_id or "").strip()
    if raw:
        return raw
    fallback = (os.getenv("DEFAULT_MEMORY_CHAT_ID") or "default").strip()
    return fallback or "default"


def _safe_chat_id(chat_id: str) -> str:
    return "".join(ch for ch in chat_id if ch.isalnum() or ch in ("-", "_"))[:128] or "default"


def _memory_paths(chat_id: str) -> tuple[Path, Path]:
    safe_id = _safe_chat_id(chat_id)
    chat_dir = MEMORY_DIR / safe_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir / "memory.json", chat_dir / "README.md"


def _default_memory() -> Dict[str, Any]:
    return {
        "summary": "",
        "recent_turns": [],
        "updated_at": None,
    }


def _load_memory(chat_id: str) -> Dict[str, Any]:
    memory_path, _ = _memory_paths(chat_id)
    if not memory_path.exists():
        return _default_memory()
    try:
        return json.loads(memory_path.read_text(encoding="utf-8"))
    except Exception:
        return _default_memory()


def _save_memory(chat_id: str, memory: Dict[str, Any]) -> None:
    memory_path, readme_path = _memory_paths(chat_id)
    memory["updated_at"] = datetime.utcnow().isoformat() + "Z"
    memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_readme(chat_id, memory, readme_path)


def _write_readme(chat_id: str, memory: Dict[str, Any], readme_path: Path) -> None:
    summary = (memory.get("summary") or "").strip()
    buffer: List[Dict[str, str]] = memory.get("recent_turns", [])
    updated = memory.get("updated_at", "unknown")

    lines = [
        f"# Conversation Memory: {chat_id}",
        f"_Last updated: {updated} — {len(buffer)} turn(s) in verbatim buffer_",
        "",
        "_Auto-generated. Edit memory.json to correct history._",
        "",
    ]

    if summary:
        lines += [
            "## Summary of Earlier Turns",
            "",
            summary,
            "",
        ]

    if buffer:
        lines.append("## Recent Turns (Verbatim)")
        lines.append("")
        for i, turn in enumerate(buffer, 1):
            lines += [
                f"### Turn {i}",
                f"**User:** {turn.get('user', '').strip()}",
                "",
                f"**Assistant:** {turn.get('assistant', '').strip()}",
                "",
            ]

    readme_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

def _extract_last_user_from_messages(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _message_to_openai_dict(m: Message) -> Dict[str, str]:
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()


def _build_messages_with_memory(req: ChatRequest) -> List[Dict[str, str]]:
    cid = _resolve_chat_id(req)

    if req.user_message is not None:
        memory = _load_memory(cid)
        system_prompt = req.system_prompt or "You are a helpful assistant."
        summary = (memory.get("summary") or "").strip()
        buffer: List[Dict[str, str]] = memory.get("recent_turns", [])

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Inject rolling summary of older turns
        if summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{summary}"
            })

        # Inject recent verbatim turns
        for turn in buffer:
            user_text = (turn.get("user") or "").strip()
            asst_text = (turn.get("assistant") or "").strip()
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if asst_text:
                messages.append({"role": "assistant", "content": asst_text})

        messages.append({"role": "user", "content": req.user_message})
        return _truncate_messages_to_fit(messages, N_CTX)

    if req.messages:
        msgs = [_message_to_openai_dict(m) for m in req.messages]
        return _truncate_messages_to_fit(msgs, N_CTX)

    raise HTTPException(
        status_code=400,
        detail="Provide either messages or user_message.",
    )

async def _persist_turn(
    req: ChatRequest,
    built_messages: List[Dict[str, str]],
    assistant_text: str,
) -> None:
    chat_id = _resolve_chat_id(req)
    user_text = req.user_message or _extract_last_user_from_messages(built_messages)

    if not user_text.strip() or not (assistant_text or "").strip():
        return

    memory = _load_memory(chat_id)
    summary = memory.get("summary", "")
    buffer: List[Dict[str, str]] = memory.get("recent_turns", [])

    # Add new turn to verbatim buffer — always store full text
    buffer.append({"user": user_text, "assistant": assistant_text})

    # If buffer exceeds token limit, summarize oldest half and compress
    if _count_turns_tokens(buffer) > MAX_BUFFER_TOKENS:
        split = max(1, len(buffer) // 2)
        turns_to_summarize = buffer[:split]
        buffer = buffer[split:]          # keep newer half verbatim
        print(f"[MEMORY] Buffer overflow — summarizing {len(turns_to_summarize)} oldest turns")
        summary = await _summarize_turns(turns_to_summarize, summary)

    memory["summary"] = summary
    memory["recent_turns"] = buffer
    _save_memory(chat_id, memory)

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
    built_messages = _build_messages_with_memory(req)

    payload = {
        "messages": built_messages,
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
    await _persist_turn(req, built_messages, response_text)
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
    built_messages = _build_messages_with_memory(req)
    payload = {
        "messages": built_messages,
        "max_tokens": req.max_new_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    async def event_generator():
        init_time = time.time()
        first_token_time = None
        token_count = 0
        response_buffer = ""

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
                    response_buffer += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'token': f'[Error: {e}]'})}\n\n"

        finally:
            end_time = time.time()
            total = end_time - init_time
            try:
                await _persist_turn(req, built_messages, response_buffer)
            except Exception:
                logger.exception("conversation_memory: persist after /chat/stream failed")
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


@app.post("/chat/stream-text")
async def chat_stream_text(req: ChatRequest):
    """
    Streaming plain text chunks (AI SDK friendly).
    This endpoint is useful for @ai-sdk/react `useCompletion` with streamProtocol='text'.
    """
    built_messages = _build_messages_with_memory(req)
    payload = {
        "messages": built_messages,
        "max_tokens": req.max_new_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    async def text_generator():
        response_buffer = ""
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
                    response_buffer += token
                    yield token
        finally:
            try:
                await _persist_turn(req, built_messages, response_buffer)
            except Exception:
                logger.exception("conversation_memory: persist after /chat/stream-text failed")

    return StreamingResponse(
        text_generator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )