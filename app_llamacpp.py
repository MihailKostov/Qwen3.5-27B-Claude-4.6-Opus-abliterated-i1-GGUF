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
MAX_ASSISTANT_STORE = 600   # chars stored per assistant turn
MAX_SUMMARY_CHARS   = 800   # hard cap on summary size
MAX_RECENT_TURNS    = 3     # raw turns kept verbatim
COMPRESS_THRESHOLD  = 400   # only re-summarize when summary exceeds this

SUMMARY_PROMPT = """You are a memory compressor for an AI assistant.

Given the previous summary and a new exchange, produce an updated summary.

Rules:
- Maximum 5 bullet points total
- Each bullet: one fact, one constraint, or one decision — under 20 words
- Merge or drop bullets that are superseded by new information
- Never include examples, code, or explanations — facts only
- Format: "- [category]: [fact]" where category is Goal/Constraint/Decision/Context/Open

Previous summary:
{previous_summary}

New exchange:
User: {user_text}
Assistant: {assistant_summary}

Updated summary (5 bullets max):"""

ENTITY_PROMPT = """Extract key facts from this message as bullet points.
Only extract: names, IDs, technical specs, explicit constraints, explicit decisions.
Skip opinions, questions, and general information.
Maximum 3 bullets. If nothing worth extracting, return empty string.

Message: {text}

Facts:"""

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
    recent_turns = memory.get("recent_turns", [])
    lines = [
        f"# Conversation Memory: {chat_id}",
        "",
        "This file is generated by the API server to preserve long-term context.",
        "",
        "## Summary",
        summary if summary else "_No summary yet._",
        "",
        "## Recent Turns",
    ]
    if not recent_turns:
        lines.append("_No turns yet._")
    else:
        for idx, turn in enumerate(recent_turns[-8:], 1):
            lines.append(f"### Turn {idx}")
            lines.append(f"- User: {turn.get('user', '').strip()}")
            lines.append(f"- Assistant: {turn.get('assistant', '').strip()}")
            lines.append("")
    readme_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


async def _extract_facts(text: str) -> str:
    """Lightweight call — extracts discrete facts from a single message."""
    prompt = ENTITY_PROMPT.format(text=text[:1000])  # cap input
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "stream": False,
        "temperature": 0.1,
    }
    try:
        r = await http_client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


async def _refresh_summary(previous_summary: str, user_text: str, assistant_text: str) -> str:
    """Full compression — only called when summary exceeds COMPRESS_THRESHOLD."""
    # Truncate assistant text for the summary prompt — we don't need the full reply
    assistant_snippet = assistant_text[:400] + ("..." if len(assistant_text) > 400 else "")

    prompt = SUMMARY_PROMPT.format(
        previous_summary=previous_summary or "(none)",
        user_text=user_text[:600],
        assistant_summary=assistant_snippet,
    )
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "stream": False,
        "temperature": 0.2,
    }
    try:
        r = await http_client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        new_summary = r.json()["choices"][0]["message"]["content"].strip()
        # Enforce hard cap
        return new_summary[:MAX_SUMMARY_CHARS]
    except Exception:
        # Fallback: keep existing summary, append a short note
        fallback = (previous_summary or "").strip()
        note = f"- Context: user asked about '{user_text[:80]}'"
        return (fallback + "\n" + note).strip()[:MAX_SUMMARY_CHARS]


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

    # Server-side memory mode: latest user turn + stored summary / recent turns.
    if req.user_message is not None:
        memory = _load_memory(cid)
        system_prompt = req.system_prompt or "You are a helpful assistant."
        summary = (memory.get("summary") or "").strip()
        recent_turns = memory.get("recent_turns", [])[-6:]

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Conversation memory summary from previous turns. "
                        "Use it as context, but prioritize latest user instructions.\n\n"
                        f"{summary}"
                    ),
                }
            )
        for turn in recent_turns:
            user_text = (turn.get("user") or "").strip()
            assistant_text = (turn.get("assistant") or "").strip()
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

        messages.append({"role": "user", "content": req.user_message})
        return _truncate_messages_to_fit(messages, N_CTX)

    # Legacy: full messages[] each request; memory still updates after reply under cid.
    if req.messages:
        return [_message_to_openai_dict(m) for m in req.messages]

    raise HTTPException(
        status_code=400,
        detail="Provide either messages (full history) or user_message (server-side memory mode).",
    )


async def _persist_turn(
    req: ChatRequest,
    built_messages: List[Dict[str, str]],
    assistant_text: str,
) -> None:
    chat_id = _resolve_chat_id(req)
    user_text = req.user_message or _extract_last_user_from_messages(built_messages)

    if not user_text.strip() or not (assistant_text or "").strip():
        logger.debug("Skipping memory persist: missing user or assistant text")
        return

    memory = _load_memory(chat_id)
    current_summary = memory.get("summary", "")

    # Truncate assistant response before storing — full response not needed for context
    assistant_snippet = assistant_text.strip()[:MAX_ASSISTANT_STORE]
    if len(assistant_text) > MAX_ASSISTANT_STORE:
        assistant_snippet += "... [truncated]"

    # Update recent turns
    recent_turns = memory.get("recent_turns", [])
    recent_turns.append({"user": user_text, "assistant": assistant_snippet})
    memory["recent_turns"] = recent_turns[-MAX_RECENT_TURNS:]

    # Two-path summary update:
    # - Summary is long → run full compression (expensive, occasional)
    # - Summary is short → just extract facts and append (cheap, every turn)
    if len(current_summary) > COMPRESS_THRESHOLD:
        memory["summary"] = await _refresh_summary(
            current_summary, user_text, assistant_text
        )
    else:
        new_facts = await _extract_facts(user_text)
        if new_facts:
            memory["summary"] = (current_summary + "\n" + new_facts).strip()[:MAX_SUMMARY_CHARS]

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