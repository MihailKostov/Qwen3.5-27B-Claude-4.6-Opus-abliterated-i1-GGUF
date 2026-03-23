# Qwen3.5-27B Claude 4.6 Opus Abliterated — Local LLM Server

A local LLM server running **Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated** via `llama.cpp`, exposed through a FastAPI wrapper with streaming support, and optionally tunneled to the internet via Cloudflare.

---

## Repository Structure

```
Claude-4.6/
├── app_llamacpp.py             ← FastAPI server (proxies llama-server)
├── Dockerfile                  ← Docker image definition
├── requirements_llamacpp.txt   ← Python dependencies
├── client.py                   ← Run this on another PC to chat with the server
└── README.md
```

---

## How It Works

```
llama-server  (llama.cpp HTTP server, OpenAI-compatible API)
      ↑ subprocess on port 8080 (internal)
FastAPI / uvicorn  (port 8000, external)
      ↑ handles CORS, /health, /chat, /chat/stream endpoints
Lovable UI / client.py / curl
      ↑ optionally via Cloudflare Tunnel (public HTTPS URL)
```

---

## Prerequisites

- Windows 10/11 with WSL2
- Python 3.12
- NVIDIA GPU (RTX 2070 or better recommended) with up-to-date drivers
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU passthrough in Docker
- [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) for remote access
- ~16 GB free disk space for the model

---

## Option A — Run Directly (No Docker)

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements_llamacpp.txt
```

### 2. Download llama-server binary

Download the latest release from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) — get the `bin-ubuntu-x64.tar.gz` or Windows zip.

Extract and place `llama-server.exe` into `D:/llm-server/llama-bin/`.

### 3. Download the model

The model will auto-download on first start via `huggingface_hub`. Or download manually:

```powershell
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='mradermacher/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-i1-GGUF',
    filename='Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated.i1-Q4_K_M.gguf',
    local_dir='D:/llm-server/.cache/model'
)
"
```

### 4. Start the server

```powershell
python -m uvicorn app_llamacpp:app --host 0.0.0.0 --port 8000 --workers 1
```

### 5. Test it

Use PowerShell (not Git Bash — it mangles paths in curl commands):

```powershell
# Health check
curl http://localhost:8000/health

# Non-streaming chat
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'

# Streaming chat
curl -N -X POST http://localhost:8000/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"Tell me a joke"}]}'
```

> **Important:** Always use PowerShell or CMD for curl on Windows.
> Git Bash converts Unix-style paths like `/app/...` into Windows paths
> like `C:/Program Files/Git/app/...`, breaking all Docker-related commands.

---

## Option B — Docker

### 1. Increase WSL2 memory limit

Docker Desktop on Windows uses WSL2 and is limited to ~8 GB RAM by default.
The model needs ~15 GB. Create or edit `C:\Users\<YourUsername>\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=4GB
processors=8
```

Then restart WSL2 and Docker Desktop:

```powershell
wsl --shutdown
# Then restart Docker Desktop from the system tray
```

### 2. Build the image

```powershell
cd D:\llm-server\Claude-4.6
docker build -t qwen35-server:latest .
```

Build time: ~3–5 minutes (downloads llama-server binary during build).

### 3. Run the container

Run from **PowerShell** (not Git Bash):

```powershell
docker run -d `
  --name qwen35-server `
  --gpus all `
  -p 8000:8000 `
  -v "D:/llm-server/.cache:/app/model_cache" `
  -e MODEL_PATH="/app/model_cache/huggingface/hub/models--mradermacher--Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-i1-GGUF/snapshots/c49b734874219aa0609e0d46f04ef75f8d14fd59/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated.i1-Q4_K_M.gguf" `
  -e LLAMA_SERVER_BIN="/app/llama-bin/llama-b8457/llama-server" `
  -e N_GPU_LAYERS=30 `
  -e N_CTX=4096 `
  -e N_THREADS=8 `
  --memory="22g" `
  --restart unless-stopped `
  qwen35-server:latest
```

### 4. Check startup logs

The model takes ~2–3 minutes to load (15 GB from disk):

```powershell
docker logs -f qwen35-server
```

A successful startup looks like:

```
load_backend: loaded CPU backend from .../libggml-cpu-zen4.so
load_backend: loaded CPU backend from .../libggml-cpu-x64.so
load_tensors: offloaded 30/65 layers to GPU
llama-server ready after 159s ✓
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 5. Verify GPU is being used

```powershell
docker exec qwen35-server nvidia-smi
```

### 6. Test it

```powershell
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"Tell me a joke"}],"max_new_tokens":100}'
```

### Container management

```powershell
docker stop qwen35-server
docker start qwen35-server
docker rm qwen35-server       # remove container (image stays)
docker logs -f qwen35-server  # follow logs
```

---

## Expose via Cloudflare Tunnel

This creates a public HTTPS URL that anyone can use to reach your local server.

### Install cloudflared

```powershell
winget install Cloudflare.cloudflared
```

### Start a quick tunnel (URL changes on restart)

```powershell
cloudflared tunnel --url http://localhost:8000 --protocol http2
```

> **Important:** Use `--protocol http2`. The default QUIC protocol is blocked
> by many ISPs and home routers. Without this flag the tunnel registers but
> the URL stays unreachable.

You'll see output like:

```
Your quick Tunnel has been created! Visit it at:
https://example-words-here.trycloudflare.com
```

Wait 30–60 seconds for DNS propagation before testing the URL.

### Test the tunnel

```powershell
curl https://example-words-here.trycloudflare.com/health
```

---

## Use from Another PC

### Option A — client.py

```powershell
pip install requests
python client.py --host 192.168.1.50            # local network
python client.py --host example-words-here.trycloudflare.com --port 443  # via tunnel
```

### Option B — curl

```powershell
# Non-streaming
curl -X POST https://example-words-here.trycloudflare.com/chat `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"Explain quantum computing"}]}'

# Streaming
curl -N -X POST https://example-words-here.trycloudflare.com/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"Tell me a joke"}]}'
```

### Option C — Python

```python
import requests

SERVER = "https://example-words-here.trycloudflare.com"

def ask(messages, max_new_tokens=512):
    r = requests.post(
        f"{SERVER}/chat",
        json={"messages": messages, "max_new_tokens": max_new_tokens},
        timeout=600
    )
    r.raise_for_status()
    return r.json()["response"]

reply = ask([{"role": "user", "content": "What is the capital of France?"}])
print(reply)
```

---

## API Reference

| Endpoint       | Method | Description                        |
|----------------|--------|------------------------------------|
| `/health`      | GET    | Status, model name, GPU config     |
| `/chat`        | POST   | Full response, returns JSON        |
| `/chat/stream` | POST   | SSE streaming, tokens arrive live  |

### POST `/chat` — Request body

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Hello!"}
  ],
  "max_new_tokens": 512
}
```

### POST `/chat` — Response

```json
{
  "response": "Hello! How can I help you today?",
  "metrics": {
    "first_token_latency_s": 3.2,
    "total_time_s": 45.1,
    "total_tokens": 98,
    "tokens_per_second": 2.58
  }
}
```

### POST `/chat/stream` — SSE Events

```
data: {"token": "Hello"}
data: {"token": "!"}
data: {"done": true, "metrics": {...}}
```

---

## Performance (Ryzen 7 8700F + RTX 2070 8 GB)

| Mode                | tok/s  | Notes                                              |
|---------------------|--------|----------------------------------------------------|
| CPU only (16t)      | ~2.58  | ✓ Recommended — AVX512 accelerates SSM layers      |
| GPU 30 layers + CPU | ~3.84  | ✓ Good balance for this hardware                   |
| GPU only            | OOM    | ✗ Model too large for 8 GB VRAM without offload    |

> This model uses a hybrid SSM/attention architecture (Qwen3.5).
> SSM layers are sequential and benefit from AVX512 on CPU.
> GPU offloading helps the attention layers but adds PCIe overhead for SSM.

---

## Environment Variables

All can be overridden with `-e` in `docker run`:

| Variable           | Default                          | Description                        |
|--------------------|----------------------------------|------------------------------------|
| `LLAMA_SERVER_BIN` | `/app/llama-bin/.../llama-server`| Path to llama-server binary        |
| `MODEL_PATH`       | `/app/model_cache/model.gguf`    | Path to GGUF model file            |
| `N_GPU_LAYERS`     | `0`                              | Layers to offload to GPU (0=CPU)   |
| `N_CTX`            | `8192`                           | Context window size                |
| `N_THREADS`        | `8`                              | CPU threads for inference          |
| `LLAMA_PORT`       | `8080`                           | Internal llama-server port         |
| `API_PORT`         | `8000`                           | External FastAPI port              |

---

## Common Errors & Fixes

### `Cannot copy out of meta tensor; no data!`

**Cause:** Attempting to run the 35B MoE model with `transformers` + `bitsandbytes` 4-bit + CPU offload. The MoE architecture is incompatible with bitsandbytes CPU offload.

**Fix:** Use `llama.cpp` instead of transformers for this model (which is what this repo does).

---

### `make_cpu_buft_list: no CPU backend found`

**Cause:** llama-server cannot find its CPU backend `.so` files (`libggml-cpu-*.so`). Missing `libgomp1` (OpenMP runtime) in the container.

**Fix:** Add `libgomp1` to the Dockerfile apt install:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl tar ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

Also set `GGML_BACKEND_PATH` in the Python subprocess environment to point to the specific CPU backend `.so` file.

---

### `FileNotFoundError: No such file or directory: 'C:/Program Files/Git/app/...'`

**Cause:** Running Docker commands in Git Bash. Git Bash converts Unix paths like `/app/llama-bin` into Windows paths.

**Fix:** Use PowerShell or CMD for all Docker commands. Alternatively, prefix paths with `//` or set `MSYS_NO_PATHCONV=1`.

---

### `curl: (6) Could not resolve host: xxx.trycloudflare.com`

**Cause:** DNS for the new tunnel URL hasn't propagated yet, or QUIC protocol is blocked by your network.

**Fix 1:** Wait 30–60 seconds and retry.

**Fix 2:** Use HTTP/2 explicitly:
```powershell
cloudflared tunnel --url http://localhost:8000 --protocol http2
```

---

### `422 Unprocessable Entity` — `Field required, body is null`

**Cause:** Running curl in Git Bash with `^` line continuations. Git Bash treats `^` as end of command, so `-H` and `-d` flags are sent as separate shell commands.

**Fix:** Use PowerShell (backtick `` ` `` for line continuation) or put the entire curl command on one line in Git Bash.

---

### Container OOM — `14.68 GB / 15.04 GB` memory limit

**Cause:** Docker Desktop on Windows caps WSL2 memory at ~8 GB by default. The model needs ~15 GB.

**Fix:** Create `C:\Users\<YourUsername>\.wslconfig`:
```ini
[wsl2]
memory=24GB
swap=4GB
```
Then run `wsl --shutdown` and restart Docker Desktop.

---

### `llama-server did not start in time`

**Cause:** The 120-second startup timeout is too short for loading a 15 GB model.

**Fix:** Increase the timeout in `app_llamacpp.py`:
```python
for attempt in range(300):  # 5 minutes
```

---

### `load_backend: failed to load ...: Is a directory`

**Cause:** `GGML_BACKEND_PATH` was set to a directory instead of a specific `.so` file.

**Fix:** Point to the specific backend file:
```python
llama_env["GGML_BACKEND_PATH"] = os.path.join(llama_bin_dir, "libggml-cpu-x64.so")
```

---

### `libggml-cpu-x64.so: libgomp.so.1: cannot open shared object file`

**Cause:** OpenMP runtime missing in the Docker image (`python:3.12-slim` is minimal).

**Fix:** Add `libgomp1` to apt installs in the Dockerfile (see above).

---

### Port collision — llama.cpp UI appears on port 8000

**Cause:** `LLAMA_PORT` and `API_PORT` are swapped — llama-server is binding to 8000 instead of 8080.

**Fix:** Ensure in `app_llamacpp.py`:
```python
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))  # internal, llama-server
API_PORT   = int(os.getenv("API_PORT",   "8000"))  # external, FastAPI
```

---

### `500 Internal Server Error` on `/chat` while llama-server returns 200

**Cause:** Response parsing failure — usually the container is out of memory and the response is truncated or malformed.

**Fix:** Check container memory usage in Docker Desktop. If near the limit, increase WSL2 memory (see above) or reduce `N_CTX` from 8192 to 4096 to free up KV-cache RAM:
```powershell
-e N_CTX=4096
```

---

## Save & Share the Docker Image

```powershell
# Export image to file (no Docker Hub needed)
docker save qwen35-server:latest | gzip > qwen35-server.tar.gz

# Import on another machine
docker load < qwen35-server.tar.gz
docker run -d -p 8000:8000 qwen35-server:latest
```

---

## Firewall (if exposing on local network without tunnel)

```powershell
# Windows — allow port 8000 through Windows Firewall
netsh advfirewall firewall add rule name="LLM Server" dir=in action=allow protocol=TCP localport=8000
```