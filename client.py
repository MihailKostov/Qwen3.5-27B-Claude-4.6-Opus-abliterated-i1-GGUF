"""
client.py – call the LLM server from any other PC on the same network.

Usage:
    python client.py                        # interactive chat (non-streaming)
    python client.py --stream               # interactive chat with streaming
    python client.py --host 192.168.1.50    # point to a specific server IP
"""

import argparse
import json
import requests
import sseclient   # pip install sseclient-py

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_HOST = "192.168.1.50"   # ← change to your server's IP
DEFAULT_PORT = 8000

def build_url(host, port, path):
    return f"http://{host}:{port}{path}"

# ── Non-streaming chat ────────────────────────────────────────────────────────
def chat(host, port, messages, max_new_tokens=4096):
    url = build_url(host, port, "/chat")
    payload = {"messages": messages, "max_new_tokens": max_new_tokens}
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["response"], data["metrics"]

# ── Streaming chat ────────────────────────────────────────────────────────────
def chat_stream(host, port, messages, max_new_tokens=4096):
    url = build_url(host, port, "/chat/stream")
    payload = {"messages": messages, "max_new_tokens": max_new_tokens, "stream": True}
    response = requests.post(url, json=payload, stream=True, timeout=300)
    response.raise_for_status()

    full_text = ""
    metrics = {}
    client = sseclient.SSEClient(response)
    for event in client.events():
        data = json.loads(event.data)
        if data.get("done"):
            metrics = data.get("metrics", {})
            break
        token = data.get("token", "")
        print(token, end="", flush=True)
        full_text += token

    print()  # newline after streamed output
    return full_text, metrics

# ── Interactive loop ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    # Ping health endpoint
    try:
        r = requests.get(build_url(args.host, args.port, "/health"), timeout=5)
        info = r.json()
        print(f"✓ Connected to {args.host}:{args.port}  |  model: {info['model']}\n")
    except Exception as e:
        print(f"✗ Cannot reach server at {args.host}:{args.port} — {e}")
        return

    messages = []
    print("Commands: /exit  /clear")
    print(f"Mode: {'streaming' if args.stream else 'non-streaming'}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/clear":
            messages = []
            print("[History cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        try:
            if args.stream:
                response, metrics = chat_stream(args.host, args.port, messages)
            else:
                response, metrics = chat(args.host, args.port, messages)
                print(response)
        except Exception as e:
            print(f"\n[Error: {e}]")
            messages.pop()
            continue

        print(f"\n  ⏱  {metrics.get('total_time_s')}s  |  "
              f"{metrics.get('tokens_per_second')} tok/s  |  "
              f"first-token latency: {metrics.get('first_token_latency_s')}s\n")

        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
