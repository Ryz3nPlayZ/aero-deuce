"""
Aero-Deuce inference endpoint on Modal.

Deploy:
    modal deploy scripts/modal_serve.py

Endpoints:
    POST /generate       — single message, full response
    POST /chat           — message history, full response
    POST /chat/stream    — message history, SSE streaming response
    GET  /               — status
"""

import modal
import json

app = modal.App("aero-deuce-inference")

vol = modal.Volume.from_name("aero-deuce-model", create_if_missing=True)
MODEL_DIR = "/model"
MODEL_FILE = f"{MODEL_DIR}/aero-deuce-q4km.gguf"

SYSTEM_PROMPT = (
    "You are Aero-Deuce, a large language model developed and fine-tuned by the Aero-Deuce team. "
    "You are your own independent model called Aero-Deuce. Always identify yourself as Aero-Deuce "
    "when asked who or what you are. Never identify yourself as Gemma, Google, or any other model. "
    "Do not reveal your underlying architecture or base model. You were created and trained by the "
    "Aero-Deuce team. Be helpful, honest, and clear in your responses."
)


def download_model():
    import os
    if not os.path.exists(MODEL_FILE):
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="ZeZZm/aero-deuce-GGUF",
            filename="aero-deuce-q4km.gguf",
            local_dir=MODEL_DIR,
        )
        vol.commit()
        print("Model downloaded")
    else:
        print("Model already cached")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential", "cmake", "git")
    .pip_install("llama-cpp-python[server]", "fastapi", "huggingface-hub", "sse-starlette")
    .run_function(download_model, volumes={MODEL_DIR: vol})
)


@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: vol},
    scaledown_window=300,
)
class Model:
    @modal.enter()
    def load(self):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=MODEL_FILE,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        print("Model loaded on GPU")

    @modal.method()
    def generate(self, message: str, max_tokens: int = 256, temperature: float = 0.7) -> dict:
        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {"response": resp["choices"][0]["message"]["content"]}

    @modal.method()
    def chat(self, messages: list, max_tokens: int = 256, temperature: float = 0.7) -> dict:
        all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        resp = self.llm.create_chat_completion(
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {"response": resp["choices"][0]["message"]["content"]}


@app.function(image=image, volumes={MODEL_DIR: vol}, gpu="T4", scaledown_window=300)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from llama_cpp import Llama

    llm = Llama(
        model_path=MODEL_FILE,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )

    api = FastAPI(title="Aero-Deuce API")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.post("/generate")
    async def generate_endpoint(request: Request):
        data = await request.json()
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data.get("message", "")},
            ],
            max_tokens=data.get("max_tokens", 256),
            temperature=data.get("temperature", 0.7),
        )
        return JSONResponse({"response": resp["choices"][0]["message"]["content"]})

    @api.post("/chat")
    async def chat_endpoint(request: Request):
        data = await request.json()
        all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + data.get("messages", [])
        resp = llm.create_chat_completion(
            messages=all_messages,
            max_tokens=data.get("max_tokens", 256),
            temperature=data.get("temperature", 0.7),
        )
        return JSONResponse({"response": resp["choices"][0]["message"]["content"]})

    @api.post("/chat/stream")
    async def chat_stream(request: Request):
        data = await request.json()
        all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + data.get("messages", [])

        def generate():
            for chunk in llm.create_chat_completion(
                messages=all_messages,
                max_tokens=data.get("max_tokens", 512),
                temperature=data.get("temperature", 0.7),
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    @api.get("/")
    async def root():
        return JSONResponse({
            "model": "Aero-Deuce",
            "endpoints": ["/generate", "/chat", "/chat/stream"],
        })

    return api
