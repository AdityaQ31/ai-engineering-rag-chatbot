# AI Engineering RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for exploring concepts from *AI Engineering* by Chip Huyen. It combines a FastAPI backend, Qdrant Cloud vector search, `BAAI/bge-small-en-v1.5` embeddings, and a locally loaded fine-tuned Llama 3.2 1B Instruct model with a LoRA adapter.

> **Deployment note:** this repository contains application code, not model weights. The current model loader expects the base model and LoRA adapter to exist as local directories. A cloud deployment will not work until model inference is configured on a host that can access those files and has enough memory/compute.

## Features

- Retrieve relevant book excerpts from an existing Qdrant Cloud collection.
- Generate answers with a local Llama 3.2 1B Instruct model and PEFT/LoRA adapter.
- Use BGE embeddings for semantic search.
- FastAPI endpoints and a responsive HTML/CSS/JavaScript chat UI.
- Show source page references returned by the vector store.
- Keep secrets and local model weights out of the repository.

## Architecture

Browser UI → FastAPI `/api/ask` → BGE query embedding → Qdrant retrieval → fine-tuned Llama model → answer and source references.

## Project structure

```text
ai-engineering-rag-chatbot/
├── app.py
├── rag/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── pipeline.py
│   └── vector_store.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10 or 3.11 recommended.
- An existing Qdrant Cloud collection named `ai_engineering` (or your configured collection) with point payload fields `text` and `metadata`.
- Local directories containing the base Llama model and saved LoRA adapter.
- A compatible PyTorch / Transformers / PEFT / bitsandbytes environment.
- For practical local GPU inference, a supported NVIDIA GPU and enough available VRAM. The development machine used an RTX 3050 Laptop GPU with 6 GB VRAM; 4-bit quantization reduces memory use but does not guarantee every configuration will fit.

## Local setup (Windows PowerShell)

Clone the repository:

```powershell
git clone https://github.com/AdityaQ31/ai-engineering-rag-chatbot.git
cd ai-engineering-rag-chatbot
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and provide your own Qdrant credentials and the actual local model paths:

```env
QDRANT_URL=https://your-qdrant-cluster-endpoint
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=ai_engineering
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
BASE_MODEL_PATH=D:/path/to/Llama-3.2-1B-Instruct
ADAPTER_PATH=D:/path/to/Llama-3.2-1B-QLoRA-Final
```

Use paths that exist on your own machine. Never commit `.env` or paste real credentials into source code.

Run the app:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Web chat interface |
| GET | `/health` | Basic health check |
| GET | `/api/stats` | Qdrant collection stats |
| POST | `/api/ask` | Ask a question |

Example request body:

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "top_k": 3
}
```

## Deployment limitations

The application loads the model during startup using `BASE_MODEL_PATH` and `ADAPTER_PATH`. These are local filesystem paths and will not exist on a cloud host automatically. A public deployment needs a separate inference plan and secure configuration for model files, secrets, RAM/VRAM, and startup time. A free web-service tier alone should not be assumed to provide a suitable GPU for this model.

## Security

- `.env`, API keys, virtual environments, caches, and model weight files should not be committed.
- If a Qdrant API key has ever been exposed in a notebook, chat, or public commit, rotate it in Qdrant and update your local `.env`.
- The MIT license in this repository applies to this project's code. Check the separate license terms for Llama model assets and any source book content before redistributing those materials.

## Future improvements

- Add tests for retrieval, API validation, and error handling.
- Improve RAG evaluation and retrieval quality.
- Add conversation history and streaming.
- Add document ingestion and upload support.
- Configure and verify a suitable hosted inference backend before publishing a live demo.

## License

This project's code is released under the MIT License. See [LICENSE](LICENSE).
