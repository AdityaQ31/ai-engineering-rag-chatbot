# ai-engineering-rag-chatbot
A Retrieval-Augmented Generation (RAG) chatbot built with a fine-tuned Llama 3.2 1B model, LoRA, Qdrant Cloud, and FastAPI to answer questions from AI Engineering book content.

# AI Engineering RAG Chatbot

An AI-powered chatbot that answers questions about *AI Engineering* by Chip Huyen using Retrieval-Augmented Generation (RAG), a fine-tuned Llama 3.2 1B model, and Qdrant Cloud.

## Features

- **Retrieval-Augmented Generation (RAG):** Retrieves relevant book excerpts to answer user questions.
- **Fine-tuned LLM:** Uses a Llama 3.2 1B Instruct model with a LoRA adapter.
- **Semantic Search:** Uses `BAAI/bge-small-en-v1.5` embeddings.
- **Vector Database:** Stores and searches document embeddings using Qdrant Cloud.
- **FastAPI Backend:** Provides API endpoints for chatbot interaction.
- **Web Interface:** A responsive chat interface for asking questions.
- **Conversational Mode:** Supports casual conversation alongside book-based question answering.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- Sentence Transformers
- Qdrant Cloud
- FastAPI
- HTML, CSS, JavaScript

## Architecture

1. Load and preprocess the AI Engineering book PDF.
2. Split the extracted text into chunks.
3. Generate embeddings using `BAAI/bge-small-en-v1.5`.
4. Store the chunks and embeddings in Qdrant Cloud.
5. Retrieve relevant chunks for a user's question.
6. Generate an answer using the fine-tuned Llama model.
7. Display the response and source references in the web interface.

## Project Structure

```text
ai-engineering-rag-chatbot/
├── app.py
├── rag/
│   ├── __init__.py
│   ├── vector_store.py
│   ├── model_loader.py
│   └── pipeline.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-engineering-rag-chatbot.git
cd ai-engineering-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example` and configure the required values:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=ai_engineering
BASE_MODEL_PATH=your_base_model_path
ADAPTER_PATH=your_lora_adapter_path
```

Replace the example values with your actual configuration. Never commit `.env` or API keys to GitHub.

### 5. Run the application

```bash
uvicorn app:app --reload
```

Open the application at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web chat interface |
| GET | `/health` | Health check |
| GET | `/api/stats` | Application statistics |
| POST | `/api/ask` | Submit a question to the chatbot |

Example request:

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "top_k": 3
}
```

## Hardware Requirements

The model is configured for local inference. Actual requirements depend on model loading, quantization, and context length.

The development setup uses an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM and 24 GB system RAM. Four-bit quantization can help reduce GPU memory usage, but available memory may still limit performance.

## Current Limitations

- Local model paths must be configured for your machine.
- Qdrant Cloud credentials are required for the hosted vector database.
- GPU memory limits can affect inference speed and context length.
- Public deployment requires separate configuration for model files, secrets, and inference resources.

## Future Improvements

- Improve RAG evaluation and retrieval quality.
- Add conversation history and streaming responses.
- Implement automated evaluation using RAG-specific metrics.
- Deploy the application with a suitable inference backend.
- Add document upload support.

## License

Add a license appropriate for your code and verify the applicable license terms for the underlying Llama model and any source documents before redistributing them.

---

**Built as an AI Engineering project to explore LLM fine-tuning, semantic search, and Retrieval-Augmented Generation.**
