import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.model_loader import load_local_model
from rag.pipeline import RAGPipeline
from rag.vector_store import VectorManager

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model, Qdrant client, and LLM once at startup."""
    vector_store = None
    try:
        vector_store = VectorManager(
            collection_name=os.getenv("QDRANT_COLLECTION", "ai_engineering"),
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        )
        tokenizer, model = load_local_model()
        app.state.rag_bot = RAGPipeline(vector_store=vector_store, tokenizer=tokenizer, model=model)
        app.state.vector_store = vector_store
        print("RAG application is ready.")
        yield
    finally:
        if vector_store is not None:
            vector_store.close()


app = FastAPI(title="AI Engineering RAG Assistant", description="Ask questions about AI Engineering using retrieval-augmented generation.", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(status_code=503, detail="RAG services are not ready.")
    try:
        return vector_store.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}") from exc


@app.post("/api/ask")
def ask(request: AskRequest):
    rag_bot = getattr(app.state, "rag_bot", None)
    if rag_bot is None:
        raise HTTPException(status_code=503, detail="RAG services are not ready.")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    try:
        result = rag_bot.ask(question, top_k=request.top_k)
        return {"question": result["question"], "answer": result["answer"], "sources": result["sources"]}
    except Exception as exc:
        print(f"RAG request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="The RAG request failed. Check the server terminal for details.") from exc
