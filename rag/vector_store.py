import os
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings


class VectorManager:
    """Connect to the existing Qdrant collection created by the ingestion pipeline."""
    def __init__(self, collection_name="ai_engineering", model_name="BAAI/bge-small-en-v1.5", url=None, api_key=None):
        qdrant_url = url or os.getenv("QDRANT_URL")
        qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in your .env file.")
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
        if not self.client.collection_exists(self.collection_name):
            raise RuntimeError(f"Qdrant collection '{self.collection_name}' does not exist. Run your ingestion notebook first.")
        info = self.client.get_collection(self.collection_name)
        print(f"Connected to Qdrant collection '{self.collection_name}' with {info.points_count} points.")

    def search(self, query: str, top_k: int = 3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query_vector = self.embedding_model.embed_query(query)
        points = self.client.query_points(collection_name=self.collection_name, query=query_vector, limit=top_k, with_payload=True).points
        results = []
        for point in points:
            payload = point.payload or {}
            results.append({"id": str(point.id), "document": payload.get("text", ""), "metadata": payload.get("metadata", {}), "score": float(point.score) if point.score is not None else None})
        return results

    def get_stats(self):
        info = self.client.get_collection(self.collection_name)
        return {"collection": self.collection_name, "points_count": info.points_count, "status": str(info.status)}

    def close(self):
        self.client.close()
