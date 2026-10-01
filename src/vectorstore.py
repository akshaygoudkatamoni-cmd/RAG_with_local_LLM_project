import os
import chromadb
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
try:
    from src.data_loader import load_all_documents
except ImportError:
    from data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

class VectorStore:
    def __init__(self, embedding_pipeline: EmbeddingPipeline, persist_directory: str = "vectorstore"):
        self.embedding_pipeline = embedding_pipeline
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection("document_embeddings")

        if self.collection.count() == 0:
            self.data_path = "data"  # Specify the path to your data directory
            self.documents = load_all_documents(self.data_path)
            self.add_documents(self.documents)

    def add_documents(self, documents: List[Any]):
        chunks = self.embedding_pipeline.chunk_documents(documents)
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        
        for i, chunk in enumerate(chunks):
            self.collection.add(
                ids=[str(i)],
                embeddings=[embeddings[i].tolist()],
                metadatas=[{"source": chunk.metadata.get("source", "unknown")}],
                documents=[chunk.page_content]
            )
        print(f"[INFO] Added {len(chunks)} chunks to the vector store.")

    def query(self, query_text: str, top_k: int = 5) -> List[Any]:
        query_embedding = self.embedding_pipeline.model.encode([query_text], convert_to_numpy=True)[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        return results