from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama

from src.embedding import EmbeddingPipeline
from src.vectorstore import VectorStore


class RAGSearch:
    """Retrieve relevant chunks and answer questions with a local Ollama model."""

    def __init__(
        self,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        vector_store: Optional[VectorStore] = None,
        persist_directory: str = "vectorstore",
        model_name: str = "qwen2.5-coder:7b",
        temperature: float = 0.5,
        llm: Optional[Any] = None,
    ) -> None:
        if vector_store is None:
            pipeline = embedding_pipeline or EmbeddingPipeline()
            vector_store = VectorStore(
                embedding_pipeline=pipeline,
                persist_directory=persist_directory,
            )

        self.vector_store = vector_store
        self.llm = llm if llm is not None else ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return Chroma query results in the record format used by the RAG pipeline."""
        results = self.vector_store.query(query_text=query, top_k=top_k)
        documents = results.get("documents")
        if not documents or not documents[0]:
            return []

        document_texts = documents[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        retrieved: List[Dict[str, Any]] = []
        for index, document in enumerate(document_texts):
            distance = distances[index]
            similarity_score = 1.0 - distance
            if similarity_score < score_threshold:
                continue

            retrieved.append(
                {
                    "id": ids[index],
                    "document": document,
                    "metadata": metadatas[index] or {},
                    "similarity_score": similarity_score,
                    "distance": distance,
                    "rank": index + 1,
                }
            )

        return retrieved

    def rag_simple(self, query: str) -> str:
        """Retrieve context and return a concise answer, or a no-context message."""
        results = self.retrieve(query)
        if not results:
            return "No relevant context found to answer the question."

        context = "\n\n".join(doc["document"] for doc in results)
        prompt = f"""Use the following context to answer the question concisely.
                    Context: {context}
                    Question: {query}
                    Answer: """
        response = self.llm.invoke(prompt)
        return response.content

    def rag_advanced(
        self,
        query: str,
        min_score: float = 0.2,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """Return the answer, source details, confidence, and optional context."""
        results = self.retrieve(query, score_threshold=min_score)
        if not results:
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "confidence": 0.0,
                "context": "",
            }

        context = "\n\n".join(doc["document"] for doc in results)
        sources = [
            {
                "source": doc["metadata"].get(
                    "source_file", doc["metadata"].get("source", "unknown")
                ),
                "page": doc["metadata"].get("page", "unknown"),
                "score": doc["similarity_score"],
                "preview": doc["document"][:120] + "...",
            }
            for doc in results
        ]

        prompt = f"""Use the following context to answer the question concisely.
                    Context: {context}
                    Question: {query}
                    Answer: """
        response = self.llm.invoke(prompt)

        output = {
            "answer": response.content,
            "sources": sources,
            "confidence": max(doc["similarity_score"] for doc in results),
        }
        
        if return_context:
            output["context"] = context
        return output

    def search(self, query: str, min_score: float = 0.2, return_context: bool = False) -> Dict[str, Any]:
        """Run the enhanced RAG search pipeline."""

        return self.rag_advanced(
            query,
            min_score=min_score,
            return_context=return_context,
        )
