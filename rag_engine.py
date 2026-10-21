"""Core RAG logic: embeddings, vector search and grounded answer generation."""

import os
import time

import faiss
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer

# Lightweight embedding model used to convert text into vectors
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Primary LLM first, with a fallback in case the primary is overloaded
LLM_MODEL_NAMES = ["gemini-3.8-flash", "gemini-3.5-flash"]

# Fixed reply used when the documents do not contain the answer
NO_ANSWER_MESSAGE = "The document does not contain this information."


class RAGEngine:
    """Builds a vector index from document chunks and answers questions using them."""

    def __init__(self, api_key, embedding_model=None):
        # Reuse a pre-loaded embedding model when provided to save time and memory
        self.embedding_model = embedding_model or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = genai.Client(api_key=api_key)
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        """Embed all chunks and store them in a FAISS index."""
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embedding_model.encode(texts)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize vectors so that inner product equals cosine similarity
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query, top_k=3):
        """Return the top_k chunks most similar to the query."""
        if self.index is None:
            raise ValueError("The index is empty. Call build_index() first.")

        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype="float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS returns -1 when fewer than top_k vectors are available
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "source": chunk.get("source", ""),
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score),
            })

        return results

    def _call_llm(self, prompt, max_retries=3):
        """Call the LLM with retries and model fallback to handle temporary server errors."""
        for model_name in LLM_MODEL_NAMES:
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=model_name, contents=prompt
                    )
                    return response.text
                except Exception as error:
                    print(f"{model_name} failed (attempt {attempt + 1}): {error}")
                    # Wait longer after each failed attempt (exponential backoff)
                    time.sleep(2 ** attempt)

        return "The service is temporarily unavailable. Please try again later."

    def answer(self, query, top_k=4):
        """Return (answer_text, sources) where sources is a list of (file_name, page) pairs."""
        results = self.retrieve(query, top_k)

        # Combine the retrieved chunks into a single context block with their origin
        context = "\n\n".join(
            f"[{r['source']} | Page {r['page']}] {r['text']}" for r in results
        )

        # Restrict the model to the provided context (guardrail against hallucination)
        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the context below. "
            f"If the answer is not present in the context, reply exactly: '{NO_ANSWER_MESSAGE}'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        answer_text = self._call_llm(prompt)

        # Collect the unique (file, page) pairs used as sources
        sources = sorted({(r["source"], r["page"]) for r in results})
        return answer_text, sources