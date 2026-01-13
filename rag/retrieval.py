import faiss
import pickle
import numpy as np
from typing import List, Tuple

from rag.embeddings import Embedder


class Retriever:
    def __init__(
        self,
        index_path: str,
        mapping_path: str,
        top_k: int = 5
    ):
        self.index = faiss.read_index(index_path)

        with open(mapping_path, "rb") as f:
            self.id_to_text = pickle.load(f)

        self.embedder = Embedder()
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """
        Returns top-k retrieved chunks and similarity scores
        """
        query_embedding = self.embedder.embed([query])

        scores, indices = self.index.search(query_embedding, self.top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.id_to_text[idx], float(score)))

        return results

if __name__ == "__main__":
    retriever = Retriever(
        index_path="rag/index.faiss",
        mapping_path="rag/id_to_text.pkl",
        top_k=3
    )

    query = "Why do attention heads improve transformer performance?"
    results = retriever.retrieve(query)

    for i, (text, score) in enumerate(results, 1):
        print(f"\n--- Result {i} (score={score:.4f}) ---")
        print(text)
