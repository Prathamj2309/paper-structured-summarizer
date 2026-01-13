from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

if __name__ == "__main__":
    embedder = Embedder()
    vecs = embedder.embed(["Transformers use self-attention."])

    print("Embedding shape:", vecs.shape)
    print("L2 norm:", np.linalg.norm(vecs[0]))