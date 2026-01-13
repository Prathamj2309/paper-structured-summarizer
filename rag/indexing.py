import faiss
import pickle
from typing import List

from rag.embeddings import Embedder


def build_faiss_index(
    chunks: List[str],
    index_path: str,
    mapping_path: str
):
    embedder = Embedder()
    embeddings = embedder.embed(chunks)

    dim = embeddings.shape[1]

    # Using inner product because embeddings are normalized
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    id_to_text = {i: chunk for i, chunk in enumerate(chunks)}
    with open(mapping_path, "wb") as f:
        pickle.dump(id_to_text, f)

    print(f"Saved FAISS index → {index_path}")
    print(f"Saved ID mapping → {mapping_path}")

if __name__ == "__main__":
    from rag.chunking import TextChunker

    sample_text = """
    Transformers rely on self-attention mechanisms to model long-range dependencies.
    Scaling attention heads improves representational capacity but introduces
    optimization challenges. Layer normalization stabilizes training by normalizing
    activations across feature dimensions. However, deeper transformer models
    suffer from vanishing gradients unless residual connections are used.
    """ * 20

    chunker = TextChunker()
    chunks = chunker.chunk(sample_text)

    build_faiss_index(
        chunks,
        index_path="rag/index.faiss",
        mapping_path="rag/id_to_text.pkl"
    )
