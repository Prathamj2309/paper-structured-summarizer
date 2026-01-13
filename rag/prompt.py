from typing import List, Tuple


def build_rag_prompt(
    query: str,
    retrieved_chunks: List[Tuple[str, float]],
    max_chunks: int = 3
) -> str:
    context_texts = [
        text.strip() for text, _ in retrieved_chunks[:max_chunks]
    ]

    context = " ".join(context_texts)

    prompt = (
        f"question: {query} "
        f"context: {context} "
        f"answer: explain in detail"
    )

    return prompt

if __name__ == "__main__":
    from rag.retrieval import Retriever

    retriever = Retriever(
        index_path="rag/index.faiss",
        mapping_path="rag/id_to_text.pkl",
        top_k=3
    )

    query = "Why do attention heads improve transformer performance?"
    retrieved = retriever.retrieve(query)

    prompt = build_rag_prompt(query, retrieved)
    print(prompt)

