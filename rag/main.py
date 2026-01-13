from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rag.retrieval import Retriever
from rag.prompt import build_rag_prompt


def run_rag(query: str):
    retriever = Retriever(
        index_path="rag/index.faiss",
        mapping_path="rag/id_to_text.pkl",
        top_k=3
    )

    retrieved_chunks = retriever.retrieve(query)

    context = "\n".join([text for text, _ in retrieved_chunks])

    prompt = (
        "Task: Explain the concept step by step.\n\n"
        "Background information:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Detailed explanation (use multiple sentences):"
    )


    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


if __name__ == "__main__":
    query = "Why do attention heads improve transformer performance?"
    answer = run_rag(query)

    print("\n=== FINAL ANSWER ===\n")
    print(answer)
