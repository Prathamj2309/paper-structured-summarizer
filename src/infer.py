import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "result/model"
MODEL_NAME = "t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_title(abstract: str) -> str:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

    prompt = "title generation: " + abstract

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=30,
            num_beams=4
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    sample_abstract = (
        "We propose a transformer-based approach for understanding long documents "
        "using attention mechanisms."
    )

    print("Generated title:")
    print(generate_title(sample_abstract))
