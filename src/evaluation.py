import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate

from dataset import load_dataset

MODEL_PATH = "result/model_with_terms_split"
MODEL_NAME = "t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH = "results/metrics_with_terms_split.json"


def main():
    print("Loading test data...")

    _, test_dataset = load_dataset(
        csv_path="data/raw/arxiv_data.csv",
        sample_size=2000,   
        test_size=0.1
    )

    print("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    print("Running evaluation...")
    for sample in test_dataset:
        prompt = sample["input_text"]

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

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        predictions.append(pred)
        references.append(sample["target_text"])

    print("Computing ROUGE...")
    scores = rouge.compute(
        predictions=predictions,
        references=references
    )

    print("ROUGE scores:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    print(f"Saving metrics to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
