import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration

from dataset import load_dataset, tokenize_dataset

MODEL_NAME = "t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("Loading dataset...")
    dataset = load_dataset(
        csv_path="data/raw/arxiv_data.csv",
        sample_size=500  # small baseline
    )

    print("Tokenizing dataset...")
    tokenized = tokenize_dataset(dataset)
    tokenized.set_format("torch")

    train_loader = DataLoader(
        tokenized,
        batch_size=4,
        shuffle=True
    )

    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    print("Starting training...")
    model.train()

    for epoch in range(3):
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Avg loss: {avg_loss:.4f}")

    print("Saving model...")
    model.save_pretrained("result/model")
    print("Model saved to result/model")


if __name__ == "__main__":
    main()
