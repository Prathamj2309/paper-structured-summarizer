import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration

from dataset import load_dataset, tokenize_dataset

MODEL_NAME = "t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("Loading dataset...")
    train_dataset, _ = load_dataset(
        csv_path="data/raw/arxiv_data.csv",
        sample_size=10000,
        test_size = 0.1  
    )

    print("Tokenizing dataset...")
    tokenized = tokenize_dataset(train_dataset)
    tokenized.set_format("torch")

    train_loader = DataLoader(
        tokenized,
        batch_size=2,
        shuffle=True
    )

    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

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
    model.save_pretrained("result/model_with_terms_split_t5_base")
    print("Model saved to result/model_with_terms_split_t5_base")


if __name__ == "__main__":
    main()
