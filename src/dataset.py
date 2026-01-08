# src/dataset.py
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer

MODEL_NAME = "t5-small"


def load_dataset(csv_path="data/raw/arxiv_data.csv", sample_size=None):
    """
    Load and prepare dataset for title generation.

    Returns a Hugging Face Dataset with:
    - input_text
    - target_text
    """
    df = pd.read_csv(csv_path)

    required_cols = {"summaries", "titles", "terms"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, found {df.columns}"
        )

    df = df[["summaries", "titles", "terms"]].dropna()
    
    def build_input(row):
        return f"title generation [TERMS: {row['terms']}]: {row['summaries']}"

    df["input_text"] = df.apply(build_input, axis=1)
    df["target_text"] = df["titles"]

    df = df[["input_text", "target_text"]]

    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)

    return Dataset.from_pandas(df)


def tokenize_dataset(dataset, max_input_len=512, max_target_len=64):
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=max_input_len
        )
        targets = tokenizer(
            batch["target_text"],
            truncation=True,
            padding="max_length",
            max_length=max_target_len
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )
