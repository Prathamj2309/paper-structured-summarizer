from dataset import load_dataset, tokenize_dataset

ds = load_dataset(sample_size=5)
tokenized = tokenize_dataset(ds)

print(tokenized[0])
