import csv	

print("DATASET.PY IS RUNNING")

def load_dataset(path, max_samples=100):
    """
    Load abstract-title pairs from a CSV file.
    """
    samples = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Safety check: expected columns
        required_cols = {"summaries", "titles"}
        if not required_cols.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns {required_cols}, "
                f"found {reader.fieldnames}"
            )

        count = 0
        for row in reader:
            print("RAW ROW:", row)  # DEBUG

            if count >= max_samples:
                break

            article = row["summaries"].strip()
            title = row["titles"].strip()
            terms = row.get("terms", "").strip()

            # Skip empty rows
            if not article or not title:
                continue

            samples.append(
                {
                    "article": article,
                    "target": title,
                    "terms": terms,
                }
            )

            count += 1

    print(f"Loaded {len(samples)} samples")
    return samples

def inspect_samples(samples, n=3, max_chars=400):
    """
    Print a few samples to verify abstract-title alignment.
    """
    for i in range(min(n, len(samples))):
        print(f"\n--- SAMPLE {i+1} ---")
        print("ABSTRACT:")
        print(samples[i]["article"][:max_chars] + "...")
        print("\nTITLE:")
        print(samples[i]["target"])

if __name__ == "__main__":
    data = load_dataset("data/raw/arxiv_data.csv", max_samples=5)
    inspect_samples(data, n=3)

