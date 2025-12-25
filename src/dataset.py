"""
Dataset loader for contribution extraction from scientific abstracts.

Task:
Input  : abstract / summary text
Target : paper title (proxy for contribution)
"""

import csv


def load_dataset(path, max_samples=100):
    """
    Load abstract-title pairs from a CSV file.

    Args:
        path (str): path to CSV file
        max_samples (int): maximum number of samples to load

    Returns:
        list of dicts with keys:
            - article: abstract text
            - target: title text
    """
    samples = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Safety check: expected columns
        required_cols = {"article", "title"}
        if not required_cols.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns {required_cols}, "
                f"found {reader.fieldnames}"
            )

        for i, row in enumerate(reader):
            if i >= max_samples:
                break

            article = row["article"].strip()
            title = row["title"].strip()

            # Skip empty rows
            if not article or not title:
                continue

            samples.append(
                {
                    "article": article,
                    "target": title,
                }
            )

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

