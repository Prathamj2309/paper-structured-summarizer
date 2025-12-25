"""
Preprocessing utilities for academic paper summarization.
"""

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - remove excessive whitespace
    - normalize newlines
    """
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def format_input(text: str) -> str:
    """
    Format input text for the summarization model.
    """
    return f"Summarize the contribution of the following paper:\n{text}"


if __name__ == "__main__":
    sample_text = """
    This paper proposes a new attention mechanism that reduces
    computational complexity for long documents.
    """
    cleaned = clean_text(sample_text)
    formatted = format_input(cleaned)
    print(formatted)
