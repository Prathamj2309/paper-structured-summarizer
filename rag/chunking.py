from typing import List
from transformers import AutoTokenizer

class TextChunker:
    def __init__(
        self,
        tokenizer_name: str = "t5-small",
        chunk_size: int = 400,
        overlap: int = 80
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self.chunk_size - self.overlap

        return chunks

if __name__ == "__main__":
    sample_text = """
Transformers improve performance through multi-head self-attention by allowing the model to simultaneously attend to different representation subspaces of the input sequence, rather than compressing all relational information into a single attention operation. In a standard single-head attention mechanism, the model must learn to capture syntactic structure, semantic meaning, positional dependencies, and long-range relationships using one shared attention map, which can become a representational bottleneck as model depth and sequence length increase. Multi-head attention alleviates this limitation by projecting the input embeddings into multiple lower-dimensional subspaces and applying independent attention operations in parallel, enabling different heads to specialize in capturing distinct types of relationships, such as local context, long-distance dependencies, or token-level semantic similarity. This architectural design improves expressiveness while also stabilizing optimization, because gradients are distributed across multiple attention pathways instead of being forced through a single computation graph. As a result, the model is less prone to vanishing gradients, particularly in deep transformer architectures where repeated nonlinear transformations can otherwise weaken gradient signals during backpropagation. Furthermore, multi-head attention interacts synergistically with residual connections and layer normalization, ensuring that information and gradients can flow efficiently through the network even as the number of layers and parameters increases. By maintaining multiple parallel information channels, the model avoids over-reliance on any single attention pattern, which improves robustness, generalization, and convergence speed during training. Empirically, increasing the number of attention heads often leads to better performance on tasks requiring complex reasoning or long-range context, provided that the overall model capacity is balanced and sufficient regularization is applied to prevent overfitting. Consequently, multi-head self-attention is a central mechanism through which transformers achieve both high representational power and stable training dynamics, making it a key factor in their success across natural language processing tasks.
    """ 


    chunker = TextChunker()
    chunks = chunker.chunk(sample_text)

    print("Chunks:", len(chunks))
    print("\n--- Chunk 1 ---\n", chunks[0])
    print("\n--- Chunk 2 ---\n", chunks[1])

