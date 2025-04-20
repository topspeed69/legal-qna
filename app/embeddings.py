from sentence_transformers import SentenceTransformer
import torch
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        logger.info(f"Loading embedding model {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_chunks([text])[0]

    def embed_chunks(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts with batching."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings

    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Convenience method to handle both single and multiple texts."""
        if isinstance(texts, str):
            return self.embed_text(texts)
        return self.embed_chunks(texts)