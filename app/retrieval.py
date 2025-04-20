import faiss
import numpy as np
import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import h5py
import torch

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, embedder, config):
        self.embedder = embedder
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dimension = config["dimension"]
        self.data_dir = Path(config["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "faiss_index.bin"
        self.metadata_path = self.data_dir / "metadata.jsonl"
        
        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded existing FAISS index from {self.index_path}")
            self.num_documents = self._count_metadata_lines()
        else:
            if config["index_type"] == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unsupported index type: {config['index_type']}")
            self.num_documents = 0
            logger.info(f"Created new FAISS index of type {config['index_type']}")

    def _count_metadata_lines(self):
        if not os.path.exists(self.metadata_path):
            return 0
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _append_metadata(self, documents):
        with open(self.metadata_path, 'a', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')

    def _get_metadata_by_indices(self, indices):
        # Only load required lines
        results = []
        if not os.path.exists(self.metadata_path):
            return results
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in indices:
                    results.append(json.loads(line))
                if len(results) == len(indices):
                    break
        return results

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_text(query).to(self.device).cpu().numpy().reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_vector, limit)
        valid_indices = [idx for idx in indices[0] if idx != -1]
        metadata = self._get_metadata_by_indices(valid_indices)
        results = []
        for idx, score, doc in zip(valid_indices, scores[0], metadata):
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score)
            })
        return results

    async def add_documents(self, documents: List[Dict[str, Any]], vectors: List[List[float]]):
        vectors_array = np.array(vectors).astype('float32')
        self.index.add(vectors_array)
        self._append_metadata(documents)
        self.num_documents += len(documents)
        self._save_state()
        logger.info(f"Added {len(documents)} documents to FAISS index")

    def _save_state(self):
        faiss.write_index(self.index, str(self.index_path))
        logger.info("Saved FAISS index to disk (metadata is in JSONL file)")