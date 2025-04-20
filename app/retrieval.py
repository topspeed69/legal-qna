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
        self.metadata_path = self.data_dir / "metadata.h5"
        
        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded existing FAISS index from {self.index_path}")
            with h5py.File(self.metadata_path, 'r') as f:
                self.documents = json.loads(f['metadata'][()])
        else:
            if config["index_type"] == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unsupported index type: {config['index_type']}")
            self.documents = []
            logger.info(f"Created new FAISS index of type {config['index_type']}")

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        # Ensure embeddings are computed on the correct device
        query_vector = self.embedder.embed_text(query).to(self.device).cpu().numpy().reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_vector, limit)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # FAISS returns -1 for empty slots
                doc = self.documents[idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                })
        return results

    async def add_documents(self, documents: List[Dict[str, Any]], vectors: List[List[float]]):
        vectors_array = np.array(vectors).astype('float32')
        start_idx = len(self.documents)
        
        # Add to FAISS index
        self.index.add(vectors_array)
        
        # Store documents metadata
        self.documents.extend(documents)
        
        # Save index and metadata
        self._save_state()
        logger.info(f"Added {len(documents)} documents to FAISS index")

    def _save_state(self):
        """Save both the FAISS index and document metadata"""
        faiss.write_index(self.index, str(self.index_path))
        with h5py.File(self.metadata_path, 'w') as f:
            f.create_dataset('metadata', data=json.dumps(self.documents))
        logger.info("Saved FAISS index and metadata to disk")