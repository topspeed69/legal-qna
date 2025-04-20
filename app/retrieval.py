import faiss
import numpy as np
import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import h5py
import torch
import sqlite3

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
        self.sqlite_path = self.data_dir / "metadata.sqlite3"
        
        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded existing FAISS index from {self.index_path}")
        else:
            if config["index_type"] == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unsupported index type: {config['index_type']}")
            logger.info(f"Created new FAISS index of type {config['index_type']}")
        self._init_sqlite()

    def _init_sqlite(self):
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def _insert_metadata(self, documents):
        with self.conn:
            self.conn.executemany(
                "INSERT INTO metadata (text, metadata) VALUES (?, ?)",
                [(doc["text"], json.dumps(doc["metadata"])) for doc in documents]
            )

    def _get_metadata_by_indices(self, indices):
        # indices are FAISS row ids (0-based)
        if not indices:
            return []
        q = f"SELECT text, metadata FROM metadata WHERE id IN ({','.join(['?']*len(indices))}) ORDER BY id"
        cur = self.conn.execute(q, [i+1 for i in indices])  # SQLite AUTOINCREMENT starts at 1
        return [{"text": row[0], "metadata": json.loads(row[1])} for row in cur.fetchall()]

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
        self._insert_metadata(documents)
        self._save_state()
        logger.info(f"Added {len(documents)} documents to FAISS index")

    def _save_state(self):
        faiss.write_index(self.index, str(self.index_path))
        logger.info("Saved FAISS index to disk (metadata is in SQLite DB)")