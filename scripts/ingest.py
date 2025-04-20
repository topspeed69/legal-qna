import argparse
import yaml
import logging.config
from pathlib import Path
import json
from typing import List, Dict
import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import Embedder
from app.retrieval import Retriever

def load_config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)

def setup_logging():
    with open("config/logging.yaml") as f:
        logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            last_break = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end)
            )
            if last_break != -1 and last_break > start:
                end = last_break + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return [c for c in chunks if c]  # Filter empty chunks

def load_mappings(mapping_file: Path) -> Dict:
    if mapping_file.exists():
        with open(mapping_file) as f:
            return json.load(f)
    return {}

def process_judgment(file_path: Path, config: Dict, mappings: Dict) -> List[Dict]:
    if file_path.suffix == ".json":
        with open(file_path) as f:
            judgment = json.load(f)
        text = judgment["text"]
        metadata = {
            "title": judgment.get("title", ""),
            "date": judgment.get("date", ""),
            "court": judgment.get("court", "")
        }
    elif file_path.suffix == ".txt":
        with open(file_path) as f:
            text = f.read()
        metadata = {
            "title": file_path.stem,
            "date": "Unknown",
            "court": "Unknown"
        }
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    chunks = chunk_text(
        text,
        config["retrieval"]["chunk_size"],
        config["retrieval"]["chunk_overlap"]
    )

    return [{
        "text": chunk,
        "metadata": {
            "source": str(file_path),
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
    } for i, chunk in enumerate(chunks)]

async def main():
    parser = argparse.ArgumentParser(description="Ingest court judgments into FAISS index")
    parser.add_argument("--input-dir", required=True, help="Directory containing judgment JSON files")
    parser.add_argument("--mapping-file", help="IPC to BNS mapping file", default="data/mappings/ipc_bns_mapping.json")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Load components
    embedder = Embedder(
        model_name=config["embeddings"]["model"],
        device=config["embeddings"]["device"]
    )
    
    retriever = Retriever(
        dimension=config["embeddings"]["dimension"],
        index_type=config["retrieval"]["index_type"],
        data_dir=config["retrieval"]["data_dir"]
    )
    
    # Load IPC → BNS mappings
    mappings = load_mappings(Path(args.mapping_file))
    
    # Process all judgment files
    judgment_files = list(input_dir.glob("**/*.json")) + list(input_dir.glob("**/*.txt"))
    logger.info(f"Found {len(judgment_files)} judgment files to process")
    
    for i in range(0, len(judgment_files), args.batch_size):
        batch_files = judgment_files[i:i + args.batch_size]
        batch_documents = []
        
        for file_path in tqdm(batch_files, desc=f"Processing batch {i//args.batch_size + 1}"):
            try:
                documents = process_judgment(file_path, config, mappings)
                batch_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if batch_documents:
            # Embed all texts in batch
            texts = [doc["text"] for doc in batch_documents]
            vectors = embedder.embed_chunks(texts)
            
            # Add to FAISS index
            await retriever.add_documents(batch_documents, vectors)
            
            logger.info(f"Processed and indexed batch of {len(batch_documents)} chunks")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())