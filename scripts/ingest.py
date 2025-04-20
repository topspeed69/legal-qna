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
import torch
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import Embedder
from app.retrieval import Retriever

def load_config():
    """Loads the configuration from the default YAML file."""
    config_path = Path("config/default.yaml")
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file {config_path}: {e}")
        sys.exit(1)

def setup_logging():
    """Sets up logging based on the logging configuration file."""
    logging_config_path = Path("config/logging.yaml")
    if not logging_config_path.exists():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        logging.warning(f"Logging config not found: {logging_config_path}. Basic logging used.")
        return
    try:
        with open(logging_config_path) as f:
            logging_config = yaml.safe_load(f)
            logging.config.dictConfig(logging_config)
    except (yaml.YAMLError, ValueError) as e:
        logging.error(f"Error setting up logging from {logging_config_path}: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        logging.warning("Basic logging used due to logging config error.")

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunks text into smaller pieces."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            last_break = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if last_break != -1 and last_break > start:
                end = last_break + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]

def load_mappings(mapping_file: Path) -> Dict:
    """Loads mapping data from a JSON file."""
    if mapping_file.exists():
        with open(mapping_file) as f:
            return json.load(f)
    return {}

def process_judgment(file_path: Path, config: Dict, mappings: Dict) -> List[Dict]:
    """Processes a single judgment file and chunks its text."""
    if file_path.suffix == ".json":
        with open(file_path) as f:
            judgment = json.load(f)
        text = judgment["text"]
        metadata = {"title": judgment.get("title", ""), "date": judgment.get("date", ""), "court": judgment.get("court", "")}
    elif file_path.suffix == ".txt":
        with open(file_path) as f:
            text = f.read()
        metadata = {"title": file_path.stem, "date": "Unknown", "court": "Unknown"}
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    chunks = chunk_text(text, config["retrieval"]["chunk_size"], config["retrieval"]["chunk_overlap"])
    return [{"text": chunk, "metadata": {"source": str(file_path), **metadata, "chunk_index": i, "total_chunks": len(chunks)}} for i, chunk in enumerate(chunks)]

async def main():
    """Main function to ingest judgments."""
    parser = argparse.ArgumentParser(description="Ingest court judgments into FAISS index")
    parser.add_argument("--input-dir", required=True, help="Directory containing judgment files")
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

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_device_cfg = config["embeddings"].get("device", "auto")
    embedding_device = embedding_device_cfg if embedding_device_cfg != "auto" else default_device
    logger.info(f"Using device '{embedding_device}' for embeddings.")

    embedder = Embedder(model_name=config["embeddings"]["model"], device=embedding_device)

    retrieval_config = config["retrieval"].copy()
    retrieval_config["dimension"] = config["embeddings"]["dimension"]
    retriever = Retriever(embedder=embedder, config=retrieval_config)

    mappings = load_mappings(Path(args.mapping_file))

    judgment_files = list(input_dir.glob("**/*.json")) + list(input_dir.glob("**/*.txt"))
    logger.info(f"Found {len(judgment_files)} judgment files to process")

    for i in range(0, len(judgment_files), args.batch_size):
        batch_files = judgment_files[i:i + args.batch_size]
        batch_documents = []
        for file_path in tqdm(batch_files, desc=f"Batch {i//args.batch_size + 1}"):
            try:
                documents = process_judgment(file_path, config, mappings)
                batch_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        if batch_documents:
            texts = [doc["text"] for doc in batch_documents]
            vectors = embedder.embed_chunks(texts)
            await retriever.add_documents(batch_documents, vectors.tolist())
            logger.info(f"Processed and indexed {len(batch_documents)} chunks")

if __name__ == "__main__":
    asyncio.run(main())