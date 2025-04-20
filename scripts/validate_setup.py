import sys
from pathlib import Path
import logging
import yaml
import torch
import faiss
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from app.embeddings import Embedder
from app.retrieval import Retriever
from app.generation import Generator
from app.citation import CitationExtractor

def load_config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)

def setup_logging():
    with open("config/logging.yaml") as f:
        logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)

def validate_components():
    logger = logging.getLogger(__name__)
    config = load_config()
    
    # Test embeddings
    logger.info("Testing embeddings component...")
    try:
        embedder = Embedder(
            model_name=config["embeddings"]["model"],
            device=config["embeddings"]["device"]
        )
        test_text = "This is a test sentence for embeddings."
        embedding = embedder.embed_text(test_text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[-1] == config["embeddings"]["dimension"]
        logger.info("✓ Embeddings component working")
    except Exception as e:
        logger.error(f"✗ Embeddings component failed: {e}")
        return False

    # Test retrieval
    logger.info("Testing retrieval component...")
    try:
        retriever = Retriever(
            dimension=config["embeddings"]["dimension"],
            index_type=config["retrieval"]["index_type"],
            data_dir=config["retrieval"]["data_dir"]
        )
        # Add a test document
        test_docs = [{"text": "Test document", "metadata": {"source": "test"}}]
        test_vectors = embedder.embed_chunks(["Test document"])
        await retriever.add_documents(test_docs, test_vectors)
        
        # Test search
        query_vector = embedder.embed_text("test")
        results = await retriever.search(query_vector.tolist(), limit=1)
        assert len(results) > 0
        logger.info("✓ Retrieval component working")
    except Exception as e:
        logger.error(f"✗ Retrieval component failed: {e}")
        return False

    # Test generation
    logger.info("Testing generation component...")
    try:
        generator = Generator(
            model_name=config["generation"]["model"],
            device=config["generation"]["device"],
            max_new_tokens=config["generation"]["max_new_tokens"]
        )
        test_question = "What is the test about?"
        test_context = [{"text": "This is a test of the QA system."}]
        answer = generator.generate_answer(test_question, test_context)
        assert isinstance(answer, str)
        assert len(answer) > 0
        logger.info("✓ Generation component working")
    except Exception as e:
        logger.error(f"✗ Generation component failed: {e}")
        return False

    # Test citation extraction
    logger.info("Testing citation extraction...")
    try:
        extractor = CitationExtractor()
        test_text = "According to Section 302 of IPC and Section 156 of CrPC..."
        citations = extractor.extract_citations(test_text)
        assert isinstance(citations, dict)
        assert "IPC" in citations
        assert "CrPC" in citations
        logger.info("✓ Citation extraction working")
    except Exception as e:
        logger.error(f"✗ Citation extraction failed: {e}")
        return False

    logger.info("All components validated successfully!")
    return True

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        import asyncio
        asyncio.run(validate_components())
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)