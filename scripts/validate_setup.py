import sys
import asyncio
from pathlib import Path
import logging
import logging.config
import yaml
import torch
import faiss
import numpy as np

# Add the project root to the sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Assuming these modules exist in your project structure (e.g., app/embeddings.py)
try:
    from app.embeddings import Embedder
    from app.retrieval import Retriever
    from app.generation import Generator
    from app.citation import CitationExtractor
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Please ensure app/embeddings.py, app/retrieval.py, etc. exist")
    print("and the project root is correctly added to sys.path.")
    sys.exit(1)

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

async def validate_components():
    """Validates the main components: embeddings, retrieval, generation, citation."""
    logger = logging.getLogger(__name__)
    config = load_config()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"CUDA available: {torch.cuda.is_available()}. Default device: {default_device}")

    # --- Test embeddings ---
    logger.info("Testing embeddings component...")
    try:
        embedding_device_cfg = config["embeddings"].get("device", "auto")
        embedding_device = embedding_device_cfg if embedding_device_cfg != "auto" else default_device
        logger.info(f"Using device '{embedding_device}' for embeddings.")
        embedder = Embedder(model_name=config["embeddings"]["model"], device=embedding_device)
        test_text = "This is a test sentence."
        embedding = embedder.embed_text(test_text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[-1] == config["embeddings"]["dimension"]
        logger.info("✓ Embeddings component working")
    except Exception as e:
        logger.error(f"✗ Embeddings component failed: {e}")
        return False

    # --- Test retrieval ---
    logger.info("Testing retrieval component...")
    retriever = None
    try:
        retrieval_config = config["retrieval"].copy()
        retrieval_config["dimension"] = config["embeddings"]["dimension"]
        logger.info(f"Initializing Retriever with config: {retrieval_config}")
        retriever = Retriever(embedder=embedder, config=retrieval_config)
        test_docs = [{"text": "Test document.", "metadata": {"source": "test"}}]
        test_vectors = embedder.embed_chunks([d["text"] for d in test_docs])
        await retriever.add_documents(test_docs, test_vectors.tolist())
        results = await retriever.search("search query", limit=1)
        assert len(results) > 0 and 'text' in results[0] and 'score' in results[0]
        logger.info("✓ Retrieval component working")
    except Exception as e:
        logger.error(f"✗ Retrieval component failed: {e}")
        return False
    finally:
         if retriever is not None and hasattr(retriever, 'cleanup_test_index'):
              retriever.cleanup_test_index()

    # --- Test generation ---
    logger.info("Testing generation component...")
    try:
        generation_device_cfg = config["generation"].get("device", "auto")
        generation_device = generation_device_cfg if generation_device_cfg != "auto" else default_device
        logger.info(f"Using device '{generation_device}' for generation.")
        generator = Generator(
            model_name=config["generation"]["model"], device=generation_device,
            max_new_tokens=config["generation"]["max_new_tokens"],
            temperature=config["generation"].get("temperature", 0.7)
        )
        test_question = "Summarize."
        test_context = [{"text": "This is a test of generation."}]
        answer = generator.generate_answer(test_question, test_context)
        assert isinstance(answer, str) and len(answer) > 0
        logger.info("✓ Generation component working")
    except Exception as e:
        logger.error(f"✗ Generation component failed: {e}")
        return False

    # --- Test citation extraction ---
    logger.info("Testing citation extraction...")
    try:
        extractor = CitationExtractor()
        test_text = "Section 302 IPC and Section 156 CrPC."
        citations = extractor.extract_citations(test_text)
        assert isinstance(citations, dict) and "IPC" in citations and "CrPC" in citations
        assert '302' in citations.get('IPC', []) and '156' in citations.get('CrPC', [])
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
        success = asyncio.run(validate_components())
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during validation: {e}")
        sys.exit(1)