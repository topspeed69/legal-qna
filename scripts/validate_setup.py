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
        # Fallback to basic configuration if logging.yaml is not found
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        logging.warning(f"Logging config file not found: {logging_config_path}. Using basic logging.")
        return
    try:
        with open(logging_config_path) as f:
            logging_config = yaml.safe_load(f)
            logging.config.dictConfig(logging_config)
    except (yaml.YAMLError, ValueError) as e:
        logging.error(f"Error setting up logging from {logging_config_path}: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        logging.warning("Using basic logging due to error in logging config.")


async def validate_components():
    """Validates the main components: embeddings, retrieval, generation, citation."""
    logger = logging.getLogger(__name__)
    config = load_config()

    # Determine the default device based on CUDA availability
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"CUDA available: {torch.cuda.is_available()}. Default device: {default_device}")

    # --- Test embeddings ---
    logger.info("Testing embeddings component...")
    try:
        # Resolve device for embeddings
        embedding_device_cfg = config["embeddings"].get("device", "auto")
        embedding_device = embedding_device_cfg if embedding_device_cfg != "auto" else default_device
        logger.info(f"Using device '{embedding_device}' for embeddings.")

        embedder = Embedder(
            model_name=config["embeddings"]["model"],
            device=embedding_device
        )
        test_text = "This is a test sentence for embeddings."
        # Ensure embed_text or similar returns numpy array and handles device correctly
        embedding = embedder.embed_text(test_text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[-1] == config["embeddings"]["dimension"]
        logger.info("✓ Embeddings component working")
    except Exception as e:
        logger.error(f"✗ Embeddings component failed: {e}")
        return False

    # --- Test retrieval ---
    logger.info("Testing retrieval component...")
    # Retrieval often depends on the embedder and potentially async operations
    try:
        # Resolve device for retrieval (Note: FAISS doesn't have a global device,
        # index is built on CPU or GPU depending on faiss-gpu/faiss-cpu build)
        # This config setting might be advisory or for loading/saving logic.
        retrieval_device_cfg = config["retrieval"].get("device", "auto")
        retrieval_device = retrieval_device_cfg if retrieval_device_cfg != "auto" else default_device
        logger.info(f"Using device '{retrieval_device}' consideration for retrieval (FAISS may differ).")


        retriever = Retriever(
            dimension=config["embeddings"]["dimension"],
            index_type=config["retrieval"]["index_type"],
            data_dir=config["retrieval"]["data_dir"]
            # Device might be passed here or handled internally by Retriever
            # device=retrieval_device # Uncomment if your Retriever needs this
        )

        # Add a test document - Requires async if Retriever.add_documents is async
        test_docs = [{"text": "Test document for retrieval testing.", "metadata": {"source": "test"}}]
        # Embed chunks needs embedder, ensure it handles batches and device
        test_vectors = embedder.embed_chunks([d["text"] for d in test_docs])
        # Assuming add_documents is async based on your original code
        await retriever.add_documents(test_docs, test_vectors)

        # Test search - Requires async if Retriever.search is async
        query_vector = embedder.embed_text("search query") # Use a different query text
        # Assuming search is async and expects a list/tensor convertible to list
        results = await retriever.search(query_vector.tolist(), limit=1)

        assert len(results) > 0
        assert 'text' in results[0] and 'score' in results[0] # Basic check for result structure
        logger.info("✓ Retrieval component working")
    except Exception as e:
        logger.error(f"✗ Retrieval component failed: {e}")
        # Cleanup test index files if any were created
        retriever.cleanup_test_index() # Assuming Retriever has a cleanup method
        return False
    finally:
         # Ensure cleanup happens even if search fails
         if 'retriever' in locals() and hasattr(retriever, 'cleanup_test_index'):
              retriever.cleanup_test_index()


    # --- Test generation ---
    logger.info("Testing generation component...")
    try:
        # Resolve device for generation
        generation_device_cfg = config["generation"].get("device", "auto")
        generation_device = generation_device_cfg if generation_device_cfg != "auto" else default_device
        logger.info(f"Using device '{generation_device}' for generation.")

        generator = Generator(
            model_name=config["generation"]["model"],
            device=generation_device,
            max_new_tokens=config["generation"]["max_new_tokens"],
            temperature=config["generation"].get("temperature", 0.7) # Use get with default
        )
        test_question = "Summarize the test context."
        test_context = [{"text": "This is a test of the QA system's generation capabilities. It should be able to process this context and produce a relevant summary or answer based on a question."}]
        answer = generator.generate_answer(test_question, test_context) # Assuming this is sync

        assert isinstance(answer, str)
        assert len(answer) > 10 # Basic check for non-empty answer
        logger.info("✓ Generation component working")
    except Exception as e:
        logger.error(f"✗ Generation component failed: {e}")
        return False

    # --- Test citation extraction ---
    logger.info("Testing citation extraction...")
    try:
        extractor = CitationExtractor()
        test_text = "According to Section 302 of IPC and Section 156 of CrPC..."
        citations = extractor.extract_citations(test_text)

        assert isinstance(citations, dict)
        assert "IPC" in citations
        assert "CrPC" in citations
        assert '302' in citations.get('IPC', []) # Check for specific extracted sections
        assert '156' in citations.get('CrPC', [])
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
        # asyncio.run is suitable for running the top-level async function
        success = asyncio.run(validate_components())
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during validation: {e}")
        sys.exit(1)