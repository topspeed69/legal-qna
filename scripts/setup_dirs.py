from pathlib import Path
import shutil
import logging
import logging.config
import yaml

def load_config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)

def setup_logging():
    with open("config/logging.yaml") as f:
        logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    
    # Create required directories
    dirs = [
        Path("data/judgments"),  # Raw judgment files
        Path("data/mappings"),   # IPC/CRPC mappings
        Path(config["retrieval"]["data_dir"]),  # FAISS indices and metadata
        Path("data/processed/training"),  # Processed training data
        Path("models"),  # Fine-tuned models
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    # Copy mapping files if they exist in root
    root_dir = Path(__file__).parent.parent.parent
    mapping_files = [
        ("ipc_bns_mapping.json", "data/mappings/ipc_bns_mapping.json"),
        ("crpc_bnss_mapping.json", "data/mappings/crpc_bns_mapping.json")
    ]
    
    for src_name, dest_path in mapping_files:
        src_path = root_dir / src_name
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            logger.info(f"Copied mapping file: {src_path} -> {dest_path}")

if __name__ == "__main__":
    main()