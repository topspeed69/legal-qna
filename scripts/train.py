import argparse
import yaml
import logging.config
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import numpy as np
from tqdm import tqdm

def load_config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)

def setup_logging():
    with open("config/logging.yaml") as f:
        logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)

class LegalQADataset(Dataset):
    def __init__(self, data_files, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_examples(data_files)
        
    def _load_examples(self, data_files):
        examples = []
        for file in data_files:
            with open(file) as f:
                data = json.load(f)
                for item in data:
                    context = item.get("context", "")
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    
                    prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}
Answer: {answer}"""
                    examples.append(prompt)
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.examples[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0]
        }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the generation model")
    parser.add_argument("--train-data", required=True, help="Directory containing training data")
    parser.add_argument("--output-dir", required=True, help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    
    # Load model and tokenizer
    model_name = config["generation"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Prepare dataset
    train_files = list(Path(args.train_data).glob("*.json"))
    dataset = LegalQADataset(train_files, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()