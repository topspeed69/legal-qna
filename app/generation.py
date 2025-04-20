from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name: str, device: str = "cpu", max_new_tokens: int = 256):
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"Loading generation model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16
        )
        self.model.eval()

    def generate_answer(self, question: str, contexts: List[Dict[str, str]]) -> str:
        # Create prompt with context
        prompt = self._create_prompt(question, contexts)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Leave room for generation
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract answer
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_answer(response)

    def _create_prompt(self, question: str, contexts: List[Dict[str, str]]) -> str:
        # Format contexts with their sources
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            text = ctx.get("text", "").strip()
            source = ctx.get("metadata", {}).get("source", "Unknown")
            formatted_contexts.append(f"Context {i} (Source: {source}):\n{text}")
        
        contexts_str = "\n\n".join(formatted_contexts)
        
        return f"""Based on the following contexts from legal documents, provide a comprehensive answer to the question. Include relevant citations and references.

{contexts_str}

Question: {question}
Answer: Let me analyze the provided context and answer your question."""

    def _extract_answer(self, response: str) -> str:
        """Extract the generated answer from the full response."""
        # Split on "Answer:" and take the last part
        parts = response.split("Answer:")
        if len(parts) > 1:
            return parts[-1].strip()
        return response.strip()  # Fallback to full response if no split possible