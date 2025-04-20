import unittest
from pathlib import Path
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import Embedder
from app.citation import CitationExtractor

class TestEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = Embedder("law-ai/InLegalBERT", device="cpu")
    
    def test_embed_text(self):
        text = "This is a test sentence"
        embedding = self.embedder.embed_text(text)
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.size(-1), 768)  # InLegalBERT dimension

class TestCitationExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = CitationExtractor()
    
    def test_ipc_extraction(self):
        text = "According to Section 302 of the IPC..."
        citations = self.extractor.extract_citations(text)
        self.assertIn("IPC", citations)
        self.assertIn("302", citations["IPC"])
    
    def test_crpc_extraction(self):
        text = "As per Section 156 of the CrPC..."
        citations = self.extractor.extract_citations(text)
        self.assertIn("CrPC", citations)
        self.assertIn("156", citations["CrPC"])

if __name__ == "__main__":
    unittest.main()