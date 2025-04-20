import re
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CitationFormatter:
    def format(self, document):
        return f"[1] {document}"

class CitationExtractor:
    def __init__(self):
        self.ipc_pattern = r"Section\s+(\d+[A-Z]?(?:\/\d+[A-Z]?)*)\s+(?:of\s+)?(?:the\s+)?IPC"
        self.crpc_pattern = r"Section\s+(\d+[A-Z]?(?:\/\d+[A-Z]?)*)\s+(?:of\s+)?(?:the\s+)?CrPC"
        
    def extract_citations(self, text: str) -> Dict[str, List[str]]:
        citations = {
            'IPC': self._extract_matches(text, self.ipc_pattern),
            'CrPC': self._extract_matches(text, self.crpc_pattern)
        }
        return citations

    def _extract_matches(self, text: str, pattern: str) -> List[str]:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        return list(set(match.group(1) for match in matches))

    def get_section_details(self, code: str, section: str, mapping_file: str) -> Optional[Dict]:
        try:
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
                return mappings.get(section, None)
        except Exception as e:
            logger.error(f"Error loading section details: {e}")
            return None