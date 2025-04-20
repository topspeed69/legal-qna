from fastapi import FastAPI, HTTPException
from .schema import QuestionRequest, AnswerResponse
from .embeddings import Embedder
from .retrieval import Retriever
from .generation import Generator
from .citation import CitationExtractor
import yaml
import logging.config
from pathlib import Path

# Load configuration
with open("config/default.yaml") as f:
    config = yaml.safe_load(f)

with open("config/logging.yaml") as f:
    logging_config = yaml.safe_load(f)
    logging.config.dictConfig(logging_config)

logger = logging.getLogger(__name__)

app = FastAPI(title="Legal QA System")

# Initialize components
embedder = Embedder(
    model_name=config["embeddings"]["model"],
    device=config["embeddings"]["device"]
)

retriever = Retriever(
    dimension=config["embeddings"]["dimension"],
    index_type=config["retrieval"]["index_type"],
    data_dir=config["retrieval"]["data_dir"]
)

generator = Generator(
    model_name=config["generation"]["model"],
    device=config["generation"]["device"],
    max_new_tokens=config["generation"]["max_new_tokens"]
)

citation_extractor = CitationExtractor()

@app.post("/api/query", response_model=AnswerResponse)
async def query(request: QuestionRequest):
    try:
        # Embed the question
        question_embedding = embedder.embed_text(request.question)
        
        # Retrieve relevant contexts
        contexts = await retriever.search(
            query_vector=question_embedding.tolist(),
            limit=config["retrieval"]["top_k"]
        )
        
        # Generate answer
        answer = generator.generate_answer(request.question, contexts)
        
        # Extract citations
        citations = citation_extractor.extract_citations(answer)
        
        # Format response with sources
        return AnswerResponse(
            answer=answer,
            sources=[ctx["metadata"] for ctx in contexts]
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))