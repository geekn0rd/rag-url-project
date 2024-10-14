import re
import logging
from typing import List, Dict
from chunking_algorithms import *
from firework_api import get_openai_embedding

# Set up logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def clean_text(text):
    # Remove unwanted characters
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_documents(documents: List[str], openai_client, chunking_algorithm: str = "fixed_size", question="") -> List[Dict]:
    """Process documents by chunking and generating embeddings."""
    if chunking_algorithm == "fix":
        docs = fixed_size_chunking(documents)
    elif chunking_algorithm == "semantic":
        docs = semantic_chunking(documents)
    elif chunking_algorithm == "qa":
        docs = qa_chunking(documents, question)
    else:
        raise ValueError(f"Unsupported chunking algorithm: {chunking_algorithm}")

    logger.info(f"Using {chunking_algorithm} chunking algorithm.")
    logger.info("Generating embeddings...")
    for doc in docs:
        doc["embedding"] = get_openai_embedding(openai_client, doc["text"])
    logger.info("Embeddings generation completed.")
    return docs
