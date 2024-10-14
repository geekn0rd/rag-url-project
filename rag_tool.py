import os
import logging

from serp_api import *
from content_preprocessor import process_documents
from firework_api import *

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_vector_database():
    """Create and persist the vector database."""
    try:
        huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=os.environ.get("HF_API_KEY"),
            model_name="nomic-ai/nomic-embed-text-v1"
        )
        db_client = chromadb.PersistentClient(path="./vdatabase/")
        return db_client, huggingface_ef
    except Exception as e:
        logger.error(f"An error occurred while creating the vector database: {e}")
        raise

def do_rag(question, url, chunking_algorithm="fix", k=2):
    # Create vector database
    db_client, huggingface_ef = create_vector_database()
    fix_chunk_collection = db_client.get_or_create_collection(
        name="fix_chunk_collections", embedding_function=huggingface_ef
    )

    # Scrape content
    documents = get_content_from_url(url)

    # Process documents
    openai_client = get_openai_client()
    docs = process_documents(documents, openai_client, chunking_algorithm, question)
    
    # Insert data into the database
    fix_chunk_collection.upsert(
        documents=[doc["text"] for doc in docs],
        ids=[doc["id"] for doc in docs],
        embeddings=[doc["embedding"] for doc in docs]
    )

    # Retrieve relevant documents based on the query
    relevant_docs = fix_chunk_collection.query(
        query_embeddings=[get_openai_embedding(openai_client, question)],
        n_results=k
    )["documents"][0]

    # Generate and print response
    response = generate_response(openai_client, question, relevant_docs)
    print(f"Question: {question}")
    print(f"Response: {response}")

    # Count chunks
    chunks_count = len(docs)

    return {
        "answer": response,
        "chunks_count": chunks_count
    }

if __name__ == "__main__":
    result = do_rag(
        question="Who was steve job's wife?",
        url="https://en.wikipedia.org/wiki/Steve_Jobs",
        chunking_algorithm="semantic"
    )
    # result = do_rag(
    #     question="What are the health benefits of physical activity?",
    #     url="https://www.cdc.gov/physical-activity-basics/benefits/?CDC_AAref_Val=https://www.cdc.gov/physicalactivity/basics/pa-health/index.html",
    # )
