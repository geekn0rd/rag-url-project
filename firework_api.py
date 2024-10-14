import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

CHAT_MODEL_NAME = "accounts/fireworks/models/llama-v3p1-8b-instruct"
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_openai_client():
    load_dotenv()
    try:
        openai_client = OpenAI(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url=os.environ.get("FIREWORKS_API_BASE"),
        )
        return openai_client
    except Exception as e:
        logger.error(f"An error occurred while creating OpenAI client: {e}")
        raise

def get_openai_embedding(openai_client, text):
    try:
        response = openai_client.embeddings.create(
            model=EMBED_MODEL_NAME,
            input=text
        )
        embedding = response.data[0].embedding
        logger.debug("Generated an embedding")
        return embedding
    except Exception as e:
        logger.error(f"An error occurred while generating embedding: {e}")
        raise

def generate_response(openai_client, question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        logger.error(f"An error occurred while generating response: {e}")
        raise