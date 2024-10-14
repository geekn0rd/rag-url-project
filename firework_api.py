from openai import OpenAI
import os
from dotenv import load_dotenv

CHAT_MODEL_NAME = "accounts/fireworks/models/llama-v3p1-8b-instruct"
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

def get_openai_client():
    load_dotenv()
    try:
        openai_client = OpenAI(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        base_url=os.environ.get("FIREWORKS_API_BASE"),
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    
    return openai_client

def get_openai_embedding(openai_client, text):
    response = openai_client.embeddings.create(
        model=EMBED_MODEL_NAME,
        input=text
        )
    embedding = response.data[0].embedding
    print("==== Generated an embedding... ====")
    return embedding


def generate_response(openai_client, question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

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