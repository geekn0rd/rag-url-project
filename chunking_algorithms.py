import spacy
from sentence_transformers import SentenceTransformer, util
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fixed_size_chunking(text, chunk_size=500):
    words = text.split()
    chunks = [{"text": ' '.join(words[i:i + chunk_size]), "id": f"id_{i // chunk_size}"} for i in range(0, len(words), chunk_size)]
    logger.info(f"Created {len(chunks)} chunks using fixed-size chunking.")
    return chunks

def semantic_chunking(text, max_chunk_length=500):
    # Load English tokenizer, POS tagger, parser, NER, and word vectors
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    chunks = []

    current_chunk = []
    current_length = 0

    for sent in doc.sents:
        sent_length = len(sent.text.split())
        if current_length + sent_length > max_chunk_length and current_chunk:
            # Join the current chunk and add it to the chunks list
            chunk_text = ' '.join([s.text for s in current_chunk])
            chunks.append({"text": chunk_text, "id": f"chunk_{len(chunks)}"})
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sent)
        current_length += sent_length

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunk_text = ' '.join([s.text for s in current_chunk])
        chunks.append({"text": chunk_text, "id": f"chunk_{len(chunks)}"})

    logger.info(f"Created {len(chunks)} chunks using semantic chunking.")
    return chunks

def qa_chunking(text, query, threshold=0.4):
    # Load a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Split into paragraphs
    chunks = [chunk["text"] for chunk in fixed_size_chunking(text)]
    
    # Handle empty or whitespace-only text
    if not chunks or all(chunk.strip() == '' for chunk in chunks):
        logger.warning("Empty or whitespace-only text provided. Returning entire text as a single chunk.")
        return [{"text": text, "id": "id0"}]  # Return the entire text as a single chunk

    # Encode the query and chunks
    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(query_embedding, chunk_embeddings)

    # Filter relevant chunks based on the threshold
    relevant_chunks = [{"text": chunks[i], "id": f"id{i}"} 
                       for i in range(len(chunks)) if similarities[0][i] > threshold]

    # If no chunks meet the threshold, return the most similar chunk
    if not relevant_chunks:
        logger.warning("No chunks met the similarity threshold. Returning the most similar chunk.")
        most_similar_index = similarities[0].argmax().item()
        relevant_chunks = [{"text": chunks[most_similar_index], "id": f"id{most_similar_index}"}]

    logger.info(f"Found {len(relevant_chunks)} relevant chunks using semantic similarity chunking.")
    return relevant_chunks
