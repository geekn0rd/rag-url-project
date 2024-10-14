import unittest
from rag_tool import do_rag
import time
import logging
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from serp_api import get_content_from_url

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestChunkingAlgorithms(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            {
                "url": "https://en.wikipedia.org/wiki/Climate_change",
                "question": "What are the impacts of climate change?",
            },
            {
                "url": "https://www.cdc.gov/physicalactivity/basics/pa-health/index.html",
                "question": "What are the health benefits of physical activity?",
            },
            # Add more test cases as needed
        ]
        self.chunking_algorithms = ["fix", "semantic", "qa"]

    def evaluate_chunking(self, algorithm):
        total_chunks = 0
        total_time = 0
        accuracy_score = 0
        context_retention_score = 0

        for case in self.test_cases:
            start_time = time.time()
            response = do_rag(case["question"], case["url"], chunking_algorithm=algorithm)
            end_time = time.time()

            # Count chunks (this would need to be implemented in do_rag() to return)
            chunks_count = response.get("chunks_count", 0)
            total_chunks += chunks_count

            # Measure time
            processing_time = end_time - start_time
            total_time += processing_time

            # Evaluate accuracy (this would need manual scoring or automated evaluation)
            accuracy_score += self.evaluate_accuracy(response["answer"], case["question"])

            # Evaluate context retention
            context_retention_score += self.evaluate_context_retention(response["answer"], case["url"])

        avg_chunks = total_chunks / len(self.test_cases)
        avg_time = total_time / len(self.test_cases)
        avg_accuracy = accuracy_score / len(self.test_cases)
        avg_context_retention = context_retention_score / len(self.test_cases)

        return {
            "algorithm": algorithm,
            "avg_chunks": avg_chunks,
            "avg_time": avg_time,
            "avg_accuracy": avg_accuracy,
            "avg_context_retention": avg_context_retention,
        }

    def evaluate_accuracy(self, answer, question):
        # Use a pre-trained question-answering model for automated evaluation
        qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

        # Tokenize the question and answer
        inputs = tokenizer(question, answer, return_tensors="pt")
        
        # Get the model's prediction
        with torch.no_grad():
            outputs = qa_model(**inputs)
        
        # Calculate the confidence score
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        confidence = torch.max(start_scores) + torch.max(end_scores)
        
        # Normalize the confidence score to be between 0 and 1
        normalized_score = torch.sigmoid(confidence).item()
        
        return normalized_score

    def evaluate_context_retention(self, answer, url):
        # Fetch the original content
        original_content = get_content_from_url(url)
        
        # Use TF-IDF to extract key phrases from the original content
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([original_content])
        
        # Get the top 10 key phrases
        feature_names = vectorizer.get_feature_names_out()
        sorted_items = self.sort_coo(tfidf_matrix.tocoo())
        key_phrases = [feature_names[idx] for score, idx in sorted_items[:10]]
        
        # Check how many key phrases are in the answer
        retained_phrases = sum(1 for phrase in key_phrases if phrase.lower() in answer.lower())
        
        # Calculate the retention score
        retention_score = retained_phrases / len(key_phrases)
        
        return retention_score

    @staticmethod
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.data, coo_matrix.col)
        return sorted(tuples, key=lambda x: (x[0], x[1]), reverse=True)

    def test_chunking_algorithms(self):
        results = []
        for algorithm in self.chunking_algorithms:
            result = self.evaluate_chunking(algorithm)
            results.append(result)
            logger.info(f"Results for {algorithm}: {result}")

        # Select the optimal algorithm based on a weighted score
        optimal_algorithm = max(results, key=lambda x: 
            0.4 * x["avg_accuracy"] + 
            0.3 * x["avg_context_retention"] + 
            0.2 * (1 / x["avg_time"]) +  # Lower time is better
            0.1 * (1 / x["avg_chunks"])  # Lower chunk count is generally better
        )

        logger.info(f"Optimal chunking algorithm: {optimal_algorithm['algorithm']}")

if __name__ == "__main__":
    unittest.main()
