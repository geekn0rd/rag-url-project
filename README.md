# 🚀 RAG URL Project

## 📖 Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions based on content from specified URLs. It combines web scraping, document processing, vector database storage, and natural language processing to provide accurate and context-aware responses to user queries.

## ✨ Features

- 🕷️ Web scraping to extract content from given URLs
- 📄 Document processing with various chunking algorithms (fixed-size, semantic, and question-aware)
- 💾 Vector database storage using Chroma DB
- 🧠 Embedding generation using Hugging Face models
- 💬 Question answering using OpenAI's API (via Fireworks AI)
- 🧪 Comprehensive testing suite for evaluating chunking algorithms

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/geekn0rd/rag-url-project.git
   cd rag-url-project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```plaintext
   FIREWORKS_API_KEY=your_fireworks_api_key
   FIREWORKS_API_BASE=your_fireworks_api_base_url
   HF_API_KEY=your_huggingface_api_key
   ```

## 🚀 Usage

To use the RAG system, run the `rag_tool.py` script with the following parameters:

```bash
python rag_tool.py --url <URL> --question <QUESTION>
```

Replace `<URL>` with the URL of the page you want to query and `<QUESTION>` with the question you want to ask.

## 🧪 Testing

To run the testing suite, execute:

```bash
python testing_suite.py
```

This will evaluate the performance of each chunking algorithm and select the optimal one based on a weighted score.

## ⚠️ Important Note

As a beginner working on this project, please be aware that this implementation may not follow best practices and could have potential issues. It's a great opportunity to learn, but you might encounter challenges related to:

- **Code Quality:** The code may not be optimized or structured in the most efficient way.
- **Error Handling:** There may be limited error handling in place.
- **Performance:** The system may not be fully optimized for speed or scalability.

Feel free to experiment and learn from any issues that arise! Happy coding!
