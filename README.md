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
   ```
   git clone https://github.com/yourusername/rag-url-project.git
   cd rag-url-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   FIREWORKS_API_KEY=your_fireworks_api_key
   FIREWORKS_API_BASE=your_fireworks_api_base_url
   HF_API_KEY=your_huggingface_api_key
   ```

## 🚀 Usage

To use the RAG system, run the `rag_tool.py` script with the following parameters: