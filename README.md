# RAG Knowledge Assistant on Databricks

This repository demonstrates how to build a **Retrieval-Augmented Generation (RAG)** knowledge assistant using **PySpark** and **Databricks**.  
The workflow includes:
- Parsing documents
- Chunking text
- Generating embeddings
- Creating a vector search index
- Querying with LLMs

---

## 🚀 Architecture Overview
1. **Document Ingestion**  
   Load raw documents (PDF, text, HTML, etc.) into Databricks.

2. **Parsing & Cleaning**  
   Extract text, remove noise, normalize formatting.

3. **Chunking**  
   Split text into manageable chunks (e.g., 500–1000 tokens) for embedding.

4. **Embeddings**  
   Generate vector representations using an embedding model (e.g., OpenAI, HuggingFace).

5. **Vector Search Index**  
   Store embeddings in Databricks Vector Search for fast retrieval.

6. **RAG Query Flow**  
   - User query → Embed query  
   - Retrieve top-k relevant chunks  
   - Pass retrieved context + query to LLM  
   - Generate augmented answer

---

## 📦 Requirements
- Databricks Runtime ML (latest version)
- PySpark
- Databricks Vector Search
- Langchain
- Python 3.9+

