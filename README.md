🧠 VectorDB — Built from Scratch in Python
A fully functional Vector Database with three search algorithms, real-time visualization, and a RAG (Retrieval Augmented Generation) pipeline — all built from scratch in pure Python with zero external database dependencies.
�
�
�
�
Load image
Load image
Load image
📌 What is this?
Most developers use vector databases as a black box. This project builds one from scratch — every algorithm, every data structure, every distance metric is hand-coded in Python. It's a learning tool, a demo, and a fully working system at the same time.
You can:
Insert vectors and search for nearest neighbors in real time
Compare 3 different search algorithms side by side
Upload documents and ask an AI questions about them (RAG)
Visualize your entire vector space in 2D using PCA
✨ Features
3 Search Algorithms — HNSW, KD-Tree, and Brute Force, all implemented from scratch
3 Distance Metrics — Cosine, Euclidean, and Manhattan
Real-time Benchmarking — compare algorithm speeds on the same query
RAG Pipeline — embed documents with Ollama and query them with a local LLM
2D Visualization — PCA scatter plot of your entire vector space, live in the browser
REST API — full FastAPI backend with Swagger docs at /docs
Zero external DB — no Pinecone, no Weaviate, no ChromaDB — pure Python
🏗️ Architecture
Code
🔍 Search Algorithms
Brute Force
The simplest approach — compute the distance from the query to every stored vector and return the closest K. Guaranteed to be correct but O(n) per query. Gets slow with large datasets.
KD-Tree
A binary tree that recursively splits the vector space along alternating dimensions. Much faster than brute force for low-dimensional data. Performance degrades in high dimensions (the "curse of dimensionality").
HNSW — Hierarchical Navigable Small World
The algorithm powering production vector databases like Pinecone, Weaviate, and ChromaDB. Builds a multi-layer graph:
Upper layers → sparse, long-range connections (fast navigation)
Lower layers → dense, short-range connections (precise search)
Search starts at the top layer and greedily descends, getting closer to the query at each level. Achieves near-linear search time even with millions of vectors.
📐 Distance Metrics
Metric
Formula
Best For
Cosine
1 - (A·B / |A||B|)
Text, semantic similarity
Euclidean
√Σ(aᵢ - bᵢ)²
Spatial data, images
Manhattan
Σ|aᵢ - bᵢ|
Sparse vectors, grid data
🤖 RAG Pipeline
RAG (Retrieval Augmented Generation) lets you ask an AI questions about your own documents.
Code
🚀 Getting Started
Prerequisites
Python 3.10+
Ollama (optional, for RAG features)
Installation
Bash
Run
Bash
Open your browser at http://localhost:8080
📡 API Reference
Method
Endpoint
Description
GET
/
Serves the UI
GET
/items
List all vectors
GET
/search
Search nearest neighbors
POST
/insert
Insert a new vector
DELETE
/delete/{id}
Delete a vector
GET
/benchmark
Compare all algorithms
GET
/hnsw-info
HNSW graph statistics
GET
/status
Ollama status
POST
/doc/insert
Embed and store a document
GET
/doc/list
List stored document chunks
DELETE
/doc/delete/{id}
Delete a document chunk
POST
/doc/ask
Ask a question (RAG)
Full interactive docs available at http://localhost:8080/docs
Example: Search
Bash
Example: Insert
Bash
🗂️ Project Structure
Code
Everything lives in two files — no frameworks, no build steps, no configuration.
🧩 Tech Stack
Layer
Technology
Backend
Python, FastAPI, Uvicorn
Algorithms
Pure Python (no NumPy)
Embeddings
Ollama — nomic-embed-text (768D)
LLM
Ollama — llama3.2
Frontend
Vanilla HTML/CSS/JS
Visualization
HTML Canvas API + PCA
Data Validation
Pydantic
💡 Key Concepts Explained
Why vectors? Any data — text, images, audio — can be converted into a list of numbers that captures its meaning. Things with similar meanings get similar numbers. This lets us do "semantic search" — finding things by meaning, not just keywords.
Why HNSW? Brute force search is exact but slow. Tree-based methods are fast but break down in high dimensions. HNSW gets the best of both worlds using a probabilistic graph structure that navigates efficiently even in hundreds of dimensions.
Why RAG? Large language models have a knowledge cutoff and don't know about your private data. RAG solves this by retrieving relevant context from your own documents and feeding it to the LLM as part of the prompt. The LLM then answers based on your data, not just its training.
📊 Demo Vectors
The project ships with 20 pre-loaded 16-dimensional demo vectors across 4 categories — CS/Algorithms, Mathematics, Food, and Sports — so you can start searching immediately without inserting any data.
🛣️ Roadmap
[ ] Persistent storage (save/load vectors to disk)
[ ] Product Quantization for memory compression
[ ] Multi-vector search (batch queries)
[ ] Vector namespace support
[ ] Docker compose setup
📄 License
MIT License — feel free to use, modify, and distribute.
🙏 Acknowledgements
HNSW Paper — Malkov & Yashunin, 2016
Ollama — for making local LLMs accessible
FastAPI — for the excellent Python web framework
