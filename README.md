# Hobbit Multimodal RAG

A multimodal Retrieval-Augmented Generation (RAG) system built on *The Hobbit* by J.R.R. Tolkien.

The project combines textual knowledge from the original book with illustrations from visual editions to provide context-aware answers and retrieve relevant images for user queries.


## Demo
[🎥 Watch Demo](https://github.com/user-attachments/assets/c5f9e024-3602-4439-88ce-c51c5b7eee1d)

## Features

* Text-based RAG pipeline using the original *The Hobbit* book
* Image retrieval from illustrated editions
* Text extraction from visual pages
* Multimodal retrieval combining text and image context
* Gemini-powered question answering
* Streamlit chat interface
* Persistent vector storage with ChromaDB
* Query expansion using Multi-Query Retrieval

## Architecture

```text
User Question
      │
      ├── Text Retriever (ChromaDB)
      │
      ├── Image Retriever (ChromaDB)
      │
      └── Similarity Filtering
                │
                ▼
         Retrieved Context
                │
                ▼
          Gemini LLM
                │
                ▼
        Answer + Images
```

## Dataset

The knowledge base is built from two sources:

1. The original text of *The Hobbit*
2. Illustrated editions containing visual depictions of scenes, characters, and locations

During ingestion:

* texts extracted from illustrated pages
* Image descriptions are generated
* Text chunks are embedded and stored in ChromaDB
* Image metadata is indexed for retrieval

## Tech Stack

* Python 3.11+
* Streamlit
* LangChain
* ChromaDB
* Google Gemini API
* Ollama
* Poetry

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hosein541/hobbit-multimodal-rag.git

cd hobbit-multimodal-rag
```

### 2. Install Poetry

Follow the official installation guide:

https://python-poetry.org/docs/#installation

Verify the installation:

```bash
poetry --version
```

### 3. Install dependencies

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

Alternatively, run commands with:

```bash
poetry run <command>
```

## Environment Variables


Get your Gemini API key from:

https://aistudio.google.com/app/apikey

## Project Structure

```text
.
├── app.py
├── data/
│   ├── images/
│   ├── metadata/
│   └── TheHobbit.pdf
├── pipelines/
|   └── chat.py
├── chains/
│   ├── image_ingestion.py
│   ├── text_ingestion.py
│   └── ingestion.py
├── vectore_db/
├── pyproject.toml
├── poetry.lock
├── config.py
└── README.md
```

## Running the Application

Start the Streamlit app:

```bash
poetry run streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

## Example Questions

* How did Bilbo acquire the One Ring, and why was it important for his escape from the goblins?
* How does Bilbo change from the beginning of the story to the end?
* What major events occurred between the company's arrival at Lake-town and the Battle of Five Armies?
* What prompted Gandalf to choose Bilbo as the burglar for Thorin's company?

## Future Improvements

* Hybrid search (BM25 + vector search)
* Cross-encoder reranking
* Native image embeddings with CLIP or SigLIP
* Conversation memory
* Evaluation pipeline with RAGAS
* Support for additional Tolkien books

## Disclaimer

This project is intended for research and educational purposes only.

All rights to *The Hobbit* and its associated illustrations belong to their respective copyright holders.
