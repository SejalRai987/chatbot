-This project implements a lightweight assistant that combines Retrieval-Augmented Generation (RAG) with a simple agentic workflow to answer user queries from a small set of documents.

-Architecture Overview
>Data Ingestion
Loads 3–5 .txt files from the local directory and splits them into smaller chunks using LangChain’s RecursiveCharacterTextSplitter.

>Vector Store & Retrieval

Embeds document chunks using all-MiniLM-L6-v2 (SentenceTransformerEmbeddings).

Stores them in a FAISS vector index.

Retrieves top 3 relevant chunks per query.

>LLM Integration
Uses Hugging Face’s flan-t5-base model via transformers.pipeline and LangChain's QA chain.

>Agentic Workflow
Based on keyword detection, the query is routed to:

A calculator tool for math-related expressions

A dictionary tool for definitions (mocked)

Or defaults to the RAG + LLM pipeline

>CLI Interface
A simple command-line interface where users can:

Ask questions

See which tool was used

View retrieved document chunks

Get the final answer

