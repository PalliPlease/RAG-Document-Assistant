## RAG Document Assistant
This project is a local AI-powered Question–Answering system built to help students quickly find information from a college handbook PDF.
It uses Retrieval-Augmented Generation (RAG) to search the document and generate accurate answers using a local LLM.

The project consists of:

An ingest pipeline to load and index the document

A RAG chain that connects the retriever to an LLM

A Streamlit interface where users can ask questions

No external APIs are required — everything runs locally using Ollama.
