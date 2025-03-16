# Advanced RAG System and AI-Powered Data Visualizer

This repository contains two separate applications: an advanced Retrieval-Augmented Generation (RAG) system and an AI-powered data visualizer. Both applications leverage large language models (LLMs) and various AI technologies to provide powerful document analysis and data visualization capabilities.

## 1. Advanced RAG System

### Overview
The RAG system is a Streamlit-based application that allows users to upload documents, process them, and ask questions about their content. It uses a combination of document parsing, vector storage, and multi-modal language models to provide accurate and context-aware responses.

### Key Features
- Document upload and processing (PDF, DOCX, XLSX, CSV)
- Multi-modal content extraction (text and images)
- Vector-based indexing for efficient retrieval
- Multi-query retrieval for improved accuracy
- Integration with Gemini and Llama language models

### Technologies Used
- Streamlit: Web application framework
- LlamaParse: Document parsing
- Google's Generative AI (Gemini): Multi-modal language model
- LlamaIndex: Vector indexing and retrieval
- Langchain: LLM orchestration
- Ollama: Local LLM integration

### How to Run
To run the RAG system, use the following command:
```
streamlit run streamlit.py
```

## 2. AI-Powered Data Visualizer

### Overview
The AI-powered data visualizer is a Dash application that uses natural language processing to generate Plotly graphs based on user requests. It loads a space mission dataset and allows users to create visualizations by describing what they want to see.

### Key Features
- Interactive data grid display
- Natural language input for graph generation
- Dynamic Plotly graph creation based on user requests
- Integration with Llama language model for code generation

### Technologies Used
- Dash: Web application framework
- Plotly: Data visualization library
- Pandas: Data manipulation and analysis
- Langchain: LLM orchestration
- Ollama: Local LLM integration

### How to Run
To run the AI-powered data visualizer, simply execute the Python file:
```
python Plotly_Visuala.py
```
Then, follow the link provided in the console to access the application in your web browser.

## Setup and Dependencies

Both applications require various Python libraries and external dependencies. Make sure to install the required packages using:

```
pip install -r requirements.txt
```

Additionally, ensure that you have the necessary API keys (Google Gemini 1.5 Flash model and llama cloud api key.) and environment variables set up for the language models and other services used in these applications.

## Usage

1. For the RAG system, upload a document and ask questions about its content. The system will process the document and provide answers based on the extracted information.

2. For the data visualizer, enter a natural language request describing the graph you want to create. The system will generate the appropriate Plotly code and display the resulting visualization.

## Note

These applications demonstrate advanced AI capabilities in document analysis and data visualization. They showcase the power of combining large language models with specialized libraries for tasks such as information retrieval and graph generation.
