# LegalMind

AI-powered tool to query legal documents

## Overview
LegalMind lets you upload legal PDFs, splits them into smart chunks, embeds them as vectors, and enables context-aware search & Q&A through an LLM. All conversations are saved for easy, persistent legal research.

## Features
- PDF upload and parsing  
- Document chunking (loader.py)  
- Embedding and vectorization (embedding.py)  
- Context-aware retrieval (retriever.py)  
- LLM integration & prompt customization (prompt.py, llm.py)  
- Persistent chat history (llm_chain.py)

## Technologies Used
- Python
- LangChain
- HuggingFace Transformers / OpenAI
- FAISS / Chroma

## Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/legalmind.git
   cd legalmind
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**  
   (Replace with your actual run command)
   ```bash
   python app.py
   ```

## Usage
- Upload your PDF legal document  
- Ask questions in chat  
- Get answers with citations and supporting excerpts  
- All chat history is saved for each document

## Example
```
Q: What is the compliance requirement for section 3B?
A: Section 3B requires annual reporting as outlined in page 12.
```

## License
Apache 2.0

***
