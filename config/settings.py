import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")

# Model configuration
GROQ_MODEL = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS vectorstore
VECTOR_DB_DIR = "data/vectordb"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
RETRIEVAL_TOP_K = 4

import os

MODEL_TYPE = "groq"
TEMPERATURE = 0.2
HISTORY_FILE = "chat_history.json"
VECTOR_DB_PATH = "data/vectordb"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
