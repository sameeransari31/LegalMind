import os
import hashlib
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from config.settings import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from utils.helpers import get_file_hash, clean_text


def split_text_with_metadata(text, file_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": file_name,
            "chunk_id": i,
        }
        documents.append(Document(page_content=chunk, metadata=metadata))

    return documents


def embed_and_store(text, original_filepath):
    file_hash = get_file_hash(original_filepath)
    db_path = f"data/vectordb/{file_hash}"

    if os.path.exists(db_path):
        print(f"Embedding already exists at {db_path} — skipping.")
        return db_path

    print("Splitting text into chunks...")
    documents = split_text_with_metadata(text, os.path.basename(original_filepath))

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Embedding chunks & building vectorstore...")
    vectordb = FAISS.from_documents(documents, embedding=embedding_model)

    os.makedirs("data/vectordb", exist_ok=True)
    vectordb.save_local(db_path)

    print(f"Vectorstore saved at: {db_path}")
    return db_path