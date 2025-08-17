import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

from config.settings import (
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_PATH,
    TEMPERATURE,
    HISTORY_FILE,
)
from modules.prompt import document_qa_prompt
from utils.helpers import get_file_hash


def load_vector_db(file_path):
    file_hash = get_file_hash(file_path)
    db_path = os.path.join(VECTOR_DB_PATH, file_hash)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector DB not found at {db_path}")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"Loaded vector DB from: {db_path}")
    return vectordb


def load_llm():
    return ChatGroq(
        temperature=TEMPERATURE,
        model_name="llama3-70b-8192",
    )


def build_conversational_retriever(file_path):
    vectordb = load_vector_db(file_path)

    if not os.path.exists("history"):
        os.makedirs("history")

    file_hash = get_file_hash(file_path)
    session_path = os.path.join("history", f"{file_hash}.json")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=FileChatMessageHistory(session_path)
    )

    llm = load_llm()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": document_qa_prompt}
    )

    print(f"Conversational Retriever ready with persistent memory at: {session_path}")
    return qa_chain


def get_context_retriever(file_path=None):
    """
    Returns a retriever object from the vector DB.
    If no file_path is given, it loads the default vectorstore.
    """
    if file_path is None:
        raise ValueError("File path must be provided to retrieve context.")

    vectordb = load_vector_db(file_path)
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})