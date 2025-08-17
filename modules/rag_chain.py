from modules.retriever import get_context_retriever
from modules.LLM import get_llm_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
import os
import json
from typing import List


def load_chat_history(session_id: str) -> List:
    """
    Load chat history from memory/session-specific JSON file.
    """
    path = f"memory/{session_id}.json"
    if not os.path.exists(path):
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chat_history = []
    for item in data:
        if item["type"] == "human":
            chat_history.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            chat_history.append(AIMessage(content=item["content"]))
    return chat_history


def save_chat_history(session_id: str, question: str, answer: str) -> None:
    """
    Save new question-answer pair to the session's memory file.
    """
    path = f"memory/{session_id}.json"
    if not os.path.exists("memory"):
        os.makedirs("memory")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append({"type": "human", "content": question})
    data.append({"type": "ai", "content": answer})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_rag_chain(question: str, session_id: str, file_path: str) -> str:
    """
    Core function to run full context-aware RAG chain.
    """
    retriever = get_context_retriever(file_path)
    context_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    llm_chain = get_llm_chain()

    chat_history = load_chat_history(session_id)

    chain_input = {
        "question": question,
        "context": context,
        "chat_history": chat_history
    }

    response = llm_chain.invoke(chain_input, config=RunnableConfig(tags=["RAG"]))

    save_chat_history(session_id, question, response)

    return response