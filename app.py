import os
import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from modules.loader import load_document
from modules.embedder import embed_and_store
from modules.rag_chain import run_rag_chain
from modules.rag_chain import save_chat_history, load_chat_history


CHAT_HISTORY_DIR = "history"


chat_histories = {}
uploaded_file_path = None


def handle_upload(file):
    global uploaded_file_path

    if not file:
        return "Please upload a file first."

    uploaded_file_path = file.name
    text = load_document(file.name)
    embed_and_store(text, file.name)
    return "Document uploaded and indexed successfully."

from langchain_core.messages import AIMessage, HumanMessage

def chat_with_doc(user_input, chat_id):
    """
    Handles user queries via context-aware RAG
    """
    if not user_input:
        return []

    history = chat_histories.get(chat_id, [])
    
    answer = run_rag_chain(question=user_input, session_id=chat_id, file_path=uploaded_file_path)

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))
    chat_histories[chat_id] = history

    save_chat_history(chat_id, user_input, answer)

    formatted_history = []
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            human_msg = history[i].content
            ai_msg = history[i + 1].content
            formatted_history.append({"role": "user", "content": human_msg})
            formatted_history.append({"role": "assistant", "content": ai_msg})

    return formatted_history


def load_history(chat_id):
    """
    Load chat history if it exists (in 'messages' format)
    """
    history = load_chat_history(chat_id)
    chat_histories[chat_id] = history

    formatted_history = []
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            human_msg = history[i].content
            ai_msg = history[i + 1].content
            formatted_history.append({"role": "user", "content": human_msg})
            formatted_history.append({"role": "assistant", "content": ai_msg})

    return formatted_history


def clear_chat(chat_id):
    chat_histories[chat_id] = []
    save_chat_history(chat_id, "", "")
    return []


# Gradio UI
with gr.Blocks(title="Smart RAG Agent") as demo:
    gr.Markdown("# Smart RAG Agent")
    gr.Markdown("Upload a document, ask questions, and get context-aware answers.")

    chat_id = gr.Textbox(value="default_user", label="Session ID (for history)", visible=False)

    with gr.Row():
        file_upload = gr.File(label="Upload Document")
        upload_status = gr.Textbox(label="Status", interactive=False)

    file_upload.change(fn=handle_upload, inputs=file_upload, outputs=upload_status)

    chatbot = gr.Chatbot(label="Chat with Document", type="messages")
    query = gr.Textbox(label="Ask a question")

    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear Chat")

    send_btn.click(fn=chat_with_doc, inputs=[query, chat_id], outputs=chatbot)
    clear_btn.click(fn=clear_chat, inputs=chat_id, outputs=chatbot)

    demo.load(fn=load_history, inputs=chat_id, outputs=chatbot)


if __name__ == "__main__":
    demo.launch(debug=True)
