from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from modules.prompt import document_qa_prompt
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.2,
)


rag_chain = (
    RunnableMap({
        "context": lambda input: input["context"],
        "question": lambda input: input["question"],
        "chat_history": lambda input: input.get("chat_history", [])
    })
    | document_qa_prompt
    | llm
    | StrOutputParser()
)


def get_context_aware_chain(get_session_history: callable) -> RunnableWithMessageHistory:
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
def get_llm_chain():
    return rag_chain