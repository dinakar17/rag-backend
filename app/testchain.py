import logging

from operator import itemgetter
from typing import List, Dict, Optional, Sequence
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import LanguageModelLike
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
    chain,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from decouple import config
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class ChatRequest(BaseModel):
    input: str
    chat_history: Optional[List[Dict[str, str]]]
    
    
def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history

def get_retriever() -> BaseRetriever:
    logging.info("Loading retriever...")
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    
    db = Chroma(persist_directory="./storage/db", embedding_function=embedding_function)
    
    logging.info("Retriever loaded")
    
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 6})


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


# chat_history and question are the keys in the request body
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""



def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    ).with_config(run_name="FindCDocs")
    
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | rag_chain 
    )
    
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True, openai_api_key=config("OPENAI_API_KEY"))

retriever = get_retriever()


answer_chain = create_chain(llm, retriever)