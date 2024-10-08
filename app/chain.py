from operator import itemgetter
from typing import Dict, List, Optional, Sequence

from decouple import config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

from .constants import DB_DIR
from .engine.embeddings import get_embeddings_model
from .engine.prompts import IMPROVED_QA_PROMPT, OPTIMIZED_CONDENSE_PROMPT

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
    question: str
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


# WEAVIATE_URL = os.environ["WEAVIATE_URL"]
# WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

# def get_weaviate_retriever() -> BaseRetriever:
#     weaviate_client = weaviate.Client(
#         url=WEAVIATE_URL,
#         auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
#     )
#     weaviate_client = Weaviate(
#         client=weaviate_client,
#         index_name=WEAVIATE_DOCS_INDEX_NAME,
#         text_key="text",
#         embedding=get_embeddings_model(),
#         by_text=False,
#         attributes=["source", "title"],
#     )
#     return weaviate_client.as_retriever(search_kwargs=dict(k=6))


def get_retriever() -> BaseRetriever:
    embedding_function = get_embeddings_model(choice="bge")

    db = Chroma(persist_directory=f"{DB_DIR}", embedding_function=embedding_function)

    return db.as_retriever(search_type="similarity", search_kwargs={"k": 6})


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


# If there is just a question return the retriever by passing the question to it
# If there is question + chat history, extract the question from the chat history and pass it
# to the retriever
def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(OPTIMIZED_CONDENSE_PROMPT)
    condense_question_chain = (
        # chain functions together, where the output of one function becomes the input for the next.
        # (introduced in 3.10)
        CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    ).with_config(
        # For debugging purposes, we can assign a name to the chain
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever

    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
        # the choice of path depends on whether the input has a chat history.
    ).with_config(run_name="RouteDependingOnChatHistory")


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    # A retriever equipped with a question to it i.e., question | retriever
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")

    # Apply transformations to the retrieved documents
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )

    # A prompt template that will be used to generate the response
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", IMPROVED_QA_PROMPT),
            # Place holder for the chat history
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    default_response_synthesizer = prompt | llm

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="openai_gpt_3_5_turbo",
        )
        | StrOutputParser()
    ).with_config(run_name="GenerateResponse")
    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )


gpt_3_5 = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0,
    streaming=True,
    openai_api_key=config("OPENAI_API_KEY"),
)

llm = gpt_3_5.configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="openai_gpt_3_5_turbo",
).with_fallbacks([gpt_3_5])

retriever = get_retriever()


answer_chain = create_chain(llm, retriever)
