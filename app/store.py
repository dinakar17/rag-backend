from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from load_and_split import all_documents
from constants import DB_DIR


model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


# store = LocalFileStore("storage/embeddings")

# cached_embedder = CacheBackedEmbeddings.from_bytes_store(
#     embeddings, store, namespace="embeddings"
# )


db = Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory=DB_DIR)

print("VectorDB created")


'''
Load the db from the persisted directory

db = Chroma.from_directory("./storage/db")

vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                                  embedding_function=embedding_function)
'''