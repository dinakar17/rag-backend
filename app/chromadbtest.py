from chromadb import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


db = Chroma(persist_directory="storage/db", embedding=embeddings)

# insert the documents into the db with id as the filename
db.add_documents(documents, embedding=embeddings, persist_directory="storage/db", id=file_name)