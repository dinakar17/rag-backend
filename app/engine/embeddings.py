from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


def get_embeddings_model(choice: str = "bge"):
    if choice == "bge":
        return bge_embeddings
    else:
        raise ValueError(f"Unknown embeddings model: {choice}")
