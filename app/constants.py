from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent # `app` directory
DB_DIR = f"{BASE_DIR}/storage/db"
DATA_DIR = f"{BASE_DIR}/storage/data"
WEAVIATE_DOCS_INDEX_NAME = "LangChain_Combined_Docs_OpenAI_text_embedding_3_small"