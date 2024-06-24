import os
from pathlib import Path

from constants import DATA_DIR
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent

data_folder_path = f"{DATA_DIR}"

# Get the path of all pdf files present in the data folder
pdf_files = [
    os.path.join(data_folder_path, f)
    for f in os.listdir(data_folder_path)
    if f.endswith(".pdf")
]


pdf_loaders = [PyPDFLoader(pdf_file) for pdf_file in pdf_files]

all_documents = []

for pdf_loader in pdf_loaders:
    print("Loading raw document..." + pdf_loader.file_path)
    raw_documents = pdf_loader.load()

    print("Splitting text...")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    documents = text_splitter.split_documents(raw_documents)
    all_documents.extend(documents)


# https://github.com/langchain-ai/langchain/issues/3016
