import os
import hashlib
import shutil
import logging
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


from .chain import ChatRequest, answer_chain
from .constants import DATA_DIR, DB_DIR, BASE_DIR
from .engine.embeddings import bge_embeddings

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(
    app,
    answer_chain,
    path="/chat",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.post("/upload-files/")
async def create_upload_files(file: UploadFile = File(...)):
    folder_path = f"{DATA_DIR}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a unique temporary file path
    temp_file_path = os.path.join(folder_path, f"temp_{uuid.uuid4()}_{file.filename}")
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)

    # Calculate hash of the uploaded file
    with open(temp_file_path, "rb") as f:
        content = f.read()
        file_hash = hashlib.md5(content).hexdigest()
    
    logging.info(f"File hash for {file.filename}: {file_hash}")

    # Check for duplicates
    for existing_file in os.listdir(folder_path):
        existing_file_path = os.path.join(folder_path, existing_file)
        if os.path.isfile(existing_file_path) and not existing_file.startswith("temp_"):
            with open(existing_file_path, "rb") as ef:
                existing_content = ef.read()
                existing_hash = hashlib.md5(existing_content).hexdigest()
                logging.info(f"Comparing with {existing_file}, hash {existing_hash}")
                if file_hash == existing_hash:
                    os.remove(temp_file_path)
                    raise HTTPException(status_code=400, detail="Duplicate file detected.")

    # Rename the temporary file to its final name
    final_file_path = os.path.join(folder_path, file.filename)
    os.rename(temp_file_path, final_file_path)
    return {"filename": file.filename}

@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    # Create the full path to the file
    file_location = DATA_DIR / file_path

    # Check if the file exists and is in the 'data' directory
    if file_location.exists() and file_location.is_file() and BASE_DIR in file_location.parents:
        return FileResponse(file_location)
    else:
        # If the file does not exist, return a 404 error
        raise HTTPException(status_code=404, detail="File not found")


@app.post("/embed/")
async def embed_file(file_name: str):
    # Create the full path to the file
    file_location = DATA_DIR / file_name
    
    # Check if the file exists and is in the 'data' directory
    if file_location.exists() and file_location.is_file() and BASE_DIR in file_location.parents:
        try:
            loader = PyPDFLoader(file_location)
            raw_document = loader.load()
            
            text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            )
        
            documents = text_splitter.split_documents(raw_document)
            
        
            db = Chroma(persist_directory=f"{DB_DIR}", embedding_function=bge_embeddings)


            # check if the file is already embedded
            for i in range(len(db._collection.get()['ids'])):
                metadata = db._collection.get()['metadatas'][i]
                if file_name in metadata['source']:
                    raise HTTPException(status_code=400, detail="File already embedded")

            before_count = db._collection.count()
            
            # insert the documents into the db with random ids
            db.add_documents(documents)
        except Exception as e:
            print(f"Error embedding the file: {e}")
            raise HTTPException(status_code=500, detail="Error embedding the file")
    else:
        # If the file does not exist, return a 404 error
        raise HTTPException(status_code=404, detail="File not found")
    
    return {"filename": file_name, "embeddings_before": before_count, "embeddings_after": db._collection.count()}



@app.delete("/files/{file_path:path}")
async def delete_file(file_path: str):
    # Create the full path to the file
    file_location = DATA_DIR / file_path

    # Check if the file exists and is in the 'data' directory
    if file_location.exists() and file_location.is_file() and DATA_DIR in file_location.parents:
        try:
            db = Chroma(persist_directory=f"{DB_DIR}", embedding_function=bge_embeddings)
            collection = db._collection.get()
            
            embeddings_count = db._collection.count()

            ids_to_delete = []

            for i in range(len(collection['ids'])):

                id = collection['ids'][i]
                metadata = collection['metadatas'][i]

                if file_path in metadata['source']:
                    ids_to_delete.append(id)
                    
            db._collection.delete(ids=ids_to_delete)
            
            os.remove(file_location)
            
            # return the number of embeddings before and after the deletion along with the status
            return {"status": "success", "before": embeddings_count, "after": db._collection.count()}
        except Exception as e:
            # log the error and return a 500 error
            logging.error(f"Error deleting the file: {e}")
            raise HTTPException(status_code=500, detail="Error deleting the file")
    else:
        # If the file does not exist, return a 404 error
        raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# poetry run langchain serve --port=8100

