import os
import shutil
import json
import hashlib
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import ollama

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Open a file and return paragraphs
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append(" ".join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append(" ".join(buffer))
        return paragraphs

# Create necessary directories
def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created at: {directory_path}")
    except OSError as e:
        print(f"Error creating directory: {e}")

# Save uploaded files locally
@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        upload_directory = 'uploads'
        create_directory(upload_directory)
        file_path = os.path.join(upload_directory, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": file.filename, "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}

# Load the contents of a file (loading from local file system)
def load_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        print(f"Loaded file from: {file_path}")
        return content
    except IOError as e:
        print(f"Error loading file: {e}")
        return None

# Chunk text into smaller pieces
def chunk_text(text, chunk_size=1000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Store embeddings in ChromaDB
def chromadb_vector_store(embeddings, paragraphs, collection_name='embeddings_collection'):
    try:
        client = chromadb.HttpClient(host='localhost', port=8001)  # ChromaDB port
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        # Add embeddings to collection
        n = len(paragraphs)
        collection.add(
            ids=[str(id) for id in range(n)],
            embeddings=[embedding for embedding in embeddings],
            documents=[paragraph for paragraph in paragraphs],
            metadatas=[{"doc_id": i} for i in range(n)],
            )

        print("Stored embeddings in ChromaDB collection")
        return collection
    except Exception as e:
        print(f"Error storing embeddings in ChromaDB: {e}")
        return None