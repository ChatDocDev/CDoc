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
    
# Generate a hash from an input string
def generate_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()

# Hash map to keep track of file metadata
file_hash_map = {}

# Add file metadata to the hash map
def add_to_hash_map(file_name):
    file_hash_map[file_name] = generate_hash(file_name)

# Load and save embeddings using JSON files
def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

# # Find cosine similarity of every chunk to a given embedding
# def find_most_similar(needle, haystack):
#     needle_norm = norm(needle)
#     similarity_scores = [
#         np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
#     ]
#     return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# Get a chat response for a question
def get_chat_response(question, collection):
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"]
    results = collection.query(
        query_embeddings=[prompt_embedding],
        n_results=5
    )
    # print(results)
    most_similar_chunks = results.get("documents", [])[0]
    # print([chunk['text'] for sublist in most_similar_chunks for chunk in sublist])
    print(most_similar_chunks)
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join([chunk for chunk in most_similar_chunks]),
            },
            {"role": "user", "content": question},
        ],
    )
    return response["message"]["content"]

# FastAPI endpoint to handle file upload and embedding
@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    try:
        upload_response = await upload_file(file)
        file_path = upload_response["file_path"]
        content = load_file(file_path)
        chunks = chunk_text(content)
        embeddings = get_embeddings(file.filename, "nomic-embed-text", chunks)
        chromadb_vector_store(embeddings, chunks)
        add_to_hash_map(file.filename)
        return {"message": "File processed and embeddings stored successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# FastAPI endpoint to handle retrieval question answering
@app.post("/ask-question/")
async def ask_question(question: str):
    try:
        if not file_hash_map:
            return JSONResponse(status_code=400, content={"error": "No file uploaded"})

        latest_file = max(file_hash_map, key=file_hash_map.get)
        client = chromadb.HttpClient(host='localhost', port=8001)
        collection = client.get_collection(name='embeddings_collection')

        response = get_chat_response(question, collection)
        return {"response": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)  # FastAPI will run on this address
