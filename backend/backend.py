import os
import shutil
import json
import hashlib
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import ollama
from typing import Generator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
import docx

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to create necessary directories
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
        
        # Save the uploaded file to the specified directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": file.filename, "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        text = "\n".join(page.page_content for page in pages)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    try:
        loader = TextLoader(txt_path)
        documents = loader.load()
        text = "\n".join(doc.page_content for doc in documents)
        return text
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

# Function to extract text from a CSV file
def extract_text_from_csv(csv_path):
    try:
        loader = CSVLoader(file_path=csv_path)
        data = loader.load()
        text = "\n".join([str(record) for record in data])
        return text
    except Exception as e:
        print(f"Error extracting text from CSV: {e}")
        return ""

# Function to extract text from an XLSX file
def extract_text_from_xlsx(xlsx_path):
    try:
        df = pd.read_excel(xlsx_path)
        text = df.to_string(index=False)
        return text
    except Exception as e:
        print(f"Error extracting text from XLSX: {e}")
        return ""

# Function to get file extension
def get_file_extension(file_path):
    try:
        return os.path.splitext(file_path)[1].lower()
    except Exception as e:
        print(f"Error getting file extension: {e}")
        return ""

# Function to extract text based on file type
def extract_text(file_path):
    try:
        ext = get_file_extension(file_path)
        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return extract_text_from_docx(file_path)
        elif ext == '.txt':
            return extract_text_from_txt(file_path)
        elif ext == '.csv':
            return extract_text_from_csv(file_path)
        elif ext == '.xlsx':
            return extract_text_from_xlsx(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=1000, chunk_overlap=50):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        doc_output = splitter.split_documents([Document(page_content=text)])
        print(doc_output)
        # Convert the Document objects to a list of strings
        result = [doc.page_content for doc in doc_output]
        print(result)
        return result
    except Exception as e:
        print(f"Error chunking text: {e}")
        return []

# Store embeddings in ChromaDB
def chromadb_vector_store(embeddings, paragraphs, collection_name):
    try:
        client = chromadb.HttpClient(host='localhost', port=8001)  # ChromaDB port
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})

        # Add embeddings to the collection
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
    try:
        hash_object = hashlib.sha256(input_string.encode())
        return hash_object.hexdigest()
    except Exception as e:
        print(f"Error generating hash: {e}")
        return ""

# Hash map to keep track of file metadata
file_hash_map = {}

# Add file metadata to the hash map
def add_to_hash_map(file_name):
    try:
        file_hash_map[file_name] = generate_hash(file_name)
    except Exception as e:
        print(f"Error adding to hash map: {e}")

# Load and save embeddings using JSON files
def save_embeddings(filename, embeddings):
    try:
        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")
        with open(f"embeddings/{filename}.json", "w") as f:
            json.dump(embeddings, f)
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def load_embeddings(filename):
    try:
        if not os.path.exists(f"embeddings/{filename}.json"):
            return False
        with open(f"embeddings/{filename}.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return False

# Function to get or generate embeddings
def get_embeddings(filename, modelname, chunks):
    try:
        if (embeddings := load_embeddings(filename)) is not False:
            return embeddings
        embeddings = [
            ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
            for chunk in chunks
        ]
        save_embeddings(filename, embeddings)
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []

# Function to generate chat responses
def get_chat_response(question, collection) -> Generator[str, None, None]:
    try:
        SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
            based on snippets of text provided in context. Answer only using the context provided, 
            being as concise as possible. If you're unsure, just say that you don't know.
            Context:
        """
        prompt_embedding = ollama.embeddings(
            model="nomic-embed-text", prompt=question)["embedding"]
        results = collection.query(
            query_embeddings=[prompt_embedding],
            n_results=5
        )
        print(results)
        most_similar_chunks = results.get("documents", [])[0]
        print(most_similar_chunks)
        print(results.get("distances", [])[0])
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
            stream=True
        )

        for chunk in response:
            print(chunk["message"]["content"])
            yield chunk["message"]["content"]
    except Exception as e:
        print(f"Error generating chat response: {e}")
        yield ""

# FastAPI endpoint to handle file upload and embedding
@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    try:
        upload_response = await upload_file(file)
        file_path = upload_response["file_path"]
        content = extract_text(file_path)
        chunks = chunk_text(content)
        embeddings = get_embeddings(file.filename, "nomic-embed-text", chunks)
        chromadb_vector_store(embeddings, chunks, collection_name=file.filename)
        add_to_hash_map(file.filename)
        return {"message": "File processed and embeddings stored successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# FastAPI endpoint to ask questions based on the document
@app.post("/ask-question/")
async def ask_question(file_name: str, question: str):
    try:
        if not file_hash_map:
            return JSONResponse(status_code=400, content={"error": "No file uploaded"})
        if file_name not in file_hash_map:
            return JSONResponse(status_code=400, content={"error": "File not found"})
        client = chromadb.HttpClient(host='localhost', port=8001)
        collection = client.get_collection(name=file_name)
        # StreamingResponse to stream the response
        return StreamingResponse(get_chat_response(question, collection), media_type="text/plain")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)  # FastAPI will run on this address
