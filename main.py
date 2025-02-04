from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException
from fastapi.security import APIKeyHeader
import pdfminer.high_level
import os
import time
from dotenv import load_dotenv
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests
import ollama

# Load environment variables from the .env file
load_dotenv()

# FastAPI app setup
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for saving the uploaded PDF files
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# API Key setup from .env file
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-KEY" 
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# Dependency to verify the API Key
def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Verify API key in the request headers.
    """
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Initialize embedding model and vector store for document processing
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(collection_name="docs", embedding_function=embedding_model)

# Minimal preprocessing to avoid unnecessary token usage and chunk splitting
def preprocess_text(text: str) -> str:
    """
    Preprocess the text by removing unwanted newlines and extra spaces.
    Minimal changes are made to avoid altering the structure and affecting chunking.
    """
    text = text.strip()  
    return text

# Combined endpoint for uploading and processing the file
@app.post("/upload/")  
async def upload_and_process_file(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """
    Upload and process the PDF file into chunks, then store the chunks in the vector store.
    This endpoint does not interact with Ollama and is only responsible for document storage.
    """
    # Save the uploaded file to the directory
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())  

    # Extract text from the uploaded PDF
    text = pdfminer.high_level.extract_text(file_path)

    # Preprocess the extracted text
    cleaned_text = preprocess_text(text)

    # Split cleaned text into chunks (paragraph-based chunking)
    chunks = cleaned_text.split("\n\n") 

    # Store the chunks in the vector store with metadata
    docs = [{"content": chunk, "metadata": {"filename": file.filename, "chunk_index": index}} 
            for index, chunk in enumerate(chunks) if chunk.strip()]

    # Add texts and metadata to the vector store
    vector_store.add_texts([doc["content"] for doc in docs], metadatas=[doc["metadata"] for doc in docs])

    # Log the info and return the number of chunks processed
    logger.info(f"User uploaded {file.filename}. Stored {len(docs)} chunks.")
    return {"filename": file.filename, "message": f"Stored {len(docs)} chunks from {file.filename}"}

# Endpoint for querying the system and generating answers
@app.post("/question/")  
def ask_question(question: str = Query(...), api_key: str = Depends(get_api_key)):
    """
    Ask a question based on the processed documents and return the answer.
    The context is derived from the top relevant document sections.
    After that, it requests Ollama for a response based on the combined context.
    """
    start_time = time.time()

    # Retrieve the top 3 most relevant document sections from the vector store
    docs = vector_store.similarity_search(question, k=3)

    # Combine the relevant document content to use as context
    context = " ".join([doc.page_content for doc in docs])

    # Use HTTP request to Ollama service with context as prompt
    try:
        response = requests.post(
            "http://ollama:11434/api/generate",  # Make sure the Ollama container is accessible
            json={
                "prompt": f"Context: {context}\n\nQuestion: {question}",
                "stream": False,
                "model": "llama3"
            }
        )

        if response.status_code == 200:
            ollama_response = response.json()
            response_message = ollama_response.get('response', 'No response generated')
            tokens_used = len(response_message.split())
            logger.info(f"Tokens used for this response: {tokens_used}")
        else:
            logger.error(f"Error calling Ollama: {response.text}")
            response_message = "Error processing the response."

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during Ollama chat: {e}")
        response_message = "Error processing the response."

    # Calculate response time and log it
    response_time = time.time() - start_time
    logger.info(f"Response time for question: {response_time:.4f} seconds")

    # Return the generated answer and the sources of the context used
    sources = [doc.metadata for doc in docs]
    return {"answer": response_message, "sources": sources}


