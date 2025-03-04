# Retrieval-Augmented Generation (RAG) System

## Overview

This project implements a basic Retrieval-Augmented Generation (RAG) system that answers questions about technical documentation. The system uses FastAPI for API development, LangChain for RAG implementation, Ollama for local LLM deployment, and Docker for containerization. The document chunks are stored in a vector store (Chroma).

## System Architecture

The system is composed of the following components:

1. **FastAPI Application**: Exposes two main endpoints (`/upload/` for document ingestion and `/question/` for answering queries).
2. **Ollama LLM Service**: Provides the model for generating responses based on context.
3. **Vector Store (Chroma)**: Stores the document embeddings for fast retrieval during querying.
4. **Docker**: Ensures the system is containerized, including all dependencies (FastAPI, Ollama, Chroma, etc.) and orchestrated with Docker Compose.

## Setup Instructions

### Prerequisites

1. Install Docker and Docker Compose.
2. Install Python (recommended version: 3.10 or above).
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/fadawkas/RAG-System_Kuasar.git
   cd RAG-System_Kuasar
   ```

2. **Start the Docker containers**:
   In the project directory, run the following command to start all services:

   ```bash
   docker-compose up --build
   ```

3. **FastAPI Server**: The FastAPI application will be running on `http://localhost:8000`. You can test the endpoints through the Swagger UI available at `http://localhost:8000/docs`.

### API Usage

#### 1. **API KEY**

To interact with the API, you need to include the API key in your requests. The API key for this project is:
```
cfbf1ae5-5aa8-40fa-a23f-f2534a0635dd
```

#### 2. **Upload a Document**

- Endpoint: `POST /upload/`
- Description: Upload a PDF file to the system, which will be processed and stored in the vector store.
- Request:

  - `file`: The PDF file to upload.
  - `api_key`: The API key for authentication.

  Example (using cURL):

  ```bash
  curl -X 'POST' \
    'http://localhost:8000/upload/' \
    -H 'X-API-KEY: API-KEY' \
    -F 'file=@/path/to/your/document.pdf'
  ```

#### 3. **Ask a Question**

- Endpoint: `POST /question/`
- Description: Ask a question related to the uploaded document(s). The system will retrieve relevant context and use Ollama to generate a response.
- Request:

  - `question`: The question you want to ask.
  - `api_key`: The API key for authentication.

  Example (using cURL):

  ```bash
  curl -X 'POST' \
    'http://localhost:8000/question/?question=What%20can%20you%20tell%20me%20about%20FastAPI' \
    -H 'X-API-KEY: API-KEY'
  ```

#### Response:

The response will contain the answer generated by Ollama along with the document sources used to generate the answer.

### Example Response:

```json
{
  "answer": "FastAPI is a modern web framework for building APIs with Python...",
  "sources": [
    {
      "filename": "FastAPI Essay.pdf",
      "chunk_index": 1
    },
    {
      "filename": "FastAPI Essay.pdf",
      "chunk_index": 2
    }
  ]
}
```

## Features

- **Document Ingestion**: Uploads and processes PDFs into text chunks for efficient retrieval.
- **Contextual Question Answering**: Combines context from relevant document chunks and queries Ollama for generating responses.
- **Containerized Setup**: Uses Docker to containerize the application and its services, ensuring consistency across environments.
- **Authentication**: Implement basic authentication using API KEY.

## Evaluation Metrics

- **Token Usage**: Tracks the number of tokens used for each response.
- **Response Time**: Measures the time taken to generate a response.
- **Error Handling**: Basic error handling for Ollama communication and API request issues.

## Conclusion

This RAG system provides a foundational implementation of document-based question answering using a combination of FastAPI, Ollama, and vector stores. It's designed to be scalable, and the components can be extended or replaced based on future needs.
