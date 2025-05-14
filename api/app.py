# import os
# import uvicorn
# from typing import List, Optional
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv

# from rag.document_processor import DocumentProcessor
# from rag.retriever import Retriever
# from rag.db import MongoDB

# # Load environment variables
# load_dotenv()

# app = FastAPI(title="RAG API")

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize MongoDB connection
# mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
# db_name = os.getenv("MONGODB_DB_NAME", "rag_database")
# mongo_db = MongoDB(mongo_uri, db_name)

# # Initialize processor and retriever
# document_processor = DocumentProcessor(mongo_db)
# retriever = Retriever(mongo_db)

# # Pydantic models
# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 3

# class QueryResponse(BaseModel):
#     query: str
#     context: List[dict]
#     answer: Optional[str] = None

# @app.get("/")
# async def root():
#     return {"message": "RAG API is running"}

# @app.get("/health")
# async def health_check():
#     try:
#         is_connected = await mongo_db.check_connection()
#         return {
#             "status": "healthy" if is_connected else "unhealthy",
#             "mongodb_connected": is_connected
#         }
#     except Exception as e:
#         return {"status": "unhealthy", "error": str(e)}

# @app.post("/documents/upload")
# async def upload_document(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     document_name: Optional[str] = Form(None)
# ):
#     if not file:
#         raise HTTPException(status_code=400, detail="No file provided")
    
#     try:
#         # If no document name provided, use the filename
#         if not document_name:
#             document_name = file.filename
            
#         file_content = await file.read()
        
#         # Process document in the background
#         background_tasks.add_task(
#             document_processor.process_document,
#             file_content,
#             document_name,
#             file.content_type
#         )
        
#         return {
#             "message": "Document upload initiated. Processing in background.",
#             "document_name": document_name
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

# @app.get("/documents")
# async def list_documents():
#     try:
#         documents = await mongo_db.list_documents()
#         return {"documents": documents}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

# @app.post("/query", response_model=QueryResponse)
# async def query(request: QueryRequest):
#     try:
#         # Retrieve relevant documents and generate answer
#         result = await retriever.retrieve(request.query, top_k=request.top_k)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# @app.delete("/documents/{document_id}")
# async def delete_document(document_id: str):
#     try:
#         result = await mongo_db.delete_document(document_id)
#         if result:
#             return {"message": f"Document {document_id} deleted successfully"}
#         else:
#             raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

import os
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.document_processor import DocumentProcessor
from rag.retriever import Retriever
from rag.db import MongoDB

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and services
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
db_name = os.getenv("MONGODB_DB_NAME", "rag_database")
mongo_db = MongoDB(mongo_uri, db_name)

document_processor = DocumentProcessor(mongo_db)
retriever = Retriever(mongo_db)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class ChunkContext(BaseModel):
    text: str
    document_id: str
    document_name: str
    chunk_index: Optional[int]
    score: Optional[float] = None
    rerank_score: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    context: List[ChunkContext]
    answer: Optional[str]

# Routes
@app.get("/")
async def root():
    return {"message": "RAG API is running"}

@app.get("/health")
async def health_check():
    try:
        is_connected = await mongo_db.check_connection()
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "mongodb_connected": is_connected
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        if not document_name:
            document_name = file.filename

        file_content = await file.read()

        background_tasks.add_task(
            document_processor.process_document,
            file_content,
            document_name,
            file.content_type
        )

        return {
            "message": "Document upload initiated. Processing in background.",
            "document_name": document_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/documents")
async def list_documents():
    try:
        documents = await mongo_db.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = await retriever.retrieve(request.query, top_k=request.top_k)
        return QueryResponse(
            query=result["query"],
            context=result["context"],
            answer=result["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        result = await mongo_db.delete_document(document_id)
        if result:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
