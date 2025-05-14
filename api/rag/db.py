from typing import List, Dict, Any, Optional
import os
import motor.motor_asyncio
from pymongo import ReturnDocument
import numpy as np

class MongoDB:
    def __init__(self, uri: str, db_name: str):
        """Initialize MongoDB connection."""
        self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        
        # Collections
        self.documents = self.db.documents
        self.chunks = self.db.chunks
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes asynchronously."""
        # Using create_task would be better in a real app
        async def create_indexes_async():
            # Document index
            await self.documents.create_index("id", unique=True)
            
            # Chunk indexes
            await self.chunks.create_index("document_id")
            await self.chunks.create_index([("document_id", 1), ("chunk_index", 1)], unique=True)
            # Vector index would typically be created here
            # But MongoDB requires a special configuration for vector search
            
        # Execute in the background
        import asyncio
        loop = asyncio.get_event_loop()
        loop.create_task(create_indexes_async())
    
    async def check_connection(self) -> bool:
        """Check if the MongoDB connection is working."""
        try:
            await self.client.admin.command('ping')
            return True
        except Exception:
            return False
    
    async def insert_document(self, document: Dict[str, Any]):
        """Insert document metadata."""
        return await self.documents.update_one(
            {"id": document["id"]},
            {"$set": document},
            upsert=True
        )
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        return await self.documents.find_one({"id": document_id}, {"_id": 0})
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        cursor = self.documents.find({}, {"_id": 0})
        return await cursor.to_list(length=None)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        # Delete document
        result_doc = await self.documents.delete_one({"id": document_id})
        
        # Delete chunks
        result_chunks = await self.chunks.delete_many({"document_id": document_id})
        
        return result_doc.deleted_count > 0
    
    async def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert multiple chunks."""
        if not chunks:
            return
            
        # Use bulk operations for efficiency
        operations = []
        for chunk in chunks:
            operations.append(
                pymongo.UpdateOne(
                    {
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"]
                    },
                    {"$set": chunk},
                    upsert=True
                )
            )
            
        if operations:
            return await self.chunks.bulk_write(operations)
    
    async def query_similar(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query for similar chunks using vector similarity.
        
        Note: For a full implementation, you'd use MongoDB's $vectorSearch operator,
        but this requires MongoDB Atlas or MongoDB 7.0+ with vector search enabled.
        This implementation uses a simplified approach for demonstration.
        """
        # Simplified similarity search without using MongoDB's vector capabilities
        # In a real app, you would use MongoDB's vector search capabilities
        
        # Get all chunks (in a real app, you'd use proper vector search)
        all_chunks = await self.chunks.find({}, {"_id": 0}).to_list(length=None)
        
        # Convert query_embedding to numpy array
        query_embedding_np = np.array(query_embedding)
        
        # Calculate similarity scores
        results_with_scores = []
        for chunk in all_chunks:
            chunk_embedding = np.array(chunk["embedding"])
            
            # Calculate cosine similarity
            dot_product = np.dot(query_embedding_np, chunk_embedding)
            norm_query = np.linalg.norm(query_embedding_np)
            norm_chunk = np.linalg.norm(chunk_embedding)
            
            similarity = dot_product / (norm_query * norm_chunk) if norm_query * norm_chunk != 0 else 0
            
            # Create result with score
            result = {**chunk, "score": float(similarity)}
            results_with_scores.append(result)
        
        # Sort by score and get top_k
        results_with_scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = results_with_scores[:top_k]
        
        return top_results

# Imported here for the bulk operations
import pymongo