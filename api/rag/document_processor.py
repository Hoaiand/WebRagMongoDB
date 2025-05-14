# import os
# from typing import List, Dict, Any, Optional
# import hashlib
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer, AutoModel
# import torch

# class DocumentProcessor:
#     def __init__(self, db):
#         self.db = db
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#         )
#         # Initialize HuggingFace model and tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
#         self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
        
#     def _mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output[0]
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
#     def generate_embedding(self, text: str) -> List[float]:
#         # Tokenize and prepare input
#         encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
#         encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
#         # Generate embeddings
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)
        
#         # Mean pooling
#         sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
#         sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
#         return sentence_embeddings[0].cpu().numpy().tolist()
        
#     async def process_document(self, content: bytes, document_name: str, content_type: str):
#         """Process a document: extract text, split into chunks, embed, and store in MongoDB."""
#         try:
#             # Generate document ID
#             document_id = self._generate_document_id(content, document_name)
            
#             # For simplicity, assuming content is text. In a real app, handle different content types
#             text = content.decode('utf-8')
            
#             # Split text into chunks
#             chunks = self.text_splitter.split_text(text)
            
#             # Create document record
#             document = {
#                 "id": document_id,
#                 "name": document_name,
#                 "content_type": content_type,
#                 "chunk_count": len(chunks)
#             }
            
#             # Store document metadata
#             await self.db.insert_document(document)
            
#             # Process and store chunks with embeddings
#             await self._process_chunks(document_id, chunks)
            
#             return document_id
            
#         except Exception as e:
#             print(f"Error processing document: {str(e)}")
#             raise
    
#     async def _process_chunks(self, document_id: str, chunks: List[str]):
#         """Process text chunks, generate embeddings, and store them."""
#         chunk_objects = []
        
#         for i, chunk_text in enumerate(chunks):
#             # Generate embedding using HuggingFace model
#             embedding = self.generate_embedding(chunk_text)
            
#             # Create chunk object
#             chunk = {
#                 "document_id": document_id,
#                 "chunk_index": i,
#                 "text": chunk_text,
#                 "embedding": embedding
#             }
            
#             chunk_objects.append(chunk)
        
#         # Store chunks in batches if there are many
#         if chunk_objects:
#             await self.db.insert_chunks(chunk_objects)
    
#     def _generate_document_id(self, content: bytes, document_name: str) -> str:
#         """Generate a unique ID for the document based on content and name."""
#         content_hash = hashlib.md5(content).hexdigest()
#         name_hash = hashlib.md5(document_name.encode()).hexdigest()
#         return f"{name_hash[:8]}-{content_hash[:16]}"

import os
from typing import List, Dict, Any, Optional
import hashlib
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader  # Thêm thư viện xử lý PDF

class DocumentProcessor:
    def __init__(self, db):
        self.db = db
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # Initialize HuggingFace model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def generate_embedding(self, text: str) -> List[float]:
        # Tokenize and prepare input
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Mean pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings[0].cpu().numpy().tolist()
        
    async def process_document(self, content: bytes, document_name: str, content_type: str):
        """Process a document: extract text, split into chunks, embed, and store in MongoDB."""
        try:
            # Generate document ID
            document_id = self._generate_document_id(content, document_name)
            
            # Xử lý theo loại nội dung
            if content_type == "application/pdf":
                # Xử lý file PDF
                pdf_file = io.BytesIO(content)
                pdf_reader = PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            else:
                # Xử lý các định dạng text thông thường
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    raise ValueError(f"Không thể xử lý định dạng file: {content_type}")
            
            # Kiểm tra nếu text rỗng sau khi xử lý
            if not text.strip():
                raise ValueError(f"Không thể trích xuất nội dung từ file: {document_name}")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create document record
            document = {
                "id": document_id,
                "name": document_name,
                "content_type": content_type,
                "chunk_count": len(chunks)
            }
            
            # Store document metadata
            await self.db.insert_document(document)
            
            # Process and store chunks with embeddings
            await self._process_chunks(document_id, chunks)
            
            return document_id
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise
    
    async def _process_chunks(self, document_id: str, chunks: List[str]):
        """Process text chunks, generate embeddings, and store them."""
        chunk_objects = []
        
        for i, chunk_text in enumerate(chunks):
            # Generate embedding using HuggingFace model
            embedding = self.generate_embedding(chunk_text)
            
            # Create chunk object
            chunk = {
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk_text,
                "embedding": embedding
            }
            
            chunk_objects.append(chunk)
        
        # Store chunks in batches if there are many
        if chunk_objects:
            await self.db.insert_chunks(chunk_objects)
    
    def _generate_document_id(self, content: bytes, document_name: str) -> str:
        """Generate a unique ID for the document based on content and name."""
        content_hash = hashlib.md5(content).hexdigest()
        name_hash = hashlib.md5(document_name.encode()).hexdigest()
        return f"{name_hash[:8]}-{content_hash[:16]}"