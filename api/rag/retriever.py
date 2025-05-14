# from typing import List, Dict, Any
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# import torch
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Retriever:
#     def __init__(self, db):
#         self.db = db
#         # Initialize HuggingFace model and tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
#         self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
        
#         # Initialize Gemini AI
#         genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
#         self.model_gemini = genai.GenerativeModel('gemini-2.0-flash')
    
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
    
#     async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
#         """
#         Retrieve the most relevant chunks and generate an answer using Gemini.
#         """
#         try:
#             # Generate query embedding using HuggingFace
#             query_embedding = self.generate_embedding(query)
            
#             # Retrieve relevant chunks from database
#             relevant_chunks = await self.db.query_similar(query_embedding, top_k)
            
#             # Format results and prepare context for Gemini
#             results = []
#             context_text = ""
#             for chunk in relevant_chunks:
#                 # Get document metadata
#                 document = await self.db.get_document(chunk["document_id"])
#                 document_name = document.get("name", "Unknown document") if document else "Unknown document"
                
#                 # Add to context
#                 context_text += f"\nFrom {document_name}:\n{chunk['text']}\n"
                
#                 results.append({
#                     "text": chunk["text"],
#                     "document_id": chunk["document_id"],
#                     "document_name": document_name,
#                     "chunk_index": chunk["chunk_index"],
#                     "score": chunk["score"] if "score" in chunk else None
#                 })
            
#             # Generate answer using Gemini
#             prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

# Context:
# {context_text}

# Question: {query}

# Answer:"""
            
#             response = self.model_gemini.generate_content(prompt)
#             answer = response.text if response else None
            
#             return {
#                 "query": query,
#                 "context": results,
#                 "answer": answer
#             }
            
#         except Exception as e:
#             print(f"Error in retrieval: {str(e)}")
#             raise




# from typing import List, Dict, Any
# import numpy as np
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Retriever:
#     def __init__(self, db):
#         self.db = db
#         # Initialize HuggingFace model and tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
#         self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)

#         # Initialize Gemini AI
#         genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
#         self.model_gemini = genai.GenerativeModel('gemini-2.0-flash')

#     def _mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output[0]
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#     def generate_embedding(self, text: str) -> List[float]:
#         encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
#         encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)
#         sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
#         sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
#         return sentence_embeddings[0].cpu().numpy().tolist()

#     def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         # TF-IDF similarity
#         texts = [chunk["text"] for chunk in chunks]
#         tfidf = TfidfVectorizer().fit([query] + texts)
#         query_vec = tfidf.transform([query])
#         text_vecs = tfidf.transform(texts)
#         tfidf_scores = cosine_similarity(query_vec, text_vecs).flatten()

#         # Semantic similarity (cosine)
#         query_emb = torch.tensor(self.generate_embedding(query)).unsqueeze(0)
#         chunk_embs = torch.tensor([self.generate_embedding(chunk["text"]) for chunk in chunks])
#         cos_scores = F.cosine_similarity(query_emb, chunk_embs).cpu().numpy()

#         # Combine scores
#         for i, chunk in enumerate(chunks):
#             chunk["rerank_score"] = 0.7 * cos_scores[i] + 0.3 * tfidf_scores[i]

#         return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

#     async def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
#         """
#         Retrieve top_k chunks relevant to the query, rerank, and generate answer using Gemini.
#         """
#         try:
#             # Step 1: Vector search in DB
#             query_embedding = self.generate_embedding(query)
#             retrieved_chunks = await self.db.query_similar(query_embedding, top_k=10)

#             # Step 2: Rerank by semantic + keyword similarity
#             reranked_chunks = self.rerank_chunks(query, retrieved_chunks)[:top_k]

#             # Step 3: Prepare context
#             context_text = ""
#             results = []
#             for chunk in reranked_chunks:
#                 document = await self.db.get_document(chunk["document_id"])
#                 doc_name = document.get("name", "Unknown") if document else "Unknown"
#                 context_text += f"\n[From {doc_name}]:\n{chunk['text']}\n"
#                 results.append({
#                     "text": chunk["text"],
#                     "document_id": chunk["document_id"],
#                     "document_name": doc_name,
#                     "chunk_index": chunk.get("chunk_index"),
#                     "score": chunk.get("score"),
#                     "rerank_score": chunk["rerank_score"]
#                 })

#             # Step 4: Generate Gemini prompt
#             prompt = f"""
# You are a document-based assistant. You can only answer questions using the given context.
# If the context does not contain enough information, reply: "Không tìm thấy thông tin trong tài liệu."

# Context:
# {context_text.strip()}

# Question: {query}

# Answer (in Vietnamese, based only on the context above):
# """

#             # Step 5: Get answer from Gemini
#             response = self.model_gemini.generate_content(prompt)
#             answer = response.text.strip() if response else "Không tìm thấy thông tin."

#             return {
#                 "query": query,
#                 "context": results,
#                 "answer": answer
#             }

#         except Exception as e:
#             print(f"[Retriever Error] {e}")
#             raise




from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class Retriever:
    def __init__(self, db):
        self.db = db
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Init Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model_gemini = genai.GenerativeModel('gemini-2.0-flash')

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_embedding(self, text: str) -> List[float]:
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].cpu().numpy().tolist()

    def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [chunk["text"] for chunk in chunks]
        tfidf = TfidfVectorizer().fit([query] + texts)
        query_vec = tfidf.transform([query])
        text_vecs = tfidf.transform(texts)
        tfidf_scores = cosine_similarity(query_vec, text_vecs).flatten()

        query_emb = torch.tensor(self.generate_embedding(query)).unsqueeze(0)
        chunk_embs = torch.tensor([self.generate_embedding(chunk["text"]) for chunk in chunks])
        cos_scores = F.cosine_similarity(query_emb, chunk_embs).cpu().numpy()

        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = 0.7 * cos_scores[i] + 0.3 * tfidf_scores[i]

        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    def classify_question(self, query: str) -> str:
        q = query.lower()
        if q.startswith(("liệt kê", "các loại", "những gì")):
            return "list"
        elif any(x in q for x in ["là gì", "định nghĩa"]):
            return "definition"
        elif any(x in q for x in ["so sánh", "khác nhau", "giống nhau"]):
            return "comparison"
        else:
            return "general"

    def trim_context(self, text: str, max_words: int = 1000) -> str:
        words = text.split()
        return ' '.join(words[:max_words])

    async def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        try:
            # Step 1: Vector search
            query_embedding = self.generate_embedding(query)
            retrieved_chunks = await self.db.query_similar(query_embedding, top_k=10)

            # Step 2: Rerank
            reranked_chunks = self.rerank_chunks(query, retrieved_chunks)[:top_k]

            # Step 3: Prepare context
            context_text = ""
            results = []
            for chunk in reranked_chunks:
                document = await self.db.get_document(chunk["document_id"])
                doc_name = document.get("name", "Unknown") if document else "Unknown"
                context_text += f"\n[From {doc_name}]:\n{chunk['text']}\n"
                results.append({
                    "text": chunk["text"],
                    "document_id": chunk["document_id"],
                    "document_name": doc_name,
                    "chunk_index": chunk.get("chunk_index"),
                    "score": chunk.get("score"),
                    "rerank_score": chunk["rerank_score"]
                })

            trimmed_context = self.trim_context(context_text, max_words=1000)

            # Step 4: Prompt crafting
            q_type = self.classify_question(query)
            instructions = {
                "definition": "Trả lời ngắn gọn, định nghĩa rõ ràng.",
                "list": "Liệt kê rõ ràng từng mục.",
                "comparison": "So sánh theo từng tiêu chí cụ thể.",
                "general": ""
            }
            extra = instructions.get(q_type, "")

            prompt = f"""
Bạn là trợ lý AI trả lời dựa trên tài liệu. Chỉ sử dụng thông tin trong ngữ cảnh để trả lời. 
Nếu không đủ thông tin, hãy trả lời: "Không tìm thấy thông tin trong tài liệu."

Ngữ cảnh:
{trimmed_context.strip()}

Câu hỏi: {query}

{extra}

Trả lời (ngắn gọn, đúng trọng tâm, bằng tiếng Việt):
"""

            # Step 5: Gemini generation
            response = self.model_gemini.generate_content(prompt)
            answer = response.text.strip() if response else "Không tìm thấy thông tin."

            # Optional Step 6: Rút gọn lại nếu cần (nâng cao)
            # short_prompt = f"""Câu trả lời sau cần rút gọn, giữ nguyên ý, trả lời đúng trọng tâm:
            # Câu hỏi: {query}
            # Câu trả lời: {answer}
            # Câu trả lời đã rút gọn:"""
            # short_response = self.model_gemini.generate_content(short_prompt)
            # if short_response: answer = short_response.text.strip()

            return {
                "query": query,
                "context": results,
                "answer": answer
            }

        except Exception as e:
            print(f"[Retriever Error] {e}")
            raise
