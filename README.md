# RAG System with Python and MongoDB

A Retrieval-Augmented Generation (RAG) system built with Python and MongoDB, featuring a React frontend for document management and querying.

## Features

- Document ingestion and processing pipeline
- Vector embeddings generation and storage in MongoDB
- Semantic search functionality with relevance ranking
- Document retrieval and context-aware responses
- API endpoints for frontend integration
- User-friendly React interface for uploading documents and querying the system

## Tech Stack

- **Frontend**: React, TypeScript, Tailwind CSS
- **Backend**: Python, FastAPI
- **Database**: MongoDB
- **Embeddings**: Sentence Transformers
- **Text Processing**: LangChain

## Project Structure

- `/api` - Python backend
  - `/rag` - Core RAG implementation
  - `app.py` - FastAPI server
- `/src` - React frontend
  - `/components` - Reusable UI components
  - `/pages` - Application pages
  - `/context` - React context for state management

## Getting Started

### Prerequisites

- Node.js
- Python 3.8+
- MongoDB (local or Atlas)

### Installation

1. Clone the repository
2. Install frontend dependencies:
   ```
   npm install
   ```
3. Install backend dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Configure MongoDB connection in `.env` file

### Running the Application

1. Start the frontend development server:
   ```
   npm run dev
   ```
2. In a separate terminal, start the backend server:
   ```
   npm run api
   ```
   or
   ```
   python api/app.py
   ```

## Usage

1. Navigate to the Upload page to add documents to the knowledge base
2. Use the Query page to ask questions about your documents
3. View and manage your documents from the Dashboard

## License

MIT
# WebRagMongoDB