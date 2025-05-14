import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import axios from 'axios';

export interface Document {
  id: string;
  name: string;
  content_type: string;
  chunk_count: number;
}

interface DocumentsContextType {
  documents: Document[];
  loading: boolean;
  error: string | null;
  fetchDocuments: () => Promise<void>;
  deleteDocument: (id: string) => Promise<void>;
}

const DocumentsContext = createContext<DocumentsContextType | undefined>(undefined);

export const useDocuments = () => {
  const context = useContext(DocumentsContext);
  if (context === undefined) {
    throw new Error('useDocuments must be used within a DocumentsProvider');
  }
  return context;
};

interface DocumentsProviderProps {
  children: ReactNode;
}

export const DocumentsProvider: React.FC<DocumentsProviderProps> = ({ children }) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_URL = 'http://localhost:8000';

  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/documents`);
      setDocuments(response.data.documents);
    } catch (err) {
      console.error('Error fetching documents:', err);
      setError('Failed to fetch documents');
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      await axios.delete(`${API_URL}/documents/${id}`);
      setDocuments(documents.filter(doc => doc.id !== id));
    } catch (err) {
      console.error('Error deleting document:', err);
      setError('Failed to delete document');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  return (
    <DocumentsContext.Provider
      value={{
        documents,
        loading,
        error,
        fetchDocuments,
        deleteDocument
      }}
    >
      {children}
    </DocumentsContext.Provider>
  );
};