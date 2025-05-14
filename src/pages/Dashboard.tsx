import React from 'react';
import { Link } from 'react-router-dom';
import { useDocuments } from '../context/DocumentsContext';
import { FileText, Upload, Search, Trash2, AlertCircle, Loader } from 'lucide-react';

const Dashboard: React.FC = () => {
  const { documents, loading, error, deleteDocument } = useDocuments();

  const handleDelete = async (id: string, name: string) => {
    if (window.confirm(`Are you sure you want to delete "${name}"?`)) {
      await deleteDocument(id);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        
        <div className="flex flex-col sm:flex-row gap-3">
          <Link
            to="/upload"
            className="inline-flex items-center justify-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-md transition-colors shadow-sm"
          >
            <Upload size={18} className="mr-2" />
            Upload Document
          </Link>
          <Link
            to="/query"
            className="inline-flex items-center justify-center px-4 py-2 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 text-gray-800 dark:text-white font-medium rounded-md border border-gray-300 dark:border-gray-600 transition-colors shadow-sm"
          >
            <Search size={18} className="mr-2" />
            Query Knowledge Base
          </Link>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white dark:bg-gray-700 rounded-lg shadow-sm p-6 transition-transform hover:scale-[1.02] duration-300">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Total Documents</h2>
            <div className="bg-blue-100 dark:bg-blue-900 p-2 rounded-full">
              <FileText className="text-blue-600 dark:text-blue-300" size={20} />
            </div>
          </div>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">{documents.length}</p>
        </div>
        
        <div className="bg-white dark:bg-gray-700 rounded-lg shadow-sm p-6 transition-transform hover:scale-[1.02] duration-300">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Total Chunks</h2>
            <div className="bg-green-100 dark:bg-green-900 p-2 rounded-full">
              <FileText className="text-green-600 dark:text-green-300" size={20} />
            </div>
          </div>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">
            {documents.reduce((sum, doc) => sum + doc.chunk_count, 0)}
          </p>
        </div>
        
        <div className="bg-white dark:bg-gray-700 rounded-lg shadow-sm p-6 transition-transform hover:scale-[1.02] duration-300">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-200">System Status</h2>
            <div className="bg-purple-100 dark:bg-purple-900 p-2 rounded-full">
              <div className="text-purple-600 dark:text-purple-300">
                <span className="block w-2 h-2 bg-green-500 rounded-full"></span>
              </div>
            </div>
          </div>
          <p className="text-xl font-medium text-green-600 dark:text-green-400">Ready</p>
        </div>
      </div>

      {/* Documents List */}
      <div className="bg-white dark:bg-gray-700 rounded-lg shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-600">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Documents</h2>
        </div>
        
        {loading ? (
          <div className="flex items-center justify-center p-8">
            <Loader className="animate-spin text-indigo-600 dark:text-indigo-400 mr-2" size={24} />
            <p className="text-gray-600 dark:text-gray-300">Loading documents...</p>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center p-8 text-red-600 dark:text-red-400">
            <AlertCircle className="mr-2" size={24} />
            <p>{error}</p>
          </div>
        ) : documents.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-8 text-gray-500 dark:text-gray-400">
            <FileText size={48} className="mb-4 opacity-40" />
            <p className="mb-2">No documents found</p>
            <Link
              to="/upload"
              className="text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 dark:hover:text-indigo-300 font-medium"
            >
              Upload your first document
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-600">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Chunks
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-700 divide-y divide-gray-200 dark:divide-gray-600">
                {documents.map((doc) => (
                  <tr key={doc.id} className="hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FileText className="text-gray-400 mr-3" size={18} />
                        <div className="text-sm font-medium text-gray-900 dark:text-white">{doc.name}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                      {doc.chunk_count}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                      {doc.content_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <button
                        onClick={() => handleDelete(doc.id, doc.name)}
                        className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300 transition-colors"
                      >
                        <Trash2 size={18} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;