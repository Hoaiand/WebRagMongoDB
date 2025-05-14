import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { useDocuments } from '../context/DocumentsContext';
import { Upload as UploadIcon, File, X, CheckCircle, AlertTriangle, Loader } from 'lucide-react';

const Upload: React.FC = () => {
  const navigate = useNavigate();
  const { fetchDocuments } = useDocuments();
  const [file, setFile] = useState<File | null>(null);
  const [documentName, setDocumentName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0];
      setFile(selectedFile);
      if (!documentName) {
        setDocumentName(selectedFile.name);
      }
    }
  }, [documentName]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1,
    accept: {
      'text/plain': ['.txt'],
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
  });

  const removeFile = () => {
    setFile(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setUploadResult({
        success: false,
        message: 'Please select a file to upload',
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    if (documentName) {
      formData.append('document_name', documentName);
    }

    setUploading(true);
    setUploadResult(null);

    try {
      await axios.post('http://localhost:8000/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadResult({
        success: true,
        message: 'Document uploaded successfully! Processing has begun.',
      });
      
      // Refresh documents list
      await fetchDocuments();
      
      // Clear form
      setFile(null);
      setDocumentName('');
      
      // Redirect after short delay
      setTimeout(() => navigate('/'), 3000);
    } catch (error: any) {
      console.error('Upload error:', error);
      setUploadResult({
        success: false,
        message: error.response?.data?.detail || 'An error occurred during upload',
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Upload Document</h1>
        <p className="text-gray-600 dark:text-gray-300">
          Upload documents to the knowledge base for later retrieval. Supported formats: TXT, PDF, DOC, DOCX.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-4">
          {/* Dropzone */}
          <div 
            {...getRootProps()} 
            className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
              ${isDragActive 
                ? 'border-indigo-400 bg-indigo-50 dark:border-indigo-500 dark:bg-indigo-900/20' 
                : 'border-gray-300 dark:border-gray-600 hover:border-indigo-400 dark:hover:border-indigo-500'
              }`}
          >
            <input {...getInputProps()} />
            
            <div className="flex flex-col items-center space-y-2">
              <div className="bg-indigo-100 dark:bg-indigo-900/30 p-3 rounded-full">
                <UploadIcon className="text-indigo-600 dark:text-indigo-400" size={24} />
              </div>
              
              {isDragActive ? (
                <p className="text-indigo-600 dark:text-indigo-400 font-medium">Drop the files here...</p>
              ) : (
                <>
                  <p className="text-gray-700 dark:text-gray-300 font-medium">
                    Drag & drop a file here, or click to select
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Supports: TXT, PDF, DOC, DOCX (max 10MB)
                  </p>
                </>
              )}
            </div>
          </div>

          {/* Selected File */}
          {file && (
            <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
              <div className="flex items-center space-x-3">
                <File className="text-indigo-600 dark:text-indigo-400" size={20} />
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-200">{file.name}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={removeFile}
                className="text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400 transition-colors"
                aria-label="Remove file"
              >
                <X size={18} />
              </button>
            </div>
          )}

          {/* Document Name */}
          <div>
            <label htmlFor="documentName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Document Name (Optional)
            </label>
            <input
              id="documentName"
              type="text"
              value={documentName}
              onChange={(e) => setDocumentName(e.target.value)}
              placeholder="Enter a custom name for your document"
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              If left empty, the original filename will be used
            </p>
          </div>
        </div>

        {/* Upload Result Message */}
        {uploadResult && (
          <div className={`p-4 rounded-md ${
            uploadResult.success 
              ? 'bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-300' 
              : 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300'
          }`}>
            <div className="flex items-center">
              {uploadResult.success ? (
                <CheckCircle className="mr-3" size={20} />
              ) : (
                <AlertTriangle className="mr-3" size={20} />
              )}
              <p>{uploadResult.message}</p>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-end space-x-4">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 font-medium"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={uploading || !file}
            className={`px-4 py-2 rounded-md font-medium text-white ${
              uploading || !file
                ? 'bg-indigo-400 dark:bg-indigo-600 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600'
            }`}
          >
            {uploading ? (
              <span className="flex items-center">
                <Loader className="animate-spin mr-2" size={18} />
                Uploading...
              </span>
            ) : (
              'Upload Document'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default Upload;