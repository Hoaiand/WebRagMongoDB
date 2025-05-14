import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import QueryPage from './pages/QueryPage';
import { DocumentsProvider } from './context/DocumentsContext';

function App() {
  return (
    <DocumentsProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/query" element={<QueryPage />} />
          </Routes>
        </Layout>
      </Router>
    </DocumentsProvider>
  );
}

export default App;