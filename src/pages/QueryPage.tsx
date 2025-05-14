import React, { useState } from "react";
import axios from "axios";
import {
  Search,
  SendHorizontal,
  FileText,
  Loader,
  AlertOctagon,
} from "lucide-react";

interface SearchResult {
  query: string;
  context: Array<{
    text: string;
    document_id: string;
    document_name: string;
    chunk_index: number;
    score: number;
  }>;
  answer: string | null;
}

const QueryPage: React.FC = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(3);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setError(null);

    try {
      const response = await axios.post("http://localhost:8000/query", {
        query: query.trim(),
        top_k: topK,
      });

      setResults(response.data);
    } catch (err: any) {
      console.error("Search error:", err);
      setError(
        err.response?.data?.detail || "An error occurred while searching"
      );
    } finally {
      setIsSearching(false);
    }
  };

  const highlightQueryTerms = (
    text: string,
    query: string
  ): React.ReactNode => {
    if (!query) return text;

    const queryTerms = query
      .toLowerCase()
      .split(/\s+/)
      .filter((term) => term.length > 2);
    if (queryTerms.length === 0) return text;

    const parts = [];
    let lastIndex = 0;

    // Create a regex to match all query terms
    const regex = new RegExp(`(${queryTerms.join("|")})`, "gi");

    // Find all matches
    let match;
    while ((match = regex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      // Add the highlighted match
      parts.push(
        <span
          key={match.index}
          className="bg-yellow-200 dark:bg-yellow-700 font-medium"
        >
          {match[0]}
        </span>
      );

      lastIndex = regex.lastIndex;
    }

    // Add the remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return <>{parts}</>;
  };

  return (
    <div className="space-y-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Query Knowledge Base
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Ask questions about your documents to retrieve relevant information.
        </p>
      </div>

      <div className="bg-white dark:bg-gray-700 rounded-lg shadow-sm p-6">
        <form onSubmit={handleSearch} className="space-y-4">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="text-gray-400" size={20} />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="block w-full pl-10 pr-12 py-3 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400"
            />
            <div className="absolute inset-y-0 right-0 flex items-center">
              <button
                type="submit"
                disabled={isSearching || !query.trim()}
                className={`h-full px-4 rounded-r-md flex items-center justify-center transition-colors ${
                  isSearching || !query.trim()
                    ? "bg-gray-100 dark:bg-gray-600 text-gray-400 dark:text-gray-500 cursor-not-allowed"
                    : "bg-indigo-100 dark:bg-indigo-700 text-indigo-600 dark:text-indigo-200 hover:bg-indigo-200 dark:hover:bg-indigo-600"
                }`}
              >
                {isSearching ? (
                  <Loader className="animate-spin" size={20} />
                ) : (
                  <SendHorizontal size={20} />
                )}
              </button>
            </div>
          </div>

          {/* <div className="flex items-center space-x-2">
            <label
              htmlFor="topK"
              className="text-sm text-gray-600 dark:text-gray-300"
            >
              Number of results:
            </label>
            <select
              id="topK"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            >
              {[1, 3, 5, 10].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div> */}
        </form>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300 p-4 rounded-md flex items-start">
          <AlertOctagon className="mt-0.5 mr-2 flex-shrink-0" size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white">
            Results for "{results.query}"
          </h2>

          {results.context.length === 0 ? (
            <div className="bg-white dark:bg-gray-700 rounded-lg shadow-sm p-6 text-center text-gray-500 dark:text-gray-400">
              No relevant information found. Try a different query or upload
              more documents.
            </div>
          ) : (
            <div className="space-y-4">
              {/* {results.context.map((item, index) => (
                <div key={index} className="bg-white dark:bg-gray-700 rounded-lg shadow-sm p-6 transition-all hover:shadow-md">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
                      <FileText size={16} className="mr-2" />
                      <span>From: {item.document_name}</span>
                    </div>
                    <div className="text-sm bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded-md text-gray-600 dark:text-gray-300">
                      Score: {(item.score * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="prose dark:prose-invert max-w-none">
                    <p className="text-gray-800 dark:text-gray-200 leading-relaxed">
                      {highlightQueryTerms(item.text, results.query)}
                    </p>
                  </div>
                </div>
              ))} */}
              {results.answer ? (
                <div className="bg-white dark:bg-gray-700 rounded-lg shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
                    Answer:
                  </h3>
                  <p className="text-gray-800 dark:text-gray-200 leading-relaxed">
                    {highlightQueryTerms(results.answer, results.query)}
                  </p>
                </div>
              ) : (
                <div className="bg-white dark:bg-gray-700 rounded-lg shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
                    Answer:
                  </h3>
                  <p className="text-gray-500 dark:text-gray-400">
                    No answer found.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryPage;
