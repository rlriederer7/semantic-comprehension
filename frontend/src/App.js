import React, { useState } from 'react';
import { Search, Upload, FileText, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB in bytes

function App() {
  const [activeTab, setActiveTab] = useState('search');

  // Upload state
  const [uploadText, setUploadText] = useState('');
  const [documentName, setDocumentName] = useState('');
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMode, setUploadMode] = useState('text'); // 'text' or 'file'
  const [selectedFile, setSelectedFile] = useState(null);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [useLLM, setUseLLM] = useState(true);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      const validTypes = ['.pdf', '.txt'];
      const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

      if (!validTypes.includes(fileExtension)) {
        setUploadStatus({
          type: 'error',
          message: 'Invalid file type. Please upload .pdf or .txt files only.'
        });
        return;
      }

      setSelectedFile(file);
      if (!documentName) {
        setDocumentName(file.name);
      }
      setUploadStatus(null);
    }
  };

  const handleUpload = async () => {
    if (uploadMode === 'text') {
      if (!uploadText.trim() || !documentName.trim()) {
        setUploadStatus({ type: 'error', message: 'Please provide both text and document name' });
        return;
      }
    } else {
      if (!selectedFile) {
        setUploadStatus({ type: 'error', message: 'Please select a file to upload' });
        return;
      }
    }

    setUploading(true);
    setUploadStatus(null);

    try {
      let response;

      if (uploadMode === 'text') {
        response = await fetch(`${API_BASE_URL}/embeddings/upload`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: uploadText,
            document_name: documentName
          })
        });
      } else {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('document_name', documentName);

        response = await fetch(`${API_BASE_URL}/embeddings/upload_file`, {
          method: 'POST',
          body: formData
        });
      }

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }

      const data = await response.json();
      setUploadStatus({
        type: 'success',
        message: `Successfully uploaded "${data.document_name}" with ${data.chunks_created} chunks${data.file_type ? ` (${data.file_type.toUpperCase()})` : ''}`
      });
      setUploadText('');
      setDocumentName('');
      setSelectedFile(null);
    } catch (error) {
      setUploadStatus({
        type: 'error',
        message: error.message || 'Failed to upload document'
      });
    } finally {
      setUploading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchError('Please enter a search query');
      return;
    }

    setSearching(true);
    setSearchError(null);
    setSearchResults(null);

    try {
      const endpoint = useLLM ? '/search/search_with_llm' : '/search/search_semantic_only';
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          top_k: topK,
          provider: 'openai'
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Search failed');
      }

      const data = await response.json();
      setSearchResults(data);
    } catch (error) {
      setSearchError(error.message || 'Failed to search documents');
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6">
            <h1 className="text-3xl font-bold mb-2">Semantic Search System</h1>
            <p className="text-blue-100">Upload documents and search with AI-powered semantic understanding</p>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setActiveTab('search')}
              className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
                activeTab === 'search'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Search size={20} />
              Search
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
                activeTab === 'upload'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Upload size={20} />
              Upload
            </button>
          </div>

          {/* Content */}
          <div className="p-6">
            {activeTab === 'upload' && (
              <div className="space-y-4">
                {/* Upload Mode Toggle */}
                <div className="flex gap-2 p-1 bg-gray-100 rounded-lg w-fit">
                  <button
                    onClick={() => {
                      setUploadMode('text');
                      setUploadStatus(null);
                    }}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      uploadMode === 'text'
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    Text Input
                  </button>
                  <button
                    onClick={() => {
                      setUploadMode('file');
                      setUploadStatus(null);
                    }}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      uploadMode === 'file'
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    File Upload
                  </button>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Document Name
                  </label>
                  <input
                    type="text"
                    value={documentName}
                    onChange={(e) => setDocumentName(e.target.value)}
                    placeholder="Enter document name..."
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                {uploadMode === 'text' ? (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Document Text
                    </label>
                    <textarea
                      value={uploadText}
                      onChange={(e) => setUploadText(e.target.value)}
                      placeholder="Paste your document text here..."
                      rows={12}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
                    />
                  </div>
                ) : (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Upload File (.pdf or .txt)
                    </label>
                    <div className="relative">
                      <input
                        type="file"
                        accept=".pdf,.txt"
                        onChange={handleFileSelect}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
                      />
                    </div>
                    {selectedFile && (
                      <div className="mt-2 text-sm text-gray-600 flex items-center gap-2">
                        <FileText size={16} />
                        <span>{selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</span>
                      </div>
                    )}
                  </div>
                )}

                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
                >
                  {uploading ? (
                    <>
                      <Loader2 size={20} className="animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload size={20} />
                      Upload Document
                    </>
                  )}
                </button>

                {uploadStatus && (
                  <div
                    className={`p-4 rounded-lg flex items-start gap-3 ${
                      uploadStatus.type === 'success'
                        ? 'bg-green-50 text-green-800 border border-green-200'
                        : 'bg-red-50 text-red-800 border border-red-200'
                    }`}
                  >
                    {uploadStatus.type === 'success' ? (
                      <CheckCircle size={20} className="flex-shrink-0 mt-0.5" />
                    ) : (
                      <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
                    )}
                    <span>{uploadStatus.message}</span>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'search' && (
              <div className="space-y-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Search Query
                    </label>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                      placeholder="What would you like to search for?"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div className="flex gap-4 items-center">
                    <div className="flex-1">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Number of Results
                      </label>
                      <input
                        type="number"
                        value={topK}
                        onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value) || 1))}
                        min="1"
                        max="100"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>

                    <div className="flex items-center gap-2 pt-7">
                      <input
                        type="checkbox"
                        id="useLLM"
                        checked={useLLM}
                        onChange={(e) => setUseLLM(e.target.checked)}
                        className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                      />
                      <label htmlFor="useLLM" className="text-sm font-medium text-gray-700">
                        Generate AI Answer
                      </label>
                    </div>
                  </div>

                  <button
                    onClick={handleSearch}
                    disabled={searching}
                    className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
                  >
                    {searching ? (
                      <>
                        <Loader2 size={20} className="animate-spin" />
                        Searching...
                      </>
                    ) : (
                      <>
                        <Search size={20} />
                        Search
                      </>
                    )}
                  </button>
                </div>

                {searchError && (
                  <div className="p-4 rounded-lg flex items-start gap-3 bg-red-50 text-red-800 border border-red-200">
                    <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
                    <span>{searchError}</span>
                  </div>
                )}

                {searchResults && (
                  <div className="space-y-6">
                    {searchResults.answer && (
                      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200">
                        <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
                          <FileText size={20} className="text-blue-600" />
                          AI-Generated Answer
                        </h3>
                        <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                          {searchResults.answer}
                        </p>
                        {searchResults.provider_used && (
                          <p className="text-xs text-gray-600 mt-3">
                            Provider: {searchResults.provider_used}
                          </p>
                        )}
                      </div>
                    )}

                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">
                        Relevant Chunks ({searchResults.chunks.length})
                      </h3>
                      <div className="space-y-3">
                        {searchResults.chunks.map((chunk, idx) => {
                          const [text, docName, score] = chunk;
                          const percentage = (score * 100).toFixed(1);

                          return (
                            <div
                              key={idx}
                              className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                            >
                              <div className="flex items-start justify-between gap-4 mb-2">
                                <span className="text-sm font-medium text-blue-600">
                                  {docName}
                                </span>
                                <div className="flex items-center gap-2">
                                  <div className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-medium">
                                    {percentage}% match
                                  </div>
                                </div>
                              </div>
                              <p className="text-gray-700 text-sm leading-relaxed">
                                {text}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;