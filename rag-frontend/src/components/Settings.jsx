import React, { useState, useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';

const Settings = () => {
  const [settings, setSettings] = useState({
  model: 'qwen2.5:14b',
  embeddings_model: 'BAAI/bge-base-en-v1.5'
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await fetch('/settings');
      const data = await response.json();
      if (data.settings) {
        setSettings(data.settings);
      }
    } catch (err) {
      setError('Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const formData = new FormData();
      Object.entries(settings).forEach(([key, value]) => {
        formData.append(key, value);
      });

      const response = await fetch('/settings', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      } else {
        setError(data.message);
        setTimeout(() => setError(null), 3000);
      }
    } catch (err) {
      setError('Failed to save settings');
      setTimeout(() => setError(null), 3000);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: value
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-gray-500">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-2xl mx-auto p-6">
        <div className="flex items-center space-x-4 mb-8">
          <button
            onClick={() => window.location.href = '/'}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <ArrowLeft className="w-6 h-6 text-gray-600" />
          </button>
          <h1 className="text-2xl font-semibold text-gray-800">Settings</h1>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
            {error}
          </div>
        )}

        {success && (
          <div className="mb-4 p-4 bg-green-50 border-l-4 border-green-500 text-green-700">
            Settings saved successfully!
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6 bg-white p-6 rounded-lg shadow-sm">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model
            </label>
            <select
              name="model"
              value={settings.model}
              onChange={handleInputChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="llama3.1:8b-instruct-q6_K">Llama3.1-8B</option>
              <option value="qwen2.5:14b">Qwen2.5-14B</option>
              <option value="llama3.2:latest">Llama3.2-3B</option>
              <option value="granite3-moe:3b">Granite-MoE</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Embeddings Model
            </label>
            <select
              name="embeddings_model"
              value={settings.embeddings_model}
              onChange={handleInputChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="BAAI/bge-base-en-v1.5">English BAAI</option>
              <option value="BAAI/bge-m3">Multilingual BAAI</option>
              <option value="nomic-embed-text:latest ">Nomic Text</option>
            </select>
          </div>
{/* 
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Resize Height (px)
            </label>
            <input
              type="number"
              name="resized_height"
              value={settings.resized_height}
              onChange={handleInputChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Resize Width (px)
            </label>
            <input
              type="number"
              name="resized_width"
              value={settings.resized_width}
              onChange={handleInputChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div> */}

          <button
            type="submit"
            className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Save Settings
          </button>
        </form>
      </div>
    </div>
  );
};

export default Settings;