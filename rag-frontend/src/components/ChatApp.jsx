import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, Settings, Plus, Trash2, Edit2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Transition } from '@headlessui/react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
// import ReactMarkdown from 'react-markdown';

// const ChatMessage = ({ content, role }) => {
//   // Convert string newlines to <br/> tags and preserve markdown
//   const formattedContent = content.split('\n').map((line, i) => (
//     <React.Fragment key={i}>
//       <ReactMarkdown className="inline">{line}</ReactMarkdown>
//       <br />
//     </React.Fragment>
//   ));

//   return (
//     <div className={`p-4 mb-4 rounded-lg ${role === 'user' ? 'bg-blue-50' : 'bg-green-50'}`}>
//       {formattedContent}
//     </div>
//   );
// };
import ChatMessage from './ChatMessage';  // Import ChatMessage component

const ChatApp = () => {
  const navigate = useNavigate();
  const { isDark, setIsDark } = useTheme();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [indexedFiles, setIndexedFiles] = useState([]);
  const fileInputRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showNewSession, setShowNewSession] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const fetchSessions = async () => {
    try {
      const response = await fetch('/chat');
      const data = await response.json();
      setSessions(data.chat_sessions);
      if (data.current_session) {
        setCurrentSession(data.current_session);
        setMessages(data.chat_history);
        setIndexedFiles(data.indexed_files);
      }
    } catch (err) {
      setError('Failed to load sessions');
      setTimeout(() => setError(null), 3000);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('query', input);
      formData.append('send_query', 'true');

      const response = await fetch('/chat', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setMessages(prev => [...prev, ...data.messages]);
        setInput('');
      } else {
        setError(data.message);
        setTimeout(() => setError(null), 3000);
      }
    } catch (err) {
      setError('Failed to send message');
      setTimeout(() => setError(null), 3000);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (files) => {
    if (!files || !files.length) return;

    try {
      setLoading(true);
      const formData = new FormData();
      Array.from(files).forEach(file => {
        formData.append('file', file);
      });
      formData.append('upload', 'true');

      const response = await fetch('/chat', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setIndexedFiles(data.files);
      } else {
        setError(data.message);
        setTimeout(() => setError(null), 3000);
      }
    } catch (err) {
      setError('Failed to upload files');
      setTimeout(() => setError(null), 3000);
    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    handleFileUpload(files);
  };

  const createNewSession = async () => {
    if (!newSessionName.trim()) {
      setError('Please enter a session name');
      setTimeout(() => setError(null), 3000);
      return;
    }

    try {
      const response = await fetch('/new_session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: newSessionName }),
      });

      if (response.ok) {
        await fetchSessions();
        setShowNewSession(false);
        setNewSessionName('');
      } else {
        setError('Failed to create session');
        setTimeout(() => setError(null), 3000);
      }
    } catch (err) {
      setError('Failed to create session');
      setTimeout(() => setError(null), 3000);
    }
  };

  const switchSession = async (sessionId) => {
    try {
      const response = await fetch(`/switch_session/${sessionId}`);
      if (response.ok) {
        setCurrentSession(sessionId);
        await fetchSessions();
      }
    } catch (err) {
      setError('Failed to switch sessions');
      setTimeout(() => setError(null), 3000);
    }
  };

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this session?')) return;

    try {
      const response = await fetch(`/delete_session/${sessionId}`, {
        method: 'POST',
      });
      const data = await response.json();
      if (data.success) {
        await fetchSessions();
      } else {
        setError(data.message);
        setTimeout(() => setError(null), 3000);
      }
    } catch (err) {
      setError('Failed to delete session');
      setTimeout(() => setError(null), 3000);
    }
  };

  return (
    <div 
      className={`flex h-screen ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag & Drop Overlay */}
      {isDragging && (
        <div className="absolute inset-0 bg-blue-500 bg-opacity-20 border-4 border-blue-500 border-dashed rounded-lg flex items-center justify-center z-50">
          <div className={`text-2xl ${isDark ? 'text-white' : 'text-gray-800'}`}>
            Drop files here to upload
          </div>
        </div>
      )}

      {/* Sidebar */}
      <div className={`w-64 ${
        isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      } border-r flex flex-col`}>
        <div className={`p-4 border-b ${isDark ? 'border-gray-700' : 'border-gray-200'}`}>
          <div className="flex justify-between items-center">
            <h2 className={`text-lg font-semibold ${
              isDark ? 'text-white' : 'text-gray-700'
            }`}>Sessions</h2>
            <div className="flex space-x-2">
              <button
                onClick={() => setIsDark(!isDark)}
                className={`p-2 rounded-full ${
                  isDark ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
                } transition-colors`}
                title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {isDark ? (
                  <Sun className="w-5 h-5 text-gray-300" />
                ) : (
                  <Moon className="w-5 h-5 text-gray-600" />
                )}
              </button>
              <button
                onClick={() => setShowNewSession(true)}
                className={`p-2 rounded-full ${
                  isDark ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
                } transition-colors`}
                title="Create new session"
              >
                <Plus className={`w-5 h-5 ${
                  isDark ? 'text-gray-300' : 'text-gray-600'
                }`} />
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => switchSession(session.id)}
              className={`flex items-center justify-between p-3 cursor-pointer transition-colors ${
                currentSession === session.id
                  ? isDark 
                    ? 'bg-gray-700 border-l-4 border-blue-500'
                    : 'bg-blue-50 border-l-4 border-blue-500'
                  : isDark
                    ? 'hover:bg-gray-700'
                    : 'hover:bg-gray-50'
              }`}
            >
              <span className={`truncate ${
                isDark ? 'text-gray-200' : 'text-gray-700'
              }`}>{session.name}</span>
              <button
                onClick={(e) => deleteSession(session.id, e)}
                className={`p-1 rounded-full ${
                  isDark ? 'hover:bg-gray-600' : 'hover:bg-gray-200'
                } transition-colors`}
                title="Delete session"
              >
                <Trash2 className={`w-4 h-4 ${
                  isDark ? 'text-gray-400' : 'text-gray-500'
                }`} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className={`${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        } border-b p-4 shadow-sm`}>
          <div className="flex justify-between items-center">
            <h1 className={`text-xl font-semibold ${
              isDark ? 'text-white' : 'text-gray-800'
            }`}>Chat</h1>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg ${
                  isDark 
                    ? 'bg-gray-700 text-blue-400 hover:bg-gray-600'
                    : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
                } transition-colors`}
              >
                <Upload className="w-5 h-5" />
                <span>Upload Files</span>
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => handleFileUpload(e.target.files)}
                className="hidden"
                multiple
                accept=".pdf,.csv,.txt"
              />
              <button
                onClick={() => navigate('/settings')}
                className={`p-2 rounded-full ${
                  isDark ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
                } transition-colors`}
                title="Settings"
              >
                <Settings className={`w-5 h-5 ${
                  isDark ? 'text-gray-300' : 'text-gray-600'
                }`} />
              </button>
            </div>
          </div>

          {indexedFiles.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {indexedFiles.map((file, index) => (
                <span
                  key={index}
                  className={`px-3 py-1 rounded-full text-sm ${
                    isDark 
                      ? 'bg-gray-700 text-gray-200'
                      : 'bg-gray-100 text-gray-700'
                  }`}
                >
                  {file}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Chat Messages */}
        <div
          ref={chatContainerRef}
          className={`flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar ${
            isDark ? 'bg-gray-900' : 'bg-gray-50'
          }`}
        >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            } w-full message-animation`}
          >
            <ChatMessage content={msg.content} role={msg.role} />
          </div>
        ))}
        {/* {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              } message-animation`}
            >
              <div
                className={`max-w-2xl p-4 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : isDark
                      ? 'bg-gray-700 text-gray-200'
                      : 'bg-white shadow-md'
                }`}
              >
                {message.content}
              </div>
            </div>
          ))} */}
          {loading && (
            <div className="flex justify-start message-animation">
              <div className={`max-w-2xl p-4 rounded-lg ${
                isDark ? 'bg-gray-700' : 'bg-white shadow-md'
              }`}>
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
                       style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
                       style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
                       style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Error Display */}
        <Transition
          show={!!error}
          enter="transition-opacity duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="transition-opacity duration-300"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed top-4 right-4 max-w-sm bg-red-50 border-l-4 border-red-500 p-4 rounded shadow-lg">
            <div className="flex">
              <div className="flex-1">
                <p className="text-sm text-red-700">{error}</p>
              </div>
              <button 
                onClick={() => setError(null)}
                className="text-red-700 hover:text-red-900"
              >
                ×
              </button>
              </div>
          </div>
        </Transition>

        {/* Input Area */}
        <div className={`${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        } border-t p-4`}>
          <div className="flex space-x-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              placeholder="Type your message..."
              className={`flex-1 p-3 rounded-lg ${
                isDark 
                  ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                  : 'bg-white border-gray-300 text-gray-900'
              } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
            />
            <button
              onClick={handleSendMessage}
              disabled={loading || !input.trim()}
              className={`px-4 py-2 rounded-lg ${
                isDark 
                  ? 'bg-blue-600 hover:bg-blue-700'
                  : 'bg-blue-600 hover:bg-blue-700'
              } text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors`}
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* New Session Modal */}
      <Transition
        show={showNewSession}
        enter="transition-opacity duration-300"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-300"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      >
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className={`${
            isDark ? 'bg-gray-800 text-white' : 'bg-white'
          } rounded-lg p-6 w-96`}>
            <h3 className="text-lg font-semibold mb-4">Create New Session</h3>
            <input
              type="text"
              value={newSessionName}
              onChange={(e) => setNewSessionName(e.target.value)}
              placeholder="Enter session name"
              className={`w-full p-2 rounded mb-4 ${
                isDark 
                  ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                  : 'border-gray-300'
              } border`}
            />
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowNewSession(false)}
                className={`px-4 py-2 rounded ${
                  isDark 
                    ? 'text-gray-300 hover:bg-gray-700'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Cancel
              </button>
              <button
                onClick={createNewSession}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      </Transition>
    </div>
  );
};

export default ChatApp;