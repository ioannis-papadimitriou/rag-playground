from flask import Flask, request, jsonify, session
from flask_cors import CORS
import uuid
import pandas as pd
import json
from pathlib import Path
from werkzeug.utils import secure_filename
from llama_index.core import SimpleDirectoryReader

from agent_networks import AgentNetwork
from logging_config import logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'  # Change in production

# Configuration
UPLOAD_FOLDER = Path('uploads')
SESSION_FOLDER = Path('sessions')
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.csv'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
SESSION_FOLDER.mkdir(exist_ok=True)

# Global state
sessions = {}
agent_networks = {}

from typing import List, Optional
import pdfplumber
def preprocess_pdf(file_path: str) -> Optional[List[str]]:
    """Extract clean text from PDF while preserving basic structure"""
    pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text)
    except Exception as e:
        logger.error(f"Error preprocessing PDF {file_path}: {e}")
        return None
    return pages if pages else None

def save_session_data(session_id: str, data: dict):
    """Save session data to disk"""
    session_file = SESSION_FOLDER / f"{session_id}.json"
    with open(session_file, 'w') as f:
        # Create serializable copy of session data
        session_data = {
            'name': data.get('name', ''),
            'messages': data.get('messages', []),
            'files': data.get('files', [])
        }
        json.dump(session_data, f)

def load_session_data(session_id: str) -> dict:
    """Load session data from disk"""
    session_file = SESSION_FOLDER / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, 'r') as f:
            return json.load(f)
    return {'name': '', 'messages': [], 'files': []}

def initialize_base_agent():
    from llama_index.llms.ollama import Ollama
    """Initialize or get the base LLM for general queries"""
    return  Ollama(
            model=session.get('model', 'qwen2.5:14b'),
            request_timeout=120.0,
            temperature=0.1
        )


async def recreate_agent_network(session_id: str) -> bool:
    """Recreate agent network from stored files"""
    try:
        session_upload_dir = UPLOAD_FOLDER / session_id
        if not session_upload_dir.exists():
            return False

        # Load session data to get file list
        session_data = load_session_data(session_id)
        if not session_data.get('files'):
            return False

        # Initialize agent network
        network = AgentNetwork(
            model=session.get('model', 'qwen2.5:14b'),
            embeddings_model=session.get('embeddings_model', 'BAAI/bge-base-en-v1.5')
        )

        # Process stored files
        file_contents = {}
        for filename in session_data['files']:
            file_path = session_upload_dir / filename
            if file_path.exists():
                if file_path.suffix.lower() == '.csv':
                    file_contents[filename] = pd.read_csv(file_path)
                else:
                    docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                    file_contents[filename] = docs

        if file_contents:
            # Process files and store top agent
            top_agent = await network.process_files(file_contents)
            agent_networks[session_id] = top_agent
            return True

        return False

    except Exception as e:
        logger.error(f"Error recreating agent network: {e}")
        return False

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "RAG API is running"})

@app.route('/chat', methods=['GET', 'POST'])
async def chat():
    """Handle chat interactions"""
    try:
        # Initialize session if needed
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']
        
        # Load or initialize session data
        if session_id not in sessions:
            sessions[session_id] = load_session_data(session_id)
        
        if request.method == 'POST':
            if 'send_query' in request.form:
                # Handle chat message
                query = request.form['query']
                logger.info(f"Processing query: {query}")
                
                try:                    
                    # Determine which agent to use based on whether files are present
                    has_files = bool(sessions[session_id].get('files', []))
                    
                    # If there are files but no agent_network, try to recreate it
                    if has_files and session_id not in agent_networks:
                        success = await recreate_agent_network(session_id)
                        has_files = success  # Update based on recreation success
                    
                    # Use appropriate agent
                    if has_files and session_id in agent_networks:
                        logger.info(f"Using document-specific agent for session {session_id}")
                        response = agent_networks[session_id].query(query)
                    else:
                        logger.info(f"Using base agent for session {session_id}")
                        base_llm = initialize_base_agent()
                        response = base_llm.complete(query).text
                    
                    # Update messages
                    new_messages = [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": str(response)}
                    ]
                    sessions[session_id]['messages'].extend(new_messages)
                    
                    # Update session name if it's the first message
                    if len(sessions[session_id]['messages']) == 2:
                        sessions[session_id]['name'] = query[:50]
                    
                    # Save session data
                    save_session_data(session_id, sessions[session_id])
                    
                    return jsonify({
                        "success": True,
                        "messages": new_messages
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    return jsonify({"success": False, "message": str(e)})
                    
            elif 'upload' in request.form:
                # Handle file upload
                files = request.files.getlist('file')
                uploaded_files = []
                
                try:
                    # Create session upload directory
                    session_upload_dir = UPLOAD_FOLDER / session_id
                    session_upload_dir.mkdir(exist_ok=True)
                    
                    # Process all files
                    file_contents = {}
                    for file in files:
                        if file and file.filename:
                            filename = secure_filename(file.filename)
                            if not Path(filename).suffix.lower() in ALLOWED_EXTENSIONS:
                                continue
                                
                            file_path = session_upload_dir / filename
                            file.save(file_path)
                            uploaded_files.append(file_path)
                            
                            # Load file content based on type
                            if file_path.suffix.lower() == '.csv':
                                file_contents[filename] = pd.read_csv(file_path)
                            else:
                                # docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                                # file_contents[filename] = docs
                                pages = preprocess_pdf(str(file_path))
                                logger.info(f"Processed sample {pages[0]}")
                                if pages:
                                    from llama_index.core import Document
                                    file_contents[filename] = [Document(text=page) for page in pages]
                                else:
                                    # Fallback to SimpleDirectoryReader
                                    docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                                    file_contents[filename] = docs
                            
                    if file_contents:
                        # Initialize agent network with both models from settings
                        model = session.get('model', 'qwen2.5:14b')
                        embeddings_model = session.get('embeddings_model', 'BAAI/bge-base-en-v1.5')
                        network = AgentNetwork(
                            model=model,
                            embeddings_model=embeddings_model
                        )
                        
                        # Process files and get top agent
                        top_agent = await network.process_files(file_contents)
                        agent_networks[session_id] = top_agent
                        
                        # Update session data
                        sessions[session_id]['files'].extend(list(file_contents.keys()))
                        save_session_data(session_id, sessions[session_id])
                        
                        return jsonify({
                            "success": True,
                            "message": "Files processed successfully",
                            "files": sessions[session_id]['files']
                        })
                    else:
                        return jsonify({
                            "success": False,
                            "message": "No valid files uploaded"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing files: {e}")
                    # Cleanup on error
                    for file_path in uploaded_files:
                        try:
                            if file_path.exists():
                                file_path.unlink()
                        except Exception as cleanup_error:
                            logger.error(f"Error during cleanup: {cleanup_error}")
                    return jsonify({"success": False, "message": str(e)})
        
        # GET request - return session data
        chat_sessions = []
        for file in SESSION_FOLDER.glob('*.json'):
            session_data = load_session_data(file.stem)
            chat_sessions.append({
                'id': file.stem,
                'name': session_data.get('name', f'Session {file.stem[:8]}')
            })
            
        return jsonify({
            'chat_sessions': chat_sessions,
            'current_session': session_id,
            'chat_history': sessions[session_id]['messages'],
            'session_name': sessions[session_id].get('name', ''),
            'indexed_files': sessions[session_id].get('files', [])
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Handle model settings"""
    if request.method == 'POST':
        try:
            model = request.form.get('model', 'qwen2.5:14b')
            embeddings_model = request.form.get('embeddings_model', 'BAAI/bge-base-en-v1.5')
            
            # Store settings in session
            session['model'] = model
            session['embeddings_model'] = embeddings_model
            
            return jsonify({
                "success": True,
                "message": "Settings updated successfully"
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            })
            
    # GET request
    return jsonify({
        'settings': {
            'model': session.get('model', 'qwen2.5:14b'),
            'embeddings_model': session.get('embeddings_model', 'BAAI/bge-base-en-v1.5')
        }
    })

@app.route('/new_session', methods=['GET', 'POST'])
def new_session():
    """Create a new chat session"""
    try:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        sessions[session_id] = {'name': '', 'messages': [], 'files': []}
        save_session_data(session_id, sessions[session_id])
        return jsonify({"success": True, "session_id": session_id})
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/switch_session/<session_id>')
async def switch_session(session_id):
    """Switch to a different session"""
    try:
        if session_id not in sessions:
            sessions[session_id] = load_session_data(session_id)
        
        # Recreate agent network if needed
        if session_id not in agent_networks:
            await recreate_agent_network(session_id)
            
        session['session_id'] = session_id
        return jsonify({"success": True, "session_id": session_id})
    except Exception as e:
        logger.error(f"Error switching session: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/delete_session/<session_id>', methods=['POST'])
def delete_session(session_id):
    """Delete a chat session"""
    try:
        # Remove session file
        session_file = SESSION_FOLDER / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            
        # Remove uploaded files
        session_upload_dir = UPLOAD_FOLDER / session_id
        if session_upload_dir.exists():
            import shutil
            shutil.rmtree(session_upload_dir)
            
        # Clean up memory
        sessions.pop(session_id, None)
        agent_networks.pop(session_id, None)
        
        # Reset current session if needed
        if session.get('session_id') == session_id:
            session['session_id'] = str(uuid.uuid4())
            
        return jsonify({"success": True, "message": "Session deleted"})
        
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)