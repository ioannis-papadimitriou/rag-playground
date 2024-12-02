import uuid
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'csv'}
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME', 'memory_agent'),
    'user': os.getenv('DB_USER', 'ollama_agent'),
    'password': os.getenv('DB_PASSWORD', 'AGRAGMEMNON'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432))
}

SAVE_FOLDER = os.environ.get('SAVE_FOLDER', './data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CHROMA_PERSIST_DIR = os.path.join(SAVE_FOLDER, "chromadb")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Model configurations
model_options = {
    "Llama3.1-8B": "llama3.1:8b-instruct-q6_K",
    "Llama3.1-1B": "llama3.2:1b" # use only for testing
}

embed_model_options = {
    "multilingual_baai": "BAAI/bge-m3",
    "multilingual_LaBSE": "sentence-transformers/LaBSE",
    "english_baai": "BAAI/bge-base-en-v1.5"
}

selected_model_name = "Llama3.1-8B"
selected_embed_model_name = "english_baai"
selected_model = model_options[selected_model_name]
selected_embed_model = embed_model_options[selected_embed_model_name]

embed_model = HuggingFaceEmbedding(
            model_name=selected_embed_model,
            trust_remote_code=True,
            cache_folder='./.cache'
        )

def file_name(name, extension):
    """
    Simple function to enable saving items in designated folder
    """
    unique_id = uuid.uuid4().hex  # Generate a unique identifier
    file_name = f"{name}_{unique_id}.{extension}"
    return os.path.join(SAVE_FOLDER, file_name)
