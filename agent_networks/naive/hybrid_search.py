from typing import List, Dict, Any, Optional, Tuple
import os
import pandas as pd

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from logging_config import loggers, logger
from utils import *
from retrievers import RetrieverType, get_retriever  # Add this import

class AgentNetwork:
    """Simplified RAG system for document analysis"""
    def __init__(
        self, 
        model: str = None, 
        embeddings_model: str = None,
        retriever_type: RetrieverType = RetrieverType.HYBRID  # Add this parameter
    ):
        # Initialize models
        self.llm = Ollama(
            model=model or "qwen2.5:14b", 
            request_timeout=120.0,
            temperature=0.1
        )
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=embeddings_model or "BAAI/bge-base-en-v1.5",
            trust_remote_code=True,
            cache_folder='./.cache'
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.logger = loggers['agent_network']
        self.sentence_parser = SentenceSplitter()
        self.retriever_type = retriever_type  # Add this line
        
    async def process_files(self, files: Dict[str, Any]):
        """Process documents and create merged query engine"""
        try:
            all_nodes = []
            
            for file, content in files.items():
                if isinstance(content, pd.DataFrame):
                    continue  # Skip CSV files
                    
                self.logger.info(f"Processing {file}")
                
                # Create nodes from document
                nodes = self.sentence_parser.get_nodes_from_documents(content)
                all_nodes.extend(nodes)
            
            if all_nodes:
                # Replace all the index/retriever/query_engine creation with this single line
                query_engine = get_retriever(
                    self.retriever_type, 
                    all_nodes,
                    similarity_top_k=20,  # Optional: pass configuration parameters
                    rerank_top_n=4        # Optional: pass configuration parameters
                )
                return query_engine
            else:
                raise ValueError("No valid documents to process")
            
        except Exception as e:
            self.logger.error(f"Error processing files: {e}")
            raise