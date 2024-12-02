# retrievers/naive.py
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from typing import List

class NaiveRetriever:
    """Basic vector retriever without reranking"""
    
    @staticmethod
    def get_query_engine(nodes: List, similarity_top_k: int = 2):
        # Create vector index
        vector_index = VectorStoreIndex(nodes)
        
        # Create retriever
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=similarity_top_k
        )
        
        # Get response synthesizer
        response_synthesizer = get_response_synthesizer()
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            response_synthesizer=response_synthesizer
        )
        
        return query_engine