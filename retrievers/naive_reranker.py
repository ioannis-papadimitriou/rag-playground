# retrievers/naive_reranker.py
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import List

class NaiveRerankerRetriever:
    """Vector retriever with reranking"""
    
    @staticmethod
    def get_query_engine(nodes: List, similarity_top_k: int = 10, rerank_top_n: int = 2):
        # Create vector index
        vector_index = VectorStoreIndex(nodes)
        
        # Create retriever
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=similarity_top_k
        )
        
        # Create reranker
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=rerank_top_n
        )
        
        # Get response synthesizer
        response_synthesizer = get_response_synthesizer()
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            node_postprocessors=[rerank],
            response_synthesizer=response_synthesizer
        )
        
        return query_engine