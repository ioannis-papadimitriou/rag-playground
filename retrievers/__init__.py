# retrievers/__init__.py
from enum import Enum
from typing import List

from .naive import NaiveRetriever
from .naive_reranker import NaiveRerankerRetriever
from .hybrid import HybridRetriever

class RetrieverType(Enum):
    NAIVE = "naive"
    NAIVE_RERANKER = "naive_reranker"
    HYBRID = "hybrid"

def get_retriever(retriever_type: RetrieverType, nodes: List, **kwargs):
    """Factory function to get the appropriate retriever"""
    if retriever_type == RetrieverType.NAIVE:
        return NaiveRetriever.get_query_engine(nodes, **kwargs)
    elif retriever_type == RetrieverType.NAIVE_RERANKER:
        return NaiveRerankerRetriever.get_query_engine(nodes, **kwargs)
    elif retriever_type == RetrieverType.HYBRID:
        return HybridRetriever.get_query_engine(nodes, **kwargs)
    else:
        raise ValueError(f"Invalid retriever type: {retriever_type}")