# retrievers/hybrid.py
from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    QueryBundle,
    get_response_synthesizer
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import List

class CustomRetriever(BaseRetriever):
    """Custom retriever that combines vector and keyword search"""
    
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "OR"
    ) -> None:
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using both vector and keyword search"""
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

class HybridRetriever:
    """Hybrid retriever with reranking"""
    
    @staticmethod
    def get_query_engine(
        nodes: List,
        similarity_top_k: int = 10,
        rerank_top_n: int = 2,
        mode: str = "OR"
    ):
        # Create indices
        vector_index = VectorStoreIndex(nodes)
        keyword_index = SimpleKeywordTableIndex(nodes)
        
        # Create retrievers
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=similarity_top_k
        )
        keyword_retriever = KeywordTableSimpleRetriever(
            index=keyword_index
        )
        
        # Create custom hybrid retriever
        custom_retriever = CustomRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            mode=mode
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
            retriever=custom_retriever,
            node_postprocessors=[rerank],
            response_synthesizer=response_synthesizer
        )
        
        return query_engine