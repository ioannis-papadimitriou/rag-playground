"""
Select which agent implementation to use by changing the import below.
Current options:
- agents_full_suite_tool_per_doc
- agents_full_suite
- agents_search_only
- naive_rag
"""

# Choose your implementation here:

# Agentic Vector Search only
# from .agents_search_only.custom import AgentNetwork
# from .agents_search_only.react import AgentNetwork

# Agentic Full suite
from .agents_full_suite.react import AgentNetwork

# Naive RAG
# from .naive.naive import AgentNetwork
# from .naive.naive_reranker import AgentNetwork
# from .naive.hybrid_search import AgentNetwork