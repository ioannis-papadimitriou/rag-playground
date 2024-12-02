from typing import List, Dict, Any, Optional, Tuple
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    load_index_from_storage,
    StorageContext,
    Settings
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from logging_config import loggers, logger
from llama_index.llms.ollama import Ollama
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.agent import ReActAgent
from utils import *


from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.tools import BaseTool
from typing import Optional, Sequence

# Custom system header with our structured reasoning
STRUCTURED_REACT_SYSTEM_HEADER = """You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools

You have access to the following tools:
{tool_desc}

## Output Format

You MUST use this EXACT format and syntax for EVERY step:

Thought: Let me analyze step-by-step:
1. Current Status:
   - Progress: [what's been tried]
   - Findings: [what's been found]
   - Missing: [what's still needed]

2. Strategy:
   - Next tool: [selected tool]
   - Expected outcome: [what we hope to find]
   - Confidence: [current confidence score 0-1]
   - Reasoning: [why this choice]

Action: [tool name from: {tool_names}]
Action Input: {{"input": "your input here"}}


CRITICAL FORMAT RULES:
1. ALWAYS start with "Thought: "
2. ALWAYS have one empty line between Thought, Action, and Action Input
3. Action MUST be exactly one of the tool names - no extra text
4. Action Input MUST be valid JSON with "input" key
5. DO NOT add any extra text or annotations to Action or Action Input lines
6. DO NOT include backticks (```) or quotes around the format
7. DO NOT explain what you're doing - put that in the Thought
8. After Observation, ALWAYS start next step with "Thought: "

INVALID (will cause errors):
```
Thought: Let me search...
Action: Let me use doc_search_file1
Action Input: {{'input': 'query'}}
```

VALID:
Thought: Let me search document 1 for relevant information.

Action: doc_search_file1

Action Input: {{"input": "query"}}


After each Observation, ALWAYS evaluate with:

Thought: Analyzing the result:
1. Quality Assessment:
   - Information found: [summary]
   - Missing elements: [gaps]
   - Relevance: [how well it matches needs]
   - Confidence: [score 0-1]
   - Reasoning: [score justification]

2. Next Steps:
   - Decision: [what to do next]
   - Reasoning: [why this choice]

Continue until you have enough information or determine you cannot answer.

Then use ONLY:

Thought: [Final reasoning about answer quality]
Answer: [your response (final answer ONLY)]
{context}
"""

class StructuredReActChatFormatter(ReActChatFormatter):
    """ReAct formatter that enforces structured reasoning and systematic document search."""
    
    def __init__(
        self,
        system_header: Optional[str] = None,
        context: str = "",
    ) -> None:
        super().__init__(
            system_header=system_header or STRUCTURED_REACT_SYSTEM_HEADER,
            context=context
        )

# Helper function to create formatted agent
def create_structured_react_agent(
    tools: Sequence[BaseTool],
    llm: Any,
    context: str = "",
    verbose: bool = True,
    **kwargs
) -> ReActAgent:
    """Create ReAct agent with structured reasoning."""
    formatter = StructuredReActChatFormatter(context=context)
    return ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=verbose,
        react_chat_formatter=formatter,
        **kwargs
    )




class AgentNetwork:
    """Network of agents for document and data analysis"""
    def __init__(self, model: str = None, embeddings_model: str = None):
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
        self.sentence_parser = SentenceSplitter(
            chunk_size=256,     # Smaller chunks (default is 1024)
            chunk_overlap=50,   # Decent overlap to maintain context
            separator=" ",      # Split on spaces for more natural breaks
            paragraph_separator="\n\n", # Keep paragraph structure
            chunking_tokenizer_fn=None, # Use default for simplicity
        )
        self.semantic_parser = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model
        )
        
    def _create_csv_tools(self, file: str, df: pd.DataFrame) -> List[FunctionTool]:
        """Create tools for CSV analysis"""
        file_base = Path(file).stem
        
        def describe_data():
            return df.describe().to_string()
            
        def calculate_correlations():
            numeric_df = df.select_dtypes(include=[float, int])
            if numeric_df.empty:
                return "No numeric columns available for correlation analysis"
            
            corr_matrix = numeric_df.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Correlation Heatmap - {file}')
            
            plot_path = f"./data/{file_base}_correlation.png"
            plt.savefig(plot_path)
            plt.close()
            
            return f"Correlation matrix for {file}:\n{corr_matrix.to_string()}\nCorrelation plot saved as {plot_path}"
            
        def missing_data():
            missing_stats = df.isnull().sum()
            missing_pct = (missing_stats / len(df) * 100).round(2)
            return f"Missing data in {file}:\n" + \
                "\n".join([f"{col}: {pct}%" for col, pct in missing_pct.items() if pct > 0])
            
        def get_columns():
            return f"Columns in {file}:\n" + "\n".join([
                f"- {col} ({str(dtype)})" 
                for col, dtype in df.dtypes.items()
            ])

        def analyze_distribution():
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            if not len(numeric_cols):
                return "No numeric columns available for distribution analysis"
                
            result = []
            for col in numeric_cols:
                stats = df[col].describe()
                skew = df[col].skew()
                result.append(f"\n{col}:")
                result.append(f"- Skewness: {skew:.2f}")
                result.append(f"- Range: {stats['min']:.2f} to {stats['max']:.2f}")
                result.append(f"- Distribution: {'Normal' if abs(skew) < 0.5 else 'Skewed'}")
                
            return "\n".join(result)

        def detect_outliers():
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            if not len(numeric_cols):
                return "No numeric columns available for outlier detection"
                
            result = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
                if len(outliers):
                    result.append(f"\n{col}:")
                    result.append(f"- Number of outliers: {len(outliers)}")
                    result.append(f"- Outlier range: < {Q1 - 1.5*IQR:.2f} or > {Q3 + 1.5*IQR:.2f}")
                    
            return "\n".join(result) if result else "No significant outliers detected"

        return [
            FunctionTool.from_defaults(
                fn=describe_data,
                name=f"csv_describe_{file_base}",
                description=f"Get statistical description of the dataset '{file}'"
            ),
            FunctionTool.from_defaults(
                fn=calculate_correlations,
                name=f"csv_correlations_{file_base}",
                description=f"Calculate correlations between numeric columns in dataset '{file}'"
            ),
            FunctionTool.from_defaults(
                fn=missing_data,
                name=f"csv_missing_{file_base}",
                description=f"Analyze missing data in dataset '{file}'"
            ),
            FunctionTool.from_defaults(
                fn=get_columns,
                name=f"csv_columns_{file_base}",
                description=f"List all columns and their types in dataset '{file}'"
            ),
            FunctionTool.from_defaults(
                fn=analyze_distribution,
                name=f"csv_distribution_{file_base}",
                description=f"Analyze the distribution of numeric columns in dataset '{file}'"
            ),
            FunctionTool.from_defaults(
                fn=detect_outliers,
                name=f"csv_outliers_{file_base}",
                description=f"Detect outliers in numeric columns of dataset '{file}'"
            )
        ]
        
    async def process_files(self, files: Dict[str, Any]):
        """Process both document and CSV files"""
        try:
            all_tools = []
            file_descriptions = []
            
            for file, content in files.items():
                self.logger.info(f"Processing {file}")
                file_base = Path(file).stem
                
                if isinstance(content, pd.DataFrame):
                    # Handle CSV files
                    tools = self._create_csv_tools(file, content)
                    all_tools.extend(tools)
                    file_descriptions.append(
                        f"- Dataset '{file}': Use csv_* tools with suffix '_{file_base}'"
                    )
                    
                else:
                    # Handle documents
                    # Create both sentence-based and semantic nodes
                    nodes = self.sentence_parser.get_nodes_from_documents(content)
                    semantic_nodes = self.semantic_parser.get_nodes_from_documents(content)
                    
                    # Create or load vector indices
                    sentence_index_path = f"{UPLOAD_FOLDER}/{file}/sentence"
                    semantic_index_path = f"{UPLOAD_FOLDER}/{file}/semantic"
                    
                    # Create or load vector index
                    if not os.path.exists(sentence_index_path):
                        vector_index = VectorStoreIndex(nodes)
                        vector_index.storage_context.persist(persist_dir=sentence_index_path)
                    else:
                        vector_index = load_index_from_storage(
                            StorageContext.from_defaults(persist_dir=sentence_index_path)
                        )
                        
                    if not os.path.exists(semantic_index_path):
                        semantic_index = VectorStoreIndex(semantic_nodes)
                        semantic_index.storage_context.persist(persist_dir=semantic_index_path)
                    else:
                        semantic_index = load_index_from_storage(
                            StorageContext.from_defaults(persist_dir=semantic_index_path)
                        )
                    
                    # Create three different summary indices with different response modes
                    general_summary_index = SummaryIndex(
                        nodes,
                        response_synthesizer=get_response_synthesizer(
                            response_mode="tree_summarize",
                            summary_template=(
                                "Provide a high-level overview of the following text.\n"
                                "Text: {context_str}\n"
                                "Summary: "
                            )
                        )
                    )
                    
                    detailed_summary_index = SummaryIndex(
                        nodes,
                        response_synthesizer=get_response_synthesizer(
                            response_mode="compact_accumulate",
                            summary_template=(
                                "Analyze the following text and preserve important details.\n"
                                "Focus on: {query_str}\n"
                                "Text: {context_str}\n"
                                "Detailed Analysis: "
                            )
                        )
                    )
                    
                    section_summary_index = SummaryIndex(
                        nodes,
                        response_synthesizer=get_response_synthesizer(
                            response_mode="compact",
                            summary_template=(
                                "Focus on the specific section or aspect requested.\n"
                                "Request: {query_str}\n"
                                "Text: {context_str}\n"
                                "Response: "
                            )
                        )
                    )
                    from llama_index.core.postprocessor import SentenceTransformerRerank
                    rerank = SentenceTransformerRerank(
                        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                        # model="BAAI/bge-reranker-base",
                        top_n=4
                    )
                    all_tools.extend(
                        [
                        QueryEngineTool(
                            query_engine=vector_index.as_query_engine(
                                llm=self.llm,
                                # similarity_top_k=2,
                                similarity_top_k=20,  # Increase initial retrieval
                                node_postprocessors=[rerank]
                            ),
                            metadata=ToolMetadata(
                                name=f"doc_search_{file_base}",
                                description=(
                                    f"Search for specific information in document '{file}' "
                                    "using sentence-level chunking. Better for finding "
                                    "specific facts or statements."
                                )
                            ),
                        ),
                    #     QueryEngineTool(
                    #         query_engine=semantic_index.as_query_engine(
                    #             llm=self.llm,
                    #             similarity_top_k=2
                    #         ),
                    #         metadata=ToolMetadata(
                    #             name=f"doc_search_semantic_{file_base}",
                    #             description=(
                    #                 f"Search for information in document '{file}' using "
                    #                 "semantic chunking. Better for finding related concepts "
                    #                 "and maintaining contextual relationships."
                    #             )
                    #         ),
                    #     ),
                    #     QueryEngineTool(
                    #         query_engine=general_summary_index.as_query_engine(
                    #             llm=self.llm,
                    #             response_mode="tree_summarize"
                    #         ),
                    #         metadata=ToolMetadata(
                    #             name=f"doc_summarize_{file_base}",
                    #             description=(
                    #                 f"Get a high-level overview of document '{file}'. "
                    #                 "Best for quick understanding of main points."
                    #             )
                    #         ),
                    #     ),
                    #     QueryEngineTool(
                    #         query_engine=detailed_summary_index.as_query_engine(
                    #             llm=self.llm,
                    #             response_mode="compact_accumulate"
                    #         ),
                    #         metadata=ToolMetadata(
                    #             name=f"doc_summarize_detailed_{file_base}",
                    #             description=(
                    #                 f"Get a detailed analysis of document '{file}' "
                    #                 "preserving important specifics and nuances. "
                    #                 "Best for technical sections, methodologies, or "
                    #                 "when detail preservation is important."
                    #             )
                    #         ),
                    #     ),
                    #     QueryEngineTool(
                    #         query_engine=section_summary_index.as_query_engine(
                    #             llm=self.llm,
                    #             response_mode="compact"
                    #         ),
                    #         metadata=ToolMetadata(
                    #             name=f"doc_summarize_section_{file_base}",
                    #             description=(
                    #                 f"Get a focused summary of specific sections in document '{file}'. "
                    #                 "Best for targeting particular parts like conclusions, "
                    #                 "methodology, or specific topics."
                    #             )
                    #         ),
                    #     ),
                    # 
                    ])
                    
                    file_descriptions.append(
                        f"- Document '{file}':\n"
                        f"  * doc_search_{file_base} for precise sentence-level search\n"
                        # f"  * doc_search_semantic_{file_base} for context-aware semantic search\n"
                        # f"  * doc_summarize_{file_base} for high-level overviews\n"
                        # f"  * doc_summarize_detailed_{file_base} for detail-preserving analysis\n"
                        # f"  * doc_summarize_section_{file_base} for specific sections"
                    )
            
            # Default prompt
            # context = f"""You are a helpful assistant that can analyze documents and datasets.
            #             Available files and their tools:
            #             {chr(10).join(file_descriptions)}

            #             Important Guidelines:
            #             1. Default Approach: Use multiple complementary tools to validate and enrich your answers unless you have extremely high confidence in a single tool's response.

            #             2. For document search and analysis:
            #             - Start with both doc_search_* and doc_search_semantic_* to:
            #                 * Get both precise matches (doc_search_*) and contextual matches (doc_search_semantic_*)
            #                 * Cross-validate information across different chunking methods
            #                 * Ensure no important context is missed

            #             - Follow with appropriate summarization tools:
            #                 * doc_summarize_* for high-level context
            #                 * doc_summarize_detailed_* when technical details matter
            #                 * doc_summarize_section_* for focused analysis
                        
            #             - Default Action for Information Gaps: If initial doc_search_* or doc_search_semantic_* tools on one document do not yield sufficient answers or high-confidence results, immediately switch to the next document and apply the same initial tools there.
                        
            #             - Tool Reuse Condition: If a tool yields low-confidence or insufficient results on a given document, avoid reapplying that same tool on the same document. Instead, use the tool on another document unless explicitly instructed to focus on that document alone.

            #             - Fallback Sequence: When confidence is low after an initial search pass in one document:
            #                 1. Apply the same search tools (doc_search_* and doc_search_semantic_*) on a different document.
            #                 2. Only return to the first document for analysis if confidence remains low after trying all other documents.

            #             - Confidence-Based Switching: If the confidence level remains below 70% after the initial doc_search_* and doc_search_semantic_* on a document, switch to the next document to ensure comprehensive coverage and avoid single-document fixation.

            #             - Confidence Check: After each tool application, assess confidence in the findings. If confidence is low after two consecutive tool applications on the same document, switch to a different document and repeat the initial tool set.
                        
            #             Only skip this comprehensive approach if:
            #             - The query is extremely simple and specific
            #             - A single tool has provided a complete, verifiable answer
            #             - You are explicitly asked to use only specific tools

            #             3. For datasets:
            #             - Begin with csv_columns_* to understand the data structure
            #             - Use multiple analysis tools (describe, correlations, distributions, etc.)
            #             - Only skip additional analysis if the query targets a very specific metric

            #             Tool Selection Guide:
            #             1. Search Tools:
            #             - doc_search_*: For specific facts, quotes, or precise matches. If results are inconclusive or confidence is low, ALWAYS apply the same tool to other documents (e.g., if unsure with doc_search_doc1, try doc_search_doc2)
            #             - doc_search_semantic_*: For related concepts, themes, and contextual understanding. Apply to multiple documents when confidence is low or when information verification is needed across sources.

            #             2. Summarization Tools:
            #             - doc_summarize_*: Quick overviews and main points
            #             - doc_summarize_detailed_*: Technical details, methodologies, specific findings
            #             - doc_summarize_section_*: Focused analysis of specific parts

            #             3. Dataset Tools:
            #             - csv_describe_*: Statistical summaries
            #             - csv_correlations_*: Relationship analysis
            #             - csv_missing_*: Data quality assessment
            #             - csv_columns_*: Data structure
            #             - csv_distribution_*: Data distribution analysis
            #             - csv_outliers_*: Anomaly detection

            #             ALWAYS:
            #             - Replace * ONLY with the specific file suffix
            #             - Use exact tool names
            #             - Explain your tool selection strategy
            #             - Present findings from each tool used
            #             - Synthesize a comprehensive response
            #             - Highlight any discrepancies found between different tools
            #             - Indicate confidence level in your final response

            #             When in doubt, prefer using more tools rather than fewer, including cross-referencing across multiple documents if confidence is low or if the answer appears incomplete."""

            # Full suite custom prompt
            # context = f"""You are a helpful assistant that can analyze documents and datasets.
            #             Available files and their tools:
            #             {chr(10).join(file_descriptions)}

            #             Important Guidelines:

            #             1. Document Search Strategy:
            #             ALWAYS follow these exact steps:
            #             a) Start with doc_search_* on first document
            #             b) Evaluate results with confidence score (0-1):
            #                 - Score ONLY represents confidence in FINDING the answer, NOT confidence in its absence
            #                 - 0: No relevant information found
            #                 - 0.1-0.3: Found tangentially related information
            #                 - 0.4-0.6: Found partially relevant information but missing key elements
            #                 - 0.7-0.8: Found most of the required information with some gaps
            #                 - 0.9-1.0: Found complete or nearly complete answer
            #                 - IMPORTANT: If you're confident the information is NOT in the document, score should be 0, NOT high confidence
            #                 - NEVER proceed to final answer if confidence is due to certainty of absence
            #             c) If confidence < 0.95:
            #                 - Try doc_search_semantic_* on same document
            #                 - If still < 0.95, move to next document
            #             d) Track highest confidence answer seen across all documents

            #             2. Confidence Scoring Rules:
            #             - Confidence MUST be based on finding the requested information
            #             - Being certain information is absent = 0 confidence
            #             - Always start the first step with 0 confidence
            #             - Only give high confidence (>0.8) when you actually find the requested information
            #             - Partial matches should be scored based on how much of the required information was found
            #             - Before giving final answer, confirm that high confidence comes from finding information, not absence of it

            #             3. Summarization Tool Usage:
            #             doc_summarize_* tools should ONLY be used when:
            #             - User explicitly requests a summary/overview
            #             - OR all documents searched with confidence < 0.95

            #             4. For datasets:
            #             - Begin with csv_columns_* to understand structure
            #             - Use analysis tools based on specific needs

            #             Tool Selection Guide:
            #             1. Primary Search Tools:
            #             - doc_search_*: For specific facts, quotes, precise matches
            #             - doc_search_semantic_*: For related concepts and context

            #             2. Restricted Summarization Tools:
            #             - doc_summarize_*: ONLY for requested overviews or as last resort
            #             - doc_summarize_detailed_*: ONLY for requested technical details
            #             - doc_summarize_section_*: ONLY for requested section analysis

            #             3. Dataset Tools:
            #             - csv_describe_*: Statistical summaries
            #             - csv_correlations_*: Relationship analysis
            #             - csv_missing_*: Data quality assessment
            #             - csv_distribution_*: Data distribution analysis
            #             - csv_outliers_*: Anomaly detection

            #             Remember:
            #             - ALWAYS start with doc_search_*
            #             - NEVER skip searching a document if previous confidence < 0.95
            #             - NEVER answer from prior knowledge
            #             - NEVER determine you cannot answer unless you have searched all available documents
            #             - Track and report confidence scores for every tool use
            #             - Only use summarization tools if explicitly requested or as last resort
            #             - Confidence of 0 means "no relevant information found", not "confident information is absent"
            #             - Before giving final answer, verify that high confidence means "found the information" not "confident it's not here
            #             """

            # Vsearch only custom prompt
            context = f"""You are a helpful assistant that can analyze documents.
            Available files and their tools:
            {chr(10).join(file_descriptions)}

            Important Guidelines:

            1. Document Search Strategy:
            ALWAYS follow these exact steps:
            a) Start with doc_search_* on first document
            b) Evaluate results with confidence score (0-1):
                - Score ONLY represents confidence in FINDING the answer, NOT confidence in its absence
                - 0: No relevant information found
                - 0.1-0.3: Found tangentially related information
                - 0.4-0.6: Found partially relevant information but missing key elements
                - 0.7-0.8: Found most of the required information with some gaps
                - 0.9-1.0: Found complete or nearly complete answer
                - IMPORTANT: If you're confident the information is NOT in the document, score should be 0, NOT high confidence
                - NEVER proceed to final answer if confidence is due to certainty of absence
            c) If confidence < 0.9:
                - Move to next document
            d) Track highest confidence answer seen across all documents

            2. Response Format Rules:
            For the final answer:
            - If confidence >= 0.9:
                * BE VERY CONCISE (keep the response to ONE AS SMALL AS POSSIBLE sentence) while giving a COMPLETE answer

            3. Confidence Scoring Rules:
            - Confidence MUST be based on finding the requested information
            - Being certain information is absent = 0 confidence
            - Always start the first step with 0 confidence
            - Only give high confidence (>0.8) when you actually find the requested information
            - Partial matches should be scored based on how much of the required information was found
            - Before giving final answer, confirm that high confidence comes from finding information, not absence of it

            Tool Selection Guide:
            - doc_search_*: Use for all searches including specific facts, quotes, precise matches, and related concepts
            - Search with variations of keywords to maximize chance of finding relevant information

            Remember:
            - ALWAYS start with first document
            - NEVER skip searching a document if previous confidence < 0.9
            - NEVER answer from prior knowledge
            - NEVER determine you cannot answer unless you have searched all available documents
            - Track and report confidence scores for every search
            - Confidence of 0 means "no relevant information found", not "confident information is absent"
            - Before giving final answer, verify that high confidence means "found the information" not "confident it's not here"
            """

            # agent = ReActAgent.from_tools(
            #     tools=all_tools,
            #     llm=self.llm,
            #     verbose=True,
            #     context=context,
            #     max_iterations=2*len(all_tools)
            # )
            agent = create_structured_react_agent(
                tools=all_tools,
                llm=self.llm,
                verbose=True,
                context=context,
                max_iterations=3*len(all_tools)
            )

            return agent
            
        except Exception as e:
            self.logger.error(f"Error processing files: {e}")
            raise