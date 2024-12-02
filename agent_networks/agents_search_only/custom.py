from typing import List, Dict, Any, Optional, Tuple
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from logging_config import loggers, logger
from llama_index.core.agent import ReActAgent
from utils import *
from retrievers import RetrieverType, get_retriever

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
4. Action Input MUST be valid JSON with DOUBLE QUOTES around both key and value:
   CORRECT: {{"input": "query"}}
   WRONG: {{'input': 'query'}}
   WRONG: {{input: "query"}}
5. DO NOT add any extra text or annotations to Action or Action Input lines
6. DO NOT include backticks (```) or quotes around the format
7. DO NOT explain what you're doing - put that in the Thought
8. After Observation, ALWAYS start next step with "Thought: "

INVALID (will cause errors):
```
Thought: Let me search...
Action: Let me use doc_search
Action Input: {{'input': 'query'}}
```

VALID:
Thought: Let me search document for relevant information.

Action: doc_search

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
    def __init__(
        self, 
        model: str = None, 
        embeddings_model: str = None,
        retriever_type: RetrieverType = RetrieverType.HYBRID
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
        self.sentence_parser = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=50,
            separator=" ",
            paragraph_separator="\n\n",
            chunking_tokenizer_fn=None,
        )
        self.semantic_parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model
        )
        self.retriever_type = retriever_type
        
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
        try:
            all_tools = []
            all_nodes = []
            
            for file, content in files.items():
                if isinstance(content, pd.DataFrame):
                    continue  # Skip CSV files
                    
                self.logger.info(f"Processing {file}")
                nodes = self.sentence_parser.get_nodes_from_documents(content)
                all_nodes.extend(nodes)
            
            if all_nodes:
                # Get query engine using the modular retriever system
                query_engine = get_retriever(
                    self.retriever_type,
                    all_nodes,
                    similarity_top_k=20,
                    rerank_top_n=4
                )
                
                all_tools.extend([
                    QueryEngineTool(
                        query_engine=query_engine,
                        metadata=ToolMetadata(
                            name="doc_search",
                            description="Search through all documents for relevant information"
                        ),
                    ),
                ])

               
                ctx_vs_only = f"""You are a helpful assistant that can analyze documents.
                Available tool:
                - doc_search: Use to search through all documents with sentence-level chunking and reranking
                Important Guidelines:

                1. Search Strategy:
                a) Search using doc_search using THE WHOLE QUESTION for input
                b) Evaluate results with confidence score (0-1):
                    - Score represents ONLY how many keywords from the original question exist in the answer
                    - 0: No keywords from question found in retrieved text
                    - 0.1-0.3: Few minor keywords present
                    - 0.4-0.6: Some important keywords present
                    - 0.7-0.8: Most important keywords present
                    - 0.9-1.0: ALL keywords from question present in the answer
                    Example:
                    Question: "What are some of the key AI methods used in Computer Vision technology?"
                    Keywords to match: [key, AI methods, Computer Vision]
                    
                    Retrieved text: "AI techniques for image processing include CNNs"
                    Confidence: 0.3 (only general AI reference, missing methods and computer vision)
                    
                    Retrieved text: "The key AI methods include CNNs for computer vision tasks"
                    Confidence: 1.0 (contains: key AI methods, computer vision)

                c) If confidence < 0.9:
                    - Try alternative phrasing of the ORIGINAL QUESTION KEYWORDS 
                d) If confidence >= 0.9:
                    - COPY THE EXACT TEXT from the observation that answered the question
                    - DO NOT modify, rephrase, or synthesize the answer
                    - Be concise and if possible select the part of THE EXACT TEXT that contains ALL THE ORIGINAL QUESTION KEYWORDS

                2. Confidence Scoring Method:
                - Always start the first step with 0 confidence
                - Extract essential keywords from the question
                - Check if these EXACT keywords (or very close variants) appear in the retrieved text
                - 1.0 confidence ONLY if ALL question keywords are present
                - Lower confidence if important keywords are missing
                - Higher confidence only if ALL keywords are covered

                Example Final Responses:

                GOOD (all keywords present):
                Question: What AI methods are used in computer vision for object detection?
                Thought: Found answer containing all keywords (AI methods, computer vision, object detection). Confidence: 1.0
                Answer: Deep learning AI methods such as CNNs are the primary computer vision techniques used for object detection and recognition tasks.

                BAD (missing keywords):
                Question: What AI methods are used in computer vision for object detection?
                Thought: Found partial answer missing some keywords. Confidence: 0.7
                Answer: CNNs are used for detecting objects in images.

                GOOD:
                Question: How does federated learning ensure data privacy?
                Thought: Found text with all keywords (federated learning, data privacy). Confidence: 1.0
                Answer: Federated learning enables privacy-preserving machine learning by keeping all data local while only sharing model updates.

                BAD (with unnecessary additions):
                Question: How does federated learning ensure data privacy?
                Thought: Found relevant text. Confidence: 1.0
                Answer: According to the documentation, federated learning enables privacy-preserving machine learning by keeping all data local.


                Remember:
                - NEVER synthesize or rephrase answers
                - NEVER answer from prior knowledge
                - ALWAYS use exact text from documents
                - NEVER add phrases like "according to", "the text mentions", etc.
                - 1.0 confidence ONLY when ALL question keywords exist in the answer
                - Make multiple search attempts before concluding information isn't available
                - If multiple relevant pieces are found, use the one that most completely answers the question
                """

            agent = create_structured_react_agent(
                tools=all_tools,
                llm=self.llm,
                verbose=True,
                context=ctx_vs_only,
                max_iterations=20
            )

            return agent
            
        except Exception as e:
            self.logger.error(f"Error processing files: {e}")
            raise