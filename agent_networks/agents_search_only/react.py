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

                no_ctx_vs_only = """You are a helpful assistant that can analyze documents. NEVER use prior knowledge.
                Available tool:
                - doc_search: Use to search through all documents with sentence-level chunking and reranking
                """

                agent = ReActAgent.from_tools(
                    tools=all_tools,
                    llm=self.llm,
                    verbose=True,
                    max_iterations=20,
                    context=no_ctx_vs_only,
                )

                return agent
            
        except Exception as e:
            self.logger.error(f"Error processing files: {e}")
            raise