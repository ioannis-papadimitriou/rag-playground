import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import argparse
import time

from llama_index.core import SimpleDirectoryReader
import pandas as pd

from agent_networks import AgentNetwork
from logging_config import logger

class ResponseGenerator:
    """Generates and records agent responses for curated QA pairs"""
    
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        embeddings_model: str = "BAAI/bge-base-en-v1.5",
        docs_dir: str = "docs"
    ):
        self.docs_dir = Path(docs_dir)
        self.agent_network = None
        self.model = model
        self.embeddings_model = embeddings_model
        
    async def initialize_agent(self) -> bool:
        """Initialize agent with all documents in docs directory"""
        try:
            self.agent_network = AgentNetwork(
                model=self.model,
                embeddings_model=self.embeddings_model
            )
            
            file_contents = {}
            for file_path in self.docs_dir.glob('*'):
                if file_path.suffix.lower() == '.csv':
                    file_contents[file_path.name] = pd.read_csv(file_path)
                else:
                    docs = SimpleDirectoryReader(
                        input_files=[str(file_path)]
                    ).load_data()
                    file_contents[file_path.name] = docs
            
            if not file_contents:
                logger.error(f"No files found in {self.docs_dir}")
                return False
            
            self.top_agent = await self.agent_network.process_files(file_contents)
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            return False

    def update_qa_file(self, qa_pair: Dict, output_file: str):
        """Update a single QA pair in the JSON file"""
        try:
            # Read current state
            with open(output_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            # Update the specific QA pair
            for i, pair in enumerate(qa_data['qa_pairs']):
                if pair['id'] == qa_pair['id']:
                    qa_data['qa_pairs'][i] = qa_pair
                    break
            
            # Write updated data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error updating QA file: {e}")
            raise

    def needs_processing(self, qa_pair: Dict) -> bool:
        """Check if a QA pair needs processing"""
        if not qa_pair['curation_status']['is_approved']:
            return False
        if 'agent_answer' in qa_pair and not qa_pair.get('agent_error'):
            return False
        return True
    
    async def process_qa_pairs(self, qa_file: str, output_file: str = None):
        """Process all approved QA pairs and record agent responses"""
        try:
            # Load QA pairs
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            output_file = output_file or qa_file.replace('.json', '_with_responses.json')
            
            # Create initial output file if it doesn't exist
            if not Path(output_file).exists():
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(qa_data, f, indent=2, ensure_ascii=False)
            
            # Initialize agent if needed
            if not self.agent_network:
                success = await self.initialize_agent()
                if not success:
                    raise Exception("Failed to initialize agent")
            
            # Count total approved pairs
            total_approved = sum(1 for qa in qa_data['qa_pairs'] 
                               if qa['curation_status']['is_approved'])
            current_approved = 0
            
            # Process each approved QA pair
            for qa_pair in qa_data['qa_pairs']:
                if not qa_pair['curation_status']['is_approved']:
                    continue
                
                current_approved += 1
                
                if not self.needs_processing(qa_pair):
                    logger.info(f"Skipping question {current_approved}/{total_approved} (already processed)")
                    continue
                
                logger.info(f"Processing question {current_approved}/{total_approved}: {qa_pair['question']}")
                
                try:
                    # Get agent response
                    response = self.top_agent.query(qa_pair['question'])
                    
                    # Extract evaluation-relevant data
                    source_nodes = []
                    if hasattr(response, 'source_nodes'):
                        source_nodes = [
                            {
                                'text': node.node.text,
                                'metadata': node.node.metadata,
                                'score': float(node.score) if hasattr(node, 'score') else None
                            }
                            for node in response.source_nodes
                        ]
                    
                    reasoning_steps = []
                    if hasattr(response, 'intermediate_steps'):
                        reasoning_steps = response.intermediate_steps
                    
                    # Update QA pair with agent data
                    qa_pair.update({
                        'agent_answer': str(response),
                        'agent_timestamp': datetime.now().isoformat(),
                        'agent_metadata': {
                            'model': self.model,
                            'embeddings_model': self.embeddings_model
                        },
                        'agent_source_nodes': source_nodes,
                        'agent_reasoning_steps': reasoning_steps,
                        'agent_raw_response': str(response),
                    })
                    logger.info("Response generated successfully")
                    
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    qa_pair.update({
                        'agent_error': str(e),
                        'agent_timestamp': datetime.now().isoformat()
                    })
                
                # Update the JSON file immediately after each response
                self.update_qa_file(qa_pair, output_file)
                logger.info("Progress saved to JSON")
                logger.info("-" * 40)
                
                # Small delay to prevent file access conflicts
                await asyncio.sleep(0.1)
            
            logger.info(f"Processing complete. Processed {total_approved} questions.")
            
        except Exception as e:
            logger.error(f"Error processing QA pairs: {e}")
            raise

async def main():
    parser = argparse.ArgumentParser(
        description="Generate agent responses for curated QA pairs"
    )
    parser.add_argument(
        '--qa-file',
        '-q',
        required=True,
        help='Path to curated QA pairs JSON file'
    )
    parser.add_argument(
        '--docs-dir',
        '-d',
        required=True,
        help='Directory containing documents to process'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output file path (default: input_with_responses.json)'
    )
    parser.add_argument(
        '--model',
        '-m',
        default='qwen2.5:14b',
        help='Model to use (default: qwen2.5:14b)'
    )
    parser.add_argument(
        '--embeddings-model',
        default='BAAI/bge-base-en-v1.5',
        help='Embeddings model to use (default: BAAI/bge-base-en-v1.5)'
    )
    
    args = parser.parse_args()
    
    generator = ResponseGenerator(
        model=args.model,
        embeddings_model=args.embeddings_model,
        docs_dir=args.docs_dir
    )
    
    await generator.process_qa_pairs(args.qa_file, args.output)

if __name__ == "__main__":
    asyncio.run(main())
    # Usage: python response_generator.py -q ./curated_qa/llama3.1_naive/qa_pairs_curated.json -d ./qa_docs/
    # Usage: python response_generator.py -q ./curated_summaries/qwen_hybrid_search/summary_qa_pairs_curated.json -d ./qa_docs/