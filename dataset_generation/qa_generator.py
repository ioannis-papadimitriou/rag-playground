from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import asyncio
from datetime import datetime
import argparse
import logging
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedQA:
    """Data class for extracted QA pairs"""
    question: str
    answer: str
    context: str
    metadata: Dict[str, Any]
    confidence: float
    extraction_timestamp: str

@dataclass
class CuratedQA(ExtractedQA):
    """Data class for curated QA pairs"""
    curator: str
    curation_notes: str
    curation_timestamp: str
    is_approved: bool
    modifications: Dict[str, str]  # original->modified mappings

class QAExtractionPipeline:
    """Extracts QA pairs from documents using LLM"""
    
    def __init__(
        self,
        model: Ollama,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def extract_qa_pairs(
        self,
        documents: List[Document]
    ) -> List[ExtractedQA]:
        """Extract QA pairs from documents"""
        qa_pairs: List[ExtractedQA] = []
        
        for doc in documents:
            # Split text into chunks based on size
            text = doc.text
            chunks = [text[i:i + self.chunk_size] 
                     for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            
            for chunk_idx, chunk in enumerate(chunks):
                # Generate QA pairs for chunk
                chunk_pairs = await self._generate_qa_pairs(
                    chunk,
                    {
                        "document_id": doc.doc_id,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        **doc.metadata
                    }
                )
                qa_pairs.extend(chunk_pairs)
        
        return qa_pairs
    
    async def _generate_qa_pairs(
        self,
        chunk: str,
        metadata: Dict[str, Any]
    ) -> List[ExtractedQA]:
        """Generate QA pairs for a single chunk"""
        prompt = f"""Generate detailed, specific question-answer pairs from the following text.
        Focus on factual information and key concepts. Each QA pair should be answerable
        solely from the provided context.

        Text: {chunk}

        Generate 2-3 QA pairs in the following JSON format:
        {{
            "qa_pairs": [
                {{
                    "question": "Specific, detailed question",
                    "answer": "Precise answer from the text",
                    "confidence": <float between 0 and 1>
                }}
            ]
        }}
        """
        
        try:
            response = self.model.complete(prompt)
            response_text = response.text
            
            # Add basic validation and cleaning of the response
            if not response_text.strip():
                logger.warning(f"Empty response received for chunk")
                return []
                
            # Try to extract JSON portion if response contains additional text
            try:
                # Find the first { and last } to extract JSON
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]
                    
                qa_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {response_text}")
                return []
            
            if not isinstance(qa_data, dict) or 'qa_pairs' not in qa_data:
                logger.error("Response missing required 'qa_pairs' field")
                return []
                
            qa_pairs = []
            for qa in qa_data['qa_pairs']:
                # Validate required fields
                if not all(k in qa for k in ('question', 'answer', 'confidence')):
                    continue
                    
                # Validate confidence is float between 0 and 1
                try:
                    confidence = float(qa['confidence'])
                    if not 0 <= confidence <= 1:
                        confidence = 0.5  # fallback to middle value
                except (TypeError, ValueError):
                    confidence = 0.5
                
                qa_pairs.append(
                    ExtractedQA(
                        question=qa['question'],
                        answer=qa['answer'],
                        context=chunk,
                        metadata=metadata,
                        confidence=confidence,
                        extraction_timestamp=datetime.now().isoformat()
                    )
                )
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error generating QA pairs for chunk: {e}")
            return []
        
class QACurationTool:
    """Tool for human curation of extracted QA pairs"""
    
    def __init__(self, output_dir: str = "curated_qa"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_curation_template(
        self,
        qa_pairs: List[ExtractedQA],
        output_file: str
    ):
        """Create template for human curation"""
        output_data = {
            "instructions": """
            Curation Instructions:
            1. Review each QA pair and its context
            2. For each pair, update the curation_status:
               - Set is_approved to true/false
               - Add your name as curator
               - Add any notes about modifications
               - If modifying, update modified_question/answer
               - Ensure answers are fully supported by context
            3. Save the file with _curated suffix
            """,
            "qa_pairs": [
                {
                    "id": idx,
                    "question": qa.question,
                    "answer": qa.answer,
                    "context": qa.context,
                    "metadata": {
                        **qa.metadata,
                        "confidence": qa.confidence,
                        "extraction_timestamp": qa.extraction_timestamp
                    },
                    "curation_status": {
                        "is_approved": False,
                        "curator": "",
                        "notes": "",
                        "modified_question": "",
                        "modified_answer": ""
                    }
                }
                for idx, qa in enumerate(qa_pairs)
            ]
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Curation template saved to {output_path}")
        
    def process_curated_file(
        self,
        input_file: str
    ) -> List[CuratedQA]:
        """Process curated file into CuratedQA objects"""
        input_path = self.output_dir / input_file
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        curated_pairs: List[CuratedQA] = []
        for qa in data['qa_pairs']:
            if not qa['curation_status']['is_approved']:
                continue
                
            modifications = {}
            if qa['curation_status']['modified_question']:
                modifications['question'] = {
                    'original': qa['question'],
                    'modified': qa['curation_status']['modified_question']
                }
            if qa['curation_status']['modified_answer']:
                modifications['answer'] = {
                    'original': qa['answer'],
                    'modified': qa['curation_status']['modified_answer']
                }
            
            curated_pairs.append(
                CuratedQA(
                    question=qa['curation_status']['modified_question'] or qa['question'],
                    answer=qa['curation_status']['modified_answer'] or qa['answer'],
                    context=qa['context'],
                    metadata=qa['metadata'],
                    confidence=qa['metadata']['confidence'],
                    extraction_timestamp=qa['metadata']['extraction_timestamp'],
                    curator=qa['curation_status']['curator'],
                    curation_notes=qa['curation_status']['notes'],
                    curation_timestamp=datetime.now().isoformat(),
                    is_approved=True,
                    modifications=modifications
                )
            )
        
        return curated_pairs

async def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from documents"
    )
    parser.add_argument(
        "--input", 
        "-i", 
        required=True,
        help="Input directory containing documents"
    )
    parser.add_argument(
        "--output",
        "-o", 
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="qwen2.5:14b",
        help="Ollama model name (default: qwen2.5:14b)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        llm = Ollama(
            model=args.model,
            request_timeout=120.0,
            temperature=0.0
        )
        
        pipeline = QAExtractionPipeline(model=llm)
        curator = QACurationTool()
        
        # Load documents
        logger.info(f"Loading documents from {args.input}")
        documents = SimpleDirectoryReader(
            input_dir=args.input, 
            recursive=True
        ).load_data()
        
        # Generate QA pairs
        logger.info("Generating QA pairs...")
        qa_pairs = await pipeline.extract_qa_pairs(documents)
        
        # Create curation template
        curator.prepare_curation_template(
            qa_pairs=qa_pairs,
            output_file=args.output
        )
        
    except Exception as e:
        logger.error(f"Error in QA generation process: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
    # usage: python qa_generator.py -i ./qa_docs -o qa_pairs.json