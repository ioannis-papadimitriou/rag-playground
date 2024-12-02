from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import json
import asyncio
from datetime import datetime
import argparse
import logging
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama

@dataclass
class ExtractedSummarizationTask:
    context: str  # changed from text
    question: str
    answer: str   # changed from suggested_summary
    metadata: Dict[str, Any]
    confidence: float
    extraction_timestamp: str
    topic: str
    word_count: int
    key_points: List[str]

@dataclass
class SummarizationExtractionPipeline:
    def __init__(
        self,
        model: Ollama,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        overlap: int = 200
    ):
        self.model = model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.topic_sections = {}  # Store sections by topic
        
    async def _collect_topic_sections(self, documents: List[Document]) -> Dict[str, List[Dict]]:
        """Collect topics and their combinations across documents"""
        logging.info("Starting topic collection across all documents...")
        
        topic_sections = {}  # topic -> list of relevant sections
        
        # First pass: collect individual topics
        for doc_idx, doc in enumerate(documents, 1):
            logging.info(f"\nCollecting topics from document {doc_idx}/{len(documents)}")
            
            chunks = [doc.text[i:i+self.max_chunk_size] 
                    for i in range(0, len(doc.text), self.max_chunk_size - self.overlap)]
            
            for chunk_idx, chunk in enumerate(chunks, 1):
                sections = await self._identify_topic_sections(chunk)
                
                for section in sections:
                    topic = section['topic']
                    if topic not in topic_sections:
                        topic_sections[topic] = []
                    
                    section['doc_id'] = doc_idx
                    section['chunk_id'] = chunk_idx
                    section['doc_metadata'] = doc.metadata
                    
                    topic_sections[topic].append(section)
        
        # Second pass: generate topic combinations
        single_topics = list(topic_sections.keys())
        logging.info(f"\nFound {len(single_topics)} individual topics, generating combinations...")
        
        for i in range(len(single_topics)):
            for j in range(i + 1, len(single_topics)):
                topic1 = single_topics[i]
                topic2 = single_topics[j]
                
                # Create combined topic name
                combined_topic = f"{topic1} and {topic2}"
                logging.info(f"Processing combination: {combined_topic} - {(len(single_topics) - i) * (len(single_topics) - j) / 2} combinations left")
                
                # Find sections where both topics are discussed
                combined_sections = []
                
                # Look for sections that might discuss both topics
                all_sections = topic_sections[topic1] + topic_sections[topic2]
                for doc_id in set(s['doc_id'] for s in all_sections):
                    doc_sections = [s for s in all_sections if s['doc_id'] == doc_id]
                    
                    # Check if both topics appear in nearby sections
                    prompt = f"""Analyze these text sections and determine if they discuss the relationship 
                    or connection between {topic1} and {topic2}.

                    Text sections:
                    '''
                    {chr(10).join(s['text'] for s in doc_sections)}
                    '''

                    Return ONLY a JSON response:
                    {{
                        "discusses_combination": true/false,
                        "relevant_text": "extract of text discussing the combination",
                        "confidence": <float 0-1>
                    }}"""
                    
                    try:
                        response = self.model.complete(prompt)
                        result = json.loads(response.text.strip())
                        
                        if result.get('discusses_combination', False) and result.get('confidence', 0) > 0.7:
                            # Create a new combined section
                            combined_section = {
                                'topic': combined_topic,
                                'text': result['relevant_text'],
                                'doc_id': doc_id,
                                'chunk_id': doc_sections[0]['chunk_id'],
                                'doc_metadata': doc_sections[0]['doc_metadata'],
                                'confidence': result['confidence'],
                                'key_points': [],  # Will be populated in next step
                                'subtopics': [topic1, topic2]
                            }
                            
                            # Get key points specific to the combination
                            key_points_prompt = f"""Identify key points about how {topic1} and {topic2} 
                            are related or interact in this text:
                            
                            '''
                            {result['relevant_text']}
                            '''
                            
                            Return ONLY a JSON list:
                            {{
                                "key_points": [
                                    "point 1",
                                    "point 2",
                                    ...
                                ]
                            }}"""
                            
                            try:
                                key_points_response = self.model.complete(key_points_prompt)
                                key_points_result = json.loads(key_points_response.text.strip())
                                combined_section['key_points'] = key_points_result.get('key_points', [])
                            except Exception as e:
                                logging.warning(f"Error getting key points for combination: {e}")
                            
                            combined_sections.append(combined_section)
                            
                    except Exception as e:
                        logging.warning(f"Error processing combination {combined_topic}: {e}")
                        continue
                
                if combined_sections:
                    topic_sections[combined_topic] = combined_sections
                    logging.info(f"Added combined topic '{combined_topic}' with {len(combined_sections)} sections")
        
        return topic_sections

    async def _generate_aggregated_task(
        self,
        topic: str,
        sections: List[Dict]
    ) -> Optional[ExtractedSummarizationTask]:
        """Generate a task aligned with QA structure"""
        
        sections = sorted(sections, key=lambda x: x['confidence'], reverse=True)
        
        all_key_points = []
        all_subtopics = set()
        combined_text = ""
        source_locations = []
        
        for section in sections:
            all_key_points.extend(section['key_points'])
            all_subtopics.update(section.get('subtopics', []))
            combined_text += f"\nSection from document {section['doc_id']}:\n{section['text']}\n"
            source_locations.append({
                'doc_id': section['doc_id'],
                'chunk_id': section['chunk_id'],
                'text': section['text']
            })
        
        prompt = f"""You are a scientific content analyzer that MUST respond in valid JSON format.

    Generate a comprehensive QA task about this topic across all text sections.

    Topic: {topic}
    Subtopics: {list(all_subtopics)}
    Key Points: {list(set(all_key_points))}

    Text sections:
    '''
    {combined_text}
    '''

    RESPOND WITH EXACTLY THIS JSON STRUCTURE:
    {{
        "task": {{
            "context": "full context needed to understand the Q&A",
            "question": "clear question about the topic",
            "answer": "comprehensive answer from all sources",
            "confidence": <number between 0 and 1>,
            "qa_pairs": [
                {{
                    "question": "specific question about the topic",
                    "answer": "precise answer from the sources",
                    "context": "relevant context for this QA pair",
                    "confidence": <number between 0 and 1>
                }}
            ]
        }}
    }}"""

        try:
            response = self.model.complete(prompt)
            result = json.loads(response.text.strip())['task']
            
            qa_pairs = result.get('qa_pairs', [])
            logging.info(f"\nGenerated {len(qa_pairs)} QA pairs for topic '{topic}'")
            
            return ExtractedSummarizationTask(
                context=str(result['context']),
                question=str(result['question']),
                answer=str(result['answer']),
                topic=topic,
                key_points=list(set(all_key_points)),
                confidence=float(result['confidence']),
                word_count=len(combined_text.split()),
                metadata={
                    "extraction_method": "cross_document",
                    "subtopics": list(all_subtopics),
                    "source_locations": source_locations,
                    "qa_pairs": qa_pairs,
                    "document_coverage": len(set(s['doc_id'] for s in sections))
                },
                extraction_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Error generating aggregated task for topic {topic}: {e}")
            return None

    async def extract_summarization_tasks(
        self,
        documents: List[Document]
    ) -> List[ExtractedSummarizationTask]:
        """Extract summarization tasks with cross-document topic aggregation"""
        tasks: List[ExtractedSummarizationTask] = []
        
        try:
            # First pass: collect all topics and their sections
            topic_sections = await self._collect_topic_sections(documents)
            
            # Second pass: generate aggregated tasks for each topic
            for topic, sections in topic_sections.items():
                logging.info(f"\nGenerating aggregated task for topic: {topic}")
                logging.info(f"Found in {len(set(s['doc_id'] for s in sections))} documents")
                
                if len(sections) > 0:
                    task = await self._generate_aggregated_task(topic, sections)
                    if task:
                        tasks.append(task)
                        logging.info(
                            f"Successfully generated task with "
                            f"{len(task.metadata['qa_pairs'])} QA pairs"
                        )
            
            logging.info(f"\nExtraction complete: {len(tasks)} tasks generated")
            return tasks
            
        except Exception as e:
            logging.error(f"Error in extraction process: {e}")
            return []


    async def _identify_topic_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify diverse topic sections with robust JSON handling"""
        
        # Format the topic prompt properly using f-string
        topic_prompt = f"""You are a scientific text analyzer that MUST respond in valid JSON format.

    Your task is to identify main topics in this text. Focus on ALL scientific topics, not just ML.

    Text to analyze:
    '''
    {text[:3000]}
    '''

    RESPOND WITH EXACTLY THIS JSON STRUCTURE, nothing else before or after:
    {{
        "topics": [
            {{
                "topic": "Name of specific scientific topic",
                "subtopics": ["subtopic 1", "subtopic 2"]
            }}
        ]
    }}

    IMPORTANT:
    1. Your response must start with '{{' and end with '}}'
    2. Use double quotes for strings
    3. No comments or explanations outside the JSON
    4. No markdown formatting"""

        try:
            logging.info(f"Starting topic identification on text of length {len(text)} chars")
            topics_response = self.model.complete(topic_prompt)
            response_text = topics_response.text.strip()
            
            logging.info("Raw response received, cleaning...")
            logging.debug(f"Raw response: {response_text}")
            
            # Remove any markdown formatting
            if '```json' in response_text:
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Try to find the JSON object
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                response_text = json_match.group()
                logging.info("Successfully extracted JSON from response")
            else:
                logging.error("No JSON object found in response")
                logging.debug(f"Cleaned response: {response_text}")
                return []
                
            try:
                topics_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error: {e}")
                logging.debug(f"Failed to parse: {response_text}")
                return []

            main_topics = topics_data.get('topics', [])
            logging.info(f"Identified {len(main_topics)} main topics")
            
            sections = []
            for topic_info in main_topics:
                topic = topic_info['topic']
                logging.info(f"\nProcessing topic: {topic}")
                
                # Use f-string for section prompt
                section_prompt = f"""You are a scientific text analyzer. Find the exact section in this text that discusses: "{topic}"

    Text to analyze:
    '''
    {text}
    '''

    Return ONLY this JSON structure:
    {{
        "section": {{
            "word_start": 0,
            "word_end": 100,
            "has_substance": true,
            "key_points": [
                "point 1",
                "point 2"
            ],
            "confidence": 0.95,
            "relevant_text": "the exact text from the section"
        }}
    }}"""
                
                try:
                    section_response = self.model.complete(section_prompt)
                    section_text = section_response.text.strip()
                    
                    logging.info(f"Got response for topic {topic}, cleaning...")
                    if '```json' in section_text:
                        section_text = section_text.replace('```json', '').replace('```', '').strip()
                    
                    json_match = re.search(r'\{[\s\S]*\}', section_text)
                    if json_match:
                        section_text = json_match.group()
                        section_data = json.loads(section_text)['section']
                        
                        # Validate section data
                        words = text.split()
                        start_idx = max(0, min(int(section_data['word_start']), len(words)-1))
                        end_idx = max(start_idx+1, min(int(section_data['word_end']), len(words)))
                        
                        if end_idx - start_idx >= 50:  # Minimum 50 words
                            extracted_text = ' '.join(words[start_idx:end_idx])
                            
                            section = {
                                'topic': str(topic),
                                'word_start': start_idx,
                                'word_end': end_idx,
                                'text': extracted_text,
                                'has_substance': bool(section_data['has_substance']),
                                'key_points': [str(p) for p in section_data.get('key_points', [])],
                                'confidence': float(min(1.0, max(0.0, section_data['confidence']))),
                                'subtopics': [str(s) for s in topic_info.get('subtopics', [])]
                            }
                            sections.append(section)
                            logging.info(f"Successfully processed section of {end_idx - start_idx} words")
                        else:
                            logging.info(f"Section too short: {end_idx - start_idx} words")
                except Exception as e:
                    logging.error(f"Error processing section for topic {topic}: {str(e)}")
                    continue
                    
            return sections
            
        except Exception as e:
            logging.error(f"Error in topic identification: {str(e)}")
            logging.debug("Stack trace:", exc_info=True)
            return []

    # Helper function to test the topic identification
    async def test_topic_identification(self, sample_text: str):
        """Test topic identification with detailed logging"""
        logging.info("Starting topic identification test")
        logging.info(f"Input text length: {len(sample_text)} characters")
        
        sections = await self._identify_topic_sections(sample_text)
        
        if not sections:
            logging.warning("No sections identified")
            return
        
        print("\nIdentified Sections:")
        for i, section in enumerate(sections, 1):
            print(f"\nSection {i}:")
            print(f"Topic: {section['topic']}")
            print(f"Subtopics: {', '.join(section['subtopics'])}")
            print(f"Length: {section['word_end'] - section['word_start']} words")
            print(f"Confidence: {section['confidence']:.2f}")
            print("Key Points:")
            for point in section['key_points']:
                print(f"- {point}")
            print("\nExtracted Text Preview:")
            print(section['text'][:200] + "...")

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()

    def _find_best_text_match(
        self,
        words: List[str],
        target_text: str,
        start_idx: int,
        end_idx: int,
        window_size: int = 100
    ) -> tuple[int, int]:
        """Find the best matching text boundaries within a window"""
        best_start = start_idx
        best_end = end_idx
        best_similarity = 0
        
        # Try different windows around the initial indices
        for i in range(max(0, start_idx - window_size), min(len(words), start_idx + window_size)):
            for j in range(i + 50, min(len(words), i + window_size)):  # Minimum 50 words
                current_text = ' '.join(words[i:j])
                similarity = self._text_similarity(current_text, target_text)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_start = i
                    best_end = j
        
        return best_start, best_end

    async def _generate_summarization_task(
        self,
        section: Dict[str, Any]
    ) -> Optional[ExtractedSummarizationTask]:
        """Generate a focused summarization task with robust JSON handling"""
        
        prompt = f"""You are a scientific content analyzer that MUST respond in valid JSON format.

    Generate a focused summarization task and QA pairs for this scientific text section.

    Topic: {section['topic']}
    Subtopics: {', '.join(section['subtopics'])}
    Key Points: {', '.join(section['key_points'])}

    Text to analyze:
    '''
    {section['text']}
    '''

    RESPOND WITH EXACTLY THIS JSON STRUCTURE, nothing else:
    {{
        "task": {{
            "prompt": "Clear task description for summarization",
            "suggested_summary": "Detailed technical summary",
            "confidence": <number between 0 and 1>,
            "qa_pairs": [
                {{
                    "question": "Specific technical question",
                    "answer": "Precise answer from the text",
                    "confidence": <number between 0 and 1>
                }}
            ]
        }}
    }}

    Requirements:
    1. Start response with '{{' and end with '}}'
    2. Use double quotes for strings
    3. Generate 2-3 technical QA pairs
    4. Focus on specific technical details
    5. No markdown or extra text"""
        
        try:
            logging.info(f"\nGenerating task for topic: {section['topic']}")
            response = self.model.complete(prompt)
            response_text = response.text.strip()
            
            # Clean response
            logging.debug(f"Raw response: {response_text}")
            if '```json' in response_text:
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Try to find JSON object
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logging.error("No JSON object found in response")
                logging.debug(f"Cleaned response: {response_text}")
                return None
                
            response_text = json_match.group()
            
            try:
                result = json.loads(response_text)['task']
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"JSON parse error: {e}")
                logging.debug(f"Failed to parse: {response_text}")
                return None
            
            # Log the generated QA pairs
            qa_pairs = result.get('qa_pairs', [])
            logging.info(f"Generated {len(qa_pairs)} QA pairs:")
            for i, qa in enumerate(qa_pairs, 1):
                logging.info(f"\nQA Pair {i}:")
                logging.info(f"Q: {qa['question']}")
                logging.info(f"A: {qa['answer']}")
                logging.info(f"Confidence: {qa['confidence']:.2f}")
            
            # Create the summarization task
            task = ExtractedSummarizationTask(
                text=section['context'],
                prompt=str(result['question']),
                suggested_summary=str(result['answer']),
                topic=section['topic'],
                key_points=section['key_points'],
                confidence=float(min(
                    section['confidence'],
                    result['confidence'],
                    min(qa['confidence'] for qa in qa_pairs) if qa_pairs else 1.0
                )),
                word_count=len(section['text'].split()),
                metadata={
                    "extraction_method": "topic_based",
                    "subtopics": section['subtopics'],
                    "key_points_count": len(section['key_points']),
                    "qa_pairs": qa_pairs
                },
                extraction_timestamp=datetime.now().isoformat()
            )
            
            logging.info("\nGenerated Summarization Task:")
            logging.info(f"Prompt: {task.question}")
            logging.info(f"Summary length: {len(task.answer.split())} words")
            logging.info(f"Overall confidence: {task.confidence:.2f}")
            
            return task
            
        except Exception as e:
            logging.error(f"Error generating task: {e}")
            logging.debug("Stack trace:", exc_info=True)
            return None

class SummarizationCurationTool:
    """Tool for human curation of extracted summarization tasks"""
    
    def __init__(self, output_dir: str = "curated_summaries"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_curation_template(
        self,
        tasks: List[ExtractedSummarizationTask],
        output_file: str
    ):
        """Create template for human curation"""
        output_data = {
            "instructions": """
            Curation Instructions:
            1. Review each QA pair:
               - Is the context complete and self-contained?
               - Is the question clear and focused?
               - Does the answer accurately reflect the context?
            2. For each QA pair, update the curation_status:
               - Set is_approved to true/false
               - Add your name as curator
               - Add any notes about modifications
               - If modifying, update any of the fields
            3. Save the file with _curated suffix
            """,
            "tasks": [
                {
                    "id": idx,
                    "context": task.context,
                    "question": task.question,
                    "answer": task.answer,
                    "topic": task.topic,
                    "key_points": task.key_points,
                    "metadata": {
                        **task.metadata,
                        "confidence": task.confidence,
                        "extraction_timestamp": task.extraction_timestamp,
                        "word_count": task.word_count
                    },
                    "curation_status": {
                        "is_approved": False,
                        "curator": "",
                        "notes": "",
                        "modified_question": "",
                        "modified_answer": "",
                        "modified_context": ""
                    }
                }
                for idx, task in enumerate(tasks)
            ]
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Curation template saved to {output_path}")

class CheckpointHandler:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, tasks: List[ExtractedSummarizationTask], filename: str) -> None:
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "tasks": [
                {
                    "context": task.context,
                    "question": task.question,
                    "answer": task.answer,
                    "topic": task.topic,
                    "key_points": task.key_points,
                    "confidence": task.confidence,
                    "word_count": task.word_count,
                    "metadata": task.metadata,
                    "extraction_timestamp": task.extraction_timestamp
                }
                for task in tasks
            ]
        }
        
        checkpoint_path = self.checkpoint_dir / f"{filename}.checkpoint"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
            
    def load_checkpoint(self, filename: str) -> List[ExtractedSummarizationTask]:
        checkpoint_path = self.checkpoint_dir / f"{filename}.checkpoint"
        
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                return [ExtractedSummarizationTask(**task_data) for task_data in checkpoint["tasks"]]
        return []


async def main():
    parser = argparse.ArgumentParser(description="Extract QA pairs from documents")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing documents")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--model", "-m", default="qwen2.5:14b", help="Ollama model name")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/summary_qa_extraction.log')
        ]
    )
    
    try:
        logging.info("Initializing components...")
        llm = Ollama(model=args.model, request_timeout=120.0, temperature=0.0)
        pipeline = SummarizationExtractionPipeline(model=llm)
        curator = SummarizationCurationTool()
        checkpoint_handler = CheckpointHandler()
        
        # Load documents
        logging.info(f"Loading documents from {args.input}")
        documents = SimpleDirectoryReader(input_dir=args.input, recursive=True).load_data(show_progress=True)
        
        if not documents:
            logging.error("No documents loaded")
            return
            
        logging.info(f"Loaded {len(documents)} documents")
        
        # Try to load existing checkpoint
        tasks = checkpoint_handler.load_checkpoint(Path(args.output).stem)
        if tasks:
            logging.info(f"Resuming from checkpoint with {len(tasks)} existing tasks")
            
        # Process documents
        new_tasks = await pipeline.extract_summarization_tasks(documents)
        if new_tasks:
            tasks.extend(new_tasks)
            
            # Save checkpoint
            checkpoint_handler.save_checkpoint(tasks=tasks, filename=Path(args.output).stem)
            logging.info(f"Checkpoint saved with {len(tasks)} total tasks")
        
        # Create final output
        if tasks:
            logging.info(f"Creating curation template with {len(tasks)} QA pairs...")
            curator.prepare_curation_template(tasks=tasks, output_file=args.output)
            
            # Clean up checkpoint after successful completion
            checkpoint_path = checkpoint_handler.checkpoint_dir / f"{Path(args.output).stem}.checkpoint"
            checkpoint_path.unlink(missing_ok=True)
            
            logging.info("Process complete!")
        else:
            logging.warning("No QA pairs were generated!")
            
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        logging.debug("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
    # usage: python summarization_qa_generator.py -i ./qa_docs -o qa_pairs.json