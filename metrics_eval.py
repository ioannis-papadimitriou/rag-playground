from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json
from pathlib import Path
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import spacy
from llama_index.llms.ollama import Ollama
import argparse
import logging
import asyncio
import matplotlib.pyplot as plt
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OVERALL_PASS_THRESHOLD = 6

class MetricType(Enum):
    PROGRAMMATIC = "programmatic"
    LLM = "llm"
    HYBRID = "hybrid"

@dataclass
class EvaluationMetric:
    name: str
    type: MetricType
    weight: float
    threshold: float
    
@dataclass
class MetricResult:
    metric: EvaluationMetric
    score: float
    details: Dict[str, Any]
    passed: bool

@dataclass
class SourceChunk:
    text: str
    document: str
    page: Optional[int]
    chunk_id: str

@dataclass
class AgentResponse:
    answer: str
    source_chunks: List[SourceChunk]
    confidence_score: float
    reasoning_path: List[str]

class MetricsEngine:
    """Hybrid metrics system combining programmatic and LLM-based evaluation"""
    
    def __init__(self, model_name: str = "qwen2.5:14b"):
        self.llm = Ollama(model=model_name)
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')
        
        # Define standard metrics
        self.metrics = [
            # Programmatic metrics
            EvaluationMetric(
                name="key_terms_precision",
                type=MetricType.PROGRAMMATIC,
                weight=0.15,
                threshold=0.7
            ),
            EvaluationMetric(
                name="token_recall",
                type=MetricType.PROGRAMMATIC,
                weight=0.15,
                threshold=0.7
            ),
            
            # LLM-based metrics
            EvaluationMetric(
                name="truthfulness",
                type=MetricType.LLM,
                weight=0.2,
                threshold=0.7
            ),
            EvaluationMetric(
                name="completeness",
                type=MetricType.LLM,
                weight=0.1,
                threshold=0.7
            ),
            EvaluationMetric(
                name="source_relevance",
                type=MetricType.LLM,
                weight=0.05,
                threshold=0.7
            ),
            EvaluationMetric(
                name="context_faithfulness",
                type=MetricType.LLM,
                weight=0.1,
                threshold=0.7
            ),
            EvaluationMetric(
                name="semantic_f1",
                type=MetricType.HYBRID,
                weight=0.1,
                threshold=0.6
            ),
            EvaluationMetric(
                name="completeness_gain",
                type=MetricType.HYBRID,
                weight=0.05,
                threshold=0.51 # slightly more than equal coverage to ground truth, we want to measure the passes
            ),
            EvaluationMetric(
                name="answer_relevance",
                type=MetricType.HYBRID,
                weight=0.1,
                threshold=0.7
            ),
            EvaluationMetric(
                name="numerical_accuracy",
                type=MetricType.HYBRID,
                weight=0.0,  
                threshold=0.7
            ),
        ]
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers and convert them to standardized float values."""
        # Special handling for temperatures in Kelvin
        kelvin_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?\s*K(?:elvin)?'
        kelvin_matches = re.finditer(kelvin_pattern, text)
        
        standardized = []
        
        # First handle Kelvin temperatures
        for match in kelvin_matches:
            num = match.group().lower().replace('kelvin', '').replace('k', '').strip()
            try:
                standardized.append(float(num.replace(',', '')))
            except ValueError:
                continue

        # If no Kelvin temperatures found, try general number pattern
        if not standardized:
            pattern = r'(?<![a-zA-Z])(?<![a-zA-Z-])(?<!\[)(?<!\()\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[KkMBmb](?:elvin)?|\s*(?:thousand|million|billion)|%)?(?![a-zA-Z](?!elvin|illion))(?!\])(?!\))'
            
            matches = [match.group().strip().lower() for match in re.finditer(pattern, text.lower())]
            
            for num in matches:
                # Remove commas
                num = num.replace(',', '')
                
                # Convert to float with multiplier
                try:
                    if any(unit in num for unit in ['k', 'thousand']):
                        base = float(num.replace('k', '').replace('thousand', '').strip())
                        standardized.append(base * 1000)
                    elif any(unit in num for unit in ['m', 'million']):
                        base = float(num.replace('m', '').replace('million', '').strip())
                        standardized.append(base * 1000000)
                    elif any(unit in num for unit in ['b', 'billion']):
                        base = float(num.replace('b', '').replace('billion', '').strip())
                        standardized.append(base * 1000000000)
                    else:
                        standardized.append(float(num))
                except ValueError:
                    continue
                    
        logger.debug(f"Text: {text}")
        logger.debug(f"Standardized numbers: {standardized}")
        
        return standardized

    def _calculate_key_terms_match(self, response: str, ground_truth: str, context: str) -> float:
        """Calculate key terms match using important terms from context"""
        context_doc = self.nlp(context)
        key_terms = set(
            token.text.lower() for token in context_doc
            if not token.is_stop and not token.is_punct and token.has_vector
        )
        
        response_terms = set(
            token.text.lower() for token in self.nlp(response)
            if token.text.lower() in key_terms
        )
        truth_terms = set(
            token.text.lower() for token in self.nlp(ground_truth)
            if token.text.lower() in key_terms
        )
        
        if not truth_terms:
            return 1.0 if not response_terms else 0.0
            
        overlap = len(response_terms.intersection(truth_terms))
        return overlap / len(truth_terms)

    def _calculate_token_precision_recall(self, response: str, ground_truth: str) -> tuple[float, float]:
        """Calculate token-level precision and recall"""
        response_doc = self.nlp(response)
        truth_doc = self.nlp(ground_truth)
        
        response_tokens = set(token.text.lower() for token in response_doc 
                            if not token.is_stop and not token.is_punct)
        truth_tokens = set(token.text.lower() for token in truth_doc
                         if not token.is_stop and not token.is_punct)
        
        if not response_tokens:
            return 0.0, 0.0
        if not truth_tokens:
            return 0.0, 1.0
            
        common_tokens = response_tokens.intersection(truth_tokens)
        
        precision = len(common_tokens) / len(response_tokens)
        recall = len(common_tokens) / len(truth_tokens)
        # Adjust precision calculation to be more lenient with verbose but correct answers
        # If recall is high (meaning the important information is there), we apply a more generous precision calculation
        if recall >= 0.8:  # High recall threshold
            # Scale precision penalty based on recall
            precision_scale = 0.7 + (0.3 * recall)  # Ranges from 0.7 to 1.0 based on recall
            raw_precision = len(common_tokens) / len(response_tokens)
            precision = raw_precision * precision_scale
        else:
            # Use standard precision calculation for low recall cases
            precision = len(common_tokens) / len(response_tokens)
        
        return precision, recall

    def _calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between text embeddings"""
        emb1 = self.embed_model.encode([text1])[0]
        emb2 = self.embed_model.encode([text2])[0]
        return 1 - cosine(emb1, emb2)

    async def _evaluate_programmatic(
        self,
        metric: EvaluationMetric,
        response: AgentResponse,
        ground_truth: Dict[str, Any]
    ) -> MetricResult:
        """Evaluate using programmatic metrics"""
        score = 0.0

        if metric.name == "key_terms_precision":
            score = self._calculate_key_terms_match(
                response.answer,
                ground_truth['answer'],
                ground_truth['context']
            )
        # elif metric.name == "token_precision":
        #     precision, _ = self._calculate_token_precision_recall(
        #         response.answer,
        #         ground_truth['answer']
        #     )
        #     score = precision
        elif metric.name == "token_recall":
            _, recall = self._calculate_token_precision_recall(
                response.answer,
                ground_truth['answer']
            )
            score = recall
        else:
            raise ValueError(f"Unknown programmatic metric: {metric.name}")
            
        return MetricResult(
            metric=metric,
            score=score,
            details={
                "method": "programmatic",
                "calculation": f"{metric.name}_calculation"
            },
            passed=score >= metric.threshold
        )

    async def _evaluate_llm(
        self,
        metric: EvaluationMetric,
        response: AgentResponse,
        ground_truth: Dict[str, Any]
    ) -> MetricResult:
        """Evaluate using LLM-based metrics"""
        if metric.name == "source_relevance":
            if not response.source_chunks:
                return MetricResult(
                    metric=metric,
                    score=0.0,
                    details={
                        "method": "llm",
                        "reasoning": "No source chunks provided"
                    },
                    passed=False
                )
            
            # Format sources with explicit numbering and clean formatting
            sources_text = ""
            for i, chunk in enumerate(response.source_chunks):
                # Clean the chunk text to avoid JSON formatting issues
                clean_text = chunk.text.replace('"', "'").replace('\n', ' ').strip()
                sources_text += f"\nSource {i+1}:\n{clean_text}\n"

            prompt = f"""Evaluate the relevance of each source to answering this question.

    Question: {ground_truth['question']}

    Sources:
    {sources_text}

    Evaluate each source's relevance on a scale of 0 to 1:
    - 0.0: Completely irrelevant
    - 1.0: Highly relevant with key information

    Remember: A single highly relevant source containing crucial information is considered successful, even if other sources are less relevant.

    Respond with structured data in this exact format:
    {{
        "overall_score": <number between 0 and 1>,
        "reasoning": "<brief explanation>",
        "source_analysis": [
            {{
                "source_num": <number>,
                "relevance_score": <number between 0 and 1>,
                "contains_key_info": true/false,
                "key_info": "<brief description if any>"
            }}
        ],
        "best_source_score": <number between 0 and 1>,
        "has_crucial_info": true/false
    }}

    Ensure all text is properly escaped and all values are properly formatted."""

            try:
                llm_response = self.llm.complete(prompt)
                
                # First try to parse the response directly
                try:
                    result = json.loads(llm_response.text)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON using regex
                    import re
                    json_pattern = r'\{[\s\S]*\}'
                    match = re.search(json_pattern, llm_response.text)
                    if match:
                        result = json.loads(match.group())
                    else:
                        raise ValueError("Could not extract valid JSON from response")

                # Validate and clean the parsed results
                best_source_score = float(result.get('best_source_score', 0.0))
                overall_score = float(result.get('overall_score', 0.0))
                has_crucial_info = bool(result.get('has_crucial_info', False))
                
                # Ensure scores are within valid range
                best_source_score = max(0.0, min(1.0, best_source_score))
                overall_score = max(0.0, min(1.0, overall_score))

                # Calculate final score with weighted approach
                if best_source_score > 0.8 or has_crucial_info:
                    final_score = 0.8 * best_source_score + 0.2 * overall_score
                else:
                    final_score = 0.5 * best_source_score + 0.5 * overall_score

                # Clean and validate source analysis
                source_analysis = []
                for source in result.get('source_analysis', []):
                    source_analysis.append({
                        "source_num": int(source.get('source_num', 0)),
                        "relevance_score": float(source.get('relevance_score', 0.0)),
                        "contains_key_info": bool(source.get('contains_key_info', False)),
                        "key_info": str(source.get('key_info', ''))
                    })
                
                return MetricResult(
                    metric=metric,
                    score=final_score,
                    details={
                        "method": "llm",
                        "reasoning": str(result.get('reasoning', '')),
                        "source_analysis": source_analysis,
                        "best_source_score": best_source_score,
                        "has_crucial_info": has_crucial_info,
                        "overall_score": overall_score
                    },
                    passed=final_score >= metric.threshold
                )
                
            except Exception as e:
                logger.warning(f"Error in source relevance evaluation: {str(e)}")
                # Fallback scoring based on simple text similarity
                try:
                    # Calculate average embedding similarity as fallback
                    similarities = []
                    question_embedding = self.embed_model.encode([ground_truth['question']])[0]
                    
                    for chunk in response.source_chunks:
                        chunk_embedding = self.embed_model.encode([chunk.text])[0]
                        similarity = 1 - cosine(question_embedding, chunk_embedding)
                        similarities.append(similarity)
                    
                    if similarities:
                        best_similarity = max(similarities)
                        avg_similarity = sum(similarities) / len(similarities)
                        fallback_score = 0.8 * best_similarity + 0.2 * avg_similarity
                    else:
                        fallback_score = 0.0
                    
                    return MetricResult(
                        metric=metric,
                        score=fallback_score,
                        details={
                            "method": "llm_failed_fallback",
                            "error": str(e),
                            "raw_response": llm_response.text if 'llm_response' in locals() else None,
                            "fallback_method": "embedding_similarity",
                            "best_similarity": best_similarity if 'best_similarity' in locals() else None,
                            "avg_similarity": avg_similarity if 'avg_similarity' in locals() else None
                        },
                        passed=fallback_score >= metric.threshold
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback scoring also failed: {str(fallback_error)}")
                    return MetricResult(
                        metric=metric,
                        score=0.0,
                        details={
                            "method": "all_methods_failed",
                            "error": str(e),
                            "fallback_error": str(fallback_error),
                            "raw_response": llm_response.text if 'llm_response' in locals() else None
                        },
                        passed=False
                    )


        elif metric.name == "truthfulness":
            prompt = f"""Evaluate the truthfulness of the following response compared to the ground truth.
            Context: {ground_truth['context']}
            Ground Truth: {ground_truth['answer']}
            Response: {response.answer}
            
            Score the truthfulness from 0 to 1, where:
            0 = completely false/contradictory
            1 = completely truthful and accurate
            
            Provide your score and reasoning in JSON format:
            {{
                "score": <float>,
                "reasoning": "<explanation>"
            }}
            """
            
        elif metric.name == "completeness":
            prompt = f"""Evaluate the completeness of the following response compared to the ground truth.
            Context: {ground_truth['context']}
            Ground Truth: {ground_truth['answer']}
            Response: {response.answer}
            
            Score the completeness from 0 to 1, where:
            0 = missing critical information
            1 = covers all important points
            
            Provide your score and reasoning in JSON format:
            {{
                "score": <float>,
                "reasoning": "<explanation>"
            }}
            """

        elif metric.name == "context_faithfulness":
            prompt = f"""Evaluate if the following response is factually consistent with the provided context.
            Consider only factual consistency, not completeness or relevance.
            
            Context: {ground_truth['context']}
            Response: {response.answer}
            
            Rate the factual consistency from 0 to 1 where:
            0 = Contains statements contradicting or unsupported by the context
            1 = All statements are supported by the context
            
            Provide your score and reasoning in JSON format:
            {{
                "score": <float>,
                "reasoning": "<explanation>",
                "contradictions": [<list of any contradicting statements>]
            }}
            """
            
            llm_response = self.llm.complete(prompt)
            try:
                result = json.loads(llm_response.text)
                return MetricResult(
                    metric=metric,
                    score=float(result['score']),
                    details={
                        "method": "llm",
                        "reasoning": result.get('reasoning', ''),
                        "contradictions": result.get('contradictions', [])
                    },
                    passed=float(result['score']) >= metric.threshold
                )
            except Exception as e:
                return MetricResult(
                    metric=metric,
                    score=0.0,
                    details={
                        "method": "llm",
                        "error": str(e),
                        "raw_response": llm_response.text
                    },
                    passed=False
                )
    
        else:
            raise ValueError(f"Unknown LLM metric: {metric.name}")
            
        response = self.llm.complete(prompt)
        try:
            result = json.loads(response.text)
            score = float(result['score'])
            
            return MetricResult(
                metric=metric,
                score=score,
                details={
                    "method": "llm",
                    "reasoning": result.get('reasoning', ''),
                    "source_analysis": result.get('source_analysis', [])
                },
                passed=score >= metric.threshold
            )
        except (json.JSONDecodeError, KeyError) as e:
            return MetricResult(
                metric=metric,
                score=0.0,
                details={
                    "method": "llm",
                    "error": str(e),
                    "raw_response": response.text
                },
                passed=False
            )

    async def _evaluate_hybrid(
        self,
        metric: EvaluationMetric,
        response: AgentResponse,
        ground_truth: Dict[str, Any]
    ) -> MetricResult:
        """Evaluate using hybrid metrics combining programmatic and LLM approaches"""

        if metric.name == "numerical_accuracy":
            # First try programmatic extraction
            truth_numbers = self._extract_numbers(ground_truth['answer'])
            
            if not truth_numbers:  # No numbers in ground truth
                return MetricResult(
                    metric=metric,
                    score=None,  # Not applicable
                    details={
                        "method": "hybrid",
                        "found_numbers": False,
                        "truth_numbers": [],
                    },
                    passed=None  # Not applicable
                )
            
            response_numbers = self._extract_numbers(response.answer)
            
            # If numbers found, verify with LLM
            prompt = f"""Compare the numerical values in these two statements and determine if they are equivalent:

    Ground Truth: {ground_truth['answer']}
    Response: {response.answer}

    Consider:
    - Different representations of the same number (e.g., "10K" = "10,000")
    - Units and their abbreviations
    - Approximate values using words like "about", "around", "approximately"

    Return your analysis in JSON format:
    {{
        "numbers_match": true/false,
        "explanation": "brief explanation of comparison",
        "equivalent_pairs": [
            {{
                "truth_value": "number as written in ground truth",
                "response_value": "number as written in response",
                "are_equivalent": true/false
            }}
        ]
    }}"""

            try:
                llm_response = self.llm.complete(prompt)
                result = json.loads(llm_response.text)
                
                # Use LLM's overall judgment for scoring
                score = 1.0 if result.get('numbers_match', False) else 0.0
                
                return MetricResult(
                    metric=metric,
                    score=score,
                    details={
                        "method": "hybrid",
                        "programmatic": {
                            "truth_numbers": truth_numbers,
                            "response_numbers": response_numbers,
                        },
                        "llm_analysis": result
                    },
                    passed=score >= metric.threshold
                )
                
            except Exception as e:
                # Fallback to pure programmatic if LLM fails
                matches = sum(1 for truth_num in truth_numbers 
                            if any(abs(truth_num - resp_num) < 0.0001 
                                for resp_num in response_numbers))
                score = matches / len(truth_numbers)
                
                return MetricResult(
                    metric=metric,
                    score=score,
                    details={
                        "method": "hybrid_fallback",
                        "error": str(e),
                        "fallback_matches": matches,
                        "total": len(truth_numbers)
                    },
                    passed=score >= metric.threshold
                )


        elif metric.name == "answer_relevance":
            # Calculate embedding similarity
            embedding_score = self._calculate_embedding_similarity(
                response.answer,
                ground_truth['answer']
            )
            
            # Get LLM judgment
            prompt = f"""Evaluate the semantic relevance of the following response to the ground truth.
            Ground Truth: {ground_truth['answer']}
            Response: {response.answer}
            Embedding Similarity Score: {embedding_score}
            
            Consider both the embedding similarity and semantic meaning.
            Score the overall relevance from 0 to 1, where:
            0 = completely irrelevant
            1 = highly relevant
            
            Provide your score and reasoning in JSON format:
            {{
                "score": <float>,
                "reasoning": "<explanation>"
            }}
            """
            
            llm_response = self.llm.complete(prompt)
            try:
                result = json.loads(llm_response.text)
                llm_score = float(result['score'])
                
                # Combine scores with weights
                final_score = 0.5 * embedding_score + 0.5 * llm_score
                
                return MetricResult(
                    metric=metric,
                    score=final_score,
                    details={
                        "method": "hybrid",
                        "embedding_score": embedding_score,
                        "llm_score": llm_score,
                        "reasoning": result['reasoning']
                    },
                    passed=final_score >= metric.threshold
                )
            except (json.JSONDecodeError, KeyError) as e:
                return MetricResult(
                    metric=metric,
                    score=embedding_score,  # Fallback to embedding score
                    details={
                        "method": "hybrid_fallback",
                        "embedding_score": embedding_score,
                        "error": str(e),
                        "raw_response": llm_response.text
                    },
                    passed=embedding_score >= metric.threshold
                )
        
        elif metric.name == "completeness_gain":
            try:
                # First, extract key points from context to establish ground truth
                prompt = f"""Given this question and context, list ALL relevant key points that would make a complete answer.
                Then identify which of these points are covered in the ground truth and in the response.

                Question: {ground_truth['question']}
                Context: {ground_truth['context']}
                Ground Truth: {ground_truth['answer']}
                Response: {response.answer}

                Return ONLY a JSON object in this format:
                {{
                    "context_key_points": [
                        "point 1",
                        "point 2"
                    ],
                    "ground_truth_coverage": {{
                        "point 1": true,
                        "point 2": false
                    }},
                    "response_coverage": {{
                        "point 1": true,
                        "point 2": true
                    }}
                }}"""

                llm_response = self.llm.complete(prompt)
                response_text = llm_response.text.strip()
                
                # Clean and parse response
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                result = json.loads(response_text.strip())
                
                # Get all points and coverage
                all_points = result['context_key_points']
                truth_coverage = result['ground_truth_coverage']
                response_coverage = result['response_coverage']
                
                if not all_points:
                    raise ValueError("No key points identified from context")

                # Calculate coverage statistics
                total_points = len(all_points)
                truth_covered = sum(1 for covered in truth_coverage.values() if covered)
                response_covered = sum(1 for covered in response_coverage.values() if covered)

                # Calculate relative gain in coverage
                truth_coverage_ratio = truth_covered / total_points if total_points > 0 else 0
                response_coverage_ratio = response_covered / total_points if total_points > 0 else 0
                
                # Calculate relative gain
                # Will be positive if response covers more points than ground truth
                # Will be negative if response covers fewer points than ground truth
                # Will be 0 if coverage is equal
                relative_gain = response_coverage_ratio - truth_coverage_ratio
                
                # Normalize gain to [0,1] scale where:
                # 0.5 means equal coverage
                # >0.5 means response has better coverage
                # <0.5 means ground truth has better coverage
                normalized_gain = (relative_gain + 1) / 2

                # Verify gain with semantic similarity for points claimed as covered
                verification_score = 1.0  # Start with perfect score
                if response_covered > truth_covered:
                    # Get embeddings for verification
                    extra_points = [
                        point for point in all_points 
                        if response_coverage.get(point) and not truth_coverage.get(point)
                    ]
                    if extra_points:
                        response_emb = self.embed_model.encode([response.answer])[0]
                        point_embs = self.embed_model.encode(extra_points)
                        
                        # Verify each extra point has strong similarity with response
                        for point_emb in point_embs:
                            similarity = 1 - cosine(response_emb, point_emb)
                            if similarity < 0.6:  # Threshold for genuine coverage
                                verification_score *= 0.8  # Penalize questionable coverage

                # Adjust normalized gain based on verification
                final_score = normalized_gain * verification_score
                
                return MetricResult(
                    metric=metric,
                    score=final_score,
                    details={
                        "method": "hybrid",
                        "raw_relative_gain": relative_gain,
                        "normalized_gain": normalized_gain,
                        "verification_score": verification_score,
                        "coverage_stats": {
                            "total_relevant_points": total_points,
                            "ground_truth_covered": truth_covered,
                            "response_covered": response_covered,
                            "ground_truth_ratio": truth_coverage_ratio,
                            "response_ratio": response_coverage_ratio
                        },
                        "point_details": {
                            "all_points": all_points,
                            "ground_truth_coverage": truth_coverage,
                            "response_coverage": response_coverage,
                            "extra_points_in_response": [
                                point for point in all_points 
                                if response_coverage.get(point) and not truth_coverage.get(point)
                            ],
                            "missing_points_in_response": [
                                point for point in all_points 
                                if truth_coverage.get(point) and not response_coverage.get(point)
                            ]
                        }
                    },
                    passed=final_score >= metric.threshold
                )
                
            except Exception as e:
                logger.error(f"Main evaluation failed: {str(e)}")
                # Fallback to basic embedding similarity
                try:
                    truth_emb = self.embed_model.encode([ground_truth['answer']])[0]
                    response_emb = self.embed_model.encode([response.answer])[0]
                    fallback_score = 1 - cosine(truth_emb, response_emb)
                    
                    return MetricResult(
                        metric=metric,
                        score=fallback_score,
                        details={
                            "method": "hybrid_fallback",
                            "error": str(e),
                            "raw_response": llm_response.text if 'llm_response' in locals() else None
                        },
                        passed=fallback_score >= metric.threshold
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback evaluation also failed: {str(fallback_error)}")
                    return MetricResult(
                        metric=metric,
                        score=0.0,
                        details={
                            "method": "all_methods_failed",
                            "error": str(e),
                            "fallback_error": str(fallback_error)
                        },
                        passed=False
                    )

        elif metric.name == "semantic_f1":
            # First get key points from ground truth and response
            prompt = f"""Identify key factual points from both the ground truth and the given answer.
            
            Ground Truth: {ground_truth['answer']}
            Response: {response.answer}
            
            List all key factual points in JSON format:
            {{
                "ground_truth_points": [
                    "<point 1>",
                    "<point 2>",
                    ...
                ],
                "response_points": [
                    "<point 1>",
                    "<point 2>",
                    ...
                ]
            }}
            Important: Ensure the output is valid JSON. Use double quotes for strings. No trailing commas."""
            
            try:
                llm_response = self.llm.complete(prompt)
                result = json.loads(llm_response.text)
                truth_points = result.get('ground_truth_points', [])
                response_points = result.get('response_points', [])
                
                if not truth_points or not response_points:
                    return MetricResult(
                        metric=metric,
                        score=0.0,
                        details={
                            "method": "hybrid",
                            "error": "No key points identified",
                            "raw_response": llm_response.text
                        },
                        passed=False
                    )
                
                # Get embeddings for all points
                truth_embeddings = self.embed_model.encode(truth_points)
                response_embeddings = self.embed_model.encode(response_points)
                
                # Create similarity matrix
                similarity_matrix = np.zeros((len(truth_points), len(response_points)))
                for i, t_emb in enumerate(truth_embeddings):
                    for j, r_emb in enumerate(response_embeddings):
                        similarity_matrix[i][j] = 1 - cosine(t_emb, r_emb)
                
                # Fixed threshold for considering points as matching
                SIMILARITY_THRESHOLD = 0.6
                
                # Track matches using sets to prevent double counting
                matched_truth = set()
                matched_response = set()
                matches = []
                
                # Find best matches above threshold
                # Sort all possible matches by similarity score
                all_matches = []
                for i in range(len(truth_points)):
                    for j in range(len(response_points)):
                        if similarity_matrix[i][j] > SIMILARITY_THRESHOLD:
                            all_matches.append((i, j, similarity_matrix[i][j]))
                
                # Sort matches by similarity score in descending order
                all_matches.sort(key=lambda x: x[2], reverse=True)
                
                # Assign matches greedily, ensuring no point is matched twice
                for truth_idx, response_idx, similarity in all_matches:
                    if truth_idx not in matched_truth and response_idx not in matched_response:
                        matched_truth.add(truth_idx)
                        matched_response.add(response_idx)
                        matches.append({
                            "truth_point": truth_points[truth_idx],
                            "response_point": response_points[response_idx],
                            "similarity": similarity
                        })
                
                # Calculate precision and recall
                precision = len(matched_response) / len(response_points) if response_points else 0
                recall = len(matched_truth) / len(truth_points) if truth_points else 0
                
                # Calculate F1 score
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0.0
                
                # Ensure f1_score is in [0, 1]
                f1_score = max(0.0, min(1.0, f1_score))
                
                return MetricResult(
                    metric=metric,
                    score=f1_score,
                    details={
                        "method": "hybrid",
                        "precision": precision,
                        "recall": recall,
                        "matches": matches,
                        "truth_points": truth_points,
                        "response_points": response_points,
                        "similarity_threshold": SIMILARITY_THRESHOLD,
                        "matched_count": len(matches),
                        "total_truth_points": len(truth_points),
                        "total_response_points": len(response_points)
                    },
                    passed=f1_score >= metric.threshold
                )
                        
            except Exception as e:
                logger.error(f"Error in semantic_f1: {str(e)}")
                # Fallback to basic embedding similarity
                try:
                    truth_emb = self.embed_model.encode([ground_truth['answer']])[0]
                    response_emb = self.embed_model.encode([response.answer])[0]
                    fallback_score = 1 - cosine(truth_emb, response_emb)
                    
                    return MetricResult(
                        metric=metric,
                        score=fallback_score,
                        details={
                            "method": "hybrid_fallback",
                            "error": str(e),
                            "raw_response": llm_response.text if 'llm_response' in locals() else None
                        },
                        passed=fallback_score >= metric.threshold
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {str(fallback_error)}")
                    return MetricResult(
                        metric=metric,
                        score=0.0,
                        details={
                            "method": "all_methods_failed",
                            "error": str(e),
                            "fallback_error": str(fallback_error)
                        },
                        passed=False
                    )

    async def evaluate_qa_pair(
        self,
        qa_pair: Dict[str, Any]
    ) -> List[MetricResult]:
        """Evaluate a single QA pair"""
        
        # Create ground truth dictionary from curated data
        ground_truth = {
            "question": qa_pair['question'],
            "answer": qa_pair.get('curation_status', {}).get('modified_answer') or qa_pair['answer'],
            "context": qa_pair['context']
        }
        
        # Create response object from agent data
        response = AgentResponse(
            answer=qa_pair['agent_answer'],
            source_chunks=[
                SourceChunk(
                    text=node['text'],
                    document=node['metadata'].get('document', ''),
                    page=node['metadata'].get('page'),
                    chunk_id=str(idx)
                )
                for idx, node in enumerate(qa_pair.get('agent_source_nodes', []))
            ],
            confidence_score=1.0,  # Default if not available
            reasoning_path=qa_pair.get('agent_reasoning_steps', [])
        )
        
        # Run all metrics
        results = []
        for metric in self.metrics:
            if metric.type == MetricType.PROGRAMMATIC:
                result = await self._evaluate_programmatic(metric, response, ground_truth)
            elif metric.type == MetricType.LLM:
                result = await self._evaluate_llm(metric, response, ground_truth)
            else:  # HYBRID
                result = await self._evaluate_hybrid(metric, response, ground_truth)
            results.append(result)
    
        return results

class EvaluationRunner:
    """Manages the evaluation process and results storage"""
    
    def __init__(
        self,
        metrics_engine: MetricsEngine,
        output_dir: str = "evaluation_results"
    ):
        self.metrics_engine = metrics_engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
        
    def _load_or_create_report(self, qa_file: str, timestamp: str) -> Dict[str, Any]:
        """Load existing report or create new one"""
        output_file = self.output_dir / f"evaluation_report.json"
        
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        # Create new report structure
        return {
            "timestamp": timestamp,
            "total_pairs_evaluated": 0,
            "results": [],
            "summary": None
        }
    
    def _save_progress(self, report: Dict[str, Any], timestamp: str):
        """Save current progress to file"""
        output_file = self.output_dir / f"evaluation_report.json"
        
        # Convert numpy types before saving
        converted_report = self._convert_numpy_types(report)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
            # Save with basic error handling
            simplified_report = {
                "timestamp": converted_report["timestamp"],
                "total_pairs_evaluated": converted_report["total_pairs_evaluated"],
                "results": converted_report["results"],
                "error": str(e)
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_report, f, indent=2, ensure_ascii=False)

    async def _get_human_evaluation(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        print("\nHuman Evaluation")
        print("-" * 50)
        print(f"Question: {qa_pair['question']}")
        print(f"Ground Truth: {qa_pair['answer']}")
        print(f"Agent Answer: {qa_pair['agent_answer']}")
        
        while True:
            status = input("\nEnter evaluation (p=PASS, f=FAIL, e=EXCLUDE): ").lower()
            if status in ['p', 'f', 'e']:
                break
                
        # notes = input("Enter any notes (optional): ")
        
        status_map = {'p': 'PASS', 'f': 'FAIL', 'e': 'EXCLUDE'}
        
        return {
            "completed": True,
            "status": status_map[status],
            # "notes": notes
        }

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results"""
        summary = {
            "total_qa_pairs": len(results),
            "metrics_summary": {},
            "overall_scores": [],
            "overall_pass_rates": {
                "total_passed": 0,
                "pass_rate": 0.0,
                "qa_results": []
            },
            "human_evaluation_summary": {
                "completed_evaluations": 0,
                "average_scores": {},
                "flags_summary": {}
            }
        }
        
        # Calculate metrics summary, overall scores, and pass rates
        for result in results:
            metrics_scores = {}
            weighted_sum = 0
            total_weight = 0
            passed_metrics = 0
            total_metrics = 0
            
            for metric_result in result["metrics"]:
                metric_name = metric_result["name"]
                total_metrics += 1
                
                # Initialize metric summary if needed
                if metric_name not in summary["metrics_summary"]:
                    summary["metrics_summary"][metric_name] = {
                        "scores": [],
                        "pass_rate": 0
                    }
                
                # Add score to metrics summary
                summary["metrics_summary"][metric_name]["scores"].append(metric_result["score"])
                
                # Track score and weight for overall calculation
                metrics_scores[metric_name] = metric_result["score"]
                weight = metric_result.get("weight", 0.0)
                weighted_sum += metric_result["score"] * weight
                total_weight += weight
                
                # Track passed metrics
                if metric_result["passed"]:
                    passed_metrics += 1
            
            # Calculate overall score for this QA pair
            if total_weight > 0:
                overall_score = weighted_sum / total_weight
                summary["overall_scores"].append(overall_score)
            
            # Determine if QA pair passed (7/9 metrics must pass)
            qa_passed = passed_metrics >= OVERALL_PASS_THRESHOLD
            if qa_passed:
                summary["overall_pass_rates"]["total_passed"] += 1
                
            # Store QA result
            summary["overall_pass_rates"]["qa_results"].append({
                "question": result["question"],
                "passed": qa_passed,
                "passed_metrics": passed_metrics,
                "total_metrics": total_metrics,
                "metrics_passed": [
                    {
                        "name": m["name"],
                        "passed": m["passed"],
                        "score": m["score"]
                    }
                    for m in result["metrics"]
                ]
            })
        
        # Calculate overall pass rate
        total_qa_pairs = len(results)
        if total_qa_pairs > 0:
            summary["overall_pass_rates"]["pass_rate"] = (
                summary["overall_pass_rates"]["total_passed"] / total_qa_pairs * 100
            )
        
        # Calculate statistics for each metric
        for metric_name, metric_data in summary["metrics_summary"].items():
            scores = metric_data["scores"]
            passed = sum(1 for score in scores if score >= 0.7)  # Default threshold
            metric_data.update({
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "median_score": float(np.median(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "pass_rate": (passed / len(scores)) * 100 if scores else 0
            })
        
        # Add overall score statistics
        if summary["overall_scores"]:
            summary["overall_summary"] = {
                "mean_score": float(np.mean(summary["overall_scores"])),
                "std_score": float(np.std(summary["overall_scores"])),
                "median_score": float(np.median(summary["overall_scores"])),
                "min_score": float(np.min(summary["overall_scores"])),
                "max_score": float(np.max(summary["overall_scores"])),
                "overall_pass_rate": summary["overall_pass_rates"]["pass_rate"]
            }
        
        return summary


    async def run_evaluation(
        self,
        qa_file: str,
        include_human_eval: bool = False
    ) -> Dict[str, Any]:
        """Run evaluation on QA pairs from file"""
        
        # Load QA pairs
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # Load or create report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report = self._load_or_create_report(qa_file, timestamp)
        
        # Get all metric names
        available_metric_names = {m.name for m in self.metrics_engine.metrics}
        
        # Process each QA pair
        for i, result in enumerate(report['results']):
            # Check for missing metrics
            existing_metric_names = {m['name'] for m in result['metrics']}
            missing_metrics = [m for m in self.metrics_engine.metrics 
                            if m.name not in existing_metric_names]
            
            if missing_metrics:
                logger.info(f"\nCalculating missing metrics for QA pair {i+1}")
                logger.info(f"Question: {result['question']}")
                
                try:
                    # Create necessary objects for evaluation
                    agent_response = AgentResponse(
                        answer=result['agent_answer'],
                        source_chunks=[
                            SourceChunk(
                                text=node['text'],
                                document=node['metadata'].get('document', ''),
                                page=node['metadata'].get('page_label'),
                                chunk_id=str(idx)
                            )
                            for idx, node in enumerate(result.get('source_nodes', []))
                        ],
                        confidence_score=1.0,
                        reasoning_path=result.get('reasoning_steps', [])
                    )
                    
                    ground_truth = {
                        "question": result['question'],
                        "answer": result['curated_answer'],
                        "context": result['context']
                    }

                    # Calculate only missing metrics
                    for metric in missing_metrics:
                        try:
                            if metric.type == MetricType.PROGRAMMATIC:
                                metric_result = await self.metrics_engine._evaluate_programmatic(
                                    metric, agent_response, ground_truth
                                )
                            elif metric.type == MetricType.LLM:
                                metric_result = await self.metrics_engine._evaluate_llm(
                                    metric, agent_response, ground_truth
                                )
                            else:  # HYBRID
                                metric_result = await self.metrics_engine._evaluate_hybrid(
                                    metric, agent_response, ground_truth
                                )
                            
                            # Skip metrics that returned None (not applicable)
                            if metric_result.score is not None:
                                # Add new metric to results
                                result['metrics'].append({
                                    "name": metric_result.metric.name,
                                    "score": float(metric_result.score),
                                    "passed": bool(metric_result.passed),
                                    "details": metric_result.details,
                                    "weight": float(metric_result.metric.weight)
                                })
                                
                                logger.info(f"{metric.name:<20}: {metric_result.score:.3f} "
                                        f"({'PASS' if metric_result.passed else 'FAIL'})")
                            
                        except Exception as e:
                            logger.error(f"Error calculating {metric.name}: {e}")
                    
                    # Recalculate overall results if needed
                    if missing_metrics:
                        # Only include metrics that have actual scores and weights
                        valid_metrics = [m for m in result['metrics'] 
                                    if m['score'] is not None and m['weight'] > 0]
                        
                        if valid_metrics:
                            weighted_sum = sum(m['score'] * m['weight'] for m in valid_metrics)
                            total_weight = sum(m['weight'] for m in valid_metrics)
                            overall_score = weighted_sum / total_weight if total_weight > 0 else 0
                        else:
                            overall_score = 0
                        
                        # Only count applicable metrics for pass/fail
                        valid_pass_metrics = [m for m in result['metrics'] 
                                            if m['passed'] is not None and 
                                            m['name'] != 'completeness_gain']
                        
                        passed_metrics = sum(1 for m in valid_pass_metrics if m['passed'])
                        total_metrics = len(valid_pass_metrics)
                        
                        result['overall_results'] = {
                            "score": float(overall_score),
                            "passed_metrics": passed_metrics,
                            "total_metrics": total_metrics,
                            "passed": passed_metrics >= OVERALL_PASS_THRESHOLD
                        }
                    
                except Exception as e:
                    logger.error(f"Error processing QA pair: {e}")
                    continue
                
                # Save progress after each update
                self._save_progress(report, timestamp)
        
        # Generate final summary
        report['summary'] = self._generate_summary(report['results'])
        self._save_progress(report, timestamp)
        
        logger.info(f"\nEvaluation complete. Report saved to evaluation_report.json")
        return report

async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent responses against curated QA pairs"
    )
    parser.add_argument(
        '--qa-file',
        '-q',
        required=True,
        help='Path to QA pairs JSON file with agent responses'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        default='evaluation_results',
        help='Directory for evaluation results'
    )
    parser.add_argument(
        '--model',
        '-m',
        default='qwen2.5:14b',
        help='Model to use for LLM-based evaluation metrics'
    )
    parser.add_argument(
        '--human-eval',
        '-e',
        action='store_true',
        help='Include interactive human evaluation'
    )
    
    args = parser.parse_args()
    
    metrics_engine = MetricsEngine(model_name=args.model)
    evaluator = EvaluationRunner(
        metrics_engine=metrics_engine,
        output_dir=args.output_dir
    )
    
    await evaluator.run_evaluation(args.qa_file, include_human_eval=args.human_eval)

def analyze_evaluation_results(results_file: str, show_std: bool = False, show_human: bool = False) -> None:
    """Analyze evaluation results from JSON file and display summary with visualization"""
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out results with EXCLUDE status
    filtered_results = [
        r for r in data['results']
        if r['human_evaluation'].get('status', '') != 'EXCLUDE'
    ]

    # Calculate overall statistics
    overall_scores = [r['overall_results']['score'] for r in filtered_results]
    passed_qa_pairs = sum(1 for r in filtered_results if r['overall_results']['passed'])
    total_qa_pairs = len(filtered_results)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Define metrics to exclude
    excluded_metrics = {}
    
    # Get metric names
    metric_names = set()
    for result in filtered_results:
        metric_names.update(m['name'] for m in result['metrics'] 
                          if m['name'] not in excluded_metrics)
    metric_names = sorted(metric_names)
    
    # Add human evaluation if requested
    if show_human:
        metric_names = list(metric_names) + ['human_evaluation']
    
    # Calculate statistics for automated metrics
    metric_scores = {name: [] for name in metric_names if name != 'human_evaluation'}
    metric_pass_rates = {name: [] for name in metric_names}
    
    for result in filtered_results:
        # Collect automated metrics
        for metric in result['metrics']:
            name = metric['name']
            if name not in excluded_metrics:
                metric_scores[name].append(metric['score'])
                metric_pass_rates[name].append(metric['passed'])
        
        # Add human evaluation results if available
        if show_human and result['human_evaluation'].get('completed', False):
            metric_pass_rates['human_evaluation'].append(result['human_evaluation'].get('status') == 'PASS')
    
    statistics = {}
    for name in metric_names:
        if name != 'human_evaluation':
            scores = metric_scores[name]
            if scores:
                pass_rate = np.mean([1 if x else 0 for x in metric_pass_rates[name]]) * 100
                statistics[name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'pass_rate': pass_rate,
                    'pass_rate_std': np.std([1 if x else 0 for x in metric_pass_rates[name]]) * 100
                }
        else:
            # Only calculate pass rate for human evaluation
            human_results = metric_pass_rates[name]
            if human_results:
                pass_rate = np.mean([1 if x else 0 for x in human_results]) * 100
                statistics[name] = {
                    'mean': pass_rate/100,  # For consistency in plotting
                    'std': 0,
                    'min': 0,
                    'max': 1,
                    'pass_rate': pass_rate,
                    'pass_rate_std': np.std([1 if x else 0 for x in human_results]) * 100
                }
    
    # Plot metrics
    x = np.arange(len(metric_names))
    width = 0.35
    
    for i, name in enumerate(metric_names):
        if name == 'human_evaluation':
            if name in statistics:
                # For human eval, only plot pass rate
                ax.bar(x[i], statistics[name]['pass_rate']/100, width, 
                      color='lightgreen', alpha=0.7)
        else:
            if name in statistics:
                # For other metrics, plot both mean and pass rate
                ax.bar(x[i] - width/2, statistics[name]['mean'], width, 
                      label='Mean Score' if i == 0 else '', color='skyblue')
                ax.bar(x[i] + width/2, statistics[name]['pass_rate']/100, width,
                      label='Pass Rate' if i == 0 else '', color='lightgreen')
                
                if show_std:
                    ax.errorbar(x[i] - width/2, statistics[name]['mean'], 
                              yerr=statistics[name]['std'], fmt='none', 
                              color='black', capsize=5)
                    ax.errorbar(x[i] + width/2, statistics[name]['pass_rate']/100, 
                              yerr=statistics[name]['pass_rate_std']/100, fmt='none',
                              color='black', capsize=5)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score (0-1) / Pass Rate')
    ax.set_title('Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(results_file).parent / 'evaluation_summary.png'
    plt.savefig(plot_path)
    
    # Print numerical summary
    print("\nEvaluation Results Summary")
    print("="*50)
    print(f"Total QA pairs evaluated: {total_qa_pairs}")
    print(f"Passed QA pairs: {passed_qa_pairs} ({(passed_qa_pairs/total_qa_pairs*100):.1f}%)")
    print(f"Overall Score: {np.mean(overall_scores):.3f} ± {np.std(overall_scores):.3f}")
    
    print("\nMetric-level Statistics:")
    print("-"*70)
    print(f"{'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Pass%':>8}")
    print("-"*70)
    
    for name in metric_names:
        if name in statistics:
            stats = statistics[name]
            if name == 'human_evaluation':
                print(f"{name:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {stats['pass_rate']:8.1f}")
            else:
                print(f"{name:<20} {stats['mean']:8.3f} {stats['std']:8.3f} {stats['min']:8.3f} {stats['max']:8.3f} {stats['pass_rate']:8.1f}")
    
    print(f"\nPlot saved to: {plot_path}")

if __name__ == "__main__":
    asyncio.run(main())
    # Usage: python metrics_eval.py -q curated_search/qwen_react_vanilla_hybrid/qa_pairs_curated_with_responses.json -o EVAL/qwen_react_vanilla_hybrid --human-eval 
    # Usage: python metrics_eval.py -q curated_summaries/qwen_hybrid_search/summary_qa_pairs_curated_with_responses.json -o evaluation_results_summary/qwen_hybrid_search --human-eval 
    analyze_evaluation_results('./EVAL/qwen_react_vanilla_hybrid/evaluation_report.json')#, show_human=True)