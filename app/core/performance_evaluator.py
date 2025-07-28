"""
Performance Evaluator for comprehensive prompt performance measurement.

Evaluates prompts across multiple dimensions including accuracy, relevance,
coherence, token efficiency, and user satisfaction metrics.
"""

import asyncio
import json
import time
import math
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation system for prompts.
    
    Evaluates prompts across multiple dimensions:
    - Accuracy: How correct and factual the responses are
    - Relevance: How well responses address the specific query
    - Coherence: How logical and well-structured responses are
    - Token Efficiency: Response quality per token used
    - Clarity: How clear and understandable responses are
    - User Satisfaction: Overall usefulness and quality
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.logger = logger.bind(component="performance_evaluator")
        
        # Evaluation configuration
        self.evaluation_timeout_seconds = 60
        self.min_test_cases = 3
        self.max_test_cases = 50
        
        # Metric weights for overall score calculation
        self.metric_weights = {
            'accuracy': 0.25,
            'relevance': 0.20,
            'coherence': 0.15,
            'token_efficiency': 0.15,
            'clarity': 0.15,
            'user_satisfaction': 0.10
        }
        
        # Evaluation rubrics
        self.evaluation_rubrics = {
            'accuracy': {
                'description': 'Factual correctness and precision of information',
                'criteria': [
                    'Information is factually correct',
                    'Claims are supported by evidence',
                    'No misleading or false statements',
                    'Appropriate level of certainty expressed'
                ]
            },
            'relevance': {
                'description': 'How well the response addresses the specific query',
                'criteria': [
                    'Directly addresses the question asked',
                    'Stays focused on the topic',
                    'Includes relevant information only',
                    'Covers all aspects of the query'
                ]
            },
            'coherence': {
                'description': 'Logical flow and structure of the response',
                'criteria': [
                    'Ideas flow logically from one to another',
                    'Clear organizational structure',
                    'Consistent reasoning throughout',
                    'Appropriate use of transitions'
                ]
            },
            'token_efficiency': {
                'description': 'Quality of information per token used',
                'criteria': [
                    'Concise without losing important information',
                    'No unnecessary repetition',
                    'Efficient use of language',
                    'High information density'
                ]
            },
            'clarity': {
                'description': 'How clear and understandable the response is',
                'criteria': [
                    'Uses clear, appropriate language',
                    'Avoids unnecessary jargon',
                    'Well-structured sentences',
                    'Easy to follow and understand'
                ]
            },
            'user_satisfaction': {
                'description': 'Overall usefulness and quality of the response',
                'criteria': [
                    'Helpful and actionable information',
                    'Appropriate tone and style',
                    'Meets user expectations',
                    'Provides value to the user'
                ]
            }
        }
    
    async def evaluate_prompt(
        self,
        prompt_content: str,
        test_cases: List[Dict[str, Any]],
        metrics: List[str] = None,
        evaluation_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt's performance across multiple test cases and metrics.
        
        Args:
            prompt_content: The prompt to evaluate
            test_cases: List of test cases with input/expected output
            metrics: List of metrics to evaluate (defaults to all)
            evaluation_config: Additional evaluation configuration
            
        Returns:
            Dict containing performance scores and detailed results
        """
        try:
            start_time = time.time()
            
            # Validate inputs
            if not test_cases:
                raise ValueError("At least one test case is required")
            
            if len(test_cases) > self.max_test_cases:
                test_cases = test_cases[:self.max_test_cases]
                self.logger.warning(
                    "Truncated test cases to maximum allowed",
                    original_count=len(test_cases),
                    max_allowed=self.max_test_cases
                )
            
            # Use all metrics if none specified
            if not metrics:
                metrics = list(self.metric_weights.keys())
            
            # Validate metrics
            invalid_metrics = [m for m in metrics if m not in self.metric_weights]
            if invalid_metrics:
                raise ValueError(f"Invalid metrics: {invalid_metrics}")
            
            self.logger.info(
                "Starting prompt evaluation",
                prompt_length=len(prompt_content),
                num_test_cases=len(test_cases),
                metrics=metrics
            )
            
            # Initialize results storage
            evaluation_results = {
                'detailed_metrics': defaultdict(list),
                'test_case_results': [],
                'metric_statistics': {},
                'token_usage': {'total_input_tokens': 0, 'total_output_tokens': 0},
                'evaluation_metadata': {
                    'prompt_length': len(prompt_content),
                    'num_test_cases': len(test_cases),
                    'metrics_evaluated': metrics,
                    'evaluation_config': evaluation_config or {}
                }
            }
            
            # Evaluate each test case
            for i, test_case in enumerate(test_cases):
                case_result = await self._evaluate_single_test_case(
                    prompt_content=prompt_content,
                    test_case=test_case,
                    metrics=metrics,
                    case_index=i
                )
                
                evaluation_results['test_case_results'].append(case_result)
                
                # Aggregate metrics
                for metric, score in case_result['metric_scores'].items():
                    evaluation_results['detailed_metrics'][metric].append(score)
                
                # Aggregate token usage
                usage = case_result.get('token_usage', {})
                evaluation_results['token_usage']['total_input_tokens'] += usage.get('input_tokens', 0)
                evaluation_results['token_usage']['total_output_tokens'] += usage.get('output_tokens', 0)
            
            # Calculate aggregated metric scores
            aggregated_metrics = {}
            for metric, scores in evaluation_results['detailed_metrics'].items():
                if scores:
                    aggregated_metrics[metric] = {
                        'mean': sum(scores) / len(scores),
                        'std': self._calculate_std(scores),
                        'min': min(scores),
                        'max': max(scores),
                        'median': self._calculate_median(scores)
                    }
            
            evaluation_results['metric_statistics'] = aggregated_metrics
            
            # Calculate overall performance score
            overall_score = self._calculate_overall_score(aggregated_metrics, metrics)
            
            # Calculate token efficiency
            total_tokens = (
                evaluation_results['token_usage']['total_input_tokens'] +
                evaluation_results['token_usage']['total_output_tokens']
            )
            
            if 'token_efficiency' in aggregated_metrics:
                token_efficiency_score = aggregated_metrics['token_efficiency']['mean']
            else:
                # Fallback calculation if not in metrics
                token_efficiency_score = min(1.0, overall_score * (1000 / max(total_tokens, 100)))
            
            evaluation_time = time.time() - start_time
            
            result = {
                'performance_score': overall_score,
                'detailed_metrics': {metric: stats['mean'] for metric, stats in aggregated_metrics.items()},
                'metric_statistics': aggregated_metrics,
                'test_case_results': evaluation_results['test_case_results'],
                'token_usage': evaluation_results['token_usage'],
                'evaluation_time': evaluation_time,
                'context': {
                    'total_test_cases': len(test_cases),
                    'metrics_evaluated': metrics,
                    'evaluation_method': 'automated_comprehensive',
                    'token_efficiency_score': token_efficiency_score,
                    'evaluation_timestamp': time.time()
                }
            }
            
            self.logger.info(
                "Prompt evaluation completed",
                performance_score=overall_score,
                evaluation_time=evaluation_time,
                total_tokens=total_tokens
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to evaluate prompt",
                error=str(e),
                prompt_length=len(prompt_content) if prompt_content else 0
            )
            raise
    
    async def evaluate_batch(
        self,
        prompts: List[str],
        test_cases: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prompts in batch for efficiency.
        
        Args:
            prompts: List of prompts to evaluate
            test_cases: Test cases to use for all prompts
            metrics: Metrics to evaluate
            
        Returns:
            List of evaluation results for each prompt
        """
        try:
            self.logger.info(
                "Starting batch evaluation",
                num_prompts=len(prompts),
                num_test_cases=len(test_cases)
            )
            
            # Run evaluations concurrently with limited concurrency
            max_concurrent = 3
            results = []
            
            for i in range(0, len(prompts), max_concurrent):
                batch = prompts[i:i + max_concurrent]
                batch_tasks = [
                    self.evaluate_prompt(prompt, test_cases, metrics)
                    for prompt in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error("Batch evaluation failed for prompt", error=str(result))
                        results.append({
                            'performance_score': 0.0,
                            'detailed_metrics': {},
                            'error': str(result)
                        })
                    else:
                        results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error("Failed batch evaluation", error=str(e))
            raise
    
    async def compare_prompts(
        self,
        prompt_a: str,
        prompt_b: str,
        test_cases: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two prompts head-to-head across test cases.
        
        Args:
            prompt_a: First prompt to compare
            prompt_b: Second prompt to compare
            test_cases: Test cases for comparison
            metrics: Metrics to compare on
            
        Returns:
            Dict containing comparison results
        """
        try:
            self.logger.info(
                "Starting prompt comparison",
                prompt_a_length=len(prompt_a),
                prompt_b_length=len(prompt_b),
                num_test_cases=len(test_cases)
            )
            
            # Evaluate both prompts
            results_a, results_b = await asyncio.gather(
                self.evaluate_prompt(prompt_a, test_cases, metrics),
                self.evaluate_prompt(prompt_b, test_cases, metrics)
            )
            
            # Calculate comparison metrics
            comparison = {
                'prompt_a_score': results_a['performance_score'],
                'prompt_b_score': results_b['performance_score'],
                'score_difference': results_b['performance_score'] - results_a['performance_score'],
                'relative_improvement': (
                    (results_b['performance_score'] - results_a['performance_score']) /
                    max(results_a['performance_score'], 0.001) * 100
                ),
                'metric_comparisons': {},
                'winner': 'prompt_b' if results_b['performance_score'] > results_a['performance_score'] else 'prompt_a',
                'statistical_significance': None  # Would need multiple runs for real significance
            }
            
            # Compare individual metrics
            for metric in (metrics or list(self.metric_weights.keys())):
                if metric in results_a['detailed_metrics'] and metric in results_b['detailed_metrics']:
                    score_a = results_a['detailed_metrics'][metric]
                    score_b = results_b['detailed_metrics'][metric]
                    
                    comparison['metric_comparisons'][metric] = {
                        'prompt_a': score_a,
                        'prompt_b': score_b,
                        'difference': score_b - score_a,
                        'relative_improvement': (score_b - score_a) / max(score_a, 0.001) * 100,
                        'winner': 'prompt_b' if score_b > score_a else 'prompt_a'
                    }
            
            self.logger.info(
                "Prompt comparison completed",
                winner=comparison['winner'],
                score_difference=comparison['score_difference']
            )
            
            return comparison
            
        except Exception as e:
            self.logger.error("Failed to compare prompts", error=str(e))
            raise
    
    # Private helper methods
    
    async def _evaluate_single_test_case(
        self,
        prompt_content: str,
        test_case: Dict[str, Any],
        metrics: List[str],
        case_index: int
    ) -> Dict[str, Any]:
        """Evaluate a single test case."""
        try:
            # Simulate LLM response generation
            response = await self._generate_response(prompt_content, test_case)
            
            # Calculate metrics for this test case
            metric_scores = {}
            for metric in metrics:
                score = await self._calculate_metric_score(
                    metric=metric,
                    response=response,
                    test_case=test_case,
                    prompt_content=prompt_content
                )
                metric_scores[metric] = score
            
            return {
                'test_case_index': case_index,
                'test_case_id': test_case.get('id'),
                'response': response['content'],
                'metric_scores': metric_scores,
                'token_usage': response['token_usage'],
                'response_time': response['response_time'],
                'evaluation_notes': response.get('evaluation_notes', {})
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to evaluate test case",
                case_index=case_index,
                error=str(e)
            )
            # Return minimal result with error
            return {
                'test_case_index': case_index,
                'metric_scores': {metric: 0.0 for metric in metrics},
                'token_usage': {'input_tokens': 0, 'output_tokens': 0},
                'error': str(e)
            }
    
    async def _generate_response(
        self,
        prompt_content: str,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response using the prompt (simulated)."""
        # Simulate API call latency
        await asyncio.sleep(0.1)
        
        # Extract input from test case
        input_data = test_case.get('input_data', {})
        input_text = input_data.get('text', '') if isinstance(input_data, dict) else str(input_data)
        
        # Simulate token counting
        input_tokens = len(prompt_content.split()) + len(input_text.split())
        
        # Simulate response generation (in real implementation, would call LLM API)
        response_content = f"Simulated response to: {input_text[:100]}... based on prompt guidance."
        output_tokens = len(response_content.split())
        
        return {
            'content': response_content,
            'token_usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            },
            'response_time': 0.1,
            'evaluation_notes': {
                'generated_via': 'simulated_llm',
                'prompt_length': len(prompt_content),
                'input_length': len(input_text)
            }
        }
    
    async def _calculate_metric_score(
        self,
        metric: str,
        response: Dict[str, Any],
        test_case: Dict[str, Any],
        prompt_content: str
    ) -> float:
        """Calculate score for a specific metric."""
        response_content = response['content']
        expected_output = test_case.get('expected_output', '')
        
        if metric == 'accuracy':
            return await self._calculate_accuracy_score(
                response_content, expected_output, test_case
            )
        elif metric == 'relevance':
            return await self._calculate_relevance_score(
                response_content, test_case
            )
        elif metric == 'coherence':
            return await self._calculate_coherence_score(response_content)
        elif metric == 'token_efficiency':
            return await self._calculate_token_efficiency_score(
                response_content, response['token_usage']
            )
        elif metric == 'clarity':
            return await self._calculate_clarity_score(response_content)
        elif metric == 'user_satisfaction':
            return await self._calculate_user_satisfaction_score(
                response_content, test_case
            )
        else:
            self.logger.warning("Unknown metric", metric=metric)
            return 0.5  # Default neutral score
    
    async def _calculate_accuracy_score(
        self,
        response: str,
        expected_output: str,
        test_case: Dict[str, Any]
    ) -> float:
        """Calculate accuracy score based on expected output comparison."""
        if not expected_output:
            # No expected output to compare against, use heuristics
            return await self._calculate_heuristic_accuracy(response, test_case)
        
        # Simple similarity calculation
        response_words = set(response.lower().split())
        expected_words = set(expected_output.lower().split())
        
        if not expected_words:
            return 0.5
        
        # Calculate word overlap
        overlap = len(response_words.intersection(expected_words))
        union = len(response_words.union(expected_words))
        
        jaccard_similarity = overlap / max(union, 1)
        
        # Check for key information preservation
        key_info_score = 0.0
        if 'evaluation_criteria' in test_case:
            criteria = test_case['evaluation_criteria']
            if isinstance(criteria, dict) and 'key_points' in criteria:
                key_points = criteria['key_points']
                key_points_found = sum(
                    1 for point in key_points
                    if point.lower() in response.lower()
                )
                key_info_score = key_points_found / max(len(key_points), 1)
        
        # Combine scores
        accuracy_score = (jaccard_similarity * 0.6) + (key_info_score * 0.4)
        return min(1.0, accuracy_score)
    
    async def _calculate_heuristic_accuracy(
        self,
        response: str,
        test_case: Dict[str, Any]
    ) -> float:
        """Calculate accuracy using heuristics when no expected output available."""
        # Basic heuristics for accuracy assessment
        score = 0.7  # Base score
        
        # Penalize if response is too short or too long
        word_count = len(response.split())
        if word_count < 5:
            score -= 0.3
        elif word_count > 500:
            score -= 0.1
        
        # Bonus for structured response
        if any(marker in response for marker in ['1.', '2.', '-', '*']):
            score += 0.1
        
        # Penalty for obvious errors or nonsense
        error_indicators = ['error', 'cannot', 'unable', 'sorry', 'unclear']
        if any(indicator in response.lower() for indicator in error_indicators):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    async def _calculate_relevance_score(
        self,
        response: str,
        test_case: Dict[str, Any]
    ) -> float:
        """Calculate how relevant the response is to the input query."""
        input_data = test_case.get('input_data', {})
        
        if isinstance(input_data, dict):
            query_text = input_data.get('text', '')
        else:
            query_text = str(input_data)
        
        if not query_text:
            return 0.5
        
        # Extract key terms from query
        query_words = set(query_text.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= stop_words
        response_words -= stop_words
        
        if not query_words:
            return 0.5
        
        # Calculate relevance based on term overlap
        relevance = len(query_words.intersection(response_words)) / len(query_words)
        
        # Bonus for direct addressing of the query
        if any(question_word in query_text.lower() for question_word in ['what', 'how', 'why', 'when', 'where', 'who']):
            # Check if response appropriately addresses the question type
            if 'what' in query_text.lower() and ('is' in response.lower() or 'are' in response.lower()):
                relevance += 0.1
            elif 'how' in query_text.lower() and any(word in response.lower() for word in ['step', 'process', 'method']):
                relevance += 0.1
        
        return min(1.0, relevance)
    
    async def _calculate_coherence_score(self, response: str) -> float:
        """Calculate coherence based on logical flow and structure."""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.7  # Single sentence gets moderate score
        
        coherence_score = 0.8  # Base score
        
        # Check for logical connectors
        connectors = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 'consequently']
        connector_count = sum(1 for sentence in sentences for connector in connectors if connector in sentence.lower())
        connector_bonus = min(0.1, connector_count / len(sentences))
        coherence_score += connector_bonus
        
        # Check for consistent tense usage
        past_tense_indicators = sum(1 for sentence in sentences if any(word.endswith('ed') for word in sentence.split()))
        present_tense_indicators = sum(1 for sentence in sentences if any(word in sentence.lower() for word in ['is', 'are', 'am']))
        
        tense_consistency = 1.0 - abs(past_tense_indicators - present_tense_indicators) / max(len(sentences), 1)
        coherence_score += tense_consistency * 0.1
        
        # Penalty for abrupt topic changes (simplified heuristic)
        topic_changes = 0
        for i in range(1, len(sentences)):
            prev_words = set(sentences[i-1].lower().split())
            curr_words = set(sentences[i].lower().split())
            overlap = len(prev_words.intersection(curr_words))
            if overlap / max(len(prev_words), 1) < 0.1:  # Very little overlap
                topic_changes += 1
        
        if topic_changes > len(sentences) // 3:  # Too many topic changes
            coherence_score -= 0.2
        
        return max(0.0, min(1.0, coherence_score))
    
    async def _calculate_token_efficiency_score(
        self,
        response: str,
        token_usage: Dict[str, int]
    ) -> float:
        """Calculate token efficiency based on information density."""
        total_tokens = token_usage.get('total_tokens', len(response.split()))
        
        if total_tokens == 0:
            return 0.0
        
        # Calculate information density metrics
        word_count = len(response.split())
        unique_words = len(set(response.lower().split()))
        
        # Information density score
        info_density = unique_words / max(word_count, 1)
        
        # Penalty for excessive length without proportional information
        length_penalty = 0.0
        if total_tokens > 200:
            length_penalty = min(0.3, (total_tokens - 200) / 1000)
        
        # Bonus for conciseness with good information
        conciseness_bonus = 0.0
        if 50 <= total_tokens <= 150 and info_density > 0.7:
            conciseness_bonus = 0.2
        
        efficiency_score = info_density + conciseness_bonus - length_penalty
        return max(0.0, min(1.0, efficiency_score))
    
    async def _calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity based on readability and language simplicity."""
        words = response.split()
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Average sentence length (shorter is often clearer)
        avg_sentence_length = len(words) / len(sentences)
        sentence_length_score = max(0.0, 1.0 - (avg_sentence_length - 15) / 30)  # Optimal around 15 words
        
        # Average word length (shorter words are often clearer)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        word_length_score = max(0.0, 1.0 - (avg_word_length - 5) / 10)  # Optimal around 5 characters
        
        # Check for complex constructions
        complex_indicators = ['however', 'nevertheless', 'furthermore', 'consequently']
        complex_count = sum(1 for indicator in complex_indicators if indicator in response.lower())
        complexity_penalty = min(0.2, complex_count / max(len(sentences), 1))
        
        # Check for passive voice (simplified detection)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(1 for indicator in passive_indicators if indicator in response.lower())
        passive_penalty = min(0.1, passive_count / max(len(words), 1) * 10)
        
        clarity_score = (sentence_length_score * 0.4 + word_length_score * 0.4) - complexity_penalty - passive_penalty
        return max(0.0, min(1.0, clarity_score))
    
    async def _calculate_user_satisfaction_score(
        self,
        response: str,
        test_case: Dict[str, Any]
    ) -> float:
        """Calculate user satisfaction based on helpfulness and completeness."""
        satisfaction_score = 0.7  # Base score
        
        # Check for completeness
        if len(response.split()) < 10:
            satisfaction_score -= 0.3  # Too brief
        elif len(response.split()) > 300:
            satisfaction_score -= 0.1  # Potentially too verbose
        
        # Bonus for actionable information
        actionable_indicators = ['you can', 'try', 'consider', 'recommend', 'suggest', 'should']
        if any(indicator in response.lower() for indicator in actionable_indicators):
            satisfaction_score += 0.1
        
        # Bonus for examples or specifics
        example_indicators = ['example', 'for instance', 'such as', 'like']
        if any(indicator in response.lower() for indicator in example_indicators):
            satisfaction_score += 0.1
        
        # Check for politeness/helpfulness
        polite_indicators = ['please', 'thank you', 'hope this helps', 'let me know']
        if any(indicator in response.lower() for indicator in polite_indicators):
            satisfaction_score += 0.05
        
        # Penalty for disclaimers or uncertainty (sometimes necessary but may reduce satisfaction)
        uncertainty_indicators = ['might', 'could', 'possibly', 'perhaps', 'maybe']
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in response.lower())
        if uncertainty_count > len(response.split()) / 20:  # More than 5% uncertainty words
            satisfaction_score -= 0.1
        
        return max(0.0, min(1.0, satisfaction_score))
    
    def _calculate_overall_score(
        self,
        aggregated_metrics: Dict[str, Dict[str, float]],
        evaluated_metrics: List[str]
    ) -> float:
        """Calculate weighted overall performance score."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric in evaluated_metrics:
            if metric in aggregated_metrics and metric in self.metric_weights:
                weight = self.metric_weights[metric]
                score = aggregated_metrics[metric]['mean']
                total_score += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _calculate_median(self, values: List[float]) -> float:
        """Calculate median value."""
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            return sorted_values[n // 2]