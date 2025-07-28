"""
Core Prompt Optimization Service for LeanVibe Agent Hive 2.0.

Orchestrates the prompt optimization process including generation, evaluation,
A/B testing, evolutionary optimization, and feedback integration.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, func
from sqlalchemy.orm import selectinload

from ..models.prompt_optimization import (
    PromptTemplate, OptimizationExperiment, PromptVariant, PromptEvaluation,
    ABTestResult, PromptFeedback, PromptTestCase, OptimizationMetric,
    PromptStatus, ExperimentStatus, OptimizationMethod
)
from ..core.database import get_session_dependency
from .prompt_generator import PromptGenerator
from .performance_evaluator import PerformanceEvaluator
from .ab_testing_engine import ABTestingEngine  
from .evolutionary_optimizer import EvolutionaryOptimizer
from .feedback_analyzer import FeedbackAnalyzer
from .context_adapter import ContextAdapter

logger = structlog.get_logger()


class OptimizationPriority(str, Enum):
    """Priority levels for optimization tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PromptOptimizer:
    """
    Core service for automated prompt optimization.
    
    Coordinates multiple optimization strategies including:
    - LLM-powered prompt generation and refinement
    - Performance evaluation with metrics collection
    - A/B testing for statistical validation
    - Evolutionary optimization using genetic algorithms
    - User feedback integration and analysis
    - Domain-specific adaptation
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.logger = logger.bind(service="prompt_optimizer")
        
        # Initialize component services
        self.prompt_generator = PromptGenerator(db_session)
        self.performance_evaluator = PerformanceEvaluator(db_session)
        self.ab_testing_engine = ABTestingEngine(db_session)
        self.evolutionary_optimizer = EvolutionaryOptimizer(db_session)
        self.feedback_analyzer = FeedbackAnalyzer(db_session)
        self.context_adapter = ContextAdapter(db_session)
        
        # Configuration
        self.max_concurrent_optimizations = 5
        self.default_convergence_threshold = 0.01
        self.min_improvement_threshold = 0.05
        self.experiment_timeout_hours = 24
        
    async def generate_prompts(
        self,
        task_description: str,
        domain: Optional[str] = None,
        performance_goals: List[str] = None,
        baseline_examples: List[Dict[str, Any]] = None,
        constraints: Dict[str, Any] = None,
        num_candidates: int = 5
    ) -> Dict[str, Any]:
        """
        Generate optimized prompt candidates for a given task.
        
        Args:
            task_description: Description of the task the prompt should perform
            domain: Target domain for the prompt (e.g., 'medical', 'legal')
            performance_goals: List of optimization goals ('accuracy', 'token_efficiency', etc.)
            baseline_examples: Example inputs/outputs for few-shot learning
            constraints: Generation constraints (length, style, etc.)
            num_candidates: Number of prompt candidates to generate
            
        Returns:
            Dict containing prompt candidates and experiment metadata
        """
        try:
            self.logger.info(
                "Starting prompt generation",
                task_description=task_description,
                domain=domain,
                performance_goals=performance_goals,
                num_candidates=num_candidates
            )
            
            # Create base prompt template
            base_template = PromptTemplate(
                name=f"Generated Template - {task_description[:50]}",
                task_type="auto_generated", 
                domain=domain,
                template_content=task_description,
                template_variables={},
                status=PromptStatus.DRAFT,
                created_by="prompt_optimizer",
                description=f"Auto-generated template for: {task_description}"
            )
            
            self.db.add(base_template)
            await self.db.commit()
            await self.db.refresh(base_template)
            
            # Create optimization experiment
            experiment = OptimizationExperiment(
                experiment_name=f"Auto Generation - {datetime.utcnow().isoformat()}",
                base_prompt_id=base_template.id,
                optimization_method=OptimizationMethod.META_PROMPTING,
                target_metrics={goal: 0.8 for goal in (performance_goals or ['accuracy'])},
                experiment_config={
                    'generation_config': {
                        'num_candidates': num_candidates,
                        'constraints': constraints or {},
                        'baseline_examples': baseline_examples or []
                    }
                },
                status=ExperimentStatus.RUNNING,
                created_by="prompt_optimizer",
                started_at=datetime.utcnow()
            )
            
            self.db.add(experiment)
            await self.db.commit()
            await self.db.refresh(experiment)
            
            # Generate prompt candidates
            candidates = await self.prompt_generator.generate_candidates(
                base_prompt=base_template.template_content,
                task_description=task_description,
                domain=domain,
                performance_goals=performance_goals or ['accuracy'],
                baseline_examples=baseline_examples or [],
                constraints=constraints or {},
                num_candidates=num_candidates
            )
            
            # Create prompt variants
            prompt_variants = []
            for i, candidate in enumerate(candidates):
                variant = PromptVariant(
                    experiment_id=experiment.id,
                    parent_prompt_id=base_template.id,
                    variant_content=candidate['content'],
                    generation_method=candidate.get('method', 'meta_prompting'),
                    generation_reasoning=candidate.get('reasoning', ''),
                    confidence_score=candidate.get('confidence', 0.5),
                    iteration=0,
                    generation_time_seconds=candidate.get('generation_time', 0),
                    token_count=candidate.get('token_count', 0),
                    complexity_score=candidate.get('complexity_score', 0),
                    readability_score=candidate.get('readability_score', 0),
                    parameters=candidate.get('parameters', {}),
                    ancestry=[]
                )
                
                self.db.add(variant)
                prompt_variants.append(variant)
            
            await self.db.commit()
            
            # Update experiment status
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            experiment.progress_percentage = 100.0
            await self.db.commit()
            
            self.logger.info(
                "Prompt generation completed",
                experiment_id=str(experiment.id),
                num_candidates=len(prompt_variants)
            )
            
            return {
                'prompt_candidates': [
                    {
                        'id': str(variant.id),
                        'content': variant.variant_content,
                        'generation_method': variant.generation_method,
                        'generation_reasoning': variant.generation_reasoning,
                        'confidence_score': variant.confidence_score,
                        'domain_adaptations': [domain] if domain else []
                    }
                    for variant in prompt_variants
                ],
                'experiment_id': str(experiment.id),
                'generation_metadata': {
                    'base_template_id': str(base_template.id),
                    'generation_time': datetime.utcnow().isoformat(),
                    'total_candidates': len(prompt_variants),
                    'method': 'meta_prompting'
                }
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to generate prompts",
                error=str(e),
                task_description=task_description
            )
            
            # Update experiment status to failed if it exists
            if 'experiment' in locals():
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = str(e)
                experiment.completed_at = datetime.utcnow()
                await self.db.commit()
            
            raise
    
    async def evaluate_prompt(
        self,
        prompt_variant_id: uuid.UUID,
        test_cases: List[Dict[str, Any]] = None,
        evaluation_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt variant's performance.
        
        Args:
            prompt_variant_id: ID of the prompt variant to evaluate
            test_cases: Test cases to use for evaluation
            evaluation_metrics: Metrics to compute
            
        Returns:
            Dict containing performance scores and detailed metrics
        """
        try:
            # Get prompt variant
            result = await self.db.execute(
                select(PromptVariant).where(PromptVariant.id == prompt_variant_id)
            )
            variant = result.scalar_one_or_none()
            
            if not variant:
                raise ValueError(f"Prompt variant {prompt_variant_id} not found")
            
            # Use default test cases if none provided
            if not test_cases:
                test_case_query = select(PromptTestCase).where(
                    and_(
                        PromptTestCase.is_active == True,
                        PromptTestCase.domain == variant.parent_prompt.domain
                    )
                ).limit(10)
                
                test_case_result = await self.db.execute(test_case_query)
                db_test_cases = test_case_result.scalars().all()
                test_cases = [
                    {
                        'id': str(tc.id),
                        'input_data': tc.input_data,
                        'expected_output': tc.expected_output,
                        'evaluation_criteria': tc.evaluation_criteria
                    }
                    for tc in db_test_cases
                ]
            
            # Default metrics
            if not evaluation_metrics:
                evaluation_metrics = ['accuracy', 'relevance', 'coherence']
            
            self.logger.info(
                "Starting prompt evaluation",
                prompt_variant_id=str(prompt_variant_id),
                num_test_cases=len(test_cases),
                metrics=evaluation_metrics
            )
            
            # Evaluate performance
            evaluation_results = await self.performance_evaluator.evaluate_prompt(
                prompt_content=variant.variant_content,
                test_cases=test_cases,
                metrics=evaluation_metrics
            )
            
            # Store evaluation results
            evaluations = []
            for metric_name, metric_value in evaluation_results['detailed_metrics'].items():
                evaluation = PromptEvaluation(
                    prompt_variant_id=variant.id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    evaluation_context=evaluation_results.get('context', {}),
                    evaluation_method="automated",
                    evaluation_time_seconds=evaluation_results.get('evaluation_time', 0),
                    token_usage=evaluation_results.get('token_usage', {}),
                    evaluated_by="prompt_optimizer"
                )
                
                self.db.add(evaluation)
                evaluations.append(evaluation)
            
            await self.db.commit()
            
            self.logger.info(
                "Prompt evaluation completed",
                prompt_variant_id=str(prompt_variant_id),
                performance_score=evaluation_results['performance_score']
            )
            
            return {
                'performance_score': evaluation_results['performance_score'],
                'detailed_metrics': evaluation_results['detailed_metrics'],
                'evaluation_id': str(evaluations[0].id) if evaluations else None,
                'test_cases_used': len(test_cases),
                'evaluation_time': evaluation_results.get('evaluation_time', 0)
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to evaluate prompt",
                prompt_variant_id=str(prompt_variant_id),
                error=str(e)
            )
            raise
    
    async def run_ab_test(
        self,
        experiment_id: uuid.UUID,
        prompt_a_id: uuid.UUID,
        prompt_b_id: uuid.UUID,
        sample_size: int = 100,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Run A/B test between two prompt variants.
        
        Args:
            experiment_id: Parent experiment ID
            prompt_a_id: First prompt variant
            prompt_b_id: Second prompt variant
            sample_size: Number of samples for the test
            significance_level: Statistical significance threshold
            
        Returns:
            Dict containing A/B test results
        """
        try:
            self.logger.info(
                "Starting A/B test",
                experiment_id=str(experiment_id),
                prompt_a_id=str(prompt_a_id),
                prompt_b_id=str(prompt_b_id),
                sample_size=sample_size
            )
            
            # Validate prompts exist
            prompts_query = select(PromptVariant).where(
                PromptVariant.id.in_([prompt_a_id, prompt_b_id])
            )
            result = await self.db.execute(prompts_query)
            prompts = result.scalars().all()
            
            if len(prompts) != 2:
                raise ValueError("Both prompt variants must exist")
            
            # Run A/B test
            test_results = await self.ab_testing_engine.run_test(
                prompt_a_id=prompt_a_id,
                prompt_b_id=prompt_b_id,
                sample_size=sample_size,
                significance_level=significance_level
            )
            
            # Store A/B test results
            ab_test = ABTestResult(
                experiment_id=experiment_id,
                test_name=f"AB Test {datetime.utcnow().isoformat()}",
                prompt_a_id=prompt_a_id,
                prompt_b_id=prompt_b_id,
                sample_size=sample_size,
                significance_level=significance_level,
                p_value=test_results.get('p_value'),
                effect_size=test_results.get('effect_size'),
                confidence_interval_lower=test_results.get('confidence_interval', [None, None])[0],
                confidence_interval_upper=test_results.get('confidence_interval', [None, None])[1],
                winner_variant_id=test_results.get('winner_variant_id'),
                test_power=test_results.get('test_power'),
                mean_a=test_results.get('mean_a'),
                mean_b=test_results.get('mean_b'),
                std_a=test_results.get('std_a'),
                std_b=test_results.get('std_b'),
                test_statistic=test_results.get('test_statistic'),
                degrees_of_freedom=test_results.get('degrees_of_freedom'),
                statistical_notes=test_results.get('notes'),
                test_completed_at=datetime.utcnow()
            )
            
            self.db.add(ab_test)
            await self.db.commit()
            await self.db.refresh(ab_test)
            
            self.logger.info(
                "A/B test completed",
                ab_test_id=str(ab_test.id),
                p_value=test_results.get('p_value'),
                winner=str(test_results.get('winner_variant_id'))
            )
            
            return {
                'test_id': str(ab_test.id),
                'sample_size': sample_size,
                'significance_level': significance_level,
                'p_value': test_results.get('p_value'),
                'effect_size': test_results.get('effect_size'),
                'winner_variant_id': str(test_results.get('winner_variant_id')) if test_results.get('winner_variant_id') else None,
                'statistical_significance': test_results.get('p_value', 1.0) < significance_level,
                'confidence_interval': test_results.get('confidence_interval'),
                'test_power': test_results.get('test_power'),
                'detailed_results': test_results
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to run A/B test",
                experiment_id=str(experiment_id),
                error=str(e)
            )
            raise
    
    async def create_experiment(
        self,
        experiment_name: str,
        base_prompt_id: uuid.UUID,
        optimization_method: OptimizationMethod,
        target_metrics: Dict[str, float],
        experiment_config: Dict[str, Any] = None,
        max_iterations: int = 50,
        created_by: str = None
    ) -> uuid.UUID:
        """
        Create a new optimization experiment.
        
        Args:
            experiment_name: Name for the experiment
            base_prompt_id: Base prompt template to optimize
            optimization_method: Method to use for optimization
            target_metrics: Target performance metrics
            experiment_config: Additional configuration
            max_iterations: Maximum optimization iterations
            created_by: Creator identifier
            
        Returns:
            UUID of the created experiment
        """
        try:
            experiment = OptimizationExperiment(
                experiment_name=experiment_name,
                base_prompt_id=base_prompt_id,
                optimization_method=optimization_method,
                target_metrics=target_metrics,
                experiment_config=experiment_config or {},
                max_iterations=max_iterations,
                convergence_threshold=self.default_convergence_threshold,
                created_by=created_by or "prompt_optimizer",
                description=f"Optimization experiment using {optimization_method.value}"
            )
            
            self.db.add(experiment)
            await self.db.commit()
            await self.db.refresh(experiment)
            
            self.logger.info(
                "Created optimization experiment",
                experiment_id=str(experiment.id),
                method=optimization_method.value,
                target_metrics=target_metrics
            )
            
            return experiment.id
            
        except Exception as e:
            self.logger.error(
                "Failed to create experiment",
                experiment_name=experiment_name,
                error=str(e)
            )
            raise
    
    async def run_experiment(
        self,
        experiment_id: uuid.UUID,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run an optimization experiment.
        
        Args:
            experiment_id: Experiment to run
            max_iterations: Override max iterations
            
        Returns:
            Dict containing experiment results
        """
        try:
            # Get experiment
            result = await self.db.execute(
                select(OptimizationExperiment)
                .options(selectinload(OptimizationExperiment.base_prompt))
                .where(OptimizationExperiment.id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            if experiment.status != ExperimentStatus.PENDING:
                raise ValueError(f"Experiment {experiment_id} is not in pending status")
            
            self.logger.info(
                "Starting optimization experiment",
                experiment_id=str(experiment_id),
                method=experiment.optimization_method.value
            )
            
            # Update experiment status
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.utcnow()
            await self.db.commit()
            
            # Run optimization based on method
            if experiment.optimization_method == OptimizationMethod.EVOLUTIONARY:
                results = await self._run_evolutionary_optimization(experiment, max_iterations)
            elif experiment.optimization_method == OptimizationMethod.META_PROMPTING:
                results = await self._run_meta_prompting_optimization(experiment, max_iterations)
            elif experiment.optimization_method == OptimizationMethod.FEW_SHOT:
                results = await self._run_few_shot_optimization(experiment, max_iterations)
            elif experiment.optimization_method == OptimizationMethod.GRADIENT_BASED:
                results = await self._run_gradient_based_optimization(experiment, max_iterations)
            else:
                raise ValueError(f"Unsupported optimization method: {experiment.optimization_method}")
            
            # Update experiment with results
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            experiment.progress_percentage = 100.0
            experiment.best_score = results.get('best_score')
            experiment.baseline_score = results.get('baseline_score')
            
            if experiment.best_score and experiment.baseline_score:
                experiment.improvement_percentage = (
                    (experiment.best_score - experiment.baseline_score) / experiment.baseline_score * 100
                )
            
            await self.db.commit()
            
            self.logger.info(
                "Optimization experiment completed",
                experiment_id=str(experiment_id),
                best_score=experiment.best_score,
                improvement_percentage=experiment.improvement_percentage
            )
            
            return {
                'experiment_id': str(experiment_id),
                'status': experiment.status.value,
                'best_score': experiment.best_score,
                'baseline_score': experiment.baseline_score,
                'improvement_percentage': experiment.improvement_percentage,
                'iterations_completed': results.get('iterations_completed', 0),
                'convergence_achieved': results.get('convergence_achieved', False),
                'optimization_time_seconds': results.get('optimization_time_seconds', 0),
                'best_variant_id': str(results.get('best_variant_id')) if results.get('best_variant_id') else None
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to run experiment",
                experiment_id=str(experiment_id),
                error=str(e)
            )
            
            # Update experiment status to failed
            if 'experiment' in locals():
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = str(e)
                experiment.completed_at = datetime.utcnow()
                await self.db.commit()
            
            raise
    
    async def record_feedback(
        self,
        prompt_variant_id: uuid.UUID,
        user_id: Optional[str],
        rating: int,
        feedback_text: Optional[str] = None,
        feedback_categories: List[str] = None,
        context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Record user feedback for a prompt variant.
        
        Args:
            prompt_variant_id: Variant that received feedback
            user_id: User providing feedback
            rating: Rating from 1-5
            feedback_text: Optional textual feedback
            feedback_categories: Categorized feedback
            context_data: Additional context
            
        Returns:
            Dict containing feedback processing results
        """
        try:
            # Validate rating
            if not 1 <= rating <= 5:
                raise ValueError("Rating must be between 1 and 5")
            
            # Analyze feedback sentiment and quality scores
            analysis_results = await self.feedback_analyzer.analyze_feedback(
                rating=rating,
                feedback_text=feedback_text,
                feedback_categories=feedback_categories or [],
                context_data=context_data or {}
            )
            
            # Calculate feedback weight based on user history and context
            feedback_weight = await self._calculate_feedback_weight(
                user_id=user_id,
                context_data=context_data or {}
            )
            
            # Create feedback record
            feedback = PromptFeedback(
                prompt_variant_id=prompt_variant_id,
                user_id=user_id,
                rating=rating,
                feedback_text=feedback_text,
                feedback_categories=feedback_categories or [],
                context_data=context_data or {},
                response_quality_score=analysis_results.get('quality_score'),
                relevance_score=analysis_results.get('relevance_score'),
                clarity_score=analysis_results.get('clarity_score'),
                usefulness_score=analysis_results.get('usefulness_score'),
                sentiment_score=analysis_results.get('sentiment_score'),
                feedback_weight=feedback_weight
            )
            
            self.db.add(feedback)
            await self.db.commit()
            await self.db.refresh(feedback)
            
            self.logger.info(
                "Recorded prompt feedback",
                feedback_id=str(feedback.id),
                prompt_variant_id=str(prompt_variant_id),
                rating=rating,
                feedback_weight=feedback_weight
            )
            
            return {
                'status': 'recorded',
                'feedback_id': str(feedback.id),
                'influence_weight': feedback_weight,
                'quality_scores': {
                    'response_quality': analysis_results.get('quality_score'),
                    'relevance': analysis_results.get('relevance_score'),
                    'clarity': analysis_results.get('clarity_score'),
                    'usefulness': analysis_results.get('usefulness_score'),
                    'sentiment': analysis_results.get('sentiment_score')
                }
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to record feedback",
                prompt_variant_id=str(prompt_variant_id),
                error=str(e)
            )
            raise
    
    async def get_experiment_status(self, experiment_id: uuid.UUID) -> Dict[str, Any]:
        """Get detailed status of an optimization experiment."""
        try:
            result = await self.db.execute(
                select(OptimizationExperiment)
                .options(
                    selectinload(OptimizationExperiment.variants),
                    selectinload(OptimizationExperiment.ab_tests)
                )
                .where(OptimizationExperiment.id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Get latest metrics
            metrics_query = select(OptimizationMetric).where(
                OptimizationMetric.experiment_id == experiment_id
            ).order_by(OptimizationMetric.recorded_at.desc()).limit(10)
            
            metrics_result = await self.db.execute(metrics_query)
            recent_metrics = metrics_result.scalars().all()
            
            return {
                'experiment_id': str(experiment.id),
                'name': experiment.experiment_name,
                'status': experiment.status.value,
                'method': experiment.optimization_method.value,
                'progress_percentage': experiment.progress_percentage,
                'current_iteration': experiment.current_iteration,
                'max_iterations': experiment.max_iterations,
                'best_score': experiment.best_score,
                'baseline_score': experiment.baseline_score,
                'improvement_percentage': experiment.improvement_percentage,
                'variants_generated': len(experiment.variants),
                'ab_tests_completed': len(experiment.ab_tests),
                'started_at': experiment.started_at.isoformat() if experiment.started_at else None,
                'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None,
                'recent_metrics': [
                    {
                        'name': metric.metric_name,
                        'value': metric.metric_value,
                        'recorded_at': metric.recorded_at.isoformat()
                    }
                    for metric in recent_metrics
                ],
                'target_metrics': experiment.target_metrics,
                'convergence_threshold': experiment.convergence_threshold,
                'error_message': experiment.error_message
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to get experiment status",
                experiment_id=str(experiment_id),
                error=str(e)
            )
            raise
    
    # Private helper methods
    
    async def _run_evolutionary_optimization(
        self, 
        experiment: OptimizationExperiment, 
        max_iterations: Optional[int]
    ) -> Dict[str, Any]:
        """Run evolutionary optimization."""
        return await self.evolutionary_optimizer.optimize(
            experiment=experiment,
            max_iterations=max_iterations or experiment.max_iterations
        )
    
    async def _run_meta_prompting_optimization(
        self, 
        experiment: OptimizationExperiment, 
        max_iterations: Optional[int]
    ) -> Dict[str, Any]:
        """Run meta-prompting optimization."""
        # Implementation for meta-prompting optimization
        # This would use the prompt generator to iteratively improve prompts
        pass
    
    async def _run_few_shot_optimization(
        self, 
        experiment: OptimizationExperiment, 
        max_iterations: Optional[int]
    ) -> Dict[str, Any]:
        """Run few-shot learning optimization."""
        # Implementation for few-shot optimization
        # This would use successful examples to improve prompts
        pass
    
    async def _run_gradient_based_optimization(
        self, 
        experiment: OptimizationExperiment, 
        max_iterations: Optional[int]
    ) -> Dict[str, Any]:
        """Run gradient-based optimization."""
        # Implementation for gradient-based optimization
        # This would use gradient information to optimize prompts
        pass
    
    async def _calculate_feedback_weight(
        self, 
        user_id: Optional[str], 
        context_data: Dict[str, Any]
    ) -> float:
        """Calculate weight for user feedback based on history and context."""
        base_weight = 1.0
        
        if not user_id:
            return base_weight * 0.5  # Anonymous feedback gets lower weight
        
        # Get user's feedback history
        feedback_query = select(PromptFeedback).where(
            PromptFeedback.user_id == user_id
        ).limit(100)
        
        result = await self.db.execute(feedback_query)
        user_feedback = result.scalars().all()
        
        if not user_feedback:
            return base_weight  # New user gets standard weight
        
        # Calculate consistency score
        ratings = [f.rating for f in user_feedback]
        rating_variance = sum((r - sum(ratings)/len(ratings))**2 for r in ratings) / len(ratings)
        consistency_factor = max(0.5, 1.0 - (rating_variance / 4.0))  # Lower variance = higher weight
        
        # Factor in feedback quality
        quality_scores = [f.response_quality_score for f in user_feedback if f.response_quality_score]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        quality_factor = max(0.5, avg_quality)
        
        # Domain expertise factor (if available in context)
        expertise_factor = context_data.get('user_expertise_level', 1.0)
        
        return min(2.0, base_weight * consistency_factor * quality_factor * expertise_factor)