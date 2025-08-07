"""
Gradient-based optimization for prompt optimization using approximated gradients.

Implements gradient-based optimization techniques for prompts by approximating
gradients through finite differences and directional derivatives in prompt space.
"""

import asyncio
import math
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import numpy as np
from collections import defaultdict

from ..models.prompt_optimization import OptimizationExperiment, PromptVariant, PromptEvaluation
from .performance_evaluator import PerformanceEvaluator

logger = structlog.get_logger()


class OptimizationDirection(str, Enum):
    """Gradient optimization directions."""
    GRADIENT_ASCENT = "gradient_ascent"
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    RMSPROP = "rmsprop"


class ParameterType(str, Enum):
    """Types of prompt parameters to optimize."""
    INSTRUCTION_CLARITY = "instruction_clarity"
    SPECIFICITY_LEVEL = "specificity_level"
    CONTEXT_RICHNESS = "context_richness"
    EXAMPLE_QUALITY = "example_quality"
    TASK_DECOMPOSITION = "task_decomposition"
    OUTPUT_FORMAT = "output_format"
    CONSTRAINT_STRENGTH = "constraint_strength"
    REASONING_GUIDANCE = "reasoning_guidance"


@dataclass
class PromptParameter:
    """Represents a continuous parameter in prompt space."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    gradient: float = 0.0
    momentum: float = 0.0
    variance: float = 0.0


@dataclass
class GradientStep:
    """Information about a gradient optimization step."""
    iteration: int
    parameters: Dict[str, float]
    performance_score: float
    gradient_norm: float
    step_size: float
    improvement: float
    parameter_updates: Dict[str, float]


class GradientOptimizer:
    """
    Gradient-based optimization system for prompt optimization.
    
    Features:
    - Finite difference gradient approximation
    - Multiple optimization algorithms (SGD, Adam, RMSprop)
    - Adaptive step sizes and momentum
    - Multi-dimensional parameter optimization
    - Convergence detection and early stopping
    - Constraint handling for prompt validity
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.logger = logger.bind(component="gradient_optimizer")
        
        # Initialize performance evaluator
        self.performance_evaluator = PerformanceEvaluator(db_session)
        
        # Optimization parameters
        self.learning_rate = 0.01
        self.momentum_coefficient = 0.9
        self.beta1 = 0.9  # Adam beta1
        self.beta2 = 0.999  # Adam beta2
        self.epsilon = 1e-8  # Numerical stability
        self.gradient_clip_threshold = 1.0
        self.max_iterations = 100
        self.convergence_threshold = 1e-4
        self.finite_difference_h = 0.01
        
        # Parameter definitions
        self.parameter_space = {
            ParameterType.INSTRUCTION_CLARITY: PromptParameter(
                name="instruction_clarity",
                current_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step_size=0.05
            ),
            ParameterType.SPECIFICITY_LEVEL: PromptParameter(
                name="specificity_level", 
                current_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step_size=0.05
            ),
            ParameterType.CONTEXT_RICHNESS: PromptParameter(
                name="context_richness",
                current_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step_size=0.05
            ),
            ParameterType.EXAMPLE_QUALITY: PromptParameter(
                name="example_quality",
                current_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step_size=0.05
            )
        }
        
        # Optimization state
        self.optimization_history = []
        self.adam_m = defaultdict(float)  # First moment estimates
        self.adam_v = defaultdict(float)  # Second moment estimates
        self.rmsprop_cache = defaultdict(float)
        self.iteration_count = 0
    
    async def optimize(
        self,
        experiment: OptimizationExperiment,
        max_iterations: Optional[int] = None,
        optimization_method: OptimizationDirection = OptimizationDirection.ADAM,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run gradient-based optimization on a prompt.
        
        Args:
            experiment: The optimization experiment
            max_iterations: Maximum iterations to run
            optimization_method: Gradient optimization algorithm
            test_cases: Test cases for evaluation
            
        Returns:
            Dict containing optimization results
        """
        try:
            start_time = time.time()
            
            # Override defaults if specified
            if max_iterations:
                self.max_iterations = max_iterations
            
            self.logger.info(
                "Starting gradient-based optimization",
                experiment_id=str(experiment.id),
                method=optimization_method.value,
                max_iterations=self.max_iterations
            )
            
            # Initialize optimization state
            self._reset_optimization_state()
            
            # Initialize parameters from experiment config
            await self._initialize_parameters(experiment)
            
            # Get or generate test cases
            if not test_cases:
                test_cases = await self._generate_default_test_cases()
            
            # Optimization loop
            best_score = float('-inf')
            best_parameters = None
            best_variant_id = None
            baseline_score = 0.0
            
            for iteration in range(self.max_iterations):
                iteration_start = time.time()
                
                # Generate prompt with current parameters
                current_prompt = await self._generate_prompt_from_parameters(
                    experiment.base_prompt.template_content,
                    self.parameter_space
                )
                
                # Evaluate current prompt
                current_score = await self._evaluate_prompt_performance(
                    current_prompt, test_cases
                )
                
                # Record baseline score from first iteration
                if iteration == 0:
                    baseline_score = current_score
                
                # Update best score if improved
                if current_score > best_score:
                    best_score = current_score
                    best_parameters = {name: param.current_value for name, param in self.parameter_space.items()}
                    
                    # Store best variant
                    variant = await self._store_variant(
                        experiment, current_prompt, current_score, iteration, best_parameters
                    )
                    best_variant_id = variant.id
                
                # Calculate gradients
                gradients = await self._compute_gradients(
                    experiment.base_prompt.template_content,
                    test_cases,
                    current_score
                )
                
                # Update parameters using selected optimization method
                if optimization_method == OptimizationDirection.GRADIENT_ASCENT:
                    parameter_updates = await self._gradient_ascent_step(gradients)
                elif optimization_method == OptimizationDirection.ADAM:
                    parameter_updates = await self._adam_step(gradients, iteration + 1)
                elif optimization_method == OptimizationDirection.RMSPROP:
                    parameter_updates = await self._rmsprop_step(gradients)
                else:
                    parameter_updates = await self._gradient_ascent_step(gradients)
                
                # Apply parameter updates with constraints
                await self._apply_parameter_updates(parameter_updates)
                
                # Calculate gradient norm and improvement
                gradient_norm = math.sqrt(sum(g**2 for g in gradients.values()))
                improvement = current_score - (self.optimization_history[-1].performance_score 
                                             if self.optimization_history else baseline_score)
                
                # Record optimization step
                step = GradientStep(
                    iteration=iteration,
                    parameters={name: param.current_value for name, param in self.parameter_space.items()},
                    performance_score=current_score,
                    gradient_norm=gradient_norm,
                    step_size=self.learning_rate,
                    improvement=improvement,
                    parameter_updates=parameter_updates
                )
                self.optimization_history.append(step)
                
                # Update experiment progress
                progress = (iteration + 1) / self.max_iterations * 100
                await self._update_experiment_progress(experiment.id, progress, iteration)
                
                # Check convergence
                if await self._check_convergence():
                    self.logger.info(
                        "Convergence achieved",
                        iteration=iteration,
                        gradient_norm=gradient_norm
                    )
                    break
                
                iteration_time = time.time() - iteration_start
                self.logger.info(
                    "Gradient optimization iteration completed",
                    iteration=iteration,
                    score=current_score,
                    gradient_norm=gradient_norm,
                    improvement=improvement,
                    iteration_time=iteration_time
                )
            
            optimization_time = time.time() - start_time
            
            result = {
                'best_score': best_score,
                'baseline_score': baseline_score,
                'iterations_completed': len(self.optimization_history),
                'convergence_achieved': len(self.optimization_history) < self.max_iterations,
                'optimization_time_seconds': optimization_time,
                'best_variant_id': best_variant_id,
                'optimization_history': self.optimization_history,
                'final_parameters': best_parameters,
                'improvement_percentage': (
                    ((best_score - baseline_score) / baseline_score * 100)
                    if baseline_score > 0 else 0.0
                ),
                'gradient_statistics': await self._get_gradient_statistics(),
                'convergence_metrics': await self._get_convergence_metrics()
            }
            
            self.logger.info(
                "Gradient optimization completed",
                experiment_id=str(experiment.id),
                best_score=best_score,
                improvement=result['improvement_percentage'],
                iterations=result['iterations_completed']
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Gradient optimization failed",
                experiment_id=str(experiment.id),
                error=str(e)
            )
            raise
    
    async def _reset_optimization_state(self):
        """Reset optimization state for new run."""
        self.optimization_history = []
        self.adam_m.clear()
        self.adam_v.clear()
        self.rmsprop_cache.clear()
        self.iteration_count = 0
        
        # Reset parameter gradients and momentum
        for param in self.parameter_space.values():
            param.gradient = 0.0
            param.momentum = 0.0
            param.variance = 0.0
    
    async def _initialize_parameters(self, experiment: OptimizationExperiment):
        """Initialize parameters from experiment configuration."""
        config = experiment.experiment_config or {}
        gradient_config = config.get('gradient_optimization', {})
        
        # Override default parameters if specified
        self.learning_rate = gradient_config.get('learning_rate', self.learning_rate)
        self.momentum_coefficient = gradient_config.get('momentum', self.momentum_coefficient)
        self.beta1 = gradient_config.get('beta1', self.beta1)
        self.beta2 = gradient_config.get('beta2', self.beta2)
        self.convergence_threshold = gradient_config.get('convergence_threshold', self.convergence_threshold)
        
        # Initialize parameter values
        initial_params = gradient_config.get('initial_parameters', {})
        for param_name, param in self.parameter_space.items():
            if param.name in initial_params:
                param.current_value = max(param.min_value, 
                                        min(param.max_value, initial_params[param.name]))
    
    async def _compute_gradients(
        self,
        base_prompt: str,
        test_cases: List[Dict[str, Any]],
        current_score: float
    ) -> Dict[str, float]:
        """Compute gradients using finite differences."""
        gradients = {}
        
        for param_name, param in self.parameter_space.items():
            # Forward difference
            param.current_value += self.finite_difference_h
            param.current_value = max(param.min_value, min(param.max_value, param.current_value))
            
            forward_prompt = await self._generate_prompt_from_parameters(base_prompt, self.parameter_space)
            forward_score = await self._evaluate_prompt_performance(forward_prompt, test_cases)
            
            # Backward difference
            param.current_value -= 2 * self.finite_difference_h
            param.current_value = max(param.min_value, min(param.max_value, param.current_value))
            
            backward_prompt = await self._generate_prompt_from_parameters(base_prompt, self.parameter_space)
            backward_score = await self._evaluate_prompt_performance(backward_prompt, test_cases)
            
            # Restore original value
            param.current_value += self.finite_difference_h
            
            # Central difference gradient
            gradient = (forward_score - backward_score) / (2 * self.finite_difference_h)
            
            # Clip gradient to prevent explosion
            gradient = max(-self.gradient_clip_threshold, 
                         min(self.gradient_clip_threshold, gradient))
            
            param.gradient = gradient
            gradients[param.name] = gradient
        
        return gradients
    
    async def _gradient_ascent_step(self, gradients: Dict[str, float]) -> Dict[str, float]:
        """Perform gradient ascent step with momentum."""
        parameter_updates = {}
        
        for param_name, param in self.parameter_space.items():
            gradient = gradients.get(param.name, 0.0)
            
            # Update momentum
            param.momentum = self.momentum_coefficient * param.momentum + self.learning_rate * gradient
            
            # Parameter update
            update = param.momentum
            parameter_updates[param.name] = update
        
        return parameter_updates
    
    async def _adam_step(self, gradients: Dict[str, float], t: int) -> Dict[str, float]:
        """Perform Adam optimization step."""
        parameter_updates = {}
        
        for param_name, param in self.parameter_space.items():
            gradient = gradients.get(param.name, 0.0)
            
            # Update biased first moment estimate
            self.adam_m[param.name] = self.beta1 * self.adam_m[param.name] + (1 - self.beta1) * gradient
            
            # Update biased second raw moment estimate  
            self.adam_v[param.name] = self.beta2 * self.adam_v[param.name] + (1 - self.beta2) * gradient**2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.adam_m[param.name] / (1 - self.beta1**t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.adam_v[param.name] / (1 - self.beta2**t)
            
            # Parameter update
            update = self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
            parameter_updates[param.name] = update
        
        return parameter_updates
    
    async def _rmsprop_step(self, gradients: Dict[str, float]) -> Dict[str, float]:
        """Perform RMSprop optimization step."""
        parameter_updates = {}
        decay_rate = 0.9
        
        for param_name, param in self.parameter_space.items():
            gradient = gradients.get(param.name, 0.0)
            
            # Update cache
            self.rmsprop_cache[param.name] = (decay_rate * self.rmsprop_cache[param.name] + 
                                            (1 - decay_rate) * gradient**2)
            
            # Parameter update
            update = self.learning_rate * gradient / (math.sqrt(self.rmsprop_cache[param.name]) + self.epsilon)
            parameter_updates[param.name] = update
        
        return parameter_updates
    
    async def _apply_parameter_updates(self, parameter_updates: Dict[str, float]):
        """Apply parameter updates with constraint handling."""
        for param_name, param in self.parameter_space.items():
            update = parameter_updates.get(param.name, 0.0)
            
            # Apply update
            new_value = param.current_value + update
            
            # Enforce constraints
            param.current_value = max(param.min_value, min(param.max_value, new_value))
    
    async def _generate_prompt_from_parameters(
        self,
        base_prompt: str,
        parameters: Dict[ParameterType, PromptParameter]
    ) -> str:
        """Generate prompt content based on parameter values."""
        # Get parameter values
        clarity = parameters[ParameterType.INSTRUCTION_CLARITY].current_value
        specificity = parameters[ParameterType.SPECIFICITY_LEVEL].current_value
        context_richness = parameters[ParameterType.CONTEXT_RICHNESS].current_value
        example_quality = parameters[ParameterType.EXAMPLE_QUALITY].current_value
        
        # Build prompt modifications based on parameter values
        modifications = []
        
        # Instruction clarity modifications
        if clarity > 0.7:
            modifications.append("Be extremely clear and precise in your instructions.")
        elif clarity > 0.4:
            modifications.append("Provide clear and detailed instructions.")
        
        # Specificity level modifications
        if specificity > 0.7:
            modifications.append("Include specific details, examples, and concrete requirements.")
        elif specificity > 0.4:
            modifications.append("Provide relevant details and context where appropriate.")
        
        # Context richness modifications
        if context_richness > 0.7:
            modifications.append("Consider the full context, background information, and related factors.")
        elif context_richness > 0.4:
            modifications.append("Take into account the relevant context and background.")
        
        # Example quality modifications
        if example_quality > 0.7:
            modifications.append("Use high-quality, relevant examples to illustrate your points.")
        elif example_quality > 0.4:
            modifications.append("Include examples where helpful.")
        
        # Combine base prompt with modifications
        if modifications:
            modified_prompt = base_prompt + "\n\nAdditional guidance:\n" + "\n".join(f"- {mod}" for mod in modifications)
        else:
            modified_prompt = base_prompt
        
        return modified_prompt
    
    async def _evaluate_prompt_performance(
        self,
        prompt: str,
        test_cases: List[Dict[str, Any]]
    ) -> float:
        """Evaluate prompt performance using the performance evaluator."""
        try:
            result = await self.performance_evaluator.evaluate_prompt(
                prompt_content=prompt,
                test_cases=test_cases,
                metrics=['accuracy', 'relevance', 'coherence', 'clarity']
            )
            return result['performance_score']
        except Exception as e:
            self.logger.error("Failed to evaluate prompt performance", error=str(e))
            return 0.0
    
    async def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.optimization_history) < 5:
            return False
        
        # Check gradient norm convergence
        recent_gradients = [step.gradient_norm for step in self.optimization_history[-5:]]
        avg_gradient_norm = sum(recent_gradients) / len(recent_gradients)
        
        if avg_gradient_norm < self.convergence_threshold:
            return True
        
        # Check score improvement convergence
        recent_scores = [step.performance_score for step in self.optimization_history[-5:]]
        score_variance = sum((s - sum(recent_scores)/len(recent_scores))**2 for s in recent_scores) / len(recent_scores)
        
        return score_variance < self.convergence_threshold
    
    async def _store_variant(
        self,
        experiment: OptimizationExperiment,
        prompt_content: str,
        performance_score: float,
        iteration: int,
        parameters: Dict[str, float]
    ) -> PromptVariant:
        """Store a prompt variant from optimization."""
        variant = PromptVariant(
            experiment_id=experiment.id,
            parent_prompt_id=experiment.base_prompt_id,
            variant_content=prompt_content,
            generation_method="gradient_based_optimization",
            generation_reasoning=f"Generated using gradient-based optimization at iteration {iteration}",
            confidence_score=performance_score,
            iteration=iteration,
            parameters=parameters,
            ancestry=[]
        )
        
        self.db.add(variant)
        await self.db.commit()
        await self.db.refresh(variant)
        
        return variant
    
    async def _update_experiment_progress(
        self,
        experiment_id: uuid.UUID,
        progress: float,
        iteration: int
    ):
        """Update experiment progress in database."""
        try:
            stmt = update(OptimizationExperiment).where(
                OptimizationExperiment.id == experiment_id
            ).values(
                progress_percentage=progress,
                current_iteration=iteration
            )
            await self.db.execute(stmt)
            await self.db.commit()
        except Exception as e:
            self.logger.error("Failed to update experiment progress", error=str(e))
    
    async def _generate_default_test_cases(self) -> List[Dict[str, Any]]:
        """Generate default test cases for optimization."""
        return [
            {
                'id': 'gradient_test_1',
                'input_data': {'text': 'Test query for optimization'},
                'expected_output': 'High-quality response',
                'evaluation_criteria': {'accuracy': True, 'clarity': True}
            },
            {
                'id': 'gradient_test_2',
                'input_data': {'text': 'Complex multi-part question'},
                'expected_output': 'Comprehensive structured response',
                'evaluation_criteria': {'completeness': True, 'coherence': True}
            },
            {
                'id': 'gradient_test_3',
                'input_data': {'text': 'Domain-specific technical question'},
                'expected_output': 'Expert-level accurate response',
                'evaluation_criteria': {'accuracy': True, 'technical_depth': True}
            }
        ]
    
    async def _get_gradient_statistics(self) -> Dict[str, Any]:
        """Get statistics about gradient optimization."""
        if not self.optimization_history:
            return {}
        
        gradient_norms = [step.gradient_norm for step in self.optimization_history]
        improvements = [step.improvement for step in self.optimization_history[1:]]
        
        return {
            'average_gradient_norm': sum(gradient_norms) / len(gradient_norms),
            'max_gradient_norm': max(gradient_norms),
            'min_gradient_norm': min(gradient_norms),
            'average_improvement': sum(improvements) / len(improvements) if improvements else 0.0,
            'total_parameter_updates': sum(
                len(step.parameter_updates) for step in self.optimization_history
            ),
            'optimization_method': 'gradient_based'
        }
    
    async def _get_convergence_metrics(self) -> Dict[str, Any]:
        """Get convergence analysis metrics."""
        if len(self.optimization_history) < 2:
            return {}
        
        scores = [step.performance_score for step in self.optimization_history]
        score_improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        return {
            'convergence_achieved': len(self.optimization_history) < self.max_iterations,
            'score_trajectory': scores,
            'score_improvements': score_improvements,
            'final_gradient_norm': self.optimization_history[-1].gradient_norm,
            'convergence_rate': sum(1 for imp in score_improvements if imp > 0) / len(score_improvements),
            'optimization_efficiency': (scores[-1] - scores[0]) / len(self.optimization_history) if scores else 0.0
        }