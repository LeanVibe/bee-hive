"""
A/B Testing Engine for statistical comparison of prompt variants.

Provides rigorous statistical testing to determine which prompts perform better
with proper significance testing, effect size calculation, and power analysis.
"""

import asyncio
import math
import random
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.prompt_optimization import PromptVariant, PromptTestCase
from .performance_evaluator import PerformanceEvaluator

logger = structlog.get_logger()


class TestType(str, Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WELCH_T_TEST = "welch_t_test"
    BOOTSTRAP = "bootstrap"


@dataclass
class ABTestConfig:
    """Configuration for A/B tests."""
    test_type: TestType = TestType.WELCH_T_TEST
    significance_level: float = 0.05
    minimum_effect_size: float = 0.1
    power_threshold: float = 0.8
    bootstrap_iterations: int = 1000
    equal_variance_assumption: bool = False
    one_tailed: bool = False


@dataclass
class ABTestResult:
    """Results from A/B test analysis."""
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    sample_size_a: int
    sample_size_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    degrees_of_freedom: Optional[int]
    winner_variant_id: Optional[uuid.UUID]
    is_significant: bool
    practical_significance: bool
    notes: str


class ABTestingEngine:
    """
    Statistical A/B testing engine for comparing prompt performance.
    
    Provides:
    - Multiple statistical test types (t-test, Mann-Whitney, Welch's t-test, bootstrap)
    - Effect size calculation (Cohen's d)
    - Statistical power analysis
    - Confidence interval estimation
    - Sample size recommendations
    - Practical significance assessment
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.logger = logger.bind(component="ab_testing_engine")
        self.performance_evaluator = PerformanceEvaluator(db_session)
        
        # Default configuration
        self.default_config = ABTestConfig()
        
        # Minimum sample sizes for different effect sizes
        self.min_sample_sizes = {
            0.1: 1550,  # Small effect
            0.3: 175,   # Medium effect
            0.5: 64,    # Large effect
            0.8: 26     # Very large effect
        }
    
    async def run_test(
        self,
        prompt_a_id: uuid.UUID,
        prompt_b_id: uuid.UUID,
        sample_size: int,
        significance_level: float = 0.05,
        test_config: Optional[ABTestConfig] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive A/B test between two prompt variants.
        
        Args:
            prompt_a_id: First prompt variant to test
            prompt_b_id: Second prompt variant to test
            sample_size: Number of samples for each variant
            significance_level: Statistical significance threshold
            test_config: Additional test configuration
            test_cases: Custom test cases (if None, uses database test cases)
            
        Returns:
            Dict containing comprehensive test results
        """
        try:
            config = test_config or self.default_config
            config.significance_level = significance_level
            
            self.logger.info(
                "Starting A/B test",
                prompt_a_id=str(prompt_a_id),
                prompt_b_id=str(prompt_b_id),
                sample_size=sample_size,
                test_type=config.test_type.value
            )
            
            start_time = time.time()
            
            # Get prompt variants
            prompts = await self._get_prompt_variants([prompt_a_id, prompt_b_id])
            if len(prompts) != 2:
                raise ValueError("Both prompt variants must exist")
            
            prompt_a = prompts[0] if prompts[0].id == prompt_a_id else prompts[1]
            prompt_b = prompts[1] if prompts[1].id == prompt_b_id else prompts[0]
            
            # Get or generate test cases
            if not test_cases:
                test_cases = await self._get_test_cases(sample_size)
            
            if len(test_cases) < sample_size:
                self.logger.warning(
                    "Insufficient test cases, using available cases",
                    requested=sample_size,
                    available=len(test_cases)
                )
                sample_size = len(test_cases)
            
            # Validate minimum sample size
            min_recommended = self._get_minimum_sample_size(config.minimum_effect_size)
            if sample_size < min_recommended:
                self.logger.warning(
                    "Sample size below recommendation for reliable results",
                    provided=sample_size,
                    recommended=min_recommended
                )
            
            # Randomly assign test cases to variants
            test_assignments = await self._assign_test_cases(test_cases, sample_size)
            
            # Evaluate both variants
            performance_a = await self._evaluate_variant_performance(
                prompt_a.variant_content,
                test_assignments['variant_a']
            )
            
            performance_b = await self._evaluate_variant_performance(
                prompt_b.variant_content, 
                test_assignments['variant_b']
            )
            
            # Perform statistical analysis
            test_result = await self._perform_statistical_test(
                scores_a=performance_a['scores'],
                scores_b=performance_b['scores'],
                config=config,
                prompt_a_id=prompt_a_id,
                prompt_b_id=prompt_b_id
            )
            
            # Calculate additional metrics
            test_duration = time.time() - start_time
            
            result = {
                'test_id': str(uuid.uuid4()),
                'prompt_a_id': str(prompt_a_id),
                'prompt_b_id': str(prompt_b_id),
                'sample_size': sample_size,
                'significance_level': significance_level,
                'test_type': config.test_type.value,
                
                # Statistical results
                'p_value': test_result.p_value,
                'test_statistic': test_result.test_statistic,
                'effect_size': test_result.effect_size,
                'confidence_interval': test_result.confidence_interval,
                'degrees_of_freedom': test_result.degrees_of_freedom,
                
                # Performance metrics
                'mean_a': test_result.mean_a,
                'mean_b': test_result.mean_b,
                'std_a': test_result.std_a,
                'std_b': test_result.std_b,
                
                # Significance assessment
                'is_statistically_significant': test_result.is_significant,
                'is_practically_significant': test_result.practical_significance,
                'winner_variant_id': test_result.winner_variant_id,
                
                # Power analysis
                'test_power': test_result.statistical_power,
                'power_adequate': test_result.statistical_power >= config.power_threshold,
                
                # Additional context
                'test_duration_seconds': test_duration,
                'evaluation_details': {
                    'variant_a_performance': performance_a['summary'],
                    'variant_b_performance': performance_b['summary']
                },
                'statistical_notes': test_result.notes,
                'recommendations': await self._generate_recommendations(test_result, config)
            }
            
            self.logger.info(
                "A/B test completed",
                test_duration=test_duration,
                p_value=test_result.p_value,
                effect_size=test_result.effect_size,
                is_significant=test_result.is_significant
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to run A/B test",
                prompt_a_id=str(prompt_a_id),
                prompt_b_id=str(prompt_b_id),
                error=str(e)
            )
            raise
    
    async def calculate_required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        significance_level: float = 0.05,
        one_tailed: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate required sample size for detecting a given effect size.
        
        Args:
            effect_size: Minimum effect size to detect (Cohen's d)
            power: Desired statistical power
            significance_level: Type I error rate
            one_tailed: Whether to use one-tailed test
            
        Returns:
            Dict containing sample size recommendations
        """
        try:
            # Calculate required sample size using power analysis
            alpha = significance_level
            beta = 1 - power
            
            # Z-scores for significance and power
            if one_tailed:
                z_alpha = self._get_z_score(1 - alpha)
            else:
                z_alpha = self._get_z_score(1 - alpha/2)
            
            z_beta = self._get_z_score(power)
            
            # Sample size calculation (per group)
            n_per_group = math.ceil(
                2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
            )
            
            # Adjust for finite population if needed
            total_sample_size = n_per_group * 2
            
            # Calculate actual power with this sample size
            actual_power = await self._calculate_statistical_power(
                n_per_group, effect_size, significance_level, one_tailed
            )
            
            # Generate recommendations for different effect sizes
            effect_size_scenarios = [0.1, 0.2, 0.3, 0.5, 0.8]
            scenarios = {}
            
            for es in effect_size_scenarios:
                scenario_n = math.ceil(2 * ((z_alpha + z_beta) ** 2) / (es ** 2))
                scenarios[f"effect_size_{es}"] = {
                    'sample_size_per_group': scenario_n,
                    'total_sample_size': scenario_n * 2,
                    'effect_size': es,
                    'description': self._get_effect_size_description(es)
                }
            
            return {
                'recommended_sample_size_per_group': n_per_group,
                'total_sample_size': total_sample_size,
                'target_effect_size': effect_size,
                'target_power': power,
                'significance_level': significance_level,
                'actual_power': actual_power,
                'one_tailed': one_tailed,
                'effect_size_scenarios': scenarios,
                'minimum_detectable_effect': await self._calculate_minimum_detectable_effect(
                    n_per_group, power, significance_level
                ),
                'cost_benefit_analysis': {
                    'low_sample_high_risk': n_per_group // 2,
                    'recommended': n_per_group,
                    'high_confidence': int(n_per_group * 1.5)
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to calculate sample size", error=str(e))
            raise
    
    async def run_sequential_test(
        self,
        prompt_a_id: uuid.UUID,
        prompt_b_id: uuid.UUID,
        max_sample_size: int = 1000,
        min_sample_size: int = 50,
        check_interval: int = 10,
        early_stopping_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Run sequential A/B test with early stopping when significance is reached.
        
        Args:
            prompt_a_id: First prompt variant
            prompt_b_id: Second prompt variant
            max_sample_size: Maximum samples before stopping
            min_sample_size: Minimum samples before checking significance
            check_interval: How often to check for significance
            early_stopping_threshold: P-value threshold for early stopping
            
        Returns:
            Dict containing sequential test results
        """
        try:
            self.logger.info(
                "Starting sequential A/B test",
                prompt_a_id=str(prompt_a_id),
                prompt_b_id=str(prompt_b_id),
                max_sample_size=max_sample_size
            )
            
            # Get test cases
            all_test_cases = await self._get_test_cases(max_sample_size)
            
            # Initialize tracking
            current_sample_size = 0
            test_history = []
            early_stop = False
            final_result = None
            
            # Get prompts
            prompts = await self._get_prompt_variants([prompt_a_id, prompt_b_id])
            prompt_a = prompts[0] if prompts[0].id == prompt_a_id else prompts[1]
            prompt_b = prompts[1] if prompts[1].id == prompt_b_id else prompts[0]
            
            while current_sample_size < max_sample_size:
                # Determine next batch size
                next_batch_size = min(check_interval, max_sample_size - current_sample_size)
                next_sample_size = current_sample_size + next_batch_size
                
                # Get test cases for this batch
                batch_cases = all_test_cases[current_sample_size:next_sample_size]
                
                # Run test with current sample size
                test_result = await self.run_test(
                    prompt_a_id=prompt_a_id,
                    prompt_b_id=prompt_b_id,
                    sample_size=next_sample_size,
                    test_cases=all_test_cases[:next_sample_size]
                )
                
                test_history.append({
                    'sample_size': next_sample_size,
                    'p_value': test_result['p_value'],
                    'effect_size': test_result['effect_size'],
                    'power': test_result['test_power'],
                    'significant': test_result['is_statistically_significant']
                })
                
                # Check for early stopping
                if (next_sample_size >= min_sample_size and 
                    test_result['p_value'] <= early_stopping_threshold and
                    test_result['test_power'] >= 0.8):
                    
                    early_stop = True
                    final_result = test_result
                    self.logger.info(
                        "Early stopping triggered",
                        sample_size=next_sample_size,
                        p_value=test_result['p_value']
                    )
                    break
                
                current_sample_size = next_sample_size
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # If no early stopping, use final result
            if not early_stop:
                final_result = test_history[-1] if test_history else None
            
            return {
                'sequential_test_completed': True,
                'early_stopping_triggered': early_stop,
                'final_sample_size': current_sample_size,
                'max_sample_size': max_sample_size,
                'test_history': test_history,
                'final_result': final_result,
                'efficiency_gain': (max_sample_size - current_sample_size) / max_sample_size if early_stop else 0,
                'recommendations': {
                    'continue_testing': not early_stop and current_sample_size < max_sample_size,
                    'conclusive_result': early_stop or current_sample_size >= max_sample_size,
                    'next_steps': 'Deploy winner' if early_stop else 'Collect more data'
                }
            }
            
        except Exception as e:
            self.logger.error("Failed sequential A/B test", error=str(e))
            raise
    
    # Private helper methods
    
    async def _get_prompt_variants(
        self, 
        variant_ids: List[uuid.UUID]
    ) -> List[PromptVariant]:
        """Get prompt variants from database."""
        result = await self.db.execute(
            select(PromptVariant).where(PromptVariant.id.in_(variant_ids))
        )
        return result.scalars().all()
    
    async def _get_test_cases(self, sample_size: int) -> List[Dict[str, Any]]:
        """Get test cases for evaluation."""
        result = await self.db.execute(
            select(PromptTestCase)
            .where(PromptTestCase.is_active == True)
            .limit(sample_size * 2)  # Get more than needed for random selection
        )
        test_cases = result.scalars().all()
        
        # Convert to dict format
        formatted_cases = []
        for tc in test_cases:
            formatted_cases.append({
                'id': str(tc.id),
                'input_data': tc.input_data,
                'expected_output': tc.expected_output,
                'evaluation_criteria': tc.evaluation_criteria
            })
        
        # If we don't have enough test cases, generate synthetic ones
        while len(formatted_cases) < sample_size:
            synthetic_case = await self._generate_synthetic_test_case(len(formatted_cases))
            formatted_cases.append(synthetic_case)
        
        return formatted_cases
    
    async def _generate_synthetic_test_case(self, index: int) -> Dict[str, Any]:
        """Generate a synthetic test case when insufficient real cases exist."""
        synthetic_inputs = [
            "Explain the concept of machine learning",
            "What are the benefits of renewable energy?",
            "How does photosynthesis work?",
            "Describe the water cycle",
            "What is artificial intelligence?",
            "Explain supply and demand in economics",
            "How do vaccines work?",
            "What causes climate change?",
            "Describe the structure of an atom",
            "How does the internet work?"
        ]
        
        input_text = synthetic_inputs[index % len(synthetic_inputs)]
        
        return {
            'id': f"synthetic_{index}",
            'input_data': {'text': input_text},
            'expected_output': None,  # Will be evaluated without expected output
            'evaluation_criteria': {'type': 'general_knowledge'}
        }
    
    async def _assign_test_cases(
        self,
        test_cases: List[Dict[str, Any]],
        sample_size: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Randomly assign test cases to variants."""
        # Shuffle test cases for random assignment
        shuffled = test_cases.copy()
        random.shuffle(shuffled)
        
        # Split into two equal groups
        mid_point = sample_size
        
        return {
            'variant_a': shuffled[:mid_point],
            'variant_b': shuffled[mid_point:mid_point * 2]
        }
    
    async def _evaluate_variant_performance(
        self,
        prompt_content: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a prompt variant's performance."""
        evaluation_result = await self.performance_evaluator.evaluate_prompt(
            prompt_content=prompt_content,
            test_cases=test_cases,
            metrics=['accuracy', 'relevance', 'coherence', 'clarity']
        )
        
        # Extract individual scores for statistical analysis
        scores = []
        for test_result in evaluation_result['test_case_results']:
            # Use overall score or weighted average of metrics
            case_scores = test_result.get('metric_scores', {})
            if case_scores:
                case_score = sum(case_scores.values()) / len(case_scores)
            else:
                case_score = 0.5  # Default neutral score
            scores.append(case_score)
        
        return {
            'scores': scores,
            'summary': {
                'mean_score': evaluation_result['performance_score'],
                'num_cases': len(test_cases),
                'detailed_metrics': evaluation_result['detailed_metrics']
            }
        }
    
    async def _perform_statistical_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        config: ABTestConfig,
        prompt_a_id: uuid.UUID,
        prompt_b_id: uuid.UUID
    ) -> ABTestResult:
        """Perform the actual statistical test."""
        
        # Calculate basic statistics
        n_a, n_b = len(scores_a), len(scores_b)
        mean_a, mean_b = sum(scores_a) / n_a, sum(scores_b) / n_b
        
        # Calculate standard deviations
        std_a = math.sqrt(sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1)) if n_a > 1 else 0
        std_b = math.sqrt(sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1)) if n_b > 1 else 0
        
        # Determine winner
        winner_id = prompt_b_id if mean_b > mean_a else prompt_a_id
        
        # Perform the specified statistical test
        if config.test_type == TestType.WELCH_T_TEST:
            result = await self._welch_t_test(scores_a, scores_b, config)
        elif config.test_type == TestType.T_TEST:
            result = await self._students_t_test(scores_a, scores_b, config)
        elif config.test_type == TestType.MANN_WHITNEY:
            result = await self._mann_whitney_test(scores_a, scores_b, config)
        elif config.test_type == TestType.BOOTSTRAP:
            result = await self._bootstrap_test(scores_a, scores_b, config)
        else:
            raise ValueError(f"Unsupported test type: {config.test_type}")
        
        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        effect_size = abs(mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval for the difference
        diff_mean = mean_b - mean_a
        se_diff = math.sqrt(std_a**2 / n_a + std_b**2 / n_b)
        
        # Use t-distribution critical value
        df = min(n_a - 1, n_b - 1)  # Conservative degrees of freedom
        t_critical = self._get_t_critical(1 - config.significance_level/2, df)
        margin_error = t_critical * se_diff
        
        confidence_interval = (diff_mean - margin_error, diff_mean + margin_error)
        
        # Calculate statistical power
        power = await self._calculate_statistical_power(
            min(n_a, n_b), effect_size, config.significance_level, config.one_tailed
        )
        
        # Assess practical significance
        practical_significance = effect_size >= config.minimum_effect_size
        
        return ABTestResult(
            test_statistic=result['test_statistic'],
            p_value=result['p_value'],
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            statistical_power=power,
            sample_size_a=n_a,
            sample_size_b=n_b,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            degrees_of_freedom=result.get('degrees_of_freedom'),
            winner_variant_id=winner_id,
            is_significant=result['p_value'] < config.significance_level,
            practical_significance=practical_significance,
            notes=result.get('notes', '')
        )
    
    async def _welch_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        config: ABTestConfig
    ) -> Dict[str, Any]:
        """Perform Welch's t-test (unequal variances)."""
        n_a, n_b = len(scores_a), len(scores_b)
        mean_a, mean_b = sum(scores_a) / n_a, sum(scores_b) / n_b
        
        var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1) if n_a > 1 else 0
        var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1) if n_b > 1 else 0
        
        # Welch's t-statistic
        se_diff = math.sqrt(var_a / n_a + var_b / n_b)
        t_stat = (mean_b - mean_a) / se_diff if se_diff > 0 else 0
        
        # Welch-Satterthwaite equation for degrees of freedom
        if var_a > 0 and var_b > 0:
            df = ((var_a / n_a + var_b / n_b) ** 2) / (
                (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
            )
        else:
            df = n_a + n_b - 2
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        if config.one_tailed:
            p_value = p_value / 2
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'notes': 'Welch\'s t-test assumes unequal variances'
        }
    
    async def _students_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        config: ABTestConfig
    ) -> Dict[str, Any]:
        """Perform Student's t-test (equal variances assumed)."""
        n_a, n_b = len(scores_a), len(scores_b)
        mean_a, mean_b = sum(scores_a) / n_a, sum(scores_b) / n_b
        
        # Pooled standard deviation
        var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1) if n_a > 1 else 0
        var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1) if n_b > 1 else 0
        
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        se_diff = math.sqrt(pooled_var * (1/n_a + 1/n_b))
        
        t_stat = (mean_b - mean_a) / se_diff if se_diff > 0 else 0
        df = n_a + n_b - 2
        
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        if config.one_tailed:
            p_value = p_value / 2
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'notes': 'Student\'s t-test assumes equal variances'
        }
    
    async def _mann_whitney_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        config: ABTestConfig
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric)."""
        # Combine and rank all scores
        combined = [(score, 'a') for score in scores_a] + [(score, 'b') for score in scores_b]
        combined.sort(key=lambda x: x[0])
        
        # Assign ranks
        ranks = {}
        for i, (score, group) in enumerate(combined):
            if score not in ranks:
                ranks[score] = []
            ranks[score].append(i + 1)
        
        # Handle ties by using average rank
        for score in ranks:
            avg_rank = sum(ranks[score]) / len(ranks[score])
            ranks[score] = avg_rank
        
        # Calculate rank sums
        rank_sum_a = sum(ranks[score] for score in scores_a)
        rank_sum_b = sum(ranks[score] for score in scores_b)
        
        n_a, n_b = len(scores_a), len(scores_b)
        
        # Mann-Whitney U statistics
        u_a = rank_sum_a - n_a * (n_a + 1) / 2
        u_b = rank_sum_b - n_b * (n_b + 1) / 2
        
        u_stat = min(u_a, u_b)
        
        # Normal approximation for p-value
        mean_u = n_a * n_b / 2
        std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
        
        z_stat = (u_stat - mean_u) / std_u if std_u > 0 else 0
        p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))
        
        if config.one_tailed:
            p_value = p_value / 2
        
        return {
            'test_statistic': u_stat,
            'p_value': p_value,
            'notes': 'Mann-Whitney U test is non-parametric and doesn\'t assume normal distribution'
        }
    
    async def _bootstrap_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        config: ABTestConfig
    ) -> Dict[str, Any]:
        """Perform bootstrap hypothesis test."""
        n_iterations = config.bootstrap_iterations
        observed_diff = sum(scores_b) / len(scores_b) - sum(scores_a) / len(scores_a)
        
        # Combine samples for null hypothesis (no difference)
        combined = scores_a + scores_b
        n_a, n_b = len(scores_a), len(scores_b)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        for _ in range(n_iterations):
            # Resample without replacement
            resampled = random.sample(combined, len(combined))
            resample_a = resampled[:n_a]
            resample_b = resampled[n_a:]
            
            diff = sum(resample_b) / len(resample_b) - sum(resample_a) / len(resample_a)
            bootstrap_diffs.append(diff)
        
        # Calculate p-value
        if config.one_tailed:
            p_value = sum(1 for diff in bootstrap_diffs if diff >= abs(observed_diff)) / n_iterations
        else:
            p_value = sum(1 for diff in bootstrap_diffs if abs(diff) >= abs(observed_diff)) / n_iterations
        
        return {
            'test_statistic': observed_diff,
            'p_value': p_value,
            'notes': f'Bootstrap test with {n_iterations} iterations'
        }
    
    def _get_minimum_sample_size(self, effect_size: float) -> int:
        """Get minimum recommended sample size for effect size."""
        for es, min_size in sorted(self.min_sample_sizes.items()):
            if effect_size <= es:
                return min_size
        return self.min_sample_sizes[0.1]  # Default to smallest effect size
    
    async def _calculate_statistical_power(
        self,
        sample_size: int,
        effect_size: float,
        alpha: float,
        one_tailed: bool = False
    ) -> float:
        """Calculate statistical power for given parameters."""
        # Simplified power calculation
        z_alpha = self._get_z_score(1 - alpha/2 if not one_tailed else 1 - alpha)
        z_beta = effect_size * math.sqrt(sample_size / 2) - z_alpha
        
        power = self._normal_cdf(z_beta)
        return max(0.0, min(1.0, power))
    
    async def _calculate_minimum_detectable_effect(
        self,
        sample_size: int,
        power: float,
        alpha: float
    ) -> float:
        """Calculate minimum detectable effect size."""
        z_alpha = self._get_z_score(1 - alpha/2)
        z_beta = self._get_z_score(power)
        
        mde = (z_alpha + z_beta) * math.sqrt(2 / sample_size)
        return mde
    
    async def _generate_recommendations(
        self,
        test_result: ABTestResult,
        config: ABTestConfig
    ) -> Dict[str, str]:
        """Generate actionable recommendations based on test results."""
        recommendations = {}
        
        if test_result.is_significant and test_result.practical_significance:
            recommendations['action'] = 'Deploy the winning variant'
            recommendations['confidence'] = 'High'
        elif test_result.is_significant and not test_result.practical_significance:
            recommendations['action'] = 'Consider business context - statistical significance but small practical effect'
            recommendations['confidence'] = 'Medium'
        elif not test_result.is_significant and test_result.statistical_power >= 0.8:
            recommendations['action'] = 'No significant difference detected with adequate power'
            recommendations['confidence'] = 'High'
        else:
            recommendations['action'] = 'Collect more data - insufficient power to detect differences'
            recommendations['confidence'] = 'Low'
        
        # Power-specific recommendations
        if test_result.statistical_power < 0.8:
            recommendations['power_note'] = f'Low statistical power ({test_result.statistical_power:.2f}). Consider increasing sample size.'
        
        # Effect size interpretation
        if test_result.effect_size < 0.2:
            recommendations['effect_size'] = 'Small effect size - may not be practically significant'
        elif test_result.effect_size < 0.5:
            recommendations['effect_size'] = 'Medium effect size - potentially meaningful difference'
        else:
            recommendations['effect_size'] = 'Large effect size - likely meaningful difference'
        
        return recommendations
    
    def _get_effect_size_description(self, effect_size: float) -> str:
        """Get qualitative description of effect size."""
        if effect_size < 0.2:
            return "Small effect"
        elif effect_size < 0.5:
            return "Medium effect"
        elif effect_size < 0.8:
            return "Large effect"
        else:
            return "Very large effect"
    
    # Statistical utility functions
    
    def _get_z_score(self, percentile: float) -> float:
        """Get z-score for given percentile (approximation)."""
        # Simplified inverse normal CDF approximation
        if percentile <= 0.5:
            return -self._get_z_score(1 - percentile)
        
        t = math.sqrt(-2 * math.log(1 - percentile))
        return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    
    def _get_t_critical(self, alpha: float, df: int) -> float:
        """Get critical t-value (approximation)."""
        # Simplified t-distribution approximation
        z = self._get_z_score(alpha)
        return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal distribution."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate CDF for t-distribution."""
        # Simple approximation using normal distribution for large df
        if df > 30:
            return self._normal_cdf(t)
        else:
            # Very rough approximation for small df
            correction = 1 + (t**2) / (4 * df)
            return self._normal_cdf(t / correction)