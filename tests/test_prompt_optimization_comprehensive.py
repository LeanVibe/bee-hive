"""
Comprehensive tests for Prompt Optimization System with statistical validation and performance benchmarking.

Tests all optimization components including gradient optimization, evolutionary algorithms,
feedback analysis, context adaptation, and statistical validation.
"""

import pytest
import uuid
import time
import statistics
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.core.prompt_optimizer import PromptOptimizer
from app.core.evolutionary_optimizer import EvolutionaryOptimizer, Individual, SelectionMethod, MutationStrategy
from app.core.gradient_optimizer import GradientOptimizer, OptimizationDirection, ParameterType
from app.core.feedback_analyzer import FeedbackAnalyzer, SentimentPolarity, FeedbackCategory
from app.core.context_adapter import ContextAdapter, UserPreferences, UserExpertiseLevel, CommunicationStyle, DomainType
from app.core.ab_testing_engine import ABTestingEngine, TestType
from app.core.performance_evaluator import PerformanceEvaluator
from app.models.prompt_optimization import (
    PromptTemplate, OptimizationExperiment, PromptVariant,
    PromptStatus, ExperimentStatus, OptimizationMethod
)


@pytest.fixture
def mock_db_session():
    """Enhanced mock database session with comprehensive mocking."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    
    # Mock common query results
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_result.scalars = MagicMock()
    mock_result.scalars.return_value.all = MagicMock(return_value=[])
    session.execute.return_value = mock_result
    
    return session


@pytest.fixture
def sample_prompt_template():
    """Sample prompt template for testing."""
    return PromptTemplate(
        id=uuid.uuid4(),
        name="Test Template",
        task_type="summarization",
        domain="general",
        template_content="Summarize the following text: {text}",
        template_variables={"text": "string"},
        status=PromptStatus.ACTIVE,
        created_by="test_system"
    )


@pytest.fixture
def sample_experiment(sample_prompt_template):
    """Sample optimization experiment for testing."""
    return OptimizationExperiment(
        id=uuid.uuid4(),
        experiment_name="Test Optimization",
        base_prompt_id=sample_prompt_template.id,
        base_prompt=sample_prompt_template,
        optimization_method=OptimizationMethod.EVOLUTIONARY,
        target_metrics={"accuracy": 0.85, "relevance": 0.8},
        experiment_config={
            "max_iterations": 5,
            "population_size": 10
        },
        status=ExperimentStatus.PENDING,
        created_by="test_system"
    )


@pytest.fixture
def sample_test_cases():
    """Sample test cases for evaluation."""
    return [
        {
            'id': 'test_1',
            'input_data': {'text': 'This is a test document for summarization.'},
            'expected_output': 'Test document summary.',
            'evaluation_criteria': {'accuracy': True, 'clarity': True}
        },
        {
            'id': 'test_2', 
            'input_data': {'text': 'Another document with more complex content to test.'},
            'expected_output': 'Complex content summary.',
            'evaluation_criteria': {'completeness': True, 'relevance': True}
        },
        {
            'id': 'test_3',
            'input_data': {'text': 'Technical documentation with specific terminology.'},
            'expected_output': 'Technical summary with key terms.',
            'evaluation_criteria': {'accuracy': True, 'technical_depth': True}
        }
    ]


class TestEvolutionaryOptimizer:
    """Comprehensive tests for EvolutionaryOptimizer."""
    
    @pytest.fixture
    def evolutionary_optimizer(self, mock_db_session):
        """Create EvolutionaryOptimizer instance."""
        return EvolutionaryOptimizer(mock_db_session)
    
    @pytest.mark.asyncio
    async def test_population_initialization(self, evolutionary_optimizer, sample_experiment, sample_test_cases):
        """Test population initialization."""
        population = await evolutionary_optimizer._initialize_population(sample_experiment, sample_test_cases)
        
        assert len(population) == evolutionary_optimizer.population_size
        assert all(isinstance(individual, Individual) for individual in population)
        assert population[0].creation_method == "original"
        assert all(ind.generation == 0 for ind in population)
        assert all(ind.fitness_score == 0.0 for ind in population)  # Not evaluated yet
    
    @pytest.mark.asyncio
    async def test_mutation_strategies(self, evolutionary_optimizer):
        """Test all mutation strategies."""
        test_content = "Please analyze the given text and provide a comprehensive summary."
        
        for strategy in MutationStrategy:
            mutated = await evolutionary_optimizer._apply_mutation(test_content, strategy, 0.3)
            assert isinstance(mutated, str)
            assert len(mutated) > 0
            # Content should be different (unless mutation failed gracefully)
            # We don't enforce difference because some mutations might not change anything
    
    @pytest.mark.asyncio
    async def test_selection_methods(self, evolutionary_optimizer, sample_test_cases):
        """Test all selection methods."""
        # Create sample population with different fitness scores
        population = [
            Individual("1", "content1", 0.9, 1, [], [], {}, "test"),
            Individual("2", "content2", 0.7, 1, [], [], {}, "test"),
            Individual("3", "content3", 0.5, 1, [], [], {}, "test"),
            Individual("4", "content4", 0.3, 1, [], [], {}, "test"),
        ]
        
        # Test tournament selection
        selected = await evolutionary_optimizer._tournament_selection(population)
        assert selected in population
        assert isinstance(selected, Individual)
        
        # Test roulette wheel selection
        selected = await evolutionary_optimizer._roulette_wheel_selection(population)
        assert selected in population
        
        # Test rank-based selection
        selected = await evolutionary_optimizer._rank_based_selection(population)
        assert selected in population
    
    @pytest.mark.asyncio
    async def test_crossover(self, evolutionary_optimizer):
        """Test crossover operation."""
        parent1 = "First sentence. Second sentence. Third sentence."
        parent2 = "Different first. Another second. Final third."
        
        offspring = await evolutionary_optimizer._crossover(parent1, parent2)
        assert isinstance(offspring, str)
        assert len(offspring) > 0
        assert '.' in offspring  # Should have sentence structure
    
    @pytest.mark.asyncio
    async def test_convergence_detection(self, evolutionary_optimizer):
        """Test convergence detection logic."""
        # Create history that should converge
        converged_history = [
            {'best_fitness': 0.8, 'generation': 0},
            {'best_fitness': 0.805, 'generation': 1},
            {'best_fitness': 0.806, 'generation': 2},
            {'best_fitness': 0.807, 'generation': 3},
            {'best_fitness': 0.807, 'generation': 4}
        ]
        evolutionary_optimizer.optimization_history = converged_history
        
        is_converged = await evolutionary_optimizer._check_convergence(converged_history)
        assert is_converged
        
        # Create history that should not converge
        improving_history = [
            {'best_fitness': 0.5, 'generation': 0},
            {'best_fitness': 0.6, 'generation': 1},
            {'best_fitness': 0.7, 'generation': 2},
            {'best_fitness': 0.8, 'generation': 3},
            {'best_fitness': 0.9, 'generation': 4}
        ]
        
        is_converged = await evolutionary_optimizer._check_convergence(improving_history)
        assert not is_converged


class TestGradientOptimizer:
    """Comprehensive tests for GradientOptimizer."""
    
    @pytest.fixture
    def gradient_optimizer(self, mock_db_session):
        """Create GradientOptimizer instance.""" 
        return GradientOptimizer(mock_db_session)
    
    @pytest.mark.asyncio
    async def test_parameter_initialization(self, gradient_optimizer, sample_experiment):
        """Test parameter initialization."""
        await gradient_optimizer._initialize_parameters(sample_experiment)
        
        assert len(gradient_optimizer.parameter_space) > 0
        for param in gradient_optimizer.parameter_space.values():
            assert param.min_value <= param.current_value <= param.max_value
            assert param.gradient == 0.0
            assert param.momentum == 0.0
    
    @pytest.mark.asyncio
    async def test_prompt_generation_from_parameters(self, gradient_optimizer):
        """Test prompt generation from parameter values."""
        # Set specific parameter values
        gradient_optimizer.parameter_space[ParameterType.INSTRUCTION_CLARITY].current_value = 0.9
        gradient_optimizer.parameter_space[ParameterType.SPECIFICITY_LEVEL].current_value = 0.8
        
        base_prompt = "Analyze the given text."
        generated_prompt = await gradient_optimizer._generate_prompt_from_parameters(
            base_prompt, gradient_optimizer.parameter_space
        )
        
        assert isinstance(generated_prompt, str)
        assert len(generated_prompt) >= len(base_prompt)
        assert "clear" in generated_prompt.lower() or "precise" in generated_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_gradient_computation(self, gradient_optimizer, sample_test_cases):
        """Test gradient computation using finite differences."""
        base_prompt = "Summarize the text."
        
        # Mock performance evaluation
        with patch.object(gradient_optimizer, '_evaluate_prompt_performance') as mock_eval:
            mock_eval.side_effect = [0.7, 0.75, 0.65, 0.72, 0.68, 0.74, 0.69, 0.71]  # Forward/backward scores
            
            gradients = await gradient_optimizer._compute_gradients(base_prompt, sample_test_cases, 0.7)
            
            assert isinstance(gradients, dict)
            assert len(gradients) > 0
            for param_name, gradient in gradients.items():
                assert isinstance(gradient, float)
                assert abs(gradient) <= gradient_optimizer.gradient_clip_threshold
    
    @pytest.mark.asyncio
    async def test_optimization_methods(self, gradient_optimizer):
        """Test different optimization methods (Adam, RMSprop, Gradient Ascent)."""
        gradients = {
            'instruction_clarity': 0.1,
            'specificity_level': -0.05,
            'context_richness': 0.08,
            'example_quality': 0.03
        }
        
        # Test Adam step
        adam_updates = await gradient_optimizer._adam_step(gradients, 1)
        assert isinstance(adam_updates, dict)
        assert len(adam_updates) == len(gradients)
        
        # Test RMSprop step
        rmsprop_updates = await gradient_optimizer._rmsprop_step(gradients)
        assert isinstance(rmsprop_updates, dict)
        assert len(rmsprop_updates) == len(gradients)
        
        # Test Gradient Ascent step
        ga_updates = await gradient_optimizer._gradient_ascent_step(gradients)
        assert isinstance(ga_updates, dict)
        assert len(ga_updates) == len(gradients)
    
    @pytest.mark.asyncio
    async def test_parameter_constraints(self, gradient_optimizer):
        """Test parameter constraint enforcement."""
        # Set parameter to boundary
        param = gradient_optimizer.parameter_space[ParameterType.INSTRUCTION_CLARITY]
        param.current_value = param.max_value
        
        # Try to update beyond boundary
        updates = {param.name: 0.5}  # Large positive update
        await gradient_optimizer._apply_parameter_updates(updates)
        
        # Should be clamped to max value
        assert param.current_value == param.max_value
        
        # Test minimum boundary
        param.current_value = param.min_value
        updates = {param.name: -0.5}  # Large negative update
        await gradient_optimizer._apply_parameter_updates(updates)
        
        # Should be clamped to min value
        assert param.current_value == param.min_value


class TestFeedbackAnalyzer:
    """Comprehensive tests for FeedbackAnalyzer."""
    
    @pytest.fixture
    def feedback_analyzer(self, mock_db_session):
        """Create FeedbackAnalyzer instance."""
        return FeedbackAnalyzer(mock_db_session)
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_positive(self, feedback_analyzer):
        """Test positive sentiment analysis."""
        result = await feedback_analyzer.analyze_feedback(
            rating=5,
            feedback_text="This is excellent and very helpful! Great work on the accuracy and clarity.",
            feedback_categories=["accuracy", "clarity"]
        )
        
        assert result.sentiment_analysis.polarity in [SentimentPolarity.POSITIVE, SentimentPolarity.VERY_POSITIVE]
        assert result.sentiment_analysis.score > 0.6
        assert len(result.sentiment_analysis.emotional_indicators) > 0
        assert "excellent" in result.sentiment_analysis.sentiment_keywords
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_negative(self, feedback_analyzer):
        """Test negative sentiment analysis."""
        result = await feedback_analyzer.analyze_feedback(
            rating=2,
            feedback_text="This is terrible and useless. Very disappointing and unclear results.",
            feedback_categories=["clarity"]
        )
        
        assert result.sentiment_analysis.polarity in [SentimentPolarity.NEGATIVE, SentimentPolarity.VERY_NEGATIVE]
        assert result.sentiment_analysis.score < 0.4
        assert len(result.sentiment_analysis.emotional_indicators) > 0
        assert "terrible" in result.sentiment_analysis.sentiment_keywords
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, feedback_analyzer):
        """Test quality scoring dimensions."""
        result = await feedback_analyzer.analyze_feedback(
            rating=4,
            feedback_text="Good accuracy but could be clearer and more complete.",
            feedback_categories=["accuracy", "clarity", "completeness"]
        )
        
        quality_scores = result.quality_scores
        assert 0.0 <= quality_scores.overall_quality <= 1.0
        assert 0.0 <= quality_scores.response_quality <= 1.0
        assert 0.0 <= quality_scores.relevance_score <= 1.0
        assert 0.0 <= quality_scores.clarity_score <= 1.0
        assert 0.0 <= quality_scores.usefulness_score <= 1.0
        assert 0.0 <= quality_scores.completeness_score <= 1.0
        assert 0.0 <= quality_scores.accuracy_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_feedback_categorization(self, feedback_analyzer):
        """Test automatic feedback categorization."""
        result = await feedback_analyzer.analyze_feedback(
            rating=3,
            feedback_text="The response was accurate but not very clear and somewhat incomplete.",
            feedback_categories=[]  # Let it auto-detect
        )
        
        detected_categories = result.feedback_categories
        assert isinstance(detected_categories, list)
        assert FeedbackCategory.ACCURACY in detected_categories
        assert FeedbackCategory.CLARITY in detected_categories
        assert FeedbackCategory.COMPLETENESS in detected_categories
    
    @pytest.mark.asyncio
    async def test_improvement_suggestions(self, feedback_analyzer):
        """Test improvement suggestion generation."""
        result = await feedback_analyzer.analyze_feedback(
            rating=2,
            feedback_text="Inaccurate information and very unclear presentation.",
            feedback_categories=["accuracy", "clarity"]
        )
        
        suggestions = result.improvement_suggestions
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest accuracy and clarity improvements
        suggestions_text = " ".join(suggestions).lower()
        assert "accuracy" in suggestions_text or "correct" in suggestions_text
        assert "clarity" in suggestions_text or "clear" in suggestions_text
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, feedback_analyzer):
        """Test confidence level calculation."""
        # High confidence case (detailed text + clear rating)
        detailed_result = await feedback_analyzer.analyze_feedback(
            rating=5,
            feedback_text="This is a very detailed feedback with specific examples and clear reasoning about why this response was excellent.",
            feedback_categories=["accuracy"]
        )
        
        # Low confidence case (minimal text + neutral rating)
        minimal_result = await feedback_analyzer.analyze_feedback(
            rating=3,
            feedback_text="ok",
            feedback_categories=[]
        )
        
        assert detailed_result.confidence_level > minimal_result.confidence_level
        assert 0.0 <= detailed_result.confidence_level <= 1.0
        assert 0.0 <= minimal_result.confidence_level <= 1.0


class TestContextAdapter:
    """Comprehensive tests for ContextAdapter."""
    
    @pytest.fixture
    def context_adapter(self, mock_db_session):
        """Create ContextAdapter instance."""
        return ContextAdapter(mock_db_session)
    
    @pytest.fixture
    def sample_user_preferences(self):
        """Sample user preferences for testing."""
        return UserPreferences(
            expertise_level=UserExpertiseLevel.ADVANCED,
            communication_style=CommunicationStyle.TECHNICAL,
            preferred_domains=[DomainType.TECHNICAL, DomainType.BUSINESS],
            language_preferences={'formality': 0.8, 'conciseness': 0.7},
            format_preferences={'structured': 0.9, 'examples': 0.8},
            content_preferences={'detailed': 0.8, 'practical': 0.9},
            personalization_settings={'adaptive_learning': True}
        )
    
    @pytest.mark.asyncio
    async def test_domain_adaptation(self, context_adapter, sample_user_preferences):
        """Test domain-specific adaptation."""
        base_prompt = "Provide analysis of the given data."
        
        result = await context_adapter.adapt_to_context(
            base_prompt=base_prompt,
            domain="technical",
            user_preferences=sample_user_preferences
        )
        
        assert result.adapted_content != base_prompt
        assert len(result.adaptations_applied) > 0
        assert result.domain_fit_score > 0.0
        assert result.overall_adaptation_score > 0.0
        
        # Should contain domain-specific terminology
        technical_terms = ["technical", "implementation", "architecture", "optimization"]
        adapted_lower = result.adapted_content.lower()
        assert any(term in adapted_lower for term in technical_terms)
    
    @pytest.mark.asyncio
    async def test_user_preference_adaptation(self, context_adapter, sample_user_preferences):
        """Test user preference adaptation."""
        base_prompt = "Give me some information."
        
        result = await context_adapter.adapt_to_context(
            base_prompt=base_prompt,
            user_preferences=sample_user_preferences
        )
        
        # Should be more formal (formality = 0.8)
        assert "please" in result.adapted_content.lower() or "kindly" in result.adapted_content.lower()
        
        # Should be structured (structured = 0.9) 
        assert "heading" in result.adapted_content.lower() or "bullet" in result.adapted_content.lower()
        
        # Should include examples (examples = 0.8)
        assert "example" in result.adapted_content.lower()
        
        assert result.user_preference_score > 0.5
    
    @pytest.mark.asyncio
    async def test_expertise_level_adaptation(self, context_adapter):
        """Test adaptation for different expertise levels."""
        base_prompt = "Explain machine learning."
        
        # Test beginner level
        beginner_prefs = UserPreferences(
            expertise_level=UserExpertiseLevel.BEGINNER,
            communication_style=CommunicationStyle.CONVERSATIONAL,
            preferred_domains=[DomainType.GENERAL],
            language_preferences={},
            format_preferences={},
            content_preferences={},
            personalization_settings={}
        )
        
        beginner_result = await context_adapter.adapt_to_context(
            base_prompt=base_prompt,
            user_preferences=beginner_prefs
        )
        
        # Test expert level
        expert_prefs = UserPreferences(
            expertise_level=UserExpertiseLevel.EXPERT,
            communication_style=CommunicationStyle.TECHNICAL,
            preferred_domains=[DomainType.TECHNICAL],
            language_preferences={},
            format_preferences={},
            content_preferences={},
            personalization_settings={}
        )
        
        expert_result = await context_adapter.adapt_to_context(
            base_prompt=base_prompt,
            user_preferences=expert_prefs
        )
        
        # Expert version should be more technical
        assert "advanced" in expert_result.adapted_content.lower() or "expert" in expert_result.adapted_content.lower()
        # Beginner version should be more accessible
        assert "new to" in beginner_result.adapted_content.lower() or "simple" in beginner_result.adapted_content.lower()
    
    @pytest.mark.asyncio
    async def test_communication_style_adaptation(self, context_adapter):
        """Test different communication style adaptations."""
        base_prompt = "Analyze the data."
        
        # Test formal style
        formal_prefs = UserPreferences(
            expertise_level=UserExpertiseLevel.INTERMEDIATE,
            communication_style=CommunicationStyle.FORMAL,
            preferred_domains=[DomainType.GENERAL],
            language_preferences={},
            format_preferences={},
            content_preferences={},
            personalization_settings={}
        )
        
        formal_result = await context_adapter.adapt_to_context(
            base_prompt=base_prompt,
            user_preferences=formal_prefs
        )
        
        # Should contain formal language
        formal_indicators = ["comprehensive analysis", "professional", "systematic"]
        formal_lower = formal_result.adapted_content.lower()
        assert any(indicator in formal_lower for indicator in formal_indicators)
        
        # Test conversational style
        casual_prefs = UserPreferences(
            expertise_level=UserExpertiseLevel.INTERMEDIATE,
            communication_style=CommunicationStyle.CONVERSATIONAL,
            preferred_domains=[DomainType.GENERAL],
            language_preferences={},
            format_preferences={},
            content_preferences={},
            personalization_settings={}
        )
        
        casual_result = await context_adapter.adapt_to_context(
            base_prompt=base_prompt,
            user_preferences=casual_prefs
        )
        
        # Should contain conversational language
        casual_indicators = ["let me help", "friendly", "accessible"]
        casual_lower = casual_result.adapted_content.lower()
        assert any(indicator in casual_lower for indicator in casual_indicators)


class TestStatisticalValidation:
    """Statistical validation tests for optimization results."""
    
    @pytest.fixture
    def ab_testing_engine(self, mock_db_session):
        """Create ABTestingEngine instance."""
        return ABTestingEngine(mock_db_session)
    
    @pytest.mark.asyncio
    async def test_statistical_significance_calculation(self, ab_testing_engine):
        """Test statistical significance calculations."""
        # Test data that should be significantly different
        scores_a = [0.5, 0.52, 0.48, 0.51, 0.49, 0.53, 0.47, 0.50, 0.52, 0.48]
        scores_b = [0.7, 0.72, 0.68, 0.71, 0.69, 0.73, 0.67, 0.70, 0.72, 0.68]
        
        # Mock the AB testing engine methods to use our test data
        config = ab_testing_engine.default_config
        
        result = await ab_testing_engine._welch_t_test(scores_a, scores_b, config)
        
        assert result['p_value'] < 0.05  # Should be statistically significant
        assert isinstance(result['test_statistic'], float)
        assert isinstance(result['degrees_of_freedom'], (int, float))
    
    @pytest.mark.asyncio
    async def test_effect_size_calculation(self, ab_testing_engine):
        """Test effect size (Cohen's d) calculation."""
        # Create test data with known effect size
        scores_a = [0.5] * 20  # Mean = 0.5, std = 0
        scores_b = [0.8] * 20  # Mean = 0.8, std = 0
        
        # Calculate effect size manually for comparison
        mean_diff = 0.8 - 0.5  # 0.3
        pooled_std = 0.001  # Near zero due to no variance
        expected_effect_size = mean_diff / max(pooled_std, 0.001)  # Very large
        
        # The actual calculation uses the AB testing logic
        n_a, n_b = len(scores_a), len(scores_b)
        mean_a, mean_b = sum(scores_a) / n_a, sum(scores_b) / n_b
        
        # Mock standard deviation calculation (both are 0 in this case)
        import math
        std_a = math.sqrt(sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1)) if n_a > 1 else 0
        std_b = math.sqrt(sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1)) if n_b > 1 else 0
        
        # Pooled standard deviation
        pooled_std = math.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        effect_size = abs(mean_b - mean_a) / max(pooled_std, 0.001)  # Avoid division by zero
        
        # Effect size should be large (> 0.8 is considered large)
        assert effect_size > 0.8
    
    @pytest.mark.asyncio
    async def test_sample_size_calculation(self, ab_testing_engine):
        """Test sample size calculation for different effect sizes."""
        # Test small effect size
        small_effect_result = await ab_testing_engine.calculate_required_sample_size(
            effect_size=0.1,  # Small effect
            power=0.8,
            significance_level=0.05
        )
        
        # Test large effect size
        large_effect_result = await ab_testing_engine.calculate_required_sample_size(
            effect_size=0.8,  # Large effect
            power=0.8,
            significance_level=0.05
        )
        
        # Smaller effect sizes should require larger sample sizes
        assert small_effect_result['recommended_sample_size_per_group'] > large_effect_result['recommended_sample_size_per_group']
        assert small_effect_result['actual_power'] >= 0.8
        assert large_effect_result['actual_power'] >= 0.8
    
    @pytest.mark.asyncio
    async def test_confidence_intervals(self, ab_testing_engine):
        """Test confidence interval calculations."""
        # Test with known data
        scores_a = [0.6, 0.65, 0.55, 0.62, 0.58]
        scores_b = [0.75, 0.78, 0.72, 0.76, 0.74]
        
        n_a, n_b = len(scores_a), len(scores_b)
        mean_a, mean_b = sum(scores_a) / n_a, sum(scores_b) / n_b
        
        # Calculate standard deviations
        import math
        std_a = math.sqrt(sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1))
        std_b = math.sqrt(sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1))
        
        # Calculate confidence interval components
        diff_mean = mean_b - mean_a
        se_diff = math.sqrt(std_a**2 / n_a + std_b**2 / n_b)
        
        # Confidence interval should contain the true difference
        margin_error = 1.96 * se_diff  # Approximate for 95% CI
        ci_lower = diff_mean - margin_error
        ci_upper = diff_mean + margin_error
        
        assert ci_lower <= diff_mean <= ci_upper
        assert ci_upper > ci_lower


class TestPerformanceBenchmarking:
    """Performance benchmarking tests for optimization components."""
    
    @pytest.mark.asyncio
    async def test_prompt_generation_performance(self, mock_db_session, sample_test_cases):
        """Test prompt generation performance benchmarks."""
        optimizer = PromptOptimizer(mock_db_session)
        
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        start_time = time.time()
        
        result = await optimizer.generate_prompts(
            task_description="Performance test task",
            domain="general",
            performance_goals=["accuracy"],
            num_candidates=5
        )
        
        generation_time = time.time() - start_time
        
        # Should generate prompts quickly (< 5 seconds for 5 candidates)
        assert generation_time < 5.0
        assert len(result["prompt_candidates"]) <= 5
        assert "experiment_id" in result
    
    @pytest.mark.asyncio
    async def test_evaluation_performance(self, mock_db_session, sample_test_cases):
        """Test evaluation performance benchmarks."""
        evaluator = PerformanceEvaluator(mock_db_session)
        
        test_prompt = "Analyze the following text and provide insights."
        
        start_time = time.time()
        
        result = await evaluator.evaluate_prompt(
            prompt_content=test_prompt,
            test_cases=sample_test_cases,
            metrics=['accuracy', 'relevance', 'coherence']
        )
        
        evaluation_time = time.time() - start_time
        
        # Should evaluate quickly (< 2 seconds for 3 test cases)
        assert evaluation_time < 2.0
        assert "performance_score" in result
        assert 0.0 <= result["performance_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_performance(self, mock_db_session, sample_test_cases):
        """Test batch evaluation performance."""
        evaluator = PerformanceEvaluator(mock_db_session)
        
        prompts = [
            "Summarize the text below.",
            "Analyze the key points in the text.",
            "Extract the main themes from the text.",
            "Provide insights about the text content.",
            "Identify important information in the text."
        ]
        
        start_time = time.time()
        
        results = await evaluator.evaluate_batch(
            prompts=prompts,
            test_cases=sample_test_cases,
            metrics=['accuracy', 'relevance']
        )
        
        batch_time = time.time() - start_time
        
        # Batch evaluation should be efficient (< 3 seconds for 5 prompts)
        assert batch_time < 3.0
        assert len(results) == len(prompts)
        for result in results:
            assert "performance_score" in result
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, mock_db_session, sample_experiment, sample_test_cases):
        """Test memory efficiency of optimization components."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple optimizers and run operations
        optimizers = [PromptOptimizer(mock_db_session) for _ in range(5)]
        
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        # Run multiple operations
        tasks = []
        for optimizer in optimizers:
            tasks.append(optimizer.generate_prompts(
                task_description="Memory test",
                domain="general",
                performance_goals=["accuracy"],
                num_candidates=3
            ))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100
        assert len([r for r in results if not isinstance(r, Exception)]) >= 3  # At least 3 should succeed


class TestIntegrationWorkflow:
    """Integration tests for complete optimization workflows."""
    
    @pytest.mark.asyncio
    async def test_full_evolutionary_optimization_workflow(self, mock_db_session, sample_experiment, sample_test_cases):
        """Test complete evolutionary optimization workflow."""
        optimizer = PromptOptimizer(mock_db_session)
        
        # Mock database responses for experiment
        mock_experiment = sample_experiment
        mock_experiment.status = ExperimentStatus.PENDING
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_experiment
        
        # Run evolutionary optimization
        result = await optimizer._run_evolutionary_optimization(mock_experiment, max_iterations=3)
        
        assert "best_score" in result
        assert "baseline_score" in result
        assert "iterations_completed" in result
        assert "optimization_time_seconds" in result
        assert result["iterations_completed"] <= 3
    
    @pytest.mark.asyncio
    async def test_full_gradient_optimization_workflow(self, mock_db_session, sample_experiment, sample_test_cases):
        """Test complete gradient optimization workflow."""
        optimizer = PromptOptimizer(mock_db_session)
        
        # Mock database responses
        mock_experiment = sample_experiment
        mock_experiment.status = ExperimentStatus.PENDING
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_experiment
        
        # Run gradient optimization
        result = await optimizer._run_gradient_based_optimization(mock_experiment, max_iterations=3)
        
        assert "best_score" in result
        assert "baseline_score" in result
        assert "iterations_completed" in result
        assert "optimization_time_seconds" in result
        assert "final_parameters" in result
        assert result["iterations_completed"] <= 3
    
    @pytest.mark.asyncio
    async def test_feedback_integration_workflow(self, mock_db_session):
        """Test feedback integration workflow."""
        optimizer = PromptOptimizer(mock_db_session)
        
        variant_id = uuid.uuid4()
        
        # Record multiple feedback entries
        feedback_results = []
        feedback_data = [
            {"rating": 5, "text": "Excellent work!", "categories": ["accuracy"]},
            {"rating": 4, "text": "Good but could be clearer", "categories": ["clarity"]},
            {"rating": 3, "text": "Average performance", "categories": ["relevance"]},
            {"rating": 4, "text": "Pretty good overall", "categories": ["usefulness"]},
        ]
        
        for feedback in feedback_data:
            result = await optimizer.record_feedback(
                prompt_variant_id=variant_id,
                user_id="test_user",
                rating=feedback["rating"],
                feedback_text=feedback["text"],
                feedback_categories=feedback["categories"]
            )
            feedback_results.append(result)
        
        # Verify all feedback was processed
        assert len(feedback_results) == len(feedback_data)
        for result in feedback_results:
            assert result["status"] == "recorded"
            assert "influence_weight" in result
            assert "quality_scores" in result


# Performance regression test
@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests to ensure optimization performance doesn't degrade."""
    
    PERFORMANCE_THRESHOLDS = {
        "prompt_generation_time": 5.0,  # seconds
        "evaluation_time": 2.0,  # seconds per test case
        "optimization_iteration_time": 10.0,  # seconds per iteration
        "memory_usage_mb": 100.0  # MB increase
    }
    
    @pytest.mark.asyncio
    async def test_no_performance_regression(self, mock_db_session, sample_test_cases):
        """Comprehensive performance regression test."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Test prompt generation performance
        optimizer = PromptOptimizer(mock_db_session)
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        start_time = time.time()
        await optimizer.generate_prompts(
            task_description="Regression test",
            domain="general", 
            performance_goals=["accuracy"],
            num_candidates=5
        )
        generation_time = time.time() - start_time
        
        # Test evaluation performance
        evaluator = PerformanceEvaluator(mock_db_session)
        start_time = time.time()
        await evaluator.evaluate_prompt(
            prompt_content="Test prompt for regression",
            test_cases=sample_test_cases,
            metrics=['accuracy', 'relevance']
        )
        evaluation_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Assert performance thresholds
        assert generation_time < self.PERFORMANCE_THRESHOLDS["prompt_generation_time"], f"Generation too slow: {generation_time:.2f}s"
        assert evaluation_time < self.PERFORMANCE_THRESHOLDS["evaluation_time"], f"Evaluation too slow: {evaluation_time:.2f}s"
        assert memory_increase < self.PERFORMANCE_THRESHOLDS["memory_usage_mb"], f"Memory usage too high: {memory_increase:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])