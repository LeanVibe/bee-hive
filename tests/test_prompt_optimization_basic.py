"""
Basic tests for Prompt Optimization System.

Tests core functionality including prompt generation, evaluation,
A/B testing, and API endpoints.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from app.core.prompt_optimizer import PromptOptimizer
from app.core.prompt_generator import PromptGenerator
from app.core.performance_evaluator import PerformanceEvaluator
from app.core.ab_testing_engine import ABTestingEngine
from app.models.prompt_optimization import (
    PromptTemplate, OptimizationExperiment, PromptVariant,
    PromptStatus, ExperimentStatus, OptimizationMethod
)


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def prompt_optimizer(mock_db_session):
    """Create PromptOptimizer instance with mocked dependencies."""
    return PromptOptimizer(mock_db_session)


@pytest.fixture
def prompt_generator(mock_db_session):
    """Create PromptGenerator instance."""
    return PromptGenerator(mock_db_session)


@pytest.fixture
def performance_evaluator(mock_db_session):
    """Create PerformanceEvaluator instance."""
    return PerformanceEvaluator(mock_db_session)


@pytest.fixture
def ab_testing_engine(mock_db_session):
    """Create ABTestingEngine instance."""
    return ABTestingEngine(mock_db_session)


class TestPromptOptimizer:
    """Test PromptOptimizer core functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_prompts_basic(self, prompt_optimizer, mock_db_session):
        """Test basic prompt generation."""
        # Mock database operations
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        result = await prompt_optimizer.generate_prompts(
            task_description="Summarize text",
            domain="general",
            performance_goals=["accuracy"],
            num_candidates=3
        )
        
        assert "prompt_candidates" in result
        assert "experiment_id" in result
        assert "generation_metadata" in result
        assert len(result["prompt_candidates"]) <= 3
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, prompt_optimizer, mock_db_session):
        """Test experiment creation."""
        base_prompt_id = uuid.uuid4()
        
        experiment_id = await prompt_optimizer.create_experiment(
            experiment_name="Test Experiment",
            base_prompt_id=base_prompt_id,
            optimization_method=OptimizationMethod.META_PROMPTING,
            target_metrics={"accuracy": 0.8},
            max_iterations=10
        )
        
        assert isinstance(experiment_id, uuid.UUID)
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()


class TestPromptGenerator:
    """Test PromptGenerator functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_candidates(self, prompt_generator):
        """Test prompt candidate generation."""
        candidates = await prompt_generator.generate_candidates(
            base_prompt="Test base prompt",
            task_description="Test task",
            domain="general",
            performance_goals=["accuracy"],
            num_candidates=3
        )
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 3
        
        for candidate in candidates:
            assert "content" in candidate
            assert "method" in candidate
            assert "reasoning" in candidate
            assert "confidence" in candidate
    
    @pytest.mark.asyncio
    async def test_refine_prompt(self, prompt_generator):
        """Test prompt refinement."""
        result = await prompt_generator.refine_prompt(
            current_prompt="Original prompt",
            performance_feedback={"accuracy": 0.6},
            target_improvements=["clarity", "accuracy"]
        )
        
        assert "content" in result
        assert "method" in result
        assert "reasoning" in result
        assert result["method"] == "iterative_refinement"


class TestPerformanceEvaluator:
    """Test PerformanceEvaluator functionality."""
    
    @pytest.mark.asyncio
    async def test_evaluate_prompt(self, performance_evaluator):
        """Test prompt evaluation."""
        test_cases = [
            {
                "id": "test1",
                "input_data": {"text": "Test input"},
                "expected_output": "Test output",
                "evaluation_criteria": {}
            }
        ]
        
        result = await performance_evaluator.evaluate_prompt(
            prompt_content="Test prompt",
            test_cases=test_cases,
            metrics=["accuracy", "relevance"]
        )
        
        assert "performance_score" in result
        assert "detailed_metrics" in result
        assert "test_case_results" in result
        assert 0.0 <= result["performance_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_evaluate_batch(self, performance_evaluator):
        """Test batch evaluation."""
        prompts = ["Prompt 1", "Prompt 2"]
        test_cases = [
            {"input_data": {"text": "Test"}, "expected_output": "Output"}
        ]
        
        results = await performance_evaluator.evaluate_batch(
            prompts=prompts,
            test_cases=test_cases,
            metrics=["accuracy"]
        )
        
        assert len(results) == 2
        for result in results:
            assert "performance_score" in result
    
    @pytest.mark.asyncio
    async def test_compare_prompts(self, performance_evaluator):
        """Test prompt comparison."""
        test_cases = [
            {"input_data": {"text": "Test"}, "expected_output": "Output"}
        ]
        
        comparison = await performance_evaluator.compare_prompts(
            prompt_a="Prompt A",
            prompt_b="Prompt B",
            test_cases=test_cases,
            metrics=["accuracy"]
        )
        
        assert "prompt_a_score" in comparison
        assert "prompt_b_score" in comparison
        assert "winner" in comparison
        assert comparison["winner"] in ["prompt_a", "prompt_b"]


class TestABTestingEngine:
    """Test ABTestingEngine functionality."""
    
    @pytest.mark.asyncio
    async def test_calculate_required_sample_size(self, ab_testing_engine):
        """Test sample size calculation."""
        result = await ab_testing_engine.calculate_required_sample_size(
            effect_size=0.3,
            power=0.8,
            significance_level=0.05
        )
        
        assert "recommended_sample_size_per_group" in result
        assert "total_sample_size" in result
        assert "actual_power" in result
        assert result["recommended_sample_size_per_group"] > 0
    
    @pytest.mark.asyncio
    async def test_run_test(self, ab_testing_engine, mock_db_session):
        """Test A/B test execution."""
        # Mock database queries
        prompt_variants = [
            MagicMock(id=uuid.uuid4(), variant_content="Prompt A"),
            MagicMock(id=uuid.uuid4(), variant_content="Prompt B")
        ]
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = prompt_variants
        
        prompt_a_id = prompt_variants[0].id
        prompt_b_id = prompt_variants[1].id
        
        # Mock test cases
        ab_testing_engine._get_test_cases = AsyncMock(return_value=[
            {"id": "test1", "input_data": {"text": "Test"}, "expected_output": "Output"}
        ])
        
        result = await ab_testing_engine.run_test(
            prompt_a_id=prompt_a_id,
            prompt_b_id=prompt_b_id,
            sample_size=10,
            significance_level=0.05
        )
        
        assert "test_id" in result
        assert "p_value" in result
        assert "effect_size" in result
        assert "is_statistically_significant" in result
        assert "winner_variant_id" in result


class TestModels:
    """Test database models."""
    
    def test_prompt_template_creation(self):
        """Test PromptTemplate model creation."""
        template = PromptTemplate(
            name="Test Template",
            task_type="summarization",
            domain="general",
            template_content="Test content",
            status=PromptStatus.DRAFT
        )
        
        assert template.name == "Test Template"
        assert template.task_type == "summarization"
        assert template.domain == "general"
        assert template.template_content == "Test content"
        assert template.status == PromptStatus.DRAFT
    
    def test_optimization_experiment_creation(self):
        """Test OptimizationExperiment model creation."""
        base_prompt_id = uuid.uuid4()
        
        experiment = OptimizationExperiment(
            experiment_name="Test Experiment",
            base_prompt_id=base_prompt_id,
            optimization_method=OptimizationMethod.EVOLUTIONARY,
            target_metrics={"accuracy": 0.8},
            status=ExperimentStatus.PENDING
        )
        
        assert experiment.experiment_name == "Test Experiment"
        assert experiment.base_prompt_id == base_prompt_id
        assert experiment.optimization_method == OptimizationMethod.EVOLUTIONARY
        assert experiment.target_metrics == {"accuracy": 0.8}
        assert experiment.status == ExperimentStatus.PENDING


# Integration test
class TestPromptOptimizationIntegration:
    """Integration tests for the full prompt optimization flow."""
    
    @pytest.mark.asyncio
    async def test_full_optimization_flow(self, prompt_optimizer, mock_db_session):
        """Test complete optimization workflow."""
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        # Step 1: Generate initial prompts
        generation_result = await prompt_optimizer.generate_prompts(
            task_description="Analyze sentiment",
            domain="social_media",
            performance_goals=["accuracy", "relevance"],
            num_candidates=2
        )
        
        assert len(generation_result["prompt_candidates"]) <= 2
        experiment_id = uuid.UUID(generation_result["experiment_id"])
        
        # Step 2: Create optimization experiment
        optimization_experiment_id = await prompt_optimizer.create_experiment(
            experiment_name="Sentiment Analysis Optimization",
            base_prompt_id=uuid.uuid4(),
            optimization_method=OptimizationMethod.META_PROMPTING,
            target_metrics={"accuracy": 0.85, "relevance": 0.8}
        )
        
        assert isinstance(optimization_experiment_id, uuid.UUID)
        
        # Step 3: Record feedback
        variant_id = uuid.uuid4()
        feedback_result = await prompt_optimizer.record_feedback(
            prompt_variant_id=variant_id,
            user_id="test_user",
            rating=4,
            feedback_text="Good but could be more specific",
            feedback_categories=["clarity"]
        )
        
        assert feedback_result["status"] == "recorded"
        assert "influence_weight" in feedback_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])