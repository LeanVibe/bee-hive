"""
Performance benchmarking tests for Prompt Optimization System.

Comprehensive performance and load testing for all optimization components
with statistical analysis and performance metrics collection.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock
import uuid
from typing import List, Dict, Any

from app.core.prompt_optimizer import PromptOptimizer
from app.core.evolutionary_optimizer import EvolutionaryOptimizer
from app.core.gradient_optimizer import GradientOptimizer, OptimizationDirection
from app.core.performance_evaluator import PerformanceEvaluator
from app.core.ab_testing_engine import ABTestingEngine
from app.core.feedback_analyzer import FeedbackAnalyzer
from app.core.context_adapter import ContextAdapter, UserPreferences, UserExpertiseLevel, CommunicationStyle
from app.models.prompt_optimization import OptimizationExperiment, PromptTemplate, ExperimentStatus, OptimizationMethod


class PerformanceMetrics:
    """Performance metrics collection and analysis."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timing(self, operation: str):
        """End timing an operation and record the duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            return duration
        return 0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_increase(self) -> float:
        """Get memory increase from initial measurement."""
        return self.get_memory_usage() - self.initial_memory
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of all metrics."""
        stats = {}
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'std': statistics.stdev(times) if len(times) > 1 else 0,
                    'min': min(times),
                    'max': max(times),
                    'p95': sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0],
                    'p99': sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else times[0]
                }
        
        stats['memory'] = {
            'current_mb': self.get_memory_usage(),
            'increase_mb': self.get_memory_increase(),
            'initial_mb': self.initial_memory
        }
        
        return stats


@pytest.fixture
def performance_metrics():
    """Performance metrics collector."""
    return PerformanceMetrics()


@pytest.fixture
def mock_db_session():
    """Mock database session optimized for performance testing."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock(return_value=None)
    session.refresh = AsyncMock(return_value=None)
    session.rollback = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    
    # Fast mock responses
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_result.scalars = MagicMock()
    mock_result.scalars.return_value.all = MagicMock(return_value=[])
    session.execute.return_value = mock_result
    
    return session


@pytest.fixture
def sample_experiment():
    """Sample experiment for performance testing."""
    template = PromptTemplate(
        id=uuid.uuid4(),
        name="Perf Test Template",
        task_type="performance_test",
        domain="general",
        template_content="Analyze the following data: {data}",
        template_variables={"data": "string"}
    )
    
    return OptimizationExperiment(
        id=uuid.uuid4(),
        experiment_name="Performance Test",
        base_prompt_id=template.id,
        base_prompt=template,
        optimization_method=OptimizationMethod.EVOLUTIONARY,
        target_metrics={"accuracy": 0.8},
        experiment_config={"max_iterations": 5},
        status=ExperimentStatus.PENDING
    )


@pytest.fixture
def performance_test_cases():
    """Test cases optimized for performance testing."""
    return [
        {
            'id': f'perf_test_{i}',
            'input_data': {'data': f'Performance test data sample {i}'},
            'expected_output': f'Expected output {i}',
            'evaluation_criteria': {'accuracy': True, 'relevance': True}
        }
        for i in range(10)
    ]


@pytest.mark.benchmark
class TestPromptGenerationPerformance:
    """Performance tests for prompt generation."""
    
    @pytest.mark.asyncio
    async def test_single_prompt_generation_performance(self, mock_db_session, performance_metrics):
        """Test single prompt generation performance."""
        optimizer = PromptOptimizer(mock_db_session)
        
        # Mock database responses for fast execution
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        performance_metrics.start_timing("single_generation")
        
        result = await optimizer.generate_prompts(
            task_description="Performance benchmark task",
            domain="general",
            performance_goals=["accuracy"],
            num_candidates=5
        )
        
        duration = performance_metrics.end_timing("single_generation")
        
        # Performance assertion: should complete in < 3 seconds
        assert duration < 3.0, f"Single prompt generation too slow: {duration:.2f}s"
        assert len(result["prompt_candidates"]) <= 5
        assert "experiment_id" in result
    
    @pytest.mark.asyncio
    async def test_batch_prompt_generation_performance(self, mock_db_session, performance_metrics):
        """Test batch prompt generation performance."""
        optimizer = PromptOptimizer(mock_db_session)
        
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        tasks = [
            "Summarize text content",
            "Analyze sentiment in reviews", 
            "Extract key information",
            "Generate report summary",
            "Classify document type"
        ]
        
        performance_metrics.start_timing("batch_generation")
        
        # Execute batch generation
        results = []
        for task in tasks:
            performance_metrics.start_timing(f"generation_{len(results)}")
            result = await optimizer.generate_prompts(
                task_description=task,
                domain="general",
                performance_goals=["accuracy"],
                num_candidates=3
            )
            performance_metrics.end_timing(f"generation_{len(results)}")
            results.append(result)
        
        total_duration = performance_metrics.end_timing("batch_generation")
        
        # Performance assertions
        assert total_duration < 10.0, f"Batch generation too slow: {total_duration:.2f}s"
        assert len(results) == len(tasks)
        
        # Check individual generation times
        stats = performance_metrics.get_statistics()
        individual_times = [stats[f'generation_{i}']['mean'] for i in range(len(tasks))]
        avg_individual_time = statistics.mean(individual_times)
        assert avg_individual_time < 2.5, f"Average individual generation too slow: {avg_individual_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_prompt_generation_performance(self, mock_db_session, performance_metrics):
        """Test concurrent prompt generation performance."""
        # Create multiple optimizers to simulate concurrent usage
        optimizers = [PromptOptimizer(mock_db_session) for _ in range(3)]
        
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        performance_metrics.start_timing("concurrent_generation")
        
        # Create concurrent tasks
        tasks = []
        for i, optimizer in enumerate(optimizers):
            task = optimizer.generate_prompts(
                task_description=f"Concurrent task {i}",
                domain="general",
                performance_goals=["accuracy"],
                num_candidates=3
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = performance_metrics.end_timing("concurrent_generation")
        
        # Performance assertions
        assert duration < 8.0, f"Concurrent generation too slow: {duration:.2f}s"
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 2, "Too many concurrent generations failed"
        
        # Memory usage should remain reasonable
        memory_increase = performance_metrics.get_memory_increase()
        assert memory_increase < 150, f"Memory usage too high during concurrent generation: {memory_increase:.2f}MB"


@pytest.mark.benchmark
class TestEvaluationPerformance:
    """Performance tests for prompt evaluation."""
    
    @pytest.mark.asyncio
    async def test_single_evaluation_performance(self, mock_db_session, performance_metrics, performance_test_cases):
        """Test single prompt evaluation performance."""
        evaluator = PerformanceEvaluator(mock_db_session)
        
        test_prompt = "Analyze the given data and provide comprehensive insights."
        
        performance_metrics.start_timing("single_evaluation")
        
        result = await evaluator.evaluate_prompt(
            prompt_content=test_prompt,
            test_cases=performance_test_cases[:5],  # Use 5 test cases
            metrics=['accuracy', 'relevance', 'coherence']
        )
        
        duration = performance_metrics.end_timing("single_evaluation")
        
        # Performance assertion: should complete in < 1.5 seconds
        assert duration < 1.5, f"Single evaluation too slow: {duration:.2f}s"
        assert "performance_score" in result
        assert 0.0 <= result["performance_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_performance(self, mock_db_session, performance_metrics, performance_test_cases):
        """Test batch evaluation performance."""
        evaluator = PerformanceEvaluator(mock_db_session)
        
        prompts = [
            "Summarize the key points in the text.",
            "Extract important information from the data.",
            "Analyze trends and patterns in the content.",
            "Provide insights based on the given information.",
            "Identify critical elements in the text.",
            "Generate a comprehensive analysis report.",
            "Highlight the most relevant aspects.",
            "Create a structured summary of findings."
        ]
        
        performance_metrics.start_timing("batch_evaluation")
        
        results = await evaluator.evaluate_batch(
            prompts=prompts,
            test_cases=performance_test_cases[:3],  # Use 3 test cases to balance speed vs coverage
            metrics=['accuracy', 'relevance']
        )
        
        duration = performance_metrics.end_timing("batch_evaluation")
        
        # Performance assertions
        assert duration < 5.0, f"Batch evaluation too slow: {duration:.2f}s"
        assert len(results) == len(prompts)
        
        # Average time per prompt should be reasonable
        avg_time_per_prompt = duration / len(prompts)
        assert avg_time_per_prompt < 0.8, f"Average evaluation time per prompt too slow: {avg_time_per_prompt:.2f}s"
    
    @pytest.mark.asyncio
    async def test_comparison_performance(self, mock_db_session, performance_metrics, performance_test_cases):
        """Test prompt comparison performance."""
        evaluator = PerformanceEvaluator(mock_db_session)
        
        prompt_a = "Analyze the data systematically and provide detailed insights."
        prompt_b = "Examine the information carefully and deliver comprehensive analysis."
        
        performance_metrics.start_timing("comparison")
        
        comparison = await evaluator.compare_prompts(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            test_cases=performance_test_cases[:5],
            metrics=['accuracy', 'relevance', 'coherence']
        )
        
        duration = performance_metrics.end_timing("comparison")
        
        # Performance assertion: should complete in < 2.5 seconds
        assert duration < 2.5, f"Comparison too slow: {duration:.2f}s"
        assert "winner" in comparison
        assert comparison["winner"] in ["prompt_a", "prompt_b"]


@pytest.mark.benchmark
class TestOptimizationPerformance:
    """Performance tests for optimization algorithms."""
    
    @pytest.mark.asyncio
    async def test_evolutionary_optimization_performance(self, mock_db_session, performance_metrics, sample_experiment):
        """Test evolutionary optimization performance."""
        optimizer = EvolutionaryOptimizer(mock_db_session)
        
        # Configure for performance testing (smaller scale)
        optimizer.population_size = 8
        optimizer.max_generations = 5
        
        performance_metrics.start_timing("evolutionary_optimization")
        
        result = await optimizer.optimize(
            experiment=sample_experiment,
            max_iterations=5,
            population_size=8
        )
        
        duration = performance_metrics.end_timing("evolutionary_optimization")
        
        # Performance assertions
        assert duration < 15.0, f"Evolutionary optimization too slow: {duration:.2f}s"
        assert "best_score" in result
        assert "iterations_completed" in result
        assert result["iterations_completed"] <= 5
        
        # Check average time per iteration
        if result["iterations_completed"] > 0:
            avg_iteration_time = duration / result["iterations_completed"]
            assert avg_iteration_time < 5.0, f"Average iteration time too slow: {avg_iteration_time:.2f}s"
    
    @pytest.mark.asyncio 
    async def test_gradient_optimization_performance(self, mock_db_session, performance_metrics, sample_experiment):
        """Test gradient optimization performance."""
        optimizer = GradientOptimizer(mock_db_session)
        
        # Configure for performance testing
        optimizer.max_iterations = 8
        optimizer.finite_difference_h = 0.05  # Larger step for faster computation
        
        performance_metrics.start_timing("gradient_optimization")
        
        result = await optimizer.optimize(
            experiment=sample_experiment,
            max_iterations=8,
            optimization_method=OptimizationDirection.ADAM
        )
        
        duration = performance_metrics.end_timing("gradient_optimization")
        
        # Performance assertions
        assert duration < 20.0, f"Gradient optimization too slow: {duration:.2f}s"
        assert "best_score" in result
        assert "iterations_completed" in result
        assert result["iterations_completed"] <= 8
        
        # Check optimization efficiency
        if result["optimization_time_seconds"] > 0:
            efficiency = result.get("improvement_percentage", 0) / result["optimization_time_seconds"]
            assert efficiency >= 0, "Optimization should show some improvement per second"
    
    @pytest.mark.asyncio
    async def test_ab_testing_performance(self, mock_db_session, performance_metrics):
        """Test A/B testing performance."""
        ab_engine = ABTestingEngine(mock_db_session)
        
        prompt_a_id = uuid.uuid4()
        prompt_b_id = uuid.uuid4()
        
        # Mock prompt variants
        mock_variants = [
            MagicMock(id=prompt_a_id, variant_content="Prompt A content"),
            MagicMock(id=prompt_b_id, variant_content="Prompt B content")
        ]
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_variants
        
        performance_metrics.start_timing("ab_testing")
        
        result = await ab_engine.run_test(
            prompt_a_id=prompt_a_id,
            prompt_b_id=prompt_b_id,
            sample_size=50,  # Smaller sample size for performance testing
            significance_level=0.05
        )
        
        duration = performance_metrics.end_timing("ab_testing")
        
        # Performance assertions
        assert duration < 8.0, f"A/B testing too slow: {duration:.2f}s"
        assert "p_value" in result
        assert "effect_size" in result
        assert "winner_variant_id" in result


@pytest.mark.benchmark
class TestAnalysisPerformance:
    """Performance tests for analysis components."""
    
    @pytest.mark.asyncio
    async def test_feedback_analysis_performance(self, mock_db_session, performance_metrics):
        """Test feedback analysis performance."""
        analyzer = FeedbackAnalyzer(mock_db_session)
        
        feedback_samples = [
            {"rating": 5, "text": "Excellent work with great accuracy and clarity!", "categories": ["accuracy", "clarity"]},
            {"rating": 2, "text": "Poor results, unclear and inaccurate information provided.", "categories": ["clarity", "accuracy"]},
            {"rating": 4, "text": "Good overall performance but could be more complete.", "categories": ["completeness"]},
            {"rating": 3, "text": "Average quality, meets basic requirements but not outstanding.", "categories": ["relevance"]},
            {"rating": 5, "text": "Outstanding analysis with comprehensive insights and perfect clarity.", "categories": ["accuracy", "clarity", "completeness"]},
        ]
        
        performance_metrics.start_timing("feedback_analysis_batch")
        
        results = []
        for sample in feedback_samples:
            performance_metrics.start_timing("single_feedback_analysis")
            
            result = await analyzer.analyze_feedback(
                rating=sample["rating"],
                feedback_text=sample["text"],
                feedback_categories=sample["categories"]
            )
            
            performance_metrics.end_timing("single_feedback_analysis")
            results.append(result)
        
        total_duration = performance_metrics.end_timing("feedback_analysis_batch")
        
        # Performance assertions
        assert total_duration < 3.0, f"Batch feedback analysis too slow: {total_duration:.2f}s"
        assert len(results) == len(feedback_samples)
        
        # Check individual analysis times
        stats = performance_metrics.get_statistics()
        avg_single_time = stats["single_feedback_analysis"]["mean"]
        assert avg_single_time < 0.8, f"Single feedback analysis too slow: {avg_single_time:.2f}s"
        
        # Verify analysis quality
        for result in results:
            assert result.sentiment_analysis is not None
            assert result.quality_scores is not None
            assert 0.0 <= result.confidence_level <= 1.0
    
    @pytest.mark.asyncio
    async def test_context_adaptation_performance(self, mock_db_session, performance_metrics):
        """Test context adaptation performance."""
        adapter = ContextAdapter(mock_db_session)
        
        # Sample user preferences
        user_prefs = UserPreferences(
            expertise_level=UserExpertiseLevel.ADVANCED,
            communication_style=CommunicationStyle.TECHNICAL,
            preferred_domains=[],
            language_preferences={'formality': 0.8},
            format_preferences={'structured': 0.9},
            content_preferences={'detailed': 0.8},
            personalization_settings={}
        )
        
        base_prompts = [
            "Analyze the data provided.",
            "Generate a summary of the information.",
            "Provide insights about the content.",
            "Extract key findings from the text.",
            "Create a comprehensive report.",
        ]
        
        performance_metrics.start_timing("context_adaptation_batch")
        
        results = []
        for prompt in base_prompts:
            performance_metrics.start_timing("single_adaptation")
            
            result = await adapter.adapt_to_context(
                base_prompt=prompt,
                domain="technical",
                user_preferences=user_prefs
            )
            
            performance_metrics.end_timing("single_adaptation")
            results.append(result)
        
        total_duration = performance_metrics.end_timing("context_adaptation_batch")
        
        # Performance assertions
        assert total_duration < 2.0, f"Batch context adaptation too slow: {total_duration:.2f}s"
        assert len(results) == len(base_prompts)
        
        # Check individual adaptation times
        stats = performance_metrics.get_statistics()
        avg_single_time = stats["single_adaptation"]["mean"]
        assert avg_single_time < 0.5, f"Single adaptation too slow: {avg_single_time:.2f}s"
        
        # Verify adaptation quality
        for result in results:
            assert result.adapted_content != ""
            assert result.overall_adaptation_score > 0.0
            assert len(result.adaptations_applied) > 0


@pytest.mark.benchmark
class TestScalabilityPerformance:
    """Scalability and load testing for optimization system."""
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization_scalability(self, mock_db_session, performance_metrics):
        """Test system scalability under concurrent optimization load."""
        # Create multiple experiments
        experiments = []
        for i in range(5):
            template = PromptTemplate(
                id=uuid.uuid4(),
                name=f"Scale Test {i}",
                domain="general",
                template_content=f"Scale test prompt {i}: {{data}}",
                template_variables={"data": "string"}
            )
            
            experiment = OptimizationExperiment(
                id=uuid.uuid4(),
                experiment_name=f"Scale Test {i}",
                base_prompt_id=template.id,
                base_prompt=template,
                optimization_method=OptimizationMethod.META_PROMPTING,
                target_metrics={"accuracy": 0.8},
                experiment_config={"max_iterations": 3},
                status=ExperimentStatus.PENDING
            )
            experiments.append(experiment)
        
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        performance_metrics.start_timing("concurrent_scalability")
        
        # Run concurrent optimizations
        optimizers = [PromptOptimizer(mock_db_session) for _ in range(len(experiments))]
        
        tasks = []
        for optimizer, experiment in zip(optimizers, experiments):
            task = optimizer._run_meta_prompting_optimization(experiment, max_iterations=3)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = performance_metrics.end_timing("concurrent_scalability")
        
        # Performance assertions
        assert duration < 25.0, f"Concurrent scalability test too slow: {duration:.2f}s"
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 3, f"Too many concurrent optimizations failed: {len(successful_results)}/{len(experiments)}"
        
        # Memory usage should be reasonable
        memory_increase = performance_metrics.get_memory_increase()
        assert memory_increase < 200, f"Memory usage too high during scalability test: {memory_increase:.2f}MB"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, mock_db_session, performance_metrics):
        """Test for memory leaks during extended operation."""
        optimizer = PromptOptimizer(mock_db_session)
        
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        memory_samples = []
        
        # Run multiple optimization cycles
        for i in range(10):
            performance_metrics.start_timing(f"memory_test_cycle_{i}")
            
            await optimizer.generate_prompts(
                task_description=f"Memory test cycle {i}",
                domain="general",
                performance_goals=["accuracy"],
                num_candidates=3
            )
            
            performance_metrics.end_timing(f"memory_test_cycle_{i}")
            
            # Sample memory usage
            current_memory = performance_metrics.get_memory_usage()
            memory_samples.append(current_memory)
            
            # Small delay to allow garbage collection
            await asyncio.sleep(0.1)
        
        # Check for memory leak indicators
        if len(memory_samples) > 5:
            early_avg = statistics.mean(memory_samples[:3])
            late_avg = statistics.mean(memory_samples[-3:])
            memory_growth = late_avg - early_avg
            
            # Memory growth should be minimal (< 50MB over 10 cycles)
            assert memory_growth < 50, f"Potential memory leak detected: {memory_growth:.2f}MB growth"
        
        # Final memory increase should be reasonable
        final_memory_increase = performance_metrics.get_memory_increase()
        assert final_memory_increase < 100, f"Total memory increase too high: {final_memory_increase:.2f}MB"


@pytest.mark.benchmark
class TestPerformanceReporting:
    """Performance reporting and metrics analysis."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_performance_report(self, mock_db_session):
        """Generate comprehensive performance report."""
        metrics = PerformanceMetrics()
        
        # Run a representative workload
        optimizer = PromptOptimizer(mock_db_session)
        evaluator = PerformanceEvaluator(mock_db_session)
        
        # Mock database responses
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        test_cases = [
            {'id': 'test_1', 'input_data': {'text': 'Test'}, 'expected_output': 'Output', 'evaluation_criteria': {}}
        ]
        
        # Performance benchmark workload
        operations = [
            ("prompt_generation", lambda: optimizer.generate_prompts("Test task", "general", ["accuracy"], num_candidates=3)),
            ("prompt_evaluation", lambda: evaluator.evaluate_prompt("Test prompt", test_cases, ["accuracy"])),
            ("batch_evaluation", lambda: evaluator.evaluate_batch(["Prompt 1", "Prompt 2"], test_cases, ["accuracy"])),
        ]
        
        for op_name, operation in operations:
            for i in range(3):  # Run each operation 3 times
                metrics.start_timing(f"{op_name}_{i}")
                await operation()
                metrics.end_timing(f"{op_name}_{i}")
        
        # Generate performance report
        stats = metrics.get_statistics()
        
        # Verify report structure
        assert 'memory' in stats
        assert stats['memory']['current_mb'] > 0
        
        # Verify operation statistics
        for op_name, _ in operations:
            for i in range(3):
                key = f"{op_name}_{i}"
                if key in stats:
                    assert 'mean' in stats[key]
                    assert 'count' in stats[key]
                    assert stats[key]['count'] == 1
        
        # Print performance report for manual review
        print("\n" + "="*50)
        print("PROMPT OPTIMIZATION PERFORMANCE REPORT")
        print("="*50)
        
        for operation, op_stats in stats.items():
            if operation != 'memory':
                print(f"\n{operation.upper()}:")
                for metric, value in op_stats.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}s")
                    else:
                        print(f"  {metric}: {value}")
        
        print(f"\nMEMORY USAGE:")
        memory_stats = stats['memory']
        for metric, value in memory_stats.items():
            print(f"  {metric}: {value:.2f}MB")
        
        print("="*50)


# Performance test configuration
@pytest.fixture(autouse=True)
def setup_performance_test_environment():
    """Setup environment for performance testing."""
    # Ensure consistent test environment
    import gc
    gc.collect()  # Clean up before tests
    
    yield
    
    # Cleanup after tests
    gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark", "--tb=short"])