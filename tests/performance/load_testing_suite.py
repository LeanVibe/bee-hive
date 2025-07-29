"""
Load Testing Suite for Semantic Memory System.

Comprehensive load testing framework to validate performance under realistic
production workloads and stress conditions.

Features:
- Locust-based distributed load testing
- Realistic user behavior simulation  
- Multiple load patterns (constant, burst, ramp-up, spike)
- Concurrent multi-agent scenarios
- Real-time performance monitoring
- Resource utilization tracking
- Failure scenario testing
- Performance degradation analysis
"""

import asyncio
import logging
import time
import json
import statistics
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import numpy as np
from locust import User, task, between, events, HttpUser
from locust.runners import MasterRunner, WorkerRunner
from locust.env import Environment

from app.services.semantic_memory_service import SemanticMemoryService, get_semantic_memory_service
from app.schemas.semantic_memory import (
    DocumentIngestRequest, BatchIngestRequest, SemanticSearchRequest,
    ContextCompressionRequest, SimilaritySearchRequest, ContextualizationRequest,
    DocumentMetadata, ProcessingOptions, BatchOptions, SearchFilters,
    CompressionMethod, ContextualizationMethod, TimeRange, KnowledgeType
)
from .semantic_memory_benchmarks import (
    RealisticDataGenerator, ResourceMonitor, PerformanceTarget,
    TestDataSize, LoadPattern, BenchmarkResult, LoadTestResult
)

logger = logging.getLogger(__name__)


class LoadTestScenario(Enum):
    """Load testing scenarios."""
    SINGLE_USER = "single_user"
    NORMAL_LOAD = "normal_load"      # 10 concurrent users
    HIGH_LOAD = "high_load"          # 50 concurrent users  
    STRESS_TEST = "stress_test"      # 100+ concurrent users
    SPIKE_TEST = "spike_test"        # Sudden load increase
    ENDURANCE_TEST = "endurance_test" # Long duration test


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    scenario: LoadTestScenario
    users: int
    spawn_rate: float
    duration_seconds: int
    load_pattern: LoadPattern = LoadPattern.CONSTANT
    
    # Operation weights (probabilities)
    search_weight: float = 0.5
    ingest_weight: float = 0.3
    compression_weight: float = 0.1
    knowledge_weight: float = 0.1
    
    # Performance thresholds
    max_response_time_ms: float = 1000.0
    max_error_rate_percent: float = 5.0
    
    # Test data
    document_pool_size: int = 10000
    agent_pool_size: int = 50


@dataclass
class UserBehavior:
    """Simulated user behavior patterns."""
    think_time_min: float = 1.0
    think_time_max: float = 5.0
    session_duration_min: int = 60
    session_duration_max: int = 300
    operations_per_session: int = 20
    
    # Behavior probabilities
    repeat_query_probability: float = 0.2
    modify_query_probability: float = 0.3
    switch_agent_probability: float = 0.1


class SemanticMemoryUser(User):
    """Locust user simulating semantic memory operations."""
    
    abstract = True
    
    def __init__(self, environment):
        super().__init__(environment)
        self.data_generator = RealisticDataGenerator()
        self.semantic_service: Optional[SemanticMemoryService] = None
        self.agent_ids = []
        self.document_cache = deque(maxlen=100)
        self.query_cache = deque(maxlen=50)
        self.current_agent_id = None
        
        # User behavior
        self.behavior = UserBehavior()
        self.session_start = time.time()
        self.operations_completed = 0
        
    async def on_start(self):
        """Initialize user session."""
        try:
            # Initialize semantic service
            self.semantic_service = await get_semantic_memory_service()
            
            # Generate agent pool for this user
            self.agent_ids = [str(uuid.uuid4()) for _ in range(5)]
            self.current_agent_id = random.choice(self.agent_ids)
            
            # Pre-populate some documents for testing
            await self._populate_initial_documents()
            
            logger.debug(f"User initialized with {len(self.agent_ids)} agents")
            
        except Exception as e:
            logger.error(f"User initialization failed: {e}")
            raise
    
    async def _populate_initial_documents(self):
        """Pre-populate documents for realistic testing."""
        try:
            # Create 10-20 initial documents per agent
            for agent_id in self.agent_ids:
                doc_count = random.randint(10, 20)
                documents = []
                
                for _ in range(doc_count):
                    content = self.data_generator.generate_document_content()
                    metadata = self.data_generator.generate_metadata()
                    tags = self.data_generator.generate_tags()
                    
                    doc_request = DocumentIngestRequest(
                        agent_id=agent_id,
                        content=content,
                        metadata=DocumentMetadata(**metadata),
                        tags=tags
                    )
                    documents.append(doc_request)
                
                # Batch ingest
                batch_request = BatchIngestRequest(
                    documents=documents,
                    batch_options=BatchOptions(parallel_processing=True)
                )
                
                response = await self.semantic_service.batch_ingest_documents(batch_request)
                
                # Cache successful document IDs
                for result in response.results:
                    if result.status == "success" and result.document_id:
                        self.document_cache.append(str(result.document_id))
                        
        except Exception as e:
            logger.warning(f"Failed to populate initial documents: {e}")
    
    def should_continue_session(self) -> bool:
        """Check if user should continue current session."""
        session_duration = time.time() - self.session_start
        return (session_duration < self.behavior.session_duration_max and 
                self.operations_completed < self.behavior.operations_per_session)
    
    @task(50)  # 50% weight
    async def perform_semantic_search(self):
        """Perform semantic search operation."""
        if not self.should_continue_session():
            return
            
        try:
            start_time = time.time()
            
            # Generate or reuse query
            if self.query_cache and random.random() < self.behavior.repeat_query_probability:
                query = random.choice(list(self.query_cache))
                if random.random() < self.behavior.modify_query_probability:
                    # Slightly modify the query
                    query += f" {random.choice(['advanced', 'optimization', 'best practices'])}"
            else:
                query = random.choice(self.data_generator.generate_search_queries(1))
                self.query_cache.append(query)
            
            # Switch agent occasionally
            if random.random() < self.behavior.switch_agent_probability:
                self.current_agent_id = random.choice(self.agent_ids)
            
            # Create search request
            search_request = SemanticSearchRequest(
                query=query,
                limit=random.randint(5, 20),
                similarity_threshold=random.uniform(0.4, 0.8),
                agent_id=self.current_agent_id
            )
            
            # Execute search
            response = await self.semantic_service.semantic_search(search_request)
            
            response_time = (time.time() - start_time) * 1000
            
            # Record metrics
            events.request_success.fire(
                request_type="semantic_search",
                name="search_query",
                response_time=response_time,
                response_length=len(response.results) if response.results else 0
            )
            
            self.operations_completed += 1
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="semantic_search",
                name="search_query",
                response_time=response_time,
                response_length=0,
                exception=e
            )
            logger.warning(f"Search operation failed: {e}")
    
    @task(30)  # 30% weight
    async def ingest_document(self):
        """Ingest new document."""
        if not self.should_continue_session():
            return
            
        try:
            start_time = time.time()
            
            # Generate document
            content = self.data_generator.generate_document_content()
            metadata = self.data_generator.generate_metadata()
            tags = self.data_generator.generate_tags()
            
            doc_request = DocumentIngestRequest(
                agent_id=self.current_agent_id,
                content=content,
                metadata=DocumentMetadata(**metadata),
                tags=tags
            )
            
            # Ingest document
            response = await self.semantic_service.ingest_document(doc_request)
            
            response_time = (time.time() - start_time) * 1000
            
            # Cache document ID
            if response.document_id:
                self.document_cache.append(str(response.document_id))
            
            # Record metrics
            events.request_success.fire(
                request_type="document_ingest",
                name="ingest_single",
                response_time=response_time,
                response_length=len(content)
            )
            
            self.operations_completed += 1
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="document_ingest",
                name="ingest_single",
                response_time=response_time,
                response_length=0,
                exception=e
            )
            logger.warning(f"Document ingestion failed: {e}")
    
    @task(15)  # 15% weight
    async def batch_ingest_documents(self):
        """Batch ingest multiple documents."""
        if not self.should_continue_session():
            return
            
        try:
            start_time = time.time()
            
            # Generate batch of documents
            batch_size = random.randint(3, 10)
            documents = []
            total_content_length = 0
            
            for _ in range(batch_size):
                content = self.data_generator.generate_document_content()
                metadata = self.data_generator.generate_metadata()
                tags = self.data_generator.generate_tags()
                
                doc_request = DocumentIngestRequest(
                    agent_id=self.current_agent_id,
                    content=content,
                    metadata=DocumentMetadata(**metadata),
                    tags=tags
                )
                documents.append(doc_request)
                total_content_length += len(content)
            
            # Create batch request
            batch_request = BatchIngestRequest(
                documents=documents,
                batch_options=BatchOptions(
                    parallel_processing=True,
                    batch_size=batch_size
                )
            )
            
            # Execute batch ingestion
            response = await self.semantic_service.batch_ingest_documents(batch_request)
            
            response_time = (time.time() - start_time) * 1000
            
            # Cache successful document IDs
            for result in response.results:
                if result.status == "success" and result.document_id:
                    self.document_cache.append(str(result.document_id))
            
            # Record metrics
            events.request_success.fire(
                request_type="batch_ingest",
                name="ingest_batch",
                response_time=response_time,
                response_length=total_content_length
            )
            
            self.operations_completed += 1
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="batch_ingest",
                name="ingest_batch",
                response_time=response_time,
                response_length=0,
                exception=e
            )
            logger.warning(f"Batch ingestion failed: {e}")
    
    @task(10)  # 10% weight
    async def compress_context(self):
        """Perform context compression."""
        if not self.should_continue_session():
            return
            
        try:
            start_time = time.time()
            
            # Create compression request
            compression_request = ContextCompressionRequest(
                context_id=f"test_context_{self.current_agent_id}_{int(time.time())}",
                compression_method=random.choice(list(CompressionMethod)),
                target_reduction=random.uniform(0.5, 0.8),
                preserve_importance_threshold=random.uniform(0.6, 0.9)
            )
            
            # Execute compression
            response = await self.semantic_service.compress_context(compression_request)
            
            response_time = (time.time() - start_time) * 1000
            
            # Record metrics
            events.request_success.fire(
                request_type="context_compression",
                name="compress_context",
                response_time=response_time,
                response_length=response.compressed_size
            )
            
            self.operations_completed += 1
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="context_compression",
                name="compress_context",
                response_time=response_time,
                response_length=0,
                exception=e
            )
            logger.warning(f"Context compression failed: {e}")
    
    @task(10)  # 10% weight
    async def get_agent_knowledge(self):
        """Retrieve agent knowledge."""
        if not self.should_continue_session():
            return
            
        try:
            start_time = time.time()
            
            # Get knowledge for current agent
            response = await self.semantic_service.get_agent_knowledge(
                agent_id=self.current_agent_id,
                knowledge_type=random.choice(list(KnowledgeType)),
                time_range=random.choice(list(TimeRange))
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Record metrics
            events.request_success.fire(
                request_type="agent_knowledge",
                name="get_knowledge",
                response_time=response_time,
                response_length=len(response.knowledge_base.patterns) if response.knowledge_base else 0
            )
            
            self.operations_completed += 1
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="agent_knowledge",
                name="get_knowledge",
                response_time=response_time,
                response_length=0,
                exception=e
            )
            logger.warning(f"Knowledge retrieval failed: {e}")
    
    @task(5)  # 5% weight
    async def find_similar_documents(self):
        """Find similar documents."""
        if not self.should_continue_session() or not self.document_cache:
            return
            
        try:
            start_time = time.time()
            
            # Use a cached document for similarity search
            document_id = random.choice(list(self.document_cache))
            
            similarity_request = SimilaritySearchRequest(
                document_id=uuid.UUID(document_id),
                limit=random.randint(5, 15),
                similarity_threshold=random.uniform(0.5, 0.8),
                exclude_self=True
            )
            
            # Execute similarity search
            response = await self.semantic_service.find_similar_documents(similarity_request)
            
            response_time = (time.time() - start_time) * 1000
            
            # Record metrics
            events.request_success.fire(
                request_type="similarity_search",
                name="find_similar",
                response_time=response_time,
                response_length=len(response.similar_documents) if response.similar_documents else 0
            )
            
            self.operations_completed += 1
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="similarity_search",
                name="find_similar",
                response_time=response_time,
                response_length=0,
                exception=e
            )
            logger.warning(f"Similarity search failed: {e}")
    
    def wait_time(self):
        """Dynamic wait time based on user behavior."""
        return between(self.behavior.think_time_min, self.behavior.think_time_max)


class LoadTestRunner:
    """Load test execution and management."""
    
    def __init__(self):
        self.results: List[LoadTestResult] = []
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = defaultdict(list)
        
    def setup_locust_events(self):
        """Setup Locust event handlers for metrics collection."""
        
        @events.request_success.add_listener
        def on_request_success(request_type, name, response_time, response_length, **kwargs):
            self.metrics_collector['success_times'].append(response_time)
            self.metrics_collector['success_operations'].append({
                'timestamp': time.time(),
                'request_type': request_type,
                'name': name,
                'response_time': response_time,
                'response_length': response_length
            })
        
        @events.request_failure.add_listener
        def on_request_failure(request_type, name, response_time, response_length, exception, **kwargs):
            self.metrics_collector['failure_times'].append(response_time)
            self.metrics_collector['failure_operations'].append({
                'timestamp': time.time(),
                'request_type': request_type,
                'name': name,
                'response_time': response_time,
                'exception': str(exception)
            })
    
    async def run_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Execute load test with given configuration."""
        logger.info(f"🚀 Starting load test: {config.scenario.value} with {config.users} users")
        
        # Setup Locust environment
        env = Environment(user_classes=[SemanticMemoryUser])
        self.setup_locust_events()
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Configure load pattern
            if config.load_pattern == LoadPattern.CONSTANT:
                await self._run_constant_load(env, config)
            elif config.load_pattern == LoadPattern.RAMP_UP:
                await self._run_ramp_up_load(env, config)
            elif config.load_pattern == LoadPattern.SPIKE:
                await self._run_spike_load(env, config)
            elif config.load_pattern == LoadPattern.BURST:
                await self._run_burst_load(env, config)
            else:
                await self._run_mixed_load(env, config)
            
            duration = time.time() - start_time
            
            # Stop monitoring and collect metrics
            await self.resource_monitor.stop_monitoring()
            resource_stats = self.resource_monitor.get_stats()
            
            # Calculate performance metrics
            result = self._calculate_load_test_result(config, duration, resource_stats)
            self.results.append(result)
            
            logger.info(f"✅ Load test completed: {result.operations_per_second:.1f} ops/sec, "
                       f"{result.error_rate_percent:.1f}% errors")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Load test failed: {e}")
            raise
        finally:
            # Cleanup
            if env.runner:
                env.runner.quit()
    
    async def _run_constant_load(self, env: Environment, config: LoadTestConfig):
        """Run constant load test."""
        runner = env.create_local_runner()
        
        # Start users gradually
        runner.start(config.users, spawn_rate=config.spawn_rate)
        
        # Wait for ramp-up to complete
        await asyncio.sleep(config.users / config.spawn_rate)
        
        # Run for specified duration
        await asyncio.sleep(config.duration_seconds)
        
        # Stop users
        runner.stop()
        await asyncio.sleep(2)  # Allow cleanup
    
    async def _run_ramp_up_load(self, env: Environment, config: LoadTestConfig):
        """Run ramp-up load test."""
        runner = env.create_local_runner()
        
        # Gradual ramp-up over first 25% of duration
        ramp_duration = config.duration_seconds * 0.25
        steady_duration = config.duration_seconds * 0.75
        
        # Start with 1 user and ramp up
        runner.start(1, spawn_rate=1)
        await asyncio.sleep(1)
        
        # Gradual increase
        steps = 5
        users_per_step = config.users // steps
        step_duration = ramp_duration / steps
        
        for step in range(1, steps + 1):
            target_users = min(users_per_step * step, config.users)
            runner.start(target_users, spawn_rate=config.spawn_rate)
            await asyncio.sleep(step_duration)
        
        # Steady state
        await asyncio.sleep(steady_duration)
        
        # Stop users
        runner.stop()
        await asyncio.sleep(2)
    
    async def _run_spike_load(self, env: Environment, config: LoadTestConfig):
        """Run spike load test."""
        runner = env.create_local_runner()
        
        # Normal load for 30% of duration
        normal_users = config.users // 4
        normal_duration = config.duration_seconds * 0.3
        
        runner.start(normal_users, spawn_rate=config.spawn_rate)
        await asyncio.sleep(normal_duration)
        
        # Spike to full load for 40% of duration
        spike_duration = config.duration_seconds * 0.4
        
        runner.start(config.users, spawn_rate=config.users)  # Rapid spike
        await asyncio.sleep(spike_duration)
        
        # Return to normal for remaining 30%
        recovery_duration = config.duration_seconds * 0.3
        
        runner.start(normal_users, spawn_rate=config.spawn_rate)
        await asyncio.sleep(recovery_duration)
        
        # Stop users
        runner.stop()
        await asyncio.sleep(2)
    
    async def _run_burst_load(self, env: Environment, config: LoadTestConfig):
        """Run burst load test."""
        runner = env.create_local_runner()
        
        # Alternate between high and low load
        burst_cycles = 4
        cycle_duration = config.duration_seconds / burst_cycles
        high_users = config.users
        low_users = config.users // 4
        
        for cycle in range(burst_cycles):
            # High load burst
            runner.start(high_users, spawn_rate=config.spawn_rate * 2)
            await asyncio.sleep(cycle_duration * 0.4)
            
            # Low load period
            runner.start(low_users, spawn_rate=config.spawn_rate)
            await asyncio.sleep(cycle_duration * 0.6)
        
        # Stop users
        runner.stop()
        await asyncio.sleep(2)
    
    async def _run_mixed_load(self, env: Environment, config: LoadTestConfig):
        """Run mixed load patterns."""
        # Combine ramp-up, steady, spike, and ramp-down
        runner = env.create_local_runner()
        
        phase_duration = config.duration_seconds / 4
        
        # Phase 1: Ramp-up
        for users in range(1, config.users // 2, max(1, config.users // 20)):
            runner.start(users, spawn_rate=config.spawn_rate)
            await asyncio.sleep(phase_duration / 10)
        
        # Phase 2: Steady
        runner.start(config.users // 2, spawn_rate=config.spawn_rate)
        await asyncio.sleep(phase_duration)
        
        # Phase 3: Spike
        runner.start(config.users, spawn_rate=config.users)
        await asyncio.sleep(phase_duration)
        
        # Phase 4: Ramp-down
        for users in range(config.users, 1, -max(1, config.users // 20)):
            runner.start(users, spawn_rate=config.spawn_rate)
            await asyncio.sleep(phase_duration / 10)
        
        # Stop users
        runner.stop()
        await asyncio.sleep(2)
    
    def _calculate_load_test_result(
        self,
        config: LoadTestConfig,
        duration: float,
        resource_stats: Dict[str, Any]
    ) -> LoadTestResult:
        """Calculate load test result metrics."""
        
        # Aggregate operation metrics
        successful_ops = self.metrics_collector.get('success_operations', [])
        failed_ops = self.metrics_collector.get('failure_operations', [])
        
        total_operations = len(successful_ops) + len(failed_ops)
        successful_operations = len(successful_ops)
        failed_operations = len(failed_ops)
        
        # Calculate latency metrics
        success_times = self.metrics_collector.get('success_times', [])
        
        if success_times:
            avg_latency = statistics.mean(success_times)
            p50_latency = np.percentile(success_times, 50)
            p95_latency = np.percentile(success_times, 95)
            p99_latency = np.percentile(success_times, 99)
            max_latency = max(success_times)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = max_latency = 0.0
        
        # Calculate rates
        operations_per_second = total_operations / duration if duration > 0 else 0
        error_rate_percent = (failed_operations / total_operations) * 100 if total_operations > 0 else 0
        
        # Extract timestamps and latencies for analysis
        timestamps = [datetime.fromtimestamp(op['timestamp']) for op in successful_ops]
        latencies = success_times
        
        return LoadTestResult(
            scenario_name=config.scenario.value,
            concurrent_users=config.users,
            duration_seconds=duration,
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            operations_per_second=operations_per_second,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            error_rate_percent=error_rate_percent,
            resource_usage=resource_stats,
            timestamps=timestamps,
            latencies=latencies
        )
    
    async def run_comprehensive_load_test_suite(self) -> List[LoadTestResult]:
        """Run comprehensive load testing suite."""
        logger.info("🚀 Starting comprehensive load test suite")
        
        # Define test configurations
        test_configs = [
            LoadTestConfig(
                scenario=LoadTestScenario.SINGLE_USER,
                users=1,
                spawn_rate=1,
                duration_seconds=60,
                load_pattern=LoadPattern.CONSTANT
            ),
            LoadTestConfig(
                scenario=LoadTestScenario.NORMAL_LOAD,
                users=10,
                spawn_rate=2,
                duration_seconds=120,
                load_pattern=LoadPattern.RAMP_UP
            ),
            LoadTestConfig(
                scenario=LoadTestScenario.HIGH_LOAD,
                users=50,
                spawn_rate=5,
                duration_seconds=180,
                load_pattern=LoadPattern.BURST
            ),
            LoadTestConfig(
                scenario=LoadTestScenario.SPIKE_TEST,
                users=100,
                spawn_rate=10,
                duration_seconds=120,
                load_pattern=LoadPattern.SPIKE
            ),
            LoadTestConfig(
                scenario=LoadTestScenario.STRESS_TEST,
                users=150,
                spawn_rate=15,
                duration_seconds=300,
                load_pattern=LoadPattern.MIXED
            )
        ]
        
        results = []
        
        for config in test_configs:
            try:
                # Clear metrics from previous test
                self.metrics_collector.clear()
                
                # Run load test
                result = await self.run_load_test(config)
                results.append(result)
                
                # Brief pause between tests
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Load test {config.scenario.value} failed: {e}")
        
        logger.info(f"✅ Comprehensive load test suite completed: {len(results)}/{len(test_configs)} tests successful")
        
        return results
    
    def analyze_performance_degradation(self, results: List[LoadTestResult]) -> Dict[str, Any]:
        """Analyze performance degradation across load levels."""
        if len(results) < 2:
            return {}
        
        # Sort results by concurrent users
        sorted_results = sorted(results, key=lambda r: r.concurrent_users)
        
        degradation_analysis = {
            'latency_degradation': [],
            'throughput_degradation': [],
            'error_rate_progression': [],
            'resource_usage_progression': []
        }
        
        baseline = sorted_results[0]
        
        for result in sorted_results[1:]:
            # Latency degradation
            latency_increase = (result.p95_latency_ms - baseline.p95_latency_ms) / baseline.p95_latency_ms
            degradation_analysis['latency_degradation'].append({
                'users': result.concurrent_users,
                'p95_latency_ms': result.p95_latency_ms,
                'degradation_factor': latency_increase
            })
            
            # Throughput degradation
            throughput_change = (result.operations_per_second - baseline.operations_per_second) / baseline.operations_per_second
            degradation_analysis['throughput_degradation'].append({
                'users': result.concurrent_users,
                'ops_per_sec': result.operations_per_second,
                'change_factor': throughput_change
            })
            
            # Error rate progression
            degradation_analysis['error_rate_progression'].append({
                'users': result.concurrent_users,
                'error_rate_percent': result.error_rate_percent
            })
            
            # Resource usage progression
            degradation_analysis['resource_usage_progression'].append({
                'users': result.concurrent_users,
                'avg_cpu_percent': result.resource_usage.get('avg_cpu_percent', 0),
                'max_memory_mb': result.resource_usage.get('max_memory_mb', 0)
            })
        
        return degradation_analysis
    
    def generate_load_test_report(self, results: List[LoadTestResult]) -> Dict[str, Any]:
        """Generate comprehensive load test report."""
        if not results:
            return {}
        
        # Performance summary
        best_performance = min(results, key=lambda r: r.p95_latency_ms)
        worst_performance = max(results, key=lambda r: r.p95_latency_ms)
        highest_throughput = max(results, key=lambda r: r.operations_per_second)
        
        # Target validation
        target_violations = []
        for result in results:
            if result.p95_latency_ms > PerformanceTarget.P95_SEARCH_LATENCY_MS.value:
                target_violations.append(f"P95 latency {result.p95_latency_ms:.1f}ms exceeds target in {result.scenario_name}")
            
            if result.error_rate_percent > PerformanceTarget.ERROR_RATE_PCT.value:
                target_violations.append(f"Error rate {result.error_rate_percent:.1f}% exceeds target in {result.scenario_name}")
        
        # Performance degradation analysis
        degradation_analysis = self.analyze_performance_degradation(results)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_scenarios': len(results),
            'performance_summary': {
                'best_performance': {
                    'scenario': best_performance.scenario_name,
                    'p95_latency_ms': best_performance.p95_latency_ms,
                    'ops_per_sec': best_performance.operations_per_second
                },
                'worst_performance': {
                    'scenario': worst_performance.scenario_name,
                    'p95_latency_ms': worst_performance.p95_latency_ms,
                    'ops_per_sec': worst_performance.operations_per_second
                },
                'highest_throughput': {
                    'scenario': highest_throughput.scenario_name,
                    'ops_per_sec': highest_throughput.operations_per_second,
                    'users': highest_throughput.concurrent_users
                }
            },
            'target_validation': {
                'violations': target_violations,
                'targets_met': len(target_violations) == 0
            },
            'degradation_analysis': degradation_analysis,
            'detailed_results': [
                {
                    'scenario': r.scenario_name,
                    'users': r.concurrent_users,
                    'duration_seconds': r.duration_seconds,
                    'operations_per_second': r.operations_per_second,
                    'p95_latency_ms': r.p95_latency_ms,
                    'error_rate_percent': r.error_rate_percent,
                    'max_memory_mb': r.resource_usage.get('max_memory_mb', 0)
                }
                for r in results
            ]
        }