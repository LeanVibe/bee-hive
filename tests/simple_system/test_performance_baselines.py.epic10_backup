"""
Performance Baselines - Component Isolation Tests
=================================================

Tests performance characteristics of core components in isolation.
This establishes baseline performance metrics and validates that
components meet performance requirements without external dependencies.

Testing Strategy:
- Mock all external dependencies for consistent baseline measurements
- Test component performance under controlled conditions
- Validate memory usage, execution time, and throughput
- Establish performance regression detection thresholds
- Test scalability characteristics of core algorithms
"""

import asyncio
import time
import uuid
import pytest
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from app.core.orchestrator import AgentOrchestrator
from app.services.semantic_memory_service import SemanticMemoryService
from app.core.context_compression_engine import ContextCompressionEngine
from app.api.dashboard_websockets import WebSocketManager
from app.core.intelligent_task_router import IntelligentTaskRouter


@pytest.mark.isolation
@pytest.mark.benchmark
@pytest.mark.performance
class TestOrchestratorPerformanceBaselines:
    """Test orchestrator performance baselines in isolation."""
    
    @pytest.fixture
    async def performance_orchestrator(
        self,
        mock_database_session,
        mock_redis_streams,
        mock_anthropic_client,
        isolated_test_environment
    ):
        """Create orchestrator optimized for performance testing."""
        
        with patch('app.core.orchestrator.get_database_session', return_value=mock_database_session), \
             patch('app.core.orchestrator.get_redis_client', return_value=mock_redis_streams), \
             patch('app.core.orchestrator.get_anthropic_client', return_value=mock_anthropic_client):
            
            orchestrator = AgentOrchestrator()
            await orchestrator.initialize()
            
            yield orchestrator
            
            await orchestrator.shutdown()
    
    async def test_agent_registration_performance_baseline(
        self,
        performance_orchestrator,
        isolated_agent_config
    ):
        """Test agent registration performance baseline."""
        orchestrator = performance_orchestrator
        
        # Performance targets
        MAX_REGISTRATION_TIME = 0.100  # 100ms per registration
        MAX_BATCH_TIME = 2.0           # 2 seconds for 100 registrations
        MAX_MEMORY_INCREASE = 50       # 50MB max memory increase
        
        # Measure baseline memory
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test single registration performance
        agent_config = isolated_agent_config()
        
        start_time = time.time()
        result = await orchestrator.register_agent(**agent_config)
        single_registration_time = time.time() - start_time
        
        assert result["success"] is True
        assert single_registration_time < MAX_REGISTRATION_TIME, \
            f"Single registration took {single_registration_time:.3f}s, exceeds {MAX_REGISTRATION_TIME}s limit"
        
        # Test batch registration performance
        batch_configs = [
            isolated_agent_config(name=f"batch-agent-{i}")
            for i in range(100)
        ]
        
        start_time = time.time()
        registration_tasks = [
            orchestrator.register_agent(**config)
            for config in batch_configs
        ]
        results = await asyncio.gather(*registration_tasks)
        batch_time = time.time() - start_time
        
        # Verify all registrations succeeded
        successful_registrations = sum(1 for result in results if result["success"])
        assert successful_registrations == 100
        
        # Verify batch performance
        assert batch_time < MAX_BATCH_TIME, \
            f"Batch registration took {batch_time:.3f}s, exceeds {MAX_BATCH_TIME}s limit"
        
        # Verify memory usage
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < MAX_MEMORY_INCREASE, \
            f"Memory increased by {memory_increase:.1f}MB, exceeds {MAX_MEMORY_INCREASE}MB limit"
        
        # Performance metrics
        avg_registration_time = batch_time / 100
        registrations_per_second = 100 / batch_time
        
        print(f"Performance Metrics:")
        print(f"  Single registration: {single_registration_time*1000:.1f}ms")
        print(f"  Batch registration: {batch_time:.2f}s")
        print(f"  Average per registration: {avg_registration_time*1000:.1f}ms")
        print(f"  Registrations/second: {registrations_per_second:.1f}")
        print(f"  Memory increase: {memory_increase:.1f}MB")
    
    async def test_task_scheduling_performance_baseline(
        self,
        performance_orchestrator,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test task scheduling performance baseline."""
        orchestrator = performance_orchestrator
        
        # Performance targets
        MAX_SCHEDULING_TIME = 0.050   # 50ms per task
        MAX_BATCH_SCHEDULING = 1.0    # 1 second for 100 tasks
        MIN_THROUGHPUT = 200          # 200 tasks/second minimum
        
        # Register agents for task assignment
        agents = []
        for i in range(20):
            agent_config = isolated_agent_config(
                name=f"worker-agent-{i}",
                capabilities=["python", "testing", "api", "database"]
            )
            result = await orchestrator.register_agent(**agent_config)
            agents.append(result["agent_id"])
        
        # Test single task scheduling
        task_config = isolated_task_config(
            title="Performance Test Task",
            required_capabilities=["python"]
        )
        
        start_time = time.time()
        result = await orchestrator.submit_task(**task_config)
        single_scheduling_time = time.time() - start_time
        
        assert result["success"] is True
        assert single_scheduling_time < MAX_SCHEDULING_TIME, \
            f"Single task scheduling took {single_scheduling_time:.3f}s, exceeds {MAX_SCHEDULING_TIME}s limit"
        
        # Test batch task scheduling
        batch_tasks = [
            isolated_task_config(
                title=f"Batch Task {i}",
                required_capabilities=["python", "testing"]
            )
            for i in range(100)
        ]
        
        start_time = time.time()
        scheduling_tasks = [
            orchestrator.submit_task(**task_config)
            for task_config in batch_tasks
        ]
        results = await asyncio.gather(*scheduling_tasks)
        batch_scheduling_time = time.time() - start_time
        
        # Verify all tasks were scheduled
        successful_submissions = sum(1 for result in results if result["success"])
        assert successful_submissions == 100
        
        # Verify batch performance
        assert batch_scheduling_time < MAX_BATCH_SCHEDULING, \
            f"Batch scheduling took {batch_scheduling_time:.3f}s, exceeds {MAX_BATCH_SCHEDULING}s limit"
        
        # Verify throughput
        throughput = 100 / batch_scheduling_time
        assert throughput >= MIN_THROUGHPUT, \
            f"Throughput {throughput:.1f} tasks/s, below {MIN_THROUGHPUT} tasks/s minimum"
        
        # Performance metrics
        avg_scheduling_time = batch_scheduling_time / 100
        
        print(f"Task Scheduling Performance:")
        print(f"  Single task: {single_scheduling_time*1000:.1f}ms")
        print(f"  Batch scheduling: {batch_scheduling_time:.2f}s")
        print(f"  Average per task: {avg_scheduling_time*1000:.1f}ms")
        print(f"  Throughput: {throughput:.1f} tasks/second")
    
    async def test_concurrent_operations_performance_baseline(
        self,
        performance_orchestrator,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test concurrent operations performance baseline."""
        orchestrator = performance_orchestrator
        
        # Performance targets
        MAX_CONCURRENT_TIME = 3.0      # 3 seconds for mixed operations
        MIN_CONCURRENT_THROUGHPUT = 150 # 150 ops/second minimum
        
        # Prepare mixed operations
        mixed_operations = []
        
        # Agent registrations
        for i in range(50):
            agent_config = isolated_agent_config(name=f"concurrent-agent-{i}")
            mixed_operations.append(("register_agent", agent_config))
        
        # Task submissions
        for i in range(100):
            task_config = isolated_task_config(title=f"Concurrent Task {i}")
            mixed_operations.append(("submit_task", task_config))
        
        # Status queries (simulate dashboard requests)
        for i in range(50):
            mixed_operations.append(("get_system_status", {}))
        
        # Execute all operations concurrently
        async def execute_operation(op_type, config):
            if op_type == "register_agent":
                return await orchestrator.register_agent(**config)
            elif op_type == "submit_task":
                return await orchestrator.submit_task(**config)
            elif op_type == "get_system_status":
                return await orchestrator.get_system_status()
        
        start_time = time.time()
        concurrent_tasks = [
            execute_operation(op_type, config)
            for op_type, config in mixed_operations
        ]
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        
        # Verify performance
        assert concurrent_time < MAX_CONCURRENT_TIME, \
            f"Concurrent operations took {concurrent_time:.3f}s, exceeds {MAX_CONCURRENT_TIME}s limit"
        
        # Verify throughput
        total_operations = len(mixed_operations)
        concurrent_throughput = total_operations / concurrent_time
        assert concurrent_throughput >= MIN_CONCURRENT_THROUGHPUT, \
            f"Concurrent throughput {concurrent_throughput:.1f} ops/s, below {MIN_CONCURRENT_THROUGHPUT} ops/s minimum"
        
        # Verify success rate
        successful_operations = sum(1 for result in results if isinstance(result, dict) and not isinstance(result, Exception))
        success_rate = successful_operations / total_operations
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%}, below 95% minimum"
        
        print(f"Concurrent Operations Performance:")
        print(f"  Total time: {concurrent_time:.2f}s")
        print(f"  Operations: {total_operations}")
        print(f"  Throughput: {concurrent_throughput:.1f} ops/second")
        print(f"  Success rate: {success_rate:.2%}")


@pytest.mark.isolation
@pytest.mark.benchmark
@pytest.mark.performance
class TestSemanticMemoryPerformanceBaselines:
    """Test semantic memory performance baselines in isolation."""
    
    @pytest.fixture
    async def performance_memory_service(
        self,
        mock_database_session,
        mock_vector_search,
        isolated_test_environment
    ):
        """Create semantic memory service optimized for performance testing."""
        
        # Mock high-performance embedding service
        mock_embeddings = AsyncMock()
        mock_embeddings.embed_text = AsyncMock(
            side_effect=lambda text: [0.1] * 1536  # Fast mock embedding
        )
        
        with patch('app.services.semantic_memory_service.get_database_session', return_value=mock_database_session), \
             patch('app.services.semantic_memory_service.get_embedding_service', return_value=mock_embeddings):
            
            memory_service = SemanticMemoryService()
            await memory_service.initialize()
            
            yield memory_service
            
            await memory_service.shutdown()
    
    async def test_memory_storage_performance_baseline(
        self,
        performance_memory_service,
        isolated_agent_config
    ):
        """Test memory storage performance baseline."""
        memory_service = performance_memory_service
        agent_config = isolated_agent_config()
        
        # Performance targets
        MAX_STORAGE_TIME = 0.020       # 20ms per memory
        MAX_BATCH_STORAGE = 2.0        # 2 seconds for 1000 memories
        MIN_STORAGE_THROUGHPUT = 500   # 500 memories/second minimum
        
        # Test single memory storage
        memory_data = {
            "content": "Test memory for performance baseline",
            "type": "test",
            "agent_id": agent_config["id"],
            "context": {"performance": "test"}
        }
        
        start_time = time.time()
        result = await memory_service.store_memory(**memory_data)
        single_storage_time = time.time() - start_time
        
        assert result["success"] is True
        assert single_storage_time < MAX_STORAGE_TIME, \
            f"Single memory storage took {single_storage_time:.3f}s, exceeds {MAX_STORAGE_TIME}s limit"
        
        # Test batch memory storage
        batch_memories = [
            {
                "content": f"Performance test memory {i}: Some content to store",
                "type": "performance_test",
                "agent_id": agent_config["id"],
                "context": {"batch": i, "test": "performance"}
            }
            for i in range(1000)
        ]
        
        start_time = time.time()
        storage_tasks = [
            memory_service.store_memory(**memory)
            for memory in batch_memories
        ]
        results = await asyncio.gather(*storage_tasks)
        batch_storage_time = time.time() - start_time
        
        # Verify batch performance
        successful_stores = sum(1 for result in results if result["success"])
        assert successful_stores == 1000
        
        assert batch_storage_time < MAX_BATCH_STORAGE, \
            f"Batch storage took {batch_storage_time:.3f}s, exceeds {MAX_BATCH_STORAGE}s limit"
        
        # Verify throughput
        storage_throughput = 1000 / batch_storage_time
        assert storage_throughput >= MIN_STORAGE_THROUGHPUT, \
            f"Storage throughput {storage_throughput:.1f} memories/s, below {MIN_STORAGE_THROUGHPUT} minimum"
        
        print(f"Memory Storage Performance:")
        print(f"  Single storage: {single_storage_time*1000:.1f}ms")
        print(f"  Batch storage: {batch_storage_time:.2f}s")
        print(f"  Throughput: {storage_throughput:.1f} memories/second")
    
    async def test_memory_search_performance_baseline(
        self,
        performance_memory_service,
        isolated_agent_config
    ):
        """Test memory search performance baseline."""
        memory_service = performance_memory_service
        agent_config = isolated_agent_config()
        
        # Performance targets
        MAX_SEARCH_TIME = 0.100        # 100ms per search
        MAX_COMPLEX_SEARCH = 0.200     # 200ms for complex searches
        MIN_SEARCH_THROUGHPUT = 50     # 50 searches/second minimum
        
        # Populate with test memories for searching
        test_memories = [
            f"Python programming concept {i}: variables, functions, classes",
            f"Web development topic {i}: HTML, CSS, JavaScript, frameworks",
            f"Database design principle {i}: normalization, indexing, queries",
            f"Testing methodology {i}: unit tests, integration tests, TDD"
        ]
        
        for i in range(250):  # 1000 total memories
            for template in test_memories:
                await memory_service.store_memory(
                    content=template,
                    type="knowledge",
                    agent_id=agent_config["id"],
                    context={"category": template.split()[0].lower()}
                )
        
        # Test single search performance
        start_time = time.time()
        results = await memory_service.search_memories(
            query="Python programming variables",
            limit=10
        )
        single_search_time = time.time() - start_time
        
        assert len(results) > 0
        assert single_search_time < MAX_SEARCH_TIME, \
            f"Single search took {single_search_time:.3f}s, exceeds {MAX_SEARCH_TIME}s limit"
        
        # Test complex search performance
        start_time = time.time()
        complex_results = await memory_service.search_memories(
            query="web development frameworks and database design",
            limit=50,
            filters={"type": "knowledge"},
            semantic_boost=True
        )
        complex_search_time = time.time() - start_time
        
        assert len(complex_results) > 0
        assert complex_search_time < MAX_COMPLEX_SEARCH, \
            f"Complex search took {complex_search_time:.3f}s, exceeds {MAX_COMPLEX_SEARCH}s limit"
        
        # Test batch search performance
        search_queries = [
            "Python functions and classes",
            "JavaScript frameworks",
            "Database indexing strategies",
            "Unit testing best practices",
            "Web security principles"
        ] * 10  # 50 searches
        
        start_time = time.time()
        search_tasks = [
            memory_service.search_memories(query=query, limit=5)
            for query in search_queries
        ]
        all_results = await asyncio.gather(*search_tasks)
        batch_search_time = time.time() - start_time
        
        # Verify throughput
        search_throughput = 50 / batch_search_time
        assert search_throughput >= MIN_SEARCH_THROUGHPUT, \
            f"Search throughput {search_throughput:.1f} searches/s, below {MIN_SEARCH_THROUGHPUT} minimum"
        
        print(f"Memory Search Performance:")
        print(f"  Single search: {single_search_time*1000:.1f}ms")
        print(f"  Complex search: {complex_search_time*1000:.1f}ms")
        print(f"  Batch search: {batch_search_time:.2f}s")
        print(f"  Throughput: {search_throughput:.1f} searches/second")


@pytest.mark.isolation
@pytest.mark.benchmark
@pytest.mark.performance
class TestContextCompressionPerformanceBaselines:
    """Test context compression performance baselines in isolation."""
    
    @pytest.fixture
    async def performance_compression_engine(self, isolated_test_environment):
        """Create context compression engine optimized for performance testing."""
        
        # Mock fast embedding service
        mock_embeddings = AsyncMock()
        mock_embeddings.embed_text = AsyncMock(return_value=[0.1] * 768)
        
        with patch('app.core.context_compression_engine.get_embedding_service', return_value=mock_embeddings):
            engine = ContextCompressionEngine()
            await engine.initialize()
            
            yield engine
            
            await engine.shutdown()
    
    async def test_compression_algorithm_performance_baseline(
        self,
        performance_compression_engine
    ):
        """Test compression algorithm performance baseline."""
        engine = performance_compression_engine
        
        # Performance targets
        MAX_SMALL_COMPRESSION = 0.100   # 100ms for small context
        MAX_LARGE_COMPRESSION = 2.0     # 2 seconds for large context
        MIN_COMPRESSION_RATIO = 0.3     # At least 30% compression
        
        # Test small context compression
        small_context = {
            "conversation": [
                {"role": "user", "content": f"Message {i}: Testing compression performance"}
                for i in range(50)
            ],
            "code_snippets": [
                {"language": "python", "code": f"def test_function_{i}():\n    return {i}"}
                for i in range(20)
            ]
        }
        
        start_time = time.time()
        small_result = await engine.compress_context(
            context=small_context,
            target_ratio=0.5
        )
        small_compression_time = time.time() - start_time
        
        assert small_result["success"] is True
        assert small_compression_time < MAX_SMALL_COMPRESSION, \
            f"Small context compression took {small_compression_time:.3f}s, exceeds {MAX_SMALL_COMPRESSION}s limit"
        
        # Test large context compression
        large_context = {
            "conversation": [
                {"role": "user", "content": f"Large context message {i}: " + "x" * 200}
                for i in range(500)
            ],
            "documentation": [
                {"title": f"Doc {i}", "content": f"Documentation content {i}: " + "y" * 1000}
                for i in range(100)
            ],
            "code_files": [
                {"filename": f"file_{i}.py", "content": f"# File {i}\n" + "def func():\n    pass\n" * 50}
                for i in range(50)
            ]
        }
        
        start_time = time.time()
        large_result = await engine.compress_context(
            context=large_context,
            target_ratio=0.3
        )
        large_compression_time = time.time() - start_time
        
        assert large_result["success"] is True
        assert large_compression_time < MAX_LARGE_COMPRESSION, \
            f"Large context compression took {large_compression_time:.3f}s, exceeds {MAX_LARGE_COMPRESSION}s limit"
        
        # Verify compression effectiveness
        assert large_result["compression_ratio"] >= MIN_COMPRESSION_RATIO, \
            f"Compression ratio {large_result['compression_ratio']:.2f}, below {MIN_COMPRESSION_RATIO} minimum"
        
        # Calculate throughput metrics
        small_size = len(str(small_context))
        large_size = len(str(large_context))
        
        small_throughput = small_size / small_compression_time / 1024  # KB/s
        large_throughput = large_size / large_compression_time / 1024   # KB/s
        
        print(f"Context Compression Performance:")
        print(f"  Small context: {small_compression_time*1000:.1f}ms")
        print(f"  Large context: {large_compression_time:.2f}s")
        print(f"  Small throughput: {small_throughput:.1f} KB/s")
        print(f"  Large throughput: {large_throughput:.1f} KB/s")
        print(f"  Compression ratio: {large_result['compression_ratio']:.2f}")


@pytest.mark.isolation
@pytest.mark.benchmark
@pytest.mark.performance
class TestSystemScalabilityBaselines:
    """Test system scalability baselines in isolation."""
    
    async def test_memory_usage_scaling_baseline(
        self,
        isolated_test_environment
    ):
        """Test memory usage scaling characteristics."""
        
        # Performance targets
        MAX_MEMORY_PER_AGENT = 5.0      # 5MB per agent maximum
        MAX_MEMORY_PER_TASK = 1.0       # 1MB per task maximum
        MAX_TOTAL_MEMORY = 500.0        # 500MB total maximum
        
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate scaling with multiple components
        mock_agents = []
        mock_tasks = []
        
        # Create mock agents (simulating memory usage)
        for i in range(100):
            agent_data = {
                "id": str(uuid.uuid4()),
                "name": f"scale-agent-{i}",
                "capabilities": [f"cap_{j}" for j in range(10)],
                "history": [f"task_{k}" for k in range(50)],
                "context": {"data": "x" * 1000}  # 1KB of context per agent
            }
            mock_agents.append(agent_data)
        
        gc.collect()
        agent_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_agent = (agent_memory - initial_memory) / 100
        
        assert memory_per_agent < MAX_MEMORY_PER_AGENT, \
            f"Memory per agent {memory_per_agent:.2f}MB, exceeds {MAX_MEMORY_PER_AGENT}MB limit"
        
        # Create mock tasks
        for i in range(500):
            task_data = {
                "id": str(uuid.uuid4()),
                "title": f"scale-task-{i}",
                "description": "x" * 500,  # 500 bytes per task
                "context": {"data": "y" * 500},
                "history": [f"event_{j}" for j in range(10)]
            }
            mock_tasks.append(task_data)
        
        gc.collect()
        task_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_task = (task_memory - agent_memory) / 500
        
        assert memory_per_task < MAX_MEMORY_PER_TASK, \
            f"Memory per task {memory_per_task:.2f}MB, exceeds {MAX_MEMORY_PER_TASK}MB limit"
        
        # Verify total memory usage
        total_memory = task_memory - initial_memory
        assert total_memory < MAX_TOTAL_MEMORY, \
            f"Total memory usage {total_memory:.1f}MB, exceeds {MAX_TOTAL_MEMORY}MB limit"
        
        print(f"Memory Scaling Performance:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Memory per agent: {memory_per_agent:.2f}MB")
        print(f"  Memory per task: {memory_per_task:.2f}MB")
        print(f"  Total memory: {total_memory:.1f}MB")
        print(f"  Agents: 100, Tasks: 500")
    
    async def test_concurrent_load_scaling_baseline(self, isolated_test_environment):
        """Test concurrent load scaling characteristics."""
        
        # Performance targets
        MIN_CONCURRENT_OPS = 100        # 100 concurrent ops minimum
        MAX_RESPONSE_TIME = 1.0         # 1 second max response time
        MIN_SUCCESS_RATE = 0.95         # 95% success rate minimum
        
        async def mock_operation(operation_id: int):
            """Mock operation that simulates real work."""
            # Simulate some CPU work
            await asyncio.sleep(0.01 + (operation_id % 10) * 0.001)
            
            # Simulate some memory allocation
            data = [i for i in range(100)]
            
            return {
                "operation_id": operation_id,
                "success": True,
                "data_size": len(data)
            }
        
        # Test increasing levels of concurrency
        concurrency_levels = [10, 25, 50, 100, 200]
        performance_results = []
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # Execute concurrent operations
            tasks = [
                mock_operation(i)
                for i in range(concurrency)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            successful_ops = sum(1 for result in results if isinstance(result, dict) and result.get("success"))
            success_rate = successful_ops / concurrency
            throughput = concurrency / total_time
            avg_response_time = total_time / concurrency
            
            performance_results.append({
                "concurrency": concurrency,
                "total_time": total_time,
                "success_rate": success_rate,
                "throughput": throughput,
                "avg_response_time": avg_response_time
            })
            
            # Verify performance criteria
            if concurrency >= MIN_CONCURRENT_OPS:
                assert success_rate >= MIN_SUCCESS_RATE, \
                    f"Success rate {success_rate:.2%} below {MIN_SUCCESS_RATE:.0%} at concurrency {concurrency}"
                
                assert avg_response_time <= MAX_RESPONSE_TIME, \
                    f"Average response time {avg_response_time:.3f}s exceeds {MAX_RESPONSE_TIME}s at concurrency {concurrency}"
        
        # Print scaling characteristics
        print(f"Concurrent Load Scaling Performance:")
        for result in performance_results:
            print(f"  Concurrency {result['concurrency']:3d}: "
                  f"{result['throughput']:6.1f} ops/s, "
                  f"{result['avg_response_time']*1000:5.1f}ms avg, "
                  f"{result['success_rate']:5.1%} success")
        
        # Verify system can handle minimum concurrent load
        min_load_result = next(r for r in performance_results if r["concurrency"] == MIN_CONCURRENT_OPS)
        assert min_load_result["success_rate"] >= MIN_SUCCESS_RATE
        assert min_load_result["avg_response_time"] <= MAX_RESPONSE_TIME