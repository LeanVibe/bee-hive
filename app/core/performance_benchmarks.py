"""
Performance Benchmark Suite for Context Engine.

Validates production KPI targets:
- <50ms semantic search response time
- 60-80% token reduction through compression  
- >90% context retrieval precision
- Support 50+ concurrent agents with <100ms latency
- <1GB storage per 10,000 contexts
"""

import asyncio
import time
import uuid
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import json
import psutil
import random

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.context import Context, ContextType
from ..schemas.context import ContextCreate, ContextSearchRequest
from ..core.context_manager import ContextManager
from ..core.embeddings import EmbeddingService
from ..core.context_compression import ContextCompressor, CompressionLevel
from ..core.vector_search import VectorSearchEngine


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    operation: str
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput: float = 0.0
    memory_usage_mb: float = 0.0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_stats(self):
        """Calculate performance statistics."""
        if self.response_times:
            self.average_response_time = statistics.mean(self.response_times)
            self.p95_response_time = statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
            self.p99_response_time = statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile
            self.throughput = len(self.response_times) / sum(self.response_times) if sum(self.response_times) > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation": self.operation,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "average_response_time_ms": self.average_response_time * 1000,
            "p95_response_time_ms": self.p95_response_time * 1000,
            "p99_response_time_ms": self.p99_response_time * 1000,
            "throughput_ops_per_sec": self.throughput,
            "memory_usage_mb": self.memory_usage_mb,
            "additional_metrics": self.additional_metrics
        }


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance validation suite for Context Engine.
    
    Validates all production KPI targets with detailed reporting.
    """
    
    def __init__(
        self,
        context_manager: ContextManager,
        db_session: AsyncSession
    ):
        """
        Initialize benchmark suite.
        
        Args:
            context_manager: Context manager instance
            db_session: Database session for direct queries
        """
        self.context_manager = context_manager
        self.db = db_session
        self.metrics: Dict[str, PerformanceMetrics] = {}
        
        # Test data for benchmarks
        self.test_contexts: List[Context] = []
        self.test_queries = [
            "Redis cluster configuration best practices",
            "PostgreSQL performance optimization techniques",
            "Agent communication error debugging",
            "Database backup and recovery procedures",
            "API security implementation patterns",
            "Microservices deployment strategies",
            "Load balancing configuration",
            "Memory optimization for Python applications",
            "Docker container orchestration",
            "Monitoring and alerting setup"
        ]
    
    async def run_full_benchmark_suite(
        self,
        num_test_contexts: int = 1000,
        concurrent_agents: int = 50
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite covering all KPI targets.
        
        Args:
            num_test_contexts: Number of test contexts to create
            concurrent_agents: Number of concurrent agents to simulate
            
        Returns:
            Complete benchmark results with pass/fail status
        """
        logger.info(f"Starting full benchmark suite with {num_test_contexts} contexts, {concurrent_agents} concurrent agents")
        
        start_time = time.time()
        
        try:
            # Setup test data
            await self._setup_test_data(num_test_contexts)
            
            # Run individual benchmarks
            await self._benchmark_search_performance()
            await self._benchmark_token_reduction()
            await self._benchmark_retrieval_precision()
            await self._benchmark_concurrent_access(concurrent_agents)
            await self._benchmark_storage_efficiency()
            await self._benchmark_compression_performance()
            await self._benchmark_database_operations()
            
            # Generate comprehensive report
            report = await self._generate_benchmark_report()
            
            total_time = time.time() - start_time
            report["benchmark_execution_time_seconds"] = total_time
            
            logger.info(f"Benchmark suite completed in {total_time:.2f} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
            raise
        finally:
            # Cleanup test data
            await self._cleanup_test_data()
    
    async def _benchmark_search_performance(self) -> None:
        """
        Benchmark semantic search performance.
        Target: <50ms response time
        """
        logger.info("Benchmarking search performance (target: <50ms)")
        
        metrics = PerformanceMetrics("semantic_search")
        
        for query in self.test_queries:
            # Warm up
            await self.context_manager.search_contexts(
                ContextSearchRequest(query=query, limit=10)
            )
            
            # Measure performance
            for _ in range(10):  # 10 iterations per query
                start_time = time.perf_counter()
                
                try:
                    results = await self.context_manager.search_contexts(
                        ContextSearchRequest(query=query, limit=10)
                    )
                    
                    response_time = time.perf_counter() - start_time
                    metrics.response_times.append(response_time)
                    metrics.success_count += 1
                    
                    # Record additional metrics
                    if "results_returned" not in metrics.additional_metrics:
                        metrics.additional_metrics["results_returned"] = []
                    metrics.additional_metrics["results_returned"].append(len(results))
                    
                except Exception as e:
                    logger.warning(f"Search failed: {e}")
                    metrics.failure_count += 1
        
        # Measure memory usage
        process = psutil.Process()
        metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        metrics.calculate_stats()
        self.metrics["search_performance"] = metrics
        
        # Validate target
        target_ms = 50
        actual_ms = metrics.average_response_time * 1000
        
        logger.info(
            f"Search performance: {actual_ms:.1f}ms avg (target: <{target_ms}ms) - "
            f"{'PASS' if actual_ms < target_ms else 'FAIL'}"
        )
    
    async def _benchmark_token_reduction(self) -> None:
        """
        Benchmark context compression effectiveness.
        Target: 60-80% token reduction
        """
        logger.info("Benchmarking token reduction (target: 60-80%)")
        
        metrics = PerformanceMetrics("token_reduction")
        
        # Create long test contexts for compression
        long_contexts = []
        for i in range(20):
            content = self._generate_long_content(5000)  # ~5000 characters
            context_data = ContextCreate(
                title=f"Long Test Context {i}",
                content=content,
                context_type=ContextType.DOCUMENTATION,
                agent_id=uuid.uuid4(),
                importance_score=0.8
            )
            long_contexts.append(context_data)
        
        total_original_tokens = 0
        total_compressed_tokens = 0
        compression_times = []
        
        for context_data in long_contexts:
            start_time = time.perf_counter()
            
            try:
                # Store context with compression
                context = await self.context_manager.store_context(
                    context_data=context_data,
                    auto_compress=True
                )
                
                compression_time = time.perf_counter() - start_time
                compression_times.append(compression_time)
                
                # Estimate token counts (rough approximation: 1 token ≈ 4 characters)
                original_tokens = len(context_data.content) // 4
                compressed_tokens = len(context.consolidation_summary or context.content) // 4
                
                total_original_tokens += original_tokens
                total_compressed_tokens += compressed_tokens
                
                metrics.success_count += 1
                
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
                metrics.failure_count += 1
        
        # Calculate token reduction
        if total_original_tokens > 0:
            reduction_ratio = (total_original_tokens - total_compressed_tokens) / total_original_tokens
            metrics.additional_metrics["token_reduction_percentage"] = reduction_ratio * 100
            metrics.additional_metrics["original_tokens"] = total_original_tokens
            metrics.additional_metrics["compressed_tokens"] = total_compressed_tokens
        
        metrics.response_times = compression_times
        metrics.calculate_stats()
        self.metrics["token_reduction"] = metrics
        
        # Validate target
        target_min, target_max = 60, 80
        actual_reduction = metrics.additional_metrics.get("token_reduction_percentage", 0)
        
        logger.info(
            f"Token reduction: {actual_reduction:.1f}% (target: {target_min}-{target_max}%) - "
            f"{'PASS' if target_min <= actual_reduction <= target_max else 'FAIL'}"
        )
    
    async def _benchmark_retrieval_precision(self) -> None:
        """
        Benchmark context retrieval precision.
        Target: >90% relevant context retrieval precision
        """
        logger.info("Benchmarking retrieval precision (target: >90%)")
        
        metrics = PerformanceMetrics("retrieval_precision")
        
        # Create test contexts with known relationships
        relevant_contexts = []
        irrelevant_contexts = []
        
        # Create relevant contexts (Redis-related)
        redis_topics = [
            "Redis cluster configuration for high availability",
            "Redis memory optimization and persistence settings",
            "Redis security best practices and authentication",
            "Redis monitoring and performance tuning",
            "Redis backup and disaster recovery procedures"
        ]
        
        for topic in redis_topics:
            context_data = ContextCreate(
                title=topic,
                content=f"Detailed guide about {topic.lower()}. This covers implementation details, best practices, and troubleshooting tips.",
                context_type=ContextType.DOCUMENTATION,
                agent_id=uuid.uuid4(),
                importance_score=0.9
            )
            context = await self.context_manager.store_context(context_data)
            relevant_contexts.append(context)
        
        # Create irrelevant contexts (unrelated topics)
        unrelated_topics = [
            "Frontend React component development patterns",
            "Mobile app deployment to app stores",
            "Machine learning model training workflows",
            "Marketing campaign optimization strategies",
            "Legal compliance for data protection"
        ]
        
        for topic in unrelated_topics:
            context_data = ContextCreate(
                title=topic,
                content=f"Comprehensive information about {topic.lower()}. Includes strategies, tools, and implementation guidelines.",
                context_type=ContextType.DOCUMENTATION,
                agent_id=uuid.uuid4(),
                importance_score=0.7
            )
            context = await self.context_manager.store_context(context_data)
            irrelevant_contexts.append(context)
        
        # Test precision with Redis-related queries
        redis_queries = [
            "Redis cluster setup and configuration",
            "How to optimize Redis performance",
            "Redis security configuration guide",
            "Redis backup and recovery methods"
        ]
        
        total_precision_scores = []
        
        for query in redis_queries:
            try:
                results = await self.context_manager.search_contexts(
                    ContextSearchRequest(query=query, limit=10, min_relevance=0.5)
                )
                
                # Calculate precision (relevant results / total results)
                relevant_count = 0
                for match in results:
                    # Check if result is from our relevant contexts
                    if any(match.context.id == ctx.id for ctx in relevant_contexts):
                        relevant_count += 1
                
                precision = relevant_count / len(results) if results else 0
                total_precision_scores.append(precision)
                metrics.success_count += 1
                
            except Exception as e:
                logger.warning(f"Precision test failed: {e}")
                metrics.failure_count += 1
        
        # Calculate overall precision
        if total_precision_scores:
            avg_precision = statistics.mean(total_precision_scores)
            metrics.additional_metrics["average_precision"] = avg_precision
            metrics.additional_metrics["precision_scores"] = total_precision_scores
        
        self.metrics["retrieval_precision"] = metrics
        
        # Validate target
        target_precision = 0.9  # 90%
        actual_precision = metrics.additional_metrics.get("average_precision", 0)
        
        logger.info(
            f"Retrieval precision: {actual_precision:.1%} (target: >{target_precision:.0%}) - "
            f"{'PASS' if actual_precision > target_precision else 'FAIL'}"
        )
    
    async def _benchmark_concurrent_access(self, num_agents: int) -> None:
        """
        Benchmark concurrent agent access.
        Target: Support 50+ agents with <100ms latency
        """
        logger.info(f"Benchmarking concurrent access (target: {num_agents}+ agents <100ms)")
        
        metrics = PerformanceMetrics("concurrent_access")
        
        async def simulate_agent_activity(agent_id: uuid.UUID) -> List[float]:
            """Simulate typical agent search activity."""
            response_times = []
            
            for _ in range(5):  # 5 operations per agent
                query = random.choice(self.test_queries)
                start_time = time.perf_counter()
                
                try:
                    await self.context_manager.search_contexts(
                        ContextSearchRequest(query=query, limit=5)
                    )
                    response_time = time.perf_counter() - start_time
                    response_times.append(response_time)
                    
                except Exception as e:
                    logger.warning(f"Concurrent search failed: {e}")
                
                # Small delay between operations
                await asyncio.sleep(0.01)
            
            return response_times
        
        # Create agent tasks
        agent_tasks = []
        agent_ids = [uuid.uuid4() for _ in range(num_agents)]
        
        start_time = time.perf_counter()
        
        # Run all agent tasks concurrently
        for agent_id in agent_ids:
            task = asyncio.create_task(simulate_agent_activity(agent_id))
            agent_tasks.append(task)
        
        # Wait for all agents to complete
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        total_time = time.perf_counter() - start_time
        
        # Collect all response times
        all_response_times = []
        successful_agents = 0
        
        for result in results:
            if isinstance(result, list):
                all_response_times.extend(result)
                successful_agents += 1
            else:
                metrics.failure_count += 1
        
        metrics.response_times = all_response_times
        metrics.success_count = successful_agents
        metrics.additional_metrics["total_operations"] = len(all_response_times)
        metrics.additional_metrics["concurrent_agents"] = num_agents
        metrics.additional_metrics["total_execution_time"] = total_time
        
        metrics.calculate_stats()
        self.metrics["concurrent_access"] = metrics
        
        # Validate target
        target_latency_ms = 100
        actual_latency_ms = metrics.average_response_time * 1000
        
        logger.info(
            f"Concurrent access: {num_agents} agents, {actual_latency_ms:.1f}ms avg latency "
            f"(target: <{target_latency_ms}ms) - "
            f"{'PASS' if actual_latency_ms < target_latency_ms else 'FAIL'}"
        )
    
    async def _benchmark_storage_efficiency(self) -> None:
        """
        Benchmark storage efficiency.
        Target: <1GB per 10,000 contexts
        """
        logger.info("Benchmarking storage efficiency (target: <1GB per 10k contexts)")
        
        metrics = PerformanceMetrics("storage_efficiency")
        
        try:
            # Get database size information
            size_query = text("""
                SELECT 
                    pg_total_relation_size('contexts') as contexts_size,
                    pg_total_relation_size('context_relationships') as relationships_size,
                    pg_total_relation_size('context_retrievals') as retrievals_size,
                    COUNT(*) as total_contexts
                FROM contexts
            """)
            
            result = await self.db.execute(size_query)
            row = result.first()
            
            if row:
                contexts_size_mb = row.contexts_size / 1024 / 1024
                relationships_size_mb = row.relationships_size / 1024 / 1024
                retrievals_size_mb = row.retrievals_size / 1024 / 1024
                total_size_mb = contexts_size_mb + relationships_size_mb + retrievals_size_mb
                context_count = row.total_contexts
                
                # Calculate per-10k-contexts size
                if context_count > 0:
                    size_per_10k_mb = (total_size_mb / context_count) * 10000
                else:
                    size_per_10k_mb = 0
                
                metrics.additional_metrics.update({
                    "contexts_table_size_mb": contexts_size_mb,
                    "relationships_table_size_mb": relationships_size_mb,
                    "retrievals_table_size_mb": retrievals_size_mb,
                    "total_size_mb": total_size_mb,
                    "context_count": context_count,
                    "size_per_10k_contexts_mb": size_per_10k_mb
                })
                
                metrics.success_count = 1
                
            else:
                metrics.failure_count = 1
        
        except Exception as e:
            logger.error(f"Storage efficiency benchmark failed: {e}")
            metrics.failure_count = 1
        
        self.metrics["storage_efficiency"] = metrics
        
        # Validate target
        target_gb_per_10k = 1.0  # 1GB
        actual_gb_per_10k = metrics.additional_metrics.get("size_per_10k_contexts_mb", 0) / 1024
        
        logger.info(
            f"Storage efficiency: {actual_gb_per_10k:.2f}GB per 10k contexts "
            f"(target: <{target_gb_per_10k}GB) - "
            f"{'PASS' if actual_gb_per_10k < target_gb_per_10k else 'FAIL'}"
        )
    
    async def _benchmark_compression_performance(self) -> None:
        """Benchmark compression performance and quality."""
        logger.info("Benchmarking compression performance")
        
        metrics = PerformanceMetrics("compression_performance")
        
        # Test different compression levels
        compression_levels = [CompressionLevel.LIGHT, CompressionLevel.STANDARD, CompressionLevel.AGGRESSIVE]
        
        for level in compression_levels:
            level_metrics = []
            
            for i in range(10):  # 10 test compressions per level
                content = self._generate_long_content(3000)
                
                start_time = time.perf_counter()
                
                try:
                    # Use context compressor directly
                    compressor = await self.context_manager.compressor
                    compressed = await compressor.compress_conversation(
                        conversation_content=content,
                        compression_level=level
                    )
                    
                    compression_time = time.perf_counter() - start_time
                    
                    level_metrics.append({
                        "compression_time": compression_time,
                        "original_length": len(content),
                        "compressed_length": len(compressed.summary),
                        "compression_ratio": compressed.compression_ratio,
                        "importance_score": compressed.importance_score
                    })
                    
                    metrics.success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Compression failed: {e}")
                    metrics.failure_count += 1
            
            # Store level-specific metrics
            if level_metrics:
                avg_time = statistics.mean([m["compression_time"] for m in level_metrics])
                avg_ratio = statistics.mean([m["compression_ratio"] for m in level_metrics])
                
                metrics.additional_metrics[f"{level.value}_avg_time"] = avg_time
                metrics.additional_metrics[f"{level.value}_avg_ratio"] = avg_ratio
        
        self.metrics["compression_performance"] = metrics
    
    async def _benchmark_database_operations(self) -> None:
        """Benchmark core database operations."""
        logger.info("Benchmarking database operations")
        
        metrics = PerformanceMetrics("database_operations")
        
        operations = [
            ("context_insert", self._benchmark_context_insert),
            ("context_update", self._benchmark_context_update),
            ("context_delete", self._benchmark_context_delete),
            ("relationship_query", self._benchmark_relationship_query)
        ]
        
        for op_name, op_func in operations:
            try:
                times = await op_func()
                metrics.additional_metrics[f"{op_name}_times"] = times
                metrics.additional_metrics[f"{op_name}_avg_time"] = statistics.mean(times) if times else 0
                metrics.success_count += 1
            except Exception as e:
                logger.warning(f"Database operation {op_name} failed: {e}")
                metrics.failure_count += 1
        
        self.metrics["database_operations"] = metrics
    
    async def _benchmark_context_insert(self) -> List[float]:
        """Benchmark context insertion performance."""
        times = []
        
        for i in range(50):
            context_data = ContextCreate(
                title=f"Benchmark Context {i}",
                content=f"Test content for benchmark context {i}",
                context_type=ContextType.DOCUMENTATION,
                agent_id=uuid.uuid4(),
                importance_score=0.5
            )
            
            start_time = time.perf_counter()
            await self.context_manager.store_context(context_data)
            times.append(time.perf_counter() - start_time)
        
        return times
    
    async def _benchmark_context_update(self) -> List[float]:
        """Benchmark context update performance."""
        # Use existing test contexts
        times = []
        
        for context in self.test_contexts[:20]:  # Update first 20 contexts
            start_time = time.perf_counter()
            
            context.importance_score = 0.9
            await self.db.commit()
            
            times.append(time.perf_counter() - start_time)
        
        return times
    
    async def _benchmark_context_delete(self) -> List[float]:
        """Benchmark context deletion (archiving) performance."""
        times = []
        
        for context in self.test_contexts[-20:]:  # Delete last 20 contexts
            start_time = time.perf_counter()
            await self.context_manager.delete_context(context.id)
            times.append(time.perf_counter() - start_time)
        
        return times
    
    async def _benchmark_relationship_query(self) -> List[float]:
        """Benchmark relationship query performance."""
        times = []
        
        for context in self.test_contexts[:10]:  # Query relationships for first 10
            start_time = time.perf_counter()
            
            # Use search engine to find similar contexts
            search_engine = await self.context_manager._ensure_search_engine()
            await search_engine.find_similar_contexts(context.id, limit=5)
            
            times.append(time.perf_counter() - start_time)
        
        return times
    
    async def _setup_test_data(self, num_contexts: int) -> None:
        """Setup test data for benchmarks."""
        logger.info(f"Setting up {num_contexts} test contexts")
        
        self.test_contexts = []
        
        for i in range(num_contexts):
            context_data = ContextCreate(
                title=f"Test Context {i}",
                content=self._generate_test_content(i),
                context_type=random.choice(list(ContextType)),
                agent_id=uuid.uuid4(),
                importance_score=random.uniform(0.3, 1.0)
            )
            
            context = await self.context_manager.store_context(context_data)
            self.test_contexts.append(context)
            
            if i % 100 == 0:
                logger.debug(f"Created {i+1}/{num_contexts} test contexts")
        
        logger.info(f"Test data setup completed: {len(self.test_contexts)} contexts created")
    
    async def _cleanup_test_data(self) -> None:
        """Cleanup test data after benchmarks."""
        logger.info("Cleaning up test data")
        
        for context in self.test_contexts:
            try:
                await self.context_manager.delete_context(context.id)
            except Exception as e:
                logger.warning(f"Failed to cleanup context {context.id}: {e}")
        
        self.test_contexts = []
        logger.info("Test data cleanup completed")
    
    def _generate_test_content(self, index: int) -> str:
        """Generate realistic test content."""
        topics = [
            "database optimization", "API development", "security best practices",
            "performance tuning", "error handling", "deployment strategies",
            "monitoring setup", "backup procedures", "scaling techniques",
            "troubleshooting guide"
        ]
        
        topic = topics[index % len(topics)]
        
        return f"""
        This is a comprehensive guide about {topic} for production systems.
        
        Key points include:
        - Implementation best practices
        - Common pitfalls to avoid
        - Performance considerations
        - Security implications
        - Monitoring and alerting
        
        The content covers detailed technical information that would be valuable
        for agents working on similar problems in the future. It includes
        specific commands, configuration examples, and troubleshooting steps.
        
        This context was generated for testing purposes but represents the type
        of high-quality technical content that agents would store and retrieve
        in a production environment.
        """
    
    def _generate_long_content(self, target_length: int) -> str:
        """Generate long content for compression testing."""
        base_text = """
        This is a detailed technical document that contains extensive information
        about system architecture, implementation details, and operational procedures.
        The document includes comprehensive guidelines for development teams,
        detailed API specifications, security protocols, performance benchmarks,
        monitoring strategies, troubleshooting procedures, and maintenance schedules.
        """
        
        content = ""
        while len(content) < target_length:
            content += base_text + f" Section {len(content) // len(base_text) + 1}. "
        
        return content[:target_length]
    
    async def _generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "benchmark_timestamp": datetime.utcnow().isoformat(),
            "kpi_validation": {},
            "performance_metrics": {},
            "overall_status": "PASS"
        }
        
        # Validate each KPI target
        kpi_results = {}
        
        # Search performance: <50ms
        if "search_performance" in self.metrics:
            search_metrics = self.metrics["search_performance"]
            avg_ms = search_metrics.average_response_time * 1000
            kpi_results["search_response_time"] = {
                "target": "<50ms",
                "actual": f"{avg_ms:.1f}ms",
                "status": "PASS" if avg_ms < 50 else "FAIL"
            }
        
        # Token reduction: 60-80%
        if "token_reduction" in self.metrics:
            token_metrics = self.metrics["token_reduction"]
            reduction = token_metrics.additional_metrics.get("token_reduction_percentage", 0)
            kpi_results["token_reduction"] = {
                "target": "60-80%",
                "actual": f"{reduction:.1f}%",
                "status": "PASS" if 60 <= reduction <= 80 else "FAIL"
            }
        
        # Retrieval precision: >90%
        if "retrieval_precision" in self.metrics:
            precision_metrics = self.metrics["retrieval_precision"]
            precision = precision_metrics.additional_metrics.get("average_precision", 0)
            kpi_results["retrieval_precision"] = {
                "target": ">90%",
                "actual": f"{precision:.1%}",
                "status": "PASS" if precision > 0.9 else "FAIL"
            }
        
        # Concurrent access: 50+ agents <100ms
        if "concurrent_access" in self.metrics:
            concurrent_metrics = self.metrics["concurrent_access"]
            avg_ms = concurrent_metrics.average_response_time * 1000
            agents = concurrent_metrics.additional_metrics.get("concurrent_agents", 0)
            kpi_results["concurrent_access"] = {
                "target": "50+ agents <100ms",
                "actual": f"{agents} agents, {avg_ms:.1f}ms",
                "status": "PASS" if agents >= 50 and avg_ms < 100 else "FAIL"
            }
        
        # Storage efficiency: <1GB per 10k contexts
        if "storage_efficiency" in self.metrics:
            storage_metrics = self.metrics["storage_efficiency"]
            gb_per_10k = storage_metrics.additional_metrics.get("size_per_10k_contexts_mb", 0) / 1024
            kpi_results["storage_efficiency"] = {
                "target": "<1GB per 10k contexts",
                "actual": f"{gb_per_10k:.2f}GB per 10k contexts",
                "status": "PASS" if gb_per_10k < 1.0 else "FAIL"
            }
        
        report["kpi_validation"] = kpi_results
        
        # Check overall status
        if any(result["status"] == "FAIL" for result in kpi_results.values()):
            report["overall_status"] = "FAIL"
        
        # Add detailed metrics
        for name, metrics in self.metrics.items():
            report["performance_metrics"][name] = metrics.to_dict()
        
        # Add recommendations
        recommendations = []
        
        for kpi, result in kpi_results.items():
            if result["status"] == "FAIL":
                if kpi == "search_response_time":
                    recommendations.append("Consider optimizing vector indexes or increasing database resources")
                elif kpi == "token_reduction":
                    recommendations.append("Review compression algorithms and target ratios")
                elif kpi == "retrieval_precision":
                    recommendations.append("Improve embedding quality or relevance scoring")
                elif kpi == "concurrent_access":
                    recommendations.append("Optimize connection pooling and query performance")
                elif kpi == "storage_efficiency":
                    recommendations.append("Implement data compression or archiving strategies")
        
        if not recommendations:
            recommendations.append("All KPI targets met - system is production ready")
        
        report["recommendations"] = recommendations
        
        return report


# Factory function for creating benchmark suite
async def create_benchmark_suite(
    context_manager: ContextManager,
    db_session: AsyncSession
) -> PerformanceBenchmarkSuite:
    """
    Create performance benchmark suite instance.
    
    Args:
        context_manager: Context manager to benchmark
        db_session: Database session for direct queries
        
    Returns:
        PerformanceBenchmarkSuite instance
    """
    return PerformanceBenchmarkSuite(context_manager, db_session)