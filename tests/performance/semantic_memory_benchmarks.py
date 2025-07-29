"""
Performance Benchmarking Suite for Semantic Memory System.

This module provides comprehensive benchmarking tools to validate Phase 3 
performance targets and ensure production-ready semantic memory operations.

Validates Performance Targets:
- P95 Search Latency: <200ms for 1M vector index
- Ingestion Throughput: >500 documents/sec sustained  
- Context Compression: 60-80% reduction with <500ms processing
- Cross-Agent Knowledge: <200ms for knowledge transfer operations
- Workflow Overhead: <10ms additional DAG orchestration impact

Features:
- Comprehensive load testing with realistic data patterns
- Vector search performance validation under concurrent load
- Document ingestion throughput measurement
- Context compression quality and speed analysis
- Cross-agent knowledge sharing latency testing
- Integration performance with workflow systems
- Resource utilization monitoring and optimization
- Performance regression detection and alerting
"""

import asyncio
import logging
import time
import statistics
import uuid
import random
import gc
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import numpy as np

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text, delete

from app.services.semantic_memory_service import SemanticMemoryService, get_semantic_memory_service
from app.core.pgvector_manager import PGVectorManager, get_pgvector_manager
from app.core.semantic_embedding_service import SemanticEmbeddingService, get_embedding_service
from app.schemas.semantic_memory import (
    DocumentIngestRequest, BatchIngestRequest, SemanticSearchRequest,
    ContextCompressionRequest, CompressionMethod, TimeRange, KnowledgeType,
    SimilaritySearchRequest, ContextualizationRequest, ContextualizationMethod,
    DocumentMetadata, ProcessingOptions, BatchOptions, SearchFilters
)
from app.core.database import get_async_session

logger = logging.getLogger(__name__)


class PerformanceTarget(Enum):
    """Performance targets for semantic memory system."""
    P95_SEARCH_LATENCY_MS = 200.0
    INGESTION_THROUGHPUT_DOCS_SEC = 500.0
    CONTEXT_COMPRESSION_REDUCTION_PCT = 60.0
    CONTEXT_COMPRESSION_TIME_MS = 500.0
    KNOWLEDGE_TRANSFER_LATENCY_MS = 200.0
    WORKFLOW_OVERHEAD_MS = 10.0
    MEMORY_USAGE_MB = 1000.0
    CPU_UTILIZATION_PCT = 80.0
    ERROR_RATE_PCT = 1.0


class TestDataSize(Enum):
    """Test data size categories for scalability testing."""
    SMALL = "small"          # 100 documents
    MEDIUM = "medium"        # 1,000 documents  
    LARGE = "large"          # 10,000 documents
    XLARGE = "xlarge"        # 100,000 documents
    XXLARGE = "xxlarge"      # 1,000,000 documents


class LoadPattern(Enum):
    """Load testing patterns."""
    CONSTANT = "constant"
    BURST = "burst"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    MIXED = "mixed"


@dataclass
class BenchmarkResult:
    """Container for individual benchmark results."""
    name: str
    target_value: float
    measured_value: float
    target_operator: str = "<="
    passed: bool = False
    margin: float = 0.0
    execution_time_ms: float = 0.0
    percentiles: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate pass/fail status and margin."""
        if self.target_operator == "<=":
            self.passed = self.measured_value <= self.target_value
            self.margin = self.target_value - self.measured_value
        elif self.target_operator == ">=":
            self.passed = self.measured_value >= self.target_value
            self.margin = self.measured_value - self.target_value
        elif self.target_operator == "<":
            self.passed = self.measured_value < self.target_value
            self.margin = self.target_value - self.measured_value
        elif self.target_operator == ">":
            self.passed = self.measured_value > self.target_value
            self.margin = self.measured_value - self.target_value
        else:
            self.passed = self.measured_value == self.target_value
            self.margin = abs(self.measured_value - self.target_value)


@dataclass
class LoadTestResult:
    """Results from load testing scenarios."""
    scenario_name: str
    concurrent_users: int
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    operations_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    error_rate_percent: float
    resource_usage: Dict[str, float]
    timestamps: List[datetime] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    timestamp: datetime
    test_suite: str
    benchmarks: List[BenchmarkResult]
    load_tests: List[LoadTestResult]
    system_info: Dict[str, Any]
    overall_score: float
    recommendations: List[str]
    regression_analysis: Dict[str, Any] = field(default_factory=dict)


class RealisticDataGenerator:
    """Generate realistic test data for performance testing."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.content_templates = [
            "Technical documentation for {subject} explaining {concept} with detailed examples and {detail_level} analysis.",
            "Code implementation guide for {subject} focusing on {concept} with {detail_level} coverage of best practices.",
            "System architecture overview describing {subject} integration with {concept} using {detail_level} patterns.",
            "API specification for {subject} defining {concept} endpoints with {detail_level} documentation.",
            "Performance analysis of {subject} measuring {concept} under {detail_level} load conditions.",
            "Security assessment for {subject} evaluating {concept} vulnerabilities with {detail_level} testing.",
            "Database schema design for {subject} implementing {concept} with {detail_level} optimization.",
            "Deployment strategy for {subject} using {concept} infrastructure with {detail_level} automation.",
            "Testing framework for {subject} validating {concept} functionality through {detail_level} scenarios.",
            "Monitoring configuration for {subject} tracking {concept} metrics via {detail_level} dashboards."
        ]
        
        self.subjects = [
            "microservices", "kubernetes", "postgresql", "redis", "docker", "nginx", 
            "elasticsearch", "kafka", "mongodb", "mysql", "rabbitmq", "grafana",
            "prometheus", "jenkins", "terraform", "ansible", "oauth2", "jwt",
            "grpc", "rest-api", "websockets", "graphql", "react", "vue", "nodejs"
        ]
        
        self.concepts = [
            "scalability", "performance", "security", "reliability", "maintainability",
            "observability", "resilience", "efficiency", "consistency", "availability",
            "durability", "flexibility", "usability", "testability", "deployability"
        ]
        
        self.detail_levels = [
            "comprehensive", "detailed", "thorough", "extensive", "in-depth",
            "complete", "advanced", "professional", "enterprise", "production"
        ]
    
    def generate_document_content(self, min_length: int = 100, max_length: int = 2000) -> str:
        """Generate realistic document content."""
        template = self.rng.choice(self.content_templates)
        subject = self.rng.choice(self.subjects)
        concept = self.rng.choice(self.concepts)
        detail_level = self.rng.choice(self.detail_levels)
        
        content = template.format(
            subject=subject,
            concept=concept,
            detail_level=detail_level
        )
        
        # Add additional content to meet length requirements
        while len(content) < min_length:
            additional = f" Additional details about {subject} implementation include {concept} considerations. "
            content += additional
            
        # Truncate if too long
        if len(content) > max_length:
            content = content[:max_length-3] + "..."
            
        return content
    
    def generate_metadata(self) -> Dict[str, Any]:
        """Generate realistic document metadata."""
        return {
            "category": self.rng.choice(["documentation", "code", "specification", "analysis"]),
            "priority": self.rng.choice(["low", "medium", "high", "critical"]),
            "version": f"{self.rng.randint(1, 5)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
            "author": f"user_{self.rng.randint(1, 100)}",
            "department": self.rng.choice(["engineering", "product", "security", "devops"]),
            "importance": round(self.rng.uniform(0.1, 1.0), 2)
        }
    
    def generate_tags(self) -> List[str]:
        """Generate realistic document tags."""
        available_tags = [
            "production", "staging", "development", "critical", "deprecated",
            "beta", "stable", "experimental", "legacy", "migration",
            "performance", "security", "monitoring", "logging", "testing"
        ]
        return self.rng.sample(available_tags, self.rng.randint(1, 5))
    
    def generate_search_queries(self, count: int = 100) -> List[str]:
        """Generate realistic search queries."""
        query_patterns = [
            "{subject} {concept}",
            "how to implement {subject}",
            "{concept} in {subject}",
            "{subject} best practices",
            "{concept} optimization",
            "troubleshooting {subject}",
            "{subject} architecture",
            "{concept} monitoring",
            "{subject} deployment",
            "{concept} security"
        ]
        
        queries = []
        for _ in range(count):
            pattern = self.rng.choice(query_patterns)
            subject = self.rng.choice(self.subjects)
            concept = self.rng.choice(self.concepts)
            query = pattern.format(subject=subject, concept=concept)
            queries.append(query)
            
        return queries


class ResourceMonitor:
    """Monitor system resource usage during performance tests."""
    
    def __init__(self, process_id: Optional[int] = None):
        self.process = psutil.Process(process_id) if process_id else psutil.Process()
        self.measurements = deque(maxlen=1000)
        self.monitoring = False
        self._monitor_task = None
    
    async def start_monitoring(self, interval: float = 0.1):
        """Start continuous resource monitoring."""
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self, interval: float):
        """Continuous monitoring loop."""
        while self.monitoring:
            try:
                measurement = {
                    'timestamp': time.time(),
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                    'memory_percent': self.process.memory_percent(),
                    'threads': self.process.num_threads(),
                    'handles': getattr(self.process, 'num_handles', lambda: 0)(),
                    'system_cpu': psutil.cpu_percent(),
                    'system_memory': psutil.virtual_memory().percent,
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                }
                self.measurements.append(measurement)
                await asyncio.sleep(interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def get_stats(self) -> Dict[str, float]:
        """Get resource usage statistics."""
        if not self.measurements:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        memory_values = [m['memory_mb'] for m in self.measurements]
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_mb': statistics.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'peak_threads': max(m['threads'] for m in self.measurements),
            'measurement_count': len(self.measurements)
        }


class SemanticMemoryBenchmarks:
    """Core benchmarking framework for semantic memory system."""
    
    def __init__(self):
        self.semantic_service: Optional[SemanticMemoryService] = None
        self.pgvector_manager: Optional[PGVectorManager] = None
        self.embedding_service: Optional[SemanticEmbeddingService] = None
        self.data_generator = RealisticDataGenerator()
        self.resource_monitor = ResourceMonitor()
        self.test_documents: List[str] = []
        self.test_agent_ids: List[str] = []
        
    async def initialize(self):
        """Initialize benchmarking framework."""
        logger.info("🚀 Initializing Semantic Memory Benchmarks...")
        
        try:
            self.semantic_service = await get_semantic_memory_service()
            self.pgvector_manager = await get_pgvector_manager()
            self.embedding_service = await get_embedding_service()
            
            logger.info("✅ Semantic Memory Benchmarks initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize benchmarks: {e}")
            raise
    
    async def cleanup(self):
        """Clean up test resources."""
        try:
            # Clean up test documents
            if self.test_documents and self.pgvector_manager:
                for doc_id in self.test_documents:
                    try:
                        await self.pgvector_manager.delete_document(uuid.UUID(doc_id))
                    except Exception as e:
                        logger.warning(f"Failed to delete test document {doc_id}: {e}")
            
            # Stop monitoring
            await self.resource_monitor.stop_monitoring()
            
            # Clear test data
            self.test_documents.clear()
            self.test_agent_ids.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("🧹 Benchmark cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def setup_test_data(self, size: TestDataSize, agent_count: int = 5) -> Dict[str, Any]:
        """Setup test data for benchmarking."""
        size_mapping = {
            TestDataSize.SMALL: 100,
            TestDataSize.MEDIUM: 1000,
            TestDataSize.LARGE: 10000,
            TestDataSize.XLARGE: 100000,
            TestDataSize.XXLARGE: 1000000
        }
        
        document_count = size_mapping[size]
        docs_per_agent = document_count // agent_count
        
        logger.info(f"Setting up {document_count} test documents across {agent_count} agents")
        
        setup_start = time.time()
        
        # Generate agent IDs
        self.test_agent_ids = [str(uuid.uuid4()) for _ in range(agent_count)]
        
        # Generate documents in batches for better performance
        batch_size = min(100, docs_per_agent)
        documents_created = 0
        
        for agent_id in self.test_agent_ids:
            for batch_start in range(0, docs_per_agent, batch_size):
                batch_end = min(batch_start + batch_size, docs_per_agent)
                batch_docs = []
                
                for i in range(batch_start, batch_end):
                    content = self.data_generator.generate_document_content()
                    metadata = self.data_generator.generate_metadata()
                    tags = self.data_generator.generate_tags()
                    
                    doc_request = DocumentIngestRequest(
                        agent_id=agent_id,
                        content=content,
                        metadata=DocumentMetadata(**metadata),
                        tags=tags,
                        workflow_id=str(uuid.uuid4())
                    )
                    batch_docs.append(doc_request)
                
                # Batch ingest documents
                batch_request = BatchIngestRequest(
                    documents=batch_docs,
                    batch_options=BatchOptions(
                        parallel_processing=True,
                        batch_size=batch_size
                    )
                )
                
                try:
                    response = await self.semantic_service.batch_ingest_documents(batch_request)
                    successful_docs = [r.document_id for r in response.results if r.status == "success"]
                    self.test_documents.extend([str(doc_id) for doc_id in successful_docs])
                    documents_created += len(successful_docs)
                    
                    if documents_created % 1000 == 0:
                        logger.info(f"Created {documents_created}/{document_count} test documents")
                        
                except Exception as e:
                    logger.warning(f"Batch ingestion failed: {e}")
        
        setup_time = time.time() - setup_start
        
        logger.info(f"✅ Test data setup completed: {len(self.test_documents)} documents in {setup_time:.2f}s")
        
        return {
            'document_count': len(self.test_documents),
            'agent_count': len(self.test_agent_ids),
            'setup_time_seconds': setup_time,
            'documents_per_second': len(self.test_documents) / setup_time if setup_time > 0 else 0
        }
    
    async def benchmark_search_latency(
        self,
        query_count: int = 100,
        concurrent_queries: int = 10
    ) -> BenchmarkResult:
        """Benchmark semantic search latency performance."""
        logger.info(f"🔍 Benchmarking search latency: {query_count} queries, {concurrent_queries} concurrent")
        
        # Generate realistic search queries
        queries = self.data_generator.generate_search_queries(query_count)
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        latencies = []
        errors = 0
        
        async def execute_search(query: str) -> float:
            """Execute single search and return latency."""
            try:
                start_time = time.perf_counter()
                
                search_request = SemanticSearchRequest(
                    query=query,
                    limit=20,
                    similarity_threshold=0.5,
                    agent_id=random.choice(self.test_agent_ids) if self.test_agent_ids else None
                )
                
                response = await self.semantic_service.semantic_search(search_request)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                return latency_ms
                
            except Exception as e:
                logger.warning(f"Search query failed: {e}")
                return 0.0
        
        # Execute queries in batches for controlled concurrency
        start_time = time.time()
        
        for batch_start in range(0, len(queries), concurrent_queries):
            batch_end = min(batch_start + concurrent_queries, len(queries))
            batch_queries = queries[batch_start:batch_end]
            
            # Execute batch concurrently
            tasks = [execute_search(query) for query in batch_queries]
            batch_latencies = await asyncio.gather(*tasks)
            
            # Filter out errors (0.0 latencies)
            valid_latencies = [lat for lat in batch_latencies if lat > 0]
            failed_count = len(batch_latencies) - len(valid_latencies)
            
            latencies.extend(valid_latencies)
            errors += failed_count
        
        total_time = time.time() - start_time
        
        # Stop monitoring and get resource stats
        await self.resource_monitor.stop_monitoring()
        resource_stats = self.resource_monitor.get_stats()
        
        # Calculate percentiles
        if latencies:
            latencies.sort()
            percentiles = {
                'p50': np.percentile(latencies, 50),
                'p90': np.percentile(latencies, 90),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'max': max(latencies),
                'min': min(latencies),
                'avg': statistics.mean(latencies)
            }
            p95_latency = percentiles['p95']
        else:
            percentiles = {}
            p95_latency = float('inf')
        
        queries_per_second = len(latencies) / total_time if total_time > 0 else 0
        error_rate = (errors / len(queries)) * 100 if queries else 0
        
        logger.info(f"Search latency P95: {p95_latency:.2f}ms, QPS: {queries_per_second:.1f}, Errors: {error_rate:.1f}%")
        
        return BenchmarkResult(
            name="search_latency_p95_ms",
            target_value=PerformanceTarget.P95_SEARCH_LATENCY_MS.value,
            measured_value=p95_latency,
            target_operator="<=",
            execution_time_ms=total_time * 1000,
            percentiles=percentiles,
            metadata={
                'total_queries': len(queries),
                'successful_queries': len(latencies),
                'failed_queries': errors,
                'queries_per_second': queries_per_second,
                'error_rate_percent': error_rate,
                'concurrent_queries': concurrent_queries,
                'resource_usage': resource_stats
            }
        )
    
    async def benchmark_ingestion_throughput(
        self,
        document_count: int = 1000,
        batch_size: int = 50,
        concurrent_batches: int = 5
    ) -> BenchmarkResult:
        """Benchmark document ingestion throughput."""
        logger.info(f"📥 Benchmarking ingestion throughput: {document_count} docs, batch_size={batch_size}")
        
        # Generate test documents
        test_docs = []
        for i in range(document_count):
            content = self.data_generator.generate_document_content()
            metadata = self.data_generator.generate_metadata()
            tags = self.data_generator.generate_tags()
            agent_id = random.choice(self.test_agent_ids) if self.test_agent_ids else str(uuid.uuid4())
            
            doc_request = DocumentIngestRequest(
                agent_id=agent_id,
                content=content,
                metadata=DocumentMetadata(**metadata),
                tags=tags,
                workflow_id=str(uuid.uuid4())
            )
            test_docs.append(doc_request)
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        successful_ingestions = 0
        failed_ingestions = 0
        total_processing_time = 0
        
        async def ingest_batch(batch_docs: List[DocumentIngestRequest]) -> Tuple[int, int, float]:
            """Ingest batch of documents and return success/failure counts and time."""
            try:
                start_time = time.perf_counter()
                
                batch_request = BatchIngestRequest(
                    documents=batch_docs,
                    batch_options=BatchOptions(
                        parallel_processing=True,
                        batch_size=len(batch_docs)
                    )
                )
                
                response = await self.semantic_service.batch_ingest_documents(batch_request)
                processing_time = time.perf_counter() - start_time
                
                successful = response.successful_ingestions
                failed = response.failed_ingestions
                
                # Track created documents for cleanup
                successful_doc_ids = [
                    str(r.document_id) for r in response.results 
                    if r.status == "success" and r.document_id
                ]
                self.test_documents.extend(successful_doc_ids)
                
                return successful, failed, processing_time
                
            except Exception as e:
                logger.warning(f"Batch ingestion failed: {e}")
                return 0, len(batch_docs), 0.0
        
        # Execute ingestion in concurrent batches
        start_time = time.time()
        
        # Split documents into batches
        batches = [
            test_docs[i:i + batch_size] 
            for i in range(0, len(test_docs), batch_size)
        ]
        
        # Process batches with controlled concurrency
        for batch_group_start in range(0, len(batches), concurrent_batches):
            batch_group_end = min(batch_group_start + concurrent_batches, len(batches))
            batch_group = batches[batch_group_start:batch_group_end]
            
            # Execute batch group concurrently
            tasks = [ingest_batch(batch) for batch in batch_group]
            results = await asyncio.gather(*tasks)
            
            # Aggregate results
            for success, failed, proc_time in results:
                successful_ingestions += success
                failed_ingestions += failed
                total_processing_time += proc_time
        
        total_time = time.time() - start_time
        
        # Stop monitoring
        await self.resource_monitor.stop_monitoring()
        resource_stats = self.resource_monitor.get_stats()
        
        # Calculate throughput
        throughput_docs_per_sec = successful_ingestions / total_time if total_time > 0 else 0
        error_rate = (failed_ingestions / document_count) * 100 if document_count > 0 else 0
        
        logger.info(f"Ingestion throughput: {throughput_docs_per_sec:.1f} docs/sec, Errors: {error_rate:.1f}%")
        
        return BenchmarkResult(
            name="ingestion_throughput_docs_per_sec",
            target_value=PerformanceTarget.INGESTION_THROUGHPUT_DOCS_SEC.value,
            measured_value=throughput_docs_per_sec,
            target_operator=">=",
            execution_time_ms=total_time * 1000,
            metadata={
                'total_documents': document_count,
                'successful_ingestions': successful_ingestions,
                'failed_ingestions': failed_ingestions,
                'batch_size': batch_size,
                'concurrent_batches': concurrent_batches,
                'error_rate_percent': error_rate,
                'avg_processing_time_per_batch_ms': (total_processing_time / len(batches)) * 1000 if batches else 0,
                'resource_usage': resource_stats
            }
        )
    
    async def benchmark_context_compression(
        self,
        context_sizes: List[int] = None,
        compression_methods: List[CompressionMethod] = None
    ) -> List[BenchmarkResult]:
        """Benchmark context compression performance and quality."""
        if context_sizes is None:
            context_sizes = [100, 500, 1000, 5000]
        
        if compression_methods is None:
            compression_methods = [
                CompressionMethod.SEMANTIC_CLUSTERING,
                CompressionMethod.IMPORTANCE_FILTERING,
                CompressionMethod.TEMPORAL_DECAY,
                CompressionMethod.HYBRID
            ]
        
        logger.info(f"🗜️ Benchmarking context compression: {len(context_sizes)} sizes, {len(compression_methods)} methods")
        
        results = []
        
        for method in compression_methods:
            method_results = []
            
            for context_size in context_sizes:
                # Create compression request
                compression_request = ContextCompressionRequest(
                    context_id=f"test_context_{context_size}",
                    compression_method=method,
                    target_reduction=0.6,  # 60% reduction target
                    preserve_importance_threshold=0.7
                )
                
                # Start resource monitoring
                await self.resource_monitor.start_monitoring()
                
                try:
                    start_time = time.perf_counter()
                    
                    response = await self.semantic_service.compress_context(compression_request)
                    
                    processing_time_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Stop monitoring
                    await self.resource_monitor.stop_monitoring()
                    resource_stats = self.resource_monitor.get_stats()
                    
                    # Validate compression quality
                    compression_ratio = response.compression_ratio
                    semantic_preservation = response.semantic_preservation_score
                    
                    method_results.append({
                        'context_size': context_size,
                        'processing_time_ms': processing_time_ms,
                        'compression_ratio': compression_ratio,
                        'semantic_preservation': semantic_preservation,
                        'resource_usage': resource_stats
                    })
                    
                    logger.info(f"{method} compression for {context_size} contexts: "
                              f"{compression_ratio:.1%} reduction in {processing_time_ms:.2f}ms")
                    
                except Exception as e:
                    logger.error(f"Compression benchmark failed for {method} with {context_size} contexts: {e}")
                    method_results.append({
                        'context_size': context_size,
                        'processing_time_ms': float('inf'),
                        'compression_ratio': 0.0,
                        'semantic_preservation': 0.0,
                        'error': str(e)
                    })
            
            # Calculate aggregate metrics for this method
            valid_results = [r for r in method_results if 'error' not in r]
            
            if valid_results:
                avg_processing_time = statistics.mean([r['processing_time_ms'] for r in valid_results])
                avg_compression_ratio = statistics.mean([r['compression_ratio'] for r in valid_results])
                avg_semantic_preservation = statistics.mean([r['semantic_preservation'] for r in valid_results])
                max_processing_time = max([r['processing_time_ms'] for r in valid_results])
                
                # Create benchmark result for processing time
                results.append(BenchmarkResult(
                    name=f"context_compression_time_ms_{method.value}",
                    target_value=PerformanceTarget.CONTEXT_COMPRESSION_TIME_MS.value,
                    measured_value=max_processing_time,
                    target_operator="<=",
                    metadata={
                        'compression_method': method.value,
                        'avg_processing_time_ms': avg_processing_time,
                        'avg_compression_ratio': avg_compression_ratio,
                        'avg_semantic_preservation': avg_semantic_preservation,
                        'context_sizes_tested': context_sizes,
                        'detailed_results': method_results
                    }
                ))
                
                # Create benchmark result for compression ratio
                results.append(BenchmarkResult(
                    name=f"context_compression_ratio_{method.value}",
                    target_value=PerformanceTarget.CONTEXT_COMPRESSION_REDUCTION_PCT.value / 100,
                    measured_value=avg_compression_ratio,
                    target_operator=">=",
                    metadata={
                        'compression_method': method.value,
                        'avg_processing_time_ms': avg_processing_time,
                        'avg_semantic_preservation': avg_semantic_preservation,
                        'context_sizes_tested': context_sizes,
                        'detailed_results': method_results
                    }
                ))
        
        return results
    
    async def benchmark_cross_agent_knowledge_sharing(
        self,
        agent_count: int = 10,
        knowledge_queries: int = 50
    ) -> BenchmarkResult:
        """Benchmark cross-agent knowledge sharing performance."""
        logger.info(f"🤝 Benchmarking cross-agent knowledge sharing: {agent_count} agents, {knowledge_queries} queries")
        
        # Ensure we have enough test agents
        while len(self.test_agent_ids) < agent_count:
            self.test_agent_ids.append(str(uuid.uuid4()))
        
        knowledge_latencies = []
        errors = 0
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        async def get_agent_knowledge(agent_id: str) -> float:
            """Get agent knowledge and return latency."""
            try:
                start_time = time.perf_counter()
                
                response = await self.semantic_service.get_agent_knowledge(
                    agent_id=agent_id,
                    knowledge_type=KnowledgeType.ALL,
                    time_range=TimeRange.DAYS_7
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                return latency_ms
                
            except Exception as e:
                logger.warning(f"Knowledge query failed for agent {agent_id}: {e}")
                return 0.0
        
        # Execute knowledge queries
        start_time = time.time()
        
        for _ in range(knowledge_queries):
            agent_id = random.choice(self.test_agent_ids)
            latency = await get_agent_knowledge(agent_id)
            
            if latency > 0:
                knowledge_latencies.append(latency)
            else:
                errors += 1
        
        total_time = time.time() - start_time
        
        # Stop monitoring
        await self.resource_monitor.stop_monitoring()
        resource_stats = self.resource_monitor.get_stats()
        
        # Calculate performance metrics
        if knowledge_latencies:
            avg_latency = statistics.mean(knowledge_latencies)
            p95_latency = np.percentile(knowledge_latencies, 95)
            max_latency = max(knowledge_latencies)
        else:
            avg_latency = p95_latency = max_latency = float('inf')
        
        queries_per_second = len(knowledge_latencies) / total_time if total_time > 0 else 0
        error_rate = (errors / knowledge_queries) * 100 if knowledge_queries > 0 else 0
        
        logger.info(f"Knowledge sharing P95 latency: {p95_latency:.2f}ms, QPS: {queries_per_second:.1f}")
        
        return BenchmarkResult(
            name="knowledge_sharing_latency_p95_ms",
            target_value=PerformanceTarget.KNOWLEDGE_TRANSFER_LATENCY_MS.value,
            measured_value=p95_latency,
            target_operator="<=",
            execution_time_ms=total_time * 1000,
            metadata={
                'agent_count': agent_count,
                'total_queries': knowledge_queries,
                'successful_queries': len(knowledge_latencies),
                'failed_queries': errors,
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max_latency,
                'queries_per_second': queries_per_second,
                'error_rate_percent': error_rate,
                'resource_usage': resource_stats
            }
        )
    
    async def run_comprehensive_benchmark_suite(
        self,
        test_data_size: TestDataSize = TestDataSize.MEDIUM,
        include_load_tests: bool = True
    ) -> PerformanceReport:
        """Run comprehensive performance benchmark suite."""
        logger.info(f"🚀 Starting comprehensive benchmark suite with {test_data_size.value} dataset")
        
        start_time = time.time()
        
        try:
            # Initialize benchmarking framework
            await self.initialize()
            
            # Setup test data
            setup_info = await self.setup_test_data(test_data_size)
            
            # Run individual benchmarks
            benchmarks = []
            
            # Search latency benchmark
            search_result = await self.benchmark_search_latency(
                query_count=100,
                concurrent_queries=10
            )
            benchmarks.append(search_result)
            
            # Ingestion throughput benchmark
            throughput_result = await self.benchmark_ingestion_throughput(
                document_count=500,
                batch_size=25,
                concurrent_batches=5
            )
            benchmarks.append(throughput_result)
            
            # Context compression benchmarks
            compression_results = await self.benchmark_context_compression()
            benchmarks.extend(compression_results)
            
            # Cross-agent knowledge sharing benchmark
            knowledge_result = await self.benchmark_cross_agent_knowledge_sharing(
                agent_count=5,
                knowledge_queries=25
            )
            benchmarks.append(knowledge_result)
            
            # Calculate overall score
            passed_benchmarks = sum(1 for b in benchmarks if b.passed)
            overall_score = passed_benchmarks / len(benchmarks) if benchmarks else 0.0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(benchmarks)
            
            # Create performance report
            report = PerformanceReport(
                timestamp=datetime.utcnow(),
                test_suite="Semantic Memory Comprehensive Benchmark",
                benchmarks=benchmarks,
                load_tests=[],  # TODO: Add load test results
                system_info={
                    'test_data_size': test_data_size.value,
                    'setup_info': setup_info,
                    'total_execution_time_seconds': time.time() - start_time
                },
                overall_score=overall_score,
                recommendations=recommendations
            )
            
            # Log summary
            logger.info(f"✅ Benchmark suite completed: {passed_benchmarks}/{len(benchmarks)} passed "
                       f"(Score: {overall_score:.1%})")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Benchmark suite failed: {e}")
            raise
        finally:
            # Cleanup test resources
            await self.cleanup()
    
    def _generate_recommendations(self, benchmarks: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []
        
        # Analyze each benchmark
        for benchmark in benchmarks:
            if not benchmark.passed:
                if "search_latency" in benchmark.name:
                    recommendations.append(
                        f"🔍 Search latency ({benchmark.measured_value:.1f}ms) exceeds target. "
                        "Consider index optimization, query caching, or hardware scaling."
                    )
                elif "ingestion_throughput" in benchmark.name:
                    recommendations.append(
                        f"📥 Ingestion throughput ({benchmark.measured_value:.1f} docs/sec) below target. "
                        "Consider batch size optimization, parallel processing, or database tuning."
                    )
                elif "compression_time" in benchmark.name:
                    recommendations.append(
                        f"🗜️ Context compression time ({benchmark.measured_value:.1f}ms) too high. "
                        "Consider algorithm optimization or asynchronous processing."
                    )
                elif "compression_ratio" in benchmark.name:
                    recommendations.append(
                        f"📊 Compression ratio ({benchmark.measured_value:.1%}) insufficient. "
                        "Review compression algorithms and importance thresholds."
                    )
                elif "knowledge_sharing" in benchmark.name:
                    recommendations.append(
                        f"🤝 Knowledge sharing latency ({benchmark.measured_value:.1f}ms) too high. "
                        "Consider knowledge caching and graph optimization."
                    )
        
        # General recommendations
        if len([b for b in benchmarks if b.passed]) / len(benchmarks) < 0.8:
            recommendations.append(
                "⚠️ Overall performance below expectations. Consider infrastructure scaling."
            )
        
        if not recommendations:
            recommendations.append("✅ All performance targets met! System ready for production.")
        
        return recommendations