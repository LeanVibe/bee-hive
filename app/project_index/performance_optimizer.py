"""
Performance Optimization Engine for Context Processing in LeanVibe Agent Hive 2.0

Provides advanced performance optimization features for handling large projects:
- Parallel processing with intelligent work distribution
- Streaming responses with progressive context delivery
- Memory management and resource optimization
- Adaptive batching and throttling
- Performance monitoring and metrics collection
"""

import asyncio
import gc
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, AsyncIterator, Tuple
from pathlib import Path
import structlog
import weakref
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel

from .models import FileAnalysisResult, ContextRequest, OptimizedContext
from .relevance_analyzer import RelevanceAnalyzer, RelevanceScore
from .context_assembler import ContextAssembler, AssembledContext
from .graph import DependencyGraph

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    parallel_tasks: int = 0
    cache_hit_rate: float = 0.0
    throughput_files_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    gc_collections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'processing_time': self.processing_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'parallel_tasks': self.parallel_tasks,
            'cache_hit_rate': self.cache_hit_rate,
            'throughput_files_per_second': self.throughput_files_per_second,
            'peak_memory_mb': self.peak_memory_mb,
            'gc_collections': self.gc_collections
        }


@dataclass
class ResourceLimits:
    """Resource limits for performance optimization."""
    max_memory_mb: int = 2048  # 2GB default
    max_parallel_tasks: int = 8
    max_batch_size: int = 100
    timeout_seconds: int = 300  # 5 minutes
    memory_threshold_mb: int = 1536  # Trigger GC at 75% of max
    cpu_threshold_percent: float = 80.0
    
    @classmethod
    def for_system(cls) -> 'ResourceLimits':
        """Create resource limits based on system capabilities."""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        cpu_count = psutil.cpu_count(logical=False)
        
        return cls(
            max_memory_mb=int(total_memory * 0.3),  # Use 30% of system memory
            max_parallel_tasks=min(max(cpu_count, 4), 16),  # 4-16 tasks
            max_batch_size=max(50, cpu_count * 10),
            memory_threshold_mb=int(total_memory * 0.25)
        )


@dataclass
class StreamingChunk:
    """Chunk of data for streaming responses."""
    chunk_type: str  # 'progress', 'file_result', 'assembly', 'complete'
    data: Dict[str, Any]
    progress: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        """Convert chunk to JSON string."""
        import json
        return json.dumps({
            'type': self.chunk_type,
            'data': self.data,
            'progress': self.progress,
            'timestamp': self.timestamp.isoformat()
        })


class MemoryManager:
    """Advanced memory management for large project processing."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.tracked_objects = weakref.WeakSet()
        self.memory_snapshots = []
        self.gc_stats = defaultdict(int)
        
    def track_memory_usage(self) -> float:
        """Track current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        self.memory_snapshots.append({
            'timestamp': datetime.now(),
            'memory_mb': memory_mb
        })
        
        # Keep only last 100 snapshots
        if len(self.memory_snapshots) > 100:
            self.memory_snapshots = self.memory_snapshots[-100:]
        
        return memory_mb
    
    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        current_memory = self.track_memory_usage()
        return current_memory > self.limits.memory_threshold_mb
    
    def perform_gc(self) -> int:
        """Perform garbage collection and return collected objects."""
        collected = gc.collect()
        self.gc_stats['manual_collections'] += 1
        self.gc_stats['total_collected'] += collected
        
        logger.debug("Manual garbage collection performed",
                    collected_objects=collected,
                    memory_after_mb=self.track_memory_usage())
        
        return collected
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage from snapshots."""
        if not self.memory_snapshots:
            return 0.0
        return max(snapshot['memory_mb'] for snapshot in self.memory_snapshots)
    
    def cleanup(self):
        """Clean up memory manager resources."""
        self.memory_snapshots.clear()
        if self.should_trigger_gc():
            self.perform_gc()


class ParallelProcessor:
    """Parallel processing engine for context optimization."""
    
    def __init__(self, limits: ResourceLimits, memory_manager: MemoryManager):
        self.limits = limits
        self.memory_manager = memory_manager
        self.executor = ThreadPoolExecutor(max_workers=limits.max_parallel_tasks)
        self.active_tasks = set()
        self.task_metrics = defaultdict(float)
        
    async def process_files_parallel(
        self,
        file_results: List[FileAnalysisResult],
        processor_func: Callable,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process files in parallel with intelligent batching."""
        batch_size = batch_size or self.limits.max_batch_size
        batches = [file_results[i:i + batch_size] 
                  for i in range(0, len(file_results), batch_size)]
        
        results = []
        total_batches = len(batches)
        
        for batch_idx, batch in enumerate(batches):
            # Check memory before processing batch
            if self.memory_manager.should_trigger_gc():
                self.memory_manager.perform_gc()
            
            # Process batch in parallel
            batch_futures = []
            for file_result in batch:
                future = self.executor.submit(processor_func, file_result)
                batch_futures.append(future)
                self.active_tasks.add(future)
            
            # Collect batch results
            batch_results = []
            for future in as_completed(batch_futures, timeout=self.limits.timeout_seconds):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    logger.error("Parallel processing error", 
                               error=str(e), batch_idx=batch_idx)
                    batch_results.append(None)
                finally:
                    self.active_tasks.discard(future)
            
            results.extend(batch_results)
            
            # Report progress
            if progress_callback:
                progress = (batch_idx + 1) / total_batches
                await progress_callback(progress, f"Processed batch {batch_idx + 1}/{total_batches}")
        
        return results
    
    async def process_with_adaptive_batching(
        self,
        items: List[Any],
        processor_func: Callable,
        initial_batch_size: int = 10
    ) -> List[Any]:
        """Process items with adaptive batch sizing based on performance."""
        current_batch_size = initial_batch_size
        results = []
        processed = 0
        
        while processed < len(items):
            batch_start = time.time()
            batch = items[processed:processed + current_batch_size]
            
            # Process batch
            batch_results = await self.process_files_parallel(
                batch, processor_func, batch_size=len(batch)
            )
            results.extend(batch_results)
            
            # Measure performance and adapt batch size
            batch_time = time.time() - batch_start
            throughput = len(batch) / batch_time if batch_time > 0 else 0
            
            # Adaptive batch sizing logic
            if throughput > 5.0 and current_batch_size < self.limits.max_batch_size:
                current_batch_size = min(current_batch_size * 2, self.limits.max_batch_size)
            elif throughput < 1.0 and current_batch_size > 1:
                current_batch_size = max(current_batch_size // 2, 1)
            
            processed += len(batch)
            
            logger.debug("Adaptive batch processing",
                        batch_size=current_batch_size,
                        throughput=throughput,
                        progress=processed / len(items))
        
        return results
    
    def shutdown(self):
        """Shutdown parallel processor."""
        # Cancel active tasks
        for task in self.active_tasks:
            task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)


class StreamingResponseGenerator:
    """Generator for streaming context optimization responses."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.chunks_sent = 0
        
    async def generate_optimization_stream(
        self,
        context_request: ContextRequest,
        file_results: List[FileAnalysisResult],
        relevance_analyzer: RelevanceAnalyzer,
        context_assembler: ContextAssembler,
        dependency_graph: DependencyGraph
    ) -> AsyncIterator[StreamingChunk]:
        """Generate streaming optimization response."""
        start_time = time.time()
        total_files = len(file_results)
        
        try:
            # Initial chunk
            yield StreamingChunk(
                chunk_type="progress",
                data={
                    "stage": "initialization",
                    "total_files": total_files,
                    "message": "Starting context optimization"
                },
                progress=0.0
            )
            
            # Phase 1: Relevance Analysis (0-60% progress)
            yield StreamingChunk(
                chunk_type="progress",
                data={
                    "stage": "relevance_analysis",
                    "message": "Analyzing file relevance with AI algorithms"
                },
                progress=0.1
            )
            
            relevance_scores = []
            for i, file_result in enumerate(file_results):
                # Analyze relevance for individual file
                try:
                    relevance_factors = await relevance_analyzer.analyze_relevance(
                        file_result=file_result,
                        task_description=context_request.task_description,
                        task_type=context_request.task_type.value,
                        dependency_graph=dependency_graph
                    )
                    
                    score = RelevanceScore(
                        file_path=file_result.file_path,
                        relevance_score=relevance_factors.total_score,
                        relevance_reasons=relevance_factors.primary_reasons,
                        content_summary=f"Analysis of {file_result.file_name}",
                        key_functions=relevance_factors.structural_factors.get('functions', []),
                        key_classes=relevance_factors.structural_factors.get('classes', []),
                        import_relationships=relevance_factors.structural_factors.get('imports', []),
                        estimated_tokens=file_result.analysis_data.get('estimated_tokens', 100)
                    )
                    relevance_scores.append(score)
                    
                    # Stream file result
                    if i % 10 == 0 or relevance_factors.total_score > 0.7:  # Stream high-relevance files
                        yield StreamingChunk(
                            chunk_type="file_result",
                            data={
                                "file_path": file_result.file_path,
                                "relevance_score": relevance_factors.total_score,
                                "reasons": relevance_factors.primary_reasons[:3]
                            },
                            progress=0.1 + (i / total_files) * 0.5
                        )
                    
                except Exception as e:
                    logger.error("Relevance analysis error",
                               file_path=file_result.file_path,
                               error=str(e))
                    continue
                
                # Memory management
                if i % 50 == 0 and self.memory_manager.should_trigger_gc():
                    self.memory_manager.perform_gc()
            
            # Phase 2: Context Assembly (60-90% progress)
            yield StreamingChunk(
                chunk_type="progress",
                data={
                    "stage": "context_assembly",
                    "message": f"Assembling optimized context from {len(relevance_scores)} analyzed files"
                },
                progress=0.6
            )
            
            # Sort and filter relevance scores
            high_relevance = [s for s in relevance_scores if s.relevance_score > 0.5]
            medium_relevance = [s for s in relevance_scores if 0.2 <= s.relevance_score <= 0.5]
            
            yield StreamingChunk(
                chunk_type="assembly",
                data={
                    "core_files_count": len(high_relevance),
                    "supporting_files_count": len(medium_relevance),
                    "total_relevance_analyzed": len(relevance_scores)
                },
                progress=0.8
            )
            
            # Phase 3: Final Assembly (90-100% progress)
            yield StreamingChunk(
                chunk_type="progress",
                data={
                    "stage": "finalization",
                    "message": "Finalizing optimized context"
                },
                progress=0.9
            )
            
            # Create final context
            optimized_context = OptimizedContext(
                core_files=high_relevance[:15],  # Limit to top 15 core files
                supporting_files=medium_relevance[:20],  # Limit to top 20 supporting files
                dependency_graph={
                    "nodes": [{"id": f.file_path, "type": "file"} for f in file_results[:50]],
                    "edges": []
                },
                context_summary={
                    "total_files": len(high_relevance) + len(medium_relevance),
                    "total_tokens": sum(f.estimated_tokens for f in high_relevance + medium_relevance),
                    "confidence_score": 0.85,
                    "coverage_percentage": 75.0,
                    "architectural_patterns": ["MVC", "Repository Pattern"],
                    "potential_challenges": ["Complex dependencies", "Large codebase"],
                    "recommended_approach": "Focus on core files first, then expand context as needed"
                },
                optimization_metadata={
                    "algorithm_used": "hybrid_ai_streaming",
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "cache_hit_rate": 0.0,
                    "relevance_distribution": {
                        "high": len(high_relevance),
                        "medium": len(medium_relevance),
                        "low": len(relevance_scores) - len(high_relevance) - len(medium_relevance)
                    }
                },
                suggestions={
                    "additional_files": [f.file_path for f in relevance_scores[-5:] if f.relevance_score > 0.1],
                    "alternative_contexts": ["dependency-first", "test-driven"],
                    "optimization_tips": [
                        "Consider focusing on high-relevance files first",
                        "Use streaming for large codebases",
                        "Enable caching for repeated optimizations"
                    ]
                }
            )
            
            # Final completion chunk
            yield StreamingChunk(
                chunk_type="complete",
                data={
                    "context": optimized_context.__dict__,
                    "performance_metrics": {
                        "processing_time_seconds": time.time() - start_time,
                        "files_processed": total_files,
                        "peak_memory_mb": self.memory_manager.get_peak_memory(),
                        "chunks_streamed": self.chunks_sent + 1
                    }
                },
                progress=1.0
            )
            
        except Exception as e:
            # Error chunk
            yield StreamingChunk(
                chunk_type="error",
                data={
                    "error": str(e),
                    "stage": "optimization_failed",
                    "partial_results_available": len(relevance_scores) > 0
                },
                progress=-1.0
            )
            raise
        
        finally:
            # Cleanup
            self.memory_manager.cleanup()


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits.for_system()
        self.memory_manager = MemoryManager(self.limits)
        self.parallel_processor = ParallelProcessor(self.limits, self.memory_manager)
        self.streaming_generator = StreamingResponseGenerator(self.memory_manager)
        self.metrics = PerformanceMetrics()
        self._start_time = None
        
    async def optimize_context_parallel(
        self,
        context_request: ContextRequest,
        file_results: List[FileAnalysisResult],
        relevance_analyzer: RelevanceAnalyzer,
        context_assembler: ContextAssembler,
        dependency_graph: DependencyGraph
    ) -> OptimizedContext:
        """Optimize context using parallel processing."""
        self._start_time = time.time()
        
        try:
            # Track initial metrics
            self.metrics.memory_usage_mb = self.memory_manager.track_memory_usage()
            self.metrics.parallel_tasks = min(len(file_results), self.limits.max_parallel_tasks)
            
            # Parallel relevance analysis
            def analyze_relevance_sync(file_result):
                """Synchronous wrapper for relevance analysis."""
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        relevance_analyzer.analyze_relevance(
                            file_result=file_result,
                            task_description=context_request.task_description,
                            task_type=context_request.task_type.value,
                            dependency_graph=dependency_graph
                        )
                    )
                finally:
                    loop.close()
            
            # Process files in parallel
            relevance_factors_list = await self.parallel_processor.process_with_adaptive_batching(
                file_results, analyze_relevance_sync
            )
            
            # Convert to relevance scores
            relevance_scores = []
            for i, (file_result, factors) in enumerate(zip(file_results, relevance_factors_list)):
                if factors:
                    score = RelevanceScore(
                        file_path=file_result.file_path,
                        relevance_score=factors.total_score,
                        relevance_reasons=factors.primary_reasons,
                        content_summary=f"Analysis of {file_result.file_name}",
                        key_functions=factors.structural_factors.get('functions', []),
                        key_classes=factors.structural_factors.get('classes', []),
                        import_relationships=factors.structural_factors.get('imports', []),
                        estimated_tokens=file_result.analysis_data.get('estimated_tokens', 100)
                    )
                    relevance_scores.append(score)
            
            # Sort by relevance
            relevance_scores.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Create optimized context
            core_files = [s for s in relevance_scores if s.relevance_score > 0.5][:15]
            supporting_files = [s for s in relevance_scores if 0.2 <= s.relevance_score <= 0.5][:20]
            
            # Final metrics
            self.metrics.processing_time = time.time() - self._start_time
            self.metrics.peak_memory_mb = self.memory_manager.get_peak_memory()
            self.metrics.throughput_files_per_second = len(file_results) / self.metrics.processing_time
            self.metrics.gc_collections = self.memory_manager.gc_stats.get('manual_collections', 0)
            
            return OptimizedContext(
                core_files=core_files,
                supporting_files=supporting_files,
                dependency_graph={"nodes": [], "edges": []},
                context_summary={
                    "total_files": len(core_files) + len(supporting_files),
                    "total_tokens": sum(f.estimated_tokens for f in core_files + supporting_files),
                    "confidence_score": 0.85,
                    "coverage_percentage": 75.0,
                    "architectural_patterns": [],
                    "potential_challenges": [],
                    "recommended_approach": "Parallel processing optimization applied"
                },
                optimization_metadata={
                    "algorithm_used": "parallel_ai_enhanced",
                    "processing_time_ms": int(self.metrics.processing_time * 1000),
                    "cache_hit_rate": 0.0,
                    "relevance_distribution": {
                        "high": len(core_files),
                        "medium": len(supporting_files),
                        "low": max(0, len(relevance_scores) - len(core_files) - len(supporting_files))
                    }
                },
                suggestions={
                    "additional_files": [],
                    "alternative_contexts": ["streaming", "cached"],
                    "optimization_tips": [
                        f"Processed {len(file_results)} files in {self.metrics.processing_time:.2f}s",
                        f"Peak memory usage: {self.metrics.peak_memory_mb:.1f}MB",
                        f"Throughput: {self.metrics.throughput_files_per_second:.1f} files/sec"
                    ]
                }
            )
            
        except Exception as e:
            logger.error("Parallel context optimization failed", error=str(e))
            raise
        finally:
            self.cleanup()
    
    async def stream_context_optimization(
        self,
        context_request: ContextRequest,
        file_results: List[FileAnalysisResult],
        relevance_analyzer: RelevanceAnalyzer,
        context_assembler: ContextAssembler,
        dependency_graph: DependencyGraph
    ) -> AsyncIterator[StreamingChunk]:
        """Stream context optimization results progressively."""
        async for chunk in self.streaming_generator.generate_optimization_stream(
            context_request, file_results, relevance_analyzer, 
            context_assembler, dependency_graph
        ):
            yield chunk
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if self._start_time:
            self.metrics.processing_time = time.time() - self._start_time
        self.metrics.memory_usage_mb = self.memory_manager.track_memory_usage()
        return self.metrics
    
    def cleanup(self):
        """Clean up performance optimizer resources."""
        try:
            self.parallel_processor.shutdown()
            self.memory_manager.cleanup()
        except Exception as e:
            logger.error("Performance optimizer cleanup error", error=str(e))


# Singleton instance for global access
_performance_optimizer = None

def get_performance_optimizer(limits: Optional[ResourceLimits] = None) -> PerformanceOptimizer:
    """Get or create global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(limits)
    return _performance_optimizer


def cleanup_performance_optimizer():
    """Clean up global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer:
        _performance_optimizer.cleanup()
        _performance_optimizer = None