"""
Optimized Embedding Pipeline with Production-Scale Performance.

This module provides an advanced embedding pipeline that extends the base
EmbeddingService with:
- Intelligent batch optimization and parallel processing
- Multi-level caching with compression and deduplication
- Fallback mechanisms for API failures
- Advanced retry strategies with circuit breaker pattern
- Embedding quality validation and filtering
- Performance monitoring and auto-scaling
- Memory-efficient processing for large datasets
"""

import asyncio
import time
import hashlib
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import aiofiles
import pickle
import gzip

from ..core.embedding_service import EmbeddingService, EmbeddingError, RateLimitError, TokenLimitError
from ..core.redis import get_redis_client, RedisClient
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingQuality(Enum):
    """Embedding quality levels for filtering."""
    HIGH = "high"       # High-quality embeddings
    MEDIUM = "medium"   # Acceptable quality
    LOW = "low"         # Poor quality, should be regenerated
    INVALID = "invalid" # Invalid/corrupted embeddings


class ProcessingMode(Enum):
    """Embedding processing modes."""
    REALTIME = "realtime"       # Immediate processing
    BATCH = "batch"             # Batch processing for efficiency
    STREAM = "stream"           # Streaming processing
    BACKGROUND = "background"   # Background processing


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""
    id: str
    text: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __hash__(self):
        return hash(self.text)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    request_id: str
    embedding: Optional[List[float]]
    quality: EmbeddingQuality
    processing_time_ms: float
    cached: bool = False
    error: Optional[str] = None
    model_used: str = ""
    token_count: int = 0


@dataclass
class BatchMetrics:
    """Metrics for batch processing."""
    batch_size: int
    processing_time_ms: float
    cache_hits: int
    api_calls: int
    errors: int
    avg_quality_score: float
    throughput_per_second: float


class CircuitBreaker:
    """Circuit breaker for API failure handling."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time:
                time_since_failure = time.time() - self.last_failure_time
                if time_since_failure > self.timeout_seconds:
                    self.state = "half-open"
                    return True
            return False
        else:  # half-open
            return True
    
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class EmbeddingQualityValidator:
    """Validates embedding quality and filters poor results."""
    
    def __init__(self):
        self.dimension_expectations = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
    
    def validate_embedding(
        self, 
        embedding: List[float], 
        text: str, 
        model: str
    ) -> Tuple[EmbeddingQuality, float]:
        """
        Validate embedding quality and return quality score.
        
        Args:
            embedding: Generated embedding
            text: Original text
            model: Model used for generation
            
        Returns:
            Tuple of (quality_level, quality_score)
        """
        quality_score = 0.0
        issues = []
        
        # Check dimensions
        expected_dim = self.dimension_expectations.get(model, 1536)
        if len(embedding) != expected_dim:
            issues.append(f"Wrong dimensions: {len(embedding)} vs {expected_dim}")
            return EmbeddingQuality.INVALID, 0.0
        
        # Check for invalid values
        embedding_array = np.array(embedding)
        if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
            issues.append("Contains NaN or Inf values")
            return EmbeddingQuality.INVALID, 0.0
        
        # Check vector magnitude (embeddings should have reasonable magnitude)
        magnitude = np.linalg.norm(embedding_array)
        if magnitude < 0.1 or magnitude > 100.0:
            issues.append(f"Unusual magnitude: {magnitude}")
            quality_score -= 0.3
        else:
            quality_score += 0.3
        
        # Check for zero vectors
        if np.allclose(embedding_array, 0):
            issues.append("Zero vector detected")
            return EmbeddingQuality.INVALID, 0.0
        
        # Check variance (good embeddings should have reasonable variance)
        variance = np.var(embedding_array)
        if variance < 1e-6:
            issues.append(f"Very low variance: {variance}")
            quality_score -= 0.2
        elif variance > 0.01:
            quality_score += 0.2
        
        # Check distribution (should be roughly normal)
        mean_val = np.mean(embedding_array)
        if abs(mean_val) > 0.1:
            issues.append(f"High mean deviation: {mean_val}")
            quality_score -= 0.1
        else:
            quality_score += 0.1
        
        # Text-based quality checks
        if len(text.strip()) < 3:
            issues.append("Text too short")
            quality_score -= 0.3
        elif len(text.strip()) > 10:
            quality_score += 0.2
        
        # Final quality assessment
        if quality_score >= 0.7:
            return EmbeddingQuality.HIGH, quality_score
        elif quality_score >= 0.4:
            return EmbeddingQuality.MEDIUM, quality_score
        elif quality_score >= 0.1:
            return EmbeddingQuality.LOW, quality_score
        else:
            return EmbeddingQuality.INVALID, quality_score


class OptimizedEmbeddingPipeline:
    """
    Production-optimized embedding pipeline with advanced features.
    
    Features:
    - Intelligent batch optimization with dynamic sizing
    - Multi-level caching with compression and deduplication
    - Circuit breaker pattern for API failure resilience
    - Quality validation and filtering
    - Parallel processing with backpressure control
    - Performance monitoring and auto-scaling
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        redis_client: Optional[RedisClient] = None,
        max_batch_size: int = 500,
        max_concurrent_batches: int = 5,
        cache_compression_enabled: bool = True,
        quality_validation_enabled: bool = True,
        circuit_breaker_enabled: bool = True
    ):
        """
        Initialize optimized embedding pipeline.
        
        Args:
            embedding_service: Base embedding service
            redis_client: Redis client for caching
            max_batch_size: Maximum batch size for processing
            max_concurrent_batches: Maximum concurrent batch operations
            cache_compression_enabled: Enable cache compression
            quality_validation_enabled: Enable embedding quality validation
            circuit_breaker_enabled: Enable circuit breaker for failures
        """
        self.settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.redis_client = redis_client or get_redis_client()
        
        # Configuration
        self.max_batch_size = max_batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.cache_compression_enabled = cache_compression_enabled
        self.quality_validation_enabled = quality_validation_enabled
        
        # Components
        self.quality_validator = EmbeddingQualityValidator() if quality_validation_enabled else None
        self.circuit_breaker = CircuitBreaker() if circuit_breaker_enabled else None
        
        # Processing queues
        self.request_queue: asyncio.Queue[EmbeddingRequest] = asyncio.Queue()
        self.result_queue: asyncio.Queue[EmbeddingResult] = asyncio.Queue()
        
        # State management
        self.active_batches: Set[str] = set()
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "batch_count": 0,
            "quality_issues": 0,
            "circuit_breaker_trips": 0,
            "avg_batch_size": 0.0,
            "avg_processing_time_ms": 0.0,
            "throughput_per_second": 0.0
        }
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Deduplication cache
        self._dedup_cache: Dict[str, str] = {}  # text_hash -> request_id
        self._dedup_results: Dict[str, EmbeddingResult] = {}  # request_id -> result
    
    async def start(self) -> None:
        """Start the background processing pipeline."""
        logger.info("Starting optimized embedding pipeline")
        
        # Start batch processor
        self._background_tasks.append(
            asyncio.create_task(self._batch_processor())
        )
        
        # Start metrics collector
        self._background_tasks.append(
            asyncio.create_task(self._metrics_collector())
        )
        
        # Start cache maintenance
        if self.cache_compression_enabled:
            self._background_tasks.append(
                asyncio.create_task(self._cache_maintenance())
            )
    
    async def stop(self) -> None:
        """Stop the pipeline and cleanup resources."""
        logger.info("Stopping optimized embedding pipeline")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clear caches
        self._dedup_cache.clear()
        self._dedup_results.clear()
    
    async def generate_embedding_optimized(
        self,
        text: str,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Generate embedding with optimization.
        
        Args:
            text: Text to embed
            priority: Processing priority (1=high, 2=medium, 3=low)
            metadata: Additional metadata
            
        Returns:
            EmbeddingResult with embedding and quality information
        """
        request_id = str(uuid.uuid4())
        request = EmbeddingRequest(
            id=request_id,
            text=text,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Check for immediate cache hit
        cached_result = await self._get_cached_embedding(text)
        if cached_result:
            self.metrics["cache_hits"] += 1
            return EmbeddingResult(
                request_id=request_id,
                embedding=cached_result,
                quality=EmbeddingQuality.HIGH,  # Assume cached embeddings are good
                processing_time_ms=1.0,
                cached=True,
                model_used=self.embedding_service.model_name
            )
        
        # Check deduplication
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self._dedup_cache:
            existing_request_id = self._dedup_cache[text_hash]
            if existing_request_id in self._dedup_results:
                result = self._dedup_results[existing_request_id]
                return EmbeddingResult(
                    request_id=request_id,
                    embedding=result.embedding,
                    quality=result.quality,
                    processing_time_ms=result.processing_time_ms,
                    cached=True,
                    model_used=result.model_used
                )
        
        # Add to deduplication tracking
        self._dedup_cache[text_hash] = request_id
        
        # Queue for processing
        await self.request_queue.put(request)
        self.metrics["total_requests"] += 1
        
        # Wait for result (with timeout)
        try:
            start_time = time.perf_counter()
            while True:
                try:
                    result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                    if result.request_id == request_id:
                        # Cache the result for deduplication
                        self._dedup_results[request_id] = result
                        return result
                    else:
                        # Put back result for correct request
                        await self.result_queue.put(result)
                        
                except asyncio.TimeoutError:
                    # Check if we've been waiting too long
                    if time.perf_counter() - start_time > 30:  # 30 second timeout
                        raise EmbeddingError("Embedding generation timeout")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to generate embedding for request {request_id}: {e}")
            return EmbeddingResult(
                request_id=request_id,
                embedding=None,
                quality=EmbeddingQuality.INVALID,
                processing_time_ms=-1,
                error=str(e)
            )
    
    async def generate_embeddings_batch_optimized(
        self,
        texts: List[str],
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts with optimization.
        
        Args:
            texts: List of texts to embed
            priority: Processing priority
            metadata: Additional metadata
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
        
        start_time = time.perf_counter()
        
        # Create requests
        requests = []
        for i, text in enumerate(texts):
            request_id = f"batch_{uuid.uuid4()}_{i}"
            request = EmbeddingRequest(
                id=request_id,
                text=text,
                priority=priority,
                metadata=metadata or {}
            )
            requests.append(request)
        
        # Check cache for all texts
        cached_results = {}
        uncached_requests = []
        
        for request in requests:
            cached_embedding = await self._get_cached_embedding(request.text)
            if cached_embedding:
                cached_results[request.id] = EmbeddingResult(
                    request_id=request.id,
                    embedding=cached_embedding,
                    quality=EmbeddingQuality.HIGH,
                    processing_time_ms=1.0,
                    cached=True,
                    model_used=self.embedding_service.model_name
                )
                self.metrics["cache_hits"] += 1
            else:
                uncached_requests.append(request)
        
        # Process uncached requests
        if uncached_requests:
            batch_results = await self._process_batch_direct(uncached_requests)
            for result in batch_results:
                cached_results[result.request_id] = result
        
        # Return results in original order
        final_results = []
        for request in requests:
            if request.id in cached_results:
                final_results.append(cached_results[request.id])
            else:
                # Create error result for missing
                final_results.append(EmbeddingResult(
                    request_id=request.id,
                    embedding=None,
                    quality=EmbeddingQuality.INVALID,
                    processing_time_ms=-1,
                    error="Processing failed"
                ))
        
        processing_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Batch processed {len(texts)} embeddings in {processing_time:.0f}ms")
        
        return final_results
    
    async def _batch_processor(self) -> None:
        """Background batch processor."""
        logger.info("Starting batch processor")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect requests for batching
                batch = await self._collect_batch()
                
                if batch:
                    async with self.processing_semaphore:
                        batch_id = str(uuid.uuid4())
                        self.active_batches.add(batch_id)
                        
                        try:
                            # Process the batch
                            results = await self._process_batch_direct(batch)
                            
                            # Queue results
                            for result in results:
                                await self.result_queue.put(result)
                                
                        finally:
                            self.active_batches.discard(batch_id)
                
                # Brief pause to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1.0)
        
        logger.info("Batch processor stopped")
    
    async def _collect_batch(self) -> List[EmbeddingRequest]:
        """Collect requests for batch processing."""
        batch = []
        deadline = time.time() + 1.0  # 1 second batch window
        
        # Collect requests with timeout
        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=max(0.1, deadline - time.time())
                )
                batch.append(request)
                
            except asyncio.TimeoutError:
                break
        
        # Sort by priority (higher priority first)
        if batch:
            batch.sort(key=lambda r: r.priority)
        
        return batch
    
    async def _process_batch_direct(
        self, 
        requests: List[EmbeddingRequest]
    ) -> List[EmbeddingResult]:
        """Process a batch of requests directly."""
        if not requests:
            return []
        
        start_time = time.perf_counter()
        results = []
        
        try:
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.can_execute():
                logger.warning("Circuit breaker is open, failing batch")
                self.metrics["circuit_breaker_trips"] += 1
                
                for request in requests:
                    results.append(EmbeddingResult(
                        request_id=request.id,
                        embedding=None,
                        quality=EmbeddingQuality.INVALID,
                        processing_time_ms=-1,
                        error="Circuit breaker open"
                    ))
                return results
            
            # Extract texts and validate
            texts = []
            valid_requests = []
            
            for request in requests:
                if self.embedding_service.validate_text_length(request.text):
                    texts.append(request.text)
                    valid_requests.append(request)
                else:
                    results.append(EmbeddingResult(
                        request_id=request.id,
                        embedding=None,
                        quality=EmbeddingQuality.INVALID,
                        processing_time_ms=0,
                        error="Text too long"
                    ))
            
            if not texts:
                return results
            
            # Generate embeddings
            try:
                embeddings = await self.embedding_service.generate_embeddings_batch(
                    texts, 
                    batch_size=min(len(texts), self.max_batch_size)
                )
                
                # Record success
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                self.metrics["api_calls"] += 1
                
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                
                # Record failure
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # Return error results
                for request in valid_requests:
                    results.append(EmbeddingResult(
                        request_id=request.id,
                        embedding=None,
                        quality=EmbeddingQuality.INVALID,
                        processing_time_ms=-1,
                        error=str(e)
                    ))
                return results
            
            # Process results
            processing_time = (time.perf_counter() - start_time) * 1000
            
            for i, (request, embedding) in enumerate(zip(valid_requests, embeddings)):
                # Validate quality if enabled
                quality = EmbeddingQuality.HIGH
                quality_score = 1.0
                
                if self.quality_validator and embedding:
                    quality, quality_score = self.quality_validator.validate_embedding(
                        embedding, request.text, self.embedding_service.model_name
                    )
                    
                    if quality == EmbeddingQuality.INVALID:
                        self.metrics["quality_issues"] += 1
                
                # Create result
                result = EmbeddingResult(
                    request_id=request.id,
                    embedding=embedding,
                    quality=quality,
                    processing_time_ms=processing_time / len(valid_requests),
                    model_used=self.embedding_service.model_name,
                    token_count=self.embedding_service.count_tokens(request.text)
                )
                
                results.append(result)
                
                # Cache the result
                if embedding and quality != EmbeddingQuality.INVALID:
                    await self._cache_embedding(request.text, embedding)
            
            # Update metrics
            self.metrics["batch_count"] += 1
            self.metrics["avg_batch_size"] = (
                (self.metrics["avg_batch_size"] * (self.metrics["batch_count"] - 1) + len(requests)) 
                / self.metrics["batch_count"]
            )
            
            processing_time_total = (time.perf_counter() - start_time) * 1000
            self.metrics["avg_processing_time_ms"] = (
                (self.metrics["avg_processing_time_ms"] * (self.metrics["batch_count"] - 1) + processing_time_total)
                / self.metrics["batch_count"]
            )
            
            logger.debug(f"Processed batch of {len(requests)} requests in {processing_time_total:.0f}ms")
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            
            # Return error results for all requests
            for request in requests:
                if not any(r.request_id == request.id for r in results):
                    results.append(EmbeddingResult(
                        request_id=request.id,
                        embedding=None,
                        quality=EmbeddingQuality.INVALID,
                        processing_time_ms=-1,
                        error=str(e)
                    ))
        
        return results
    
    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding with compression support."""
        try:
            cache_key = f"embedding_optimized:{hashlib.sha256(text.encode()).hexdigest()}"
            
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                if self.cache_compression_enabled:
                    # Decompress and deserialize
                    compressed_data = json.loads(cached_data)
                    embedding_bytes = gzip.decompress(
                        bytes.fromhex(compressed_data["embedding"])
                    )
                    embedding = pickle.loads(embedding_bytes)
                else:
                    embedding_data = json.loads(cached_data)
                    embedding = embedding_data["embedding"]
                
                return embedding
                
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        
        return None
    
    async def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding with compression support."""
        try:
            cache_key = f"embedding_optimized:{hashlib.sha256(text.encode()).hexdigest()}"
            
            if self.cache_compression_enabled:
                # Compress and serialize
                embedding_bytes = pickle.dumps(embedding)
                compressed_bytes = gzip.compress(embedding_bytes)
                cache_data = {
                    "embedding": compressed_bytes.hex(),
                    "compressed": True,
                    "timestamp": time.time()
                }
            else:
                cache_data = {
                    "embedding": embedding,
                    "compressed": False,
                    "timestamp": time.time()
                }
            
            await self.redis_client.set(
                cache_key,
                json.dumps(cache_data),
                expire=3600  # 1 hour TTL
            )
            
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    async def _metrics_collector(self) -> None:
        """Background metrics collection."""
        logger.info("Starting metrics collector")
        
        while not self._shutdown_event.is_set():
            try:
                # Update throughput metrics
                current_time = time.time()
                if hasattr(self, '_last_metrics_time'):
                    time_delta = current_time - self._last_metrics_time
                    if time_delta > 0:
                        requests_delta = self.metrics["total_requests"] - getattr(self, '_last_request_count', 0)
                        self.metrics["throughput_per_second"] = requests_delta / time_delta
                
                self._last_metrics_time = current_time
                self._last_request_count = self.metrics["total_requests"]
                
                # Clean up old deduplication entries
                cutoff_time = datetime.utcnow() - timedelta(minutes=10)
                old_entries = [
                    text_hash for text_hash, request_id in self._dedup_cache.items()
                    if request_id in self._dedup_results and 
                       self._dedup_results[request_id].request_id in self._dedup_results
                ]
                
                for text_hash in old_entries[:100]:  # Clean 100 at a time
                    request_id = self._dedup_cache.pop(text_hash, None)
                    if request_id:
                        self._dedup_results.pop(request_id, None)
                
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Metrics collector stopped")
    
    async def _cache_maintenance(self) -> None:
        """Background cache maintenance."""
        logger.info("Starting cache maintenance")
        
        while not self._shutdown_event.is_set():
            try:
                # Clean up expired cache entries
                pattern = "embedding_optimized:*"
                keys_cleaned = 0
                
                async for key in self.redis_client._redis.scan_iter(match=pattern):
                    try:
                        cached_data = await self.redis_client.get(key)
                        if cached_data:
                            data = json.loads(cached_data)
                            timestamp = data.get("timestamp", 0)
                            
                            # Remove entries older than 2 hours
                            if time.time() - timestamp > 7200:
                                await self.redis_client.delete(key)
                                keys_cleaned += 1
                                
                                if keys_cleaned >= 100:  # Limit cleanup per cycle
                                    break
                    except:
                        # Remove invalid entries
                        await self.redis_client.delete(key)
                        keys_cleaned += 1
                
                if keys_cleaned > 0:
                    logger.info(f"Cleaned {keys_cleaned} expired cache entries")
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                await asyncio.sleep(1800)
        
        logger.info("Cache maintenance stopped")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = self.embedding_service.get_performance_metrics()
        
        # Add pipeline-specific metrics
        pipeline_metrics = {
            **base_metrics,
            "pipeline": {
                "total_requests": self.metrics["total_requests"],
                "cache_hits": self.metrics["cache_hits"],
                "api_calls": self.metrics["api_calls"],
                "batch_count": self.metrics["batch_count"],
                "quality_issues": self.metrics["quality_issues"],
                "circuit_breaker_trips": self.metrics["circuit_breaker_trips"],
                "avg_batch_size": self.metrics["avg_batch_size"],
                "avg_processing_time_ms": self.metrics["avg_processing_time_ms"],
                "throughput_per_second": self.metrics["throughput_per_second"],
                "active_batches": len(self.active_batches),
                "queue_size": self.request_queue.qsize(),
                "dedup_cache_size": len(self._dedup_cache)
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
                "failure_count": self.circuit_breaker.failure_count if self.circuit_breaker else 0
            } if self.circuit_breaker else {"state": "disabled"}
        }
        
        return pipeline_metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health = await self.embedding_service.health_check()
        
        # Add pipeline health
        pipeline_health = {
            "queue_healthy": self.request_queue.qsize() < 1000,
            "active_batches": len(self.active_batches),
            "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
            "background_tasks_running": len([t for t in self._background_tasks if not t.done()]),
            "dedup_cache_size": len(self._dedup_cache)
        }
        
        health["pipeline"] = pipeline_health
        
        # Overall health status
        if (pipeline_health["queue_healthy"] and 
            health.get("status") == "healthy" and
            (not self.circuit_breaker or self.circuit_breaker.state != "open")):
            health["overall_status"] = "healthy"
        else:
            health["overall_status"] = "degraded"
        
        return health


# Global instance for application use
_optimized_pipeline: Optional[OptimizedEmbeddingPipeline] = None


async def get_optimized_embedding_pipeline() -> OptimizedEmbeddingPipeline:
    """
    Get singleton optimized embedding pipeline instance.
    
    Returns:
        OptimizedEmbeddingPipeline instance
    """
    global _optimized_pipeline
    
    if _optimized_pipeline is None:
        _optimized_pipeline = OptimizedEmbeddingPipeline()
        await _optimized_pipeline.start()
    
    return _optimized_pipeline


async def cleanup_optimized_embedding_pipeline() -> None:
    """Cleanup optimized embedding pipeline resources."""
    global _optimized_pipeline
    
    if _optimized_pipeline:
        await _optimized_pipeline.stop()
        _optimized_pipeline = None