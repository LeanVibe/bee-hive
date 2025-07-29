"""
Semantic Embedding Service for LeanVibe Agent Hive 2.0

High-performance embedding generation with OpenAI text-embedding-ada-002 integration,
batch processing optimization for >500 docs/sec ingestion, and comprehensive caching.

Features:
- OpenAI text-embedding-ada-002 integration with retries
- Async batch processing with thread pool optimization
- Intelligent caching and deduplication
- Performance monitoring and rate limit handling
- Content preprocessing and token optimization
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import openai
import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingConfig:
    """Configuration for embedding service."""
    
    def __init__(self):
        self.model = "text-embedding-ada-002"
        self.dimensions = 1536
        self.max_tokens = 8191  # Model limit
        self.batch_size = 50  # Optimal for OpenAI API
        self.max_concurrent_requests = 10
        self.rate_limit_rpm = 3000  # Requests per minute
        self.rate_limit_tpm = 1000000  # Tokens per minute
        self.retry_attempts = 3
        self.cache_ttl_hours = 24
        self.content_preprocessing = True
        self.token_optimization = True
        
        # Performance targets
        self.target_throughput_docs_per_sec = 500.0
        self.target_avg_latency_ms = 100.0
        self.target_cache_hit_rate = 0.3


class EmbeddingCache:
    """In-memory cache for embeddings with TTL and deduplication."""
    
    def __init__(self, ttl_hours: int = 24):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl_seconds = ttl_hours * 3600
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()
    
    def _get_cache_key(self, content: str, model: str) -> str:
        """Generate cache key from content and model."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"{model}:{content_hash}"
    
    async def get(self, content: str, model: str) -> Optional[List[float]]:
        """Get cached embedding."""
        async with self._lock:
            cache_key = self._get_cache_key(content, model)
            
            if cache_key in self._cache:
                cached_item = self._cache[cache_key]
                
                # Check TTL
                if time.time() - cached_item['timestamp'] < self._ttl_seconds:
                    self._hits += 1
                    return cached_item['embedding']
                else:
                    # Remove expired item
                    del self._cache[cache_key]
            
            self._misses += 1
            return None
    
    async def set(self, content: str, model: str, embedding: List[float]):
        """Cache embedding."""
        async with self._lock:
            cache_key = self._get_cache_key(content, model)
            self._cache[cache_key] = {
                'embedding': embedding,
                'timestamp': time.time()
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_size': len(self._cache),
                'cache_hits': self._hits,
                'cache_misses': self._misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    async def cleanup_expired(self):
        """Remove expired cache entries."""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, item in self._cache.items()
                if current_time - item['timestamp'] >= self._ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class RateLimiter:
    """Token-aware rate limiter for OpenAI API."""
    
    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self.request_times: List[float] = []
        self.token_usage: List[Tuple[float, int]] = []  # (timestamp, tokens)
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int) -> float:
        """
        Acquire rate limit permission.
        
        Args:
            tokens: Number of tokens for the request
            
        Returns:
            Delay in seconds if rate limited, 0 if OK
        """
        async with self._lock:
            current_time = time.time()
            
            # Clean old entries (older than 1 minute)
            cutoff_time = current_time - 60.0
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
            
            # Check request rate limit
            if len(self.request_times) >= self.rpm:
                oldest_request = min(self.request_times)
                delay = 60.0 - (current_time - oldest_request)
                if delay > 0:
                    return delay
            
            # Check token rate limit
            total_tokens = sum(tokens for _, tokens in self.token_usage)
            if total_tokens + tokens > self.tpm:
                oldest_token_time = min(t for t, _ in self.token_usage) if self.token_usage else current_time
                delay = 60.0 - (current_time - oldest_token_time)
                if delay > 0:
                    return delay
            
            # Record usage
            self.request_times.append(current_time)
            self.token_usage.append((current_time, tokens))
            
            return 0.0


class SemanticEmbeddingService:
    """
    High-performance semantic embedding service.
    
    Provides optimized embedding generation with OpenAI integration,
    intelligent caching, batch processing, and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.cache = EmbeddingCache(self.config.cache_ttl_hours)
        self.rate_limiter = RateLimiter(self.config.rate_limit_rpm, self.config.rate_limit_tpm)
        
        # Initialize OpenAI client
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Token encoder for text processing
        self.tokenizer = tiktoken.encoding_for_model(self.config.model)
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_latency_ms': 0.0,
            'throughput_docs_per_sec': 0.0,
            'last_reset': time.time()
        }
        
        # Semaphore for concurrent request limiting
        self.request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        logger.info("✅ Semantic Embedding Service initialized")
    
    def _preprocess_content(self, content: str) -> str:
        """
        Preprocess content for optimal embedding generation.
        
        Args:
            content: Raw content text
            
        Returns:
            Preprocessed content
        """
        if not self.config.content_preprocessing:
            return content
        
        # Remove excessive whitespace
        content = ' '.join(content.split())
        
        # Truncate if too long (leave room for potential additions)
        max_chars = self.config.max_tokens * 3  # Rough character to token ratio
        if len(content) > max_chars:
            content = content[:max_chars].rsplit(' ', 1)[0]  # Don't cut words
        
        return content.strip()
    
    def _count_tokens(self, content: str) -> int:
        """Count tokens in content."""
        try:
            return len(self.tokenizer.encode(content))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough estimation
            return len(content.split()) * 1.3
    
    def _optimize_content_for_tokens(self, content: str) -> str:
        """
        Optimize content to fit within token limits.
        
        Args:
            content: Input content
            
        Returns:
            Optimized content within token limits
        """
        if not self.config.token_optimization:
            return content
        
        token_count = self._count_tokens(content)
        
        if token_count <= self.config.max_tokens:
            return content
        
        # Truncate content to fit token limit
        words = content.split()
        target_words = int(len(words) * (self.config.max_tokens / token_count))
        
        optimized_content = ' '.join(words[:target_words])
        
        # Verify token count
        if self._count_tokens(optimized_content) > self.config.max_tokens:
            # More aggressive truncation
            target_words = int(target_words * 0.9)
            optimized_content = ' '.join(words[:target_words])
        
        logger.debug(f"Content optimized: {token_count} -> {self._count_tokens(optimized_content)} tokens")
        
        return optimized_content
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    async def _generate_embedding_with_retry(self, content: str) -> List[float]:
        """
        Generate embedding with retry logic.
        
        Args:
            content: Content to embed
            
        Returns:
            Embedding vector
        """
        token_count = self._count_tokens(content)
        
        # Rate limiting
        delay = await self.rate_limiter.acquire(token_count)
        if delay > 0:
            logger.debug(f"Rate limited, waiting {delay:.2f}s")
            await asyncio.sleep(delay)
        
        async with self.request_semaphore:
            try:
                response = await self.client.embeddings.create(
                    model=self.config.model,
                    input=content,
                    encoding_format="float"
                )
                
                embedding = response.data[0].embedding
                
                # Update metrics
                self.metrics['successful_requests'] += 1
                self.metrics['total_tokens'] += token_count
                
                return embedding
                
            except Exception as e:
                self.metrics['failed_requests'] += 1
                logger.error(f"Embedding generation failed: {e}")
                raise
    
    async def generate_embedding(self, content: str) -> Optional[List[float]]:
        """
        Generate embedding for a single piece of content.
        
        Args:
            content: Content to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            start_time = time.time()
            
            # Preprocess content
            processed_content = self._preprocess_content(content)
            optimized_content = self._optimize_content_for_tokens(processed_content)
            
            # Check cache first
            cached_embedding = await self.cache.get(optimized_content, self.config.model)
            if cached_embedding:
                self.metrics['cache_hits'] += 1
                logger.debug("Cache hit for embedding")
                return cached_embedding
            
            self.metrics['cache_misses'] += 1
            self.metrics['total_requests'] += 1
            
            # Generate embedding
            embedding = await self._generate_embedding_with_retry(optimized_content)
            
            # Cache the result
            await self.cache.set(optimized_content, self.config.model, embedding)
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_latency_metric(processing_time)
            
            logger.debug(f"Embedding generated in {processing_time:.2f}ms")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def generate_embeddings_batch(
        self,
        contents: List[str],
        batch_id: Optional[str] = None
    ) -> Tuple[List[Optional[List[float]]], Dict[str, Any]]:
        """
        Generate embeddings for multiple contents with optimal batching.
        
        Args:
            contents: List of content strings
            batch_id: Optional batch identifier for tracking
            
        Returns:
            Tuple of (embeddings_list, batch_stats)
        """
        try:
            start_time = time.time()
            batch_id = batch_id or str(uuid.uuid4())
            
            logger.info(f"🚀 Starting batch embedding generation: {len(contents)} documents")
            
            # Preprocess all contents
            processed_contents = []
            for content in contents:
                processed = self._preprocess_content(content)
                optimized = self._optimize_content_for_tokens(processed)
                processed_contents.append(optimized)
            
            # Check cache for all contents
            embeddings = []
            cache_tasks = []
            uncached_indices = []
            
            for i, content in enumerate(processed_contents):
                cache_tasks.append(self.cache.get(content, self.config.model))
            
            cached_results = await asyncio.gather(*cache_tasks)
            
            for i, cached_embedding in enumerate(cached_results):
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    self.metrics['cache_hits'] += 1
                else:
                    embeddings.append(None)
                    uncached_indices.append(i)
                    self.metrics['cache_misses'] += 1
            
            if uncached_indices:
                logger.info(f"📡 Generating embeddings for {len(uncached_indices)} uncached documents")
                
                # Process uncached contents in batches
                for batch_start in range(0, len(uncached_indices), self.config.batch_size):
                    batch_end = min(batch_start + self.config.batch_size, len(uncached_indices))
                    batch_indices = uncached_indices[batch_start:batch_end]
                    
                    # Create tasks for this batch
                    batch_tasks = []
                    for idx in batch_indices:
                        content = processed_contents[idx]
                        task = self._generate_embedding_with_retry(content)
                        batch_tasks.append(task)
                    
                    # Execute batch with rate limiting
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results and cache them
                    for i, result in enumerate(batch_results):
                        idx = batch_indices[i]
                        
                        if isinstance(result, Exception):
                            logger.error(f"Batch embedding failed for index {idx}: {result}")
                            embeddings[idx] = None
                        else:
                            embeddings[idx] = result
                            # Cache the result
                            await self.cache.set(
                                processed_contents[idx], 
                                self.config.model, 
                                result
                            )
            
            # Calculate batch statistics
            processing_time = (time.time() - start_time) * 1000
            successful_count = sum(1 for e in embeddings if e is not None)
            failed_count = len(embeddings) - successful_count
            throughput = len(contents) / (processing_time / 1000) if processing_time > 0 else 0
            
            batch_stats = {
                'batch_id': batch_id,
                'total_documents': len(contents),
                'successful_embeddings': successful_count,
                'failed_embeddings': failed_count,
                'processing_time_ms': processing_time,
                'throughput_docs_per_sec': throughput,
                'cache_hits': sum(1 for i in range(len(embeddings)) if i not in uncached_indices),
                'cache_misses': len(uncached_indices),
                'model': self.config.model
            }
            
            # Update global metrics
            self._update_throughput_metric(throughput)
            
            logger.info(f"✅ Batch completed: {successful_count}/{len(contents)} successful")
            logger.info(f"📊 Throughput: {throughput:.1f} docs/sec")
            
            return embeddings, batch_stats
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(contents), {
                'batch_id': batch_id or 'unknown',
                'error': str(e),
                'total_documents': len(contents),
                'successful_embeddings': 0,
                'failed_embeddings': len(contents)
            }
    
    def _update_latency_metric(self, latency_ms: float):
        """Update average latency metric."""
        if self.metrics['successful_requests'] == 1:
            self.metrics['avg_latency_ms'] = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['avg_latency_ms'] = (
                alpha * latency_ms + 
                (1 - alpha) * self.metrics['avg_latency_ms']
            )
    
    def _update_throughput_metric(self, throughput: float):
        """Update throughput metric."""
        if self.metrics['throughput_docs_per_sec'] == 0:
            self.metrics['throughput_docs_per_sec'] = throughput
        else:
            # Exponential moving average
            alpha = 0.2
            self.metrics['throughput_docs_per_sec'] = (
                alpha * throughput + 
                (1 - alpha) * self.metrics['throughput_docs_per_sec']
            )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = await self.cache.get_stats()
        
        current_time = time.time()
        uptime_seconds = current_time - self.metrics['last_reset']
        
        return {
            'embedding_service': {
                'model': self.config.model,
                'total_requests': self.metrics['total_requests'],
                'successful_requests': self.metrics['successful_requests'],
                'failed_requests': self.metrics['failed_requests'],
                'success_rate': (
                    self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1)
                ),
                'total_tokens_processed': self.metrics['total_tokens'],
                'avg_latency_ms': self.metrics['avg_latency_ms'],
                'throughput_docs_per_sec': self.metrics['throughput_docs_per_sec'],
                'uptime_seconds': uptime_seconds
            },
            'cache_performance': cache_stats,
            'performance_targets': {
                'target_throughput_docs_per_sec': self.config.target_throughput_docs_per_sec,
                'target_avg_latency_ms': self.config.target_avg_latency_ms,
                'target_cache_hit_rate': self.config.target_cache_hit_rate,
                'throughput_achievement': (
                    self.metrics['throughput_docs_per_sec'] / self.config.target_throughput_docs_per_sec
                ),
                'latency_achievement': (
                    self.config.target_avg_latency_ms / max(self.metrics['avg_latency_ms'], 1)
                ),
                'cache_hit_rate_achievement': (
                    cache_stats['hit_rate'] / self.config.target_cache_hit_rate
                    if self.config.target_cache_hit_rate > 0 else 1.0
                )
            },
            'configuration': {
                'batch_size': self.config.batch_size,
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'cache_ttl_hours': self.config.cache_ttl_hours,
                'rate_limit_rpm': self.config.rate_limit_rpm,
                'rate_limit_tpm': self.config.rate_limit_tpm
            }
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            # Test embedding generation with a simple query
            test_content = "Health check test content"
            start_time = time.time()
            
            test_embedding = await self.generate_embedding(test_content)
            
            response_time = (time.time() - start_time) * 1000
            is_healthy = test_embedding is not None and len(test_embedding) == self.config.dimensions
            
            cache_stats = await self.cache.get_stats()
            
            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'response_time_ms': response_time,
                'embedding_dimensions': len(test_embedding) if test_embedding else 0,
                'cache_size': cache_stats['cache_size'],
                'active_requests': self.config.max_concurrent_requests - self.request_semaphore._value,
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }
    
    async def cleanup_cache(self):
        """Clean up expired cache entries."""
        await self.cache.cleanup_expired()
    
    async def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_latency_ms': 0.0,
            'throughput_docs_per_sec': 0.0,
            'last_reset': time.time()
        }
        
        # Reset cache stats
        async with self.cache._lock:
            self.cache._hits = 0
            self.cache._misses = 0
        
        logger.info("📊 Performance metrics reset")
    
    async def shutdown(self):
        """Shutdown the embedding service."""
        await self.client.close()
        self.thread_pool.shutdown(wait=True)
        logger.info("🛑 Semantic Embedding Service shutdown completed")


# Global instance
_embedding_service: Optional[SemanticEmbeddingService] = None

async def get_embedding_service() -> SemanticEmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = SemanticEmbeddingService()
    
    return _embedding_service

async def cleanup_embedding_service():
    """Clean up the global embedding service."""
    global _embedding_service
    
    if _embedding_service:
        await _embedding_service.shutdown()
        _embedding_service = None