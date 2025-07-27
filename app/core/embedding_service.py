"""
Production-Ready OpenAI Embedding Service for Context Engine.

Provides text embedding generation using OpenAI's text-embedding-ada-002 model
with Redis caching, batch processing, rate limiting, retry logic, and comprehensive
error handling for optimal performance and reliability.
"""

import asyncio
import hashlib
import json
import time
import random
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

import openai
from openai import AsyncOpenAI
import tiktoken
from prometheus_client import Counter, Histogram, Gauge

from ..core.config import get_settings
from ..core.redis import get_redis_client, RedisClient


logger = logging.getLogger(__name__)

# Prometheus metrics for monitoring (lazy initialization to avoid duplicates)
_metrics_initialized = False
embedding_requests_total = None
embedding_duration_seconds = None
embedding_cache_hits_total = None
embedding_tokens_processed = None
embedding_batch_size = None
embedding_rate_limit_errors = None
embedding_cache_size = None

def _init_metrics():
    """Initialize Prometheus metrics if not already done."""
    global _metrics_initialized, embedding_requests_total, embedding_duration_seconds
    global embedding_cache_hits_total, embedding_tokens_processed, embedding_batch_size
    global embedding_rate_limit_errors, embedding_cache_size
    
    if not _metrics_initialized:
        try:
            embedding_requests_total = Counter('embedding_requests_total', 'Total embedding requests', ['model', 'status'])
            embedding_duration_seconds = Histogram('embedding_duration_seconds', 'Time spent generating embeddings', ['model'])
            embedding_cache_hits_total = Counter('embedding_cache_hits_total', 'Total cache hits', ['cache_type'])
            embedding_tokens_processed = Counter('embedding_tokens_processed_total', 'Total tokens processed')
            embedding_batch_size = Histogram('embedding_batch_size', 'Batch sizes used for embeddings')
            embedding_rate_limit_errors = Counter('embedding_rate_limit_errors_total', 'Rate limit errors encountered')
            embedding_cache_size = Gauge('embedding_cache_size', 'Current cache size', ['cache_type'])
            _metrics_initialized = True
        except ValueError:
            # Metrics already exist, get them from registry
            from prometheus_client import REGISTRY
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name'):
                    if collector._name == 'embedding_requests_total':
                        embedding_requests_total = collector
                    elif collector._name == 'embedding_duration_seconds':
                        embedding_duration_seconds = collector
                    elif collector._name == 'embedding_cache_hits_total':
                        embedding_cache_hits_total = collector
                    elif collector._name == 'embedding_tokens_processed_total':
                        embedding_tokens_processed = collector
                    elif collector._name == 'embedding_batch_size':
                        embedding_batch_size = collector
                    elif collector._name == 'embedding_rate_limit_errors_total':
                        embedding_rate_limit_errors = collector
                    elif collector._name == 'embedding_cache_size':
                        embedding_cache_size = collector
            _metrics_initialized = True


class EmbeddingError(Exception):
    """Base exception for embedding service errors."""
    pass


class RateLimitError(EmbeddingError):
    """Raised when API rate limit is exceeded."""
    pass


class TokenLimitError(EmbeddingError):
    """Raised when text exceeds token limits."""
    pass


class EmbeddingService:
    """
    Production-ready service for generating text embeddings using OpenAI's embedding models.
    
    Features:
    - Redis-based caching with memory fallback
    - Intelligent batch processing for efficiency  
    - Rate limiting and exponential backoff retries
    - Comprehensive error handling and monitoring
    - Performance metrics and health checks
    - Token counting and validation
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        max_tokens: int = 8191,
        cache_ttl: int = 3600,
        redis_client: Optional[RedisClient] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        rate_limit_rpm: int = 3000
    ):
        """
        Initialize embedding service.
        
        Args:
            model_name: OpenAI embedding model to use
            max_tokens: Maximum tokens per embedding request
            cache_ttl: Cache time-to-live in seconds
            redis_client: Redis client for caching (optional)
            max_retries: Maximum retry attempts for failed requests
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            rate_limit_rpm: Rate limit in requests per minute
        """
        # Initialize metrics
        _init_metrics()
        
        self.settings = get_settings()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.rate_limit_rpm = rate_limit_rpm
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        
        # Initialize Redis client for caching
        self.redis = redis_client or get_redis_client()
        
        # Initialize tokenizer for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # In-memory cache for embeddings (fallback)
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self._request_times: List[float] = []
        
        # Performance metrics
        self._api_calls = 0
        self._cache_hits = 0
        self._redis_cache_hits = 0
        self._memory_cache_hits = 0
        self._total_tokens = 0
        self._failed_requests = 0
        
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text input with caching and error handling.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            TokenLimitError: If text is too long
            EmbeddingError: If text is empty or API call fails
            RateLimitError: If rate limit is exceeded
        """
        start_time = time.time()
        try:
            if not text or not text.strip():
                if embedding_requests_total:
                    embedding_requests_total.labels(model=self.model_name, status='error').inc()
                raise EmbeddingError("Text cannot be empty")
            
            # Check cache first
            cache_key = self._get_cache_key(text)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self._cache_hits += 1
                embedding_cache_hits_total.labels(cache_type='total').inc()
                embedding_requests_total.labels(model=self.model_name, status='cache_hit').inc()
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_result
            
            # Validate token count
            token_count = len(self.tokenizer.encode(text))
            if token_count > self.max_tokens:
                embedding_requests_total.labels(model=self.model_name, status='error').inc()
                raise TokenLimitError(f"Text too long: {token_count} tokens (max: {self.max_tokens})")
            
            # Check and enforce rate limits
            await self._enforce_rate_limit()
            
            # Generate embedding with retries
            embedding = await self._generate_with_retries(text, token_count)
            
            # Cache the result
            await self._cache_result(cache_key, embedding)
            
            # Update metrics
            self._api_calls += 1
            self._total_tokens += token_count
            embedding_tokens_processed.inc(token_count)
            embedding_requests_total.labels(model=self.model_name, status='success').inc()
            
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently with intelligent batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per API call
            
        Returns:
            List of embedding vectors in same order as input texts
            
        Raises:
            TokenLimitError: If any text is too long
            EmbeddingError: If any text is invalid or API call fails
            RateLimitError: If rate limit is exceeded
        """
        with embedding_duration_seconds.labels(model=self.model_name).time():
            if not texts:
                return []
            
            embedding_batch_size.observe(len(texts))
            all_embeddings = []
            
            # Optimize batch size based on token counts
            optimized_batch_size = await self._optimize_batch_size(texts, batch_size)
            
            # Process in batches
            for i in range(0, len(texts), optimized_batch_size):
                batch_texts = texts[i:i + optimized_batch_size]
                
                # Check cache for each text in batch
                batch_results = []
                uncached_texts = []
                uncached_indices = []
                
                for idx, text in enumerate(batch_texts):
                    if not text or not text.strip():
                        embedding_requests_total.labels(model=self.model_name, status='error').inc()
                        raise EmbeddingError(f"Text at index {i + idx} cannot be empty")
                    
                    cache_key = self._get_cache_key(text)
                    cached_result = await self._get_from_cache(cache_key)
                    
                    if cached_result:
                        batch_results.append(cached_result)
                        self._cache_hits += 1
                        embedding_cache_hits_total.labels(cache_type='total').inc()
                    else:
                        batch_results.append(None)  # Placeholder
                        uncached_texts.append(text)
                        uncached_indices.append(idx)
                
                # Generate embeddings for uncached texts
                if uncached_texts:
                    # Validate token counts
                    total_tokens = 0
                    for text in uncached_texts:
                        token_count = len(self.tokenizer.encode(text))
                        if token_count > self.max_tokens:
                            embedding_requests_total.labels(model=self.model_name, status='error').inc()
                            raise TokenLimitError(f"Text too long: {token_count} tokens (max: {self.max_tokens})")
                        total_tokens += token_count
                    
                    # Check and enforce rate limits
                    await self._enforce_rate_limit()
                    
                    # Generate embeddings with retries
                    embeddings = await self._generate_batch_with_retries(uncached_texts, total_tokens)
                    
                    # Cache and insert results
                    for i, embedding in enumerate(embeddings):
                        batch_idx = uncached_indices[i]
                        batch_results[batch_idx] = embedding
                        
                        # Cache the result
                        cache_key = self._get_cache_key(uncached_texts[i])
                        await self._cache_result(cache_key, embedding)
                
                all_embeddings.extend(batch_results)
            
            embedding_requests_total.labels(model=self.model_name, status='batch_success').inc()
            logger.info(f"Generated {len(all_embeddings)} embeddings ({self._cache_hits} cache hits)")
            return all_embeddings
    
    # Legacy method name for backward compatibility
    async def batch_generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """Legacy method name - use generate_embeddings_batch instead."""
        return await self.generate_embeddings_batch(texts, batch_size)
    
    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text without generating if not found.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text)
        return await self._get_from_cache(cache_key)
    
    async def invalidate_cache(self, text: str) -> bool:
        """
        Invalidate cached embedding for specific text.
        
        Args:
            text: Text to invalidate cache for
            
        Returns:
            True if cache was invalidated successfully
        """
        cache_key = self._get_cache_key(text)
        return await self._invalidate_cache_entry(cache_key)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def validate_text_length(self, text: str) -> bool:
        """
        Validate that text is within token limits.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False
        
        return self.count_tokens(text) <= self.max_tokens
    
    async def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting to prevent API quota exhaustion.
        """
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        
        # Check if we're within rate limits
        if len(self._request_times) >= self.rate_limit_rpm:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                embedding_rate_limit_errors.inc()
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self._request_times.append(current_time)
    
    async def _generate_with_retries(self, text: str, token_count: int) -> List[float]:
        """
        Generate embedding with exponential backoff retries.
        
        Args:
            text: Text to embed
            token_count: Number of tokens in text
            
        Returns:
            Generated embedding
            
        Raises:
            EmbeddingError: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Generating embedding (attempt {attempt + 1}/{self.max_retries + 1}): {text[:50]}... ({token_count} tokens)")
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                return response.data[0].embedding
                
            except openai.RateLimitError as e:
                last_exception = e
                embedding_rate_limit_errors.inc()
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                continue
                
            except (openai.APIError, openai.APIConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"API error, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error generating embedding: {e}")
                self._failed_requests += 1
                embedding_requests_total.labels(model=self.model_name, status='error').inc()
                raise EmbeddingError(f"Failed to generate embedding: {e}")
        
        # All retries exhausted
        self._failed_requests += 1
        embedding_requests_total.labels(model=self.model_name, status='error').inc()
        logger.error(f"All retries exhausted for embedding generation: {last_exception}")
        if isinstance(last_exception, openai.RateLimitError):
            raise RateLimitError(f"Rate limit exceeded after {self.max_retries} retries")
        raise EmbeddingError(f"Failed to generate embedding after {self.max_retries} retries: {last_exception}")
    
    async def _generate_batch_with_retries(self, texts: List[str], total_tokens: int) -> List[List[float]]:
        """
        Generate batch embeddings with exponential backoff retries.
        
        Args:
            texts: List of texts to embed
            total_tokens: Total number of tokens
            
        Returns:
            List of generated embeddings
            
        Raises:
            EmbeddingError: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Batch generating {len(texts)} embeddings (attempt {attempt + 1}/{self.max_retries + 1})")
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                
                # Update metrics
                self._api_calls += 1
                self._total_tokens += total_tokens
                embedding_tokens_processed.inc(total_tokens)
                
                return [data.embedding for data in response.data]
                
            except openai.RateLimitError as e:
                last_exception = e
                embedding_rate_limit_errors.inc()
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    logger.warning(f"Rate limit hit, retrying batch in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                continue
                
            except (openai.APIError, openai.APIConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"API error in batch, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error in batch generation: {e}")
                self._failed_requests += 1
                embedding_requests_total.labels(model=self.model_name, status='error').inc()
                raise EmbeddingError(f"Failed to generate batch embeddings: {e}")
        
        # All retries exhausted
        self._failed_requests += 1
        embedding_requests_total.labels(model=self.model_name, status='error').inc()
        logger.error(f"All retries exhausted for batch embedding generation: {last_exception}")
        if isinstance(last_exception, openai.RateLimitError):
            raise RateLimitError(f"Rate limit exceeded after {self.max_retries} retries")
        raise EmbeddingError(f"Failed to generate batch embeddings after {self.max_retries} retries: {last_exception}")
    
    async def _optimize_batch_size(self, texts: List[str], max_batch_size: int) -> int:
        """
        Optimize batch size based on token counts to maximize efficiency.
        
        Args:
            texts: List of texts to analyze
            max_batch_size: Maximum allowed batch size
            
        Returns:
            Optimized batch size
        """
        if not texts:
            return max_batch_size
        
        # Calculate average token count
        sample_size = min(10, len(texts))
        sample_texts = texts[:sample_size]
        total_tokens = sum(len(self.tokenizer.encode(text)) for text in sample_texts)
        avg_tokens = total_tokens / sample_size
        
        # Optimize batch size to keep under token limits
        # Assuming OpenAI has a practical limit around 100k tokens per batch
        max_tokens_per_batch = 100000
        optimal_batch_size = min(max_batch_size, max(1, int(max_tokens_per_batch / avg_tokens)))
        
        return optimal_batch_size
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the embedding service.
        
        Returns:
            Dictionary with performance data
        """
        total_requests = self._api_calls + self._cache_hits
        cache_hit_rate = self._cache_hits / max(1, total_requests)
        error_rate = self._failed_requests / max(1, total_requests + self._failed_requests)
        
        return {
            "total_api_calls": self._api_calls,
            "cache_hits": self._cache_hits,
            "redis_cache_hits": self._redis_cache_hits,
            "memory_cache_hits": self._memory_cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "error_rate": error_rate,
            "failed_requests": self._failed_requests,
            "total_tokens_processed": self._total_tokens,
            "memory_cache_size": len(self._memory_cache),
            "average_tokens_per_call": self._total_tokens / max(1, self._api_calls),
            "requests_per_minute": len([t for t in self._request_times if time.time() - t < 60])
        }
    
    async def clear_cache(self) -> None:
        """Clear both Redis and memory embedding caches."""
        try:
            # Clear Redis cache with pattern matching
            keys = []
            async for key in self.redis._redis.scan_iter(match="embedding_cache:*"):
                keys.append(key)
            if keys:
                await self.redis._redis.delete(*keys)
            
            # Clear memory cache
            self._memory_cache.clear()
            logger.info("Embedding caches cleared")
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
            # At least clear memory cache
            self._memory_cache.clear()
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Hash-based cache key
        """
        # Use SHA-256 hash of text and model name for cache key
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """
        Retrieve embedding from Redis cache first, then memory cache.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached embedding or None if not found/expired
        """
        # Try Redis cache first
        try:
            redis_key = f"embedding_cache:{cache_key}"
            cached_data = await self.redis.get(redis_key)
            if cached_data:
                embedding_data = json.loads(cached_data)
                self._redis_cache_hits += 1
                embedding_cache_hits_total.labels(cache_type='redis').inc()
                return embedding_data["embedding"]
        except Exception as e:
            logger.warning(f"Redis cache lookup failed: {e}")
        
        # Fallback to memory cache
        if cache_key in self._memory_cache:
            cache_entry = self._memory_cache[cache_key]
            
            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                self._memory_cache_hits += 1
                embedding_cache_hits_total.labels(cache_type='memory').inc()
                return cache_entry["embedding"]
            else:
                # Remove expired entry
                del self._memory_cache[cache_key]
        
        return None
    
    async def _cache_result(self, cache_key: str, embedding: List[float]) -> None:
        """
        Cache embedding result in both Redis and memory.
        
        Args:
            cache_key: Key to store under
            embedding: Embedding to cache
        """
        cache_data = {
            "embedding": embedding,
            "timestamp": time.time()
        }
        
        # Cache in Redis with TTL
        try:
            redis_key = f"embedding_cache:{cache_key}"
            await self.redis.set(
                redis_key, 
                json.dumps(cache_data), 
                expire=self.cache_ttl
            )
            embedding_cache_size.labels(cache_type='redis').inc()
        except Exception as e:
            logger.warning(f"Failed to cache in Redis: {e}")
        
        # Also cache in memory as fallback
        self._memory_cache[cache_key] = cache_data
        embedding_cache_size.labels(cache_type='memory').set(len(self._memory_cache))
    
    async def _invalidate_cache_entry(self, cache_key: str) -> bool:
        """
        Invalidate specific cache entry from both Redis and memory.
        
        Args:
            cache_key: Cache key to invalidate
            
        Returns:
            True if successfully invalidated
        """
        success = True
        
        # Remove from Redis
        try:
            redis_key = f"embedding_cache:{cache_key}"
            await self.redis.delete(redis_key)
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache entry: {e}")
            success = False
        
        # Remove from memory cache
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
            embedding_cache_size.labels(cache_type='memory').set(len(self._memory_cache))
        
        return success
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the embedding service.
        
        Returns:
            Health status information
        """
        health_status = {
            "status": "unknown",
            "model": self.model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        try:
            # Test Redis connectivity
            redis_healthy = False
            try:
                await self.redis.set("health_check", "test")
                await self.redis.delete("health_check")
                redis_healthy = True
                health_status["checks"]["redis"] = "healthy"
            except Exception as e:
                health_status["checks"]["redis"] = f"unhealthy: {e}"
            
            # Test embedding generation
            embedding_healthy = False
            test_text = "health check test"
            try:
                test_embedding = await self.generate_embedding(test_text)
                embedding_healthy = True
                health_status["checks"]["embedding_generation"] = "healthy"
                health_status["embedding_dimensions"] = len(test_embedding)
            except Exception as e:
                health_status["checks"]["embedding_generation"] = f"unhealthy: {e}"
            
            # Test cache functionality
            cache_healthy = False
            try:
                cached_result = await self.get_cached_embedding(test_text)
                if cached_result:
                    cache_healthy = True
                    health_status["checks"]["cache"] = "healthy"
                else:
                    health_status["checks"]["cache"] = "no_cache_entry"
            except Exception as e:
                health_status["checks"]["cache"] = f"unhealthy: {e}"
            
            # Overall status
            if embedding_healthy and redis_healthy:
                health_status["status"] = "healthy"
            elif embedding_healthy:
                health_status["status"] = "degraded"  # Embedding works but Redis issues
            else:
                health_status["status"] = "unhealthy"
            
            health_status["performance"] = self.get_performance_metrics()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status


# Singleton instance for application use
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get singleton embedding service instance with configuration settings.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(
            model_name=settings.OPENAI_EMBEDDING_MODEL,
            max_tokens=settings.OPENAI_EMBEDDING_MAX_TOKENS,
            cache_ttl=settings.OPENAI_EMBEDDING_CACHE_TTL,
            max_retries=settings.OPENAI_EMBEDDING_MAX_RETRIES,
            base_delay=settings.OPENAI_EMBEDDING_BASE_DELAY,
            max_delay=settings.OPENAI_EMBEDDING_MAX_DELAY,
            rate_limit_rpm=settings.OPENAI_EMBEDDING_RATE_LIMIT_RPM
        )
    
    return _embedding_service


async def cleanup_embedding_service() -> None:
    """Cleanup embedding service resources."""
    global _embedding_service
    
    if _embedding_service:
        await _embedding_service.clear_cache()
        _embedding_service = None