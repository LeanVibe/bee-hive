"""
Embedding Service for Context Engine.

Provides text embedding generation using OpenAI's text-embedding-ada-002 model
with caching, batch processing, and error handling for optimal performance.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any
import logging

import openai
from openai import AsyncOpenAI
import tiktoken

from ..core.config import get_settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI's embedding models.
    
    Features:
    - Automatic caching to reduce API calls
    - Batch processing for efficiency  
    - Token counting and validation
    - Rate limiting and error handling
    - Performance monitoring
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        max_tokens: int = 8191,
        cache_ttl: int = 3600
    ):
        """
        Initialize embedding service.
        
        Args:
            model_name: OpenAI embedding model to use
            max_tokens: Maximum tokens per embedding request
            cache_ttl: Cache time-to-live in seconds
        """
        self.settings = get_settings()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        
        # Initialize tokenizer for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # In-memory cache for embeddings
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self._api_calls = 0
        self._cache_hits = 0
        self._total_tokens = 0
        
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text input.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If text is too long or empty
            openai.OpenAIError: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            self._cache_hits += 1
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return cached_result
        
        # Validate token count
        token_count = len(self.tokenizer.encode(text))
        if token_count > self.max_tokens:
            raise ValueError(f"Text too long: {token_count} tokens (max: {self.max_tokens})")
        
        try:
            # Call OpenAI API
            logger.debug(f"Generating embedding for text: {text[:50]}... ({token_count} tokens)")
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Update metrics
            self._api_calls += 1
            self._total_tokens += token_count
            
            # Cache the result
            self._cache_result(cache_key, embedding)
            
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error generating embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise
    
    async def batch_generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per API call
            
        Returns:
            List of embedding vectors in same order as input texts
            
        Raises:
            ValueError: If any text is invalid
            openai.OpenAIError: If API call fails
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache for each text in batch
            batch_results = []
            uncached_texts = []
            uncached_indices = []
            
            for idx, text in enumerate(batch_texts):
                if not text or not text.strip():
                    raise ValueError(f"Text at index {i + idx} cannot be empty")
                
                cache_key = self._get_cache_key(text)
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result:
                    batch_results.append(cached_result)
                    self._cache_hits += 1
                else:
                    batch_results.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    # Validate token counts
                    for text in uncached_texts:
                        token_count = len(self.tokenizer.encode(text))
                        if token_count > self.max_tokens:
                            raise ValueError(f"Text too long: {token_count} tokens (max: {self.max_tokens})")
                    
                    logger.debug(f"Batch generating {len(uncached_texts)} embeddings")
                    response = await self.client.embeddings.create(
                        model=self.model_name,
                        input=uncached_texts
                    )
                    
                    # Update metrics
                    self._api_calls += 1
                    self._total_tokens += sum(len(self.tokenizer.encode(text)) for text in uncached_texts)
                    
                    # Cache and insert results
                    for i, embedding_data in enumerate(response.data):
                        embedding = embedding_data.embedding
                        batch_idx = uncached_indices[i]
                        batch_results[batch_idx] = embedding
                        
                        # Cache the result
                        cache_key = self._get_cache_key(uncached_texts[i])
                        self._cache_result(cache_key, embedding)
                    
                except openai.OpenAIError as e:
                    logger.error(f"OpenAI API error in batch generation: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in batch generation: {e}")
                    raise
            
            all_embeddings.extend(batch_results)
        
        logger.info(f"Generated {len(all_embeddings)} embeddings ({self._cache_hits} cache hits)")
        return all_embeddings
    
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
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the embedding service.
        
        Returns:
            Dictionary with performance data
        """
        cache_hit_rate = self._cache_hits / max(1, self._api_calls + self._cache_hits)
        
        return {
            "total_api_calls": self._api_calls,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "total_tokens_processed": self._total_tokens,
            "cache_size": len(self._cache),
            "average_tokens_per_call": self._total_tokens / max(1, self._api_calls)
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
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
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """
        Retrieve embedding from cache if not expired.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached embedding or None if not found/expired
        """
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            
            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["embedding"]
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, embedding: List[float]) -> None:
        """
        Cache embedding result.
        
        Args:
            cache_key: Key to store under
            embedding: Embedding to cache
        """
        self._cache[cache_key] = {
            "embedding": embedding,
            "timestamp": time.time()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the embedding service.
        
        Returns:
            Health status information
        """
        try:
            # Test with a simple embedding
            test_embedding = await self.generate_embedding("health check test")
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "embedding_dimensions": len(test_embedding),
                "cache_size": len(self._cache),
                "performance": self.get_performance_metrics()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name
            }


# Singleton instance for application use
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get singleton embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    return _embedding_service


async def cleanup_embedding_service() -> None:
    """Cleanup embedding service resources."""
    global _embedding_service
    
    if _embedding_service:
        _embedding_service.clear_cache()
        _embedding_service = None