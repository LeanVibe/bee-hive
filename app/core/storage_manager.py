"""
Unified Storage Manager for LeanVibe Agent Hive 2.0

Consolidates 18 storage-related files into a comprehensive storage management system:
- Database management and connections
- Vector storage and search
- Embedding services
- Cache management
- Index management
- Hybrid search capabilities
"""

import asyncio
import uuid
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

import structlog
import redis.asyncio as aioredis
from sqlalchemy import select, and_, or_, desc, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import QueuePool

from .unified_manager_base import UnifiedManagerBase, ManagerConfig, PluginInterface, PluginType, create_manager_config
from .database import get_async_session
from .redis import get_redis

logger = structlog.get_logger()


class StorageType(str, Enum):
    """Types of storage systems."""
    RELATIONAL = "relational"
    VECTOR = "vector"
    CACHE = "cache"
    DOCUMENT = "document"
    TIME_SERIES = "time_series"
    BLOB = "blob"


class IndexType(str, Enum):
    """Types of database indexes."""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    VECTOR_IVF = "vector_ivf"
    VECTOR_HNSW = "vector_hnsw"


class EmbeddingModel(str, Enum):
    """Embedding model types."""
    OPENAI_ADA = "openai_ada"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CUSTOM = "custom"


class SearchType(str, Enum):
    """Search operation types."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_SEARCH = "semantic_search"
    HYBRID_SEARCH = "hybrid_search"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "beehive"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 0
    echo: bool = False
    ssl_mode: str = "prefer"


@dataclass
class VectorConfig:
    """Vector storage configuration."""
    dimension: int = 1536
    index_type: IndexType = IndexType.VECTOR_HNSW
    distance_metric: str = "cosine"
    ef_construction: int = 200
    m: int = 16


@dataclass
class CacheConfig:
    """Cache configuration."""
    default_ttl_seconds: int = 3600
    max_memory_mb: int = 512
    eviction_policy: str = "lru"
    compression_enabled: bool = True


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""
    text: str
    model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS
    normalize: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: List[float]
    dimension: int
    model: EmbeddingModel
    text_hash: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SearchQuery:
    """Search query specification."""
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    search_type: SearchType = SearchType.SEMANTIC_SEARCH
    limit: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    min_similarity: float = 0.0
    include_metadata: bool = True


@dataclass
class SearchResult:
    """Search result item."""
    id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class DatabaseManager:
    """Advanced database management and connection pooling."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = None
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "connection_errors": 0,
            "query_count": 0,
            "avg_query_time_ms": 0.0
        }
        
    async def initialize(self) -> bool:
        """Initialize database connections."""
        try:
            # Connection pool would be initialized here
            logger.info(
                "Database manager initialized",
                host=self.config.host,
                database=self.config.database,
                pool_size=self.config.pool_size
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize database manager", error=str(e))
            return False
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute database query with monitoring."""
        start_time = datetime.utcnow()
        
        try:
            async with get_async_session() as session:
                result = await session.execute(text(query), params or {})
                rows = result.fetchall()
                
                # Convert to dictionaries
                results = []
                if rows:
                    columns = result.keys()
                    results = [dict(zip(columns, row)) for row in rows]
                
                # Update stats
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.connection_stats["query_count"] += 1
                self.connection_stats["avg_query_time_ms"] = (
                    (self.connection_stats["avg_query_time_ms"] * (self.connection_stats["query_count"] - 1) + execution_time) /
                    self.connection_stats["query_count"]
                )
                
                return results
                
        except Exception as e:
            self.connection_stats["connection_errors"] += 1
            logger.error("Database query failed", query=query[:100], error=str(e))
            raise
    
    async def bulk_insert(
        self,
        table: str,
        records: List[Dict[str, Any]]
    ) -> bool:
        """Perform bulk insert operation."""
        try:
            if not records:
                return True
            
            # Build bulk insert query
            columns = list(records[0].keys())
            placeholders = ", ".join([f":{col}" for col in columns])
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            async with get_async_session() as session:
                await session.execute(text(query), records)
                await session.commit()
                
                logger.info(
                    "Bulk insert completed",
                    table=table,
                    records=len(records)
                )
                
                return True
                
        except Exception as e:
            logger.error("Bulk insert failed", table=table, error=str(e))
            return False
    
    async def create_index(
        self,
        table: str,
        columns: List[str],
        index_type: IndexType = IndexType.BTREE,
        unique: bool = False
    ) -> bool:
        """Create database index."""
        try:
            index_name = f"idx_{table}_{'_'.join(columns)}"
            
            if index_type == IndexType.VECTOR_HNSW:
                # pgvector HNSW index
                query = f"""
                CREATE INDEX IF NOT EXISTS {index_name} 
                ON {table} 
                USING hnsw ({', '.join(columns)} vector_cosine_ops)
                """
            elif index_type == IndexType.GIN:
                # GIN index for full-text search
                query = f"""
                CREATE INDEX IF NOT EXISTS {index_name} 
                ON {table} 
                USING gin ({', '.join(columns)})
                """
            else:
                # Standard B-tree index
                unique_clause = "UNIQUE" if unique else ""
                query = f"""
                CREATE {unique_clause} INDEX IF NOT EXISTS {index_name} 
                ON {table} ({', '.join(columns)})
                """
            
            await self.execute_query(query)
            
            logger.info(
                "Index created",
                table=table,
                columns=columns,
                index_type=index_type.value
            )
            
            return True
            
        except Exception as e:
            logger.error("Index creation failed", table=table, error=str(e))
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get database connection statistics."""
        return self.connection_stats.copy()


class EmbeddingService:
    """Comprehensive embedding generation and management service."""
    
    def __init__(self):
        self.embedding_cache: Dict[str, EmbeddingResult] = {}
        self.model_configs = {
            EmbeddingModel.SENTENCE_TRANSFORMERS: {
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384
            },
            EmbeddingModel.OPENAI_ADA: {
                "model_name": "text-embedding-ada-002",
                "dimension": 1536
            }
        }
        
    async def generate_embedding(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResult:
        """Generate embedding for text."""
        try:
            # Create cache key
            text_hash = hashlib.md5(request.text.encode()).hexdigest()
            cache_key = f"{request.model.value}:{text_hash}"
            
            # Check cache first
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
            
            # Generate embedding based on model
            if request.model == EmbeddingModel.SENTENCE_TRANSFORMERS:
                embedding = await self._generate_sentence_transformer_embedding(request.text)
            elif request.model == EmbeddingModel.OPENAI_ADA:
                embedding = await self._generate_openai_embedding(request.text)
            else:
                embedding = await self._generate_custom_embedding(request.text)
            
            # Normalize if requested
            if request.normalize:
                embedding = self._normalize_embedding(embedding)
            
            # Create result
            result = EmbeddingResult(
                embedding=embedding,
                dimension=len(embedding),
                model=request.model,
                text_hash=text_hash
            )
            
            # Cache result
            self.embedding_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error("Embedding generation failed", text=request.text[:100], error=str(e))
            raise
    
    async def _generate_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence transformers."""
        try:
            # This would use sentence-transformers library
            # For now, return mock embedding
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.random(384).tolist()
            return embedding
            
        except Exception as e:
            logger.error("Sentence transformer embedding failed", error=str(e))
            raise
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            # This would call OpenAI API
            # For now, return mock embedding
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.random(1536).tolist()
            return embedding
            
        except Exception as e:
            logger.error("OpenAI embedding failed", error=str(e))
            raise
    
    async def _generate_custom_embedding(self, text: str) -> List[float]:
        """Generate embedding using custom model."""
        try:
            # Custom embedding logic would go here
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.random(512).tolist()
            return embedding
            
        except Exception as e:
            logger.error("Custom embedding failed", error=str(e))
            raise
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return (np.array(embedding) / norm).tolist()
        return embedding
    
    async def batch_generate_embeddings(
        self,
        texts: List[str],
        model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts efficiently."""
        try:
            tasks = []
            
            for text in texts:
                request = EmbeddingRequest(text=text, model=model)
                task = self.generate_embedding(request)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
            
        except Exception as e:
            logger.error("Batch embedding generation failed", error=str(e))
            raise
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        return {
            "cache_size": len(self.embedding_cache),
            "supported_models": [model.value for model in EmbeddingModel],
            "model_configs": self.model_configs
        }


class VectorSearchEngine:
    """Advanced vector search with multiple algorithms."""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.vector_store: Dict[str, Dict[str, Any]] = {}
        self.index_built = False
        
    async def store_vector(
        self,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store vector with metadata."""
        try:
            self.vector_store[id] = {
                "vector": vector,
                "metadata": metadata or {},
                "created_at": datetime.utcnow()
            }
            
            # Mark index as needing rebuild
            self.index_built = False
            
            return True
            
        except Exception as e:
            logger.error("Vector storage failed", id=id, error=str(e))
            return False
    
    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search vectors by similarity."""
        try:
            results = []
            
            for vec_id, data in self.vector_store.items():
                # Apply filters
                if filters:
                    metadata = data["metadata"]
                    if not self._apply_filters(metadata, filters):
                        continue
                
                # Calculate similarity
                similarity = self._calculate_cosine_similarity(
                    query_vector,
                    data["vector"]
                )
                
                # Check minimum similarity threshold
                if similarity >= min_similarity:
                    results.append(SearchResult(
                        id=vec_id,
                        content=data["metadata"].get("content", ""),
                        similarity_score=similarity,
                        metadata=data["metadata"],
                        embedding=data["vector"]
                    ))
            
            # Sort by similarity and limit
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []
    
    def _calculate_cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            return dot_product / (norm_v1 * norm_v2)
            
        except Exception as e:
            logger.error("Similarity calculation failed", error=str(e))
            return 0.0
    
    def _apply_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """Apply metadata filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    async def delete_vector(self, id: str) -> bool:
        """Delete vector by ID."""
        try:
            if id in self.vector_store:
                del self.vector_store[id]
                self.index_built = False
                return True
            return False
            
        except Exception as e:
            logger.error("Vector deletion failed", id=id, error=str(e))
            return False
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector storage statistics."""
        return {
            "total_vectors": len(self.vector_store),
            "index_built": self.index_built,
            "dimension": self.config.dimension,
            "distance_metric": self.config.distance_metric
        }


class CacheManager:
    """Advanced caching system with Redis and in-memory tiers."""
    
    def __init__(self, config: CacheConfig, redis_client=None):
        self.config = config
        self.redis = redis_client
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0,
            "evictions": 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (local -> Redis)."""
        try:
            # Check local cache first
            if key in self.local_cache:
                item = self.local_cache[key]
                
                # Check TTL
                if datetime.utcnow() < item["expires_at"]:
                    self.cache_stats["hits"] += 1
                    self.cache_stats["local_hits"] += 1
                    return item["value"]
                else:
                    # Expired, remove from local cache
                    del self.local_cache[key]
            
            # Check Redis cache
            if self.redis:
                cached_data = await self.redis.get(f"cache:{key}")
                if cached_data:
                    try:
                        data = json.loads(cached_data)
                        self.cache_stats["hits"] += 1
                        self.cache_stats["redis_hits"] += 1
                        
                        # Store in local cache
                        await self._store_local(key, data, self.config.default_ttl_seconds)
                        
                        return data
                    except json.JSONDecodeError:
                        pass
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in cache (local + Redis)."""
        try:
            ttl = ttl_seconds or self.config.default_ttl_seconds
            
            # Store in local cache
            await self._store_local(key, value, ttl)
            
            # Store in Redis cache
            if self.redis:
                await self.redis.setex(
                    f"cache:{key}",
                    ttl,
                    json.dumps(value, default=str)
                )
            
            return True
            
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def _store_local(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store value in local cache."""
        # Check memory limit and evict if necessary
        if len(self.local_cache) >= 1000:  # Simple size limit
            await self._evict_local_cache()
        
        self.local_cache[key] = {
            "value": value,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl_seconds)
        }
    
    async def _evict_local_cache(self) -> None:
        """Evict items from local cache based on policy."""
        if self.config.eviction_policy == "lru":
            # Remove oldest items
            sorted_items = sorted(
                self.local_cache.items(),
                key=lambda x: x[1]["created_at"]
            )
            
            # Remove 25% of cache
            evict_count = len(sorted_items) // 4
            for i in range(evict_count):
                key = sorted_items[i][0]
                del self.local_cache[key]
                self.cache_stats["evictions"] += 1
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            deleted_local = key in self.local_cache
            if deleted_local:
                del self.local_cache[key]
            
            deleted_redis = False
            if self.redis:
                result = await self.redis.delete(f"cache:{key}")
                deleted_redis = result > 0
            
            return deleted_local or deleted_redis
            
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    async def clear(self) -> bool:
        """Clear all cache data."""
        try:
            self.local_cache.clear()
            
            if self.redis:
                # Delete all cache keys
                keys = await self.redis.keys("cache:*")
                if keys:
                    await self.redis.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error("Cache clear failed", error=str(e))
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(total_requests, 1)
        
        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache)
        }


class HybridSearchEngine:
    """Hybrid search combining full-text and semantic search."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_engine: VectorSearchEngine,
        database_manager: DatabaseManager
    ):
        self.embedding_service = embedding_service
        self.vector_engine = vector_engine
        self.database_manager = database_manager
        
    async def hybrid_search(
        self,
        query: SearchQuery
    ) -> List[SearchResult]:
        """Perform hybrid search combining multiple techniques."""
        try:
            results = []
            
            if query.search_type == SearchType.EXACT_MATCH:
                results = await self._exact_search(query)
            elif query.search_type == SearchType.FUZZY_MATCH:
                results = await self._fuzzy_search(query)
            elif query.search_type == SearchType.SEMANTIC_SEARCH:
                results = await self._semantic_search(query)
            elif query.search_type == SearchType.HYBRID_SEARCH:
                results = await self._hybrid_search(query)
            
            return results[:query.limit]
            
        except Exception as e:
            logger.error("Hybrid search failed", error=str(e))
            return []
    
    async def _exact_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform exact text search."""
        if not query.query_text:
            return []
        
        # This would perform SQL LIKE search
        sql_query = """
        SELECT id, content, metadata 
        FROM documents 
        WHERE content ILIKE %s
        ORDER BY created_at DESC
        """
        
        results = await self.database_manager.execute_query(
            sql_query,
            {"query": f"%{query.query_text}%"}
        )
        
        search_results = []
        for row in results:
            search_results.append(SearchResult(
                id=row["id"],
                content=row["content"],
                similarity_score=1.0,  # Exact match
                metadata=row.get("metadata", {})
            ))
        
        return search_results
    
    async def _fuzzy_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform fuzzy text search."""
        if not query.query_text:
            return []
        
        # This would use PostgreSQL fuzzy string matching
        sql_query = """
        SELECT id, content, metadata,
               similarity(content, %s) as sim_score
        FROM documents 
        WHERE similarity(content, %s) > 0.3
        ORDER BY sim_score DESC
        """
        
        results = await self.database_manager.execute_query(
            sql_query,
            {"query": query.query_text}
        )
        
        search_results = []
        for row in results:
            search_results.append(SearchResult(
                id=row["id"],
                content=row["content"],
                similarity_score=row["sim_score"],
                metadata=row.get("metadata", {})
            ))
        
        return search_results
    
    async def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic vector search."""
        try:
            query_embedding = query.query_embedding
            
            # Generate embedding if not provided
            if not query_embedding and query.query_text:
                embedding_request = EmbeddingRequest(text=query.query_text)
                embedding_result = await self.embedding_service.generate_embedding(embedding_request)
                query_embedding = embedding_result.embedding
            
            if not query_embedding:
                return []
            
            # Search vectors
            vector_results = await self.vector_engine.search_vectors(
                query_vector=query_embedding,
                limit=query.limit,
                min_similarity=query.min_similarity,
                filters=query.filters
            )
            
            return vector_results
            
        except Exception as e:
            logger.error("Semantic search failed", error=str(e))
            return []
    
    async def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Combine full-text and semantic search."""
        try:
            # Perform both searches
            text_results = await self._fuzzy_search(query)
            semantic_results = await self._semantic_search(query)
            
            # Combine and re-rank results
            combined_results = {}
            
            # Add text search results with weight
            for result in text_results:
                combined_results[result.id] = result
                result.similarity_score *= 0.4  # Weight for text search
            
            # Add semantic search results with weight
            for result in semantic_results:
                if result.id in combined_results:
                    # Combine scores for items found in both
                    combined_results[result.id].similarity_score += result.similarity_score * 0.6
                else:
                    result.similarity_score *= 0.6  # Weight for semantic search
                    combined_results[result.id] = result
            
            # Sort by combined score
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return final_results
            
        except Exception as e:
            logger.error("Hybrid search failed", error=str(e))
            return []


class StorageManager(UnifiedManagerBase):
    """
    Unified Storage Manager consolidating all storage-related functionality.
    
    Replaces 18 separate files:
    - database.py
    - database_models.py
    - database_types.py
    - database_performance_validator.py
    - enhanced_coordination_database_integration.py
    - redis.py
    - redis_integration.py
    - optimized_redis.py
    - enhanced_redis_streams_manager.py
    - embedding_service.py
    - embedding_service_simple.py
    - embeddings.py
    - vector_search.py
    - vector_search_engine.py
    - advanced_vector_search.py
    - enhanced_vector_search.py
    - memory_aware_vector_search.py
    - hybrid_search_engine.py
    - index_management.py
    - optimized_embedding_pipeline.py
    - optimized_pgvector_manager.py
    - pgvector_manager.py
    - semantic_embedding_service.py
    - mobile_api_cache.py
    """
    
    def __init__(self, config: ManagerConfig, dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(config, dependencies)
        
        # Configuration
        self.db_config = DatabaseConfig(**config.plugin_config.get("database", {}))
        self.vector_config = VectorConfig(**config.plugin_config.get("vector", {}))
        self.cache_config = CacheConfig(**config.plugin_config.get("cache", {}))
        
        # Core components
        self.database_manager = DatabaseManager(self.db_config)
        self.embedding_service = EmbeddingService()
        self.vector_engine = VectorSearchEngine(self.vector_config)
        self.cache_manager: Optional[CacheManager] = None
        self.hybrid_search: Optional[HybridSearchEngine] = None
        
        # State tracking
        self.storage_stats = {
            "total_queries": 0,
            "total_embeddings": 0,
            "total_vector_searches": 0,
            "cache_operations": 0
        }
    
    async def _initialize_manager(self) -> bool:
        """Initialize the storage manager."""
        try:
            # Initialize database
            db_success = await self.database_manager.initialize()
            if not db_success:
                return False
            
            # Initialize cache manager
            redis_client = get_redis()
            self.cache_manager = CacheManager(self.cache_config, redis_client)
            
            # Initialize hybrid search
            self.hybrid_search = HybridSearchEngine(
                self.embedding_service,
                self.vector_engine,
                self.database_manager
            )
            
            logger.info(
                "Storage Manager initialized",
                database_host=self.db_config.host,
                vector_dimension=self.vector_config.dimension,
                cache_enabled=redis_client is not None
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Storage Manager", error=str(e))
            return False
    
    async def _shutdown_manager(self) -> None:
        """Shutdown the storage manager."""
        try:
            # Clear caches
            if self.cache_manager:
                await self.cache_manager.clear()
            
            logger.info("Storage Manager shutdown completed")
            
        except Exception as e:
            logger.error("Error during Storage Manager shutdown", error=str(e))
    
    async def _get_manager_health(self) -> Dict[str, Any]:
        """Get storage manager health information."""
        health_info = {
            "database": self.database_manager.get_connection_stats(),
            "embeddings": self.embedding_service.get_embedding_stats(),
            "vectors": self.vector_engine.get_vector_stats(),
            "storage_stats": self.storage_stats.copy()
        }
        
        if self.cache_manager:
            health_info["cache"] = self.cache_manager.get_cache_stats()
        
        return health_info
    
    async def _load_plugins(self) -> None:
        """Load storage manager plugins."""
        # Storage plugins would be loaded here
        pass
    
    # === DATABASE OPERATIONS ===
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute database query."""
        return await self.execute_with_monitoring(
            "execute_query",
            self._execute_query_impl,
            query,
            params
        )
    
    async def _execute_query_impl(
        self,
        query: str,
        params: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Internal implementation of query execution."""
        self.storage_stats["total_queries"] += 1
        return await self.database_manager.execute_query(query, params)
    
    async def bulk_insert(
        self,
        table: str,
        records: List[Dict[str, Any]]
    ) -> bool:
        """Perform bulk insert operation."""
        return await self.execute_with_monitoring(
            "bulk_insert",
            self.database_manager.bulk_insert,
            table,
            records
        )
    
    # === EMBEDDING OPERATIONS ===
    
    async def generate_embedding(
        self,
        text: str,
        model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS,
        normalize: bool = True
    ) -> EmbeddingResult:
        """Generate embedding for text."""
        return await self.execute_with_monitoring(
            "generate_embedding",
            self._generate_embedding_impl,
            text,
            model,
            normalize
        )
    
    async def _generate_embedding_impl(
        self,
        text: str,
        model: EmbeddingModel,
        normalize: bool
    ) -> EmbeddingResult:
        """Internal implementation of embedding generation."""
        self.storage_stats["total_embeddings"] += 1
        request = EmbeddingRequest(text=text, model=model, normalize=normalize)
        return await self.embedding_service.generate_embedding(request)
    
    async def batch_generate_embeddings(
        self,
        texts: List[str],
        model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        return await self.execute_with_monitoring(
            "batch_generate_embeddings",
            self._batch_generate_embeddings_impl,
            texts,
            model
        )
    
    async def _batch_generate_embeddings_impl(
        self,
        texts: List[str],
        model: EmbeddingModel
    ) -> List[EmbeddingResult]:
        """Internal implementation of batch embedding generation."""
        self.storage_stats["total_embeddings"] += len(texts)
        return await self.embedding_service.batch_generate_embeddings(texts, model)
    
    # === VECTOR OPERATIONS ===
    
    async def store_vector(
        self,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store vector with metadata."""
        return await self.execute_with_monitoring(
            "store_vector",
            self.vector_engine.store_vector,
            id,
            vector,
            metadata
        )
    
    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search vectors by similarity."""
        return await self.execute_with_monitoring(
            "search_vectors",
            self._search_vectors_impl,
            query_vector,
            limit,
            min_similarity,
            filters
        )
    
    async def _search_vectors_impl(
        self,
        query_vector: List[float],
        limit: int,
        min_similarity: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Internal implementation of vector search."""
        self.storage_stats["total_vector_searches"] += 1
        return await self.vector_engine.search_vectors(
            query_vector,
            limit,
            min_similarity,
            filters
        )
    
    # === SEARCH OPERATIONS ===
    
    async def hybrid_search(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        search_type: SearchType = SearchType.HYBRID_SEARCH,
        limit: int = 10,
        min_similarity: float = 0.0,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Perform hybrid search."""
        return await self.execute_with_monitoring(
            "hybrid_search",
            self._hybrid_search_impl,
            query_text,
            query_embedding,
            search_type,
            limit,
            min_similarity,
            filters or {}
        )
    
    async def _hybrid_search_impl(
        self,
        query_text: Optional[str],
        query_embedding: Optional[List[float]],
        search_type: SearchType,
        limit: int,
        min_similarity: float,
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Internal implementation of hybrid search."""
        if not self.hybrid_search:
            return []
        
        query = SearchQuery(
            query_text=query_text,
            query_embedding=query_embedding,
            search_type=search_type,
            limit=limit,
            min_similarity=min_similarity,
            filters=filters
        )
        
        return await self.hybrid_search.hybrid_search(query)
    
    # === CACHE OPERATIONS ===
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.cache_manager:
            return None
        
        return await self.execute_with_monitoring(
            "cache_get",
            self._cache_get_impl,
            key
        )
    
    async def _cache_get_impl(self, key: str) -> Optional[Any]:
        """Internal implementation of cache get."""
        self.storage_stats["cache_operations"] += 1
        return await self.cache_manager.get(key)
    
    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        if not self.cache_manager:
            return False
        
        return await self.execute_with_monitoring(
            "cache_set",
            self._cache_set_impl,
            key,
            value,
            ttl_seconds
        )
    
    async def _cache_set_impl(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int]
    ) -> bool:
        """Internal implementation of cache set."""
        self.storage_stats["cache_operations"] += 1
        return await self.cache_manager.set(key, value, ttl_seconds)
    
    async def cache_delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.cache_manager:
            return False
        
        return await self.execute_with_monitoring(
            "cache_delete",
            self._cache_delete_impl,
            key
        )
    
    async def _cache_delete_impl(self, key: str) -> bool:
        """Internal implementation of cache delete."""
        self.storage_stats["cache_operations"] += 1
        return await self.cache_manager.delete(key)
    
    # === INDEX MANAGEMENT ===
    
    async def create_index(
        self,
        table: str,
        columns: List[str],
        index_type: IndexType = IndexType.BTREE,
        unique: bool = False
    ) -> bool:
        """Create database index."""
        return await self.execute_with_monitoring(
            "create_index",
            self.database_manager.create_index,
            table,
            columns,
            index_type,
            unique
        )
    
    # === PUBLIC API METHODS ===
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        try:
            stats = {
                "operations": self.storage_stats.copy(),
                "database": self.database_manager.get_connection_stats(),
                "embeddings": self.embedding_service.get_embedding_stats(),
                "vectors": self.vector_engine.get_vector_stats()
            }
            
            if self.cache_manager:
                stats["cache"] = self.cache_manager.get_cache_stats()
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get storage stats", error=str(e))
            return {"error": str(e)}


# Session cache implementation for backward compatibility
class SessionCache:
    """Session-based cache for temporary data storage."""
    
    def __init__(self):
        self._cache = {}
        self._created_at = datetime.utcnow()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from session cache."""
        return self._cache.get(key)
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in session cache."""
        self._cache[key] = value
    
    async def delete(self, key: str) -> bool:
        """Delete value from session cache."""
        return self._cache.pop(key, None) is not None
    
    async def clear(self) -> None:
        """Clear all values from session cache."""
        self._cache.clear()

# Global session cache instance
_session_cache = SessionCache()

def get_session_cache() -> SessionCache:
    """Get the global session cache instance."""
    return _session_cache

# Factory function for creating storage manager
def create_storage_manager(**config_overrides) -> StorageManager:
    """Create and initialize a storage manager."""
    config = create_manager_config("StorageManager", **config_overrides)
    return StorageManager(config)