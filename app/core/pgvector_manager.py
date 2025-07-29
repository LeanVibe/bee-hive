"""
PGVector Manager for LeanVibe Agent Hive 2.0 Semantic Memory Service

High-performance pgvector integration with optimized connection pooling,
HNSW indexing, and advanced query optimization for <200ms P95 search latency.

Features:
- Connection pooling with pgbouncer compatibility
- HNSW index management and optimization
- Advanced query optimization for vector operations
- Performance monitoring and metrics collection
- Batch processing support for high-throughput operations
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

import asyncpg
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool

from ..core.database import get_database_url
from ..schemas.semantic_memory import SearchResult, MetricsFormat

logger = logging.getLogger(__name__)


class PGVectorConfig:
    """Configuration for pgvector operations."""
    
    def __init__(self):
        self.connection_pool_size = 20
        self.max_overflow = 30
        self.pool_timeout = 30
        self.pool_recycle = 3600  # 1 hour
        self.embedding_dimensions = 1536
        self.hnsw_m = 16  # Number of connections for HNSW
        self.hnsw_ef_construction = 64  # Size of dynamic candidate list
        self.hnsw_ef_search = 40  # Size of dynamic candidate list for search
        self.batch_size = 50  # Optimal batch size for embeddings
        self.max_search_results = 100
        self.similarity_threshold_default = 0.7
        self.performance_targets = {
            'p95_search_latency_ms': 200.0,
            'ingestion_throughput_docs_per_sec': 500.0,
            'memory_efficiency_mb_per_100k_docs': 500.0
        }


class VectorSearchMetrics:
    """Metrics collection for vector search operations."""
    
    def __init__(self):
        self.search_times: List[float] = []
        self.embedding_times: List[float] = []
        self.result_counts: List[int] = []
        self.similarity_scores: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_operations = 0
        
    def record_search(self, search_time_ms: float, embedding_time_ms: float, 
                     result_count: int, avg_similarity: float):
        """Record a search operation."""
        self.search_times.append(search_time_ms)
        self.embedding_times.append(embedding_time_ms)
        self.result_counts.append(result_count)
        self.similarity_scores.append(avg_similarity)
        self.total_operations += 1
    
    def get_p95_latency(self) -> float:
        """Get P95 search latency."""
        if not self.search_times:
            return 0.0
        return np.percentile(self.search_times, 95)
    
    def get_avg_latency(self) -> float:
        """Get average search latency."""
        return np.mean(self.search_times) if self.search_times else 0.0
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PGVectorManager:
    """
    High-performance pgvector manager for semantic memory operations.
    
    Provides optimized vector operations with connection pooling,
    advanced indexing, and comprehensive performance monitoring.
    """
    
    def __init__(self, config: Optional[PGVectorConfig] = None):
        self.config = config or PGVectorConfig()
        self.engine = None
        self.session_factory = None
        self.metrics = VectorSearchMetrics()
        self._connection_pool = None
        self._index_stats = {}
        
    async def initialize(self):
        """Initialize the pgvector manager with optimized connection pooling."""
        try:
            database_url = get_database_url().replace('postgresql://', 'postgresql+asyncpg://')
            
            # Create engine with optimized pool configuration
            self.engine = create_async_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.config.connection_pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Validate connections
                echo=False  # Set to True for SQL debugging
            )
            
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Verify pgvector extension and optimize settings
            await self._verify_pgvector_setup()
            await self._optimize_vector_settings()
            
            logger.info("✅ PGVector Manager initialized successfully")
            logger.info(f"📊 Pool size: {self.config.connection_pool_size}, Max overflow: {self.config.max_overflow}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize PGVector Manager: {e}")
            raise
    
    async def _verify_pgvector_setup(self):
        """Verify pgvector extension and index status."""
        async with self.get_session() as session:
            # Check pgvector extension
            result = await session.execute(text("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                );
            """))
            
            if not result.scalar():
                raise RuntimeError("pgvector extension not installed")
            
            # Check HNSW indexes exist and are healthy
            result = await session.execute(text("""
                SELECT indexname, tablename, indexdef
                FROM pg_indexes 
                WHERE indexname LIKE '%semantic_documents_embedding_hnsw%'
                ORDER BY indexname;
            """))
            
            indexes = result.fetchall()
            logger.info(f"📍 Found {len(indexes)} HNSW indexes")
            
            for index in indexes:
                logger.info(f"   • {index.indexname}: {index.tablename}")
    
    async def _optimize_vector_settings(self):
        """Optimize PostgreSQL settings for vector operations."""
        async with self.get_session() as session:
            # Set optimal work_mem for vector operations
            await session.execute(text("SET work_mem = '256MB';"))
            
            # Set optimal maintenance_work_mem for index operations
            await session.execute(text("SET maintenance_work_mem = '512MB';"))
            
            # Optimize for vector search performance
            await session.execute(text(f"SET hnsw.ef_search = {self.config.hnsw_ef_search};"))
            
            await session.commit()
    
    @asynccontextmanager
    async def get_session(self):
        """Get a database session with proper error handling."""
        if not self.session_factory:
            raise RuntimeError("PGVectorManager not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def insert_document_with_embedding(
        self,
        document_id: uuid.UUID,
        agent_id: uuid.UUID,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        workflow_id: Optional[uuid.UUID] = None,
        importance_score: float = 0.5
    ) -> bool:
        """
        Insert a document with its embedding vector.
        
        Args:
            document_id: Unique document identifier
            agent_id: Agent that owns the document
            content: Document content
            embedding: 1536-dimensional embedding vector
            metadata: Optional metadata dictionary
            tags: Optional list of tags
            workflow_id: Optional workflow context
            importance_score: Document importance (0.0 to 1.0)
            
        Returns:
            Success status
        """
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                # Validate embedding dimensions
                if len(embedding) != self.config.embedding_dimensions:
                    raise ValueError(f"Embedding must have {self.config.embedding_dimensions} dimensions")
                
                # Convert embedding to pgvector format
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                query = text("""
                    INSERT INTO semantic_documents (
                        document_id, agent_id, workflow_id, content, metadata, tags,
                        embedding, importance_score, created_at, updated_at
                    ) VALUES (
                        :document_id, :agent_id, :workflow_id, :content, :metadata::json, :tags::json,
                        :embedding::vector, :importance_score, NOW(), NOW()
                    )
                    ON CONFLICT (document_id) 
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        tags = EXCLUDED.tags,
                        embedding = EXCLUDED.embedding,
                        importance_score = EXCLUDED.importance_score,
                        updated_at = NOW()
                """)
                
                await session.execute(query, {
                    'document_id': document_id,
                    'agent_id': agent_id,
                    'workflow_id': workflow_id,
                    'content': content,
                    'metadata': metadata or {},
                    'tags': tags or [],
                    'embedding': embedding_str,
                    'importance_score': importance_score
                })
                
                await session.commit()
                
                processing_time = (time.time() - start_time) * 1000
                logger.debug(f"Document inserted in {processing_time:.2f}ms")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert document {document_id}: {e}")
            return False
    
    async def batch_insert_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Tuple[int, int, List[str]]:
        """
        Batch insert documents with embeddings for high throughput.
        
        Args:
            documents: List of document dictionaries with all required fields
            
        Returns:
            Tuple of (successful_count, failed_count, error_messages)
        """
        successful = 0
        failed = 0
        errors = []
        
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                # Process documents in optimal batch sizes
                for i in range(0, len(documents), self.config.batch_size):
                    batch = documents[i:i + self.config.batch_size]
                    
                    # Prepare batch insert query
                    values = []
                    params = {}
                    
                    for j, doc in enumerate(batch):
                        param_prefix = f"doc_{i}_{j}"
                        
                        # Validate and format embedding
                        embedding = doc.get('embedding', [])
                        if len(embedding) != self.config.embedding_dimensions:
                            errors.append(f"Document {doc.get('document_id')}: Invalid embedding dimensions")
                            failed += 1
                            continue
                        
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        
                        values.append(f"""(
                            :{param_prefix}_document_id, :{param_prefix}_agent_id, :{param_prefix}_workflow_id,
                            :{param_prefix}_content, :{param_prefix}_metadata::json, :{param_prefix}_tags::json,
                            :{param_prefix}_embedding::vector, :{param_prefix}_importance_score, NOW(), NOW()
                        )""")
                        
                        params.update({
                            f'{param_prefix}_document_id': doc.get('document_id'),
                            f'{param_prefix}_agent_id': doc.get('agent_id'),
                            f'{param_prefix}_workflow_id': doc.get('workflow_id'),
                            f'{param_prefix}_content': doc.get('content'),
                            f'{param_prefix}_metadata': doc.get('metadata', {}),
                            f'{param_prefix}_tags': doc.get('tags', []),
                            f'{param_prefix}_embedding': embedding_str,
                            f'{param_prefix}_importance_score': doc.get('importance_score', 0.5)
                        })
                    
                    if values:
                        query = text(f"""
                            INSERT INTO semantic_documents (
                                document_id, agent_id, workflow_id, content, metadata, tags,
                                embedding, importance_score, created_at, updated_at
                            ) VALUES {', '.join(values)}
                            ON CONFLICT (document_id) 
                            DO UPDATE SET
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata,
                                tags = EXCLUDED.tags,
                                embedding = EXCLUDED.embedding,
                                importance_score = EXCLUDED.importance_score,
                                updated_at = NOW()
                        """)
                        
                        await session.execute(query, params)
                        successful += len(values)
                
                await session.commit()
                
                processing_time = (time.time() - start_time) * 1000
                throughput = len(documents) / (processing_time / 1000) if processing_time > 0 else 0
                
                logger.info(f"✅ Batch inserted {successful} documents in {processing_time:.2f}ms")
                logger.info(f"📊 Throughput: {throughput:.1f} docs/sec")
                
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            errors.append(str(e))
            failed = len(documents) - successful
            
        return successful, failed, errors
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        agent_id: Optional[uuid.UUID] = None,
        workflow_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        importance_min: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Perform high-performance semantic search using pgvector.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            agent_id: Filter by agent (optional)
            workflow_id: Filter by workflow (optional)
            tags: Filter by tags (optional)
            metadata_filters: Additional metadata filters (optional)
            importance_min: Minimum importance score (optional)
            
        Returns:
            List of search results with similarity scores
        """
        try:
            start_time = time.time()
            
            # Validate input
            if len(query_embedding) != self.config.embedding_dimensions:
                raise ValueError(f"Query embedding must have {self.config.embedding_dimensions} dimensions")
            
            # Limit results to prevent performance issues
            limit = min(limit, self.config.max_search_results)
            
            async with self.get_session() as session:
                # Build dynamic query with filters
                base_query = """
                    SELECT 
                        document_id,
                        content,
                        metadata,
                        agent_id,
                        tags,
                        importance_score,
                        1 - (embedding <-> :query_embedding::vector) as similarity_score,
                        created_at,
                        access_count
                    FROM semantic_documents
                    WHERE embedding IS NOT NULL
                """
                
                params = {
                    'query_embedding': '[' + ','.join(map(str, query_embedding)) + ']',
                    'similarity_threshold': similarity_threshold,
                    'limit': limit
                }
                
                # Add filters
                filters = ["1 - (embedding <-> :query_embedding::vector) >= :similarity_threshold"]
                
                if agent_id:
                    filters.append("agent_id = :agent_id")
                    params['agent_id'] = agent_id
                
                if workflow_id:
                    filters.append("workflow_id = :workflow_id")
                    params['workflow_id'] = workflow_id
                
                if importance_min is not None:
                    filters.append("importance_score >= :importance_min")
                    params['importance_min'] = importance_min
                
                if tags:
                    # Use GIN index for efficient tag filtering
                    filters.append("tags ?| :tags")
                    params['tags'] = tags
                
                if metadata_filters:
                    for key, value in metadata_filters.items():
                        filter_key = f"metadata_{key.replace('.', '_')}"
                        filters.append(f"metadata ->> :{filter_key} = :{filter_key}_value")
                        params[filter_key] = key
                        params[f"{filter_key}_value"] = str(value)
                
                # Combine query with filters and optimization
                query = text(f"""
                    {base_query}
                    AND {' AND '.join(filters)}
                    ORDER BY embedding <-> :query_embedding::vector
                    LIMIT :limit
                """)
                
                # Set optimal search parameters
                await session.execute(text(f"SET hnsw.ef_search = {self.config.hnsw_ef_search};"))
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                
                # Convert to SearchResult objects
                search_results = []
                similarity_scores = []
                
                for row in rows:
                    similarity_scores.append(row.similarity_score)
                    
                    search_results.append(SearchResult(
                        document_id=row.document_id,
                        content=row.content,
                        similarity_score=row.similarity_score,
                        metadata=row.metadata or {},
                        agent_id=str(row.agent_id),
                        tags=row.tags or [],
                        relevance_explanation=f"Semantic similarity: {row.similarity_score:.3f}",
                        highlighted_content=None,  # TODO: Implement highlighting
                        embedding_vector=None  # Not included by default for performance
                    ))
                
                # Record metrics
                search_time = (time.time() - start_time) * 1000
                avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
                
                self.metrics.record_search(
                    search_time, 0.0, len(search_results), avg_similarity
                )
                
                logger.debug(f"Semantic search completed in {search_time:.2f}ms, {len(search_results)} results")
                
                return search_results
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def find_similar_documents(
        self,
        document_id: uuid.UUID,
        limit: int = 5,
        similarity_threshold: float = 0.6,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: Source document ID
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            exclude_self: Whether to exclude the source document
            
        Returns:
            List of similar documents
        """
        try:
            async with self.get_session() as session:
                # Get the source document's embedding
                source_query = text("""
                    SELECT embedding, agent_id FROM semantic_documents 
                    WHERE document_id = :document_id AND embedding IS NOT NULL
                """)
                
                source_result = await session.execute(source_query, {'document_id': document_id})
                source_row = source_result.fetchone()
                
                if not source_row:
                    logger.warning(f"Document {document_id} not found or has no embedding")
                    return []
                
                # Extract embedding vector from database format
                embedding_str = str(source_row.embedding)
                embedding_list = [float(x) for x in embedding_str.strip('[]').split(',')]
                
                # Use semantic search to find similar documents
                results = await self.semantic_search(
                    query_embedding=embedding_list,
                    limit=limit + (1 if exclude_self else 0),
                    similarity_threshold=similarity_threshold,
                    agent_id=source_row.agent_id  # Scope to same agent by default
                )
                
                # Filter out self if requested
                if exclude_self:
                    results = [r for r in results if r.document_id != document_id][:limit]
                
                return results
                
        except Exception as e:
            logger.error(f"Similar document search failed: {e}")
            return []
    
    async def get_document_by_id(
        self,
        document_id: uuid.UUID,
        include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID with optional embedding.
        
        Args:
            document_id: Document identifier
            include_embedding: Whether to include the embedding vector
            
        Returns:
            Document data or None if not found
        """
        try:
            async with self.get_session() as session:
                # Update access tracking
                await session.execute(text("""
                    UPDATE semantic_documents 
                    SET access_count = access_count + 1, last_accessed = NOW()
                    WHERE document_id = :document_id
                """), {'document_id': document_id})
                
                # Retrieve document
                embedding_select = ", embedding" if include_embedding else ""
                query = text(f"""
                    SELECT 
                        document_id, agent_id, workflow_id, content, metadata, tags,
                        importance_score, access_count, created_at, updated_at, last_accessed
                        {embedding_select}
                    FROM semantic_documents 
                    WHERE document_id = :document_id
                """)
                
                result = await session.execute(query, {'document_id': document_id})
                row = result.fetchone()
                
                if not row:
                    return None
                
                document_data = {
                    'document_id': row.document_id,
                    'agent_id': row.agent_id,
                    'workflow_id': row.workflow_id,
                    'content': row.content,
                    'metadata': row.metadata or {},
                    'tags': row.tags or [],
                    'importance_score': row.importance_score,
                    'access_count': row.access_count,
                    'created_at': row.created_at,
                    'updated_at': row.updated_at,
                    'last_accessed': row.last_accessed
                }
                
                if include_embedding and hasattr(row, 'embedding') and row.embedding:
                    embedding_str = str(row.embedding)
                    document_data['embedding_vector'] = [
                        float(x) for x in embedding_str.strip('[]').split(',')
                    ]
                
                await session.commit()
                return document_data
                
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id}: {e}")
            return None
    
    async def delete_document(self, document_id: uuid.UUID) -> bool:
        """
        Delete a document from the semantic memory.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Success status
        """
        try:
            async with self.get_session() as session:
                query = text("DELETE FROM semantic_documents WHERE document_id = :document_id")
                result = await session.execute(query, {'document_id': document_id})
                await session.commit()
                
                deleted = result.rowcount > 0
                logger.debug(f"Document {document_id} {'deleted' if deleted else 'not found'}")
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'total_operations': self.metrics.total_operations,
            'avg_search_time_ms': self.metrics.get_avg_latency(),
            'p95_search_time_ms': self.metrics.get_p95_latency(),
            'cache_hit_rate': self.metrics.get_cache_hit_rate(),
            'performance_targets': self.config.performance_targets,
            'connection_pool_size': self.config.connection_pool_size,
            'max_overflow': self.config.max_overflow
        }
    
    async def optimize_indexes(self) -> Dict[str, Any]:
        """
        Optimize vector indexes for better performance.
        
        Returns:
            Optimization results and recommendations
        """
        try:
            async with self.get_session() as session:
                # Update table statistics
                await session.execute(text("ANALYZE semantic_documents;"))
                
                # Get index usage statistics
                result = await session.execute(text("""
                    SELECT 
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes 
                    WHERE indexname LIKE '%semantic_documents%'
                    ORDER BY idx_scan DESC;
                """))
                
                index_stats = [dict(row) for row in result.fetchall()]
                
                # Get current performance metrics
                performance_result = await session.execute(text("""
                    SELECT * FROM optimize_semantic_search_performance();
                """))
                
                optimization_results = [dict(row) for row in performance_result.fetchall()]
                
                await session.commit()
                
                return {
                    'index_statistics': index_stats,
                    'optimization_results': optimization_results,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Clean up resources and close connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("🧹 PGVector Manager cleanup completed")


# Global instance
_pgvector_manager: Optional[PGVectorManager] = None

async def get_pgvector_manager() -> PGVectorManager:
    """Get the global pgvector manager instance."""
    global _pgvector_manager
    
    if _pgvector_manager is None:
        _pgvector_manager = PGVectorManager()
        await _pgvector_manager.initialize()
    
    return _pgvector_manager

async def cleanup_pgvector_manager():
    """Clean up the global pgvector manager."""
    global _pgvector_manager
    
    if _pgvector_manager:
        await _pgvector_manager.cleanup()
        _pgvector_manager = None