"""
Context Analytics and Relationship Management.

Implements context relationship tracking, retrieval analytics, and
knowledge graph functionality for the Context Engine.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.context import Context
from ..core.embeddings import EmbeddingService


logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships between contexts."""
    SIMILAR = "SIMILAR"  # Semantically similar content
    RELATED = "RELATED"  # Topically related
    DERIVED_FROM = "DERIVED_FROM"  # One context builds on another
    SUPERSEDES = "SUPERSEDES"  # Newer version replaces older
    REFERENCES = "REFERENCES"  # One context references another
    CONTRADICTS = "CONTRADICTS"  # Conflicting information


class ContextRelationship:
    """Represents a relationship between two contexts."""
    
    def __init__(
        self,
        id: uuid.UUID,
        source_context_id: uuid.UUID,
        target_context_id: uuid.UUID,
        relationship_type: RelationshipType,
        similarity_score: Optional[float] = None,
        confidence_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None
    ):
        self.id = id
        self.source_context_id = source_context_id
        self.target_context_id = target_context_id
        self.relationship_type = relationship_type
        self.similarity_score = similarity_score
        self.confidence_score = confidence_score
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow()


class ContextRetrieval:
    """Represents a context retrieval event for analytics."""
    
    def __init__(
        self,
        id: uuid.UUID,
        context_id: uuid.UUID,
        requesting_agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        similarity_score: Optional[float] = None,
        relevance_score: Optional[float] = None,
        rank_position: Optional[int] = None,
        was_helpful: Optional[bool] = None,
        feedback_score: Optional[int] = None,
        retrieval_method: str = "semantic_search",
        response_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        retrieved_at: Optional[datetime] = None
    ):
        self.id = id
        self.context_id = context_id
        self.requesting_agent_id = requesting_agent_id
        self.session_id = session_id
        self.query_text = query_text
        self.query_embedding = query_embedding
        self.similarity_score = similarity_score
        self.relevance_score = relevance_score
        self.rank_position = rank_position
        self.was_helpful = was_helpful
        self.feedback_score = feedback_score
        self.retrieval_method = retrieval_method
        self.response_time_ms = response_time_ms
        self.metadata = metadata or {}
        self.retrieved_at = retrieved_at or datetime.utcnow()


class ContextAnalyticsManager:
    """
    Manages context relationships and retrieval analytics.
    
    Features:
    - Automatic relationship discovery
    - Retrieval event tracking
    - Performance analytics
    - Knowledge graph construction
    - Usage pattern analysis
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_service: EmbeddingService
    ):
        """
        Initialize analytics manager.
        
        Args:
            db_session: Database session
            embedding_service: Service for embedding operations
        """
        self.db = db_session
        self.embedding_service = embedding_service
        
        # Performance tracking
        self.analytics_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
    
    async def discover_context_relationships(
        self,
        context_id: uuid.UUID,
        similarity_threshold: float = 0.75,
        max_relationships: int = 10
    ) -> List[ContextRelationship]:
        """
        Discover relationships for a context using semantic similarity.
        
        Args:
            context_id: Context to find relationships for
            similarity_threshold: Minimum similarity score
            max_relationships: Maximum number of relationships to find
            
        Returns:
            List of discovered relationships
        """
        try:
            # Get the source context
            source_context = await self.db.get(Context, context_id)
            if not source_context or not source_context.embedding:
                return []
            
            # Find similar contexts using vector similarity
            similarity_query = text("""
                SELECT 
                    c.id,
                    c.title,
                    c.context_type,
                    c.importance_score,
                    c.created_at,
                    1 - (c.embedding <=> :source_embedding) as similarity_score
                FROM contexts c
                WHERE c.id != :context_id
                    AND c.embedding IS NOT NULL
                    AND 1 - (c.embedding <=> :source_embedding) >= :threshold
                ORDER BY c.embedding <=> :source_embedding
                LIMIT :max_results
            """)
            
            result = await self.db.execute(
                similarity_query,
                {
                    "source_embedding": source_context.embedding,
                    "context_id": context_id,
                    "threshold": similarity_threshold,
                    "max_results": max_relationships
                }
            )
            
            relationships = []
            
            for row in result:
                # Determine relationship type based on similarity and other factors
                relationship_type = self._classify_relationship(
                    source_context=source_context,
                    target_context_data={
                        "id": row.id,
                        "title": row.title,
                        "context_type": row.context_type,
                        "importance_score": row.importance_score,
                        "created_at": row.created_at
                    },
                    similarity_score=row.similarity_score
                )
                
                # Calculate confidence based on various factors
                confidence = self._calculate_relationship_confidence(
                    similarity_score=row.similarity_score,
                    source_importance=source_context.importance_score,
                    target_importance=row.importance_score
                )
                
                relationship = ContextRelationship(
                    id=uuid.uuid4(),
                    source_context_id=context_id,
                    target_context_id=row.id,
                    relationship_type=relationship_type,
                    similarity_score=row.similarity_score,
                    confidence_score=confidence,
                    metadata={
                        "discovery_method": "semantic_similarity",
                        "source_title": source_context.title,
                        "target_title": row.title
                    }
                )
                
                relationships.append(relationship)
            
            # Store relationships in database
            await self._store_relationships(relationships)
            
            logger.info(f"Discovered {len(relationships)} relationships for context {context_id}")
            return relationships
            
        except Exception as e:
            logger.error(f"Error discovering relationships for context {context_id}: {e}")
            return []
    
    async def record_context_retrieval(
        self,
        context_id: uuid.UUID,
        requesting_agent_id: uuid.UUID,
        query_text: Optional[str] = None,
        similarity_score: Optional[float] = None,
        relevance_score: Optional[float] = None,
        rank_position: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        session_id: Optional[uuid.UUID] = None,
        retrieval_method: str = "semantic_search"
    ) -> ContextRetrieval:
        """
        Record a context retrieval event for analytics.
        
        Args:
            context_id: Context that was retrieved
            requesting_agent_id: Agent that requested the context
            query_text: Search query (optional)
            similarity_score: Semantic similarity score
            relevance_score: Calculated relevance score
            rank_position: Position in search results
            response_time_ms: Response time in milliseconds
            session_id: Session context (optional)
            retrieval_method: Method used for retrieval
            
        Returns:
            ContextRetrieval record
        """
        try:
            # Generate query embedding if query text is provided
            query_embedding = None
            if query_text:
                try:
                    query_embedding = await self.embedding_service.generate_embedding(query_text)
                except Exception as e:
                    logger.warning(f"Failed to generate query embedding: {e}")
            
            retrieval = ContextRetrieval(
                id=uuid.uuid4(),
                context_id=context_id,
                requesting_agent_id=requesting_agent_id,
                session_id=session_id,
                query_text=query_text,
                query_embedding=query_embedding,
                similarity_score=similarity_score,
                relevance_score=relevance_score,
                rank_position=rank_position,
                response_time_ms=response_time_ms,
                retrieval_method=retrieval_method
            )
            
            # Store in database
            await self._store_retrieval_record(retrieval)
            
            return retrieval
            
        except Exception as e:
            logger.error(f"Error recording context retrieval: {e}")
            raise
    
    async def update_retrieval_feedback(
        self,
        retrieval_id: uuid.UUID,
        was_helpful: bool,
        feedback_score: Optional[int] = None
    ) -> bool:
        """
        Update retrieval feedback for improving recommendations.
        
        Args:
            retrieval_id: Retrieval record to update
            was_helpful: Whether the context was helpful
            feedback_score: Optional numerical feedback (1-5)
            
        Returns:
            True if update was successful
        """
        try:
            update_query = text("""
                UPDATE context_retrievals 
                SET was_helpful = :was_helpful,
                    feedback_score = :feedback_score
                WHERE id = :retrieval_id
            """)
            
            await self.db.execute(
                update_query,
                {
                    "retrieval_id": retrieval_id,
                    "was_helpful": was_helpful,
                    "feedback_score": feedback_score
                }
            )
            
            await self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating retrieval feedback: {e}")
            await self.db.rollback()
            return False
    
    async def get_context_usage_analytics(
        self,
        context_id: Optional[uuid.UUID] = None,
        agent_id: Optional[uuid.UUID] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get comprehensive usage analytics for contexts.
        
        Args:
            context_id: Specific context to analyze (optional)
            agent_id: Specific agent to analyze (optional)
            days_back: Number of days to analyze
            
        Returns:
            Usage analytics data
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Build base query
            conditions = ["cr.retrieved_at >= :cutoff_date"]
            params = {"cutoff_date": cutoff_date}
            
            if context_id:
                conditions.append("cr.context_id = :context_id")
                params["context_id"] = context_id
            
            if agent_id:
                conditions.append("cr.requesting_agent_id = :agent_id")
                params["agent_id"] = agent_id
            
            where_clause = " AND ".join(conditions)
            
            # Analytics query
            analytics_query = text(f"""
                SELECT 
                    COUNT(*) as total_retrievals,
                    COUNT(DISTINCT cr.context_id) as unique_contexts,
                    COUNT(DISTINCT cr.requesting_agent_id) as unique_agents,
                    AVG(cr.similarity_score) as avg_similarity_score,
                    AVG(cr.relevance_score) as avg_relevance_score,
                    AVG(cr.response_time_ms) as avg_response_time_ms,
                    COUNT(CASE WHEN cr.was_helpful = true THEN 1 END) as helpful_retrievals,
                    COUNT(CASE WHEN cr.was_helpful = false THEN 1 END) as unhelpful_retrievals,
                    AVG(cr.feedback_score) as avg_feedback_score,
                    COUNT(DISTINCT cr.session_id) as unique_sessions
                FROM context_retrievals cr
                WHERE {where_clause}
            """)
            
            result = await self.db.execute(analytics_query, params)
            row = result.first()
            
            # Top contexts query
            top_contexts_query = text(f"""
                SELECT 
                    cr.context_id,
                    c.title,
                    COUNT(*) as retrieval_count,
                    AVG(cr.similarity_score) as avg_similarity,
                    AVG(cr.relevance_score) as avg_relevance
                FROM context_retrievals cr
                JOIN contexts c ON cr.context_id = c.id
                WHERE {where_clause}
                GROUP BY cr.context_id, c.title
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)
            
            top_contexts_result = await self.db.execute(top_contexts_query, params)
            top_contexts = [
                {
                    "context_id": str(r.context_id),
                    "title": r.title,
                    "retrieval_count": r.retrieval_count,
                    "avg_similarity": float(r.avg_similarity) if r.avg_similarity else None,
                    "avg_relevance": float(r.avg_relevance) if r.avg_relevance else None
                }
                for r in top_contexts_result
            ]
            
            # Top queries query
            top_queries_query = text(f"""
                SELECT 
                    cr.query_text,
                    COUNT(*) as query_count,
                    AVG(cr.similarity_score) as avg_similarity
                FROM context_retrievals cr
                WHERE {where_clause} AND cr.query_text IS NOT NULL
                GROUP BY cr.query_text
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)
            
            top_queries_result = await self.db.execute(top_queries_query, params)
            top_queries = [
                {
                    "query": r.query_text,
                    "count": r.query_count,
                    "avg_similarity": float(r.avg_similarity) if r.avg_similarity else None
                }
                for r in top_queries_result
            ]
            
            # Calculate derived metrics
            total_retrievals = row.total_retrievals or 0
            helpful_retrievals = row.helpful_retrievals or 0
            unhelpful_retrievals = row.unhelpful_retrievals or 0
            
            helpfulness_rate = 0.0
            if helpful_retrievals + unhelpful_retrievals > 0:
                helpfulness_rate = helpful_retrievals / (helpful_retrievals + unhelpful_retrievals)
            
            analytics = {
                "analysis_period_days": days_back,
                "total_retrievals": total_retrievals,
                "unique_contexts": row.unique_contexts or 0,
                "unique_agents": row.unique_agents or 0,
                "unique_sessions": row.unique_sessions or 0,
                "avg_similarity_score": float(row.avg_similarity_score) if row.avg_similarity_score else None,
                "avg_relevance_score": float(row.avg_relevance_score) if row.avg_relevance_score else None,
                "avg_response_time_ms": float(row.avg_response_time_ms) if row.avg_response_time_ms else None,
                "helpfulness_rate": helpfulness_rate,
                "avg_feedback_score": float(row.avg_feedback_score) if row.avg_feedback_score else None,
                "top_contexts": top_contexts,
                "top_queries": top_queries
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting usage analytics: {e}")
            return {}
    
    async def get_relationship_graph(
        self,
        context_id: uuid.UUID,
        max_depth: int = 2,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> Dict[str, Any]:
        """
        Get relationship graph for a context.
        
        Args:
            context_id: Root context for the graph
            max_depth: Maximum depth to traverse
            relationship_types: Types of relationships to include
            
        Returns:
            Graph data with nodes and edges
        """
        try:
            # Build relationship type filter
            type_filter = ""
            params = {"root_context_id": context_id, "max_depth": max_depth}
            
            if relationship_types:
                type_list = [rt.value for rt in relationship_types]
                type_filter = "AND relationship_type = ANY(:relationship_types)"
                params["relationship_types"] = type_list
            
            # Recursive query to build graph
            graph_query = text(f"""
                WITH RECURSIVE context_graph AS (
                    -- Base case: start with the root context
                    SELECT 
                        source_context_id,
                        target_context_id,
                        relationship_type,
                        similarity_score,
                        confidence_score,
                        1 as depth
                    FROM context_relationships
                    WHERE source_context_id = :root_context_id {type_filter}
                    
                    UNION
                    
                    -- Recursive case: follow relationships
                    SELECT 
                        cr.source_context_id,
                        cr.target_context_id,
                        cr.relationship_type,
                        cr.similarity_score,
                        cr.confidence_score,
                        cg.depth + 1
                    FROM context_relationships cr
                    JOIN context_graph cg ON cr.source_context_id = cg.target_context_id
                    WHERE cg.depth < :max_depth {type_filter}
                )
                SELECT DISTINCT * FROM context_graph
                ORDER BY depth, similarity_score DESC
            """)
            
            result = await self.db.execute(graph_query, params)
            relationships = result.all()
            
            # Get context information for all nodes
            context_ids = set()
            for rel in relationships:
                context_ids.add(rel.source_context_id)
                context_ids.add(rel.target_context_id)
            
            if context_ids:
                contexts_query = select(Context).where(Context.id.in_(context_ids))
                contexts_result = await self.db.execute(contexts_query)
                contexts = {ctx.id: ctx for ctx in contexts_result.scalars()}
            else:
                contexts = {}
            
            # Build graph structure
            nodes = []
            edges = []
            
            for ctx_id, ctx in contexts.items():
                nodes.append({
                    "id": str(ctx_id),
                    "title": ctx.title,
                    "context_type": ctx.context_type.value if ctx.context_type else None,
                    "importance_score": ctx.importance_score,
                    "created_at": ctx.created_at.isoformat() if ctx.created_at else None,
                    "is_root": ctx_id == context_id
                })
            
            for rel in relationships:
                edges.append({
                    "source": str(rel.source_context_id),
                    "target": str(rel.target_context_id),
                    "relationship_type": rel.relationship_type,
                    "similarity_score": float(rel.similarity_score) if rel.similarity_score else None,
                    "confidence_score": float(rel.confidence_score),
                    "depth": rel.depth
                })
            
            return {
                "root_context_id": str(context_id),
                "nodes": nodes,
                "edges": edges,
                "max_depth": max_depth,
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            }
            
        except Exception as e:
            logger.error(f"Error building relationship graph: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    def _classify_relationship(
        self,
        source_context: Context,
        target_context_data: Dict[str, Any],
        similarity_score: float
    ) -> RelationshipType:
        """Classify the type of relationship between contexts."""
        # High similarity suggests SIMILAR relationship
        if similarity_score >= 0.9:
            return RelationshipType.SIMILAR
        
        # Check if one context is newer and might supersede the other
        if source_context.created_at and target_context_data.get("created_at"):
            time_diff = abs((source_context.created_at - target_context_data["created_at"]).days)
            if time_diff > 30 and similarity_score >= 0.8:
                if source_context.created_at > target_context_data["created_at"]:
                    return RelationshipType.SUPERSEDES
                else:
                    return RelationshipType.DERIVED_FROM
        
        # Check context types for potential relationships
        if (source_context.context_type and 
            source_context.context_type.value == target_context_data.get("context_type")):
            return RelationshipType.RELATED
        
        # Default to RELATED for moderate similarity
        if similarity_score >= 0.75:
            return RelationshipType.RELATED
        
        # Low similarity but still above threshold
        return RelationshipType.REFERENCES
    
    def _calculate_relationship_confidence(
        self,
        similarity_score: float,
        source_importance: float,
        target_importance: float
    ) -> float:
        """Calculate confidence score for a relationship."""
        # Base confidence on similarity score
        base_confidence = similarity_score
        
        # Boost confidence if both contexts have high importance
        importance_boost = (source_importance + target_importance) / 2 * 0.1
        
        # Ensure confidence doesn't exceed 1.0
        confidence = min(1.0, base_confidence + importance_boost)
        
        return confidence
    
    async def _store_relationships(self, relationships: List[ContextRelationship]) -> None:
        """Store relationships in the database."""
        if not relationships:
            return
        
        try:
            for rel in relationships:
                insert_query = text("""
                    INSERT INTO context_relationships (
                        id, source_context_id, target_context_id, relationship_type,
                        similarity_score, confidence_score, metadata, created_at
                    ) VALUES (
                        :id, :source_context_id, :target_context_id, :relationship_type,
                        :similarity_score, :confidence_score, :metadata, :created_at
                    ) ON CONFLICT (source_context_id, target_context_id, relationship_type) 
                    DO NOTHING
                """)
                
                await self.db.execute(
                    insert_query,
                    {
                        "id": rel.id,
                        "source_context_id": rel.source_context_id,
                        "target_context_id": rel.target_context_id,
                        "relationship_type": rel.relationship_type.value,
                        "similarity_score": rel.similarity_score,
                        "confidence_score": rel.confidence_score,
                        "metadata": rel.metadata,
                        "created_at": rel.created_at
                    }
                )
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing relationships: {e}")
            await self.db.rollback()
            raise
    
    async def _store_retrieval_record(self, retrieval: ContextRetrieval) -> None:
        """Store retrieval record in the database."""
        try:
            insert_query = text("""
                INSERT INTO context_retrievals (
                    id, context_id, requesting_agent_id, session_id, query_text,
                    query_embedding, similarity_score, relevance_score, rank_position,
                    was_helpful, feedback_score, retrieval_method, response_time_ms,
                    metadata, retrieved_at
                ) VALUES (
                    :id, :context_id, :requesting_agent_id, :session_id, :query_text,
                    :query_embedding, :similarity_score, :relevance_score, :rank_position,
                    :was_helpful, :feedback_score, :retrieval_method, :response_time_ms,
                    :metadata, :retrieved_at
                )
            """)
            
            await self.db.execute(
                insert_query,
                {
                    "id": retrieval.id,
                    "context_id": retrieval.context_id,
                    "requesting_agent_id": retrieval.requesting_agent_id,
                    "session_id": retrieval.session_id,
                    "query_text": retrieval.query_text,
                    "query_embedding": retrieval.query_embedding,
                    "similarity_score": retrieval.similarity_score,
                    "relevance_score": retrieval.relevance_score,
                    "rank_position": retrieval.rank_position,
                    "was_helpful": retrieval.was_helpful,
                    "feedback_score": retrieval.feedback_score,
                    "retrieval_method": retrieval.retrieval_method,
                    "response_time_ms": retrieval.response_time_ms,
                    "metadata": retrieval.metadata,
                    "retrieved_at": retrieval.retrieved_at
                }
            )
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing retrieval record: {e}")
            await self.db.rollback()
            raise


# Factory function
async def get_context_analytics_manager(
    db_session: AsyncSession,
    embedding_service: EmbeddingService
) -> ContextAnalyticsManager:
    """
    Get context analytics manager instance.
    
    Args:
        db_session: Database session
        embedding_service: Embedding service
        
    Returns:
        ContextAnalyticsManager instance
    """
    return ContextAnalyticsManager(db_session, embedding_service)