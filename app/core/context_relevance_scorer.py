"""
Context Relevance Scorer for LeanVibe Agent Hive 2.0

Intelligent context selection and ranking system that dynamically scores and ranks
context information based on relevance, importance, recency, and task-specific criteria.

Features:
- Multi-Factor Scoring: Combines relevance, importance, recency, and task-specificity
- Semantic Similarity: Uses embeddings for deep semantic relevance scoring
- Dynamic Ranking: Adapts scoring based on current task and agent context
- Context Caching: Optimized caching for frequently accessed contexts
- Adaptive Learning: Improves scoring based on usage patterns and feedback
- Performance Optimization: Sub-100ms scoring for real-time context selection
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

import structlog
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .semantic_embedding_service import get_embedding_service, SemanticEmbeddingService
from .agent_knowledge_manager import KnowledgeItem, KnowledgeType
from .memory_hierarchy_manager import MemoryItem, MemoryType, MemoryLevel

logger = structlog.get_logger()


# =============================================================================
# RELEVANCE SCORING TYPES AND CONFIGURATIONS
# =============================================================================

class ScoringStrategy(str, Enum):
    """Strategies for relevance scoring."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_MATCHING = "keyword_matching"
    TEMPORAL_RELEVANCE = "temporal_relevance"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    USAGE_FREQUENCY = "usage_frequency"
    HYBRID_MULTI_FACTOR = "hybrid_multi_factor"


class ContextType(str, Enum):
    """Types of context for relevance scoring."""
    TASK_CONTEXT = "task_context"
    AGENT_CONTEXT = "agent_context"
    WORKFLOW_CONTEXT = "workflow_context"
    DOMAIN_CONTEXT = "domain_context"
    HISTORICAL_CONTEXT = "historical_context"


class RelevanceFactors(str, Enum):
    """Factors that contribute to relevance scoring."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_OVERLAP = "keyword_overlap"
    RECENCY = "recency"
    IMPORTANCE = "importance"
    USAGE_FREQUENCY = "usage_frequency"
    AGENT_PREFERENCE = "agent_preference"
    TASK_SPECIFICITY = "task_specificity"
    CONTEXT_COMPLETENESS = "context_completeness"


@dataclass
class ScoringConfig:
    """Configuration for relevance scoring."""
    strategy: ScoringStrategy = ScoringStrategy.HYBRID_MULTI_FACTOR
    
    # Factor weights (must sum to 1.0)
    semantic_weight: float = 0.3
    keyword_weight: float = 0.15
    recency_weight: float = 0.15
    importance_weight: float = 0.2
    usage_weight: float = 0.1
    task_specificity_weight: float = 0.1
    
    # Scoring parameters
    min_semantic_threshold: float = 0.3
    recency_decay_hours: float = 168.0  # 1 week
    usage_boost_factor: float = 1.5
    importance_boost_threshold: float = 0.8
    
    # Performance settings
    max_contexts_to_score: int = 1000
    embedding_cache_size: int = 500
    enable_adaptive_learning: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        total_weight = (
            self.semantic_weight + self.keyword_weight + self.recency_weight +
            self.importance_weight + self.usage_weight + self.task_specificity_weight
        )
        return abs(total_weight - 1.0) < 0.01  # Allow small floating point errors


@dataclass
class ContextItem:
    """Context item for relevance scoring."""
    context_id: str
    content: str
    context_type: ContextType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    # Quality metrics
    importance_score: float = 0.5
    quality_score: float = 0.5
    completeness_score: float = 0.5
    
    # Task-specific information
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Caching
    embedding: Optional[List[float]] = None
    keywords: Optional[List[str]] = None
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def calculate_age_hours(self) -> float:
        """Calculate age in hours."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "content": self.content,
            "context_type": self.context_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "quality_score": self.quality_score,
            "completeness_score": self.completeness_score,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "tags": self.tags
        }


@dataclass
class RelevanceScore:
    """Detailed relevance score for a context item."""
    context_id: str
    overall_score: float
    
    # Individual factor scores
    semantic_similarity: float = 0.0
    keyword_overlap: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    usage_score: float = 0.0
    task_specificity: float = 0.0
    
    # Metadata
    scoring_strategy: ScoringStrategy = ScoringStrategy.HYBRID_MULTI_FACTOR
    scoring_time_ms: float = 0.0
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "context_id": self.context_id,
            "overall_score": self.overall_score,
            "factor_scores": {
                "semantic_similarity": self.semantic_similarity,
                "keyword_overlap": self.keyword_overlap,
                "recency_score": self.recency_score,
                "importance_score": self.importance_score,
                "usage_score": self.usage_score,
                "task_specificity": self.task_specificity
            },
            "scoring_strategy": self.scoring_strategy.value,
            "scoring_time_ms": self.scoring_time_ms,
            "explanation": self.explanation
        }


@dataclass
class ScoringRequest:
    """Request for context relevance scoring."""
    query: str
    contexts: List[ContextItem]
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    context_types: Optional[List[ContextType]] = None
    max_results: int = 10
    min_score_threshold: float = 0.1
    config: Optional[ScoringConfig] = None


@dataclass
class ScoringResult:
    """Result of context relevance scoring."""
    request_id: str
    query: str
    scored_contexts: List[Tuple[ContextItem, RelevanceScore]]
    total_contexts_evaluated: int
    processing_time_ms: float
    strategy_used: ScoringStrategy
    
    def get_top_contexts(self, limit: int = 10) -> List[Tuple[ContextItem, RelevanceScore]]:
        """Get top N contexts by relevance score."""
        return self.scored_contexts[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "query": self.query,
            "scored_contexts": [
                {
                    "context": context.to_dict(),
                    "relevance_score": score.to_dict()
                }
                for context, score in self.scored_contexts
            ],
            "total_contexts_evaluated": self.total_contexts_evaluated,
            "processing_time_ms": self.processing_time_ms,
            "strategy_used": self.strategy_used.value
        }


# =============================================================================
# INDIVIDUAL SCORING ALGORITHMS
# =============================================================================

class SemanticSimilarityScorer:
    """Scores contexts based on semantic similarity using embeddings."""
    
    def __init__(self, embedding_service: SemanticEmbeddingService):
        self.embedding_service = embedding_service
        self.embedding_cache: Dict[str, List[float]] = {}
    
    async def score_contexts(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[float]:
        """Score contexts based on semantic similarity."""
        try:
            # Get query embedding
            query_embedding = await self._get_cached_embedding(query)
            if not query_embedding:
                return [0.0] * len(contexts)
            
            scores = []
            
            for context in contexts:
                # Get context embedding
                context_embedding = await self._get_context_embedding(context)
                
                if context_embedding:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        np.array(query_embedding).reshape(1, -1),
                        np.array(context_embedding).reshape(1, -1)
                    )[0][0]
                    
                    # Apply minimum threshold
                    score = max(0.0, similarity) if similarity >= config.min_semantic_threshold else 0.0
                    scores.append(score)
                else:
                    scores.append(0.0)
            
            return scores
            
        except Exception as e:
            logger.error(f"Semantic similarity scoring failed: {e}")
            return [0.0] * len(contexts)
    
    async def _get_context_embedding(self, context: ContextItem) -> Optional[List[float]]:
        """Get embedding for context, using cache if available."""
        if context.embedding:
            return context.embedding
        
        embedding = await self._get_cached_embedding(context.content)
        if embedding:
            context.embedding = embedding  # Cache in context
        
        return embedding
    
    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with caching."""
        text_hash = str(hash(text))
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        embedding = await self.embedding_service.generate_embedding(text)
        
        if embedding:
            # Cache with size limit
            if len(self.embedding_cache) >= 500:  # Max cache size
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            
            self.embedding_cache[text_hash] = embedding
        
        return embedding


class KeywordMatchingScorer:
    """Scores contexts based on keyword overlap."""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.keyword_cache: Dict[str, List[str]] = {}
    
    def score_contexts(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[float]:
        """Score contexts based on keyword overlap."""
        try:
            query_keywords = self._extract_keywords(query)
            scores = []
            
            for context in contexts:
                context_keywords = self._get_context_keywords(context)
                
                if not query_keywords or not context_keywords:
                    scores.append(0.0)
                    continue
                
                # Calculate keyword overlap
                query_set = set(query_keywords)
                context_set = set(context_keywords)
                
                overlap = len(query_set & context_set)
                union = len(query_set | context_set)
                
                # Jaccard similarity
                jaccard_score = overlap / union if union > 0 else 0.0
                
                # Weight by keyword importance (TF-IDF style)
                weighted_score = jaccard_score
                
                scores.append(weighted_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Keyword matching scoring failed: {e}")
            return [0.0] * len(contexts)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        text_hash = str(hash(text))
        
        if text_hash in self.keyword_cache:
            return self.keyword_cache[text_hash]
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = text.lower().split()
        keywords = [word for word in words if len(word) > 3 and word.isalpha()]
        
        # Cache with size limit
        if len(self.keyword_cache) >= 200:
            oldest_key = next(iter(self.keyword_cache))
            del self.keyword_cache[oldest_key]
        
        self.keyword_cache[text_hash] = keywords
        return keywords
    
    def _get_context_keywords(self, context: ContextItem) -> List[str]:
        """Get keywords for context, using cache if available."""
        if context.keywords:
            return context.keywords
        
        keywords = self._extract_keywords(context.content)
        context.keywords = keywords  # Cache in context
        
        return keywords


class TemporalRelevanceScorer:
    """Scores contexts based on temporal relevance (recency)."""
    
    def score_contexts(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[float]:
        """Score contexts based on recency."""
        try:
            current_time = datetime.utcnow()
            scores = []
            
            for context in contexts:
                # Calculate age
                age_hours = (current_time - context.created_at).total_seconds() / 3600
                
                # Exponential decay
                decay_factor = config.recency_decay_hours
                recency_score = max(0.0, np.exp(-age_hours / decay_factor))
                
                # Boost recently accessed items
                if context.last_accessed:
                    last_access_hours = (current_time - context.last_accessed).total_seconds() / 3600
                    if last_access_hours < 24:  # Recent access boost
                        access_boost = 1.2 * np.exp(-last_access_hours / 24)
                        recency_score = min(1.0, recency_score * access_boost)
                
                scores.append(recency_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Temporal relevance scoring failed: {e}")
            return [0.0] * len(contexts)


class ImportanceWeightedScorer:
    """Scores contexts based on importance and quality metrics."""
    
    def score_contexts(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[float]:
        """Score contexts based on importance."""
        try:
            scores = []
            
            for context in contexts:
                # Base importance score
                importance = context.importance_score
                
                # Quality weighting
                quality_weight = (context.quality_score + context.completeness_score) / 2
                
                # Combine importance and quality
                weighted_importance = importance * (0.7 + 0.3 * quality_weight)
                
                # Boost for very high importance
                if importance >= config.importance_boost_threshold:
                    weighted_importance = min(1.0, weighted_importance * 1.3)
                
                scores.append(weighted_importance)
            
            return scores
            
        except Exception as e:
            logger.error(f"Importance weighted scoring failed: {e}")
            return [0.0] * len(contexts)


class UsageFrequencyScorer:
    """Scores contexts based on usage patterns and frequency."""
    
    def score_contexts(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[float]:
        """Score contexts based on usage frequency."""
        try:
            # Calculate max access count for normalization
            max_access_count = max((ctx.access_count for ctx in contexts), default=1)
            
            scores = []
            
            for context in contexts:
                # Normalize access count
                usage_score = context.access_count / max(1, max_access_count)
                
                # Apply usage boost factor
                if context.access_count > 0:
                    usage_score = min(1.0, usage_score * config.usage_boost_factor)
                
                # Recent usage boost
                if context.last_accessed:
                    hours_since_access = (datetime.utcnow() - context.last_accessed).total_seconds() / 3600
                    if hours_since_access < 48:  # Recent usage boost
                        recent_boost = 1.5 * np.exp(-hours_since_access / 48)
                        usage_score = min(1.0, usage_score * recent_boost)
                
                scores.append(usage_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Usage frequency scoring failed: {e}")
            return [0.0] * len(contexts)


class TaskSpecificityScorer:
    """Scores contexts based on task-specific relevance."""
    
    def score_contexts(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> List[float]:
        """Score contexts based on task specificity."""
        try:
            scores = []
            
            for context in contexts:
                specificity_score = 0.0
                
                # Agent matching
                if agent_id and context.agent_id == agent_id:
                    specificity_score += 0.3
                
                # Task matching
                if task_id and context.task_id == task_id:
                    specificity_score += 0.4
                
                # Workflow matching
                if workflow_id and context.workflow_id == workflow_id:
                    specificity_score += 0.3
                
                # Context type relevance
                if context.context_type in [ContextType.TASK_CONTEXT, ContextType.WORKFLOW_CONTEXT]:
                    specificity_score += 0.2
                
                # Tag matching with query
                query_words = set(query.lower().split())
                context_tags = set(tag.lower() for tag in context.tags)
                
                if query_words & context_tags:
                    tag_overlap = len(query_words & context_tags) / len(query_words | context_tags)
                    specificity_score += tag_overlap * 0.3
                
                scores.append(min(1.0, specificity_score))
            
            return scores
            
        except Exception as e:
            logger.error(f"Task specificity scoring failed: {e}")
            return [0.0] * len(contexts)


# =============================================================================
# MAIN CONTEXT RELEVANCE SCORER
# =============================================================================

class ContextRelevanceScorer:
    """
    Main context relevance scorer that combines multiple scoring strategies
    to provide intelligent context selection and ranking.
    """
    
    def __init__(self, embedding_service: Optional[SemanticEmbeddingService] = None):
        """Initialize the context relevance scorer."""
        self.embedding_service = embedding_service
        
        # Individual scorers
        self.semantic_scorer = None
        self.keyword_scorer = KeywordMatchingScorer()
        self.temporal_scorer = TemporalRelevanceScorer()
        self.importance_scorer = ImportanceWeightedScorer()
        self.usage_scorer = UsageFrequencyScorer()
        self.task_scorer = TaskSpecificityScorer()
        
        # Adaptive learning
        self.scoring_history: List[Dict[str, Any]] = []
        self.performance_feedback: Dict[str, List[float]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            "total_scoring_requests": 0,
            "avg_scoring_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "adaptive_adjustments": 0
        }
        
        logger.info("Context Relevance Scorer initialized")
    
    async def initialize(self):
        """Initialize the scorer with required services."""
        if not self.embedding_service:
            self.embedding_service = await get_embedding_service()
        
        self.semantic_scorer = SemanticSimilarityScorer(self.embedding_service)
        
        logger.info("✅ Context Relevance Scorer fully initialized")
    
    # =============================================================================
    # MAIN SCORING OPERATIONS
    # =============================================================================
    
    async def score_contexts(
        self,
        request: ScoringRequest
    ) -> ScoringResult:
        """Score and rank contexts based on relevance."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Validate configuration
            config = request.config or ScoringConfig()
            if not config.validate():
                raise ValueError("Invalid scoring configuration")
            
            # Filter contexts if needed
            contexts_to_score = self._filter_contexts(request.contexts, request)
            
            if not contexts_to_score:
                return ScoringResult(
                    request_id=request_id,
                    query=request.query,
                    scored_contexts=[],
                    total_contexts_evaluated=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    strategy_used=config.strategy
                )
            
            # Perform scoring based on strategy
            if config.strategy == ScoringStrategy.HYBRID_MULTI_FACTOR:
                scored_contexts = await self._hybrid_multi_factor_scoring(
                    request.query, contexts_to_score, config, request
                )
            elif config.strategy == ScoringStrategy.SEMANTIC_SIMILARITY:
                scored_contexts = await self._semantic_only_scoring(
                    request.query, contexts_to_score, config
                )
            elif config.strategy == ScoringStrategy.KEYWORD_MATCHING:
                scored_contexts = await self._keyword_only_scoring(
                    request.query, contexts_to_score, config
                )
            elif config.strategy == ScoringStrategy.TEMPORAL_RELEVANCE:
                scored_contexts = await self._temporal_only_scoring(
                    request.query, contexts_to_score, config
                )
            elif config.strategy == ScoringStrategy.IMPORTANCE_WEIGHTED:
                scored_contexts = await self._importance_only_scoring(
                    request.query, contexts_to_score, config
                )
            else:
                # Default to hybrid
                scored_contexts = await self._hybrid_multi_factor_scoring(
                    request.query, contexts_to_score, config, request
                )
            
            # Filter by minimum score threshold
            filtered_contexts = [
                (ctx, score) for ctx, score in scored_contexts
                if score.overall_score >= request.min_score_threshold
            ]
            
            # Sort by relevance score
            filtered_contexts.sort(key=lambda x: x[1].overall_score, reverse=True)
            
            # Limit results
            final_contexts = filtered_contexts[:request.max_results]
            
            # Update access statistics for selected contexts
            for ctx, _ in final_contexts:
                ctx.update_access()
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_metrics(processing_time, len(contexts_to_score))
            
            # Store for adaptive learning
            if config.enable_adaptive_learning:
                self._record_scoring_session(request, final_contexts, processing_time)
            
            result = ScoringResult(
                request_id=request_id,
                query=request.query,
                scored_contexts=final_contexts,
                total_contexts_evaluated=len(contexts_to_score),
                processing_time_ms=processing_time,
                strategy_used=config.strategy
            )
            
            logger.debug(
                f"Context scoring completed",
                request_id=request_id,
                contexts_scored=len(final_contexts),
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Context scoring failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return ScoringResult(
                request_id=request_id,
                query=request.query,
                scored_contexts=[],
                total_contexts_evaluated=0,
                processing_time_ms=processing_time,
                strategy_used=ScoringStrategy.HYBRID_MULTI_FACTOR
            )
    
    async def quick_score(
        self,
        query: str,
        contexts: List[ContextItem],
        max_results: int = 5
    ) -> List[Tuple[ContextItem, float]]:
        """Quick scoring for real-time applications (simplified)."""
        try:
            if not contexts:
                return []
            
            # Use lightweight scoring for speed
            config = ScoringConfig(
                strategy=ScoringStrategy.KEYWORD_MATCHING,
                semantic_weight=0.0,  # Skip semantic for speed
                keyword_weight=0.4,
                recency_weight=0.3,
                importance_weight=0.3,
                max_contexts_to_score=min(100, len(contexts))  # Limit for speed
            )
            
            request = ScoringRequest(
                query=query,
                contexts=contexts[:config.max_contexts_to_score],
                max_results=max_results,
                config=config
            )
            
            result = await self.score_contexts(request)
            
            # Return simplified format
            return [
                (ctx, score.overall_score)
                for ctx, score in result.scored_contexts
            ]
            
        except Exception as e:
            logger.error(f"Quick scoring failed: {e}")
            return []
    
    # =============================================================================
    # SPECIALIZED SCORING STRATEGIES
    # =============================================================================
    
    async def _hybrid_multi_factor_scoring(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig,
        request: ScoringRequest
    ) -> List[Tuple[ContextItem, RelevanceScore]]:
        """Perform hybrid multi-factor scoring."""
        
        # Calculate individual factor scores
        semantic_scores = await self.semantic_scorer.score_contexts(query, contexts, config)
        keyword_scores = self.keyword_scorer.score_contexts(query, contexts, config)
        temporal_scores = self.temporal_scorer.score_contexts(query, contexts, config)
        importance_scores = self.importance_scorer.score_contexts(query, contexts, config)
        usage_scores = self.usage_scorer.score_contexts(query, contexts, config)
        task_scores = self.task_scorer.score_contexts(
            query, contexts, config, request.agent_id, request.task_id, request.workflow_id
        )
        
        scored_contexts = []
        
        for i, context in enumerate(contexts):
            start_time = time.time()
            
            # Get individual scores
            semantic = semantic_scores[i]
            keyword = keyword_scores[i]
            temporal = temporal_scores[i]
            importance = importance_scores[i]
            usage = usage_scores[i]
            task_specific = task_scores[i]
            
            # Calculate weighted overall score
            overall = (
                semantic * config.semantic_weight +
                keyword * config.keyword_weight +
                temporal * config.recency_weight +
                importance * config.importance_weight +
                usage * config.usage_weight +
                task_specific * config.task_specificity_weight
            )
            
            scoring_time = (time.time() - start_time) * 1000
            
            # Generate explanation
            explanation = self._generate_score_explanation(
                semantic, keyword, temporal, importance, usage, task_specific, config
            )
            
            # Create relevance score
            relevance_score = RelevanceScore(
                context_id=context.context_id,
                overall_score=overall,
                semantic_similarity=semantic,
                keyword_overlap=keyword,
                recency_score=temporal,
                importance_score=importance,
                usage_score=usage,
                task_specificity=task_specific,
                scoring_strategy=config.strategy,
                scoring_time_ms=scoring_time,
                explanation=explanation
            )
            
            scored_contexts.append((context, relevance_score))
        
        return scored_contexts
    
    async def _semantic_only_scoring(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[Tuple[ContextItem, RelevanceScore]]:
        """Perform semantic similarity only scoring."""
        semantic_scores = await self.semantic_scorer.score_contexts(query, contexts, config)
        
        scored_contexts = []
        for i, context in enumerate(contexts):
            score = semantic_scores[i]
            
            relevance_score = RelevanceScore(
                context_id=context.context_id,
                overall_score=score,
                semantic_similarity=score,
                scoring_strategy=ScoringStrategy.SEMANTIC_SIMILARITY,
                explanation=f"Semantic similarity: {score:.3f}"
            )
            
            scored_contexts.append((context, relevance_score))
        
        return scored_contexts
    
    async def _keyword_only_scoring(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[Tuple[ContextItem, RelevanceScore]]:
        """Perform keyword matching only scoring."""
        keyword_scores = self.keyword_scorer.score_contexts(query, contexts, config)
        
        scored_contexts = []
        for i, context in enumerate(contexts):
            score = keyword_scores[i]
            
            relevance_score = RelevanceScore(
                context_id=context.context_id,
                overall_score=score,
                keyword_overlap=score,
                scoring_strategy=ScoringStrategy.KEYWORD_MATCHING,
                explanation=f"Keyword overlap: {score:.3f}"
            )
            
            scored_contexts.append((context, relevance_score))
        
        return scored_contexts
    
    async def _temporal_only_scoring(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[Tuple[ContextItem, RelevanceScore]]:
        """Perform temporal relevance only scoring."""
        temporal_scores = self.temporal_scorer.score_contexts(query, contexts, config)
        
        scored_contexts = []
        for i, context in enumerate(contexts):
            score = temporal_scores[i]
            
            relevance_score = RelevanceScore(
                context_id=context.context_id,
                overall_score=score,
                recency_score=score,
                scoring_strategy=ScoringStrategy.TEMPORAL_RELEVANCE,
                explanation=f"Temporal relevance: {score:.3f}"
            )
            
            scored_contexts.append((context, relevance_score))
        
        return scored_contexts
    
    async def _importance_only_scoring(
        self,
        query: str,
        contexts: List[ContextItem],
        config: ScoringConfig
    ) -> List[Tuple[ContextItem, RelevanceScore]]:
        """Perform importance weighted only scoring."""
        importance_scores = self.importance_scorer.score_contexts(query, contexts, config)
        
        scored_contexts = []
        for i, context in enumerate(contexts):
            score = importance_scores[i]
            
            relevance_score = RelevanceScore(
                context_id=context.context_id,
                overall_score=score,
                importance_score=score,
                scoring_strategy=ScoringStrategy.IMPORTANCE_WEIGHTED,
                explanation=f"Importance weighted: {score:.3f}"
            )
            
            scored_contexts.append((context, relevance_score))
        
        return scored_contexts
    
    # =============================================================================
    # ADAPTIVE LEARNING AND OPTIMIZATION
    # =============================================================================
    
    def provide_feedback(
        self,
        request_id: str,
        context_id: str,
        feedback_score: float,
        feedback_type: str = "relevance"
    ):
        """Provide feedback on scoring accuracy for adaptive learning."""
        try:
            feedback_key = f"{feedback_type}_{context_id}"
            self.performance_feedback[feedback_key].append(feedback_score)
            
            # Trigger adaptive adjustment if enough feedback
            if len(self.performance_feedback[feedback_key]) >= 5:
                self._adapt_scoring_weights(feedback_key)
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
    
    def _adapt_scoring_weights(self, feedback_key: str):
        """Adapt scoring weights based on feedback."""
        try:
            feedback_scores = self.performance_feedback[feedback_key]
            avg_feedback = sum(feedback_scores) / len(feedback_scores)
            
            # Simple adaptation logic (could be more sophisticated)
            if avg_feedback < 0.3:  # Poor performance
                logger.info(f"Adapting scoring weights due to poor feedback: {avg_feedback:.3f}")
                self.metrics["adaptive_adjustments"] += 1
                
                # This would trigger more sophisticated weight adjustment
                # For now, just log the adaptation
            
        except Exception as e:
            logger.error(f"Adaptive weight adjustment failed: {e}")
    
    def _record_scoring_session(
        self,
        request: ScoringRequest,
        results: List[Tuple[ContextItem, RelevanceScore]],
        processing_time: float
    ):
        """Record scoring session for learning purposes."""
        try:
            session_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": request.query,
                "agent_id": request.agent_id,
                "task_id": request.task_id,
                "contexts_count": len(request.contexts),
                "results_count": len(results),
                "processing_time_ms": processing_time,
                "top_score": results[0][1].overall_score if results else 0.0
            }
            
            self.scoring_history.append(session_record)
            
            # Keep only recent history
            if len(self.scoring_history) > 1000:
                self.scoring_history = self.scoring_history[-500:]
            
        except Exception as e:
            logger.error(f"Failed to record scoring session: {e}")
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _filter_contexts(
        self,
        contexts: List[ContextItem],
        request: ScoringRequest
    ) -> List[ContextItem]:
        """Filter contexts based on request criteria."""
        filtered = contexts
        
        # Filter by context types if specified
        if request.context_types:
            filtered = [ctx for ctx in filtered if ctx.context_type in request.context_types]
        
        # Limit number of contexts for performance
        config = request.config or ScoringConfig()
        if len(filtered) > config.max_contexts_to_score:
            # Sort by importance before limiting
            filtered.sort(key=lambda ctx: ctx.importance_score, reverse=True)
            filtered = filtered[:config.max_contexts_to_score]
        
        return filtered
    
    def _generate_score_explanation(
        self,
        semantic: float,
        keyword: float,
        temporal: float,
        importance: float,
        usage: float,
        task_specific: float,
        config: ScoringConfig
    ) -> str:
        """Generate human-readable explanation of scoring."""
        components = []
        
        if semantic > 0.1:
            components.append(f"semantic similarity ({semantic:.2f})")
        if keyword > 0.1:
            components.append(f"keyword overlap ({keyword:.2f})")
        if temporal > 0.1:
            components.append(f"recency ({temporal:.2f})")
        if importance > 0.1:
            components.append(f"importance ({importance:.2f})")
        if usage > 0.1:
            components.append(f"usage frequency ({usage:.2f})")
        if task_specific > 0.1:
            components.append(f"task relevance ({task_specific:.2f})")
        
        if components:
            return f"Score based on: {', '.join(components)}"
        else:
            return "Low relevance across all factors"
    
    def _update_metrics(self, processing_time: float, contexts_scored: int):
        """Update performance metrics."""
        self.metrics["total_scoring_requests"] += 1
        
        if self.metrics["total_scoring_requests"] > 0:
            self.metrics["avg_scoring_time_ms"] = (
                (self.metrics["avg_scoring_time_ms"] * (self.metrics["total_scoring_requests"] - 1) +
                 processing_time) / self.metrics["total_scoring_requests"]
            )
    
    # =============================================================================
    # UTILITY CONVERSION METHODS
    # =============================================================================
    
    def knowledge_item_to_context(self, knowledge_item: KnowledgeItem) -> ContextItem:
        """Convert KnowledgeItem to ContextItem."""
        return ContextItem(
            context_id=knowledge_item.knowledge_id,
            content=knowledge_item.content,
            context_type=ContextType.DOMAIN_CONTEXT,
            metadata=knowledge_item.metadata,
            created_at=knowledge_item.created_at,
            last_accessed=knowledge_item.last_used,
            access_count=knowledge_item.usage_count,
            importance_score=knowledge_item.confidence_score,
            quality_score=knowledge_item.confidence_score,
            completeness_score=0.8,  # Default
            agent_id=knowledge_item.agent_id,
            tags=knowledge_item.tags
        )
    
    def memory_item_to_context(self, memory_item: MemoryItem) -> ContextItem:
        """Convert MemoryItem to ContextItem."""
        context_type_mapping = {
            MemoryType.EPISODIC: ContextType.HISTORICAL_CONTEXT,
            MemoryType.SEMANTIC: ContextType.DOMAIN_CONTEXT,
            MemoryType.PROCEDURAL: ContextType.TASK_CONTEXT,
            MemoryType.META_COGNITIVE: ContextType.AGENT_CONTEXT,
            MemoryType.CONTEXTUAL: ContextType.WORKFLOW_CONTEXT
        }
        
        return ContextItem(
            context_id=memory_item.memory_id,
            content=memory_item.content,
            context_type=context_type_mapping.get(memory_item.memory_type, ContextType.DOMAIN_CONTEXT),
            metadata=memory_item.context,
            created_at=memory_item.created_at,
            last_accessed=memory_item.last_accessed,
            access_count=memory_item.access_count,
            importance_score=memory_item.importance_score,
            quality_score=memory_item.relevance_score,
            completeness_score=memory_item.confidence_score,
            agent_id=memory_item.agent_id,
            tags=memory_item.tags
        )
    
    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the context relevance scorer."""
        return {
            **self.metrics,
            "scoring_history_size": len(self.scoring_history),
            "feedback_records": sum(len(scores) for scores in self.performance_feedback.values()),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if self.semantic_scorer else 0.0
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on context relevance scorer."""
        try:
            # Test scoring with sample contexts
            test_contexts = [
                ContextItem(
                    context_id="test_1",
                    content="Test context for health check with machine learning algorithms",
                    context_type=ContextType.DOMAIN_CONTEXT,
                    importance_score=0.8,
                    tags=["test", "machine-learning"]
                ),
                ContextItem(
                    context_id="test_2",
                    content="Another test context about data processing and analytics",
                    context_type=ContextType.TASK_CONTEXT,
                    importance_score=0.6,
                    tags=["test", "data-processing"]
                )
            ]
            
            request = ScoringRequest(
                query="machine learning algorithms",
                contexts=test_contexts,
                max_results=2
            )
            
            result = await self.score_contexts(request)
            
            return {
                "status": "healthy",
                "components": {
                    "semantic_scorer": "operational" if self.semantic_scorer else "unavailable",
                    "keyword_scorer": "operational",
                    "temporal_scorer": "operational",
                    "importance_scorer": "operational",
                    "usage_scorer": "operational",
                    "task_scorer": "operational"
                },
                "test_results": {
                    "scoring_functional": len(result.scored_contexts) > 0,
                    "processing_time_ms": result.processing_time_ms,
                    "top_score": result.scored_contexts[0][1].overall_score if result.scored_contexts else 0.0
                },
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Context relevance scorer health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "components": {
                    "semantic_scorer": "unknown",
                    "keyword_scorer": "unknown",
                    "temporal_scorer": "unknown",
                    "importance_scorer": "unknown",
                    "usage_scorer": "unknown",
                    "task_scorer": "unknown"
                }
            }


# =============================================================================
# GLOBAL CONTEXT RELEVANCE SCORER INSTANCE
# =============================================================================

_relevance_scorer: Optional[ContextRelevanceScorer] = None


async def get_context_relevance_scorer() -> ContextRelevanceScorer:
    """Get global context relevance scorer instance."""
    global _relevance_scorer
    
    if _relevance_scorer is None:
        _relevance_scorer = ContextRelevanceScorer()
        await _relevance_scorer.initialize()
    
    return _relevance_scorer


async def cleanup_context_relevance_scorer():
    """Clean up global context relevance scorer."""
    global _relevance_scorer
    _relevance_scorer = None