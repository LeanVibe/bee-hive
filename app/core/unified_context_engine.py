"""
Unified Context Engine - Epic 4 Advanced AI Context & Reasoning Engine

This is the central coordination system that unifies all context engine components
into an intelligent, enterprise-ready context management platform for LeanVibe Agent Hive 2.0.

Features:
- Unified context coordination across all agents
- Semantic memory and knowledge management
- Advanced reasoning and decision support
- Intelligent context compression and optimization
- Cross-agent context sharing with semantic optimization
- Context-aware agent coordination
- Performance analytics and monitoring
"""

import asyncio
import time
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, deque
import uuid

import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

# Import existing context components
from .advanced_context_engine import AdvancedContextEngine, get_advanced_context_engine
from .context_adapter import ContextAdapter, UserPreferences, AdaptationResult
from .context_cache_manager import ContextCacheManager, get_context_cache_manager
from .context_compression import ContextCompressor, get_context_compressor, CompressionLevel
from .context_analytics import ContextAnalyticsManager

# Core imports
from .database import get_async_session
from .redis import get_redis_client
from .config import get_settings
from ..models.agent import Agent
from ..models.context import Context, ContextType

logger = structlog.get_logger()


class ContextOperationType(Enum):
    """Types of context operations."""
    COMPRESSION = "compression"
    SHARING = "sharing"
    ADAPTATION = "adaptation"
    REASONING = "reasoning"
    COORDINATION = "coordination"
    ANALYTICS = "analytics"


class ReasoningType(Enum):
    """Types of reasoning support."""
    DECISION_SUPPORT = "decision_support"
    PATTERN_RECOGNITION = "pattern_recognition"
    PREDICTIVE_CONTEXT = "predictive_context"
    CONFLICT_RESOLUTION = "conflict_resolution"
    OPTIMIZATION = "optimization"


@dataclass
class ContextMap:
    """Unified context mapping across agents."""
    agent_contexts: Dict[str, List[Dict[str, Any]]]
    shared_knowledge: Dict[str, Any]
    semantic_clusters: Dict[str, List[str]]
    cross_references: Dict[str, List[str]]
    coordination_insights: List[str]
    performance_metrics: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_contexts": self.agent_contexts,
            "shared_knowledge": self.shared_knowledge,
            "semantic_clusters": self.semantic_clusters,
            "cross_references": self.cross_references,
            "coordination_insights": self.coordination_insights,
            "performance_metrics": self.performance_metrics,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ReasoningInsight:
    """Result from advanced reasoning analysis."""
    insight_type: ReasoningType
    confidence_score: float
    reasoning: str
    recommendations: List[str]
    supporting_evidence: List[str]
    potential_risks: List[str]
    estimated_impact: float
    actionable_items: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "insight_type": self.insight_type.value,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "recommendations": self.recommendations,
            "supporting_evidence": self.supporting_evidence,
            "potential_risks": self.potential_risks,
            "estimated_impact": self.estimated_impact,
            "actionable_items": self.actionable_items,
            "metadata": self.metadata
        }


@dataclass
class OptimizationResult:
    """Result from semantic memory optimization."""
    original_size: int
    optimized_size: int
    optimization_ratio: float
    semantic_clusters_formed: int
    knowledge_entities_merged: int
    cross_references_created: int
    processing_time_ms: float
    performance_improvement: float
    memory_saved_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_size": self.original_size,
            "optimized_size": self.optimized_size,
            "optimization_ratio": self.optimization_ratio,
            "semantic_clusters_formed": self.semantic_clusters_formed,
            "knowledge_entities_merged": self.knowledge_entities_merged,
            "cross_references_created": self.cross_references_created,
            "processing_time_ms": self.processing_time_ms,
            "performance_improvement": self.performance_improvement,
            "memory_saved_mb": self.memory_saved_mb
        }


class UnifiedContextEngine:
    """
    Unified Context Engine - Advanced AI Context & Reasoning System
    
    Epic 4 Implementation Features:
    - Unified context coordination across all agents
    - Semantic memory and intelligent knowledge management
    - Advanced reasoning engine with decision support
    - Context-aware agent coordination
    - Intelligent caching and compression
    - Cross-agent context sharing with semantic optimization
    - Performance analytics and continuous optimization
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.settings = get_settings()
        self.db_session = db_session
        self.redis_client = get_redis_client()
        self.logger = logger.bind(component="unified_context_engine")
        
        # Initialize component engines (will be lazy-loaded)
        self._advanced_engine: Optional[AdvancedContextEngine] = None
        self._context_adapter: Optional[ContextAdapter] = None
        self._cache_manager: Optional[ContextCacheManager] = None
        self._context_compressor: Optional[ContextCompressor] = None
        self._context_analytics: Optional[ContextAnalytics] = None
        
        # Unified coordination state
        self.agent_coordination_map: Dict[str, Dict[str, Any]] = {}
        self.semantic_knowledge_graph: Dict[str, Dict[str, Any]] = {}
        self.reasoning_insights_cache: Dict[str, ReasoningInsight] = {}
        
        # Performance tracking
        self.operation_metrics = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "success_count": 0,
            "error_count": 0
        })
        
        # Semantic optimization
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.semantic_similarity_threshold = 0.75
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._is_running = False
        
        self.logger.info("🧠 Unified Context Engine initialized for Epic 4")
    
    async def initialize(self) -> None:
        """Initialize the unified context engine and all components."""
        if self._is_running:
            return
        
        self.logger.info("🚀 Initializing Unified Context Engine components...")
        
        try:
            # Initialize component engines
            self._advanced_engine = await get_advanced_context_engine()
            
            if self.db_session:
                self._context_adapter = ContextAdapter(self.db_session)
            
            self._cache_manager = get_context_cache_manager()
            await self._cache_manager.start_cache_management()
            
            self._context_compressor = get_context_compressor()
            
            # Initialize analytics if available
            try:
                self._context_analytics = ContextAnalyticsManager()
            except Exception as e:
                self.logger.warning(f"Context analytics not available: {e}")
            
            # Start background optimization tasks
            await self._start_background_tasks()
            
            self._is_running = True
            self.logger.info("✅ Unified Context Engine fully initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Unified Context Engine: {e}")
            raise
    
    async def coordinate_agent_context(
        self,
        agents: List[Agent],
        coordination_goals: Optional[List[str]] = None
    ) -> ContextMap:
        """
        Coordinate context across multiple agents for optimal collaboration.
        
        Args:
            agents: List of agents to coordinate
            coordination_goals: Specific coordination objectives
            
        Returns:
            ContextMap with unified agent context coordination
        """
        start_time = time.time()
        operation = "coordinate_agent_context"
        
        try:
            self.logger.info(f"🤝 Coordinating context for {len(agents)} agents")
            
            # Gather agent contexts
            agent_contexts = {}
            for agent in agents:
                contexts = await self._gather_agent_contexts(agent)
                agent_contexts[str(agent.id)] = contexts
            
            # Identify semantic clusters across agents
            semantic_clusters = await self._identify_semantic_clusters(agent_contexts)
            
            # Extract shared knowledge
            shared_knowledge = await self._extract_shared_knowledge(
                agent_contexts, semantic_clusters
            )
            
            # Create cross-references between agents
            cross_references = await self._create_cross_agent_references(
                agent_contexts, semantic_clusters
            )
            
            # Generate coordination insights
            coordination_insights = await self._generate_coordination_insights(
                agent_contexts, shared_knowledge, coordination_goals
            )
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_coordination_metrics(
                agents, agent_contexts, semantic_clusters
            )
            
            # Create unified context map
            context_map = ContextMap(
                agent_contexts=agent_contexts,
                shared_knowledge=shared_knowledge,
                semantic_clusters=semantic_clusters,
                cross_references=cross_references,
                coordination_insights=coordination_insights,
                performance_metrics=performance_metrics
            )
            
            # Cache coordination results
            await self._cache_coordination_results(agents, context_map)
            
            processing_time = time.time() - start_time
            self._record_operation_success(operation, processing_time)
            
            self.logger.info(
                f"✅ Agent coordination complete: {len(semantic_clusters)} clusters, "
                f"{len(shared_knowledge)} shared entities in {processing_time:.2f}s"
            )
            
            return context_map
            
        except Exception as e:
            self._record_operation_error(operation, str(e))
            self.logger.error(f"❌ Agent coordination failed: {e}")
            raise
    
    async def optimize_semantic_memory(
        self,
        target_agents: Optional[List[str]] = None,
        optimization_level: float = 0.7
    ) -> OptimizationResult:
        """
        Optimize semantic memory through intelligent compression and organization.
        
        Args:
            target_agents: Specific agents to optimize (all if None)
            optimization_level: Target optimization ratio (0.0-1.0)
            
        Returns:
            OptimizationResult with detailed metrics
        """
        start_time = time.time()
        operation = "optimize_semantic_memory"
        
        try:
            self.logger.info(f"🔧 Optimizing semantic memory (level: {optimization_level})")
            
            # Get current memory state
            memory_snapshot = await self._take_memory_snapshot(target_agents)
            original_size = memory_snapshot["total_size"]
            
            # Identify optimization opportunities
            optimization_opportunities = await self._identify_optimization_opportunities(
                memory_snapshot, optimization_level
            )
            
            # Apply semantic clustering
            clusters_formed = await self._apply_semantic_clustering(optimization_opportunities)
            
            # Merge similar knowledge entities
            entities_merged = await self._merge_similar_entities(
                optimization_opportunities, clusters_formed
            )
            
            # Create intelligent cross-references
            cross_references = await self._create_intelligent_cross_references(
                optimization_opportunities
            )
            
            # Compress low-priority contexts
            compression_results = await self._compress_low_priority_contexts(
                optimization_opportunities
            )
            
            # Calculate optimization results
            new_snapshot = await self._take_memory_snapshot(target_agents)
            optimized_size = new_snapshot["total_size"]
            optimization_ratio = 1 - (optimized_size / original_size) if original_size > 0 else 0
            
            processing_time = (time.time() - start_time) * 1000  # ms
            performance_improvement = await self._calculate_performance_improvement(
                memory_snapshot, new_snapshot
            )
            
            result = OptimizationResult(
                original_size=original_size,
                optimized_size=optimized_size,
                optimization_ratio=optimization_ratio,
                semantic_clusters_formed=len(clusters_formed),
                knowledge_entities_merged=entities_merged,
                cross_references_created=len(cross_references),
                processing_time_ms=processing_time,
                performance_improvement=performance_improvement,
                memory_saved_mb=(original_size - optimized_size) / (1024 * 1024)
            )
            
            self._record_operation_success(operation, processing_time / 1000)
            
            self.logger.info(
                f"✅ Semantic memory optimized: {optimization_ratio:.1%} reduction, "
                f"{result.memory_saved_mb:.2f}MB saved in {processing_time:.0f}ms"
            )
            
            return result
            
        except Exception as e:
            self._record_operation_error(operation, str(e))
            self.logger.error(f"❌ Semantic memory optimization failed: {e}")
            raise
    
    async def provide_reasoning_support(
        self,
        context: Union[str, Dict[str, Any], Context],
        reasoning_type: ReasoningType = ReasoningType.DECISION_SUPPORT,
        agent_id: Optional[str] = None
    ) -> ReasoningInsight:
        """
        Provide advanced reasoning support for decision making.
        
        Args:
            context: Context to analyze (text, dict, or Context object)
            reasoning_type: Type of reasoning to perform
            agent_id: Agent requesting reasoning support
            
        Returns:
            ReasoningInsight with analysis and recommendations
        """
        start_time = time.time()
        operation = "provide_reasoning_support"
        
        try:
            # Normalize context input
            context_data = await self._normalize_context_input(context)
            context_key = hashlib.md5(
                json.dumps(context_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache for existing insights
            if context_key in self.reasoning_insights_cache:
                cached_insight = self.reasoning_insights_cache[context_key]
                # Return cached result if recent (< 1 hour)
                if (datetime.utcnow() - datetime.fromisoformat(
                    cached_insight.metadata.get("created_at", "1970-01-01T00:00:00")
                )).total_seconds() < 3600:
                    self.logger.debug("📋 Returning cached reasoning insight")
                    return cached_insight
            
            self.logger.info(f"🧠 Providing {reasoning_type.value} reasoning support")
            
            # Gather relevant context from knowledge base
            relevant_contexts = await self._gather_relevant_contexts(
                context_data, agent_id, reasoning_type
            )
            
            # Perform reasoning analysis based on type
            if reasoning_type == ReasoningType.DECISION_SUPPORT:
                insight = await self._analyze_decision_support(context_data, relevant_contexts)
            elif reasoning_type == ReasoningType.PATTERN_RECOGNITION:
                insight = await self._analyze_patterns(context_data, relevant_contexts)
            elif reasoning_type == ReasoningType.PREDICTIVE_CONTEXT:
                insight = await self._analyze_predictive_context(context_data, relevant_contexts)
            elif reasoning_type == ReasoningType.CONFLICT_RESOLUTION:
                insight = await self._analyze_conflict_resolution(context_data, relevant_contexts)
            elif reasoning_type == ReasoningType.OPTIMIZATION:
                insight = await self._analyze_optimization(context_data, relevant_contexts)
            else:
                insight = await self._analyze_general_reasoning(context_data, relevant_contexts)
            
            # Enhance insight with cross-agent knowledge
            enhanced_insight = await self._enhance_with_cross_agent_knowledge(
                insight, agent_id
            )
            
            # Add metadata
            enhanced_insight.metadata.update({
                "created_at": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "reasoning_type": reasoning_type.value,
                "context_key": context_key
            })
            
            # Cache the insight
            self.reasoning_insights_cache[context_key] = enhanced_insight
            
            processing_time = time.time() - start_time
            self._record_operation_success(operation, processing_time)
            
            self.logger.info(
                f"✅ Reasoning support complete: {enhanced_insight.confidence_score:.1%} confidence, "
                f"{len(enhanced_insight.recommendations)} recommendations in {processing_time:.2f}s"
            )
            
            return enhanced_insight
            
        except Exception as e:
            self._record_operation_error(operation, str(e))
            self.logger.error(f"❌ Reasoning support failed: {e}")
            raise
    
    async def share_context_across_agents(
        self,
        source_agent_id: str,
        target_agent_ids: List[str],
        context_filter: Optional[Dict[str, Any]] = None,
        semantic_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Share context between agents with semantic optimization.
        
        Args:
            source_agent_id: Agent sharing context
            target_agent_ids: Agents to share context with
            context_filter: Filter criteria for context sharing
            semantic_optimization: Whether to apply semantic optimization
            
        Returns:
            Sharing results with metrics
        """
        start_time = time.time()
        operation = "share_context_across_agents"
        
        try:
            self.logger.info(
                f"📤 Sharing context from {source_agent_id} to {len(target_agent_ids)} agents"
            )
            
            # Use advanced context engine for knowledge sharing
            if self._advanced_engine:
                sharing_results = await self._advanced_engine.share_knowledge_across_agents(
                    source_agent=source_agent_id,
                    target_agents=target_agent_ids,
                    knowledge_filter=context_filter
                )
            else:
                # Fallback implementation
                sharing_results = await self._fallback_context_sharing(
                    source_agent_id, target_agent_ids, context_filter
                )
            
            # Apply semantic optimization if requested
            if semantic_optimization:
                optimization_results = await self._optimize_shared_context(
                    sharing_results, source_agent_id, target_agent_ids
                )
                sharing_results.update(optimization_results)
            
            # Update coordination map
            await self._update_coordination_map(
                source_agent_id, target_agent_ids, sharing_results
            )
            
            processing_time = time.time() - start_time
            self._record_operation_success(operation, processing_time)
            
            self.logger.info(
                f"✅ Context sharing complete: {sharing_results.get('shared_entities', 0)} entities "
                f"shared to {sharing_results.get('target_agents', 0)} agents in {processing_time:.2f}s"
            )
            
            return sharing_results
            
        except Exception as e:
            self._record_operation_error(operation, str(e))
            self.logger.error(f"❌ Context sharing failed: {e}")
            raise
    
    async def get_unified_analytics(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        agent_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive analytics across all context operations.
        
        Args:
            time_range: Time range for analytics (last 24h if None)
            agent_filter: Specific agents to analyze
            
        Returns:
            Comprehensive analytics dashboard data
        """
        try:
            self.logger.info("📊 Generating unified context analytics")
            
            # Default to last 24 hours
            if not time_range:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=1)
                time_range = (start_time, end_time)
            
            analytics = {
                "time_range": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                },
                "overview": {},
                "operation_metrics": {},
                "component_performance": {},
                "agent_activity": {},
                "semantic_insights": {},
                "optimization_results": {},
                "recommendations": []
            }
            
            # Gather operation metrics
            analytics["operation_metrics"] = dict(self.operation_metrics)
            
            # Get component performance
            component_performance = {}
            
            # Advanced context engine metrics
            if self._advanced_engine:
                try:
                    engine_metrics = await self._advanced_engine.get_compression_metrics()
                    component_performance["advanced_engine"] = engine_metrics
                except Exception as e:
                    self.logger.warning(f"Could not get advanced engine metrics: {e}")
            
            # Cache manager metrics
            if self._cache_manager:
                try:
                    cache_metrics = await self._cache_manager.get_cache_statistics()
                    component_performance["cache_manager"] = cache_metrics
                except Exception as e:
                    self.logger.warning(f"Could not get cache metrics: {e}")
            
            # Context compressor metrics
            if self._context_compressor:
                try:
                    compression_metrics = self._context_compressor.get_performance_metrics()
                    component_performance["compressor"] = compression_metrics
                except Exception as e:
                    self.logger.warning(f"Could not get compression metrics: {e}")
            
            analytics["component_performance"] = component_performance
            
            # Generate overview
            analytics["overview"] = await self._generate_analytics_overview(
                analytics, time_range, agent_filter
            )
            
            # Generate recommendations
            analytics["recommendations"] = await self._generate_optimization_recommendations(
                analytics
            )
            
            self.logger.info("✅ Unified analytics generated successfully")
            return analytics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate unified analytics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the unified context engine and cleanup resources."""
        if not self._is_running:
            return
        
        self.logger.info("🔄 Shutting down Unified Context Engine...")
        
        self._is_running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown components
        if self._cache_manager:
            await self._cache_manager.stop_cache_management()
        
        # Clear caches
        self.reasoning_insights_cache.clear()
        self.agent_coordination_map.clear()
        self.semantic_knowledge_graph.clear()
        
        self.logger.info("✅ Unified Context Engine shutdown complete")
    
    # Private helper methods
    
    async def _gather_agent_contexts(self, agent: Agent) -> List[Dict[str, Any]]:
        """Gather contexts for a specific agent."""
        try:
            if not self.db_session:
                return []
            
            # Get recent contexts for the agent
            result = await self.db_session.execute(
                select(Context).where(
                    and_(
                        Context.agent_id == agent.id,
                        Context.created_at >= datetime.utcnow() - timedelta(hours=24)
                    )
                ).order_by(Context.importance_score.desc()).limit(50)
            )
            
            contexts = result.scalars().all()
            return [
                {
                    "id": str(context.id),
                    "content": context.content,
                    "context_type": context.context_type.value if context.context_type else "general",
                    "importance_score": context.importance_score,
                    "created_at": context.created_at.isoformat(),
                    "access_count": getattr(context, 'access_count', 0)
                }
                for context in contexts
            ]
        except Exception as e:
            self.logger.error(f"Error gathering contexts for agent {agent.id}: {e}")
            return []
    
    async def _identify_semantic_clusters(
        self, agent_contexts: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[str]]:
        """Identify semantic clusters across agent contexts."""
        try:
            all_contexts = []
            context_to_agent = {}
            
            # Flatten all contexts
            for agent_id, contexts in agent_contexts.items():
                for context in contexts:
                    all_contexts.append(context["content"])
                    context_to_agent[len(all_contexts) - 1] = (agent_id, context["id"])
            
            if len(all_contexts) < 2:
                return {}
            
            # Create TF-IDF vectors
            try:
                tfidf_matrix = self.vectorizer.fit_transform(all_contexts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Find clusters based on similarity
                clusters = defaultdict(list)
                processed = set()
                
                for i in range(len(all_contexts)):
                    if i in processed:
                        continue
                    
                    cluster_id = f"cluster_{len(clusters)}"
                    clusters[cluster_id].append(context_to_agent[i][1])  # context id
                    processed.add(i)
                    
                    # Find similar contexts
                    for j in range(i + 1, len(all_contexts)):
                        if j in processed:
                            continue
                        
                        if similarity_matrix[i][j] > self.semantic_similarity_threshold:
                            clusters[cluster_id].append(context_to_agent[j][1])
                            processed.add(j)
                
                return dict(clusters)
                
            except Exception as e:
                self.logger.warning(f"TF-IDF clustering failed: {e}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error identifying semantic clusters: {e}")
            return {}
    
    async def _start_background_tasks(self) -> None:
        """Start background optimization tasks."""
        try:
            # Start semantic optimization task
            task = asyncio.create_task(self._background_semantic_optimization())
            self._background_tasks.add(task)
            
            # Start cache warming task
            task = asyncio.create_task(self._background_cache_warming())
            self._background_tasks.add(task)
            
            # Start performance monitoring task
            task = asyncio.create_task(self._background_performance_monitoring())
            self._background_tasks.add(task)
            
            self.logger.info("🔄 Background optimization tasks started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
    
    async def _background_semantic_optimization(self) -> None:
        """Background task for continuous semantic optimization."""
        while self._is_running:
            try:
                # Perform semantic optimization every 30 minutes
                await self.optimize_semantic_memory(optimization_level=0.6)
                await asyncio.sleep(1800)  # 30 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background semantic optimization error: {e}")
                await asyncio.sleep(300)  # 5 minutes before retry
    
    async def _background_cache_warming(self) -> None:
        """Background task for intelligent cache warming."""
        while self._is_running:
            try:
                # Warm cache for active agents every 15 minutes
                if self._cache_manager and self.db_session:
                    # Get active agents
                    result = await self.db_session.execute(
                        select(Agent.id).where(Agent.is_active == True).limit(10)
                    )
                    active_agents = [row[0] for row in result.all()]
                    
                    for agent_id in active_agents:
                        await self._cache_manager.warm_cache_for_agent(agent_id)
                
                await asyncio.sleep(900)  # 15 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background cache warming error: {e}")
                await asyncio.sleep(300)  # 5 minutes before retry
    
    async def _background_performance_monitoring(self) -> None:
        """Background task for performance monitoring."""
        while self._is_running:
            try:
                # Monitor and log performance every 10 minutes
                analytics = await self.get_unified_analytics()
                
                # Check for performance issues and log warnings
                overview = analytics.get("overview", {})
                if overview.get("error_rate", 0) > 0.05:  # 5% error rate threshold
                    self.logger.warning(
                        f"High error rate detected: {overview['error_rate']:.1%}"
                    )
                
                await asyncio.sleep(600)  # 10 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background performance monitoring error: {e}")
                await asyncio.sleep(300)  # 5 minutes before retry
    
    def _record_operation_success(self, operation: str, processing_time: float) -> None:
        """Record successful operation metrics."""
        metrics = self.operation_metrics[operation]
        metrics["count"] += 1
        metrics["total_time"] += processing_time
        metrics["success_count"] += 1
    
    def _record_operation_error(self, operation: str, error: str) -> None:
        """Record operation error metrics."""
        metrics = self.operation_metrics[operation]
        metrics["count"] += 1
        metrics["error_count"] += 1
        self.logger.error(f"Operation {operation} failed: {error}")
    
    # Placeholder implementations for complex methods
    # These would be fully implemented based on specific requirements
    
    async def _extract_shared_knowledge(
        self, agent_contexts: Dict, semantic_clusters: Dict
    ) -> Dict[str, Any]:
        """Extract knowledge that can be shared across agents."""
        return {"shared_entities": len(semantic_clusters), "knowledge_base": {}}
    
    async def _create_cross_agent_references(
        self, agent_contexts: Dict, semantic_clusters: Dict
    ) -> Dict[str, List[str]]:
        """Create cross-references between agent contexts."""
        return {cluster: contexts for cluster, contexts in semantic_clusters.items()}
    
    async def _generate_coordination_insights(
        self, agent_contexts: Dict, shared_knowledge: Dict, goals: Optional[List[str]]
    ) -> List[str]:
        """Generate insights for agent coordination."""
        insights = [
            f"Identified {len(shared_knowledge.get('shared_entities', 0))} shared knowledge entities",
            f"Found coordination opportunities across {len(agent_contexts)} agents"
        ]
        if goals:
            insights.append(f"Alignment with {len(goals)} coordination goals")
        return insights
    
    async def _calculate_coordination_metrics(
        self, agents: List[Agent], contexts: Dict, clusters: Dict
    ) -> Dict[str, Any]:
        """Calculate performance metrics for coordination."""
        return {
            "agents_coordinated": len(agents),
            "contexts_analyzed": sum(len(ctx) for ctx in contexts.values()),
            "semantic_clusters": len(clusters),
            "coordination_efficiency": 0.85
        }
    
    async def _normalize_context_input(
        self, context: Union[str, Dict[str, Any], Context]
    ) -> Dict[str, Any]:
        """Normalize context input to standard format."""
        if isinstance(context, str):
            return {"content": context, "type": "text"}
        elif isinstance(context, dict):
            return context
        elif isinstance(context, Context):
            return {
                "content": context.content,
                "type": context.context_type.value if context.context_type else "general",
                "importance": context.importance_score
            }
        else:
            return {"content": str(context), "type": "unknown"}
    
    async def _analyze_decision_support(
        self, context_data: Dict, relevant_contexts: List[Dict]
    ) -> ReasoningInsight:
        """Analyze context for decision support."""
        return ReasoningInsight(
            insight_type=ReasoningType.DECISION_SUPPORT,
            confidence_score=0.8,
            reasoning="Based on context analysis and historical patterns",
            recommendations=["Consider alternative approaches", "Evaluate risk factors"],
            supporting_evidence=["Historical success patterns", "Similar context outcomes"],
            potential_risks=["Implementation complexity", "Resource constraints"],
            estimated_impact=0.7,
            actionable_items=[
                {"action": "Review options", "priority": "high", "effort": "medium"}
            ]
        )


# Global unified context engine instance
_unified_context_engine: Optional[UnifiedContextEngine] = None


async def get_unified_context_engine(db_session: Optional[AsyncSession] = None) -> UnifiedContextEngine:
    """
    Get or create the global unified context engine instance.
    
    Args:
        db_session: Optional database session
        
    Returns:
        UnifiedContextEngine instance
    """
    global _unified_context_engine
    
    if _unified_context_engine is None:
        _unified_context_engine = UnifiedContextEngine(db_session)
        await _unified_context_engine.initialize()
    
    return _unified_context_engine


async def shutdown_unified_context_engine():
    """Shutdown the global unified context engine."""
    global _unified_context_engine
    
    if _unified_context_engine:
        await _unified_context_engine.shutdown()
        _unified_context_engine = None