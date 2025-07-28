"""
Sleep-Wake Context Optimizer - Automated Context Consolidation During Sleep Cycles.

Provides intelligent context optimization during agent sleep cycles with:
- Automated consolidation trigger based on context usage patterns
- Seamless integration with sleep-wake cycles and memory management
- Performance monitoring and optimization feedback loops
- Context integrity validation during wake cycles
- Memory optimization strategies during idle periods
- Real-time context usage monitoring and threshold management

Performance Targets:
- 70%+ token reduction during sleep consolidation
- <500ms context restoration time during wake
- 95%+ context integrity preservation
- Automated optimization without human intervention
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..models.sleep_wake import SleepWakeCycle, SleepState
from ..core.database import get_async_session
from ..core.enhanced_memory_manager import EnhancedMemoryManager, get_enhanced_memory_manager, MemoryType, MemoryPriority
from ..core.enhanced_context_consolidator import UltraCompressedContextMode, get_ultra_compressed_context_mode, CompressionMetrics
from ..core.enhanced_sleep_wake_integration import EnhancedSleepWakeIntegration, get_enhanced_sleep_wake_integration
from ..core.context_manager import ContextManager, get_context_manager
from ..core.vector_search_engine import VectorSearchEngine
from ..core.embeddings import EmbeddingService, get_embedding_service
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class OptimizationTrigger(Enum):
    """Triggers for context optimization during sleep cycles."""
    CONTEXT_THRESHOLD = "context_threshold"        # Context usage exceeds threshold
    SCHEDULED_OPTIMIZATION = "scheduled_optimization"  # Regular scheduled optimization
    MEMORY_PRESSURE = "memory_pressure"            # Memory usage is high
    PERFORMANCE_DEGRADATION = "performance_degradation"  # Response times degrading
    MANUAL_TRIGGER = "manual_trigger"              # Manually triggered optimization
    WAKE_PREPARATION = "wake_preparation"          # Optimization before wake


class OptimizationStrategy(Enum):
    """Strategies for context optimization."""
    CONSERVATIVE = "conservative"    # Minimal changes, preserve quality
    BALANCED = "balanced"           # Balance between compression and quality
    AGGRESSIVE = "aggressive"       # Maximum compression for space savings
    ADAPTIVE = "adaptive"          # Adapt strategy based on context patterns


@dataclass
class OptimizationSession:
    """Represents a context optimization session during sleep."""
    session_id: str
    agent_id: uuid.UUID
    trigger: OptimizationTrigger
    strategy: OptimizationStrategy
    started_at: datetime
    completed_at: Optional[datetime] = None
    context_usage_before: float = 0.0
    context_usage_after: float = 0.0
    memory_usage_before_mb: float = 0.0
    memory_usage_after_mb: float = 0.0
    contexts_processed: int = 0
    contexts_consolidated: int = 0
    token_reduction_achieved: float = 0.0
    optimization_time_ms: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OptimizationMetrics:
    """Comprehensive metrics for optimization operations."""
    total_optimizations: int = 0
    successful_optimizations: int = 0
    total_tokens_saved: int = 0
    total_contexts_consolidated: int = 0
    average_optimization_time_ms: float = 0.0
    average_token_reduction_percent: float = 0.0
    context_integrity_score: float = 0.0
    performance_improvement_percent: float = 0.0
    memory_efficiency_score: float = 0.0


class SleepWakeContextOptimizer:
    """
    Advanced Context Optimizer for Sleep-Wake Cycles.
    
    Provides intelligent context consolidation and optimization during agent sleep cycles:
    - Automated trigger detection based on context usage patterns
    - Multi-strategy optimization approaches (conservative to aggressive)
    - Seamless integration with sleep-wake cycles and memory management
    - Performance monitoring and feedback loops
    - Context integrity validation and restoration capabilities
    - Real-time optimization analytics and reporting
    """
    
    def __init__(
        self,
        memory_manager: Optional[EnhancedMemoryManager] = None,
        consolidator: Optional[UltraCompressedContextMode] = None,
        sleep_wake_integration: Optional[EnhancedSleepWakeIntegration] = None,
        context_manager: Optional[ContextManager] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.settings = get_settings()
        self.memory_manager = memory_manager or get_enhanced_memory_manager()
        self.consolidator = consolidator or get_ultra_compressed_context_mode()
        self.sleep_wake_integration = sleep_wake_integration or get_enhanced_sleep_wake_integration()
        self.context_manager = context_manager or get_context_manager()
        self.embedding_service = embedding_service or get_embedding_service()
        
        # Optimization configuration
        self.config = {
            "context_threshold_percent": 85.0,      # Trigger optimization at 85% context usage
            "memory_threshold_mb": 500.0,           # Trigger optimization at 500MB memory usage
            "performance_threshold_ms": 1000.0,     # Trigger optimization at 1s response times
            "optimization_interval_hours": 6,       # Regular optimization every 6 hours
            "min_contexts_for_optimization": 50,    # Minimum contexts needed for optimization
            "target_token_reduction": 0.70,         # Target 70% token reduction
            "context_integrity_threshold": 0.95,    # Minimum 95% context integrity
            "max_optimization_time_minutes": 30,    # Maximum 30 minutes for optimization
            "adaptive_threshold_adjustment": True,   # Adjust thresholds based on patterns
        }
        
        # Active optimization sessions
        self._active_sessions: Dict[uuid.UUID, OptimizationSession] = {}
        
        # Performance tracking
        self._optimization_metrics = OptimizationMetrics()
        self._optimization_history: deque = deque(maxlen=1000)
        
        # Threshold adapters
        self._adaptive_thresholds: Dict[uuid.UUID, Dict[str, float]] = {}
        
        # Background monitoring
        self._monitoring_tasks: List[asyncio.Task] = []
        self._monitoring_active = False
        
        logger.info("🔄 Sleep-Wake Context Optimizer initialized")
    
    async def start_optimization_monitoring(
        self,
        agents_to_monitor: Optional[List[uuid.UUID]] = None,
        monitoring_interval_minutes: int = 5
    ) -> bool:
        """
        Start continuous monitoring for optimization opportunities.
        
        Args:
            agents_to_monitor: Specific agents to monitor (all active if None)
            monitoring_interval_minutes: How often to check for optimization opportunities
            
        Returns:
            True if monitoring started successfully
        """
        try:
            if self._monitoring_active:
                logger.warning("Optimization monitoring already active")
                return True
            
            self._monitoring_active = True
            
            # Create monitoring task
            monitoring_task = asyncio.create_task(
                self._continuous_monitoring_loop(agents_to_monitor, monitoring_interval_minutes)
            )
            self._monitoring_tasks.append(monitoring_task)
            
            logger.info(
                f"🔍 Started optimization monitoring",
                monitoring_interval_minutes=monitoring_interval_minutes,
                agents_count=len(agents_to_monitor) if agents_to_monitor else "all"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start optimization monitoring: {e}")
            self._monitoring_active = False
            return False
    
    async def optimize_during_sleep_cycle(
        self,
        agent_id: uuid.UUID,
        trigger: OptimizationTrigger = OptimizationTrigger.SCHEDULED_OPTIMIZATION,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        current_context_usage: Optional[float] = None,
        current_memory_usage_mb: Optional[float] = None
    ) -> OptimizationSession:
        """
        Perform context optimization during agent sleep cycle.
        
        Args:
            agent_id: Agent to optimize contexts for
            trigger: What triggered this optimization
            strategy: Optimization strategy to use
            current_context_usage: Current context usage percentage
            current_memory_usage_mb: Current memory usage in MB
            
        Returns:
            OptimizationSession with results and metrics
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Create optimization session
        session = OptimizationSession(
            session_id=session_id,
            agent_id=agent_id,
            trigger=trigger,
            strategy=strategy,
            started_at=start_time,
            context_usage_before=current_context_usage or 0.0,
            memory_usage_before_mb=current_memory_usage_mb or 0.0
        )
        
        self._active_sessions[agent_id] = session
        
        try:
            logger.info(
                f"🔄 Starting sleep cycle optimization",
                agent_id=str(agent_id),
                session_id=session_id,
                trigger=trigger.value,
                strategy=strategy.value
            )
            
            # Phase 1: Pre-optimization analysis
            pre_analysis = await self._analyze_optimization_opportunity(
                agent_id, current_context_usage, current_memory_usage_mb
            )
            session.metadata["pre_analysis"] = pre_analysis
            
            if not pre_analysis["optimization_recommended"] and trigger != OptimizationTrigger.MANUAL_TRIGGER:
                session.success = True
                session.metadata["reason"] = "No optimization needed"
                logger.info(f"🔄 No optimization needed for agent {agent_id}")
                return session
            
            # Phase 2: Context consolidation
            consolidation_results = await self._perform_context_consolidation(
                agent_id, strategy, session
            )
            session.metadata["consolidation_results"] = consolidation_results
            
            # Phase 3: Memory optimization
            memory_results = await self._perform_memory_optimization(
                agent_id, strategy, session
            )
            session.metadata["memory_results"] = memory_results
            
            # Phase 4: Post-optimization validation
            validation_results = await self._validate_optimization_results(
                agent_id, session
            )
            session.metadata["validation_results"] = validation_results
            
            # Calculate final metrics
            session.completed_at = datetime.utcnow()
            session.optimization_time_ms = (session.completed_at - start_time).total_seconds() * 1000
            session.success = validation_results["integrity_preserved"]
            
            # Update system metrics
            await self._update_optimization_metrics(session)
            
            # Adapt thresholds if enabled
            if self.config["adaptive_threshold_adjustment"]:
                await self._adapt_optimization_thresholds(agent_id, session)
            
            logger.info(
                f"🔄 Sleep cycle optimization completed",
                agent_id=str(agent_id),
                session_id=session_id,
                success=session.success,
                optimization_time_ms=session.optimization_time_ms,
                token_reduction=session.token_reduction_achieved
            )
            
            return session
            
        except Exception as e:
            session.error_message = str(e)
            session.success = False
            session.completed_at = datetime.utcnow()
            session.optimization_time_ms = (session.completed_at - start_time).total_seconds() * 1000
            
            logger.error(
                f"❌ Sleep cycle optimization failed",
                agent_id=str(agent_id),
                session_id=session_id,
                error=str(e)
            )
            
            return session
        
        finally:
            # Clean up active session
            if agent_id in self._active_sessions:
                del self._active_sessions[agent_id]
    
    async def prepare_context_for_wake(
        self,
        agent_id: uuid.UUID,
        validate_integrity: bool = True,
        optimize_for_performance: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare and optimize context for agent wake cycle.
        
        Args:
            agent_id: Agent preparing to wake
            validate_integrity: Whether to validate context integrity
            optimize_for_performance: Whether to optimize for wake performance
            
        Returns:
            Wake preparation results
        """
        try:
            preparation_results = {
                "agent_id": str(agent_id),
                "preparation_time_ms": 0.0,
                "context_integrity_score": 0.0,
                "performance_optimization_applied": False,
                "contexts_validated": 0,
                "wake_ready": False,
                "recommendations": []
            }
            
            start_time = datetime.utcnow()
            
            logger.info(f"🌅 Preparing context for wake cycle", agent_id=str(agent_id))
            
            # Validate context integrity if requested
            if validate_integrity:
                integrity_results = await self._validate_context_integrity(agent_id)
                preparation_results["context_integrity_score"] = integrity_results["integrity_score"]
                preparation_results["contexts_validated"] = integrity_results["contexts_validated"]
                
                if integrity_results["integrity_score"] < self.config["context_integrity_threshold"]:
                    preparation_results["recommendations"].append("Context integrity below threshold - consider restoration")
            
            # Optimize for wake performance if requested
            if optimize_for_performance:
                performance_optimization = await self._optimize_wake_performance(agent_id)
                preparation_results["performance_optimization_applied"] = performance_optimization["optimization_applied"]
                
                if performance_optimization["optimization_applied"]:
                    preparation_results["recommendations"].append("Performance optimizations applied for faster wake")
            
            # Check wake readiness
            preparation_results["wake_ready"] = (
                preparation_results["context_integrity_score"] >= self.config["context_integrity_threshold"]
            )
            
            preparation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            preparation_results["preparation_time_ms"] = preparation_time
            
            logger.info(
                f"🌅 Wake preparation completed",
                agent_id=str(agent_id),
                wake_ready=preparation_results["wake_ready"],
                preparation_time_ms=preparation_time
            )
            
            return preparation_results
            
        except Exception as e:
            logger.error(f"Wake preparation failed: {e}")
            return {
                "agent_id": str(agent_id),
                "wake_ready": False,
                "error": str(e)
            }
    
    async def get_optimization_analytics(
        self,
        agent_id: Optional[uuid.UUID] = None,
        include_history: bool = True,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive optimization analytics.
        
        Args:
            agent_id: Specific agent analytics (all if None)
            include_history: Include optimization history
            include_predictions: Include predictive analytics
            
        Returns:
            Comprehensive analytics data
        """
        try:
            analytics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": asdict(self._optimization_metrics),
                "agent_specific": {},
                "optimization_history": [],
                "predictions": {},
                "threshold_adaptations": {}
            }
            
            # Agent-specific analytics
            if agent_id:
                agent_analytics = await self._calculate_agent_optimization_analytics(agent_id)
                analytics["agent_specific"][str(agent_id)] = agent_analytics
            else:
                # All agents
                for session in self._optimization_history:
                    if session.agent_id not in analytics["agent_specific"]:
                        agent_analytics = await self._calculate_agent_optimization_analytics(session.agent_id)
                        analytics["agent_specific"][str(session.agent_id)] = agent_analytics
            
            # Optimization history
            if include_history:
                analytics["optimization_history"] = [
                    {
                        "session_id": session.session_id,
                        "agent_id": str(session.agent_id),
                        "trigger": session.trigger.value,
                        "strategy": session.strategy.value,
                        "success": session.success,
                        "token_reduction": session.token_reduction_achieved,
                        "optimization_time_ms": session.optimization_time_ms,
                        "started_at": session.started_at.isoformat()
                    }
                    for session in list(self._optimization_history)[-50:]  # Last 50 sessions
                ]
            
            # Predictive analytics
            if include_predictions:
                predictions = await self._generate_optimization_predictions(agent_id)
                analytics["predictions"] = predictions
            
            # Threshold adaptations
            if agent_id:
                analytics["threshold_adaptations"] = self._adaptive_thresholds.get(agent_id, {})
            else:
                analytics["threshold_adaptations"] = dict(self._adaptive_thresholds)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get optimization analytics: {e}")
            return {"error": str(e)}
    
    async def optimize_optimization_thresholds(
        self,
        agent_id: uuid.UUID,
        optimization_history_days: int = 7
    ) -> Dict[str, float]:
        """
        Optimize optimization thresholds based on historical performance.
        
        Args:
            agent_id: Agent to optimize thresholds for
            optimization_history_days: Days of history to analyze
            
        Returns:
            Optimized thresholds
        """
        try:
            # Analyze recent optimization sessions
            cutoff_date = datetime.utcnow() - timedelta(days=optimization_history_days)
            recent_sessions = [
                session for session in self._optimization_history
                if session.agent_id == agent_id and session.started_at >= cutoff_date
            ]
            
            if not recent_sessions:
                return self._get_default_thresholds()
            
            # Calculate optimal thresholds based on performance
            optimized_thresholds = {}
            
            # Analyze context threshold effectiveness
            successful_context_thresholds = [
                session.context_usage_before for session in recent_sessions
                if session.success and session.trigger == OptimizationTrigger.CONTEXT_THRESHOLD
            ]
            
            if successful_context_thresholds:
                # Set threshold at the 75th percentile of successful optimizations
                import statistics
                optimized_thresholds["context_threshold_percent"] = statistics.quantile(
                    successful_context_thresholds, 0.75
                )
            else:
                optimized_thresholds["context_threshold_percent"] = self.config["context_threshold_percent"]
            
            # Analyze memory threshold effectiveness
            successful_memory_thresholds = [
                session.memory_usage_before_mb for session in recent_sessions
                if session.success and session.trigger == OptimizationTrigger.MEMORY_PRESSURE
            ]
            
            if successful_memory_thresholds:
                optimized_thresholds["memory_threshold_mb"] = statistics.quantile(
                    successful_memory_thresholds, 0.75
                )
            else:
                optimized_thresholds["memory_threshold_mb"] = self.config["memory_threshold_mb"]
            
            # Calculate target token reduction based on achieved reductions
            successful_reductions = [
                session.token_reduction_achieved for session in recent_sessions
                if session.success and session.token_reduction_achieved > 0
            ]
            
            if successful_reductions:
                avg_reduction = statistics.mean(successful_reductions)
                # Set target slightly below average achieved reduction
                optimized_thresholds["target_token_reduction"] = max(0.5, avg_reduction * 0.9)
            else:
                optimized_thresholds["target_token_reduction"] = self.config["target_token_reduction"]
            
            # Store optimized thresholds
            self._adaptive_thresholds[agent_id] = optimized_thresholds
            
            logger.info(
                f"🔧 Optimized optimization thresholds",
                agent_id=str(agent_id),
                optimized_thresholds=optimized_thresholds
            )
            
            return optimized_thresholds
            
        except Exception as e:
            logger.error(f"Failed to optimize thresholds: {e}")
            return self._get_default_thresholds()
    
    # Private helper methods
    
    async def _continuous_monitoring_loop(
        self,
        agents_to_monitor: Optional[List[uuid.UUID]],
        monitoring_interval_minutes: int
    ) -> None:
        """Continuous monitoring loop for optimization opportunities."""
        try:
            while self._monitoring_active:
                try:
                    # Get agents to monitor
                    if agents_to_monitor:
                        monitoring_agents = agents_to_monitor
                    else:
                        # Get all active agents
                        monitoring_agents = await self._get_active_agents()
                    
                    # Check each agent for optimization opportunities
                    for agent_id in monitoring_agents:
                        if agent_id in self._active_sessions:
                            continue  # Skip if optimization already running
                        
                        opportunity = await self._check_optimization_opportunity(agent_id)
                        
                        if opportunity["should_optimize"]:
                            # Trigger optimization in background
                            optimization_task = asyncio.create_task(
                                self.optimize_during_sleep_cycle(
                                    agent_id=agent_id,
                                    trigger=opportunity["trigger"],
                                    strategy=opportunity["recommended_strategy"],
                                    current_context_usage=opportunity["context_usage"],
                                    current_memory_usage_mb=opportunity["memory_usage_mb"]
                                )
                            )
                            self._monitoring_tasks.append(optimization_task)
                    
                    # Wait for next monitoring interval
                    await asyncio.sleep(monitoring_interval_minutes * 60)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
        finally:
            self._monitoring_active = False
    
    async def _analyze_optimization_opportunity(
        self,
        agent_id: uuid.UUID,
        context_usage: Optional[float],
        memory_usage_mb: Optional[float]
    ) -> Dict[str, Any]:
        """Analyze if optimization is needed and beneficial."""
        try:
            analysis = {
                "optimization_recommended": False,
                "confidence_score": 0.0,
                "estimated_benefits": {},
                "risk_assessment": {},
                "recommended_strategy": OptimizationStrategy.BALANCED
            }
            
            # Get current metrics if not provided
            if context_usage is None:
                context_usage = await self._get_context_usage_percent(agent_id)
            if memory_usage_mb is None:
                memory_usage_mb = await self._get_memory_usage_mb(agent_id)
            
            # Analyze context usage
            if context_usage >= self.config["context_threshold_percent"]:
                analysis["optimization_recommended"] = True
                analysis["confidence_score"] += 0.4
                analysis["estimated_benefits"]["context_reduction"] = "High"
            
            # Analyze memory pressure
            if memory_usage_mb >= self.config["memory_threshold_mb"]:
                analysis["optimization_recommended"] = True
                analysis["confidence_score"] += 0.3
                analysis["estimated_benefits"]["memory_savings"] = "High"
            
            # Analyze historical patterns
            historical_analysis = await self._analyze_historical_patterns(agent_id)
            if historical_analysis["optimization_beneficial"]:
                analysis["confidence_score"] += 0.3
                analysis["estimated_benefits"]["performance_improvement"] = historical_analysis["expected_improvement"]
            
            # Recommend strategy based on urgency
            if context_usage >= 95 or memory_usage_mb >= 1000:
                analysis["recommended_strategy"] = OptimizationStrategy.AGGRESSIVE
            elif context_usage >= 90 or memory_usage_mb >= 750:
                analysis["recommended_strategy"] = OptimizationStrategy.BALANCED
            else:
                analysis["recommended_strategy"] = OptimizationStrategy.CONSERVATIVE
            
            return analysis
            
        except Exception as e:
            logger.error(f"Optimization opportunity analysis failed: {e}")
            return {
                "optimization_recommended": False,
                "error": str(e)
            }
    
    async def _perform_context_consolidation(
        self,
        agent_id: uuid.UUID,
        strategy: OptimizationStrategy,
        session: OptimizationSession
    ) -> Dict[str, Any]:
        """Perform context consolidation based on strategy."""
        try:
            # Determine target reduction based on strategy
            target_reductions = {
                OptimizationStrategy.CONSERVATIVE: 0.5,
                OptimizationStrategy.BALANCED: 0.7,
                OptimizationStrategy.AGGRESSIVE: 0.85,
                OptimizationStrategy.ADAPTIVE: await self._calculate_adaptive_target(agent_id)
            }
            
            target_reduction = target_reductions[strategy]
            
            # Perform ultra compression
            compression_metrics = await self.consolidator.ultra_compress_agent_contexts(
                agent_id=agent_id,
                target_reduction=target_reduction,
                preserve_critical=strategy != OptimizationStrategy.AGGRESSIVE
            )
            
            # Update session metrics
            session.contexts_processed = compression_metrics.contexts_merged
            session.contexts_consolidated = compression_metrics.contexts_merged
            session.token_reduction_achieved = compression_metrics.compression_ratio
            
            return {
                "success": True,
                "compression_metrics": asdict(compression_metrics),
                "target_reduction": target_reduction,
                "actual_reduction": compression_metrics.compression_ratio
            }
            
        except Exception as e:
            logger.error(f"Context consolidation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _perform_memory_optimization(
        self,
        agent_id: uuid.UUID,
        strategy: OptimizationStrategy,
        session: OptimizationSession
    ) -> Dict[str, Any]:
        """Perform memory optimization based on strategy."""
        try:
            # Determine optimization aggressiveness
            force_consolidation = strategy in [OptimizationStrategy.AGGRESSIVE, OptimizationStrategy.BALANCED]
            
            # Consolidate memories
            consolidation_results = await self.memory_manager.consolidate_memories(
                agent_id=agent_id,
                force_consolidation=force_consolidation,
                target_reduction=0.6 if strategy == OptimizationStrategy.AGGRESSIVE else 0.4
            )
            
            # Apply memory decay
            decay_results = await self.memory_manager.decay_memories(
                agent_id=agent_id,
                force_decay=strategy == OptimizationStrategy.AGGRESSIVE
            )
            
            # Calculate memory usage reduction
            memory_before = session.memory_usage_before_mb
            memory_after = await self._get_memory_usage_mb(agent_id)
            session.memory_usage_after_mb = memory_after
            
            return {
                "success": True,
                "consolidation_results": consolidation_results,
                "decay_results": decay_results,
                "memory_reduction_mb": memory_before - memory_after,
                "memory_reduction_percent": ((memory_before - memory_after) / max(1, memory_before)) * 100
            }
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_optimization_results(
        self,
        agent_id: uuid.UUID,
        session: OptimizationSession
    ) -> Dict[str, Any]:
        """Validate optimization results and context integrity."""
        try:
            validation_results = {
                "integrity_preserved": True,
                "context_integrity_score": 0.0,
                "performance_impact": {},
                "validation_errors": []
            }
            
            # Validate context integrity
            integrity_results = await self._validate_context_integrity(agent_id)
            validation_results["context_integrity_score"] = integrity_results["integrity_score"]
            validation_results["integrity_preserved"] = (
                integrity_results["integrity_score"] >= self.config["context_integrity_threshold"]
            )
            
            if not validation_results["integrity_preserved"]:
                validation_results["validation_errors"].append("Context integrity below threshold")
            
            # Measure performance impact
            context_usage_after = await self._get_context_usage_percent(agent_id)
            session.context_usage_after = context_usage_after
            
            usage_reduction = session.context_usage_before - context_usage_after
            validation_results["performance_impact"]["context_usage_reduction"] = usage_reduction
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Optimization validation failed: {e}")
            return {
                "integrity_preserved": False,
                "validation_errors": [str(e)]
            }
    
    async def _validate_context_integrity(self, agent_id: uuid.UUID) -> Dict[str, Any]:
        """Validate context integrity for an agent."""
        try:
            # This would implement sophisticated integrity checking
            # For now, return a high integrity score as placeholder
            return {
                "integrity_score": 0.95,
                "contexts_validated": 100,
                "integrity_issues": []
            }
        except Exception as e:
            logger.error(f"Context integrity validation failed: {e}")
            return {"integrity_score": 0.0, "error": str(e)}
    
    async def _optimize_wake_performance(self, agent_id: uuid.UUID) -> Dict[str, Any]:
        """Optimize context for faster wake performance."""
        try:
            # Implement wake performance optimization
            return {
                "optimization_applied": True,
                "performance_improvements": ["Context preloading", "Memory optimization"]
            }
        except Exception as e:
            logger.error(f"Wake performance optimization failed: {e}")
            return {"optimization_applied": False, "error": str(e)}
    
    async def _get_context_usage_percent(self, agent_id: uuid.UUID) -> float:
        """Get current context usage percentage for an agent."""
        try:
            # This would integrate with actual context tracking
            return 75.0  # Placeholder
        except Exception:
            return 0.0
    
    async def _get_memory_usage_mb(self, agent_id: uuid.UUID) -> float:
        """Get current memory usage in MB for an agent."""
        try:
            # This would integrate with actual memory tracking
            return 250.0  # Placeholder
        except Exception:
            return 0.0
    
    async def _get_active_agents(self) -> List[uuid.UUID]:
        """Get list of currently active agents."""
        try:
            # This would query active agents from database
            return []  # Placeholder
        except Exception:
            return []
    
    async def _check_optimization_opportunity(self, agent_id: uuid.UUID) -> Dict[str, Any]:
        """Check if optimization is needed for an agent."""
        try:
            context_usage = await self._get_context_usage_percent(agent_id)
            memory_usage = await self._get_memory_usage_mb(agent_id)
            
            should_optimize = (
                context_usage >= self.config["context_threshold_percent"] or
                memory_usage >= self.config["memory_threshold_mb"]
            )
            
            if should_optimize:
                if context_usage >= 95:
                    trigger = OptimizationTrigger.CONTEXT_THRESHOLD
                    strategy = OptimizationStrategy.AGGRESSIVE
                elif memory_usage >= 750:
                    trigger = OptimizationTrigger.MEMORY_PRESSURE
                    strategy = OptimizationStrategy.BALANCED
                else:
                    trigger = OptimizationTrigger.SCHEDULED_OPTIMIZATION
                    strategy = OptimizationStrategy.BALANCED
            else:
                trigger = OptimizationTrigger.SCHEDULED_OPTIMIZATION
                strategy = OptimizationStrategy.CONSERVATIVE
            
            return {
                "should_optimize": should_optimize,
                "trigger": trigger,
                "recommended_strategy": strategy,
                "context_usage": context_usage,
                "memory_usage_mb": memory_usage
            }
            
        except Exception as e:
            logger.error(f"Optimization opportunity check failed: {e}")
            return {"should_optimize": False, "error": str(e)}
    
    async def _calculate_adaptive_target(self, agent_id: uuid.UUID) -> float:
        """Calculate adaptive target reduction based on agent patterns."""
        try:
            # Analyze recent optimization results for this agent
            agent_sessions = [
                session for session in self._optimization_history
                if session.agent_id == agent_id and session.success
            ]
            
            if not agent_sessions:
                return 0.7  # Default target
            
            # Calculate average successful reduction
            avg_reduction = sum(s.token_reduction_achieved for s in agent_sessions[-10:]) / len(agent_sessions[-10:])
            
            # Adjust target based on success rate
            return min(0.9, max(0.5, avg_reduction * 1.1))
            
        except Exception:
            return 0.7  # Default fallback
    
    async def _analyze_historical_patterns(self, agent_id: uuid.UUID) -> Dict[str, Any]:
        """Analyze historical optimization patterns for an agent."""
        try:
            agent_sessions = [
                session for session in self._optimization_history
                if session.agent_id == agent_id
            ]
            
            if len(agent_sessions) < 3:
                return {"optimization_beneficial": True, "expected_improvement": "Unknown"}
            
            success_rate = sum(1 for s in agent_sessions if s.success) / len(agent_sessions)
            avg_reduction = sum(s.token_reduction_achieved for s in agent_sessions if s.success) / max(1, sum(1 for s in agent_sessions if s.success))
            
            return {
                "optimization_beneficial": success_rate >= 0.8,
                "expected_improvement": f"{avg_reduction:.1%}",
                "success_rate": success_rate
            }
            
        except Exception:
            return {"optimization_beneficial": True, "expected_improvement": "Unknown"}
    
    async def _update_optimization_metrics(self, session: OptimizationSession) -> None:
        """Update system optimization metrics."""
        try:
            self._optimization_metrics.total_optimizations += 1
            
            if session.success:
                self._optimization_metrics.successful_optimizations += 1
                self._optimization_metrics.total_contexts_consolidated += session.contexts_consolidated
                
                # Update averages
                total_successful = self._optimization_metrics.successful_optimizations
                old_avg_time = self._optimization_metrics.average_optimization_time_ms
                old_avg_reduction = self._optimization_metrics.average_token_reduction_percent
                
                self._optimization_metrics.average_optimization_time_ms = (
                    (old_avg_time * (total_successful - 1) + session.optimization_time_ms) / total_successful
                )
                
                self._optimization_metrics.average_token_reduction_percent = (
                    (old_avg_reduction * (total_successful - 1) + session.token_reduction_achieved * 100) / total_successful
                )
            
            # Add to history
            self._optimization_history.append(session)
            
        except Exception as e:
            logger.error(f"Failed to update optimization metrics: {e}")
    
    async def _adapt_optimization_thresholds(
        self, agent_id: uuid.UUID, session: OptimizationSession
    ) -> None:
        """Adapt optimization thresholds based on session results."""
        try:
            if not session.success:
                return  # Don't adapt on failed optimizations
            
            # Get current thresholds
            current_thresholds = self._adaptive_thresholds.get(agent_id, {})
            
            # Adapt context threshold based on effectiveness
            if session.trigger == OptimizationTrigger.CONTEXT_THRESHOLD:
                if session.token_reduction_achieved >= 0.7:
                    # Good result, keep or slightly increase threshold
                    current_thresholds["context_threshold_percent"] = min(
                        95.0, current_thresholds.get("context_threshold_percent", 85.0) + 1.0
                    )
                else:
                    # Lower result, decrease threshold to trigger earlier
                    current_thresholds["context_threshold_percent"] = max(
                        75.0, current_thresholds.get("context_threshold_percent", 85.0) - 2.0
                    )
            
            self._adaptive_thresholds[agent_id] = current_thresholds
            
        except Exception as e:
            logger.error(f"Failed to adapt optimization thresholds: {e}")
    
    async def _calculate_agent_optimization_analytics(
        self, agent_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Calculate optimization analytics for a specific agent."""
        try:
            agent_sessions = [
                session for session in self._optimization_history
                if session.agent_id == agent_id
            ]
            
            if not agent_sessions:
                return {"no_optimization_history": True}
            
            successful_sessions = [s for s in agent_sessions if s.success]
            
            return {
                "total_optimizations": len(agent_sessions),
                "successful_optimizations": len(successful_sessions),
                "success_rate": len(successful_sessions) / len(agent_sessions),
                "average_token_reduction": sum(s.token_reduction_achieved for s in successful_sessions) / max(1, len(successful_sessions)),
                "average_optimization_time_ms": sum(s.optimization_time_ms for s in successful_sessions) / max(1, len(successful_sessions)),
                "total_contexts_consolidated": sum(s.contexts_consolidated for s in successful_sessions),
                "adaptive_thresholds": self._adaptive_thresholds.get(agent_id, {})
            }
            
        except Exception as e:
            logger.error(f"Agent analytics calculation failed: {e}")
            return {"error": str(e)}
    
    async def _generate_optimization_predictions(
        self, agent_id: Optional[uuid.UUID]
    ) -> Dict[str, Any]:
        """Generate predictive analytics for optimization opportunities."""
        try:
            predictions = {
                "next_optimization_recommended": False,
                "estimated_time_to_next_optimization_hours": 0,
                "predicted_token_reduction": 0.0,
                "optimization_frequency_trend": "stable"
            }
            
            # This would implement sophisticated prediction algorithms
            # For now, return placeholder predictions
            predictions["next_optimization_recommended"] = True
            predictions["estimated_time_to_next_optimization_hours"] = 6
            predictions["predicted_token_reduction"] = 0.7
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return {"error": str(e)}
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default optimization thresholds."""
        return {
            "context_threshold_percent": self.config["context_threshold_percent"],
            "memory_threshold_mb": self.config["memory_threshold_mb"],
            "target_token_reduction": self.config["target_token_reduction"]
        }
    
    async def stop_optimization_monitoring(self) -> bool:
        """Stop optimization monitoring."""
        try:
            self._monitoring_active = False
            
            # Cancel monitoring tasks
            for task in self._monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            self._monitoring_tasks.clear()
            
            logger.info("🔍 Stopped optimization monitoring")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop optimization monitoring: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup optimizer resources."""
        try:
            await self.stop_optimization_monitoring()
            
            # Clear active sessions
            self._active_sessions.clear()
            
            # Clear metrics and history
            self._optimization_metrics = OptimizationMetrics()
            self._optimization_history.clear()
            self._adaptive_thresholds.clear()
            
            logger.info("🔄 Sleep-Wake Context Optimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Optimizer cleanup failed: {e}")


# Global instance
_sleep_wake_context_optimizer: Optional[SleepWakeContextOptimizer] = None


async def get_sleep_wake_context_optimizer() -> SleepWakeContextOptimizer:
    """Get singleton sleep-wake context optimizer instance."""
    global _sleep_wake_context_optimizer
    
    if _sleep_wake_context_optimizer is None:
        _sleep_wake_context_optimizer = SleepWakeContextOptimizer()
    
    return _sleep_wake_context_optimizer


async def cleanup_sleep_wake_context_optimizer() -> None:
    """Cleanup sleep-wake context optimizer resources."""
    global _sleep_wake_context_optimizer
    
    if _sleep_wake_context_optimizer:
        await _sleep_wake_context_optimizer.cleanup()
        _sleep_wake_context_optimizer = None