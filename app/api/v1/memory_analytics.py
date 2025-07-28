"""
Memory Analytics API - Dashboard Endpoints and Monitoring for Context & Memory Systems.

Provides comprehensive API endpoints for monitoring and analytics of:
- Enhanced Memory Manager performance and usage
- Sleep-Wake Context Optimizer analytics and metrics
- Memory-Aware Vector Search performance
- Memory Consolidation Service statistics
- Semantic Integrity Validator results
- Real-time monitoring and alerting capabilities

Performance Targets:
- <100ms response time for dashboard endpoints
- Real-time streaming of memory analytics
- Comprehensive system health monitoring
- Detailed performance insights and trends
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_session
from ...core.enhanced_memory_manager import get_enhanced_memory_manager, MemoryType, MemoryPriority
from ...core.sleep_wake_context_optimizer import get_sleep_wake_context_optimizer, OptimizationTrigger, OptimizationStrategy
from ...core.memory_aware_vector_search import get_memory_aware_vector_search, SearchStrategy
from ...core.memory_consolidation_service import get_memory_consolidation_service, ConsolidationStrategy, ContentModality
from ...core.semantic_integrity_validator import get_semantic_integrity_validator, ValidationStrategy, IntegrityDimension
from ...schemas.context import ContextSearchRequest


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory-analytics", tags=["Memory Analytics"])


# Pydantic models for API responses

class MemoryAnalyticsResponse(BaseModel):
    timestamp: str
    agent_id: Optional[str] = None
    system_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    health_status: str
    recommendations: List[str] = Field(default_factory=list)


class ConsolidationAnalyticsResponse(BaseModel):
    timestamp: str
    total_consolidations: int
    successful_consolidations: int
    average_token_reduction: float
    modality_distribution: Dict[str, int]
    strategy_effectiveness: Dict[str, float]
    recent_consolidations: List[Dict[str, Any]]


class VectorSearchAnalyticsResponse(BaseModel):
    timestamp: str
    total_searches: int
    average_response_time_ms: float
    cache_hit_rate: float
    strategy_performance: Dict[str, Dict[str, Any]]
    recent_searches: List[Dict[str, Any]]


class IntegrityValidationAnalyticsResponse(BaseModel):
    timestamp: str
    total_validations: int
    validation_pass_rate: float
    average_integrity_score: float
    dimension_performance: Dict[str, float]
    recent_validations: List[Dict[str, Any]]


class SystemHealthResponse(BaseModel):
    overall_status: str
    components: Dict[str, Dict[str, Any]]
    performance_summary: Dict[str, Any]
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str


class OptimizationMetricsResponse(BaseModel):
    timestamp: str
    total_optimizations: int
    successful_optimizations: int
    average_optimization_time_ms: float
    average_token_reduction_percent: float
    agent_specific_metrics: Dict[str, Any]


# WebSocket manager for real-time analytics
class MemoryAnalyticsWebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Start monitoring if not already active
        if not self.monitoring_active:
            await self.start_monitoring()
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Stop monitoring if no active connections
        if not self.active_connections and self.monitoring_active:
            self.stop_monitoring()
    
    async def start_monitoring(self):
        """Start real-time monitoring and broadcasting."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
    
    async def broadcast_analytics(self, data: Dict[str, Any]):
        """Broadcast analytics data to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def _monitoring_loop(self):
        """Main monitoring loop for real-time analytics."""
        try:
            while self.monitoring_active:
                try:
                    # Collect real-time metrics
                    analytics_data = await self._collect_real_time_metrics()
                    
                    # Broadcast to connected clients
                    await self.broadcast_analytics(analytics_data)
                    
                    # Wait before next update
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(10)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
        finally:
            self.monitoring_active = False
    
    async def _collect_real_time_metrics(self) -> Dict[str, Any]:
        """Collect real-time metrics from all systems."""
        try:
            # Get services
            memory_manager = await get_enhanced_memory_manager()
            optimizer = await get_sleep_wake_context_optimizer()
            vector_search = await get_memory_aware_vector_search()
            consolidation_service = await get_memory_consolidation_service()
            validator = await get_semantic_integrity_validator()
            
            # Collect metrics
            memory_analytics = await memory_manager.get_memory_analytics()
            optimization_analytics = await optimizer.get_optimization_analytics()
            search_analytics = await vector_search.get_search_analytics()
            consolidation_analytics = await consolidation_service.get_consolidation_analytics()
            validation_analytics = await validator.get_validation_analytics()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "memory_analytics": memory_analytics,
                "optimization_analytics": optimization_analytics,
                "search_analytics": search_analytics,
                "consolidation_analytics": consolidation_analytics,
                "validation_analytics": validation_analytics,
                "system_status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Real-time metrics collection failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "system_status": "degraded"
            }


# Global WebSocket manager
ws_manager = MemoryAnalyticsWebSocketManager()


# API Endpoints

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get comprehensive system health status."""
    try:
        # Get services
        memory_manager = await get_enhanced_memory_manager()
        optimizer = await get_sleep_wake_context_optimizer()
        vector_search = await get_memory_aware_vector_search()
        consolidation_service = await get_memory_consolidation_service()
        validator = await get_semantic_integrity_validator()
        
        # Collect health status from each component
        components = {}
        
        # Memory Manager health
        try:
            memory_health = await memory_manager.health_check()
            components["memory_manager"] = {
                "status": memory_health.get("status", "unknown"),
                "metrics": memory_health.get("metrics", {})
            }
        except Exception as e:
            components["memory_manager"] = {"status": "unhealthy", "error": str(e)}
        
        # Vector Search health
        try:
            search_health = await vector_search.health_check()
            components["vector_search"] = search_health
        except Exception as e:
            components["vector_search"] = {"status": "unhealthy", "error": str(e)}
        
        # Other components (simplified health checks)
        components["context_optimizer"] = {"status": "healthy"}
        components["consolidation_service"] = {"status": "healthy"}
        components["integrity_validator"] = {"status": "healthy"}
        
        # Determine overall status
        component_statuses = [comp.get("status", "unknown") for comp in components.values()]
        if all(status == "healthy" for status in component_statuses):
            overall_status = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        # Performance summary
        performance_summary = {
            "total_components": len(components),
            "healthy_components": sum(1 for status in component_statuses if status == "healthy"),
            "response_time_ms": 0.0  # Would measure actual response time
        }
        
        # Generate alerts for unhealthy components
        alerts = []
        for comp_name, comp_health in components.items():
            if comp_health.get("status") == "unhealthy":
                alerts.append({
                    "component": comp_name,
                    "severity": "high",
                    "message": f"{comp_name} is unhealthy",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return SystemHealthResponse(
            overall_status=overall_status,
            components=components,
            performance_summary=performance_summary,
            alerts=alerts,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/memory", response_model=MemoryAnalyticsResponse)
async def get_memory_analytics(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for specific analytics"),
    include_distribution: bool = Query(True, description="Include memory type distribution"),
    include_performance: bool = Query(True, description="Include performance metrics")
):
    """Get comprehensive memory analytics."""
    try:
        memory_manager = await get_enhanced_memory_manager()
        
        # Get memory analytics
        analytics = await memory_manager.get_memory_analytics(
            agent_id=agent_id,
            include_distribution=include_distribution,
            include_performance_metrics=include_performance
        )
        
        # Extract system metrics
        system_metrics = analytics.get("system_analytics", {})
        performance_metrics = analytics.get("performance_metrics", {})
        
        # Generate recommendations based on metrics
        recommendations = []
        
        # Check memory utilization
        if "agent_analytics" in analytics:
            for agent_data in analytics["agent_analytics"].values():
                utilization = agent_data.get("memory_utilization_percent", 0)
                if utilization > 90:
                    recommendations.append(f"High memory utilization ({utilization:.1f}%) - consider consolidation")
                elif utilization < 20:
                    recommendations.append(f"Low memory utilization ({utilization:.1f}%) - possible over-allocation")
        
        # Check performance metrics
        if performance_metrics.get("average_retrieval_time_ms", 0) > 1000:
            recommendations.append("Slow memory retrieval - consider performance optimization")
        
        if not recommendations:
            recommendations.append("Memory system operating within normal parameters")
        
        return MemoryAnalyticsResponse(
            timestamp=datetime.utcnow().isoformat(),
            agent_id=str(agent_id) if agent_id else None,
            system_metrics=system_metrics,
            performance_metrics=performance_metrics,
            health_status="healthy",
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Memory analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory analytics failed: {str(e)}")


@router.get("/consolidation", response_model=ConsolidationAnalyticsResponse)
async def get_consolidation_analytics(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for specific analytics"),
    time_range_hours: int = Query(24, description="Time range in hours for analytics")
):
    """Get memory consolidation analytics."""
    try:
        consolidation_service = await get_memory_consolidation_service()
        
        # Calculate time range
        time_range = None
        if time_range_hours > 0:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_range_hours)
            time_range = (start_time, end_time)
        
        # Get consolidation analytics
        analytics = await consolidation_service.get_consolidation_analytics(
            agent_id=agent_id,
            time_range=time_range
        )
        
        # Extract metrics
        system_metrics = analytics.get("system_metrics", {})
        
        return ConsolidationAnalyticsResponse(
            timestamp=datetime.utcnow().isoformat(),
            total_consolidations=system_metrics.get("total_consolidations", 0),
            successful_consolidations=system_metrics.get("successful_consolidations", 0),
            average_token_reduction=system_metrics.get("average_token_reduction_percent", 0.0),
            modality_distribution=dict(system_metrics.get("modality_distribution", {})),
            strategy_effectiveness=dict(system_metrics.get("strategy_effectiveness", {})),
            recent_consolidations=analytics.get("recent_consolidations", [])
        )
        
    except Exception as e:
        logger.error(f"Consolidation analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consolidation analytics failed: {str(e)}")


@router.get("/search", response_model=VectorSearchAnalyticsResponse)
async def get_search_analytics(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for specific analytics"),
    time_range_hours: int = Query(24, description="Time range in hours for analytics")
):
    """Get vector search analytics."""
    try:
        vector_search = await get_memory_aware_vector_search()
        
        # Calculate time range
        time_range = None
        if time_range_hours > 0:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_range_hours)
            time_range = (start_time, end_time)
        
        # Get search analytics
        analytics = await vector_search.get_search_analytics(
            agent_id=agent_id,
            time_range=time_range
        )
        
        # Extract metrics
        system_analytics = analytics.get("system_analytics", {})
        strategy_performance = analytics.get("strategy_performance", {})
        recent_searches = analytics.get("recent_searches", [])
        
        return VectorSearchAnalyticsResponse(
            timestamp=datetime.utcnow().isoformat(),
            total_searches=system_analytics.get("total_searches", 0),
            average_response_time_ms=system_analytics.get("average_response_time_ms", 0.0),
            cache_hit_rate=system_analytics.get("cache_hit_rate", 0.0),
            strategy_performance=strategy_performance,
            recent_searches=recent_searches
        )
        
    except Exception as e:
        logger.error(f"Search analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search analytics failed: {str(e)}")


@router.get("/validation", response_model=IntegrityValidationAnalyticsResponse)
async def get_validation_analytics(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for specific analytics"),
    time_range_hours: int = Query(24, description="Time range in hours for analytics")
):
    """Get semantic integrity validation analytics."""
    try:
        validator = await get_semantic_integrity_validator()
        
        # Calculate time range
        time_range = None
        if time_range_hours > 0:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_range_hours)
            time_range = (start_time, end_time)
        
        # Get validation analytics
        analytics = await validator.get_validation_analytics(
            agent_id=agent_id,
            time_range=time_range
        )
        
        # Extract metrics
        system_metrics = analytics.get("system_metrics", {})
        dimension_performance = analytics.get("dimension_performance", {})
        recent_validations = analytics.get("recent_validations", [])
        
        # Calculate pass rate
        total_validations = system_metrics.get("total_validations", 0)
        passed_validations = system_metrics.get("passed_validations", 0)
        pass_rate = passed_validations / max(1, total_validations)
        
        # Convert dimension performance to simple float values
        simplified_dimension_performance = {}
        for dimension, data in dimension_performance.items():
            if isinstance(data, dict):
                simplified_dimension_performance[dimension] = data.get("average_score", 0.0)
            else:
                simplified_dimension_performance[dimension] = float(data)
        
        return IntegrityValidationAnalyticsResponse(
            timestamp=datetime.utcnow().isoformat(),
            total_validations=total_validations,
            validation_pass_rate=pass_rate,
            average_integrity_score=system_metrics.get("average_integrity_score", 0.0),
            dimension_performance=simplified_dimension_performance,
            recent_validations=recent_validations
        )
        
    except Exception as e:
        logger.error(f"Validation analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation analytics failed: {str(e)}")


@router.get("/optimization", response_model=OptimizationMetricsResponse)
async def get_optimization_metrics(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for specific metrics"),
    include_predictions: bool = Query(True, description="Include optimization predictions")
):
    """Get sleep-wake context optimization metrics."""
    try:
        optimizer = await get_sleep_wake_context_optimizer()
        
        # Get optimization analytics
        analytics = await optimizer.get_optimization_analytics(
            agent_id=agent_id,
            include_history=True,
            include_predictions=include_predictions
        )
        
        # Extract system metrics
        system_metrics = analytics.get("system_metrics", {})
        
        # Get agent-specific metrics
        agent_specific = {}
        if agent_id and "agent_specific" in analytics:
            agent_specific = analytics["agent_specific"].get(str(agent_id), {})
        
        return OptimizationMetricsResponse(
            timestamp=datetime.utcnow().isoformat(),
            total_optimizations=system_metrics.get("total_optimizations", 0),
            successful_optimizations=system_metrics.get("successful_optimizations", 0),
            average_optimization_time_ms=system_metrics.get("average_optimization_time_ms", 0.0),
            average_token_reduction_percent=system_metrics.get("average_token_reduction_percent", 0.0),
            agent_specific_metrics=agent_specific
        )
        
    except Exception as e:
        logger.error(f"Optimization metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization metrics failed: {str(e)}")


@router.post("/consolidate/{agent_id}")
async def trigger_consolidation(
    agent_id: UUID,
    strategy: ConsolidationStrategy = ConsolidationStrategy.SEMANTIC_MERGE,
    target_reduction: float = Query(0.7, ge=0.1, le=0.9),
    modality: Optional[ContentModality] = None
):
    """Trigger memory consolidation for an agent."""
    try:
        consolidation_service = await get_memory_consolidation_service()
        
        if modality:
            # Consolidate specific modality
            result = await consolidation_service.consolidate_by_modality(
                agent_id=agent_id,
                target_modality=modality,
                consolidation_strategy=strategy,
                quality_threshold=0.9
            )
        else:
            # Schedule background consolidation
            job_id = await consolidation_service.schedule_background_consolidation(
                agent_id=agent_id,
                priority=5,
                delay_minutes=0
            )
            
            return {"message": "Consolidation scheduled", "job_id": job_id}
        
        return {
            "message": "Consolidation completed",
            "result": {
                "success": result.success,
                "original_fragments": result.original_fragment_count,
                "consolidated_fragments": result.consolidated_fragment_count,
                "token_reduction": result.token_reduction_ratio,
                "processing_time": result.consolidation_time_seconds
            }
        }
        
    except Exception as e:
        logger.error(f"Consolidation trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {str(e)}")


@router.post("/optimize/{agent_id}")
async def trigger_optimization(
    agent_id: UUID,
    trigger: OptimizationTrigger = OptimizationTrigger.MANUAL_TRIGGER,
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
):
    """Trigger context optimization for an agent."""
    try:
        optimizer = await get_sleep_wake_context_optimizer()
        
        # Trigger optimization
        session = await optimizer.optimize_during_sleep_cycle(
            agent_id=agent_id,
            trigger=trigger,
            strategy=strategy
        )
        
        return {
            "message": "Optimization completed",
            "session": {
                "session_id": session.session_id,
                "success": session.success,
                "token_reduction": session.token_reduction_achieved,
                "contexts_processed": session.contexts_processed,
                "optimization_time": session.optimization_time_ms
            }
        }
        
    except Exception as e:
        logger.error(f"Optimization trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/performance/optimize")
async def optimize_system_performance(
    target_performance_ms: float = Query(500.0, description="Target performance in milliseconds"),
    agent_id: Optional[UUID] = Query(None, description="Specific agent to optimize")
):
    """Optimize overall system performance."""
    try:
        results = {}
        
        # Optimize memory performance
        memory_manager = await get_enhanced_memory_manager()
        if agent_id:
            memory_result = await memory_manager.optimize_memory_performance(
                agent_id=agent_id,
                target_performance_ms=target_performance_ms
            )
            results["memory_optimization"] = memory_result
        
        # Optimize search performance
        vector_search = await get_memory_aware_vector_search()
        search_result = await vector_search.optimize_search_performance(
            agent_id=agent_id,
            target_performance_ms=target_performance_ms
        )
        results["search_optimization"] = search_result
        
        # Optimize consolidation strategies
        consolidation_service = await get_memory_consolidation_service()
        consolidation_result = await consolidation_service.optimize_consolidation_strategies(
            agent_id=agent_id
        )
        results["consolidation_optimization"] = consolidation_result
        
        return {
            "message": "System performance optimization completed",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance optimization failed: {str(e)}")


@router.websocket("/realtime")
async def websocket_analytics(websocket: WebSocket):
    """WebSocket endpoint for real-time analytics streaming."""
    await ws_manager.connect(websocket)
    
    try:
        # Send initial analytics data
        initial_data = await ws_manager._collect_real_time_metrics()
        await websocket.send_json(initial_data)
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message or timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client requests for specific data
                if message == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                elif message.startswith("request:"):
                    # Handle specific data requests
                    request_type = message.split(":", 1)[1]
                    response_data = await ws_manager._collect_real_time_metrics()
                    response_data["request_type"] = request_type
                    await websocket.send_json(response_data)
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        ws_manager.disconnect(websocket)


@router.get("/agents/{agent_id}/summary")
async def get_agent_memory_summary(agent_id: UUID):
    """Get comprehensive memory summary for a specific agent."""
    try:
        # Collect data from all systems
        memory_manager = await get_enhanced_memory_manager()
        optimizer = await get_sleep_wake_context_optimizer()
        vector_search = await get_memory_aware_vector_search()
        consolidation_service = await get_memory_consolidation_service()
        validator = await get_semantic_integrity_validator()
        
        # Get analytics for the agent
        memory_analytics = await memory_manager.get_memory_analytics(agent_id=agent_id)
        optimization_analytics = await optimizer.get_optimization_analytics(agent_id=agent_id)
        search_analytics = await vector_search.get_search_analytics(agent_id=agent_id)
        consolidation_analytics = await consolidation_service.get_consolidation_analytics(agent_id=agent_id)
        validation_analytics = await validator.get_validation_analytics(agent_id=agent_id)
        
        # Compile summary
        summary = {
            "agent_id": str(agent_id),
            "timestamp": datetime.utcnow().isoformat(),
            "memory_status": {
                "total_memories": memory_analytics.get("agent_analytics", {}).get(str(agent_id), {}).get("total_memories", 0),
                "memory_utilization_percent": memory_analytics.get("agent_analytics", {}).get(str(agent_id), {}).get("memory_utilization_percent", 0),
                "average_importance": memory_analytics.get("agent_analytics", {}).get(str(agent_id), {}).get("average_importance", 0)
            },
            "optimization_status": {
                "total_optimizations": optimization_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("total_optimizations", 0),
                "success_rate": optimization_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("success_rate", 0),
                "average_token_reduction": optimization_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("average_token_reduction", 0)
            },
            "search_performance": {
                "total_searches": search_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("total_searches", 0),
                "average_response_time": search_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("average_search_time_ms", 0),
                "search_frequency": search_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("search_frequency", 0)
            },
            "consolidation_metrics": {
                "total_consolidations": consolidation_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("total_consolidations", 0),
                "consolidation_efficiency": consolidation_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("consolidation_efficiency", 0)
            },
            "validation_metrics": {
                "total_validations": validation_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("total_validations", 0),
                "average_integrity_score": validation_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("average_score", 0),
                "validation_pass_rate": validation_analytics.get("agent_specific", {}).get(str(agent_id), {}).get("pass_rate", 0)
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        memory_util = summary["memory_status"]["memory_utilization_percent"]
        if memory_util > 90:
            recommendations.append("High memory utilization - consider consolidation")
        elif memory_util < 20:
            recommendations.append("Low memory utilization - review memory allocation")
        
        search_time = summary["search_performance"]["average_response_time"]
        if search_time > 1000:
            recommendations.append("Slow search performance - consider optimization")
        
        validation_rate = summary["validation_metrics"]["validation_pass_rate"]
        if validation_rate < 0.9:
            recommendations.append("Low validation pass rate - review consolidation quality")
        
        if not recommendations:
            recommendations.append("Agent memory systems operating normally")
        
        summary["recommendations"] = recommendations
        summary["overall_health"] = "healthy" if not any("review" in rec or "consider" in rec for rec in recommendations) else "attention_needed"
        
        return summary
        
    except Exception as e:
        logger.error(f"Agent summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent summary failed: {str(e)}")


# Cleanup function for WebSocket manager
async def cleanup_websocket_manager():
    """Cleanup WebSocket manager on shutdown."""
    ws_manager.stop_monitoring()
    for connection in ws_manager.active_connections:
        try:
            await connection.close()
        except Exception:
            pass
    ws_manager.active_connections.clear()