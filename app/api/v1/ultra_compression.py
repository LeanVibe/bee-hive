"""
Ultra Compression API endpoints for real-time context compression management.

Provides RESTful API for:
- Ultra compression operations with 70% token reduction
- Real-time compression monitoring and control
- Compression analytics and performance metrics
- Adaptive threshold management
- Integration with existing agent and context systems
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi import Depends, Query, Path, Body
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from ...core.enhanced_context_consolidator import (
    UltraCompressedContextMode,
    CompressionStrategy,
    ContextPriority,
    CompressionMetrics,
    get_ultra_compressed_context_mode,
    ultra_compress_agent_contexts,
    start_real_time_compression
)
from ...core.dependencies import get_current_user
from ...schemas.base import BaseResponse
from ...models.user import User


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ultra-compression", tags=["Ultra Compression"])


# Request/Response Models
class CompressionRequest(BaseModel):
    """Request model for compression operations."""
    agent_id: UUID = Field(..., description="Agent ID to compress contexts for")
    target_reduction: float = Field(0.70, ge=0.1, le=0.95, description="Target compression ratio (0.1-0.95)")
    preserve_critical: bool = Field(True, description="Whether to preserve critical contexts")
    
    @validator('target_reduction')
    def validate_target_reduction(cls, v):
        if not 0.1 <= v <= 0.95:
            raise ValueError('Target reduction must be between 0.1 and 0.95')
        return v


class CompressionResponse(BaseModel):
    """Response model for compression operations."""
    success: bool = Field(..., description="Whether compression was successful")
    compression_metrics: Dict[str, Any] = Field(..., description="Detailed compression metrics")
    message: str = Field(..., description="Human-readable status message")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class ThresholdOptimizationRequest(BaseModel):
    """Request model for threshold optimization."""
    agent_id: UUID = Field(..., description="Agent ID to optimize thresholds for")
    force_optimization: bool = Field(False, description="Force optimization even without enough history")


class CompressionAnalyticsResponse(BaseModel):
    """Response model for compression analytics."""
    performance_metrics: Dict[str, Any] = Field(..., description="Overall performance metrics")
    adaptive_thresholds: Dict[str, float] = Field(..., description="Current adaptive thresholds")
    recent_compressions: List[Dict[str, Any]] = Field(..., description="Recent compression details")
    agent_specific: Optional[Dict[str, Any]] = Field(None, description="Agent-specific analytics")


class RealTimeCompressionConfig(BaseModel):
    """Configuration for real-time compression."""
    agent_id: UUID = Field(..., description="Agent ID to monitor")
    interval_minutes: int = Field(30, ge=1, le=1440, description="Compression interval in minutes")
    target_reduction: float = Field(0.70, ge=0.1, le=0.95, description="Target compression ratio")
    auto_stop_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Auto-stop when compression ratio falls below this")


# WebSocket connection manager for real-time compression
class CompressionWebSocketManager:
    """Manages WebSocket connections for real-time compression monitoring."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.compression_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.compression_tasks:
            self.compression_tasks[connection_id].cancel()
            del self.compression_tasks[connection_id]
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_compression_update(self, connection_id: str, data: Dict[str, Any]):
        """Send compression update to specific client."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_json(data)
            except Exception as e:
                logger.error(f"Error sending WebSocket message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast_compression_status(self, data: Dict[str, Any]):
        """Broadcast compression status to all connected clients."""
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            self.disconnect(connection_id)


# Global WebSocket manager
ws_manager = CompressionWebSocketManager()


@router.post("/compress", response_model=CompressionResponse, status_code=HTTP_201_CREATED)
async def compress_agent_contexts(
    request: CompressionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    compressor: UltraCompressedContextMode = Depends(get_ultra_compressed_context_mode)
):
    """
    Perform ultra compression on agent's contexts.
    
    Compresses contexts for the specified agent using advanced semantic clustering
    and compression strategies to achieve the target reduction ratio.
    """
    try:
        logger.info(f"Starting ultra compression for agent {request.agent_id} by user {current_user.id}")
        
        start_time = datetime.utcnow()
        
        # Perform compression
        metrics = await compressor.ultra_compress_agent_contexts(
            agent_id=request.agent_id,
            target_reduction=request.target_reduction,
            preserve_critical=request.preserve_critical
        )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Prepare response
        compression_data = {
            "original_token_count": metrics.original_token_count,
            "compressed_token_count": metrics.compressed_token_count,
            "compression_ratio": metrics.compression_ratio,
            "contexts_merged": metrics.contexts_merged,
            "contexts_archived": metrics.contexts_archived,
            "efficiency_score": metrics.calculate_efficiency_score(),
            "strategy_used": metrics.strategy_used.value if metrics.strategy_used else "unknown",
            "semantic_similarity_loss": metrics.semantic_similarity_loss,
            "target_achieved": metrics.compression_ratio >= request.target_reduction * 0.9  # 90% of target
        }
        
        success_message = (
            f"Compressed {metrics.original_token_count} → {metrics.compressed_token_count} tokens "
            f"({metrics.compression_ratio:.1%} reduction) for agent {request.agent_id}"
        )
        
        # Broadcast update to WebSocket clients
        background_tasks.add_task(
            ws_manager.broadcast_compression_status,
            {
                "event": "compression_completed",
                "agent_id": str(request.agent_id),
                "metrics": compression_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return CompressionResponse(
            success=True,
            compression_metrics=compression_data,
            message=success_message,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error compressing contexts for agent {request.agent_id}: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Compression failed: {str(e)}"
        )


@router.post("/optimize-thresholds", response_model=Dict[str, Any])
async def optimize_compression_thresholds(
    request: ThresholdOptimizationRequest,
    current_user: User = Depends(get_current_user),
    compressor: UltraCompressedContextMode = Depends(get_ultra_compressed_context_mode)
):
    """
    Optimize compression thresholds for an agent based on historical performance.
    
    Analyzes recent compression history and adjusts thresholds to improve
    compression efficiency and quality.
    """
    try:
        logger.info(f"Optimizing thresholds for agent {request.agent_id} by user {current_user.id}")
        
        # Optimize thresholds
        optimized_thresholds = await compressor.adaptive_threshold_optimization(request.agent_id)
        
        return {
            "success": True,
            "agent_id": str(request.agent_id),
            "optimized_thresholds": optimized_thresholds,
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "message": f"Thresholds optimized for agent {request.agent_id}"
        }
        
    except Exception as e:
        logger.error(f"Error optimizing thresholds for agent {request.agent_id}: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Threshold optimization failed: {str(e)}"
        )


@router.get("/analytics", response_model=CompressionAnalyticsResponse)
async def get_compression_analytics(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for specific analytics"),
    current_user: User = Depends(get_current_user),
    compressor: UltraCompressedContextMode = Depends(get_ultra_compressed_context_mode)
):
    """
    Get comprehensive compression analytics.
    
    Returns detailed analytics including performance metrics, adaptive thresholds,
    recent compression history, and agent-specific data if requested.
    """
    try:
        logger.info(f"Getting compression analytics for user {current_user.id}, agent: {agent_id}")
        
        # Get analytics
        analytics = await compressor.get_compression_analytics(agent_id)
        
        if "error" in analytics:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Analytics retrieval failed: {analytics['error']}"
            )
        
        return CompressionAnalyticsResponse(**analytics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting compression analytics: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Analytics retrieval failed: {str(e)}"
        )


@router.get("/status/{agent_id}")
async def get_agent_compression_status(
    agent_id: UUID = Path(..., description="Agent ID to check status for"),
    current_user: User = Depends(get_current_user),
    compressor: UltraCompressedContextMode = Depends(get_ultra_compressed_context_mode)
):
    """
    Get compression status for a specific agent.
    
    Returns current compression state, recent activity, and recommendations.
    """
    try:
        # Check if compression is needed
        should_compress = await compressor._should_compress(agent_id)
        
        # Get recent analytics
        analytics = await compressor.get_compression_analytics(agent_id)
        
        return {
            "agent_id": str(agent_id),
            "should_compress": should_compress,
            "compression_recommended": should_compress,
            "current_thresholds": analytics.get("adaptive_thresholds", {}),
            "recent_performance": analytics.get("recent_compressions", [])[-5:],  # Last 5
            "agent_specific": analytics.get("agent_specific", {}),
            "status_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting compression status for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Status retrieval failed: {str(e)}"
        )


@router.post("/real-time/start")
async def start_real_time_compression_monitoring(
    config: RealTimeCompressionConfig,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    compressor: UltraCompressedContextMode = Depends(get_ultra_compressed_context_mode)
):
    """
    Start real-time compression monitoring for an agent.
    
    Begins continuous monitoring and compression of the agent's contexts
    at the specified interval.
    """
    try:
        logger.info(f"Starting real-time compression for agent {config.agent_id} by user {current_user.id}")
        
        # Check if already running
        task_id = f"compression_{config.agent_id}"
        if task_id in ws_manager.compression_tasks:
            return {
                "success": False,
                "message": f"Real-time compression already running for agent {config.agent_id}",
                "agent_id": str(config.agent_id)
            }
        
        # Start real-time compression task
        async def compression_monitor():
            try:
                async for metrics in compressor.real_time_compression_stream(
                    agent_id=config.agent_id,
                    compression_interval_minutes=config.interval_minutes
                ):
                    # Send update via WebSocket
                    await ws_manager.broadcast_compression_status({
                        "event": "real_time_compression",
                        "agent_id": str(config.agent_id),
                        "metrics": {
                            "compression_ratio": metrics.compression_ratio,
                            "contexts_merged": metrics.contexts_merged,
                            "efficiency_score": metrics.calculate_efficiency_score(),
                            "processing_time_ms": metrics.processing_time_ms
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Check auto-stop threshold
                    if (config.auto_stop_threshold is not None and 
                        metrics.compression_ratio < config.auto_stop_threshold):
                        logger.info(f"Auto-stopping compression for agent {config.agent_id} due to low ratio")
                        break
                        
            except asyncio.CancelledError:
                logger.info(f"Real-time compression cancelled for agent {config.agent_id}")
            except Exception as e:
                logger.error(f"Error in real-time compression for agent {config.agent_id}: {e}")
        
        # Start the task
        task = asyncio.create_task(compression_monitor())
        ws_manager.compression_tasks[task_id] = task
        
        return {
            "success": True,
            "message": f"Real-time compression started for agent {config.agent_id}",
            "agent_id": str(config.agent_id),
            "config": config.dict(),
            "start_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting real-time compression for agent {config.agent_id}: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to start real-time compression: {str(e)}"
        )


@router.post("/real-time/stop/{agent_id}")
async def stop_real_time_compression_monitoring(
    agent_id: UUID = Path(..., description="Agent ID to stop monitoring for"),
    current_user: User = Depends(get_current_user)
):
    """
    Stop real-time compression monitoring for an agent.
    """
    try:
        task_id = f"compression_{agent_id}"
        
        if task_id not in ws_manager.compression_tasks:
            return {
                "success": False,
                "message": f"No real-time compression running for agent {agent_id}",
                "agent_id": str(agent_id)
            }
        
        # Cancel the task
        ws_manager.compression_tasks[task_id].cancel()
        del ws_manager.compression_tasks[task_id]
        
        logger.info(f"Stopped real-time compression for agent {agent_id} by user {current_user.id}")
        
        return {
            "success": True,
            "message": f"Real-time compression stopped for agent {agent_id}",
            "agent_id": str(agent_id),
            "stop_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping real-time compression for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to stop real-time compression: {str(e)}"
        )


@router.get("/real-time/status")
async def get_real_time_compression_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get status of all real-time compression monitoring tasks.
    """
    try:
        active_tasks = []
        
        for task_id, task in ws_manager.compression_tasks.items():
            agent_id = task_id.replace("compression_", "")
            active_tasks.append({
                "agent_id": agent_id,
                "task_id": task_id,
                "is_running": not task.done(),
                "is_cancelled": task.cancelled() if task.done() else False
            })
        
        return {
            "active_compressions": len(active_tasks),
            "active_websocket_connections": len(ws_manager.active_connections),
            "tasks": active_tasks,
            "status_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting real-time compression status: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Status retrieval failed: {str(e)}"
        )


@router.websocket("/ws/{connection_id}")
async def compression_websocket_endpoint(
    websocket: WebSocket,
    connection_id: str = Path(..., description="Unique connection identifier")
):
    """
    WebSocket endpoint for real-time compression monitoring.
    
    Provides real-time updates on compression operations, metrics,
    and system status.
    """
    await ws_manager.connect(websocket, connection_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "event": "connected",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to ultra compression monitoring"
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (for potential control commands)
                data = await websocket.receive_json()
                
                # Handle client commands
                if data.get("command") == "ping":
                    await websocket.send_json({
                        "event": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif data.get("command") == "get_status":
                    # Send current status
                    await websocket.send_json({
                        "event": "status_update",
                        "active_compressions": len(ws_manager.compression_tasks),
                        "connected_clients": len(ws_manager.active_connections),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {connection_id}: {e}")
                await websocket.send_json({
                    "event": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {connection_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        ws_manager.disconnect(connection_id)


@router.get("/strategies")
async def get_compression_strategies(
    current_user: User = Depends(get_current_user)
):
    """
    Get information about available compression strategies.
    
    Returns details about different compression strategies and their characteristics.
    """
    strategies = {
        "aggressive_merge": {
            "name": "Aggressive Merge",
            "reduction": "80-90%",
            "description": "Merges highly similar contexts with significant information loss but maximum space savings",
            "use_case": "Large volumes of redundant or low-importance contexts",
            "quality_impact": "High"
        },
        "semantic_cluster": {
            "name": "Semantic Cluster",
            "reduction": "70-80%",
            "description": "Groups semantically similar contexts and compresses while preserving meaning",
            "use_case": "General purpose compression with good balance of reduction and quality",
            "quality_impact": "Medium"
        },
        "hierarchical_compress": {
            "name": "Hierarchical Compress",
            "reduction": "60-70%",
            "description": "Applies different compression levels based on context importance hierarchy",
            "use_case": "Mixed importance contexts requiring preservation of critical information",
            "quality_impact": "Low-Medium"
        },
        "intelligent_summary": {
            "name": "Intelligent Summary",
            "reduction": "50-60%",
            "description": "Creates intelligent summaries preserving key insights and decisions",
            "use_case": "High-value contexts requiring detailed preservation",
            "quality_impact": "Low"
        },
        "light_optimization": {
            "name": "Light Optimization",
            "reduction": "30-40%",
            "description": "Minimal compression focusing on format optimization and redundancy removal",
            "use_case": "Critical contexts or when quality preservation is paramount",
            "quality_impact": "Minimal"
        }
    }
    
    priorities = {
        "critical": {
            "name": "Critical",
            "description": "Never compressed aggressively, preserves all important information",
            "default_strategy": "light_optimization"
        },
        "high": {
            "name": "High",
            "description": "Conservative compression with quality preservation",
            "default_strategy": "intelligent_summary"
        },
        "medium": {
            "name": "Medium",
            "description": "Balanced compression with reasonable quality retention",
            "default_strategy": "hierarchical_compress"
        },
        "low": {
            "name": "Low",
            "description": "Aggressive compression allowed for space optimization",
            "default_strategy": "semantic_cluster"
        },
        "disposable": {
            "name": "Disposable",
            "description": "Maximum compression or archival, minimal quality requirements",
            "default_strategy": "aggressive_merge"
        }
    }
    
    return {
        "compression_strategies": strategies,
        "priority_levels": priorities,
        "target_reductions": {
            "conservative": 0.5,
            "balanced": 0.7,
            "aggressive": 0.85
        },
        "recommendations": {
            "first_time_users": "Start with 'balanced' target (70%) and 'intelligent_summary' strategy",
            "high_volume_systems": "Use 'aggressive' target (85%) with automatic strategy selection",
            "quality_critical_systems": "Use 'conservative' target (50%) with 'light_optimization' strategy"
        }
    }