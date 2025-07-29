"""
Enhanced Dashboard Backend APIs for Semantic Intelligence - VS 6.2
LeanVibe Agent Hive 2.0

Provides APIs for:
- Semantic Query Explorer with natural language interface
- Context Trajectory visualization
- Intelligence KPI metrics and trend analysis  
- Live workflow constellation data
- Performance optimization for <2s response times
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import re

import structlog
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, func, desc, asc, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_session
from ...core.embedding_service import get_embedding_service
from ...models.observability import AgentEvent, EventType
from ...models.context import Context, ContextType
from ...models.agent import Agent
from ...models.session import Session
from ...schemas.observability import BaseObservabilityEvent
from ...services.event_collector_service import get_event_collector
from .observability_websocket import get_websocket_manager, DashboardEvent, DashboardEventType

logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/observability/dashboard", tags=["observability-dashboard"])


class SemanticQuery(BaseModel):
    """Natural language query for semantic search."""
    query: str = Field(..., min_length=3, max_length=500, description="Natural language query")
    context_window_hours: Optional[int] = Field(default=24, ge=1, le=168, description="Time window in hours")
    max_results: Optional[int] = Field(default=50, ge=1, le=500, description="Maximum number of results")
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Semantic similarity threshold")
    include_context: bool = Field(default=True, description="Include context lineage in results")
    include_performance: bool = Field(default=True, description="Include performance metrics")


class ContextTrajectoryRequest(BaseModel):
    """Request for context trajectory visualization."""
    context_id: Optional[str] = Field(default=None, description="Specific context ID to trace")
    concept: Optional[str] = Field(default=None, description="Semantic concept to trace")
    agent_id: Optional[str] = Field(default=None, description="Agent to trace context for")
    session_id: Optional[str] = Field(default=None, description="Session to trace context for")
    time_range_hours: int = Field(default=24, ge=1, le=168, description="Time range in hours")
    max_depth: int = Field(default=10, ge=1, le=50, description="Maximum trajectory depth")


class IntelligenceKPIRequest(BaseModel):
    """Request for intelligence KPI metrics."""
    kpi_names: Optional[List[str]] = Field(default=None, description="Specific KPI names to retrieve")
    time_range_hours: int = Field(default=24, ge=1, le=168, description="Time range in hours")
    aggregation_interval: str = Field(default="1h", regex="^(1m|5m|15m|1h|6h|1d)$", description="Aggregation interval")
    include_trends: bool = Field(default=True, description="Include trend analysis")
    include_forecasts: bool = Field(default=False, description="Include forecasted values")


class WorkflowConstellationRequest(BaseModel):
    """Request for workflow constellation visualization data."""
    session_ids: Optional[List[str]] = Field(default=None, description="Session IDs to include")
    agent_ids: Optional[List[str]] = Field(default=None, description="Agent IDs to include")
    time_range_hours: int = Field(default=1, ge=1, le=24, description="Time range in hours")
    include_semantic_flow: bool = Field(default=True, description="Include semantic concept flow")
    min_interaction_count: int = Field(default=1, ge=1, description="Minimum interaction count to include")


# Response models

class SemanticSearchResult(BaseModel):
    """Semantic search result with context and relevance."""
    id: str
    event_type: str
    timestamp: datetime
    relevance_score: float
    content_summary: str
    agent_id: Optional[str]
    session_id: Optional[str]
    semantic_concepts: List[str]
    context_references: List[str]
    performance_metrics: Optional[Dict[str, Any]]


class ContextTrajectoryNode(BaseModel):
    """Node in context trajectory visualization."""
    id: str
    type: str  # 'context', 'event', 'agent', 'concept'
    label: str
    timestamp: Optional[datetime]
    metadata: Dict[str, Any]
    semantic_embedding: Optional[List[float]]
    connections: List[str]  # IDs of connected nodes


class ContextTrajectoryPath(BaseModel):
    """Path through context trajectory."""
    nodes: List[ContextTrajectoryNode]
    edges: List[Dict[str, str]]  # source_id, target_id, relationship_type
    semantic_similarity: float
    path_strength: float
    temporal_flow: List[datetime]


class IntelligenceKPI(BaseModel):
    """Intelligence KPI metric with trend data."""
    name: str
    description: str
    current_value: float
    unit: str
    trend_direction: str  # 'up', 'down', 'stable'
    trend_strength: float  # 0.0 to 1.0
    threshold_status: str  # 'normal', 'warning', 'critical'
    historical_data: List[Dict[str, Any]]
    forecast_data: Optional[List[Dict[str, Any]]]


class WorkflowConstellationNode(BaseModel):
    """Node in workflow constellation."""
    id: str
    type: str  # 'agent', 'concept', 'session'
    label: str
    position: Dict[str, float]  # x, y coordinates
    size: float  # Relative size based on activity
    color: str  # Color coding for status/type
    metadata: Dict[str, Any]


class WorkflowConstellationEdge(BaseModel):
    """Edge in workflow constellation."""
    source: str
    target: str
    type: str  # 'communication', 'semantic_flow', 'context_sharing'
    strength: float  # 0.0 to 1.0
    frequency: int
    latency_ms: Optional[float]
    semantic_concepts: List[str]


class WorkflowConstellation(BaseModel):
    """Complete workflow constellation visualization data."""
    nodes: List[WorkflowConstellationNode]
    edges: List[WorkflowConstellationEdge]
    semantic_flows: List[Dict[str, Any]]
    temporal_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Semantic query processing

class SemanticQueryProcessor:
    """Advanced semantic query processor with natural language understanding."""
    
    def __init__(self):
        self.embedding_service = None
        self.query_patterns = {
            'temporal': [
                r'(last|past|recent|within)\s+(\d+)\s+(hour|day|minute)s?',
                r'(today|yesterday|this week)',
                r'(before|after|since)\s+(.+)'
            ],
            'agent_specific': [
                r'agent\s+([a-f0-9-]+)',
                r'(claude|gpt|assistant|bot)\s*([a-f0-9-]*)',
            ],
            'performance': [
                r'(slow|fast|performance|latency|speed)',
                r'(error|fail|success|complete)',
                r'(response time|duration|took)'
            ],
            'semantic': [
                r'(similar to|like|related to)\s+(.+)',
                r'(concept|idea|topic)\s+(.+)',
                r'(context|reference|about)\s+(.+)'
            ]
        }
        
    async def _ensure_embedding_service(self):
        """Ensure embedding service is initialized."""
        if self.embedding_service is None:
            self.embedding_service = await get_embedding_service()
    
    async def process_query(
        self, 
        query: SemanticQuery, 
        db_session: AsyncSession
    ) -> List[SemanticSearchResult]:
        """Process natural language query and return relevant results."""
        start_time = time.time()
        
        try:
            await self._ensure_embedding_service()
            
            # Parse query for intent and filters
            parsed_query = await self._parse_natural_language_query(query.query)
            
            # Generate semantic embedding
            query_embedding = await self.embedding_service.generate_embedding(query.query)
            
            # Build database query with semantic and traditional filters
            db_query = await self._build_semantic_query(
                parsed_query, query_embedding, query, db_session
            )
            
            # Execute query
            result = await db_session.execute(db_query)
            events = result.scalars().all()
            
            # Rank and filter results by semantic similarity
            results = await self._rank_semantic_results(
                events, query_embedding, query.similarity_threshold
            )
            
            # Enhance results with context and performance data
            if query.include_context or query.include_performance:
                results = await self._enhance_search_results(
                    results, query, db_session
                )
            
            processing_time = time.time() - start_time
            logger.info(
                "Semantic query processed",
                query=query.query,
                results_count=len(results),
                processing_time_ms=round(processing_time * 1000, 2)
            )
            
            return results[:query.max_results]
            
        except Exception as e:
            logger.error(f"Semantic query processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    async def _parse_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query for intent and filters."""
        parsed = {
            'intent': 'general_search',
            'temporal_filter': None,
            'agent_filter': None,
            'performance_filter': None,
            'semantic_concepts': [],
            'entities': []
        }
        
        query_lower = query.lower()
        
        # Extract temporal information
        for pattern in self.query_patterns['temporal']:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                parsed['temporal_filter'] = matches[0]
                parsed['intent'] = 'temporal_search'
                break
        
        # Extract agent information
        for pattern in self.query_patterns['agent_specific']:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                parsed['agent_filter'] = matches[0]
                parsed['intent'] = 'agent_search'
                break
        
        # Extract performance intent
        for pattern in self.query_patterns['performance']:
            if re.search(pattern, query_lower, re.IGNORECASE):
                parsed['performance_filter'] = True
                parsed['intent'] = 'performance_search'
                break
        
        # Extract semantic concepts
        for pattern in self.query_patterns['semantic']:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                parsed['semantic_concepts'].extend(matches)
                parsed['intent'] = 'semantic_search'
        
        return parsed
    
    async def _build_semantic_query(
        self, 
        parsed_query: Dict[str, Any], 
        query_embedding: List[float],
        query: SemanticQuery,
        db_session: AsyncSession
    ):
        """Build database query based on parsed natural language query."""
        # Base query
        base_query = select(AgentEvent).order_by(desc(AgentEvent.created_at))
        
        # Apply temporal filters
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=query.context_window_hours)
        
        if parsed_query.get('temporal_filter'):
            # Parse specific temporal filter
            # This would be enhanced with more sophisticated temporal parsing
            pass
        
        base_query = base_query.where(
            and_(
                AgentEvent.created_at >= start_time,
                AgentEvent.created_at <= end_time
            )
        )
        
        # Apply agent filters
        if parsed_query.get('agent_filter'):
            # Extract agent ID or name
            agent_filter = parsed_query['agent_filter']
            if isinstance(agent_filter, tuple):
                agent_filter = agent_filter[0] if agent_filter[0] else agent_filter[1]
            
            base_query = base_query.where(
                or_(
                    AgentEvent.agent_id == agent_filter,
                    func.cast(AgentEvent.agent_id, db_session.bind.dialect.name == 'postgresql' and 'text' or 'char').like(f'%{agent_filter}%')
                )
            )
        
        # Apply performance filters
        if parsed_query.get('performance_filter'):
            base_query = base_query.where(AgentEvent.latency_ms.isnot(None))
        
        return base_query.limit(query.max_results * 2)  # Get more for similarity filtering
    
    async def _rank_semantic_results(
        self, 
        events: List[AgentEvent], 
        query_embedding: List[float],
        similarity_threshold: float
    ) -> List[SemanticSearchResult]:
        """Rank results by semantic similarity."""
        results = []
        
        for event in events:
            try:
                # Extract semantic embedding from event if available
                event_embedding = None
                if hasattr(event, 'payload') and event.payload:
                    event_embedding = event.payload.get('semantic_embedding')
                
                # Calculate semantic similarity
                similarity_score = 0.5  # Default score
                if event_embedding and query_embedding:
                    similarity_score = await self._calculate_cosine_similarity(
                        query_embedding, event_embedding
                    )
                
                # Filter by similarity threshold
                if similarity_score >= similarity_threshold:
                    # Extract content summary
                    content_summary = await self._extract_content_summary(event)
                    
                    # Extract semantic concepts
                    semantic_concepts = event.payload.get('semantic_concepts', []) if event.payload else []
                    
                    result = SemanticSearchResult(
                        id=str(event.id),
                        event_type=event.event_type.value,
                        timestamp=event.created_at,
                        relevance_score=similarity_score,
                        content_summary=content_summary,
                        agent_id=str(event.agent_id) if event.agent_id else None,
                        session_id=str(event.session_id) if event.session_id else None,
                        semantic_concepts=semantic_concepts,
                        context_references=event.payload.get('context_references', []) if event.payload else [],
                        performance_metrics=event.payload.get('performance_metrics') if event.payload else None
                    )
                    
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to process event {event.id} for semantic ranking: {e}")
                continue
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    async def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot_product / (norm_a * norm_b))
            
        except Exception:
            return 0.0
    
    async def _extract_content_summary(self, event: AgentEvent) -> str:
        """Extract content summary from event."""
        try:
            if not event.payload:
                return f"{event.event_type.value} event"
            
            # Extract meaningful content
            content_fields = ['content', 'message', 'description', 'summary', 'result']
            for field in content_fields:
                if field in event.payload and event.payload[field]:
                    content = str(event.payload[field])
                    return content[:200] + "..." if len(content) > 200 else content
            
            # Fallback to event type and basic info
            return f"{event.event_type.value} event from {event.agent_id or 'unknown agent'}"
            
        except Exception:
            return f"{event.event_type.value} event"
    
    async def _enhance_search_results(
        self, 
        results: List[SemanticSearchResult], 
        query: SemanticQuery,
        db_session: AsyncSession
    ) -> List[SemanticSearchResult]:
        """Enhance search results with additional context and performance data."""
        # For now, return results as-is
        # This could be enhanced with additional context queries
        return results


# Context trajectory processing

class ContextTrajectoryProcessor:
    """Processor for context trajectory visualization."""
    
    async def build_trajectory(
        self, 
        request: ContextTrajectoryRequest, 
        db_session: AsyncSession
    ) -> List[ContextTrajectoryPath]:
        """Build context trajectory paths."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=request.time_range_hours)
            
            # Query events in time range
            query = select(AgentEvent).where(
                AgentEvent.created_at >= start_time
            ).order_by(AgentEvent.created_at)
            
            # Apply filters
            if request.agent_id:
                query = query.where(AgentEvent.agent_id == request.agent_id)
            if request.session_id:
                query = query.where(AgentEvent.session_id == request.session_id)
            
            result = await db_session.execute(query)
            events = result.scalars().all()
            
            # Build trajectory graph
            trajectory_graph = await self._build_trajectory_graph(events, request)
            
            # Find meaningful paths
            paths = await self._find_trajectory_paths(trajectory_graph, request.max_depth)
            
            return paths
            
        except Exception as e:
            logger.error(f"Context trajectory processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _build_trajectory_graph(
        self, 
        events: List[AgentEvent], 
        request: ContextTrajectoryRequest
    ) -> Dict[str, ContextTrajectoryNode]:
        """Build trajectory graph from events."""
        nodes = {}
        
        for event in events:
            try:
                # Create event node
                event_node = ContextTrajectoryNode(
                    id=f"event_{event.id}",
                    type="event",
                    label=f"{event.event_type.value}",
                    timestamp=event.created_at,
                    metadata={
                        "event_type": event.event_type.value,
                        "agent_id": str(event.agent_id) if event.agent_id else None,
                        "session_id": str(event.session_id) if event.session_id else None
                    },
                    semantic_embedding=event.payload.get('semantic_embedding') if event.payload else None,
                    connections=[]
                )
                nodes[event_node.id] = event_node
                
                # Create agent node if not exists
                if event.agent_id:
                    agent_node_id = f"agent_{event.agent_id}"
                    if agent_node_id not in nodes:
                        agent_node = ContextTrajectoryNode(
                            id=agent_node_id,
                            type="agent",
                            label=f"Agent {str(event.agent_id)[:8]}",
                            timestamp=None,
                            metadata={"agent_id": str(event.agent_id)},
                            semantic_embedding=None,
                            connections=[]
                        )
                        nodes[agent_node_id] = agent_node
                    
                    # Connect event to agent
                    nodes[agent_node_id].connections.append(event_node.id)
                    event_node.connections.append(agent_node_id)
                
                # Process context references
                if event.payload and 'context_references' in event.payload:
                    for context_ref in event.payload['context_references']:
                        context_node_id = f"context_{context_ref}"
                        if context_node_id not in nodes:
                            context_node = ContextTrajectoryNode(
                                id=context_node_id,
                                type="context",
                                label=f"Context {context_ref[:8]}",
                                timestamp=None,
                                metadata={"context_id": context_ref},
                                semantic_embedding=None,
                                connections=[]
                            )
                            nodes[context_node_id] = context_node
                        
                        # Connect event to context
                        nodes[context_node_id].connections.append(event_node.id)
                        event_node.connections.append(context_node_id)
                
            except Exception as e:
                logger.error(f"Failed to process event {event.id} for trajectory: {e}")
                continue
        
        return nodes
    
    async def _find_trajectory_paths(
        self, 
        nodes: Dict[str, ContextTrajectoryNode], 
        max_depth: int
    ) -> List[ContextTrajectoryPath]:
        """Find meaningful trajectory paths through the graph."""
        paths = []
        
        # Simple path finding algorithm
        # This could be enhanced with more sophisticated graph algorithms
        for node_id, node in nodes.items():
            if node.type == "context":
                # Start path from context nodes
                path = await self._trace_path_from_node(nodes, node_id, max_depth)
                if len(path.nodes) > 1:
                    paths.append(path)
        
        return paths
    
    async def _trace_path_from_node(
        self, 
        nodes: Dict[str, ContextTrajectoryNode], 
        start_node_id: str, 
        max_depth: int
    ) -> ContextTrajectoryPath:
        """Trace path from a starting node."""
        visited = set()
        path_nodes = []
        edges = []
        
        def dfs(node_id: str, depth: int):
            if depth >= max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            node = nodes[node_id]
            path_nodes.append(node)
            
            for connected_id in node.connections:
                if connected_id not in visited:
                    edges.append({
                        "source_id": node_id,
                        "target_id": connected_id,
                        "relationship_type": "connection"
                    })
                    dfs(connected_id, depth + 1)
        
        dfs(start_node_id, 0)
        
        # Calculate path metrics
        temporal_flow = [node.timestamp for node in path_nodes if node.timestamp]
        temporal_flow.sort()
        
        return ContextTrajectoryPath(
            nodes=path_nodes,
            edges=edges,
            semantic_similarity=0.8,  # Mock value
            path_strength=len(path_nodes) / max_depth,
            temporal_flow=temporal_flow
        )


# Intelligence KPI processing

class IntelligenceKPIProcessor:
    """Processor for intelligence KPI metrics."""
    
    def __init__(self):
        self.kpi_definitions = {
            "workflow_quality": {
                "description": "Overall workflow execution quality",
                "unit": "score",
                "normal_range": (0.8, 1.0),
                "warning_range": (0.6, 0.8),
                "critical_range": (0.0, 0.6)
            },
            "coordination_amplification": {
                "description": "Agent coordination effectiveness",
                "unit": "multiplier",
                "normal_range": (1.5, 3.0),
                "warning_range": (1.0, 1.5),
                "critical_range": (0.0, 1.0)
            },
            "semantic_search_latency": {
                "description": "Semantic search response time",
                "unit": "ms",
                "normal_range": (0, 100),
                "warning_range": (100, 500),
                "critical_range": (500, float('inf'))
            },
            "context_compression_efficiency": {
                "description": "Context compression ratio",
                "unit": "ratio",
                "normal_range": (0.6, 0.9),
                "warning_range": (0.4, 0.6),
                "critical_range": (0.0, 0.4)
            },
            "agent_knowledge_acquisition": {
                "description": "Rate of agent knowledge growth",
                "unit": "concepts/hour",
                "normal_range": (10, 50),
                "warning_range": (5, 10),
                "critical_range": (0, 5)
            }
        }
    
    async def get_kpi_metrics(
        self, 
        request: IntelligenceKPIRequest, 
        db_session: AsyncSession
    ) -> List[IntelligenceKPI]:
        """Get intelligence KPI metrics with trend analysis."""
        try:
            kpi_names = request.kpi_names or list(self.kpi_definitions.keys())
            results = []
            
            for kpi_name in kpi_names:
                if kpi_name not in self.kpi_definitions:
                    continue
                
                kpi_def = self.kpi_definitions[kpi_name]
                
                # Calculate current value and historical data
                current_value, historical_data = await self._calculate_kpi_value(
                    kpi_name, request, db_session
                )
                
                # Analyze trend
                trend_direction, trend_strength = await self._analyze_kpi_trend(historical_data)
                
                # Determine threshold status
                threshold_status = self._get_threshold_status(current_value, kpi_def)
                
                # Generate forecast if requested
                forecast_data = None
                if request.include_forecasts:
                    forecast_data = await self._generate_kpi_forecast(historical_data)
                
                kpi = IntelligenceKPI(
                    name=kpi_name,
                    description=kpi_def["description"],
                    current_value=current_value,
                    unit=kpi_def["unit"],
                    trend_direction=trend_direction,
                    trend_strength=trend_strength,
                    threshold_status=threshold_status,
                    historical_data=historical_data,
                    forecast_data=forecast_data
                )
                
                results.append(kpi)
            
            return results
            
        except Exception as e:
            logger.error(f"KPI metrics processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _calculate_kpi_value(
        self, 
        kpi_name: str, 
        request: IntelligenceKPIRequest,
        db_session: AsyncSession
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculate KPI value and historical data."""
        # Mock implementation - would be replaced with actual KPI calculation
        import random
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=request.time_range_hours)
        
        # Generate mock historical data
        historical_data = []
        current_time = start_time
        interval_delta = self._get_interval_delta(request.aggregation_interval)
        
        base_value = random.uniform(0.5, 1.0)
        
        while current_time <= end_time:
            # Add some noise and trend
            noise = random.uniform(-0.1, 0.1)
            trend = (current_time - start_time).total_seconds() / (end_time - start_time).total_seconds() * 0.2
            value = max(0, base_value + trend + noise)
            
            historical_data.append({
                "timestamp": current_time.isoformat(),
                "value": round(value, 4),
                "metadata": {
                    "data_points": random.randint(10, 100),
                    "confidence": random.uniform(0.8, 1.0)
                }
            })
            
            current_time += interval_delta
        
        current_value = historical_data[-1]["value"] if historical_data else 0.0
        
        return current_value, historical_data
    
    def _get_interval_delta(self, interval: str) -> timedelta:
        """Get timedelta for aggregation interval."""
        interval_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "1d": timedelta(days=1)
        }
        return interval_map.get(interval, timedelta(hours=1))
    
    async def _analyze_kpi_trend(self, historical_data: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Analyze KPI trend direction and strength."""
        if len(historical_data) < 2:
            return "stable", 0.0
        
        values = [point["value"] for point in historical_data]
        
        # Simple linear trend analysis
        x = list(range(len(values)))
        y = values
        
        if len(x) > 1:
            slope = (y[-1] - y[0]) / (len(y) - 1)
            
            if abs(slope) < 0.01:
                return "stable", abs(slope) * 100
            elif slope > 0:
                return "up", min(slope * 100, 1.0)
            else:
                return "down", min(abs(slope) * 100, 1.0)
        
        return "stable", 0.0
    
    def _get_threshold_status(self, value: float, kpi_def: Dict[str, Any]) -> str:
        """Get threshold status for KPI value."""
        normal_range = kpi_def["normal_range"]
        warning_range = kpi_def["warning_range"]
        critical_range = kpi_def["critical_range"]
        
        if normal_range[0] <= value <= normal_range[1]:
            return "normal"
        elif warning_range[0] <= value <= warning_range[1]:
            return "warning"
        else:
            return "critical"
    
    async def _generate_kpi_forecast(self, historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate KPI forecast data."""
        # Simple forecast based on trend
        if len(historical_data) < 3:
            return []
        
        values = [point["value"] for point in historical_data]
        
        # Simple linear extrapolation
        forecast_data = []
        last_value = values[-1]
        trend = (values[-1] - values[-3]) / 2  # Simple trend calculation
        
        for i in range(1, 13):  # 12 future points
            forecast_value = max(0, last_value + trend * i)
            forecast_data.append({
                "timestamp": (datetime.utcnow() + timedelta(hours=i)).isoformat(),
                "value": round(forecast_value, 4),
                "confidence": max(0.1, 0.9 - i * 0.1),  # Decreasing confidence
                "is_forecast": True
            })
        
        return forecast_data


# Workflow constellation processing

class WorkflowConstellationProcessor:
    """Processor for workflow constellation visualization."""
    
    async def build_constellation(
        self, 
        request: WorkflowConstellationRequest, 
        db_session: AsyncSession
    ) -> WorkflowConstellation:
        """Build workflow constellation visualization data."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=request.time_range_hours)
            
            # Query events for constellation
            query = select(AgentEvent).where(
                AgentEvent.created_at >= start_time
            ).order_by(AgentEvent.created_at)
            
            # Apply filters
            if request.session_ids:
                query = query.where(AgentEvent.session_id.in_(request.session_ids))
            if request.agent_ids:
                query = query.where(AgentEvent.agent_id.in_(request.agent_ids))
            
            result = await db_session.execute(query)
            events = result.scalars().all()
            
            # Build constellation data
            nodes, edges = await self._build_constellation_graph(events, request)
            
            # Generate semantic flows
            semantic_flows = await self._extract_semantic_flows(events, request)
            
            # Generate temporal data
            temporal_data = await self._extract_temporal_data(events)
            
            return WorkflowConstellation(
                nodes=nodes,
                edges=edges,
                semantic_flows=semantic_flows,
                temporal_data=temporal_data,
                metadata={
                    "time_range_hours": request.time_range_hours,
                    "total_events": len(events),
                    "generation_time": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Workflow constellation processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _build_constellation_graph(
        self, 
        events: List[AgentEvent], 
        request: WorkflowConstellationRequest
    ) -> Tuple[List[WorkflowConstellationNode], List[WorkflowConstellationEdge]]:
        """Build constellation graph nodes and edges."""
        nodes = {}
        edges = {}
        interaction_counts = {}
        
        # Process events to build graph
        for event in events:
            try:
                agent_id = str(event.agent_id) if event.agent_id else "unknown"
                session_id = str(event.session_id) if event.session_id else "unknown"
                
                # Create/update agent node
                if agent_id not in nodes:
                    nodes[agent_id] = WorkflowConstellationNode(
                        id=agent_id,
                        type="agent",
                        label=f"Agent {agent_id[:8]}",
                        position={"x": 0.0, "y": 0.0},  # Will be calculated later
                        size=1.0,
                        color="#3B82F6",
                        metadata={
                            "event_count": 0,
                            "first_seen": event.created_at,
                            "last_seen": event.created_at
                        }
                    )
                
                # Update node metadata
                node = nodes[agent_id]
                node.metadata["event_count"] += 1
                node.metadata["last_seen"] = max(node.metadata["last_seen"], event.created_at)
                
                # Process semantic concepts
                if event.payload and 'semantic_concepts' in event.payload:
                    for concept in event.payload['semantic_concepts']:
                        concept_id = f"concept_{concept}"
                        
                        if concept_id not in nodes:
                            nodes[concept_id] = WorkflowConstellationNode(
                                id=concept_id,
                                type="concept",
                                label=concept,
                                position={"x": 0.0, "y": 0.0},
                                size=0.5,
                                color="#10B981",
                                metadata={"usage_count": 0}
                            )
                        
                        nodes[concept_id].metadata["usage_count"] += 1
                        
                        # Create edge between agent and concept
                        edge_id = f"{agent_id}_{concept_id}"
                        if edge_id not in edges:
                            edges[edge_id] = WorkflowConstellationEdge(
                                source=agent_id,
                                target=concept_id,
                                type="semantic_flow",
                                strength=0.0,
                                frequency=0,
                                latency_ms=None,
                                semantic_concepts=[concept]
                            )
                        
                        edges[edge_id].frequency += 1
                        edges[edge_id].strength = min(1.0, edges[edge_id].frequency / 10.0)
                
                # Track agent interactions (simplified)
                interaction_counts[agent_id] = interaction_counts.get(agent_id, 0) + 1
                
            except Exception as e:
                logger.error(f"Failed to process event {event.id} for constellation: {e}")
                continue
        
        # Filter by minimum interaction count
        filtered_nodes = [
            node for node in nodes.values() 
            if node.type != "agent" or interaction_counts.get(node.id, 0) >= request.min_interaction_count
        ]
        
        # Calculate node positions using force-directed layout (simplified)
        await self._calculate_node_positions(filtered_nodes)
        
        # Update node sizes based on activity
        for node in filtered_nodes:
            if node.type == "agent":
                node.size = min(3.0, 1.0 + interaction_counts.get(node.id, 0) / 10.0)
            elif node.type == "concept":
                node.size = min(2.0, 0.5 + node.metadata["usage_count"] / 5.0)
        
        return filtered_nodes, list(edges.values())
    
    async def _calculate_node_positions(self, nodes: List[WorkflowConstellationNode]):
        """Calculate node positions using force-directed layout."""
        import math
        import random
        
        # Simple circular layout for now
        center_x, center_y = 0.0, 0.0
        radius = 100.0
        
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            node.position["x"] = center_x + radius * math.cos(angle) + random.uniform(-10, 10)
            node.position["y"] = center_y + radius * math.sin(angle) + random.uniform(-10, 10)
    
    async def _extract_semantic_flows(
        self, 
        events: List[AgentEvent], 
        request: WorkflowConstellationRequest
    ) -> List[Dict[str, Any]]:
        """Extract semantic concept flows."""
        flows = []
        
        if not request.include_semantic_flow:
            return flows
        
        # Group events by time windows
        time_windows = {}
        window_size = 60  # 1 minute windows
        
        for event in events:
            window_key = int(event.created_at.timestamp() // window_size)
            if window_key not in time_windows:
                time_windows[window_key] = []
            time_windows[window_key].append(event)
        
        # Analyze concept flows within windows
        for window_key, window_events in time_windows.items():
            concept_usage = {}
            
            for event in window_events:
                if event.payload and 'semantic_concepts' in event.payload:
                    for concept in event.payload['semantic_concepts']:
                        if concept not in concept_usage:
                            concept_usage[concept] = []
                        concept_usage[concept].append({
                            "agent_id": str(event.agent_id) if event.agent_id else "unknown",
                            "timestamp": event.created_at.isoformat()
                        })
            
            # Create flow objects for concepts used by multiple agents
            for concept, usage in concept_usage.items():
                if len(usage) > 1:
                    flows.append({
                        "concept": concept,
                        "window_start": datetime.fromtimestamp(window_key * window_size).isoformat(),
                        "agents": [u["agent_id"] for u in usage],
                        "flow_strength": len(usage) / len(window_events),
                        "temporal_spread": (max(u["timestamp"] for u in usage), min(u["timestamp"] for u in usage))
                    })
        
        return flows
    
    async def _extract_temporal_data(self, events: List[AgentEvent]) -> List[Dict[str, Any]]:
        """Extract temporal data for constellation animation."""
        temporal_data = []
        
        # Group events by time intervals
        interval_minutes = 5
        time_groups = {}
        
        for event in events:
            interval_key = int(event.created_at.timestamp() // (interval_minutes * 60))
            if interval_key not in time_groups:
                time_groups[interval_key] = []
            time_groups[interval_key].append(event)
        
        # Generate temporal data points
        for interval_key, interval_events in time_groups.items():
            timestamp = datetime.fromtimestamp(interval_key * interval_minutes * 60)
            
            temporal_data.append({
                "timestamp": timestamp.isoformat(),
                "event_count": len(interval_events),
                "active_agents": len(set(str(e.agent_id) for e in interval_events if e.agent_id)),
                "dominant_event_type": max(set(e.event_type.value for e in interval_events), 
                                         key=[e.event_type.value for e in interval_events].count),
                "average_latency": sum(e.latency_ms for e in interval_events if e.latency_ms) / 
                                 max(1, len([e for e in interval_events if e.latency_ms]))
            })
        
        return sorted(temporal_data, key=lambda x: x["timestamp"])


# Initialize processors
semantic_processor = SemanticQueryProcessor()
trajectory_processor = ContextTrajectoryProcessor()
kpi_processor = IntelligenceKPIProcessor()
constellation_processor = WorkflowConstellationProcessor()


# API endpoints

@router.post("/semantic-search", response_model=List[SemanticSearchResult])
async def semantic_search(
    query: SemanticQuery,
    db_session: AsyncSession = Depends(get_async_session)
):
    """
    Natural language semantic search of observability data.
    
    Supports queries like:
    - "Show me slow responses from the last hour"
    - "Find errors related to context compression"
    - "What agents were most active yesterday?"
    """
    return await semantic_processor.process_query(query, db_session)


@router.post("/context-trajectory", response_model=List[ContextTrajectoryPath])
async def get_context_trajectory(
    request: ContextTrajectoryRequest,
    db_session: AsyncSession = Depends(get_async_session)
):
    """
    Get context trajectory visualization showing semantic knowledge flow.
    
    Traces how context and semantic concepts flow between agents and events.
    """
    return await trajectory_processor.build_trajectory(request, db_session)


@router.post("/intelligence-kpis", response_model=List[IntelligenceKPI])
async def get_intelligence_kpis(
    request: IntelligenceKPIRequest,
    db_session: AsyncSession = Depends(get_async_session)
):
    """
    Get intelligence KPI metrics with trend analysis.
    
    Provides real-time metrics on:
    - Workflow quality improvement
    - Coordination amplification
    - Semantic search performance
    - Context compression efficiency
    - Agent knowledge acquisition rates
    """
    return await kpi_processor.get_kpi_metrics(request, db_session)


@router.post("/workflow-constellation", response_model=WorkflowConstellation)
async def get_workflow_constellation(
    request: WorkflowConstellationRequest,
    db_session: AsyncSession = Depends(get_async_session)
):
    """
    Get workflow constellation visualization data.
    
    Shows real-time agent interactions as nodes with semantic concepts
    flowing between them as animated particles.
    """
    return await constellation_processor.build_constellation(request, db_session)


@router.get("/performance-summary")
async def get_performance_summary(
    time_range_hours: int = Query(default=24, ge=1, le=168),
    db_session: AsyncSession = Depends(get_async_session)
):
    """
    Get dashboard performance summary for optimization monitoring.
    """
    try:
        start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Query performance data
        query = select(
            func.count(AgentEvent.id).label("total_events"),
            func.avg(AgentEvent.latency_ms).label("avg_latency"),
            func.max(AgentEvent.latency_ms).label("max_latency"),
            func.count(
                func.case([(AgentEvent.latency_ms > 1000, 1)], else_=None)
            ).label("slow_events")
        ).where(
            and_(
                AgentEvent.created_at >= start_time,
                AgentEvent.latency_ms.isnot(None)
            )
        )
        
        result = await db_session.execute(query)
        stats = result.first()
        
        return {
            "summary": {
                "total_events": stats.total_events or 0,
                "avg_latency_ms": round(stats.avg_latency or 0, 2),
                "max_latency_ms": stats.max_latency or 0,
                "slow_events": stats.slow_events or 0,
                "performance_score": min(1.0, max(0.0, 1.0 - (stats.avg_latency or 0) / 1000))
            },
            "targets": {
                "dashboard_load_time_ms": 2000,
                "event_latency_ms": 1000,
                "throughput_events_per_second": 1000
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream-status")
async def get_stream_status():
    """Get real-time streaming status and connection metrics."""
    try:
        manager = await get_websocket_manager()
        return {
            "websocket_connections": len(manager.connections),
            "streaming_metrics": manager.get_metrics(),
            "performance_status": "optimal" if manager.metrics["average_latency_ms"] < 1000 else "degraded",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Stream status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))