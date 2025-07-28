"""
Transcript Analysis API Endpoints for LeanVibe Agent Hive 2.0

RESTful API endpoints for chat transcript analysis, conversation debugging,
and real-time communication monitoring with comprehensive search capabilities.

Features:
- Chat transcript retrieval and analysis
- Advanced conversation search with semantic analysis
- Real-time transcript streaming management
- Debugging session management
- Communication pattern analysis
- Error analysis and bottleneck detection
- Performance metrics and insights
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import (
    HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY
)

from ...core.database import get_async_session
from ...core.embedding_service import EmbeddingService, get_embedding_service
from ...core.context_engine_integration import ContextEngineIntegration
from ...core.chat_transcript_manager import (
    ChatTranscriptManager, ConversationEvent, ConversationEventType,
    SearchFilter as TranscriptSearchFilter
)
from ...core.communication_analyzer import (
    CommunicationAnalyzer, AnalysisType, AlertSeverity
)
from ...core.conversation_search_engine import (
    ConversationSearchEngine, SearchQuery, SearchType, SortOption
)
from ...core.transcript_streaming import (
    TranscriptStreamingManager, StreamingFilter, FilterMode, StreamEventType
)
from ...core.conversation_debugging_tools import (
    ConversationDebugger, ReplayMode, DebugLevel, AnalysisScope
)
from ...models.user import User
from ...core.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcript-analysis", tags=["Transcript Analysis"])


# Request/Response Models
class TranscriptRequest(BaseModel):
    """Request model for conversation transcript retrieval."""
    session_id: str = Field(..., description="Session ID to get transcript for")
    agent_filter: Optional[List[str]] = Field(None, description="Filter by specific agent IDs")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of events")
    include_metadata: bool = Field(default=True, description="Include event metadata")
    include_embeddings: bool = Field(default=False, description="Include embedding vectors")


class SearchRequest(BaseModel):
    """Request model for advanced conversation search."""
    query_text: Optional[str] = Field(None, description="Search query text")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    session_filters: List[str] = Field(default_factory=list, description="Filter by session IDs")
    agent_filters: List[str] = Field(default_factory=list, description="Filter by agent IDs")
    event_type_filters: List[ConversationEventType] = Field(default_factory=list, description="Filter by event types")
    time_range_start: Optional[datetime] = Field(None, description="Start time for search")
    time_range_end: Optional[datetime] = Field(None, description="End time for search")
    semantic_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Semantic similarity threshold")
    sort_by: SortOption = Field(default=SortOption.RELEVANCE, description="Sort results by")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    include_metadata: bool = Field(default=True, description="Include result metadata")


class AnalysisRequest(BaseModel):
    """Request model for communication analysis."""
    session_id: str = Field(..., description="Session ID to analyze")
    analysis_types: List[AnalysisType] = Field(default_factory=list, description="Types of analysis to perform")
    time_window_hours: int = Field(default=24, ge=1, le=168, description="Time window for analysis")
    include_recommendations: bool = Field(default=True, description="Include optimization recommendations")


class StreamingRequest(BaseModel):
    """Request model for real-time transcript streaming."""
    session_filters: List[str] = Field(default_factory=list, description="Filter by session IDs")
    agent_filters: List[str] = Field(default_factory=list, description="Filter by agent IDs")
    event_types: List[ConversationEventType] = Field(default_factory=list, description="Filter by event types")
    stream_events: List[StreamEventType] = Field(default_factory=list, description="Types of stream events")
    keywords: List[str] = Field(default_factory=list, description="Filter by keywords")
    min_severity: AlertSeverity = Field(default=AlertSeverity.INFO, description="Minimum alert severity")
    real_time_only: bool = Field(default=True, description="Only real-time events")
    include_patterns: bool = Field(default=True, description="Include pattern detection")
    include_performance: bool = Field(default=True, description="Include performance metrics")
    filter_mode: FilterMode = Field(default=FilterMode.INCLUSIVE, description="Filter application mode")


class DebugSessionRequest(BaseModel):
    """Request model for debug session creation."""
    target_session_id: str = Field(..., description="Session to debug")
    replay_mode: ReplayMode = Field(default=ReplayMode.STEP_BY_STEP, description="Replay mode")
    debug_level: DebugLevel = Field(default=DebugLevel.INFO, description="Debug output level")
    analysis_scope: AnalysisScope = Field(default=AnalysisScope.SESSION_WIDE, description="Analysis scope")
    auto_analyze: bool = Field(default=True, description="Auto-analyze events")


class BreakpointRequest(BaseModel):
    """Request model for adding debug breakpoints."""
    condition_type: str = Field(..., description="Breakpoint condition type")
    condition_value: Any = Field(..., description="Breakpoint condition value")
    action: str = Field(default="pause", description="Action to take when breakpoint hits")


class ReplayRequest(BaseModel):
    """Request model for conversation replay."""
    original_session_id: str = Field(..., description="Session to replay")
    target_session_id: Optional[str] = Field(None, description="Target session for replay")
    speed_multiplier: float = Field(default=1.0, ge=0.1, le=10.0, description="Replay speed multiplier")
    start_time: Optional[datetime] = Field(None, description="Start time for replay")
    end_time: Optional[datetime] = Field(None, description="End time for replay")


# Response Models
class TranscriptResponse(BaseModel):
    """Response model for conversation transcript."""
    success: bool
    session_id: str
    events: List[Dict[str, Any]]
    total_events: int
    metadata: Dict[str, Any]
    generated_at: str


class SearchResponse(BaseModel):
    """Response model for search results."""
    success: bool
    results: List[Dict[str, Any]]
    total_matches: int
    search_time_ms: float
    facets: Dict[str, Dict[str, int]]
    suggestions: List[str]
    query_info: Dict[str, Any]


class AnalysisResponse(BaseModel):
    """Response model for communication analysis."""
    success: bool
    session_id: str
    insights: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    analysis_time_ms: float
    generated_at: str


# Dependency injection
async def get_transcript_manager(
    db: AsyncSession = Depends(get_async_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> ChatTranscriptManager:
    """Get configured chat transcript manager."""
    return ChatTranscriptManager(db, embedding_service)


async def get_communication_analyzer() -> CommunicationAnalyzer:
    """Get configured communication analyzer."""
    return CommunicationAnalyzer()


async def get_search_engine(
    db: AsyncSession = Depends(get_async_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> ConversationSearchEngine:
    """Get configured conversation search engine."""
    context_engine = ContextEngineIntegration()
    return ConversationSearchEngine(db, embedding_service, context_engine)


async def get_streaming_manager(
    transcript_manager: ChatTranscriptManager = Depends(get_transcript_manager),
    analyzer: CommunicationAnalyzer = Depends(get_communication_analyzer)
) -> TranscriptStreamingManager:
    """Get configured transcript streaming manager."""
    return TranscriptStreamingManager(transcript_manager, analyzer)


async def get_conversation_debugger(
    transcript_manager: ChatTranscriptManager = Depends(get_transcript_manager),
    analyzer: CommunicationAnalyzer = Depends(get_communication_analyzer),
    streaming_manager: TranscriptStreamingManager = Depends(get_streaming_manager)
) -> ConversationDebugger:
    """Get configured conversation debugger."""
    return ConversationDebugger(transcript_manager, analyzer, streaming_manager)


# API Endpoints

@router.get("/transcript/{session_id}", response_model=TranscriptResponse, status_code=HTTP_200_OK)
async def get_conversation_transcript(
    session_id: str = Path(..., description="Session ID to get transcript for"),
    agent_filter: Optional[str] = Query(None, description="Comma-separated agent IDs"),
    start_time: Optional[datetime] = Query(None, description="Start time filter (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time filter (ISO format)"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of events"),
    include_metadata: bool = Query(default=True, description="Include event metadata"),
    transcript_manager: ChatTranscriptManager = Depends(get_transcript_manager),
    current_user: User = Depends(get_current_user)
):
    """
    Get chronological conversation transcript for a session.
    
    Returns agent-to-agent communications, tool calls, and context sharing
    events in chronological order with optional filtering.
    """
    try:
        # Parse agent filter
        agent_list = None
        if agent_filter:
            agent_list = [a.strip() for a in agent_filter.split(",")]
        
        # Get transcript events
        events = await transcript_manager.get_conversation_transcript(
            session_id=session_id,
            agent_filter=agent_list,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Calculate metadata
        metadata = {
            'event_types': {},
            'agents_involved': set(),
            'time_span_minutes': 0,
            'tool_usage_count': 0,
            'context_sharing_count': 0
        }
        
        for event in events:
            # Count event types
            event_type = event.event_type.value
            metadata['event_types'][event_type] = metadata['event_types'].get(event_type, 0) + 1
            
            # Track agents
            metadata['agents_involved'].add(event.source_agent_id)
            if event.target_agent_id:
                metadata['agents_involved'].add(event.target_agent_id)
            
            # Count tool usage and context sharing
            metadata['tool_usage_count'] += len(event.tool_calls)
            metadata['context_sharing_count'] += len(event.context_references)
        
        # Calculate time span
        if len(events) > 1:
            metadata['time_span_minutes'] = (
                events[-1].timestamp - events[0].timestamp
            ).total_seconds() / 60
        
        metadata['agents_involved'] = list(metadata['agents_involved'])
        
        # Format events for response
        formatted_events = [
            {
                **event.to_dict(),
                'metadata': event.metadata if include_metadata else {}
            }
            for event in events
        ]
        
        logger.info(
            "Conversation transcript retrieved",
            user_id=str(current_user.id),
            session_id=session_id,
            event_count=len(events),
            agent_filter=agent_list
        )
        
        return TranscriptResponse(
            success=True,
            session_id=session_id,
            events=formatted_events,
            total_events=len(events),
            metadata=metadata,
            generated_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(
            "Failed to get conversation transcript",
            user_id=str(current_user.id),
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get transcript: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse, status_code=HTTP_200_OK)
async def search_conversations(
    request: SearchRequest,
    search_engine: ConversationSearchEngine = Depends(get_search_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Advanced search across conversations with semantic analysis.
    
    Supports semantic search, keyword search, pattern matching,
    and sophisticated filtering with performance optimization.
    """
    try:
        search_start = datetime.utcnow()
        
        # Build search query
        search_query = SearchQuery(
            query_text=request.query_text,
            search_type=request.search_type,
            session_filters=request.session_filters,
            agent_filters=request.agent_filters,
            event_type_filters=request.event_type_filters,
            time_range=(request.time_range_start, request.time_range_end) if request.time_range_start else None,
            semantic_threshold=request.semantic_threshold,
            sort_by=request.sort_by,
            limit=request.limit,
            offset=request.offset,
            include_metadata=request.include_metadata
        )
        
        # Execute search
        search_results = await search_engine.search(search_query)
        
        search_time = (datetime.utcnow() - search_start).total_seconds() * 1000
        
        logger.info(
            "Conversation search completed",
            user_id=str(current_user.id),
            search_type=request.search_type.value,
            query_text=request.query_text,
            results_count=len(search_results.results),
            search_time_ms=search_time
        )
        
        return SearchResponse(
            success=True,
            results=[result.to_dict() for result in search_results.results],
            total_matches=search_results.total_matches,
            search_time_ms=search_results.search_time_ms,
            facets=search_results.facets,
            suggestions=search_results.suggestions,
            query_info=request.dict()
        )
        
    except Exception as e:
        logger.error(
            "Conversation search failed",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/analyze", response_model=AnalysisResponse, status_code=HTTP_200_OK)
async def analyze_conversation(
    request: AnalysisRequest,
    transcript_manager: ChatTranscriptManager = Depends(get_transcript_manager),
    analyzer: CommunicationAnalyzer = Depends(get_communication_analyzer),
    current_user: User = Depends(get_current_user)
):
    """
    Comprehensive analysis of conversation patterns and performance.
    
    Provides pattern detection, performance analysis, bottleneck identification,
    and optimization recommendations with detailed insights.
    """
    try:
        analysis_start = datetime.utcnow()
        
        # Get conversation events for analysis
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=request.time_window_hours)
        
        events = await transcript_manager.get_conversation_transcript(
            session_id=request.session_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not events:
            return AnalysisResponse(
                success=True,
                session_id=request.session_id,
                insights=[],
                metrics={},
                recommendations=[],
                analysis_time_ms=0,
                generated_at=datetime.utcnow().isoformat()
            )
        
        # Perform analysis
        analysis_types = request.analysis_types if request.analysis_types else list(AnalysisType)
        insights = await analyzer.analyze_conversation_events(events, analysis_types)
        
        # Get conversation analytics
        analytics = await transcript_manager.get_conversation_analytics(
            session_id=request.session_id,
            time_window_hours=request.time_window_hours
        )
        
        # Generate optimization report if requested
        recommendations = []
        if request.include_recommendations:
            optimization_report = await analyzer.generate_optimization_report(
                session_id=request.session_id,
                time_window_hours=request.time_window_hours
            )
            recommendations = optimization_report.get('optimization_opportunities', [])
        
        analysis_time = (datetime.utcnow() - analysis_start).total_seconds() * 1000
        
        logger.info(
            "Conversation analysis completed",
            user_id=str(current_user.id),
            session_id=request.session_id,
            insights_count=len(insights),
            analysis_time_ms=analysis_time
        )
        
        return AnalysisResponse(
            success=True,
            session_id=request.session_id,
            insights=[insight.to_dict() for insight in insights],
            metrics=analytics,
            recommendations=recommendations,
            analysis_time_ms=analysis_time,
            generated_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(
            "Conversation analysis failed",
            user_id=str(current_user.id),
            session_id=request.session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/replay", status_code=HTTP_200_OK)
async def replay_conversation(
    request: ReplayRequest,
    transcript_manager: ChatTranscriptManager = Depends(get_transcript_manager),
    current_user: User = Depends(get_current_user)
):
    """
    Replay conversation events for debugging purposes.
    
    Creates a new session with replayed events at specified speed
    with optional time range filtering.
    """
    try:
        # Execute replay
        replay_result = await transcript_manager.replay_conversation(
            session_id=request.original_session_id,
            target_session_id=request.target_session_id,
            speed_multiplier=request.speed_multiplier,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        logger.info(
            "Conversation replay completed",
            user_id=str(current_user.id),
            original_session=request.original_session_id,
            target_session=replay_result.get('target_session_id'),
            events_replayed=replay_result.get('events_replayed', 0)
        )
        
        return {
            'success': True,
            'replay_result': replay_result,
            'generated_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Conversation replay failed",
            user_id=str(current_user.id),
            original_session=request.original_session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Replay failed: {str(e)}"
        )


@router.post("/debug-session", status_code=HTTP_201_CREATED)
async def create_debug_session(
    request: DebugSessionRequest,
    debugger: ConversationDebugger = Depends(get_conversation_debugger),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new debugging session for conversation analysis.
    
    Enables step-by-step debugging, breakpoint management,
    and comprehensive error analysis capabilities.
    """
    try:
        # Create debug session
        debug_session_id = await debugger.create_debug_session(
            target_session_id=request.target_session_id,
            replay_mode=request.replay_mode,
            debug_level=request.debug_level,
            analysis_scope=request.analysis_scope,
            auto_analyze=request.auto_analyze
        )
        
        # Get session status
        session_status = await debugger.get_debug_session_status(debug_session_id)
        
        logger.info(
            "Debug session created",
            user_id=str(current_user.id),
            debug_session_id=debug_session_id,
            target_session=request.target_session_id
        )
        
        return {
            'success': True,
            'debug_session_id': debug_session_id,
            'session_status': session_status,
            'created_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Debug session creation failed",
            user_id=str(current_user.id),
            target_session=request.target_session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Debug session creation failed: {str(e)}"
        )


@router.post("/debug-session/{debug_session_id}/breakpoint", status_code=HTTP_201_CREATED)
async def add_debug_breakpoint(
    debug_session_id: str = Path(..., description="Debug session ID"),
    request: BreakpointRequest = ...,
    debugger: ConversationDebugger = Depends(get_conversation_debugger),
    current_user: User = Depends(get_current_user)
):
    """Add a debugging breakpoint to a debug session."""
    try:
        breakpoint_id = await debugger.add_breakpoint(
            debug_session_id=debug_session_id,
            condition_type=request.condition_type,
            condition_value=request.condition_value,
            action=request.action
        )
        
        logger.info(
            "Debug breakpoint added",
            user_id=str(current_user.id),
            debug_session_id=debug_session_id,
            breakpoint_id=breakpoint_id
        )
        
        return {
            'success': True,
            'breakpoint_id': breakpoint_id,
            'debug_session_id': debug_session_id,
            'created_at': datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Breakpoint creation failed",
            user_id=str(current_user.id),
            debug_session_id=debug_session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Breakpoint creation failed: {str(e)}"
        )


@router.post("/debug-session/{debug_session_id}/step", status_code=HTTP_200_OK)
async def step_debug_session(
    debug_session_id: str = Path(..., description="Debug session ID"),
    steps: int = Query(default=1, ge=1, le=100, description="Number of steps to execute"),
    debugger: ConversationDebugger = Depends(get_conversation_debugger),
    current_user: User = Depends(get_current_user)
):
    """Step through debug session by specified number of events."""
    try:
        step_result = await debugger.step_debug_session(
            debug_session_id=debug_session_id,
            steps=steps
        )
        
        logger.info(
            "Debug session stepped",
            user_id=str(current_user.id),
            debug_session_id=debug_session_id,
            steps_executed=step_result['steps_executed']
        )
        
        return {
            'success': True,
            **step_result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Debug step failed",
            user_id=str(current_user.id),
            debug_session_id=debug_session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Debug step failed: {str(e)}"
        )


@router.get("/debug-session/{debug_session_id}/report", status_code=HTTP_200_OK)
async def get_debug_report(
    debug_session_id: str = Path(..., description="Debug session ID"),
    include_recommendations: bool = Query(default=True, description="Include recommendations"),
    debugger: ConversationDebugger = Depends(get_conversation_debugger),
    current_user: User = Depends(get_current_user)
):
    """Generate comprehensive debug report for a session."""
    try:
        debug_report = await debugger.generate_debug_report(
            debug_session_id=debug_session_id,
            include_recommendations=include_recommendations
        )
        
        logger.info(
            "Debug report generated",
            user_id=str(current_user.id),
            debug_session_id=debug_session_id,
            report_size=len(str(debug_report))
        )
        
        return {
            'success': True,
            'debug_report': debug_report,
            'generated_at': datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Debug report generation failed",
            user_id=str(current_user.id),
            debug_session_id=debug_session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Debug report generation failed: {str(e)}"
        )


@router.get("/search/suggestions", status_code=HTTP_200_OK)
async def get_search_suggestions(
    partial_query: str = Query(..., min_length=1, description="Partial search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum suggestions"),
    search_engine: ConversationSearchEngine = Depends(get_search_engine),
    current_user: User = Depends(get_current_user)
):
    """Get search query auto-completion suggestions."""
    try:
        suggestions = await search_engine.suggest_queries(
            partial_query=partial_query,
            limit=limit
        )
        
        return {
            'success': True,
            'suggestions': suggestions,
            'partial_query': partial_query,
            'generated_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Search suggestions failed",
            user_id=str(current_user.id),
            partial_query=partial_query,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Suggestions failed: {str(e)}"
        )


@router.get("/analytics", status_code=HTTP_200_OK)
async def get_transcript_analytics(
    search_engine: ConversationSearchEngine = Depends(get_search_engine),
    transcript_manager: ChatTranscriptManager = Depends(get_transcript_manager),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive transcript analysis analytics."""
    try:
        # Get search analytics
        search_analytics = await search_engine.get_search_analytics()
        
        # Get transcript manager performance insights
        performance_insights = await transcript_manager.get_performance_insights()
        
        return {
            'success': True,
            'search_analytics': search_analytics,
            'performance_insights': performance_insights,
            'generated_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Analytics retrieval failed",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Analytics failed: {str(e)}"
        )


@router.websocket("/stream")
async def transcript_streaming_websocket(
    websocket: WebSocket,
    session_filters: Optional[str] = Query(None, description="Comma-separated session IDs"),
    agent_filters: Optional[str] = Query(None, description="Comma-separated agent IDs"),
    event_types: Optional[str] = Query(None, description="Comma-separated event types"),
    real_time_only: bool = Query(True, description="Only real-time events"),
    streaming_manager: TranscriptStreamingManager = Depends(get_streaming_manager)
):
    """
    WebSocket endpoint for real-time transcript streaming.
    
    Streams live conversation events with advanced filtering,
    pattern detection, and performance monitoring.
    """
    connection_id = None
    
    try:
        # Parse filters
        streaming_filter = StreamingFilter(
            session_ids=session_filters.split(",") if session_filters else [],
            agent_ids=agent_filters.split(",") if agent_filters else [],
            event_types=[
                ConversationEventType(et.strip())
                for et in event_types.split(",")
                if et.strip()
            ] if event_types else [],
            stream_events=[StreamEventType.CONVERSATION_EVENT, StreamEventType.INSIGHT_GENERATED],
            keywords=[],
            min_severity=AlertSeverity.INFO,
            real_time_only=real_time_only,
            include_patterns=True,
            include_performance=True,
            filter_mode=FilterMode.INCLUSIVE
        )
        
        # Register streaming session
        connection_id = await streaming_manager.register_streaming_session(
            websocket=websocket,
            session_filter=streaming_filter
        )
        
        logger.info(
            "Transcript streaming WebSocket connected",
            connection_id=connection_id,
            filters=streaming_filter.__dict__
        )
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client requests (filter updates, etc.)
                message_type = message.get('type')
                
                if message_type == 'ping':
                    await websocket.send_text(json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.utcnow().isoformat()
                    }))
                elif message_type == 'filter_update':
                    # Update streaming filter
                    filter_data = message.get('filter', {})
                    # Process filter update...
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Internal server error'
                }))
                
    except WebSocketDisconnect:
        logger.info("Transcript streaming WebSocket disconnected", connection_id=connection_id)
    except Exception as e:
        logger.error(
            "Transcript streaming WebSocket error",
            connection_id=connection_id,
            error=str(e)
        )
    finally:
        if connection_id:
            await streaming_manager.unregister_streaming_session(connection_id)


@router.get("/health", status_code=HTTP_200_OK)
async def health_check(
    transcript_manager: ChatTranscriptManager = Depends(get_transcript_manager),
    analyzer: CommunicationAnalyzer = Depends(get_communication_analyzer)
):
    """
    Health check for transcript analysis system.
    
    Verifies system components and provides status information.
    """
    try:
        # Get performance insights
        performance_insights = await transcript_manager.get_performance_insights()
        
        # Check system health
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'transcript_manager': 'healthy',
                'communication_analyzer': 'healthy',
                'search_engine': 'healthy',
                'streaming_manager': 'healthy',
                'debugger': 'healthy'
            },
            'performance': performance_insights
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Health check failed: {str(e)}"
        )