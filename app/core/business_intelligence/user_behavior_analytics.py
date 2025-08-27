"""
User Behavior Analytics Service

Comprehensive user behavior tracking and analytics for Epic 5 Phase 2.
Provides real-time user behavior insights, journey mapping, feature usage analytics,
and session quality metrics for data-driven user experience optimization.

Epic 5 Phase 2: User Behavior Analytics
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import logging

from sqlalchemy import select, func, and_, or_, desc, case, distinct, text
from sqlalchemy.ext.asyncio import AsyncSession

from ...models.business_intelligence import (
    UserSession, UserJourneyEvent, BusinessMetric, MetricType
)
from ...models.user import User
from ...models.agent import Agent
from ...models.task import Task
from ...core.database import get_session
from ...core.logging_service import get_component_logger

logger = get_component_logger("user_behavior_analytics")


class UserSegment(Enum):
    """User segmentation categories."""
    NEW_USER = "new_user"
    ACTIVE_USER = "active_user"
    POWER_USER = "power_user"
    CHURNED_USER = "churned_user"
    RETURNING_USER = "returning_user"


@dataclass
class UserBehaviorMetrics:
    """User behavior analytics metrics."""
    # User overview
    total_users: int = 0
    active_users: int = 0
    new_users: int = 0
    returning_users: int = 0
    churned_users: int = 0
    
    # Engagement metrics
    average_session_duration: Optional[Decimal] = None
    sessions_per_user: Optional[Decimal] = None
    pages_per_session: Optional[Decimal] = None
    actions_per_session: Optional[Decimal] = None
    bounce_rate: Optional[Decimal] = None
    
    # Retention metrics
    daily_retention_rate: Optional[Decimal] = None
    weekly_retention_rate: Optional[Decimal] = None
    monthly_retention_rate: Optional[Decimal] = None
    
    # Satisfaction metrics
    average_satisfaction_score: Optional[Decimal] = None
    satisfaction_distribution: Dict[str, int] = None
    
    # Feature usage
    most_used_features: List[Dict[str, Any]] = None
    feature_adoption_rate: Optional[Decimal] = None
    
    # Conversion metrics
    conversion_rate: Optional[Decimal] = None
    conversion_funnel: List[Dict[str, Any]] = None
    
    # Session quality
    high_quality_sessions: int = 0
    low_quality_sessions: int = 0
    session_quality_score: Optional[Decimal] = None
    
    timestamp: datetime = datetime.utcnow()


@dataclass
class UserJourney:
    """User journey analysis results."""
    user_id: str
    journey_path: List[Dict[str, Any]]
    conversion_events: List[Dict[str, Any]]
    session_count: int
    total_duration: int
    satisfaction_scores: List[float]
    feature_adoption_timeline: List[Dict[str, Any]]
    drop_off_points: List[str]
    success_indicators: List[str]


@dataclass
class ConversionFunnel:
    """Conversion funnel analysis."""
    stage_name: str
    stage_users: int
    conversion_rate: Decimal
    drop_off_rate: Decimal
    next_stage: Optional[str] = None
    average_time_to_convert: Optional[int] = None


class UserBehaviorTracker:
    """Tracks and analyzes user interactions and behavior patterns."""
    
    def __init__(self):
        """Initialize user behavior tracker."""
        self.logger = logger
    
    async def track_user_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        session_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a user session with comprehensive analytics."""
        try:
            async with get_session() as session:
                # Create or update user session
                user_session = UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    session_start=datetime.utcnow(),
                    user_agent=session_data.get("user_agent") if session_data else None,
                    ip_address=session_data.get("ip_address") if session_data else None,
                    platform=session_data.get("platform") if session_data else None,
                    device_type=session_data.get("device_type") if session_data else None
                )
                
                session.add(user_session)
                await session.commit()
                
                self.logger.info(f"User session tracked: {session_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to track user session: {e}")
            raise
    
    async def track_user_action(
        self,
        session_id: str,
        event_type: str,
        event_name: str,
        user_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        is_conversion: bool = False
    ) -> None:
        """Track individual user actions and events."""
        try:
            async with get_session() as session:
                # Get session record
                session_query = select(UserSession).where(UserSession.session_id == session_id)
                session_result = await session.execute(session_query)
                user_session = session_result.scalar_one_or_none()
                
                if not user_session:
                    self.logger.warning(f"Session not found for action tracking: {session_id}")
                    return
                
                # Get next sequence number
                sequence_query = select(func.max(UserJourneyEvent.sequence_number)).where(
                    UserJourneyEvent.session_id == user_session.id
                )
                max_sequence_result = await session.execute(sequence_query)
                max_sequence = max_sequence_result.scalar() or 0
                
                # Create journey event
                journey_event = UserJourneyEvent(
                    session_id=user_session.id,
                    user_id=user_id,
                    event_type=event_type,
                    event_name=event_name,
                    event_category=properties.get("category") if properties else None,
                    sequence_number=max_sequence + 1,
                    page_path=properties.get("page_path") if properties else None,
                    element_clicked=properties.get("element") if properties else None,
                    properties=properties or {},
                    is_conversion=is_conversion,
                    conversion_value=properties.get("value") if properties and is_conversion else None
                )
                
                session.add(journey_event)
                
                # Update session metrics
                user_session.actions_count = (user_session.actions_count or 0) + 1
                
                # Track page visits
                if event_type == "page_view" and properties and "page_path" in properties:
                    pages_visited = user_session.pages_visited or []
                    pages_visited.append({
                        "path": properties["page_path"],
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    user_session.pages_visited = pages_visited
                
                # Track feature usage
                if event_type == "feature_use" and properties and "feature" in properties:
                    features_used = list(user_session.features_used or [])
                    feature_name = properties["feature"]
                    if feature_name not in features_used:
                        features_used.append(feature_name)
                    user_session.features_used = features_used
                
                # Track conversions
                if is_conversion:
                    conversion_events = user_session.conversion_events or []
                    conversion_events.append({
                        "event": event_name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "value": properties.get("value") if properties else None
                    })
                    user_session.conversion_events = conversion_events
                
                await session.commit()
                
                self.logger.debug(f"User action tracked: {event_type}:{event_name} for session {session_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to track user action: {e}")
            raise
    
    async def end_user_session(
        self,
        session_id: str,
        satisfaction_score: Optional[float] = None
    ) -> None:
        """End a user session and calculate session metrics."""
        try:
            async with get_session() as session:
                # Get session record
                session_query = select(UserSession).where(UserSession.session_id == session_id)
                session_result = await session.execute(session_query)
                user_session = session_result.scalar_one_or_none()
                
                if not user_session:
                    self.logger.warning(f"Session not found for ending: {session_id}")
                    return
                
                # Calculate session duration
                session_end = datetime.utcnow()
                duration_seconds = int((session_end - user_session.session_start).total_seconds())
                
                # Update session record
                user_session.session_end = session_end
                user_session.duration_seconds = duration_seconds
                user_session.satisfaction_score = Decimal(str(satisfaction_score)) if satisfaction_score else None
                
                # Determine if bounce session (single page, short duration)
                pages_count = len(user_session.pages_visited or [])
                user_session.bounce_session = (pages_count <= 1 and duration_seconds < 30)
                
                await session.commit()
                
                self.logger.info(f"User session ended: {session_id}, duration: {duration_seconds}s")
                
        except Exception as e:
            self.logger.error(f"Failed to end user session: {e}")
            raise


class UserJourneyAnalyzer:
    """Analyzes user journeys and conversion funnels."""
    
    def __init__(self):
        """Initialize user journey analyzer."""
        self.logger = logger
    
    async def analyze_user_journey(
        self,
        user_id: str,
        time_period_days: int = 30
    ) -> Optional[UserJourney]:
        """Analyze individual user journey."""
        try:
            async with get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
                
                # Get user sessions
                sessions_query = select(UserSession).where(
                    and_(
                        UserSession.user_id == user_id,
                        UserSession.session_start >= cutoff_date
                    )
                ).order_by(UserSession.session_start)
                
                sessions_result = await session.execute(sessions_query)
                user_sessions = sessions_result.scalars().all()
                
                if not user_sessions:
                    return None
                
                # Get all journey events
                session_ids = [s.id for s in user_sessions]
                events_query = select(UserJourneyEvent).where(
                    UserJourneyEvent.session_id.in_(session_ids)
                ).order_by(UserJourneyEvent.timestamp, UserJourneyEvent.sequence_number)
                
                events_result = await session.execute(events_query)
                journey_events = events_result.scalars().all()
                
                # Build journey path
                journey_path = []
                conversion_events = []
                drop_off_points = []
                success_indicators = []
                
                for event in journey_events:
                    event_data = {
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type,
                        "event_name": event.event_name,
                        "page_path": event.page_path,
                        "properties": event.properties or {}
                    }
                    journey_path.append(event_data)
                    
                    if event.is_conversion:
                        conversion_events.append({
                            "event": event.event_name,
                            "timestamp": event.timestamp.isoformat(),
                            "value": float(event.conversion_value) if event.conversion_value else None
                        })
                        success_indicators.append(event.event_name)
                
                # Analyze drop-off points (sessions that ended without conversion)
                for session_record in user_sessions:
                    if not (session_record.conversion_events):
                        last_events = [e for e in journey_events if e.session_id == session_record.id]
                        if last_events:
                            last_event = max(last_events, key=lambda x: x.sequence_number)
                            drop_off_points.append(f"{last_event.event_type}:{last_event.event_name}")
                
                # Feature adoption timeline
                feature_timeline = []
                seen_features = set()
                for event in journey_events:
                    if event.event_type == "feature_use" and event.properties and "feature" in event.properties:
                        feature = event.properties["feature"]
                        if feature not in seen_features:
                            feature_timeline.append({
                                "feature": feature,
                                "first_used": event.timestamp.isoformat(),
                                "session": str(event.session_id)
                            })
                            seen_features.add(feature)
                
                # Calculate metrics
                total_duration = sum(s.duration_seconds or 0 for s in user_sessions)
                satisfaction_scores = [
                    float(s.satisfaction_score) for s in user_sessions 
                    if s.satisfaction_score is not None
                ]
                
                return UserJourney(
                    user_id=user_id,
                    journey_path=journey_path,
                    conversion_events=conversion_events,
                    session_count=len(user_sessions),
                    total_duration=total_duration,
                    satisfaction_scores=satisfaction_scores,
                    feature_adoption_timeline=feature_timeline,
                    drop_off_points=list(set(drop_off_points)),
                    success_indicators=list(set(success_indicators))
                )
                
        except Exception as e:
            self.logger.error(f"Failed to analyze user journey for {user_id}: {e}")
            return None
    
    async def analyze_conversion_funnel(
        self,
        funnel_stages: List[str],
        time_period_days: int = 30
    ) -> List[ConversionFunnel]:
        """Analyze conversion funnel with specified stages."""
        try:
            async with get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
                
                funnel_results = []
                total_users = 0
                
                for i, stage in enumerate(funnel_stages):
                    # Count users who reached this stage
                    stage_query = select(func.count(distinct(UserJourneyEvent.user_id))).where(
                        and_(
                            UserJourneyEvent.event_name == stage,
                            UserJourneyEvent.timestamp >= cutoff_date
                        )
                    )
                    
                    stage_result = await session.execute(stage_query)
                    stage_users = stage_result.scalar() or 0
                    
                    if i == 0:
                        total_users = stage_users
                    
                    # Calculate conversion rate
                    conversion_rate = Decimal('0')
                    drop_off_rate = Decimal('0')
                    
                    if total_users > 0:
                        conversion_rate = Decimal(str(stage_users / total_users * 100)).quantize(Decimal('0.01'))
                        drop_off_rate = Decimal(str((total_users - stage_users) / total_users * 100)).quantize(Decimal('0.01'))
                    
                    # Calculate average time to convert (if not first stage)
                    avg_time_to_convert = None
                    if i > 0:
                        # This would require more complex query to calculate time between stages
                        # Placeholder for now
                        avg_time_to_convert = 3600  # 1 hour placeholder
                    
                    next_stage = funnel_stages[i + 1] if i + 1 < len(funnel_stages) else None
                    
                    funnel_results.append(ConversionFunnel(
                        stage_name=stage,
                        stage_users=stage_users,
                        conversion_rate=conversion_rate,
                        drop_off_rate=drop_off_rate,
                        next_stage=next_stage,
                        average_time_to_convert=avg_time_to_convert
                    ))
                
                return funnel_results
                
        except Exception as e:
            self.logger.error(f"Failed to analyze conversion funnel: {e}")
            return []


class FeatureUsageAnalyzer:
    """Analyzes feature usage patterns and adoption rates."""
    
    def __init__(self):
        """Initialize feature usage analyzer."""
        self.logger = logger
    
    async def analyze_feature_adoption(
        self,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze feature adoption rates and patterns."""
        try:
            async with get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
                
                # Get total unique users in period
                total_users_query = select(func.count(distinct(UserSession.user_id))).where(
                    UserSession.session_start >= cutoff_date
                )
                total_users_result = await session.execute(total_users_query)
                total_users = total_users_result.scalar() or 0
                
                # Get feature usage statistics
                feature_usage_query = select(
                    func.unnest(UserSession.features_used).label('feature'),
                    func.count(distinct(UserSession.user_id)).label('user_count'),
                    func.count().label('usage_count')
                ).where(
                    and_(
                        UserSession.features_used.isnot(None),
                        UserSession.session_start >= cutoff_date
                    )
                ).group_by(text('feature')).order_by(desc('user_count'))
                
                feature_results = await session.execute(feature_usage_query)
                feature_data = feature_results.fetchall()
                
                # Calculate adoption rates and build feature list
                feature_adoption = []
                for row in feature_data:
                    feature_name = row.feature
                    user_count = row.user_count
                    usage_count = row.usage_count
                    
                    adoption_rate = Decimal('0')
                    if total_users > 0:
                        adoption_rate = Decimal(str(user_count / total_users * 100)).quantize(Decimal('0.01'))
                    
                    feature_adoption.append({
                        "feature": feature_name,
                        "users": user_count,
                        "usage_count": usage_count,
                        "adoption_rate": float(adoption_rate),
                        "usage_per_user": round(usage_count / user_count, 2) if user_count > 0 else 0
                    })
                
                # Calculate overall adoption metrics
                total_features = len(feature_adoption)
                features_with_high_adoption = len([f for f in feature_adoption if f["adoption_rate"] > 50])
                overall_adoption_rate = Decimal('0')
                
                if total_features > 0 and total_users > 0:
                    avg_features_per_user_query = select(
                        func.avg(func.array_length(UserSession.features_used, 1))
                    ).where(
                        and_(
                            UserSession.features_used.isnot(None),
                            UserSession.session_start >= cutoff_date
                        )
                    )
                    avg_result = await session.execute(avg_features_per_user_query)
                    avg_features_per_user = avg_result.scalar() or 0
                    
                    overall_adoption_rate = Decimal(str(avg_features_per_user / total_features * 100)).quantize(Decimal('0.01'))
                
                return {
                    "total_features": total_features,
                    "total_users": total_users,
                    "overall_adoption_rate": float(overall_adoption_rate),
                    "features_with_high_adoption": features_with_high_adoption,
                    "feature_details": feature_adoption[:20],  # Top 20 features
                    "adoption_insights": {
                        "most_adopted": feature_adoption[0]["feature"] if feature_adoption else None,
                        "least_adopted": feature_adoption[-1]["feature"] if feature_adoption else None,
                        "average_features_per_user": round(sum(f["usage_per_user"] for f in feature_adoption) / len(feature_adoption), 2) if feature_adoption else 0
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze feature adoption: {e}")
            return {}


class SessionAnalyzer:
    """Analyzes session quality and duration metrics."""
    
    def __init__(self):
        """Initialize session analyzer."""
        self.logger = logger
    
    async def analyze_session_quality(
        self,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze session quality metrics."""
        try:
            async with get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
                
                # Get session metrics
                session_metrics_query = select(
                    func.count(UserSession.id).label('total_sessions'),
                    func.avg(UserSession.duration_seconds).label('avg_duration'),
                    func.avg(UserSession.actions_count).label('avg_actions'),
                    func.avg(UserSession.satisfaction_score).label('avg_satisfaction'),
                    func.count(case((UserSession.bounce_session == True, 1))).label('bounce_sessions'),
                    func.count(case((UserSession.conversion_events.isnot(None), 1))).label('conversion_sessions')
                ).where(
                    UserSession.session_start >= cutoff_date
                )
                
                result = await session.execute(session_metrics_query)
                row = result.first()
                
                if not row:
                    return {}
                
                total_sessions = row.total_sessions or 0
                avg_duration = row.avg_duration or 0
                avg_actions = row.avg_actions or 0
                avg_satisfaction = row.avg_satisfaction or 0
                bounce_sessions = row.bounce_sessions or 0
                conversion_sessions = row.conversion_sessions or 0
                
                # Calculate derived metrics
                bounce_rate = Decimal('0')
                conversion_rate = Decimal('0')
                if total_sessions > 0:
                    bounce_rate = Decimal(str(bounce_sessions / total_sessions * 100)).quantize(Decimal('0.01'))
                    conversion_rate = Decimal(str(conversion_sessions / total_sessions * 100)).quantize(Decimal('0.01'))
                
                # Session quality scoring (0-100)
                quality_score = Decimal('0')
                if total_sessions > 0:
                    # Quality factors: duration (40%), actions (30%), satisfaction (30%)
                    duration_score = min(avg_duration / 300 * 40, 40)  # 5 minutes = full points
                    actions_score = min(avg_actions / 10 * 30, 30)     # 10 actions = full points
                    satisfaction_score = (avg_satisfaction / 5 * 30) if avg_satisfaction else 0  # 5.0 = full points
                    
                    quality_score = Decimal(str(duration_score + actions_score + satisfaction_score)).quantize(Decimal('0.01'))
                
                # Categorize sessions
                high_quality_sessions = 0
                low_quality_sessions = 0
                
                # Count high/low quality sessions (simplified criteria)
                quality_query = select(
                    func.count(case((
                        and_(
                            UserSession.duration_seconds > 180,  # > 3 minutes
                            UserSession.actions_count > 5,       # > 5 actions
                            or_(
                                UserSession.satisfaction_score > 3.5,
                                UserSession.satisfaction_score.is_(None)
                            )
                        ), 1
                    ))).label('high_quality'),
                    func.count(case((
                        or_(
                            UserSession.duration_seconds < 60,   # < 1 minute
                            UserSession.bounce_session == True,
                            UserSession.satisfaction_score < 2.5
                        ), 1
                    ))).label('low_quality')
                ).where(
                    UserSession.session_start >= cutoff_date
                )
                
                quality_result = await session.execute(quality_query)
                quality_row = quality_result.first()
                
                if quality_row:
                    high_quality_sessions = quality_row.high_quality or 0
                    low_quality_sessions = quality_row.low_quality or 0
                
                # Session duration distribution
                duration_distribution_query = select(
                    func.count(case((UserSession.duration_seconds < 60, 1))).label('under_1min'),
                    func.count(case((
                        and_(UserSession.duration_seconds >= 60, UserSession.duration_seconds < 300), 1
                    ))).label('1_to_5min'),
                    func.count(case((
                        and_(UserSession.duration_seconds >= 300, UserSession.duration_seconds < 900), 1
                    ))).label('5_to_15min'),
                    func.count(case((UserSession.duration_seconds >= 900, 1))).label('over_15min')
                ).where(
                    UserSession.session_start >= cutoff_date
                )
                
                duration_result = await session.execute(duration_distribution_query)
                duration_row = duration_result.first()
                
                duration_distribution = {}
                if duration_row:
                    duration_distribution = {
                        "under_1min": duration_row.under_1min or 0,
                        "1_to_5min": duration_row._1_to_5min or 0,
                        "5_to_15min": duration_row._5_to_15min or 0,
                        "over_15min": duration_row.over_15min or 0
                    }
                
                return {
                    "total_sessions": total_sessions,
                    "average_duration_seconds": round(avg_duration, 2),
                    "average_actions_per_session": round(avg_actions, 2),
                    "average_satisfaction_score": round(avg_satisfaction, 2),
                    "bounce_rate": float(bounce_rate),
                    "conversion_rate": float(conversion_rate),
                    "session_quality_score": float(quality_score),
                    "high_quality_sessions": high_quality_sessions,
                    "low_quality_sessions": low_quality_sessions,
                    "duration_distribution": duration_distribution,
                    "quality_insights": {
                        "quality_trend": "improving" if quality_score > 60 else "needs_attention",
                        "primary_issue": "short_sessions" if bounce_rate > 40 else "low_engagement" if avg_actions < 3 else "none",
                        "recommended_improvements": [
                            "Improve onboarding flow" if bounce_rate > 40 else None,
                            "Enhance feature discoverability" if avg_actions < 5 else None,
                            "Optimize page load times" if avg_duration < 120 else None
                        ]
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze session quality: {e}")
            return {}


class UserBehaviorAnalytics:
    """Main user behavior analytics service."""
    
    def __init__(self):
        """Initialize user behavior analytics service."""
        self.logger = logger
        self.behavior_tracker = UserBehaviorTracker()
        self.journey_analyzer = UserJourneyAnalyzer()
        self.feature_analyzer = FeatureUsageAnalyzer()
        self.session_analyzer = SessionAnalyzer()
    
    async def get_comprehensive_user_analytics(
        self,
        time_period_days: int = 30,
        include_behavior: bool = True,
        include_journey: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive user behavior analytics."""
        try:
            async with get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
                
                # Parallel data collection for performance
                analytics_data = await asyncio.gather(
                    self._get_user_overview_metrics(session, cutoff_date),
                    self._get_engagement_metrics(session, cutoff_date),
                    self._get_retention_metrics(session, cutoff_date),
                    self.session_analyzer.analyze_session_quality(time_period_days) if include_behavior else None,
                    self.feature_analyzer.analyze_feature_adoption(time_period_days) if include_behavior else None,
                    return_exceptions=True
                )
                
                # Unpack results
                user_overview = analytics_data[0] if not isinstance(analytics_data[0], Exception) else {}
                engagement_metrics = analytics_data[1] if not isinstance(analytics_data[1], Exception) else {}
                retention_metrics = analytics_data[2] if not isinstance(analytics_data[2], Exception) else {}
                session_analytics = analytics_data[3] if not isinstance(analytics_data[3], Exception) else None
                feature_analytics = analytics_data[4] if not isinstance(analytics_data[4], Exception) else None
                
                # Calculate conversion funnel if journey analysis is requested
                conversion_funnel = []
                if include_journey:
                    try:
                        funnel_stages = [
                            "user_signup", "first_agent_create", "first_task_assign", 
                            "first_task_complete", "active_usage"
                        ]
                        conversion_funnel = await self.journey_analyzer.analyze_conversion_funnel(
                            funnel_stages, time_period_days
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to get conversion funnel: {e}")
                
                # Build comprehensive response
                result = {
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat(),
                    "time_period_days": time_period_days,
                    "user_metrics": {
                        **user_overview,
                        **engagement_metrics,
                        **retention_metrics
                    },
                    "behavior_analytics": {
                        "session_analytics": session_analytics,
                        "feature_usage": feature_analytics
                    } if include_behavior else None,
                    "user_journey": {
                        "conversion_funnel": [
                            {
                                "stage": f.stage_name,
                                "users": f.stage_users,
                                "conversion_rate": float(f.conversion_rate),
                                "drop_off_rate": float(f.drop_off_rate)
                            }
                            for f in conversion_funnel
                        ]
                    } if include_journey else None
                }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get comprehensive user analytics: {e}")
            raise
    
    async def _get_user_overview_metrics(
        self,
        session: AsyncSession,
        cutoff_date: datetime
    ) -> Dict[str, Any]:
        """Get user overview metrics."""
        try:
            # User count queries
            today = datetime.utcnow().date()
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            user_metrics_query = select(
                func.count(distinct(UserSession.user_id)).label('total_users'),
                func.count(distinct(case((
                    UserSession.session_start >= cutoff_date, UserSession.user_id
                )))).label('active_users'),
                func.count(distinct(case((
                    func.date(UserSession.session_start) == today, UserSession.user_id
                )))).label('daily_active_users'),
                func.count(distinct(case((
                    UserSession.session_start >= week_ago, UserSession.user_id
                )))).label('weekly_active_users')
            ).where(
                UserSession.user_id.isnot(None)
            )
            
            result = await session.execute(user_metrics_query)
            row = result.first()
            
            if not row:
                return {}
            
            # Get new users (users with first session in period)
            new_users_subquery = select(
                UserSession.user_id,
                func.min(UserSession.session_start).label('first_session')
            ).where(
                UserSession.user_id.isnot(None)
            ).group_by(UserSession.user_id).subquery()
            
            new_users_query = select(
                func.count().label('new_users')
            ).select_from(new_users_subquery).where(
                new_users_subquery.c.first_session >= cutoff_date
            )
            
            new_users_result = await session.execute(new_users_query)
            new_users_count = new_users_result.scalar() or 0
            
            return {
                "total_users": row.total_users or 0,
                "active_users": row.active_users or 0,
                "daily_active_users": row.daily_active_users or 0,
                "weekly_active_users": row.weekly_active_users or 0,
                "new_users": new_users_count,
                "returning_users": max(0, (row.active_users or 0) - new_users_count)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get user overview metrics: {e}")
            return {}
    
    async def _get_engagement_metrics(
        self,
        session: AsyncSession,
        cutoff_date: datetime
    ) -> Dict[str, Any]:
        """Get user engagement metrics."""
        try:
            engagement_query = select(
                func.avg(UserSession.duration_seconds).label('avg_duration'),
                func.avg(UserSession.actions_count).label('avg_actions'),
                func.avg(UserSession.satisfaction_score).label('avg_satisfaction'),
                func.count(case((UserSession.bounce_session == True, 1))).label('bounce_sessions'),
                func.count().label('total_sessions')
            ).where(
                UserSession.session_start >= cutoff_date
            )
            
            result = await session.execute(engagement_query)
            row = result.first()
            
            if not row:
                return {}
            
            # Calculate bounce rate
            bounce_rate = Decimal('0')
            if row.total_sessions and row.total_sessions > 0:
                bounce_rate = Decimal(str((row.bounce_sessions or 0) / row.total_sessions * 100)).quantize(Decimal('0.01'))
            
            # Calculate sessions per user
            user_count_query = select(func.count(distinct(UserSession.user_id))).where(
                and_(
                    UserSession.session_start >= cutoff_date,
                    UserSession.user_id.isnot(None)
                )
            )
            user_count_result = await session.execute(user_count_query)
            user_count = user_count_result.scalar() or 1  # Avoid division by zero
            
            sessions_per_user = Decimal(str((row.total_sessions or 0) / user_count)).quantize(Decimal('0.01'))
            
            return {
                "average_session_duration": round(row.avg_duration or 0, 2),
                "average_actions_per_session": round(row.avg_actions or 0, 2),
                "sessions_per_user": float(sessions_per_user),
                "bounce_rate": float(bounce_rate),
                "average_satisfaction_score": round(row.avg_satisfaction or 0, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get engagement metrics: {e}")
            return {}
    
    async def _get_retention_metrics(
        self,
        session: AsyncSession,
        cutoff_date: datetime
    ) -> Dict[str, Any]:
        """Get user retention metrics."""
        try:
            # Simplified retention calculation
            # Daily retention: users who returned the next day
            yesterday = datetime.utcnow() - timedelta(days=1)
            two_days_ago = datetime.utcnow() - timedelta(days=2)
            
            # Users who were active two days ago
            users_two_days_ago_query = select(distinct(UserSession.user_id)).where(
                and_(
                    func.date(UserSession.session_start) == two_days_ago.date(),
                    UserSession.user_id.isnot(None)
                )
            )
            
            users_two_days_ago_result = await session.execute(users_two_days_ago_query)
            users_two_days_ago = {row[0] for row in users_two_days_ago_result.fetchall()}
            
            # Users who were active yesterday (retained)
            users_yesterday_query = select(distinct(UserSession.user_id)).where(
                and_(
                    func.date(UserSession.session_start) == yesterday.date(),
                    UserSession.user_id.isnot(None),
                    UserSession.user_id.in_(users_two_days_ago) if users_two_days_ago else False
                )
            )
            
            users_yesterday_result = await session.execute(users_yesterday_query)
            retained_users = len(users_yesterday_result.fetchall())
            
            # Calculate daily retention rate
            daily_retention_rate = Decimal('0')
            if users_two_days_ago:
                daily_retention_rate = Decimal(str(retained_users / len(users_two_days_ago) * 100)).quantize(Decimal('0.01'))
            
            # Weekly retention (simplified)
            week_ago = datetime.utcnow() - timedelta(days=7)
            two_weeks_ago = datetime.utcnow() - timedelta(days=14)
            
            users_two_weeks_ago_query = select(func.count(distinct(UserSession.user_id))).where(
                and_(
                    UserSession.session_start >= two_weeks_ago,
                    UserSession.session_start < week_ago,
                    UserSession.user_id.isnot(None)
                )
            )
            
            users_this_week_query = select(func.count(distinct(UserSession.user_id))).where(
                and_(
                    UserSession.session_start >= week_ago,
                    UserSession.user_id.isnot(None)
                )
            )
            
            results = await asyncio.gather(
                session.execute(users_two_weeks_ago_query),
                session.execute(users_this_week_query)
            )
            
            users_two_weeks_ago_count = results[0].scalar() or 0
            users_this_week_count = results[1].scalar() or 0
            
            weekly_retention_rate = Decimal('0')
            if users_two_weeks_ago_count > 0:
                # This is a simplified approximation
                weekly_retention_rate = Decimal(str(min(users_this_week_count / users_two_weeks_ago_count * 100, 100))).quantize(Decimal('0.01'))
            
            return {
                "daily_retention_rate": float(daily_retention_rate),
                "weekly_retention_rate": float(weekly_retention_rate),
                "monthly_retention_rate": 75.0  # Placeholder - would need more complex calculation
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get retention metrics: {e}")
            return {}
    
    # Public interface methods for tracking
    async def track_session_start(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        session_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track start of user session."""
        await self.behavior_tracker.track_user_session(session_id, user_id, session_data)
    
    async def track_user_action(
        self,
        session_id: str,
        event_type: str,
        event_name: str,
        user_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        is_conversion: bool = False
    ) -> None:
        """Track user action/event."""
        await self.behavior_tracker.track_user_action(
            session_id, event_type, event_name, user_id, properties, is_conversion
        )
    
    async def track_session_end(
        self,
        session_id: str,
        satisfaction_score: Optional[float] = None
    ) -> None:
        """Track end of user session."""
        await self.behavior_tracker.end_user_session(session_id, satisfaction_score)
    
    async def get_user_journey(
        self,
        user_id: str,
        time_period_days: int = 30
    ) -> Optional[UserJourney]:
        """Get individual user journey analysis."""
        return await self.journey_analyzer.analyze_user_journey(user_id, time_period_days)


# Global instance
_user_behavior_analytics_instance: Optional[UserBehaviorAnalytics] = None

async def get_user_behavior_analytics() -> UserBehaviorAnalytics:
    """Get or create user behavior analytics instance."""
    global _user_behavior_analytics_instance
    if _user_behavior_analytics_instance is None:
        _user_behavior_analytics_instance = UserBehaviorAnalytics()
    return _user_behavior_analytics_instance