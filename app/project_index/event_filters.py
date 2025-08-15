"""
Advanced Event Filtering and Subscription Logic for Project Index WebSocket

This module provides sophisticated filtering capabilities for project index events,
including relevance filtering, user personalization, and performance filtering.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union
from uuid import UUID
from enum import Enum
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from .websocket_events import (
    ProjectIndexEventType, ProjectIndexWebSocketEvent,
    ProjectIndexUpdateData, AnalysisProgressData, 
    DependencyChangeData, ContextOptimizedData
)

logger = structlog.get_logger()


class FilterCriteria(Enum):
    """Event filtering criteria types."""
    PROJECT_SPECIFIC = "project_specific"
    EVENT_TYPE = "event_type"
    RELEVANCE_SCORE = "relevance_score"
    PERFORMANCE_IMPACT = "performance_impact"
    USER_PREFERENCE = "user_preference"
    FILE_LANGUAGE = "file_language"
    FILE_PATH_PATTERN = "file_path_pattern"
    DEPENDENCY_TYPE = "dependency_type"
    PROGRESS_THRESHOLD = "progress_threshold"


@dataclass
class FilterRule:
    """Individual filter rule configuration."""
    criteria: FilterCriteria
    operator: str  # equals, contains, greater_than, less_than, in, not_in
    value: Any
    weight: float = 1.0  # For relevance scoring
    enabled: bool = True


@dataclass
class UserPreferences:
    """User-specific event preferences."""
    user_id: str
    preferred_languages: List[str] = None
    ignored_file_patterns: List[str] = None
    min_progress_updates: int = 10  # Only show progress every N%
    high_impact_only: bool = False
    notification_frequency: str = "normal"  # low, normal, high
    custom_filters: List[FilterRule] = None
    
    def __post_init__(self):
        if self.preferred_languages is None:
            self.preferred_languages = []
        if self.ignored_file_patterns is None:
            self.ignored_file_patterns = []
        if self.custom_filters is None:
            self.custom_filters = []


class RelevanceScorer:
    """Calculates relevance scores for events based on various factors."""
    
    def __init__(self):
        self.scoring_weights = {
            "file_frequency": 0.3,
            "dependency_impact": 0.4,
            "language_preference": 0.2,
            "recency": 0.1
        }
        
        # Track file access patterns for relevance scoring
        self.file_access_patterns: Dict[str, Dict[str, Any]] = {}
        self.language_popularity: Dict[str, int] = {}
    
    def calculate_relevance(
        self, 
        event: ProjectIndexWebSocketEvent,
        user_preferences: Optional[UserPreferences] = None
    ) -> float:
        """Calculate relevance score for an event (0.0 to 1.0)."""
        base_score = 0.5  # Default relevance
        
        try:
            if event.type == ProjectIndexEventType.PROJECT_INDEX_UPDATED:
                base_score = self._score_project_update(event.data, user_preferences)
            elif event.type == ProjectIndexEventType.ANALYSIS_PROGRESS:
                base_score = self._score_analysis_progress(event.data, user_preferences)
            elif event.type == ProjectIndexEventType.DEPENDENCY_CHANGED:
                base_score = self._score_dependency_change(event.data, user_preferences)
            elif event.type == ProjectIndexEventType.CONTEXT_OPTIMIZED:
                base_score = self._score_context_optimization(event.data, user_preferences)
            
            # Apply recency boost
            recency_score = self._calculate_recency_score(event.timestamp)
            final_score = base_score * (1 + recency_score * self.scoring_weights["recency"])
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.warning("Error calculating relevance score", error=str(e))
            return base_score
    
    def _score_project_update(
        self, 
        data: Dict[str, Any], 
        user_prefs: Optional[UserPreferences]
    ) -> float:
        """Score project update events."""
        score = 0.7  # High base score for project updates
        
        # Boost score for significant changes
        files_analyzed = data.get('files_analyzed', 0)
        dependencies_updated = data.get('dependencies_updated', 0)
        
        if files_analyzed > 50:
            score += 0.1
        if dependencies_updated > 20:
            score += 0.1
        
        # Check for errors (higher relevance)
        if data.get('error_count', 0) > 0:
            score += 0.2
        
        return score
    
    def _score_analysis_progress(
        self, 
        data: Dict[str, Any], 
        user_prefs: Optional[UserPreferences]
    ) -> float:
        """Score analysis progress events."""
        progress = data.get('progress_percentage', 0)
        
        # Lower score for frequent progress updates
        base_score = 0.3
        
        # Higher score for milestone progress
        if progress in [25, 50, 75, 100]:
            base_score = 0.6
        
        # User preference for progress frequency
        if user_prefs and user_prefs.min_progress_updates:
            if progress % user_prefs.min_progress_updates != 0:
                base_score *= 0.5
        
        # Boost for errors
        if data.get('errors_encountered', 0) > 0:
            base_score += 0.3
        
        return base_score
    
    def _score_dependency_change(
        self, 
        data: Dict[str, Any], 
        user_prefs: Optional[UserPreferences]
    ) -> float:
        """Score dependency change events."""
        score = 0.6  # Good base score for dependency changes
        
        # Higher score for breaking changes
        change_type = data.get('change_type', '')
        if change_type in ['removed', 'file_deleted']:
            score += 0.2
        
        # Impact analysis boost
        impact = data.get('impact_analysis', {})
        affected_files = impact.get('affected_files', [])
        if len(affected_files) > 5:
            score += 0.2
        
        # Language preference scoring
        if user_prefs and user_prefs.preferred_languages:
            file_meta = data.get('file_metadata', {})
            language = file_meta.get('language', '')
            if language in user_prefs.preferred_languages:
                score += 0.1
        
        return score
    
    def _score_context_optimization(
        self, 
        data: Dict[str, Any], 
        user_prefs: Optional[UserPreferences]
    ) -> float:
        """Score context optimization events."""
        score = 0.5  # Medium base score
        
        # Higher score for high-confidence optimizations
        results = data.get('optimization_results', {})
        confidence = results.get('confidence_score', 0.5)
        score += confidence * 0.3
        
        # Boost for actionable recommendations
        recommendations = data.get('recommendations', {})
        if recommendations.get('architectural_patterns'):
            score += 0.1
        if recommendations.get('potential_challenges'):
            score += 0.1
        
        return score
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """Calculate recency boost (0.0 to 1.0)."""
        age_minutes = (datetime.utcnow() - timestamp).total_seconds() / 60
        
        # Exponential decay over 60 minutes
        if age_minutes <= 5:
            return 1.0
        elif age_minutes <= 30:
            return 0.8
        elif age_minutes <= 60:
            return 0.5
        else:
            return 0.2
    
    def update_access_patterns(self, user_id: str, file_path: str, language: str) -> None:
        """Update file access patterns for better relevance scoring."""
        if user_id not in self.file_access_patterns:
            self.file_access_patterns[user_id] = {}
        
        if file_path not in self.file_access_patterns[user_id]:
            self.file_access_patterns[user_id][file_path] = {
                'access_count': 0,
                'last_access': None,
                'language': language
            }
        
        self.file_access_patterns[user_id][file_path]['access_count'] += 1
        self.file_access_patterns[user_id][file_path]['last_access'] = datetime.utcnow()
        
        # Update language popularity
        self.language_popularity[language] = self.language_popularity.get(language, 0) + 1


class EventFilter:
    """Main event filtering engine with advanced capabilities."""
    
    def __init__(self):
        self.relevance_scorer = RelevanceScorer()
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.global_filters: List[FilterRule] = []
        
        # Performance tracking
        self.filter_metrics = {
            "events_filtered": 0,
            "events_passed": 0,
            "average_relevance_score": 0.0,
            "filter_processing_time_ms": 0.0
        }
    
    async def should_deliver_event(
        self, 
        event: ProjectIndexWebSocketEvent,
        user_id: str,
        connection_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if event should be delivered to user."""
        start_time = datetime.utcnow()
        
        try:
            # Get user preferences
            user_prefs = self.user_preferences.get(user_id)
            
            # Calculate relevance score
            relevance_score = self.relevance_scorer.calculate_relevance(event, user_prefs)
            
            # Apply filters
            should_deliver = await self._apply_filters(event, user_prefs, relevance_score, connection_metadata)
            
            # Update metrics
            if should_deliver:
                self.filter_metrics["events_passed"] += 1
            else:
                self.filter_metrics["events_filtered"] += 1
            
            # Update average relevance score
            total_events = self.filter_metrics["events_passed"] + self.filter_metrics["events_filtered"]
            if total_events > 0:
                current_avg = self.filter_metrics["average_relevance_score"]
                self.filter_metrics["average_relevance_score"] = (
                    (current_avg * (total_events - 1) + relevance_score) / total_events
                )
            
            # Track processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.filter_metrics["filter_processing_time_ms"] = (
                (self.filter_metrics["filter_processing_time_ms"] + processing_time) / 2
            )
            
            logger.debug("Event filter decision",
                        event_type=event.type.value,
                        user_id=user_id,
                        relevance_score=relevance_score,
                        should_deliver=should_deliver)
            
            return should_deliver
            
        except Exception as e:
            logger.error("Error in event filtering", error=str(e), user_id=user_id)
            # Default to deliver on error
            return True
    
    async def _apply_filters(
        self, 
        event: ProjectIndexWebSocketEvent,
        user_prefs: Optional[UserPreferences],
        relevance_score: float,
        connection_metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply all configured filters to event."""
        
        # Global filters
        for filter_rule in self.global_filters:
            if filter_rule.enabled and not await self._evaluate_filter_rule(event, filter_rule):
                return False
        
        # User-specific filters
        if user_prefs and user_prefs.custom_filters:
            for filter_rule in user_prefs.custom_filters:
                if filter_rule.enabled and not await self._evaluate_filter_rule(event, filter_rule):
                    return False
        
        # Relevance threshold filter
        min_relevance = 0.3  # Default minimum relevance
        if user_prefs:
            if user_prefs.high_impact_only:
                min_relevance = 0.7
            elif user_prefs.notification_frequency == "low":
                min_relevance = 0.6
            elif user_prefs.notification_frequency == "high":
                min_relevance = 0.1
        
        if relevance_score < min_relevance:
            return False
        
        # Progress update frequency filter
        if (event.type == ProjectIndexEventType.ANALYSIS_PROGRESS and 
            user_prefs and user_prefs.min_progress_updates):
            
            progress = event.data.get('progress_percentage', 0)
            if progress % user_prefs.min_progress_updates != 0 and progress != 100:
                return False
        
        # Language preference filter
        if user_prefs and user_prefs.preferred_languages:
            if not await self._check_language_preference(event, user_prefs.preferred_languages):
                return False
        
        # File pattern ignore filter
        if user_prefs and user_prefs.ignored_file_patterns:
            if await self._check_ignored_patterns(event, user_prefs.ignored_file_patterns):
                return False
        
        return True
    
    async def _evaluate_filter_rule(
        self, 
        event: ProjectIndexWebSocketEvent, 
        rule: FilterRule
    ) -> bool:
        """Evaluate a specific filter rule."""
        try:
            if rule.criteria == FilterCriteria.EVENT_TYPE:
                return self._evaluate_operator(event.type.value, rule.operator, rule.value)
            
            elif rule.criteria == FilterCriteria.PROJECT_SPECIFIC:
                project_id = event.data.get('project_id')
                return self._evaluate_operator(project_id, rule.operator, rule.value)
            
            elif rule.criteria == FilterCriteria.PERFORMANCE_IMPACT:
                # Check if event indicates performance issues
                if event.type == ProjectIndexEventType.ANALYSIS_PROGRESS:
                    processing_rate = event.data.get('processing_rate', 1.0)
                    return self._evaluate_operator(processing_rate, rule.operator, rule.value)
                elif event.type == ProjectIndexEventType.PROJECT_INDEX_UPDATED:
                    duration = event.data.get('analysis_duration_seconds', 0)
                    return self._evaluate_operator(duration, rule.operator, rule.value)
            
            elif rule.criteria == FilterCriteria.FILE_LANGUAGE:
                language = self._extract_language_from_event(event)
                if language:
                    return self._evaluate_operator(language, rule.operator, rule.value)
            
            elif rule.criteria == FilterCriteria.DEPENDENCY_TYPE:
                if event.type == ProjectIndexEventType.DEPENDENCY_CHANGED:
                    dep_details = event.data.get('dependency_details', {})
                    dep_type = dep_details.get('relationship_type', '')
                    return self._evaluate_operator(dep_type, rule.operator, rule.value)
            
            # Default to pass if rule doesn't apply
            return True
            
        except Exception as e:
            logger.warning("Error evaluating filter rule", error=str(e), rule=rule.criteria.value)
            return True
    
    def _evaluate_operator(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
        """Evaluate filter operator."""
        try:
            if operator == "equals":
                return actual_value == expected_value
            elif operator == "not_equals":
                return actual_value != expected_value
            elif operator == "contains":
                return str(expected_value).lower() in str(actual_value).lower()
            elif operator == "not_contains":
                return str(expected_value).lower() not in str(actual_value).lower()
            elif operator == "in":
                return actual_value in expected_value
            elif operator == "not_in":
                return actual_value not in expected_value
            elif operator == "greater_than":
                return float(actual_value) > float(expected_value)
            elif operator == "less_than":
                return float(actual_value) < float(expected_value)
            elif operator == "greater_equal":
                return float(actual_value) >= float(expected_value)
            elif operator == "less_equal":
                return float(actual_value) <= float(expected_value)
            else:
                logger.warning("Unknown filter operator", operator=operator)
                return True
        except (ValueError, TypeError) as e:
            logger.warning("Error in operator evaluation", error=str(e))
            return True
    
    async def _check_language_preference(
        self, 
        event: ProjectIndexWebSocketEvent, 
        preferred_languages: List[str]
    ) -> bool:
        """Check if event relates to preferred languages."""
        language = self._extract_language_from_event(event)
        if not language:
            return True  # Pass events without language info
        
        return language.lower() in [lang.lower() for lang in preferred_languages]
    
    async def _check_ignored_patterns(
        self, 
        event: ProjectIndexWebSocketEvent, 
        ignored_patterns: List[str]
    ) -> bool:
        """Check if event matches ignored file patterns."""
        file_path = self._extract_file_path_from_event(event)
        if not file_path:
            return False  # Don't ignore events without file paths
        
        import fnmatch
        for pattern in ignored_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        
        return False
    
    def _extract_language_from_event(self, event: ProjectIndexWebSocketEvent) -> Optional[str]:
        """Extract programming language from event data."""
        if event.type == ProjectIndexEventType.DEPENDENCY_CHANGED:
            file_meta = event.data.get('file_metadata', {})
            return file_meta.get('language')
        elif event.type == ProjectIndexEventType.PROJECT_INDEX_UPDATED:
            # Extract from statistics if available
            stats = event.data.get('statistics', {})
            languages = stats.get('languages_detected', [])
            return languages[0] if languages else None
        
        return None
    
    def _extract_file_path_from_event(self, event: ProjectIndexWebSocketEvent) -> Optional[str]:
        """Extract file path from event data."""
        if event.type == ProjectIndexEventType.DEPENDENCY_CHANGED:
            return event.data.get('file_path')
        elif event.type == ProjectIndexEventType.ANALYSIS_PROGRESS:
            return event.data.get('current_file')
        
        return None
    
    def set_user_preferences(self, user_id: str, preferences: UserPreferences) -> None:
        """Set user-specific filtering preferences."""
        self.user_preferences[user_id] = preferences
        logger.info("Updated user preferences", user_id=user_id)
    
    def add_global_filter(self, filter_rule: FilterRule) -> None:
        """Add global filter rule that applies to all users."""
        self.global_filters.append(filter_rule)
        logger.info("Added global filter", criteria=filter_rule.criteria.value)
    
    def remove_global_filter(self, criteria: FilterCriteria, value: Any = None) -> int:
        """Remove global filter rules by criteria."""
        original_count = len(self.global_filters)
        
        if value is None:
            self.global_filters = [f for f in self.global_filters if f.criteria != criteria]
        else:
            self.global_filters = [
                f for f in self.global_filters 
                if not (f.criteria == criteria and f.value == value)
            ]
        
        removed_count = original_count - len(self.global_filters)
        logger.info("Removed global filters", count=removed_count)
        return removed_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get filtering performance metrics."""
        return {
            **self.filter_metrics,
            "active_user_preferences": len(self.user_preferences),
            "global_filters": len(self.global_filters),
            "filter_pass_rate": (
                self.filter_metrics["events_passed"] / 
                (self.filter_metrics["events_passed"] + self.filter_metrics["events_filtered"])
                if (self.filter_metrics["events_passed"] + self.filter_metrics["events_filtered"]) > 0
                else 0.0
            )
        }


# Global filter instance
_event_filter: Optional[EventFilter] = None


def get_event_filter() -> EventFilter:
    """Get or create the global event filter instance."""
    global _event_filter
    if _event_filter is None:
        _event_filter = EventFilter()
    return _event_filter