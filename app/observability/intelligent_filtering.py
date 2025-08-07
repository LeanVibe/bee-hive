"""
Intelligent Event Filtering and Semantic Categorization for LeanVibe Agent Hive 2.0

Advanced filtering system with semantic analysis, pattern recognition, and intelligent
categorization for multi-agent observability events. Provides context-aware filtering
with machine learning-powered insights.
"""

import asyncio
import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from collections import defaultdict, Counter

import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.observability import BaseObservabilityEvent, EventCategory
from app.core.embedding_service import get_embedding_service

logger = structlog.get_logger()


class FilterSeverity(str, Enum):
    """Event severity levels for intelligent filtering."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEBUG = "debug"


class FilterPattern(str, Enum):
    """Common event patterns for intelligent recognition."""
    ERROR_CASCADE = "error_cascade"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RETRY_STORM = "retry_storm"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    SECURITY_ANOMALY = "security_anomaly"
    WORKFLOW_BOTTLENECK = "workflow_bottleneck"
    AGENT_COORDINATION_ISSUE = "agent_coordination_issue"


class EventFilter:
    """
    Base class for event filtering with common functionality.
    """
    
    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority
        self.matches = 0
        self.created_at = datetime.utcnow()
    
    def matches_event(self, event: BaseObservabilityEvent) -> bool:
        """Check if event matches this filter."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            "name": self.name,
            "priority": self.priority,
            "matches": self.matches,
            "created_at": self.created_at.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds()
        }


class PatternBasedFilter(EventFilter):
    """Filter events based on pattern matching in payload content."""
    
    def __init__(self, name: str, patterns: List[str], priority: int = 1, case_sensitive: bool = False):
        super().__init__(name, priority)
        self.patterns = [re.compile(p, re.IGNORECASE if not case_sensitive else 0) for p in patterns]
        self.case_sensitive = case_sensitive
    
    def matches_event(self, event: BaseObservabilityEvent) -> bool:
        """Check if event payload matches any of the patterns."""
        try:
            # Convert event payload to searchable text
            payload_text = json.dumps(event.payload if event.payload else {}, default=str)
            
            # Check each pattern
            for pattern in self.patterns:
                if pattern.search(payload_text):
                    self.matches += 1
                    return True
            
            return False
            
        except Exception as e:
            logger.error(
                "❌ Pattern filter error",
                filter_name=self.name,
                error=str(e)
            )
            return False


class PerformanceThresholdFilter(EventFilter):
    """Filter events based on performance thresholds."""
    
    def __init__(
        self, 
        name: str, 
        max_execution_time_ms: Optional[float] = None,
        max_memory_usage_mb: Optional[float] = None,
        max_cpu_usage_percent: Optional[float] = None,
        priority: int = 2
    ):
        super().__init__(name, priority)
        self.max_execution_time_ms = max_execution_time_ms
        self.max_memory_usage_mb = max_memory_usage_mb
        self.max_cpu_usage_percent = max_cpu_usage_percent
    
    def matches_event(self, event: BaseObservabilityEvent) -> bool:
        """Check if event exceeds performance thresholds."""
        try:
            if not event.performance_metrics:
                return False
            
            metrics = event.performance_metrics
            
            # Check execution time threshold
            if (self.max_execution_time_ms and 
                metrics.execution_time_ms > self.max_execution_time_ms):
                self.matches += 1
                return True
            
            # Check memory usage threshold
            if (self.max_memory_usage_mb and 
                metrics.memory_usage_mb > self.max_memory_usage_mb):
                self.matches += 1
                return True
            
            # Check CPU usage threshold
            if (self.max_cpu_usage_percent and 
                metrics.cpu_usage_percent > self.max_cpu_usage_percent):
                self.matches += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "❌ Performance filter error",
                filter_name=self.name,
                error=str(e)
            )
            return False


class TemporalPatternFilter(EventFilter):
    """Filter events based on temporal patterns and frequencies."""
    
    def __init__(
        self, 
        name: str, 
        event_types: List[str],
        time_window_seconds: int = 60,
        min_frequency: int = 5,
        priority: int = 3
    ):
        super().__init__(name, priority)
        self.event_types = set(event_types)
        self.time_window_seconds = time_window_seconds
        self.min_frequency = min_frequency
        self.event_history: List[Tuple[datetime, str]] = []
    
    def matches_event(self, event: BaseObservabilityEvent) -> bool:
        """Check if event is part of a temporal pattern."""
        try:
            if event.event_type not in self.event_types:
                return False
            
            current_time = datetime.utcnow()
            
            # Add current event to history
            self.event_history.append((current_time, event.event_type))
            
            # Clean old events outside time window
            cutoff_time = current_time - timedelta(seconds=self.time_window_seconds)
            self.event_history = [
                (timestamp, event_type) 
                for timestamp, event_type in self.event_history 
                if timestamp > cutoff_time
            ]
            
            # Count events in current window
            event_count = len(self.event_history)
            
            if event_count >= self.min_frequency:
                self.matches += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "❌ Temporal filter error",
                filter_name=self.name,
                error=str(e)
            )
            return False


class SemanticSimilarityFilter(EventFilter):
    """Filter events based on semantic similarity to reference events."""
    
    def __init__(
        self, 
        name: str, 
        reference_events: List[str],
        similarity_threshold: float = 0.8,
        priority: int = 2
    ):
        super().__init__(name, priority)
        self.reference_events = reference_events
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Fit vectorizer on reference events
        if reference_events:
            self.reference_vectors = self.vectorizer.fit_transform(reference_events)
        else:
            self.reference_vectors = None
    
    def matches_event(self, event: BaseObservabilityEvent) -> bool:
        """Check if event is semantically similar to reference events."""
        try:
            if not self.reference_vectors:
                return False
            
            # Extract text content from event
            event_text = self._extract_event_text(event)
            if not event_text:
                return False
            
            # Vectorize event text
            event_vector = self.vectorizer.transform([event_text])
            
            # Calculate similarities
            similarities = cosine_similarity(event_vector, self.reference_vectors)
            max_similarity = float(np.max(similarities))
            
            if max_similarity >= self.similarity_threshold:
                self.matches += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "❌ Semantic filter error",
                filter_name=self.name,
                error=str(e)
            )
            return False
    
    def _extract_event_text(self, event: BaseObservabilityEvent) -> str:
        """Extract searchable text from event."""
        text_parts = []
        
        if event.payload:
            # Add payload content
            payload_text = json.dumps(event.payload, default=str)
            text_parts.append(payload_text)
            
            # Extract specific text fields
            if isinstance(event.payload, dict):
                for key in ['message', 'error', 'description', 'content', 'text']:
                    if key in event.payload and isinstance(event.payload[key], str):
                        text_parts.append(event.payload[key])
        
        return " ".join(text_parts)


class IntelligentEventFilter:
    """
    Advanced intelligent event filtering system with pattern recognition,
    semantic analysis, and adaptive learning capabilities.
    """
    
    def __init__(
        self,
        enable_semantic_analysis: bool = True,
        enable_pattern_recognition: bool = True,
        enable_adaptive_learning: bool = True
    ):
        self.enable_semantic_analysis = enable_semantic_analysis
        self.enable_pattern_recognition = enable_pattern_recognition
        self.enable_adaptive_learning = enable_adaptive_learning
        
        # Filter registry
        self.filters: Dict[str, EventFilter] = {}
        self.filter_chains: Dict[str, List[EventFilter]] = defaultdict(list)
        
        # Pattern recognition
        self.pattern_detector = EventPatternDetector() if enable_pattern_recognition else None
        
        # Semantic analysis
        self.semantic_analyzer = SemanticEventAnalyzer() if enable_semantic_analysis else None
        
        # Adaptive learning
        self.adaptive_learner = AdaptiveFilterLearner() if enable_adaptive_learning else None
        
        # Statistics
        self.total_events_processed = 0
        self.events_filtered = 0
        self.filter_execution_times: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(
            "🧠 Intelligent event filtering system initialized",
            semantic_analysis=enable_semantic_analysis,
            pattern_recognition=enable_pattern_recognition,
            adaptive_learning=enable_adaptive_learning
        )
    
    def add_filter(self, filter_instance: EventFilter, chain: str = "default") -> None:
        """Add a filter to the system."""
        self.filters[filter_instance.name] = filter_instance
        self.filter_chains[chain].append(filter_instance)
        
        # Sort by priority (higher priority first)
        self.filter_chains[chain].sort(key=lambda f: f.priority, reverse=True)
        
        logger.info(
            "✅ Filter added",
            filter_name=filter_instance.name,
            chain=chain,
            priority=filter_instance.priority
        )
    
    def remove_filter(self, filter_name: str) -> bool:
        """Remove a filter from the system."""
        if filter_name not in self.filters:
            return False
        
        filter_instance = self.filters[filter_name]
        del self.filters[filter_name]
        
        # Remove from all chains
        for chain_name, chain_filters in self.filter_chains.items():
            self.filter_chains[chain_name] = [f for f in chain_filters if f.name != filter_name]
        
        logger.info("🗑️ Filter removed", filter_name=filter_name)
        return True
    
    async def filter_event(
        self, 
        event: BaseObservabilityEvent, 
        chain: str = "default"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Filter event through the intelligent filtering system.
        
        Returns:
            Tuple of (should_include, filter_metadata)
        """
        start_time = datetime.utcnow()
        self.total_events_processed += 1
        
        try:
            filter_results = {
                "matched_filters": [],
                "pattern_analysis": {},
                "semantic_analysis": {},
                "severity": FilterSeverity.LOW,
                "recommendations": []
            }
            
            # Run through filter chain
            chain_filters = self.filter_chains.get(chain, [])
            should_include = False
            
            for filter_instance in chain_filters:
                filter_start = datetime.utcnow()
                
                try:
                    if filter_instance.matches_event(event):
                        should_include = True
                        filter_results["matched_filters"].append({
                            "name": filter_instance.name,
                            "priority": filter_instance.priority,
                            "type": type(filter_instance).__name__
                        })
                        
                        # Update severity based on filter priority
                        if filter_instance.priority >= 4:
                            filter_results["severity"] = FilterSeverity.CRITICAL
                        elif filter_instance.priority >= 3:
                            filter_results["severity"] = FilterSeverity.HIGH
                        elif filter_instance.priority >= 2:
                            filter_results["severity"] = FilterSeverity.MEDIUM
                
                except Exception as e:
                    logger.error(
                        "❌ Filter execution error",
                        filter_name=filter_instance.name,
                        error=str(e)
                    )
                
                # Track execution time
                filter_time = (datetime.utcnow() - filter_start).total_seconds() * 1000
                self.filter_execution_times[filter_instance.name].append(filter_time)
            
            # Perform pattern analysis
            if self.pattern_detector:
                try:
                    pattern_analysis = await self.pattern_detector.analyze_event(event)
                    filter_results["pattern_analysis"] = pattern_analysis
                    
                    # Include event if critical patterns detected
                    if pattern_analysis.get("severity") in [FilterSeverity.CRITICAL, FilterSeverity.HIGH]:
                        should_include = True
                
                except Exception as e:
                    logger.error("❌ Pattern analysis error", error=str(e))
            
            # Perform semantic analysis
            if self.semantic_analyzer:
                try:
                    semantic_analysis = await self.semantic_analyzer.analyze_event(event)
                    filter_results["semantic_analysis"] = semantic_analysis
                    
                    # Include event if semantically important
                    if semantic_analysis.get("importance_score", 0) > 0.8:
                        should_include = True
                
                except Exception as e:
                    logger.error("❌ Semantic analysis error", error=str(e))
            
            # Adaptive learning
            if self.adaptive_learner:
                try:
                    recommendations = await self.adaptive_learner.get_recommendations(event, filter_results)
                    filter_results["recommendations"] = recommendations
                
                except Exception as e:
                    logger.error("❌ Adaptive learning error", error=str(e))
            
            # Update statistics
            if should_include:
                self.events_filtered += 1
            
            # Track processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.debug(
                "🔍 Event filtered",
                event_id=str(event.event_id),
                should_include=should_include,
                matched_filters=len(filter_results["matched_filters"]),
                processing_time_ms=processing_time,
                severity=filter_results["severity"]
            )
            
            return should_include, filter_results
            
        except Exception as e:
            logger.error(
                "❌ Event filtering failed",
                event_id=str(event.event_id),
                error=str(e),
                exc_info=True
            )
            # Default to including event on error
            return True, {"error": str(e)}
    
    async def filter_events_batch(
        self, 
        events: List[BaseObservabilityEvent], 
        chain: str = "default"
    ) -> List[Tuple[BaseObservabilityEvent, bool, Dict[str, Any]]]:
        """
        Filter multiple events in batch for optimal performance.
        """
        tasks = [self.filter_event(event, chain) for event in events]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "❌ Batch filtering error",
                    event_index=i,
                    error=str(result)
                )
                batch_results.append((events[i], True, {"error": str(result)}))
            else:
                should_include, metadata = result
                batch_results.append((events[i], should_include, metadata))
        
        return batch_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering statistics."""
        avg_filter_times = {}
        for filter_name, times in self.filter_execution_times.items():
            if times:
                avg_filter_times[filter_name] = {
                    "avg_time_ms": sum(times) / len(times),
                    "max_time_ms": max(times),
                    "min_time_ms": min(times),
                    "executions": len(times)
                }
        
        return {
            "total_events_processed": self.total_events_processed,
            "events_filtered": self.events_filtered,
            "filter_rate": (self.events_filtered / self.total_events_processed) if self.total_events_processed > 0 else 0,
            "active_filters": len(self.filters),
            "filter_chains": {chain: len(filters) for chain, filters in self.filter_chains.items()},
            "filter_performance": avg_filter_times,
            "filter_stats": {name: filter_instance.get_stats() for name, filter_instance in self.filters.items()}
        }


class EventPatternDetector:
    """
    Detects common patterns in event streams for intelligent alerting.
    """
    
    def __init__(self):
        self.pattern_history: Dict[str, List[datetime]] = defaultdict(list)
        self.known_patterns = {
            FilterPattern.ERROR_CASCADE: self._detect_error_cascade,
            FilterPattern.PERFORMANCE_DEGRADATION: self._detect_performance_degradation,
            FilterPattern.RETRY_STORM: self._detect_retry_storm,
            FilterPattern.RESOURCE_EXHAUSTION: self._detect_resource_exhaustion
        }
    
    async def analyze_event(self, event: BaseObservabilityEvent) -> Dict[str, Any]:
        """Analyze event for patterns."""
        analysis_results = {
            "detected_patterns": [],
            "severity": FilterSeverity.LOW,
            "confidence": 0.0,
            "recommendations": []
        }
        
        for pattern_name, detector_func in self.known_patterns.items():
            try:
                pattern_result = detector_func(event)
                if pattern_result["detected"]:
                    analysis_results["detected_patterns"].append({
                        "pattern": pattern_name,
                        "confidence": pattern_result["confidence"],
                        "metadata": pattern_result.get("metadata", {})
                    })
                    
                    # Update overall severity
                    if pattern_result["severity"].value > analysis_results["severity"].value:
                        analysis_results["severity"] = pattern_result["severity"]
            
            except Exception as e:
                logger.error(
                    "❌ Pattern detection error",
                    pattern=pattern_name,
                    error=str(e)
                )
        
        return analysis_results
    
    def _detect_error_cascade(self, event: BaseObservabilityEvent) -> Dict[str, Any]:
        """Detect error cascade patterns."""
        # TODO: Implement sophisticated error cascade detection
        return {"detected": False, "confidence": 0.0, "severity": FilterSeverity.LOW}
    
    def _detect_performance_degradation(self, event: BaseObservabilityEvent) -> Dict[str, Any]:
        """Detect performance degradation patterns."""
        # TODO: Implement performance degradation detection
        return {"detected": False, "confidence": 0.0, "severity": FilterSeverity.LOW}
    
    def _detect_retry_storm(self, event: BaseObservabilityEvent) -> Dict[str, Any]:
        """Detect retry storm patterns."""
        # TODO: Implement retry storm detection
        return {"detected": False, "confidence": 0.0, "severity": FilterSeverity.LOW}
    
    def _detect_resource_exhaustion(self, event: BaseObservabilityEvent) -> Dict[str, Any]:
        """Detect resource exhaustion patterns."""
        # TODO: Implement resource exhaustion detection
        return {"detected": False, "confidence": 0.0, "severity": FilterSeverity.LOW}


class SemanticEventAnalyzer:
    """
    Performs semantic analysis on events for intelligent categorization.
    """
    
    def __init__(self):
        self.embedding_service = None
        self.event_clusters = {}
        self.importance_model = None
    
    async def analyze_event(self, event: BaseObservabilityEvent) -> Dict[str, Any]:
        """Perform semantic analysis on event."""
        analysis_results = {
            "semantic_category": "unknown",
            "importance_score": 0.5,
            "similar_events": [],
            "extracted_topics": [],
            "sentiment": "neutral"
        }
        
        try:
            # Extract text content
            event_text = self._extract_event_text(event)
            if not event_text:
                return analysis_results
            
            # Generate embedding if service available
            if not self.embedding_service:
                self.embedding_service = get_embedding_service()
            
            if self.embedding_service:
                embedding = await self.embedding_service.generate_embedding(event_text)
                # TODO: Use embedding for similarity search and clustering
            
            # Basic text analysis
            analysis_results.update(self._basic_text_analysis(event_text))
            
        except Exception as e:
            logger.error("❌ Semantic analysis error", error=str(e))
        
        return analysis_results
    
    def _extract_event_text(self, event: BaseObservabilityEvent) -> str:
        """Extract searchable text from event."""
        text_parts = []
        
        if event.payload:
            payload_text = json.dumps(event.payload, default=str)
            text_parts.append(payload_text)
        
        return " ".join(text_parts)
    
    def _basic_text_analysis(self, text: str) -> Dict[str, Any]:
        """Perform basic text analysis without heavy dependencies."""
        # Simple keyword-based analysis
        error_keywords = ["error", "exception", "failed", "timeout", "crash"]
        performance_keywords = ["slow", "latency", "memory", "cpu", "performance"]
        security_keywords = ["unauthorized", "forbidden", "security", "breach"]
        
        error_score = sum(1 for keyword in error_keywords if keyword.lower() in text.lower())
        performance_score = sum(1 for keyword in performance_keywords if keyword.lower() in text.lower())
        security_score = sum(1 for keyword in security_keywords if keyword.lower() in text.lower())
        
        # Determine category
        if error_score > 0:
            category = "error"
            importance = min(0.8 + (error_score * 0.05), 1.0)
        elif security_score > 0:
            category = "security"
            importance = min(0.9 + (security_score * 0.05), 1.0)
        elif performance_score > 0:
            category = "performance"
            importance = min(0.6 + (performance_score * 0.1), 1.0)
        else:
            category = "general"
            importance = 0.3
        
        return {
            "semantic_category": category,
            "importance_score": importance,
            "keyword_scores": {
                "error": error_score,
                "performance": performance_score,
                "security": security_score
            }
        }


class AdaptiveFilterLearner:
    """
    Learns from filtering decisions to improve future performance.
    """
    
    def __init__(self):
        self.learning_history: List[Dict[str, Any]] = []
        self.filter_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: {"true_positives": 0, "false_positives": 0})
    
    async def get_recommendations(
        self, 
        event: BaseObservabilityEvent, 
        filter_results: Dict[str, Any]
    ) -> List[str]:
        """Get recommendations for improving filtering."""
        recommendations = []
        
        # Analyze filter performance
        matched_filters = filter_results.get("matched_filters", [])
        if not matched_filters:
            recommendations.append("Consider adding filters for this event type")
        
        # Check for pattern anomalies
        if filter_results.get("pattern_analysis", {}).get("detected_patterns"):
            recommendations.append("Pattern-based alert may be warranted")
        
        # Semantic importance check
        semantic_analysis = filter_results.get("semantic_analysis", {})
        if semantic_analysis.get("importance_score", 0) > 0.7:
            recommendations.append("Event has high semantic importance")
        
        return recommendations


# Pre-configured filter sets for common scenarios
def create_error_monitoring_filters() -> List[EventFilter]:
    """Create filters optimized for error monitoring."""
    return [
        PatternBasedFilter(
            name="error_patterns",
            patterns=[r"error", r"exception", r"failed", r"timeout", r"crash"],
            priority=4
        ),
        PerformanceThresholdFilter(
            name="slow_operations",
            max_execution_time_ms=5000.0,
            priority=3
        ),
        TemporalPatternFilter(
            name="error_frequency",
            event_types=["PostToolUse", "Notification"],
            time_window_seconds=300,
            min_frequency=10,
            priority=4
        )
    ]


def create_performance_monitoring_filters() -> List[EventFilter]:
    """Create filters optimized for performance monitoring."""
    return [
        PerformanceThresholdFilter(
            name="high_latency",
            max_execution_time_ms=2000.0,
            priority=3
        ),
        PerformanceThresholdFilter(
            name="memory_usage",
            max_memory_usage_mb=1000.0,
            priority=2
        ),
        TemporalPatternFilter(
            name="performance_degradation",
            event_types=["PostToolUse"],
            time_window_seconds=600,
            min_frequency=5,
            priority=3
        )
    ]


def create_security_monitoring_filters() -> List[EventFilter]:
    """Create filters optimized for security monitoring."""
    return [
        PatternBasedFilter(
            name="security_keywords",
            patterns=[r"unauthorized", r"forbidden", r"security", r"breach", r"attack"],
            priority=4
        ),
        PatternBasedFilter(
            name="suspicious_activity",
            patterns=[r"injection", r"xss", r"csrf", r"malware", r"intrusion"],
            priority=4
        )
    ]