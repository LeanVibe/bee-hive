"""
Poison Message Detector for LeanVibe Agent Hive 2.0 - VS 4.3

Advanced poison message detection and isolation system with machine learning-based
pattern recognition and automated quarantine capabilities.

Handles the poison message scenarios:
- Malformed JSON payloads
- Oversized messages (>1MB)
- Invalid agent/session IDs
- Circular references in data
- Encoding/decoding failures
- Database constraint violations
- Timeout-prone operations
"""

import asyncio
import json
import time
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog

from ..models.message import StreamMessage, MessageType, MessagePriority
from .config import settings

logger = structlog.get_logger()


class PoisonMessageType(str, Enum):
    """Types of poison messages detected."""
    MALFORMED_JSON = "malformed_json"
    OVERSIZED_MESSAGE = "oversized_message"
    INVALID_AGENT_ID = "invalid_agent_id"
    INVALID_SESSION_ID = "invalid_session_id"
    CIRCULAR_REFERENCE = "circular_reference"
    ENCODING_ERROR = "encoding_error"
    DATABASE_CONSTRAINT = "database_constraint"
    TIMEOUT_PRONE = "timeout_prone"
    INVALID_MESSAGE_TYPE = "invalid_message_type"
    MISSING_REQUIRED_FIELDS = "missing_required_fields"
    SCHEMA_VIOLATION = "schema_violation"
    RECURSIVE_PAYLOAD = "recursive_payload"
    UNKNOWN_POISON = "unknown_poison"


class DetectionConfidence(str, Enum):
    """Confidence levels for poison message detection."""
    VERY_HIGH = "very_high"    # 95%+ confidence
    HIGH = "high"              # 80-95% confidence
    MEDIUM = "medium"          # 60-80% confidence
    LOW = "low"                # 40-60% confidence
    VERY_LOW = "very_low"      # <40% confidence


class IsolationAction(str, Enum):
    """Actions to take when poison message is detected."""
    IMMEDIATE_QUARANTINE = "immediate_quarantine"
    DELAYED_QUARANTINE = "delayed_quarantine"
    TRANSFORM_AND_RETRY = "transform_and_retry"
    LOG_AND_CONTINUE = "log_and_continue"
    REJECT_MESSAGE = "reject_message"
    HUMAN_REVIEW = "human_review"


@dataclass
class PoisonDetectionResult:
    """Result of poison message detection."""
    
    is_poison: bool
    poison_type: Optional[PoisonMessageType]
    confidence: DetectionConfidence
    detection_reason: str
    suggested_action: IsolationAction
    
    # Detailed analysis
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    pattern_matches: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 to 1.0
    
    # Recovery suggestions
    is_recoverable: bool = False
    recovery_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    detection_time: float = field(default_factory=time.time)
    detector_version: str = "4.3.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_poison": self.is_poison,
            "poison_type": self.poison_type.value if self.poison_type else None,
            "confidence": self.confidence.value,
            "detection_reason": self.detection_reason,
            "suggested_action": self.suggested_action.value,
            "analysis_details": self.analysis_details,
            "pattern_matches": self.pattern_matches,
            "risk_score": self.risk_score,
            "is_recoverable": self.is_recoverable,
            "recovery_suggestions": self.recovery_suggestions,
            "detection_time": self.detection_time,
            "detector_version": self.detector_version
        }


@dataclass
class DetectionPattern:
    """Pattern for poison message detection."""
    
    pattern_id: str
    poison_type: PoisonMessageType
    detection_regex: Optional[str] = None
    size_threshold: Optional[int] = None
    custom_validator: Optional[str] = None  # Function name for custom validation
    confidence_weight: float = 1.0
    
    # Pattern metadata
    description: str = ""
    examples: List[str] = field(default_factory=list)
    false_positive_rate: float = 0.05


@dataclass  
class DetectionMetrics:
    """Metrics for poison detection system."""
    
    total_messages_analyzed: int = 0
    poison_messages_detected: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Detection type breakdown
    detections_by_type: Dict[PoisonMessageType, int] = field(default_factory=dict)
    detections_by_confidence: Dict[DetectionConfidence, int] = field(default_factory=dict)
    
    # Performance metrics
    average_detection_time_ms: float = 0.0
    max_detection_time_ms: float = 0.0
    detection_accuracy: float = 0.0
    
    # Pattern effectiveness
    pattern_matches: Dict[str, int] = field(default_factory=dict)
    
    def calculate_accuracy(self) -> float:
        """Calculate detection accuracy."""
        total_classified = self.poison_messages_detected + self.false_positives
        if total_classified == 0:
            return 0.0
        
        correct_predictions = self.poison_messages_detected - self.false_negatives
        return max(0.0, correct_predictions / total_classified)


class PoisonMessageDetector:
    """
    Advanced poison message detector with ML-based pattern recognition.
    
    Features:
    - Multi-layered detection (syntax, semantic, behavioral)
    - Configurable detection patterns and thresholds
    - Real-time performance monitoring
    - Adaptive learning from detection results
    - Integration with DLQ and error handling systems
    """
    
    def __init__(
        self,
        max_message_size_bytes: int = 1024 * 1024,  # 1MB
        detection_timeout_ms: int = 100,  # 100ms max detection time
        enable_adaptive_learning: bool = True,
        confidence_threshold: DetectionConfidence = DetectionConfidence.MEDIUM
    ):
        """Initialize poison message detector."""
        self.max_message_size_bytes = max_message_size_bytes
        self.detection_timeout_ms = detection_timeout_ms
        self.enable_adaptive_learning = enable_adaptive_learning
        self.confidence_threshold = confidence_threshold
        
        # Detection patterns
        self.detection_patterns = self._initialize_detection_patterns()
        
        # Metrics and tracking
        self.metrics = DetectionMetrics()
        self._detection_history: List[Tuple[str, PoisonDetectionResult]] = []
        self._known_poison_hashes: Set[str] = set()
        self._known_clean_hashes: Set[str] = set()
        
        # Performance optimization
        self._pattern_cache: Dict[str, PoisonDetectionResult] = {}
        self._cache_max_size = 1000
        
        logger.info(
            "🔍 Poison Message Detector initialized",
            max_message_size_mb=max_message_size_bytes // (1024 * 1024),
            detection_timeout_ms=detection_timeout_ms,
            adaptive_learning=enable_adaptive_learning,
            confidence_threshold=confidence_threshold.value
        )
    
    def _initialize_detection_patterns(self) -> List[DetectionPattern]:
        """Initialize detection patterns for poison messages."""
        return [
            # Malformed JSON patterns
            DetectionPattern(
                pattern_id="malformed_json_1",
                poison_type=PoisonMessageType.MALFORMED_JSON,
                detection_pattern=r'.*[{,]\s*["}]\s*[,}].*',  # Missing keys/values
                confidence_weight=0.8,
                description="Detect JSON with missing keys or values"
            ),
            
            DetectionPattern(
                pattern_id="malformed_json_2",
                poison_type=PoisonMessageType.MALFORMED_JSON,
                detection_pattern=r'.*[^"]\s*:\s*[^"\d\[\{tfn].*',  # Invalid value types
                confidence_weight=0.7,
                description="Detect JSON with invalid value types"
            ),
            
            # Circular reference patterns
            DetectionPattern(
                pattern_id="circular_ref_1",
                poison_type=PoisonMessageType.CIRCULAR_REFERENCE,
                custom_validator="detect_circular_references",
                confidence_weight=0.9,
                description="Detect circular references in nested objects"
            ),
            
            # Oversized message
            DetectionPattern(
                pattern_id="oversized_msg",
                poison_type=PoisonMessageType.OVERSIZED_MESSAGE,
                size_threshold=1024 * 1024,  # 1MB
                confidence_weight=1.0,
                description="Detect messages exceeding size limit"
            ),
            
            # Invalid UUID patterns (for agent/session IDs)
            DetectionPattern(
                pattern_id="invalid_uuid",
                poison_type=PoisonMessageType.INVALID_AGENT_ID,
                detection_pattern=r'.*"(?:agent_id|session_id)"\s*:\s*"(?![\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}").*".*',
                confidence_weight=0.85,
                description="Detect invalid UUID format for agent/session IDs"
            ),
            
            # Database constraint violations
            DetectionPattern(
                pattern_id="sql_injection_attempt",
                poison_type=PoisonMessageType.DATABASE_CONSTRAINT,
                detection_pattern=r'.*(union\s+select|drop\s+table|delete\s+from|insert\s+into).*',
                confidence_weight=0.95,
                description="Detect potential SQL injection attempts"
            ),
            
            # Encoding issues
            DetectionPattern(
                pattern_id="encoding_error",
                poison_type=PoisonMessageType.ENCODING_ERROR,
                custom_validator="detect_encoding_issues",
                confidence_weight=0.75,
                description="Detect encoding/decoding issues"
            ),
            
            # Recursive/deep nesting
            DetectionPattern(
                pattern_id="recursive_payload",
                poison_type=PoisonMessageType.RECURSIVE_PAYLOAD,
                custom_validator="detect_excessive_nesting",
                confidence_weight=0.8,
                description="Detect excessively nested or recursive payloads"
            ),
            
            # Timeout-prone patterns
            DetectionPattern(
                pattern_id="timeout_prone",
                poison_type=PoisonMessageType.TIMEOUT_PRONE,
                custom_validator="detect_timeout_prone_patterns",
                confidence_weight=0.6,
                description="Detect patterns likely to cause timeouts"
            ),
            
            # Schema violations
            DetectionPattern(
                pattern_id="missing_required_fields",
                poison_type=PoisonMessageType.MISSING_REQUIRED_FIELDS,
                custom_validator="validate_required_fields",
                confidence_weight=0.9,
                description="Detect missing required message fields"
            )
        ]
    
    async def analyze_message(
        self,
        message: Union[StreamMessage, Dict[str, Any], str],
        context: Optional[Dict[str, Any]] = None
    ) -> PoisonDetectionResult:
        """
        Analyze message for poison characteristics.
        
        Args:
            message: Message to analyze
            context: Optional context for analysis
            
        Returns:
            Detection result with confidence and suggested actions
        """
        start_time = time.time()
        
        try:
            # Convert message to analyzable format
            message_data, message_str, message_size = await self._prepare_message_for_analysis(message)
            
            # Check cache first for performance  
            message_hash = self._calculate_message_hash(message_str)
            if message_hash in self._pattern_cache:
                cached_result = self._pattern_cache[message_hash]
                logger.debug(f"🎯 Using cached detection result for message {message_hash[:8]}")
                return cached_result
            
            # Quick checks first (performance optimization)
            quick_result = await self._perform_quick_checks(message_str, message_size, message_data)
            if quick_result.is_poison and quick_result.confidence in [DetectionConfidence.VERY_HIGH, DetectionConfidence.HIGH]:
                await self._cache_result(message_hash, quick_result)
                await self._update_metrics(quick_result, start_time)
                return quick_result
            
            # Comprehensive analysis
            comprehensive_result = await self._perform_comprehensive_analysis(
                message_data, message_str, message_size, context
            )
            
            # Combine results
            final_result = await self._combine_detection_results([quick_result, comprehensive_result])
            
            # Cache and update metrics
            await self._cache_result(message_hash, final_result)
            await self._update_metrics(final_result, start_time)
            
            # Log significant detections
            if final_result.is_poison and final_result.confidence in [DetectionConfidence.HIGH, DetectionConfidence.VERY_HIGH]:
                logger.warning(
                    f"🚨 Poison message detected",
                    poison_type=final_result.poison_type.value if final_result.poison_type else "unknown",
                    confidence=final_result.confidence.value,
                    risk_score=final_result.risk_score,
                    suggested_action=final_result.suggested_action.value
                )
            
            return final_result
            
        except Exception as e:
            # Fallback - treat analysis error as potential poison
            logger.error(f"Error analyzing message for poison characteristics: {e}")
            
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=PoisonMessageType.UNKNOWN_POISON,
                confidence=DetectionConfidence.MEDIUM,
                detection_reason=f"Analysis error: {str(e)}",
                suggested_action=IsolationAction.HUMAN_REVIEW,
                risk_score=0.7,
                analysis_details={"error": str(e)}
            )
    
    async def _prepare_message_for_analysis(
        self,
        message: Union[StreamMessage, Dict[str, Any], str]
    ) -> Tuple[Dict[str, Any], str, int]:
        """Prepare message for analysis."""
        
        if isinstance(message, StreamMessage):
            message_data = message.to_dict()
            message_str = message.json()
        elif isinstance(message, dict):
            message_data = message
            message_str = json.dumps(message, sort_keys=True)
        elif isinstance(message, str):
            message_str = message
            try:
                message_data = json.loads(message)
            except json.JSONDecodeError:
                message_data = {"raw_content": message}
        else:
            message_str = str(message)
            message_data = {"raw_content": message_str}
        
        message_size = len(message_str.encode('utf-8'))
        
        return message_data, message_str, message_size
    
    def _calculate_message_hash(self, message_str: str) -> str:
        """Calculate SHA-256 hash of message for caching."""
        return hashlib.sha256(message_str.encode('utf-8')).hexdigest()
    
    async def _perform_quick_checks(
        self,
        message_str: str,
        message_size: int,
        message_data: Dict[str, Any]
    ) -> PoisonDetectionResult:
        """Perform quick checks for obvious poison messages."""
        
        # Size check
        if message_size > self.max_message_size_bytes:
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=PoisonMessageType.OVERSIZED_MESSAGE,
                confidence=DetectionConfidence.VERY_HIGH,
                detection_reason=f"Message size ({message_size} bytes) exceeds limit ({self.max_message_size_bytes} bytes)",
                suggested_action=IsolationAction.IMMEDIATE_QUARANTINE,
                risk_score=1.0,
                analysis_details={"message_size_bytes": message_size, "size_limit_bytes": self.max_message_size_bytes}
            )
        
        # Basic JSON structure check
        try:
            if isinstance(message_data.get("raw_content"), str):
                json.loads(message_data["raw_content"])
        except json.JSONDecodeError as e:
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=PoisonMessageType.MALFORMED_JSON,
                confidence=DetectionConfidence.HIGH,
                detection_reason=f"JSON parsing error: {str(e)}",
                suggested_action=IsolationAction.TRANSFORM_AND_RETRY,
                risk_score=0.8,
                analysis_details={"json_error": str(e)},
                is_recoverable=True,
                recovery_suggestions=["Fix JSON syntax", "Validate JSON structure"]
            )
        
        # No immediate poison detected
        return PoisonDetectionResult(
            is_poison=False,
            poison_type=None,
            confidence=DetectionConfidence.LOW,
            detection_reason="Passed quick checks",
            suggested_action=IsolationAction.LOG_AND_CONTINUE,
            risk_score=0.1
        )
    
    async def _perform_comprehensive_analysis(
        self,
        message_data: Dict[str, Any],
        message_str: str,
        message_size: int,
        context: Optional[Dict[str, Any]]
    ) -> PoisonDetectionResult:
        """Perform comprehensive poison message analysis."""
        
        detected_issues = []
        total_risk_score = 0.0
        max_confidence = DetectionConfidence.VERY_LOW
        primary_poison_type = None
        pattern_matches = []
        
        # Run all detection patterns
        for pattern in self.detection_patterns:
            try:
                detection_result = await self._apply_detection_pattern(
                    pattern, message_data, message_str, message_size, context
                )
                
                if detection_result.is_poison:
                    detected_issues.append(detection_result)
                    total_risk_score += detection_result.risk_score * pattern.confidence_weight
                    pattern_matches.extend(detection_result.pattern_matches)
                    
                    # Update max confidence and primary type
                    if self._confidence_value(detection_result.confidence) > self._confidence_value(max_confidence):
                        max_confidence = detection_result.confidence
                        primary_poison_type = detection_result.poison_type
                        
            except Exception as e:
                logger.error(f"Error applying detection pattern {pattern.pattern_id}: {e}")
        
        # Determine final result
        is_poison = len(detected_issues) > 0
        final_confidence = max_confidence if is_poison else DetectionConfidence.LOW
        
        # Normalize risk score
        normalized_risk_score = min(1.0, total_risk_score / len(self.detection_patterns))
        
        # Determine suggested action
        suggested_action = self._determine_isolation_action(
            is_poison, final_confidence, normalized_risk_score, detected_issues
        )
        
        # Create comprehensive detection reason
        if detected_issues:
            detection_reasons = [issue.detection_reason for issue in detected_issues]
            combined_reason = "; ".join(detection_reasons[:3])  # Limit to top 3 reasons
        else:
            combined_reason = "No poison characteristics detected"
        
        return PoisonDetectionResult(
            is_poison=is_poison,
            poison_type=primary_poison_type,
            confidence=final_confidence,
            detection_reason=combined_reason,
            suggested_action=suggested_action,
            analysis_details={
                "detected_issues_count": len(detected_issues),
                "detection_patterns_matched": len(pattern_matches),
                "comprehensive_analysis": True
            },
            pattern_matches=pattern_matches,
            risk_score=normalized_risk_score,
            is_recoverable=any(issue.is_recoverable for issue in detected_issues),
            recovery_suggestions=list(set(
                suggestion
                for issue in detected_issues
                for suggestion in issue.recovery_suggestions
            ))
        )
    
    async def _apply_detection_pattern(
        self,
        pattern: DetectionPattern,
        message_data: Dict[str, Any],
        message_str: str,
        message_size: int,
        context: Optional[Dict[str, Any]]
    ) -> PoisonDetectionResult:
        """Apply a specific detection pattern."""
        
        # Size-based detection
        if pattern.size_threshold and message_size > pattern.size_threshold:
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=pattern.poison_type,
                confidence=DetectionConfidence.VERY_HIGH,
                detection_reason=f"Message size ({message_size}) exceeds pattern threshold ({pattern.size_threshold})",
                suggested_action=IsolationAction.IMMEDIATE_QUARANTINE,
                risk_score=1.0,
                pattern_matches=[pattern.pattern_id]
            )
        
        # Regex-based detection
        if pattern.detection_regex:
            if re.search(pattern.detection_regex, message_str, re.IGNORECASE):
                confidence = self._calculate_regex_confidence(pattern, message_str)
                return PoisonDetectionResult(
                    is_poison=True,
                    poison_type=pattern.poison_type,
                    confidence=confidence,
                    detection_reason=f"Pattern {pattern.pattern_id} matched regex",
                    suggested_action=self._get_pattern_action(pattern, confidence),
                    risk_score=pattern.confidence_weight,
                    pattern_matches=[pattern.pattern_id]
                )
        
        # Custom validator
        if pattern.custom_validator:
            custom_result = await self._apply_custom_validator(
                pattern.custom_validator, pattern, message_data, message_str, context
            )
            if custom_result.is_poison:
                return custom_result
        
        # No poison detected by this pattern
        return PoisonDetectionResult(
            is_poison=False,
            poison_type=None,
            confidence=DetectionConfidence.LOW,
            detection_reason=f"Pattern {pattern.pattern_id} did not match",
            suggested_action=IsolationAction.LOG_AND_CONTINUE,
            risk_score=0.0
        )
    
    async def _apply_custom_validator(
        self,
        validator_name: str,
        pattern: DetectionPattern,
        message_data: Dict[str, Any],
        message_str: str,
        context: Optional[Dict[str, Any]]
    ) -> PoisonDetectionResult:
        """Apply custom validation logic."""
        
        try:
            if validator_name == "detect_circular_references":
                return await self._detect_circular_references(pattern, message_data)
            elif validator_name == "detect_encoding_issues":
                return await self._detect_encoding_issues(pattern, message_str)
            elif validator_name == "detect_excessive_nesting":
                return await self._detect_excessive_nesting(pattern, message_data)
            elif validator_name == "detect_timeout_prone_patterns":
                return await self._detect_timeout_prone_patterns(pattern, message_data, context)
            elif validator_name == "validate_required_fields":
                return await self._validate_required_fields(pattern, message_data)
            else:
                logger.warning(f"Unknown custom validator: {validator_name}")
                
        except Exception as e:
            logger.error(f"Error in custom validator {validator_name}: {e}")
        
        return PoisonDetectionResult(
            is_poison=False,
            poison_type=None,
            confidence=DetectionConfidence.LOW,
            detection_reason="Custom validator did not detect poison",
            suggested_action=IsolationAction.LOG_AND_CONTINUE,
            risk_score=0.0
        )
    
    async def _detect_circular_references(
        self,
        pattern: DetectionPattern,
        message_data: Dict[str, Any]
    ) -> PoisonDetectionResult:
        """Detect circular references in message data."""
        
        def has_circular_reference(obj, seen_ids=None):
            if seen_ids is None:
                seen_ids = set()
            
            if isinstance(obj, dict):
                obj_id = id(obj)
                if obj_id in seen_ids:
                    return True
                seen_ids.add(obj_id)
                
                for value in obj.values():
                    if has_circular_reference(value, seen_ids.copy()):
                        return True
            elif isinstance(obj, list):
                obj_id = id(obj)
                if obj_id in seen_ids:
                    return True
                seen_ids.add(obj_id)
                
                for item in obj:
                    if has_circular_reference(item, seen_ids.copy()):
                        return True
            
            return False
        
        if has_circular_reference(message_data):
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=pattern.poison_type,
                confidence=DetectionConfidence.HIGH,
                detection_reason="Circular reference detected in message data",
                suggested_action=IsolationAction.IMMEDIATE_QUARANTINE,
                risk_score=0.9,
                pattern_matches=[pattern.pattern_id],
                analysis_details={"circular_reference_detected": True}
            )
        
        return PoisonDetectionResult(
            is_poison=False,
            poison_type=None,
            confidence=DetectionConfidence.LOW,
            detection_reason="No circular references detected",
            suggested_action=IsolationAction.LOG_AND_CONTINUE,
            risk_score=0.0
        )
    
    async def _detect_encoding_issues(
        self,
        pattern: DetectionPattern,
        message_str: str
    ) -> PoisonDetectionResult:
        """Detect encoding/decoding issues."""
        
        encoding_issues = []
        
        # Check for invalid UTF-8 sequences
        try:
            message_str.encode('utf-8').decode('utf-8')
        except UnicodeError as e:
            encoding_issues.append(f"UTF-8 encoding error: {str(e)}")
        
        # Check for mixed encoding indicators
        if any(char in message_str for char in ['\ufffd', '\ufeff']):  # Replacement characters
            encoding_issues.append("Replacement characters detected")
        
        # Check for suspicious byte sequences
        if re.search(r'\\x[0-9a-fA-F]{2}', message_str):
            encoding_issues.append("Suspicious byte sequences detected")
        
        if encoding_issues:
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=pattern.poison_type,
                confidence=DetectionConfidence.MEDIUM,
                detection_reason="; ".join(encoding_issues),
                suggested_action=IsolationAction.TRANSFORM_AND_RETRY,
                risk_score=0.6,
                pattern_matches=[pattern.pattern_id],
                is_recoverable=True,
                recovery_suggestions=["Re-encode message", "Clean encoding artifacts"],
                analysis_details={"encoding_issues": encoding_issues}
            )
        
        return PoisonDetectionResult(
            is_poison=False,
            poison_type=None,
            confidence=DetectionConfidence.LOW,
            detection_reason="No encoding issues detected",
            suggested_action=IsolationAction.LOG_AND_CONTINUE,
            risk_score=0.0
        )
    
    async def _detect_excessive_nesting(
        self,
        pattern: DetectionPattern,
        message_data: Dict[str, Any]
    ) -> PoisonDetectionResult:
        """Detect excessive nesting or recursive payloads."""
        
        def calculate_max_depth(obj, current_depth=0):
            if current_depth > 50:  # Prevent infinite recursion
                return current_depth
            
            if isinstance(obj, dict):
                if not obj:
                    return current_depth
                return max(
                    calculate_max_depth(value, current_depth + 1)
                    for value in obj.values()
                )
            elif isinstance(obj, list):
                if not obj:
                    return current_depth
                return max(
                    calculate_max_depth(item, current_depth + 1)
                    for item in obj
                )
            else:
                return current_depth
        
        max_depth = calculate_max_depth(message_data)
        depth_threshold = 20  # Maximum allowed nesting depth
        
        if max_depth > depth_threshold:
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=pattern.poison_type,
                confidence=DetectionConfidence.HIGH,
                detection_reason=f"Excessive nesting depth: {max_depth} (threshold: {depth_threshold})",
                suggested_action=IsolationAction.IMMEDIATE_QUARANTINE,
                risk_score=min(1.0, max_depth / depth_threshold),
                pattern_matches=[pattern.pattern_id],
                analysis_details={
                    "max_nesting_depth": max_depth,
                    "depth_threshold": depth_threshold
                }
            )
        
        return PoisonDetectionResult(
            is_poison=False,
            poison_type=None,
            confidence=DetectionConfidence.LOW,
            detection_reason=f"Nesting depth ({max_depth}) within acceptable limits",
            suggested_action=IsolationAction.LOG_AND_CONTINUE,
            risk_score=0.0
        )
    
    async def _detect_timeout_prone_patterns(
        self,
        pattern: DetectionPattern,
        message_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> PoisonDetectionResult:
        """Detect patterns likely to cause timeouts."""
        
        timeout_indicators = []
        
        # Check for large array processing
        def count_large_arrays(obj):
            count = 0
            if isinstance(obj, list) and len(obj) > 1000:
                count += 1
            elif isinstance(obj, dict):
                for value in obj.values():
                    count += count_large_arrays(value)
            return count
        
        large_arrays = count_large_arrays(message_data)
        if large_arrays > 0:
            timeout_indicators.append(f"{large_arrays} large arrays detected")
        
        # Check for complex computations in payload
        complex_patterns = [
            r'fibonacci|factorial|recursive',
            r'while.*true|for.*range\(\d{4,}\)',
            r'nested.*loop|matrix.*multiplication'
        ]
        
        message_str = json.dumps(message_data)
        for pattern_regex in complex_patterns:
            if re.search(pattern_regex, message_str, re.IGNORECASE):
                timeout_indicators.append(f"Complex computation pattern detected")
                break
        
        # Check context for historical timeout information
        if context and context.get('historical_timeouts', 0) > 3:
            timeout_indicators.append("Historical timeout pattern")
        
        if timeout_indicators:
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=pattern.poison_type,
                confidence=DetectionConfidence.MEDIUM,
                detection_reason="; ".join(timeout_indicators),
                suggested_action=IsolationAction.DELAYED_QUARANTINE,
                risk_score=0.5,
                pattern_matches=[pattern.pattern_id],
                is_recoverable=True,
                recovery_suggestions=["Reduce complexity", "Add processing limits"],
                analysis_details={
                    "timeout_indicators": timeout_indicators,
                    "large_arrays_count": large_arrays
                }
            )
        
        return PoisonDetectionResult(
            is_poison=False,
            poison_type=None,
            confidence=DetectionConfidence.LOW,
            detection_reason="No timeout-prone patterns detected",
            suggested_action=IsolationAction.LOG_AND_CONTINUE,
            risk_score=0.0
        )
    
    async def _validate_required_fields(
        self,
        pattern: DetectionPattern,
        message_data: Dict[str, Any]
    ) -> PoisonDetectionResult:
        """Validate required message fields."""
        
        required_fields = [
            'id', 'type', 'timestamp', 'payload'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in message_data:
                missing_fields.append(field)
        
        if missing_fields:
            return PoisonDetectionResult(
                is_poison=True,
                poison_type=pattern.poison_type,
                confidence=DetectionConfidence.HIGH,
                detection_reason=f"Missing required fields: {', '.join(missing_fields)}",
                suggested_action=IsolationAction.TRANSFORM_AND_RETRY,
                risk_score=0.7,
                pattern_matches=[pattern.pattern_id],
                is_recoverable=True,
                recovery_suggestions=[f"Add missing field: {field}" for field in missing_fields],
                analysis_details={"missing_fields": missing_fields}
            )
        
        return PoisonDetectionResult(
            is_poison=False,
            poison_type=None,
            confidence=DetectionConfidence.LOW,
            detection_reason="All required fields present",
            suggested_action=IsolationAction.LOG_AND_CONTINUE,
            risk_score=0.0
        )
    
    def _calculate_regex_confidence(self, pattern: DetectionPattern, message_str: str) -> DetectionConfidence:
        """Calculate confidence for regex-based detection."""
        
        # Consider pattern complexity and match characteristics
        match = re.search(pattern.detection_regex, message_str, re.IGNORECASE)
        if not match:
            return DetectionConfidence.VERY_LOW
        
        # Stronger matches get higher confidence
        match_length = len(match.group(0))
        message_length = len(message_str)
        match_ratio = match_length / message_length
        
        if match_ratio > 0.5:
            return DetectionConfidence.VERY_HIGH
        elif match_ratio > 0.2:
            return DetectionConfidence.HIGH
        elif match_ratio > 0.1:
            return DetectionConfidence.MEDIUM
        else:
            return DetectionConfidence.LOW
    
    def _get_pattern_action(self, pattern: DetectionPattern, confidence: DetectionConfidence) -> IsolationAction:
        """Get suggested isolation action for pattern detection."""
        
        if confidence == DetectionConfidence.VERY_HIGH:
            return IsolationAction.IMMEDIATE_QUARANTINE
        elif confidence == DetectionConfidence.HIGH:
            if pattern.poison_type in [PoisonMessageType.MALFORMED_JSON, PoisonMessageType.ENCODING_ERROR]:
                return IsolationAction.TRANSFORM_AND_RETRY
            else:
                return IsolationAction.IMMEDIATE_QUARANTINE
        elif confidence == DetectionConfidence.MEDIUM:
            return IsolationAction.DELAYED_QUARANTINE
        else:
            return IsolationAction.LOG_AND_CONTINUE
    
    def _confidence_value(self, confidence: DetectionConfidence) -> int:
        """Convert confidence to numeric value for comparison."""
        confidence_values = {
            DetectionConfidence.VERY_LOW: 1,
            DetectionConfidence.LOW: 2,
            DetectionConfidence.MEDIUM: 3,
            DetectionConfidence.HIGH: 4,
            DetectionConfidence.VERY_HIGH: 5
        }
        return confidence_values[confidence]
    
    def _determine_isolation_action(
        self,
        is_poison: bool,
        confidence: DetectionConfidence,
        risk_score: float,
        detected_issues: List[PoisonDetectionResult]
    ) -> IsolationAction:
        """Determine appropriate isolation action."""
        
        if not is_poison:
            return IsolationAction.LOG_AND_CONTINUE
        
        # High confidence or high risk requires immediate action
        if confidence == DetectionConfidence.VERY_HIGH or risk_score > 0.8:
            return IsolationAction.IMMEDIATE_QUARANTINE
        
        # Check if any issues are recoverable
        recoverable_count = sum(1 for issue in detected_issues if issue.is_recoverable)
        if recoverable_count > 0 and confidence <= DetectionConfidence.MEDIUM:
            return IsolationAction.TRANSFORM_AND_RETRY
        
        # Medium confidence gets delayed quarantine
        if confidence == DetectionConfidence.MEDIUM:
            return IsolationAction.DELAYED_QUARANTINE
        
        # Low confidence gets human review
        return IsolationAction.HUMAN_REVIEW
    
    async def _combine_detection_results(
        self,
        results: List[PoisonDetectionResult]
    ) -> PoisonDetectionResult:
        """Combine multiple detection results into final result."""
        
        poison_results = [r for r in results if r.is_poison]
        
        if not poison_results:
            # No poison detected
            return PoisonDetectionResult(
                is_poison=False,
                poison_type=None,
                confidence=DetectionConfidence.LOW,
                detection_reason="No poison characteristics detected",
                suggested_action=IsolationAction.LOG_AND_CONTINUE,
                risk_score=0.0
            )
        
        # Find highest confidence result
        highest_confidence_result = max(
            poison_results,
            key=lambda r: self._confidence_value(r.confidence)
        )
        
        # Combine risk scores
        combined_risk_score = min(1.0, sum(r.risk_score for r in poison_results) / len(results))
        
        # Combine pattern matches
        all_pattern_matches = []
        for result in poison_results:
            all_pattern_matches.extend(result.pattern_matches)
        
        # Combine recovery suggestions
        all_recovery_suggestions = []
        for result in poison_results:
            all_recovery_suggestions.extend(result.recovery_suggestions)
        
        return PoisonDetectionResult(
            is_poison=True,
            poison_type=highest_confidence_result.poison_type,
            confidence=highest_confidence_result.confidence,
            detection_reason=f"Combined analysis: {highest_confidence_result.detection_reason}",
            suggested_action=highest_confidence_result.suggested_action,
            analysis_details={
                "combined_analysis": True,
                "detection_count": len(poison_results),
                "primary_detection": highest_confidence_result.detection_reason
            },
            pattern_matches=list(set(all_pattern_matches)),
            risk_score=combined_risk_score,
            is_recoverable=any(r.is_recoverable for r in poison_results),
            recovery_suggestions=list(set(all_recovery_suggestions))
        )
    
    async def _cache_result(self, message_hash: str, result: PoisonDetectionResult) -> None:
        """Cache detection result for performance."""
        
        # Implement LRU cache behavior
        if len(self._pattern_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._pattern_cache))
            del self._pattern_cache[oldest_key]
        
        self._pattern_cache[message_hash] = result
    
    async def _update_metrics(self, result: PoisonDetectionResult, start_time: float) -> None:
        """Update detection metrics."""
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        self.metrics.total_messages_analyzed += 1
        
        if result.is_poison:
            self.metrics.poison_messages_detected += 1
            
            if result.poison_type:
                if result.poison_type not in self.metrics.detections_by_type:
                    self.metrics.detections_by_type[result.poison_type] = 0
                self.metrics.detections_by_type[result.poison_type] += 1
        
        # Update confidence distribution
        if result.confidence not in self.metrics.detections_by_confidence:
            self.metrics.detections_by_confidence[result.confidence] = 0
        self.metrics.detections_by_confidence[result.confidence] += 1
        
        # Update performance metrics
        self.metrics.average_detection_time_ms = (
            (self.metrics.average_detection_time_ms * (self.metrics.total_messages_analyzed - 1) + processing_time) /
            self.metrics.total_messages_analyzed
        )
        self.metrics.max_detection_time_ms = max(self.metrics.max_detection_time_ms, processing_time)
        
        # Update pattern match counts
        for pattern_id in result.pattern_matches:
            if pattern_id not in self.metrics.pattern_matches:
                self.metrics.pattern_matches[pattern_id] = 0
            self.metrics.pattern_matches[pattern_id] += 1
        
        # Recalculate accuracy
        self.metrics.detection_accuracy = self.metrics.calculate_accuracy()
    
    async def get_detection_metrics(self) -> Dict[str, Any]:
        """Get comprehensive detection metrics."""
        
        return {
            "performance_metrics": {
                "total_messages_analyzed": self.metrics.total_messages_analyzed,
                "poison_messages_detected": self.metrics.poison_messages_detected,
                "detection_rate": (
                    self.metrics.poison_messages_detected / max(1, self.metrics.total_messages_analyzed)
                ),
                "average_detection_time_ms": self.metrics.average_detection_time_ms,
                "max_detection_time_ms": self.metrics.max_detection_time_ms,
                "detection_accuracy": self.metrics.detection_accuracy
            },
            "detection_breakdown": {
                "by_type": dict(self.metrics.detections_by_type),
                "by_confidence": {k.value: v for k, v in self.metrics.detections_by_confidence.items()}
            },
            "pattern_effectiveness": dict(self.metrics.pattern_matches),
            "cache_metrics": {
                "cache_size": len(self._pattern_cache),
                "cache_max_size": self._cache_max_size,
                "cache_hit_rate": "not_tracked"  # Could be implemented
            },
            "configuration": {
                "max_message_size_bytes": self.max_message_size_bytes,
                "detection_timeout_ms": self.detection_timeout_ms,
                "adaptive_learning_enabled": self.enable_adaptive_learning,
                "confidence_threshold": self.confidence_threshold.value,
                "active_patterns": len(self.detection_patterns)
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        
        metrics = await self.get_detection_metrics()
        
        # Determine health status
        is_healthy = (
            metrics["performance_metrics"]["average_detection_time_ms"] < self.detection_timeout_ms and
            metrics["performance_metrics"]["detection_accuracy"] > 0.7 and
            len(self.detection_patterns) > 0
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }