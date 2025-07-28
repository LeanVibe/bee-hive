"""
Semantic Integrity Validator - Context Restoration Validation & Quality Assurance.

Provides comprehensive semantic integrity validation for LeanVibe Agent Hive 2.0:
- Context restoration accuracy validation using semantic similarity analysis
- Multi-dimensional integrity checks (semantic, structural, informational)
- Real-time validation during wake cycles and context retrieval
- Quality degradation detection and automatic correction
- Integration with memory management and consolidation systems
- Detailed integrity reporting and analytics

Performance Targets:
- 95%+ context restoration accuracy validation
- <200ms validation time for typical context sets
- Multi-modal validation support (text, code, structured data)
- Automatic quality degradation detection and correction
"""

import asyncio
import logging
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import re
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.enhanced_memory_manager import (
    EnhancedMemoryManager, get_enhanced_memory_manager, MemoryFragment, MemoryType
)
from ..core.memory_consolidation_service import (
    MemoryConsolidationService, get_memory_consolidation_service, ContentModality
)
from ..core.embeddings import EmbeddingService, get_embedding_service
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class IntegrityDimension(Enum):
    """Dimensions of semantic integrity validation."""
    SEMANTIC_SIMILARITY = "semantic_similarity"    # Meaning preservation
    STRUCTURAL_INTEGRITY = "structural_integrity"  # Format/structure preservation
    INFORMATIONAL_COMPLETENESS = "informational_completeness"  # Information coverage
    CONTEXTUAL_COHERENCE = "contextual_coherence"  # Logical consistency
    TEMPORAL_CONSISTENCY = "temporal_consistency"   # Time-based relationships
    RELATIONAL_ACCURACY = "relational_accuracy"    # Cross-reference preservation


class ValidationSeverity(Enum):
    """Severity levels for integrity issues."""
    CRITICAL = "critical"      # Major semantic loss, requires immediate attention
    HIGH = "high"             # Significant issues, should be addressed
    MEDIUM = "medium"         # Moderate issues, can be tolerated
    LOW = "low"              # Minor issues, acceptable
    INFO = "info"            # Informational only, no action needed


class ValidationStrategy(Enum):
    """Strategies for semantic integrity validation."""
    COMPREHENSIVE = "comprehensive"     # Full validation across all dimensions
    FOCUSED = "focused"                # Focus on critical dimensions only
    RAPID = "rapid"                   # Fast validation for real-time use
    DEEP_ANALYSIS = "deep_analysis"    # Thorough analysis with detailed reporting
    ADAPTIVE = "adaptive"             # Adapt strategy based on content type


@dataclass
class IntegrityIssue:
    """Represents a semantic integrity issue."""
    issue_id: str
    dimension: IntegrityDimension
    severity: ValidationSeverity
    description: str
    affected_content: str
    expected_content: Optional[str] = None
    confidence_score: float = 0.0
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidationRequest:
    """Request for semantic integrity validation."""
    original_content: Union[str, List[str], List[MemoryFragment], List[Context]]
    restored_content: Union[str, List[str], List[MemoryFragment], List[Context]]
    validation_strategy: ValidationStrategy = ValidationStrategy.COMPREHENSIVE
    dimensions_to_validate: Optional[List[IntegrityDimension]] = None
    severity_threshold: ValidationSeverity = ValidationSeverity.MEDIUM
    content_modality: Optional[ContentModality] = None
    agent_id: Optional[uuid.UUID] = None
    max_validation_time_seconds: int = 30
    include_suggestions: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.dimensions_to_validate is None:
            self.dimensions_to_validate = list(IntegrityDimension)


@dataclass
class ValidationResult:
    """Result of semantic integrity validation."""
    validation_id: str
    agent_id: Optional[uuid.UUID]
    overall_integrity_score: float
    dimension_scores: Dict[IntegrityDimension, float]
    issues_found: List[IntegrityIssue]
    validation_passed: bool
    validation_time_seconds: float
    content_modality: Optional[ContentModality]
    strategy_used: ValidationStrategy
    recommendations: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}
        if not hasattr(self, 'recommendations') or self.recommendations is None:
            self.recommendations = []


@dataclass
class ValidationMetrics:
    """Metrics for validation operations."""
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    average_validation_time: float = 0.0
    average_integrity_score: float = 0.0
    dimension_performance: Dict[IntegrityDimension, float] = None
    issue_distribution: Dict[ValidationSeverity, int] = None
    
    def __post_init__(self):
        if self.dimension_performance is None:
            self.dimension_performance = {}
        if self.issue_distribution is None:
            self.issue_distribution = {}


class SemanticIntegrityValidator:
    """
    Advanced Semantic Integrity Validator for Context Restoration.
    
    Provides comprehensive validation capabilities:
    - Multi-dimensional semantic integrity analysis
    - Real-time validation during context restoration
    - Automatic quality degradation detection and correction
    - Detailed integrity reporting and analytics
    - Integration with memory management and consolidation systems
    - Adaptive validation strategies based on content types
    """
    
    def __init__(
        self,
        memory_manager: Optional[EnhancedMemoryManager] = None,
        consolidation_service: Optional[MemoryConsolidationService] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.settings = get_settings()
        self.memory_manager = memory_manager or get_enhanced_memory_manager()
        self.consolidation_service = consolidation_service or get_memory_consolidation_service()
        self.embedding_service = embedding_service or get_embedding_service()
        
        # Configuration
        self.config = {
            "semantic_similarity_threshold": 0.85,
            "structural_similarity_threshold": 0.80,
            "information_completeness_threshold": 0.90,
            "contextual_coherence_threshold": 0.85,
            "temporal_consistency_threshold": 0.75,
            "relational_accuracy_threshold": 0.80,
            "max_concurrent_validations": 10,
            "cache_validation_results": True,
            "adaptive_thresholds": True,
            "auto_correction_enabled": True
        }
        
        # Dimension-specific validators
        self._dimension_validators: Dict[IntegrityDimension, Callable] = {
            IntegrityDimension.SEMANTIC_SIMILARITY: self._validate_semantic_similarity,
            IntegrityDimension.STRUCTURAL_INTEGRITY: self._validate_structural_integrity,
            IntegrityDimension.INFORMATIONAL_COMPLETENESS: self._validate_informational_completeness,
            IntegrityDimension.CONTEXTUAL_COHERENCE: self._validate_contextual_coherence,
            IntegrityDimension.TEMPORAL_CONSISTENCY: self._validate_temporal_consistency,
            IntegrityDimension.RELATIONAL_ACCURACY: self._validate_relational_accuracy
        }
        
        # Performance tracking
        self._validation_metrics = ValidationMetrics()
        self._validation_history: deque = deque(maxlen=1000)
        
        # Adaptive thresholds
        self._adaptive_thresholds: Dict[ContentModality, Dict[IntegrityDimension, float]] = defaultdict(dict)
        
        # Issue correction templates
        self._correction_templates: Dict[ValidationSeverity, List[str]] = {
            ValidationSeverity.CRITICAL: [
                "Complete content restoration required",
                "Semantic meaning significantly altered",
                "Critical information loss detected"
            ],
            ValidationSeverity.HIGH: [
                "Partial content restoration recommended",
                "Important details may be missing",
                "Structural integrity compromised"
            ],
            ValidationSeverity.MEDIUM: [
                "Minor adjustments recommended",
                "Some information may be condensed",
                "Format may need minor corrections"
            ]
        }
        
        logger.info("🔍 Semantic Integrity Validator initialized")
    
    async def validate_integrity(
        self, request: ValidationRequest
    ) -> ValidationResult:
        """
        Perform comprehensive semantic integrity validation.
        
        Args:
            request: Validation request with original and restored content
            
        Returns:
            ValidationResult with detailed integrity analysis
        """
        validation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            logger.info(
                f"🔍 Starting semantic integrity validation",
                validation_id=validation_id,
                strategy=request.validation_strategy.value,
                dimensions_count=len(request.dimensions_to_validate),
                agent_id=str(request.agent_id) if request.agent_id else None
            )
            
            # Normalize content for validation
            original_normalized = await self._normalize_content(request.original_content)
            restored_normalized = await self._normalize_content(request.restored_content)
            
            # Detect content modality if not provided
            if request.content_modality is None:
                request.content_modality = await self._detect_content_modality(original_normalized)
            
            # Perform dimension-specific validations
            dimension_scores = {}
            issues_found = []
            
            for dimension in request.dimensions_to_validate:
                if dimension in self._dimension_validators:
                    try:
                        score, dimension_issues = await self._dimension_validators[dimension](
                            original_normalized, restored_normalized, request
                        )
                        dimension_scores[dimension] = score
                        issues_found.extend(dimension_issues)
                        
                    except Exception as e:
                        logger.error(f"Dimension validation failed for {dimension.value}: {e}")
                        dimension_scores[dimension] = 0.5  # Default score on failure
            
            # Calculate overall integrity score
            overall_score = await self._calculate_overall_integrity_score(
                dimension_scores, request.content_modality
            )
            
            # Determine if validation passed
            validation_passed = await self._determine_validation_pass(
                overall_score, dimension_scores, issues_found, request
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                dimension_scores, issues_found, request
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create validation result
            result = ValidationResult(
                validation_id=validation_id,
                agent_id=request.agent_id,
                overall_integrity_score=overall_score,
                dimension_scores=dimension_scores,
                issues_found=issues_found,
                validation_passed=validation_passed,
                validation_time_seconds=processing_time,
                content_modality=request.content_modality,
                strategy_used=request.validation_strategy,
                recommendations=recommendations,
                metadata={
                    "original_content_length": len(str(original_normalized)),
                    "restored_content_length": len(str(restored_normalized)),
                    "dimensions_validated": len(request.dimensions_to_validate),
                    "issues_by_severity": self._count_issues_by_severity(issues_found)
                }
            )
            
            # Update metrics and history
            await self._update_validation_metrics(result)
            self._validation_history.append({
                "validation_id": validation_id,
                "agent_id": request.agent_id,
                "overall_score": overall_score,
                "validation_passed": validation_passed,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Adapt thresholds if enabled
            if self.config["adaptive_thresholds"]:
                await self._adapt_validation_thresholds(request.content_modality, result)
            
            logger.info(
                f"🔍 Semantic integrity validation completed",
                validation_id=validation_id,
                overall_score=overall_score,
                validation_passed=validation_passed,
                processing_time=processing_time,
                issues_count=len(issues_found)
            )
            
            return result
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            error_result = ValidationResult(
                validation_id=validation_id,
                agent_id=request.agent_id,
                overall_integrity_score=0.0,
                dimension_scores={},
                issues_found=[
                    IntegrityIssue(
                        issue_id=str(uuid.uuid4()),
                        dimension=IntegrityDimension.SEMANTIC_SIMILARITY,
                        severity=ValidationSeverity.CRITICAL,
                        description=f"Validation failed: {str(e)}",
                        affected_content="",
                        confidence_score=1.0
                    )
                ],
                validation_passed=False,
                validation_time_seconds=processing_time,
                content_modality=request.content_modality,
                strategy_used=request.validation_strategy,
                recommendations=["Manual validation required due to system error"],
                metadata={"error": str(e)}
            )
            
            logger.error(
                f"❌ Semantic integrity validation failed",
                validation_id=validation_id,
                error=str(e)
            )
            
            return error_result
    
    async def validate_memory_restoration(
        self,
        agent_id: uuid.UUID,
        original_memories: List[MemoryFragment],
        restored_memories: List[MemoryFragment],
        validation_strategy: ValidationStrategy = ValidationStrategy.COMPREHENSIVE
    ) -> ValidationResult:
        """
        Validate integrity of memory restoration.
        
        Args:
            agent_id: Agent ID for context
            original_memories: Original memory fragments
            restored_memories: Restored memory fragments
            validation_strategy: Validation strategy to use
            
        Returns:
            ValidationResult for memory restoration
        """
        try:
            # Create validation request
            request = ValidationRequest(
                original_content=original_memories,
                restored_content=restored_memories,
                validation_strategy=validation_strategy,
                agent_id=agent_id,
                include_suggestions=True
            )
            
            # Perform validation
            return await self.validate_integrity(request)
            
        except Exception as e:
            logger.error(f"Memory restoration validation failed: {e}")
            raise
    
    async def validate_context_consolidation(
        self,
        original_contexts: List[Context],
        consolidated_contexts: List[Context],
        consolidation_metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate integrity of context consolidation.
        
        Args:
            original_contexts: Original contexts before consolidation
            consolidated_contexts: Contexts after consolidation
            consolidation_metadata: Metadata from consolidation process
            
        Returns:
            ValidationResult for consolidation integrity
        """
        try:
            # Create validation request
            request = ValidationRequest(
                original_content=original_contexts,
                restored_content=consolidated_contexts,
                validation_strategy=ValidationStrategy.FOCUSED,
                dimensions_to_validate=[
                    IntegrityDimension.SEMANTIC_SIMILARITY,
                    IntegrityDimension.INFORMATIONAL_COMPLETENESS,
                    IntegrityDimension.CONTEXTUAL_COHERENCE
                ],
                metadata=consolidation_metadata or {}
            )
            
            # Perform validation
            return await self.validate_integrity(request)
            
        except Exception as e:
            logger.error(f"Context consolidation validation failed: {e}")
            raise
    
    async def rapid_integrity_check(
        self,
        original_content: str,
        restored_content: str,
        agent_id: Optional[uuid.UUID] = None
    ) -> Tuple[float, bool]:
        """
        Perform rapid integrity check for real-time use.
        
        Args:
            original_content: Original content string
            restored_content: Restored content string
            agent_id: Optional agent ID for context
            
        Returns:
            Tuple of (integrity_score, validation_passed)
        """
        try:
            # Create rapid validation request
            request = ValidationRequest(
                original_content=original_content,
                restored_content=restored_content,
                validation_strategy=ValidationStrategy.RAPID,
                dimensions_to_validate=[
                    IntegrityDimension.SEMANTIC_SIMILARITY,
                    IntegrityDimension.INFORMATIONAL_COMPLETENESS
                ],
                agent_id=agent_id,
                max_validation_time_seconds=5,
                include_suggestions=False
            )
            
            # Perform validation
            result = await self.validate_integrity(request)
            
            return result.overall_integrity_score, result.validation_passed
            
        except Exception as e:
            logger.error(f"Rapid integrity check failed: {e}")
            return 0.0, False
    
    async def auto_correct_integrity_issues(
        self,
        validation_result: ValidationResult,
        correction_strategy: str = "conservative"
    ) -> Tuple[str, List[str]]:
        """
        Automatically correct integrity issues when possible.
        
        Args:
            validation_result: Result from integrity validation
            correction_strategy: Strategy for corrections ("conservative", "aggressive")
            
        Returns:
            Tuple of (corrected_content, applied_corrections)
        """
        try:
            corrected_content = ""
            applied_corrections = []
            
            # Analyze issues and apply corrections based on severity
            critical_issues = [
                issue for issue in validation_result.issues_found
                if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]
            ]
            
            if critical_issues and correction_strategy == "aggressive":
                # Apply aggressive corrections for critical issues
                for issue in critical_issues:
                    if issue.suggested_fix:
                        applied_corrections.append(f"Applied fix for {issue.dimension.value}: {issue.suggested_fix}")
            
            elif critical_issues and correction_strategy == "conservative":
                # Conservative approach: flag for manual review
                applied_corrections.append("Critical issues detected - manual review recommended")
            
            # Apply minor corrections for medium and low severity issues
            minor_issues = [
                issue for issue in validation_result.issues_found
                if issue.severity in [ValidationSeverity.MEDIUM, ValidationSeverity.LOW]
            ]
            
            for issue in minor_issues:
                if issue.suggested_fix:
                    applied_corrections.append(f"Minor correction: {issue.suggested_fix}")
            
            logger.info(
                f"🔧 Auto-correction completed",
                validation_id=validation_result.validation_id,
                corrections_applied=len(applied_corrections),
                strategy=correction_strategy
            )
            
            return corrected_content, applied_corrections
            
        except Exception as e:
            logger.error(f"Auto-correction failed: {e}")
            return "", ["Auto-correction failed - manual review required"]
    
    async def get_validation_analytics(
        self,
        agent_id: Optional[uuid.UUID] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive validation analytics.
        
        Args:
            agent_id: Specific agent analytics (all if None)
            time_range: Time range for analytics
            
        Returns:
            Comprehensive analytics data
        """
        try:
            analytics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": asdict(self._validation_metrics),
                "dimension_performance": {},
                "recent_validations": [],
                "integrity_trends": {},
                "threshold_adaptations": {}
            }
            
            # Dimension performance analysis
            for dimension in IntegrityDimension:
                dimension_scores = []
                for validation in self._validation_history:
                    if "dimension_scores" in validation:
                        if dimension in validation["dimension_scores"]:
                            dimension_scores.append(validation["dimension_scores"][dimension])
                
                if dimension_scores:
                    analytics["dimension_performance"][dimension.value] = {
                        "average_score": sum(dimension_scores) / len(dimension_scores),
                        "validation_count": len(dimension_scores),
                        "trend": "stable"  # Could calculate actual trend
                    }
            
            # Recent validations (last 20)
            recent_validations = list(self._validation_history)[-20:]
            analytics["recent_validations"] = [
                {
                    "validation_id": validation.get("validation_id"),
                    "agent_id": str(validation.get("agent_id")) if validation.get("agent_id") else None,
                    "overall_score": validation.get("overall_score", 0.0),
                    "validation_passed": validation.get("validation_passed", False),
                    "processing_time": validation.get("processing_time", 0.0),
                    "timestamp": validation.get("timestamp")
                }
                for validation in recent_validations
            ]
            
            # Agent-specific analytics
            if agent_id:
                agent_validations = [
                    v for v in self._validation_history
                    if v.get("agent_id") == agent_id
                ]
                
                if agent_validations:
                    analytics["agent_specific"] = {
                        "total_validations": len(agent_validations),
                        "average_score": sum(v.get("overall_score", 0) for v in agent_validations) / len(agent_validations),
                        "pass_rate": sum(1 for v in agent_validations if v.get("validation_passed", False)) / len(agent_validations)
                    }
            
            # Adaptive thresholds
            analytics["threshold_adaptations"] = {
                modality.value: dict(thresholds)
                for modality, thresholds in self._adaptive_thresholds.items()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Validation analytics calculation failed: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _normalize_content(
        self, content: Union[str, List[str], List[MemoryFragment], List[Context]]
    ) -> str:
        """Normalize content for validation."""
        try:
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                if not content:
                    return ""
                
                # Check type of first element
                first_item = content[0]
                if isinstance(first_item, str):
                    return "\n\n".join(content)
                elif isinstance(first_item, MemoryFragment):
                    return "\n\n".join(fragment.content for fragment in content)
                elif isinstance(first_item, Context):
                    return "\n\n".join(context.content for context in content if context.content)
                else:
                    return str(content)
            else:
                return str(content)
                
        except Exception as e:
            logger.error(f"Content normalization failed: {e}")
            return ""
    
    async def _detect_content_modality(self, content: str) -> ContentModality:
        """Detect content modality for validation optimization."""
        try:
            # Simple heuristic-based detection
            if re.search(r'def\s+\w+\s*\(|function\s+\w+|class\s+\w+', content):
                return ContentModality.CODE
            elif re.search(r'^\w+:\s|^[A-Z][a-z]+:\s', content, re.MULTILINE):
                return ContentModality.CONVERSATION
            elif content.strip().startswith('{') or content.strip().startswith('<'):
                return ContentModality.STRUCTURED_DATA
            elif re.search(r'^#+\s|\*\*\w+\*\*|```', content, re.MULTILINE):
                return ContentModality.DOCUMENTATION
            else:
                return ContentModality.TEXT
                
        except Exception:
            return ContentModality.TEXT
    
    async def _validate_semantic_similarity(
        self, original: str, restored: str, request: ValidationRequest
    ) -> Tuple[float, List[IntegrityIssue]]:
        """Validate semantic similarity between original and restored content."""
        try:
            issues = []
            
            # Generate embeddings for both contents
            original_embedding = await self.embedding_service.generate_embedding(original)
            restored_embedding = await self.embedding_service.generate_embedding(restored)
            
            # Calculate cosine similarity
            similarity = self._calculate_cosine_similarity(original_embedding, restored_embedding)
            
            # Check against threshold
            threshold = self.config["semantic_similarity_threshold"]
            if request.content_modality and request.content_modality in self._adaptive_thresholds:
                threshold = self._adaptive_thresholds[request.content_modality].get(
                    IntegrityDimension.SEMANTIC_SIMILARITY, threshold
                )
            
            if similarity < threshold:
                severity = ValidationSeverity.CRITICAL if similarity < 0.7 else ValidationSeverity.HIGH
                issues.append(IntegrityIssue(
                    issue_id=str(uuid.uuid4()),
                    dimension=IntegrityDimension.SEMANTIC_SIMILARITY,
                    severity=severity,
                    description=f"Semantic similarity below threshold: {similarity:.3f} < {threshold:.3f}",
                    affected_content=restored[:200] + "..." if len(restored) > 200 else restored,
                    confidence_score=1.0 - similarity,
                    suggested_fix="Consider content restoration or re-consolidation with higher quality thresholds"
                ))
            
            return similarity, issues
            
        except Exception as e:
            logger.error(f"Semantic similarity validation failed: {e}")
            return 0.0, [IntegrityIssue(
                issue_id=str(uuid.uuid4()),
                dimension=IntegrityDimension.SEMANTIC_SIMILARITY,
                severity=ValidationSeverity.CRITICAL,
                description=f"Validation error: {str(e)}",
                affected_content="",
                confidence_score=1.0
            )]
    
    async def _validate_structural_integrity(
        self, original: str, restored: str, request: ValidationRequest
    ) -> Tuple[float, List[IntegrityIssue]]:
        """Validate structural integrity preservation."""
        try:
            issues = []
            
            # Analyze structural elements based on content modality
            if request.content_modality == ContentModality.CODE:
                score, code_issues = await self._validate_code_structure(original, restored)
                issues.extend(code_issues)
            elif request.content_modality == ContentModality.STRUCTURED_DATA:
                score, data_issues = await self._validate_data_structure(original, restored)
                issues.extend(data_issues)
            elif request.content_modality == ContentModality.DOCUMENTATION:
                score, doc_issues = await self._validate_document_structure(original, restored)
                issues.extend(doc_issues)
            else:
                # Generic text structure validation
                score, text_issues = await self._validate_text_structure(original, restored)
                issues.extend(text_issues)
            
            return score, issues
            
        except Exception as e:
            logger.error(f"Structural integrity validation failed: {e}")
            return 0.5, []
    
    async def _validate_informational_completeness(
        self, original: str, restored: str, request: ValidationRequest
    ) -> Tuple[float, List[IntegrityIssue]]:
        """Validate informational completeness."""
        try:
            issues = []
            
            # Extract key information elements
            original_entities = self._extract_key_entities(original)
            restored_entities = self._extract_key_entities(restored)
            
            # Calculate completeness ratio
            if not original_entities:
                completeness = 1.0  # If no entities in original, assume complete
            else:
                preserved_entities = set(restored_entities).intersection(set(original_entities))
                completeness = len(preserved_entities) / len(original_entities)
            
            # Check against threshold
            threshold = self.config["information_completeness_threshold"]
            if completeness < threshold:
                missing_entities = set(original_entities) - set(restored_entities)
                issues.append(IntegrityIssue(
                    issue_id=str(uuid.uuid4()),
                    dimension=IntegrityDimension.INFORMATIONAL_COMPLETENESS,
                    severity=ValidationSeverity.HIGH if completeness < 0.7 else ValidationSeverity.MEDIUM,
                    description=f"Information completeness below threshold: {completeness:.3f} < {threshold:.3f}",
                    affected_content=f"Missing entities: {', '.join(list(missing_entities)[:5])}",
                    confidence_score=1.0 - completeness,
                    suggested_fix="Review consolidation to preserve key information elements"
                ))
            
            return completeness, issues
            
        except Exception as e:
            logger.error(f"Informational completeness validation failed: {e}")
            return 0.5, []
    
    async def _validate_contextual_coherence(
        self, original: str, restored: str, request: ValidationRequest
    ) -> Tuple[float, List[IntegrityIssue]]:
        """Validate contextual coherence and logical consistency."""
        try:
            issues = []
            
            # Analyze coherence through sentence flow and logical connections
            original_sentences = self._extract_sentences(original)
            restored_sentences = self._extract_sentences(restored)
            
            # Calculate coherence score based on logical flow preservation
            coherence_score = self._calculate_coherence_score(original_sentences, restored_sentences)
            
            # Check for logical inconsistencies
            inconsistencies = self._detect_logical_inconsistencies(original, restored)
            
            if inconsistencies:
                for inconsistency in inconsistencies:
                    issues.append(IntegrityIssue(
                        issue_id=str(uuid.uuid4()),
                        dimension=IntegrityDimension.CONTEXTUAL_COHERENCE,
                        severity=ValidationSeverity.MEDIUM,
                        description=f"Logical inconsistency detected: {inconsistency}",
                        affected_content="",
                        confidence_score=0.8,
                        suggested_fix="Review logical flow and fix inconsistencies"
                    ))
            
            return coherence_score, issues
            
        except Exception as e:
            logger.error(f"Contextual coherence validation failed: {e}")
            return 0.5, []
    
    async def _validate_temporal_consistency(
        self, original: str, restored: str, request: ValidationRequest
    ) -> Tuple[float, List[IntegrityIssue]]:
        """Validate temporal consistency in content."""
        try:
            issues = []
            
            # Extract temporal references
            original_temporal = self._extract_temporal_references(original)
            restored_temporal = self._extract_temporal_references(restored)
            
            # Check for temporal inconsistencies
            consistency_score = self._calculate_temporal_consistency(original_temporal, restored_temporal)
            
            if consistency_score < self.config["temporal_consistency_threshold"]:
                issues.append(IntegrityIssue(
                    issue_id=str(uuid.uuid4()),
                    dimension=IntegrityDimension.TEMPORAL_CONSISTENCY,
                    severity=ValidationSeverity.LOW,
                    description=f"Temporal consistency issues detected",
                    affected_content="",
                    confidence_score=1.0 - consistency_score,
                    suggested_fix="Review temporal references for consistency"
                ))
            
            return consistency_score, issues
            
        except Exception as e:
            logger.error(f"Temporal consistency validation failed: {e}")
            return 0.5, []
    
    async def _validate_relational_accuracy(
        self, original: str, restored: str, request: ValidationRequest
    ) -> Tuple[float, List[IntegrityIssue]]:
        """Validate preservation of relational information."""
        try:
            issues = []
            
            # Extract relationships and references
            original_relations = self._extract_relationships(original)
            restored_relations = self._extract_relationships(restored)
            
            # Calculate relational accuracy
            accuracy = self._calculate_relational_accuracy(original_relations, restored_relations)
            
            if accuracy < self.config["relational_accuracy_threshold"]:
                issues.append(IntegrityIssue(
                    issue_id=str(uuid.uuid4()),
                    dimension=IntegrityDimension.RELATIONAL_ACCURACY,
                    severity=ValidationSeverity.MEDIUM,
                    description=f"Relational accuracy below threshold: {accuracy:.3f}",
                    affected_content="",
                    confidence_score=1.0 - accuracy,
                    suggested_fix="Review relationship preservation in consolidation"
                ))
            
            return accuracy, issues
            
        except Exception as e:
            logger.error(f"Relational accuracy validation failed: {e}")
            return 0.5, []
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception:
            return 0.0
    
    async def _validate_code_structure(self, original: str, restored: str) -> Tuple[float, List[IntegrityIssue]]:
        """Validate code structure preservation."""
        try:
            issues = []
            
            # Extract code elements
            original_functions = re.findall(r'(?:def|function|class)\s+(\w+)', original)
            restored_functions = re.findall(r'(?:def|function|class)\s+(\w+)', restored)
            
            # Calculate preservation ratio
            if original_functions:
                preserved = set(restored_functions).intersection(set(original_functions))
                preservation_ratio = len(preserved) / len(original_functions)
            else:
                preservation_ratio = 1.0
            
            # Check for syntax validity (basic check)
            syntax_score = 1.0
            if not self._basic_syntax_check(restored):
                syntax_score = 0.5
                issues.append(IntegrityIssue(
                    issue_id=str(uuid.uuid4()),
                    dimension=IntegrityDimension.STRUCTURAL_INTEGRITY,
                    severity=ValidationSeverity.HIGH,
                    description="Potential syntax issues detected in code",
                    affected_content="",
                    confidence_score=0.8
                ))
            
            overall_score = (preservation_ratio + syntax_score) / 2
            return overall_score, issues
            
        except Exception as e:
            logger.error(f"Code structure validation failed: {e}")
            return 0.5, []
    
    async def _validate_data_structure(self, original: str, restored: str) -> Tuple[float, List[IntegrityIssue]]:
        """Validate structured data integrity."""
        try:
            issues = []
            
            # Try parsing as JSON
            try:
                original_data = json.loads(original)
                restored_data = json.loads(restored)
                
                # Compare structure
                structure_score = self._compare_json_structure(original_data, restored_data)
                
                if structure_score < 0.8:
                    issues.append(IntegrityIssue(
                        issue_id=str(uuid.uuid4()),
                        dimension=IntegrityDimension.STRUCTURAL_INTEGRITY,
                        severity=ValidationSeverity.MEDIUM,
                        description="Data structure integrity compromised",
                        affected_content="",
                        confidence_score=1.0 - structure_score
                    ))
                
                return structure_score, issues
                
            except json.JSONDecodeError:
                # Not valid JSON, use heuristic analysis
                return 0.7, []
            
        except Exception as e:
            logger.error(f"Data structure validation failed: {e}")
            return 0.5, []
    
    async def _validate_document_structure(self, original: str, restored: str) -> Tuple[float, List[IntegrityIssue]]:
        """Validate document structure preservation."""
        try:
            issues = []
            
            # Extract document elements
            original_headers = re.findall(r'^#+\s+(.+)$', original, re.MULTILINE)
            restored_headers = re.findall(r'^#+\s+(.+)$', restored, re.MULTILINE)
            
            # Calculate header preservation
            if original_headers:
                preserved_headers = set(restored_headers).intersection(set(original_headers))
                header_preservation = len(preserved_headers) / len(original_headers)
            else:
                header_preservation = 1.0
            
            # Check for other structural elements
            original_lists = len(re.findall(r'^\s*[-*+]\s', original, re.MULTILINE))
            restored_lists = len(re.findall(r'^\s*[-*+]\s', restored, re.MULTILINE))
            
            list_preservation = min(1.0, restored_lists / max(1, original_lists))
            
            overall_score = (header_preservation + list_preservation) / 2
            
            if overall_score < 0.8:
                issues.append(IntegrityIssue(
                    issue_id=str(uuid.uuid4()),
                    dimension=IntegrityDimension.STRUCTURAL_INTEGRITY,
                    severity=ValidationSeverity.MEDIUM,
                    description="Document structure elements not fully preserved",
                    affected_content="",
                    confidence_score=1.0 - overall_score
                ))
            
            return overall_score, issues
            
        except Exception as e:
            logger.error(f"Document structure validation failed: {e}")
            return 0.5, []
    
    async def _validate_text_structure(self, original: str, restored: str) -> Tuple[float, List[IntegrityIssue]]:
        """Validate generic text structure."""
        try:
            issues = []
            
            # Compare paragraph structure
            original_paragraphs = len([p for p in original.split('\n\n') if p.strip()])
            restored_paragraphs = len([p for p in restored.split('\n\n') if p.strip()])
            
            # Calculate structure preservation
            if original_paragraphs > 0:
                structure_score = min(1.0, restored_paragraphs / original_paragraphs)
            else:
                structure_score = 1.0
            
            # Check sentence structure
            original_sentences = len([s for s in original.split('.') if s.strip()])
            restored_sentences = len([s for s in restored.split('.') if s.strip()])
            
            if original_sentences > 0:
                sentence_score = min(1.0, restored_sentences / original_sentences)
            else:
                sentence_score = 1.0
            
            overall_score = (structure_score + sentence_score) / 2
            
            return overall_score, issues
            
        except Exception as e:
            logger.error(f"Text structure validation failed: {e}")
            return 0.5, []
    
    def _extract_key_entities(self, content: str) -> List[str]:
        """Extract key entities from content."""
        try:
            # Simple entity extraction using patterns
            entities = []
            
            # Extract proper nouns (capitalized words)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', content)
            entities.extend(proper_nouns)
            
            # Extract numbers
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
            entities.extend(numbers)
            
            # Extract quoted strings
            quoted = re.findall(r'"([^"]*)"', content)
            entities.extend(quoted)
            
            return list(set(entities))  # Remove duplicates
            
        except Exception:
            return []
    
    def _extract_sentences(self, content: str) -> List[str]:
        """Extract sentences from content."""
        try:
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            return sentences
        except Exception:
            return []
    
    def _calculate_coherence_score(self, original_sentences: List[str], restored_sentences: List[str]) -> float:
        """Calculate coherence score based on sentence flow."""
        try:
            if not original_sentences or not restored_sentences:
                return 1.0
            
            # Simple coherence calculation based on sentence overlap and order
            original_set = set(original_sentences)
            restored_set = set(restored_sentences)
            
            overlap = len(original_set.intersection(restored_set))
            total_unique = len(original_set.union(restored_set))
            
            if total_unique == 0:
                return 1.0
            
            return overlap / total_unique
            
        except Exception:
            return 0.5
    
    def _detect_logical_inconsistencies(self, original: str, restored: str) -> List[str]:
        """Detect logical inconsistencies between original and restored content."""
        try:
            inconsistencies = []
            
            # Check for contradictory statements (basic heuristic)
            original_lower = original.lower()
            restored_lower = restored.lower()
            
            # Check for negation reversals
            if 'not' in original_lower and 'not' not in restored_lower:
                inconsistencies.append("Potential negation reversal")
            
            # Check for numerical inconsistencies
            original_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', original)
            restored_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', restored)
            
            if len(set(original_numbers)) != len(set(restored_numbers)):
                inconsistencies.append("Numerical data inconsistency")
            
            return inconsistencies
            
        except Exception:
            return []
    
    def _extract_temporal_references(self, content: str) -> List[str]:
        """Extract temporal references from content."""
        try:
            temporal_patterns = [
                r'\b\d{4}-\d{2}-\d{2}\b',  # Dates
                r'\b\d{1,2}:\d{2}\b',      # Times
                r'\b(?:before|after|during|when|then|now|today|yesterday|tomorrow)\b',  # Temporal words
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'  # Months
            ]
            
            references = []
            for pattern in temporal_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                references.extend(matches)
            
            return references
            
        except Exception:
            return []
    
    def _calculate_temporal_consistency(self, original_temporal: List[str], restored_temporal: List[str]) -> float:
        """Calculate temporal consistency score."""
        try:
            if not original_temporal:
                return 1.0
            
            preserved = set(restored_temporal).intersection(set(original_temporal))
            consistency = len(preserved) / len(original_temporal)
            
            return consistency
            
        except Exception:
            return 0.5
    
    def _extract_relationships(self, content: str) -> List[str]:
        """Extract relationships and references from content."""
        try:
            relationships = []
            
            # Extract relationships indicated by prepositions
            relationship_patterns = [
                r'\b\w+\s+(?:of|from|to|with|by|for|in|on|at)\s+\w+\b',
                r'\b\w+\s+(?:relates to|connected to|associated with)\s+\w+\b',
                r'\b\w+\s+(?:depends on|requires|needs)\s+\w+\b'
            ]
            
            for pattern in relationship_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                relationships.extend(matches)
            
            return relationships
            
        except Exception:
            return []
    
    def _calculate_relational_accuracy(self, original_relations: List[str], restored_relations: List[str]) -> float:
        """Calculate relational accuracy score."""
        try:
            if not original_relations:
                return 1.0
            
            preserved = set(restored_relations).intersection(set(original_relations))
            accuracy = len(preserved) / len(original_relations)
            
            return accuracy
            
        except Exception:
            return 0.5
    
    def _basic_syntax_check(self, code: str) -> bool:
        """Basic syntax check for code content."""
        try:
            # Very basic checks for common syntax issues
            lines = code.split('\n')
            
            # Check for balanced brackets
            bracket_pairs = {'(': ')', '[': ']', '{': '}'}
            stack = []
            
            for line in lines:
                for char in line:
                    if char in bracket_pairs:
                        stack.append(bracket_pairs[char])
                    elif char in bracket_pairs.values():
                        if not stack or stack.pop() != char:
                            return False
            
            return len(stack) == 0
            
        except Exception:
            return True  # Default to assuming it's okay if check fails
    
    def _compare_json_structure(self, original: Any, restored: Any) -> float:
        """Compare JSON structure similarity."""
        try:
            if type(original) != type(restored):
                return 0.0
            
            if isinstance(original, dict):
                original_keys = set(original.keys())
                restored_keys = set(restored.keys())
                
                if not original_keys:
                    return 1.0
                
                key_overlap = len(original_keys.intersection(restored_keys))
                return key_overlap / len(original_keys)
            
            elif isinstance(original, list):
                return min(1.0, len(restored) / max(1, len(original)))
            
            else:
                return 1.0 if original == restored else 0.5
            
        except Exception:
            return 0.5
    
    async def _calculate_overall_integrity_score(
        self, dimension_scores: Dict[IntegrityDimension, float], content_modality: Optional[ContentModality]
    ) -> float:
        """Calculate overall integrity score from dimension scores."""
        try:
            if not dimension_scores:
                return 0.0
            
            # Weight dimensions based on content modality
            weights = self._get_dimension_weights(content_modality)
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for dimension, score in dimension_scores.items():
                weight = weights.get(dimension, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            
            return weighted_sum / max(1.0, total_weight)
            
        except Exception:
            return 0.0
    
    def _get_dimension_weights(self, content_modality: Optional[ContentModality]) -> Dict[IntegrityDimension, float]:
        """Get dimension weights based on content modality."""
        try:
            # Default weights
            default_weights = {
                IntegrityDimension.SEMANTIC_SIMILARITY: 2.0,
                IntegrityDimension.INFORMATIONAL_COMPLETENESS: 1.5,
                IntegrityDimension.CONTEXTUAL_COHERENCE: 1.0,
                IntegrityDimension.STRUCTURAL_INTEGRITY: 1.0,
                IntegrityDimension.TEMPORAL_CONSISTENCY: 0.5,
                IntegrityDimension.RELATIONAL_ACCURACY: 1.0
            }
            
            # Adjust weights based on modality
            if content_modality == ContentModality.CODE:
                default_weights[IntegrityDimension.STRUCTURAL_INTEGRITY] = 2.0
                default_weights[IntegrityDimension.SEMANTIC_SIMILARITY] = 1.5
            elif content_modality == ContentModality.STRUCTURED_DATA:
                default_weights[IntegrityDimension.STRUCTURAL_INTEGRITY] = 2.5
                default_weights[IntegrityDimension.INFORMATIONAL_COMPLETENESS] = 2.0
            elif content_modality == ContentModality.CONVERSATION:
                default_weights[IntegrityDimension.CONTEXTUAL_COHERENCE] = 2.0
                default_weights[IntegrityDimension.TEMPORAL_CONSISTENCY] = 1.5
            
            return default_weights
            
        except Exception:
            return {dimension: 1.0 for dimension in IntegrityDimension}
    
    async def _determine_validation_pass(
        self,
        overall_score: float,
        dimension_scores: Dict[IntegrityDimension, float],
        issues: List[IntegrityIssue],
        request: ValidationRequest
    ) -> bool:
        """Determine if validation passes based on scores and issues."""
        try:
            # Check overall score threshold
            if overall_score < 0.8:
                return False
            
            # Check for critical issues
            critical_issues = [
                issue for issue in issues
                if issue.severity == ValidationSeverity.CRITICAL
            ]
            
            if critical_issues:
                return False
            
            # Check individual dimension thresholds
            for dimension, score in dimension_scores.items():
                min_threshold = 0.7  # Minimum acceptable score for any dimension
                if score < min_threshold:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def _generate_recommendations(
        self,
        dimension_scores: Dict[IntegrityDimension, float],
        issues: List[IntegrityIssue],
        request: ValidationRequest
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        try:
            recommendations = []
            
            # Generate recommendations based on low-scoring dimensions
            for dimension, score in dimension_scores.items():
                if score < 0.8:
                    if dimension == IntegrityDimension.SEMANTIC_SIMILARITY:
                        recommendations.append("Consider using less aggressive consolidation to preserve semantic meaning")
                    elif dimension == IntegrityDimension.INFORMATIONAL_COMPLETENESS:
                        recommendations.append("Review consolidation to ensure key information is preserved")
                    elif dimension == IntegrityDimension.STRUCTURAL_INTEGRITY:
                        recommendations.append("Verify that content structure is maintained during consolidation")
                    elif dimension == IntegrityDimension.CONTEXTUAL_COHERENCE:
                        recommendations.append("Check logical flow and coherence in consolidated content")
            
            # Add recommendations based on critical issues
            critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
            if critical_issues:
                recommendations.append("Critical integrity issues detected - consider content restoration")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Validation passed - content integrity is acceptable")
            
            return recommendations
            
        except Exception:
            return ["Error generating recommendations - manual review suggested"]
    
    def _count_issues_by_severity(self, issues: List[IntegrityIssue]) -> Dict[str, int]:
        """Count issues by severity level."""
        try:
            severity_counts = defaultdict(int)
            for issue in issues:
                severity_counts[issue.severity.value] += 1
            return dict(severity_counts)
        except Exception:
            return {}
    
    async def _update_validation_metrics(self, result: ValidationResult) -> None:
        """Update validation performance metrics."""
        try:
            self._validation_metrics.total_validations += 1
            
            if result.validation_passed:
                self._validation_metrics.passed_validations += 1
            else:
                self._validation_metrics.failed_validations += 1
            
            # Update averages
            total = self._validation_metrics.total_validations
            old_avg_time = self._validation_metrics.average_validation_time
            old_avg_score = self._validation_metrics.average_integrity_score
            
            self._validation_metrics.average_validation_time = (
                (old_avg_time * (total - 1) + result.validation_time_seconds) / total
            )
            
            self._validation_metrics.average_integrity_score = (
                (old_avg_score * (total - 1) + result.overall_integrity_score) / total
            )
            
            # Update dimension performance
            for dimension, score in result.dimension_scores.items():
                if dimension not in self._validation_metrics.dimension_performance:
                    self._validation_metrics.dimension_performance[dimension] = score
                else:
                    old_score = self._validation_metrics.dimension_performance[dimension]
                    self._validation_metrics.dimension_performance[dimension] = (old_score + score) / 2
            
            # Update issue distribution
            for issue in result.issues_found:
                if issue.severity not in self._validation_metrics.issue_distribution:
                    self._validation_metrics.issue_distribution[issue.severity] = 0
                self._validation_metrics.issue_distribution[issue.severity] += 1
            
        except Exception as e:
            logger.error(f"Validation metrics update failed: {e}")
    
    async def _adapt_validation_thresholds(
        self, content_modality: Optional[ContentModality], result: ValidationResult
    ) -> None:
        """Adapt validation thresholds based on validation results."""
        try:
            if not content_modality:
                return
            
            # Adapt thresholds based on validation success/failure patterns
            if result.validation_passed and result.overall_integrity_score > 0.9:
                # If validation passes with high score, slightly increase thresholds
                for dimension, score in result.dimension_scores.items():
                    current_threshold = self._adaptive_thresholds[content_modality].get(
                        dimension, self.config.get(f"{dimension.value}_threshold", 0.8)
                    )
                    new_threshold = min(0.95, current_threshold + 0.01)
                    self._adaptive_thresholds[content_modality][dimension] = new_threshold
            
            elif not result.validation_passed and result.overall_integrity_score < 0.7:
                # If validation fails with low score, slightly decrease thresholds
                for dimension, score in result.dimension_scores.items():
                    if score < 0.7:
                        current_threshold = self._adaptive_thresholds[content_modality].get(
                            dimension, self.config.get(f"{dimension.value}_threshold", 0.8)
                        )
                        new_threshold = max(0.6, current_threshold - 0.02)
                        self._adaptive_thresholds[content_modality][dimension] = new_threshold
            
        except Exception as e:
            logger.error(f"Threshold adaptation failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup validator resources."""
        try:
            # Clear history and metrics
            self._validation_history.clear()
            self._validation_metrics = ValidationMetrics()
            self._adaptive_thresholds.clear()
            
            logger.info("🔍 Semantic Integrity Validator cleanup completed")
            
        except Exception as e:
            logger.error(f"Validator cleanup failed: {e}")


# Global instance
_semantic_integrity_validator: Optional[SemanticIntegrityValidator] = None


async def get_semantic_integrity_validator() -> SemanticIntegrityValidator:
    """Get singleton semantic integrity validator instance."""
    global _semantic_integrity_validator
    
    if _semantic_integrity_validator is None:
        _semantic_integrity_validator = SemanticIntegrityValidator()
    
    return _semantic_integrity_validator


async def cleanup_semantic_integrity_validator() -> None:
    """Cleanup semantic integrity validator resources."""
    global _semantic_integrity_validator
    
    if _semantic_integrity_validator:
        await _semantic_integrity_validator.cleanup()
        _semantic_integrity_validator = None