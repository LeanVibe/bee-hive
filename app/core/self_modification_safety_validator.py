"""
Comprehensive Safety Validator for Self-Modification System

This module provides multi-layer security and stability validation with
comprehensive checks, automated threat detection, and fail-safe mechanisms
to ensure 100% sandbox security and system stability.
"""

import ast
import re
import hashlib
import json
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for safety checks."""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    PARANOID = "paranoid"


class ThreatLevel(Enum):
    """Threat levels for security issues."""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Results of validation checks."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    BLOCKED = "blocked"


@dataclass
class SecurityThreat:
    """Detected security threat."""
    threat_id: str
    threat_type: str
    severity: ThreatLevel
    description: str
    evidence: Dict[str, Any]
    
    # Location information
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    
    # Risk assessment
    exploit_likelihood: float = 0.0  # 0.0 to 1.0
    impact_severity: float = 0.0     # 0.0 to 1.0
    risk_score: float = 0.0          # Combined risk score
    
    # Mitigation
    mitigation_required: bool = True
    mitigation_suggestions: List[str] = field(default_factory=list)
    can_auto_fix: bool = False
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)
    validator_version: str = "1.0.0"


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    validation_id: str
    validation_level: ValidationLevel
    overall_result: ValidationResult
    safety_score: float  # 0.0 to 1.0
    
    # Threat analysis
    threats_detected: List[SecurityThreat]
    threat_summary: Dict[ThreatLevel, int]
    highest_threat_level: ThreatLevel
    
    # Validation results by category
    code_quality_score: float
    security_score: float
    performance_score: float
    maintainability_score: float
    
    # Specific checks
    syntax_validation: ValidationResult
    semantic_validation: ValidationResult
    security_validation: ValidationResult
    performance_validation: ValidationResult
    sandbox_escape_validation: ValidationResult
    
    # Recommendations
    approval_required: bool
    human_review_required: bool
    recommended_actions: List[str]
    blocking_issues: List[str]
    
    # Metadata
    validation_timestamp: datetime
    validation_duration_ms: int
    files_analyzed: int
    lines_analyzed: int
    validation_metadata: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveSafetyValidator:
    """
    Multi-layer security and stability validator with comprehensive checks.
    
    Provides maximum security validation with automated threat detection,
    sandbox escape prevention, and fail-safe mechanisms for all code modifications.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.ENHANCED):
        self.validation_level = validation_level
        
        # Security patterns and rules
        self._security_rules = self._load_security_rules()
        self._banned_imports = self._load_banned_imports()
        self._dangerous_functions = self._load_dangerous_functions()
        self._sandbox_escape_patterns = self._load_sandbox_escape_patterns()
        
        # Performance thresholds
        self._performance_thresholds = {
            'max_execution_time_ms': 5000,
            'max_memory_usage_mb': 100,
            'max_cpu_usage_percent': 80,
            'max_file_size_mb': 10
        }
        
        # Safety thresholds by validation level
        self._safety_thresholds = {
            ValidationLevel.BASIC: 0.5,
            ValidationLevel.STANDARD: 0.7,
            ValidationLevel.ENHANCED: 0.8,
            ValidationLevel.PARANOID: 0.95
        }
        
        logger.info(f"ComprehensiveSafetyValidator initialized at {validation_level.value} level")
    
    def validate_code_modification(self,
                                 original_code: str,
                                 modified_code: str,
                                 file_path: str = "unknown",
                                 modification_context: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """
        Perform comprehensive validation of a code modification.
        
        Args:
            original_code: Original code content
            modified_code: Modified code content  
            file_path: Path of file being modified
            modification_context: Additional context about modification
            
        Returns:
            Comprehensive validation report
            
        Raises:
            ValidationError: If validation process fails
        """
        start_time = datetime.utcnow()
        validation_id = self._generate_validation_id(file_path)
        
        logger.info(f"Starting comprehensive validation {validation_id} for {file_path}")
        
        try:
            # Initialize report
            report = ValidationReport(
                validation_id=validation_id,
                validation_level=self.validation_level,
                overall_result=ValidationResult.PASS,
                safety_score=1.0,
                threats_detected=[],
                threat_summary={level: 0 for level in ThreatLevel},
                highest_threat_level=ThreatLevel.INFORMATIONAL,
                code_quality_score=1.0,
                security_score=1.0,
                performance_score=1.0,
                maintainability_score=1.0,
                syntax_validation=ValidationResult.PASS,
                semantic_validation=ValidationResult.PASS,
                security_validation=ValidationResult.PASS,
                performance_validation=ValidationResult.PASS,
                sandbox_escape_validation=ValidationResult.PASS,
                approval_required=False,
                human_review_required=False,
                recommended_actions=[],
                blocking_issues=[],
                validation_timestamp=start_time,
                validation_duration_ms=0,
                files_analyzed=1,
                lines_analyzed=len(modified_code.split('\n'))
            )
            
            # Run validation layers
            self._validate_syntax(modified_code, report)
            self._validate_semantics(original_code, modified_code, report)
            self._validate_security(modified_code, report)
            self._validate_performance(original_code, modified_code, report)
            self._validate_sandbox_escape_prevention(modified_code, report)
            self._validate_code_quality(modified_code, report)
            
            # Calculate overall scores
            self._calculate_overall_scores(report)
            
            # Determine final result and requirements
            self._determine_final_result(report)
            
            # Calculate validation duration
            end_time = datetime.utcnow()
            report.validation_duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(f"Validation {validation_id} completed: {report.overall_result.value} (safety: {report.safety_score:.2f})")
            return report
            
        except Exception as e:
            logger.error(f"Validation {validation_id} failed: {e}")
            raise ValidationError(f"Validation process failed: {e}")
    
    def validate_sandbox_configuration(self, sandbox_config: Dict[str, Any]) -> ValidationReport:
        """
        Validate sandbox configuration for security.
        
        Args:
            sandbox_config: Sandbox configuration to validate
            
        Returns:
            Validation report for sandbox configuration
        """
        validation_id = self._generate_validation_id("sandbox_config")
        start_time = datetime.utcnow()
        
        logger.info(f"Validating sandbox configuration {validation_id}")
        
        report = ValidationReport(
            validation_id=validation_id,
            validation_level=self.validation_level,
            overall_result=ValidationResult.PASS,
            safety_score=1.0,
            threats_detected=[],
            threat_summary={level: 0 for level in ThreatLevel},
            highest_threat_level=ThreatLevel.INFORMATIONAL,
            code_quality_score=1.0,
            security_score=1.0,
            performance_score=1.0,
            maintainability_score=1.0,
            syntax_validation=ValidationResult.PASS,
            semantic_validation=ValidationResult.PASS,
            security_validation=ValidationResult.PASS,
            performance_validation=ValidationResult.PASS,
            sandbox_escape_validation=ValidationResult.PASS,
            approval_required=False,
            human_review_required=False,
            recommended_actions=[],
            blocking_issues=[],
            validation_timestamp=start_time,
            validation_duration_ms=0,
            files_analyzed=0,
            lines_analyzed=0
        )
        
        # Validate sandbox security
        self._validate_sandbox_security_config(sandbox_config, report)
        self._validate_resource_limits(sandbox_config, report)
        self._validate_network_isolation(sandbox_config, report)
        
        # Calculate final scores
        self._calculate_overall_scores(report)
        self._determine_final_result(report)
        
        # Calculate duration
        end_time = datetime.utcnow()
        report.validation_duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return report
    
    def _validate_syntax(self, code: str, report: ValidationReport) -> None:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            report.syntax_validation = ValidationResult.PASS
            logger.debug("Syntax validation passed")
        except SyntaxError as e:
            threat = SecurityThreat(
                threat_id=self._generate_threat_id("syntax_error"),
                threat_type="syntax_error",
                severity=ThreatLevel.HIGH,
                description=f"Syntax error: {e.msg}",
                evidence={'error': str(e), 'line': e.lineno},
                line_number=e.lineno,
                exploit_likelihood=0.0,
                impact_severity=1.0,
                risk_score=1.0,
                mitigation_required=True,
                mitigation_suggestions=["Fix syntax error before proceeding"],
                can_auto_fix=False
            )
            report.threats_detected.append(threat)
            report.syntax_validation = ValidationResult.FAIL
            report.blocking_issues.append(f"Syntax error on line {e.lineno}: {e.msg}")
            
            logger.warning(f"Syntax validation failed: {e}")
    
    def _validate_semantics(self, original_code: str, modified_code: str, report: ValidationReport) -> None:
        """Validate semantic correctness and changes."""
        try:
            # Parse both versions
            original_ast = ast.parse(original_code) if original_code else None
            modified_ast = ast.parse(modified_code)
            
            # Analyze semantic changes
            semantic_issues = self._analyze_semantic_changes(original_ast, modified_ast)
            
            if semantic_issues:
                for issue in semantic_issues:
                    threat = SecurityThreat(
                        threat_id=self._generate_threat_id("semantic_issue"),
                        threat_type="semantic_change",
                        severity=ThreatLevel.MEDIUM,
                        description=issue['description'],
                        evidence=issue,
                        exploit_likelihood=0.3,
                        impact_severity=0.5,
                        risk_score=0.4,
                        mitigation_required=True,
                        mitigation_suggestions=["Review semantic changes carefully"],
                        can_auto_fix=False
                    )
                    report.threats_detected.append(threat)
            
            # Check for dangerous semantic patterns
            dangerous_patterns = self._detect_dangerous_semantic_patterns(modified_ast)
            for pattern in dangerous_patterns:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("dangerous_pattern"),
                    threat_type="dangerous_semantic_pattern",
                    severity=ThreatLevel.HIGH,
                    description=pattern['description'],
                    evidence=pattern,
                    exploit_likelihood=0.7,
                    impact_severity=0.8,
                    risk_score=0.75,
                    mitigation_required=True,
                    mitigation_suggestions=pattern.get('mitigation', []),
                    can_auto_fix=False
                )
                report.threats_detected.append(threat)
            
            if not semantic_issues and not dangerous_patterns:
                report.semantic_validation = ValidationResult.PASS
            else:
                report.semantic_validation = ValidationResult.WARNING
                
        except Exception as e:
            logger.warning(f"Semantic validation error: {e}")
            report.semantic_validation = ValidationResult.WARNING
    
    def _validate_security(self, code: str, report: ValidationReport) -> None:
        """Comprehensive security validation."""
        try:
            # Parse AST for analysis
            tree = ast.parse(code)
            
            # Check for banned imports
            banned_imports = self._detect_banned_imports(tree)
            for banned_import in banned_imports:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("banned_import"),
                    threat_type="banned_import",
                    severity=ThreatLevel.CRITICAL,
                    description=f"Banned import detected: {banned_import['module']}",
                    evidence=banned_import,
                    line_number=banned_import.get('line_number'),
                    exploit_likelihood=1.0,
                    impact_severity=1.0,
                    risk_score=1.0,
                    mitigation_required=True,
                    mitigation_suggestions=["Remove or replace banned import"],
                    can_auto_fix=False
                )
                report.threats_detected.append(threat)
                report.blocking_issues.append(f"Banned import: {banned_import['module']}")
            
            # Check for dangerous function calls
            dangerous_calls = self._detect_dangerous_function_calls(tree)
            for call in dangerous_calls:
                severity = ThreatLevel.CRITICAL if call['function'] in ['eval', 'exec'] else ThreatLevel.HIGH
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("dangerous_call"),
                    threat_type="dangerous_function_call",
                    severity=severity,
                    description=f"Dangerous function call: {call['function']}",
                    evidence=call,
                    line_number=call.get('line_number'),
                    exploit_likelihood=0.9,
                    impact_severity=1.0,
                    risk_score=0.95,
                    mitigation_required=True,
                    mitigation_suggestions=[f"Replace {call['function']} with safer alternative"],
                    can_auto_fix=False
                )
                report.threats_detected.append(threat)
            
            # Check for security patterns in code text
            security_violations = self._scan_security_patterns(code)
            for violation in security_violations:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("security_pattern"),
                    threat_type="security_violation",
                    severity=violation['severity'],
                    description=violation['description'],
                    evidence=violation,
                    line_number=violation.get('line_number'),
                    exploit_likelihood=violation.get('likelihood', 0.5),
                    impact_severity=violation.get('impact', 0.7),
                    risk_score=violation.get('risk_score', 0.6),
                    mitigation_required=True,
                    mitigation_suggestions=violation.get('suggestions', []),
                    can_auto_fix=False
                )
                report.threats_detected.append(threat)
            
            # Determine security validation result
            critical_threats = [t for t in report.threats_detected if t.severity == ThreatLevel.CRITICAL]
            high_threats = [t for t in report.threats_detected if t.severity == ThreatLevel.HIGH]
            
            if critical_threats:
                report.security_validation = ValidationResult.BLOCKED
                report.blocking_issues.extend([t.description for t in critical_threats])
            elif high_threats:
                report.security_validation = ValidationResult.FAIL
            else:
                report.security_validation = ValidationResult.PASS
                
        except Exception as e:
            logger.warning(f"Security validation error: {e}")
            report.security_validation = ValidationResult.WARNING
    
    def _validate_performance(self, original_code: str, modified_code: str, report: ValidationReport) -> None:
        """Validate performance implications."""
        try:
            # Analyze complexity changes
            original_complexity = self._calculate_code_complexity(original_code) if original_code else 0
            modified_complexity = self._calculate_code_complexity(modified_code)
            
            complexity_increase = modified_complexity - original_complexity
            
            if complexity_increase > 50:  # Significant complexity increase
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("complexity_increase"),
                    threat_type="performance_degradation",
                    severity=ThreatLevel.MEDIUM,
                    description=f"Significant complexity increase: {complexity_increase}",
                    evidence={'original': original_complexity, 'modified': modified_complexity},
                    exploit_likelihood=0.0,
                    impact_severity=0.5,
                    risk_score=0.25,
                    mitigation_required=False,
                    mitigation_suggestions=["Consider refactoring to reduce complexity"],
                    can_auto_fix=False
                )
                report.threats_detected.append(threat)
            
            # Check for performance anti-patterns
            performance_issues = self._detect_performance_antipatterns(modified_code)
            for issue in performance_issues:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("performance_issue"),
                    threat_type="performance_antipattern",
                    severity=ThreatLevel.LOW,
                    description=issue['description'],
                    evidence=issue,
                    line_number=issue.get('line_number'),
                    exploit_likelihood=0.0,
                    impact_severity=0.3,
                    risk_score=0.15,
                    mitigation_required=False,
                    mitigation_suggestions=issue.get('suggestions', []),
                    can_auto_fix=issue.get('can_auto_fix', False)
                )
                report.threats_detected.append(threat)
            
            report.performance_validation = ValidationResult.PASS
            
        except Exception as e:
            logger.warning(f"Performance validation error: {e}")
            report.performance_validation = ValidationResult.WARNING
    
    def _validate_sandbox_escape_prevention(self, code: str, report: ValidationReport) -> None:
        """Validate against sandbox escape attempts - CRITICAL for security."""
        try:
            # Check for sandbox escape patterns
            escape_attempts = []
            
            for pattern_name, pattern_info in self._sandbox_escape_patterns.items():
                pattern = pattern_info['pattern']
                matches = re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    line_number = code[:match.start()].count('\n') + 1
                    escape_attempts.append({
                        'pattern_name': pattern_name,
                        'severity': pattern_info['severity'],
                        'description': pattern_info['description'],
                        'line_number': line_number,
                        'code_snippet': match.group(),
                        'mitigation': pattern_info.get('mitigation', [])
                    })
            
            # Create threats for each escape attempt
            for attempt in escape_attempts:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("sandbox_escape"),
                    threat_type="sandbox_escape_attempt",
                    severity=ThreatLevel.CRITICAL,
                    description=f"Sandbox escape attempt: {attempt['description']}",
                    evidence=attempt,
                    line_number=attempt['line_number'],
                    code_snippet=attempt['code_snippet'],
                    exploit_likelihood=1.0,
                    impact_severity=1.0,
                    risk_score=1.0,
                    mitigation_required=True,
                    mitigation_suggestions=attempt['mitigation'],
                    can_auto_fix=False
                )
                report.threats_detected.append(threat)
                report.blocking_issues.append(f"CRITICAL: Sandbox escape attempt - {attempt['description']}")
            
            # AST-based sandbox escape detection
            tree = ast.parse(code)
            ast_escape_attempts = self._detect_ast_sandbox_escapes(tree)
            
            for attempt in ast_escape_attempts:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("ast_sandbox_escape"),
                    threat_type="ast_sandbox_escape",
                    severity=ThreatLevel.CRITICAL,
                    description=f"AST-level sandbox escape: {attempt['description']}",
                    evidence=attempt,
                    exploit_likelihood=1.0,
                    impact_severity=1.0,
                    risk_score=1.0,
                    mitigation_required=True,
                    mitigation_suggestions=["Remove sandbox escape attempt"],
                    can_auto_fix=False
                )
                report.threats_detected.append(threat)
                report.blocking_issues.append(f"CRITICAL: AST sandbox escape - {attempt['description']}")
            
            if escape_attempts or ast_escape_attempts:
                report.sandbox_escape_validation = ValidationResult.BLOCKED
            else:
                report.sandbox_escape_validation = ValidationResult.PASS
                
        except Exception as e:
            logger.error(f"Sandbox escape validation error: {e}")
            report.sandbox_escape_validation = ValidationResult.FAIL
            report.blocking_issues.append("Sandbox escape validation failed")
    
    def _validate_code_quality(self, code: str, report: ValidationReport) -> None:
        """Validate code quality metrics."""
        try:
            lines = code.split('\n')
            
            # Calculate basic quality metrics
            total_lines = len(lines)
            blank_lines = len([line for line in lines if not line.strip()])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            code_lines = total_lines - blank_lines - comment_lines
            
            # Comment ratio
            comment_ratio = comment_lines / max(code_lines, 1)
            
            # Line length issues
            long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
            
            # Quality scoring
            quality_score = 1.0
            
            if comment_ratio < 0.1:  # Less than 10% comments
                quality_score -= 0.2
                report.recommended_actions.append("Consider adding more comments for maintainability")
            
            if len(long_lines) > total_lines * 0.1:  # More than 10% long lines
                quality_score -= 0.1
                report.recommended_actions.append("Consider breaking long lines for readability")
            
            report.code_quality_score = max(0.0, quality_score)
            
        except Exception as e:
            logger.warning(f"Code quality validation error: {e}")
            report.code_quality_score = 0.5
    
    def _validate_sandbox_security_config(self, config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate sandbox security configuration."""
        # Check network isolation
        if config.get('network_access', True):
            threat = SecurityThreat(
                threat_id=self._generate_threat_id("network_access"),
                threat_type="sandbox_config_violation",
                severity=ThreatLevel.CRITICAL,
                description="Network access enabled in sandbox",
                evidence={'network_access': config.get('network_access')},
                exploit_likelihood=1.0,
                impact_severity=1.0,
                risk_score=1.0,
                mitigation_required=True,
                mitigation_suggestions=["Disable network access in sandbox"],
                can_auto_fix=True
            )
            report.threats_detected.append(threat)
            report.blocking_issues.append("CRITICAL: Network access must be disabled")
        
        # Check resource limits
        memory_limit = config.get('memory_limit_mb', 0)
        if memory_limit > 512 or memory_limit == 0:
            threat = SecurityThreat(
                threat_id=self._generate_threat_id("memory_limit"),
                threat_type="resource_limit_violation",
                severity=ThreatLevel.HIGH,
                description=f"Memory limit too high or unlimited: {memory_limit}MB",
                evidence={'memory_limit_mb': memory_limit},
                exploit_likelihood=0.7,
                impact_severity=0.8,
                risk_score=0.75,
                mitigation_required=True,
                mitigation_suggestions=["Set memory limit to 256MB or less"],
                can_auto_fix=True
            )
            report.threats_detected.append(threat)
    
    def _validate_resource_limits(self, config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate resource limit configuration."""
        for resource, threshold in self._performance_thresholds.items():
            config_value = config.get(resource, 0)
            if config_value > threshold or config_value == 0:
                severity = ThreatLevel.HIGH if resource in ['max_memory_usage_mb', 'max_execution_time_ms'] else ThreatLevel.MEDIUM
                
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id("resource_limit"),
                    threat_type="resource_limit_violation",
                    severity=severity,
                    description=f"{resource} exceeds safe threshold: {config_value} > {threshold}",
                    evidence={resource: config_value, 'threshold': threshold},
                    exploit_likelihood=0.5,
                    impact_severity=0.6,
                    risk_score=0.55,
                    mitigation_required=True,
                    mitigation_suggestions=[f"Set {resource} to {threshold} or less"],
                    can_auto_fix=True
                )
                report.threats_detected.append(threat)
    
    def _validate_network_isolation(self, config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate network isolation configuration."""
        network_settings = config.get('network', {})
        
        # Check for network access
        if network_settings.get('enabled', False):
            threat = SecurityThreat(
                threat_id=self._generate_threat_id("network_enabled"),
                threat_type="network_isolation_violation",
                severity=ThreatLevel.CRITICAL,
                description="Network access enabled - violates isolation requirements",
                evidence=network_settings,
                exploit_likelihood=1.0,
                impact_severity=1.0,
                risk_score=1.0,
                mitigation_required=True,
                mitigation_suggestions=["Disable all network access"],
                can_auto_fix=True
            )
            report.threats_detected.append(threat)
            report.blocking_issues.append("CRITICAL: Network isolation must be enforced")
    
    def _calculate_overall_scores(self, report: ValidationReport) -> None:
        """Calculate overall validation scores."""
        # Count threats by severity
        for threat in report.threats_detected:
            report.threat_summary[threat.severity] += 1
            
            # Update highest threat level
            if threat.severity.value > report.highest_threat_level.value:
                report.highest_threat_level = threat.severity
        
        # Calculate security score
        critical_count = report.threat_summary[ThreatLevel.CRITICAL]
        high_count = report.threat_summary[ThreatLevel.HIGH]
        medium_count = report.threat_summary[ThreatLevel.MEDIUM]
        
        security_penalty = (critical_count * 1.0 + high_count * 0.5 + medium_count * 0.2)
        report.security_score = max(0.0, 1.0 - security_penalty)
        
        # Calculate overall safety score
        scores = [
            report.security_score * 0.4,      # Security is most important
            report.code_quality_score * 0.2,
            report.performance_score * 0.2,
            report.maintainability_score * 0.2
        ]
        report.safety_score = sum(scores)
        
        # Ensure sandbox escape failures result in 0 safety score
        if report.sandbox_escape_validation == ValidationResult.BLOCKED:
            report.safety_score = 0.0
            report.security_score = 0.0
    
    def _determine_final_result(self, report: ValidationReport) -> None:
        """Determine final validation result and approval requirements."""
        # Check for blocking issues
        if report.blocking_issues:
            report.overall_result = ValidationResult.BLOCKED
            report.approval_required = True
            report.human_review_required = True
            return
        
        # Check safety score against threshold
        threshold = self._safety_thresholds[self.validation_level]
        
        if report.safety_score < threshold:
            if report.safety_score < threshold * 0.5:
                report.overall_result = ValidationResult.FAIL
            else:
                report.overall_result = ValidationResult.WARNING
            
            report.approval_required = True
            report.human_review_required = True
        
        # Check for critical or high threats
        if report.highest_threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            report.approval_required = True
            report.human_review_required = True
            
            if report.highest_threat_level == ThreatLevel.CRITICAL:
                report.overall_result = ValidationResult.BLOCKED
        
        # Generate recommendations
        if report.approval_required:
            report.recommended_actions.append("Human review required before applying modification")
        
        if report.safety_score < 0.8:
            report.recommended_actions.append("Consider additional safety improvements")
        
        if report.threat_summary[ThreatLevel.HIGH] > 0:
            report.recommended_actions.append("Address high-severity threats")
    
    # Helper methods for specific validations
    
    def _analyze_semantic_changes(self, original_ast: Optional[ast.AST], modified_ast: ast.AST) -> List[Dict[str, Any]]:
        """Analyze semantic changes between ASTs."""
        issues = []
        
        if original_ast is None:
            return issues  # No comparison possible
        
        # Compare function definitions
        original_functions = self._extract_functions(original_ast)
        modified_functions = self._extract_functions(modified_ast)
        
        # Check for removed functions
        removed_functions = set(original_functions.keys()) - set(modified_functions.keys())
        for func_name in removed_functions:
            issues.append({
                'type': 'function_removed',
                'description': f"Function '{func_name}' was removed",
                'severity': 'medium'
            })
        
        # Check for added functions
        added_functions = set(modified_functions.keys()) - set(original_functions.keys())
        for func_name in added_functions:
            issues.append({
                'type': 'function_added',
                'description': f"Function '{func_name}' was added",
                'severity': 'low'
            })
        
        return issues
    
    def _detect_dangerous_semantic_patterns(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect dangerous semantic patterns in AST."""
        patterns = []
        
        for node in ast.walk(ast_tree):
            # Check for dynamic code execution
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'compile']:
                    patterns.append({
                        'type': 'dynamic_execution',
                        'description': f"Dynamic code execution using {node.func.id}",
                        'line_number': getattr(node, 'lineno', None),
                        'mitigation': ['Replace with safer alternative', 'Use ast.literal_eval for safe evaluation']
                    })
            
            # Check for attribute access on dangerous objects
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id in ['__builtins__', 'globals', 'locals']:
                    patterns.append({
                        'type': 'dangerous_attribute_access',
                        'description': f"Access to dangerous attribute: {node.value.id}.{node.attr}",
                        'line_number': getattr(node, 'lineno', None),
                        'mitigation': ['Remove access to dangerous attributes']
                    })
        
        return patterns
    
    def _detect_banned_imports(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect banned import statements."""
        banned_imports = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self._banned_imports:
                        banned_imports.append({
                            'module': alias.name,
                            'line_number': getattr(node, 'lineno', None),
                            'type': 'import'
                        })
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in self._banned_imports:
                    banned_imports.append({
                        'module': node.module,
                        'line_number': getattr(node, 'lineno', None),
                        'type': 'from_import'
                    })
        
        return banned_imports
    
    def _detect_dangerous_function_calls(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect calls to dangerous functions."""
        dangerous_calls = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self._dangerous_functions:
                    dangerous_calls.append({
                        'function': node.func.id,
                        'line_number': getattr(node, 'lineno', None),
                        'severity': self._dangerous_functions[node.func.id]['severity'],
                        'description': self._dangerous_functions[node.func.id]['description']
                    })
        
        return dangerous_calls
    
    def _scan_security_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Scan code for security violation patterns."""
        violations = []
        
        for rule_name, rule in self._security_rules.items():
            pattern = rule['pattern']
            matches = re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                violations.append({
                    'rule': rule_name,
                    'severity': rule['severity'],
                    'description': rule['description'],
                    'line_number': line_number,
                    'code_snippet': match.group(),
                    'likelihood': rule.get('likelihood', 0.5),
                    'impact': rule.get('impact', 0.5),
                    'risk_score': rule.get('risk_score', 0.5),
                    'suggestions': rule.get('mitigation', [])
                })
        
        return violations
    
    def _detect_ast_sandbox_escapes(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect sandbox escape attempts at AST level."""
        escape_attempts = []
        
        for node in ast.walk(ast_tree):
            # Check for __import__ usage
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == '__import__':
                    escape_attempts.append({
                        'type': 'import_builtin',
                        'description': 'Use of __import__ builtin for dynamic imports',
                        'line_number': getattr(node, 'lineno', None)
                    })
            
            # Check for access to frame objects
            elif isinstance(node, ast.Attribute):
                if node.attr in ['f_globals', 'f_locals', 'f_back']:
                    escape_attempts.append({
                        'type': 'frame_access',
                        'description': f'Access to frame attribute: {node.attr}',
                        'line_number': getattr(node, 'lineno', None)
                    })
            
            # Check for dangerous builtin access
            elif isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == '__builtins__':
                    escape_attempts.append({
                        'type': 'builtins_access',
                        'description': 'Direct access to __builtins__',
                        'line_number': getattr(node, 'lineno', None)
                    })
        
        return escape_attempts
    
    def _calculate_code_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity of code."""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, (ast.BoolOp, ast.Compare)):
                    complexity += len(node.values) if hasattr(node, 'values') else 1
            
            return complexity
        except:
            return 0
    
    def _detect_performance_antipatterns(self, code: str) -> List[Dict[str, Any]]:
        """Detect performance anti-patterns."""
        antipatterns = []
        
        # String concatenation in loop
        if re.search(r'for\s+\w+.*:\s*\w+\s*\+=.*str', code, re.MULTILINE):
            antipatterns.append({
                'type': 'string_concat_loop',
                'description': 'String concatenation in loop - use join() instead',
                'suggestions': ['Use str.join() for better performance'],
                'can_auto_fix': True
            })
        
        # Inefficient list operations
        if re.search(r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', code):
            antipatterns.append({
                'type': 'range_len_loop',
                'description': 'Using range(len()) instead of enumerate()',
                'suggestions': ['Use enumerate() for cleaner and potentially faster code'],
                'can_auto_fix': True
            })
        
        return antipatterns
    
    def _extract_functions(self, ast_tree: ast.AST) -> Dict[str, ast.FunctionDef]:
        """Extract function definitions from AST."""
        functions = {}
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
        return functions
    
    def _load_security_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load security rules for pattern matching."""
        return {
            'hardcoded_secret': {
                'pattern': r'(?i)(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                'severity': ThreatLevel.HIGH,
                'description': 'Hardcoded secret detected',
                'likelihood': 0.8,
                'impact': 0.9,
                'risk_score': 0.85,
                'mitigation': ['Use environment variables or secure secret management']
            },
            'sql_injection': {
                'pattern': r'execute\s*\(\s*["\'].*%s.*["\']',
                'severity': ThreatLevel.HIGH,
                'description': 'Potential SQL injection vulnerability',
                'likelihood': 0.7,
                'impact': 0.9,
                'risk_score': 0.8,
                'mitigation': ['Use parameterized queries']
            },
            'command_injection': {
                'pattern': r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
                'severity': ThreatLevel.CRITICAL,
                'description': 'Command injection vulnerability',
                'likelihood': 0.9,
                'impact': 1.0,
                'risk_score': 0.95,
                'mitigation': ['Use shell=False and pass arguments as list']
            }
        }
    
    def _load_banned_imports(self) -> Set[str]:
        """Load list of banned import modules."""
        return {
            'subprocess', 'os', 'sys', 'socket', 'urllib', 'requests',
            'http', 'ftplib', 'telnetlib', 'smtplib', 'poplib', 'imaplib',
            'ctypes', 'marshal', 'pickle', 'dill', 'shelve'
        }
    
    def _load_dangerous_functions(self) -> Dict[str, Dict[str, Any]]:
        """Load list of dangerous function calls."""
        return {
            'eval': {
                'severity': ThreatLevel.CRITICAL,
                'description': 'eval() allows arbitrary code execution'
            },
            'exec': {
                'severity': ThreatLevel.CRITICAL,
                'description': 'exec() allows arbitrary code execution'
            },
            'compile': {
                'severity': ThreatLevel.HIGH,
                'description': 'compile() can be used to execute dynamic code'
            },
            '__import__': {
                'severity': ThreatLevel.HIGH,
                'description': '__import__() allows dynamic module imports'
            }
        }
    
    def _load_sandbox_escape_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns that indicate sandbox escape attempts."""
        return {
            'frame_inspection': {
                'pattern': r'\b(f_globals|f_locals|f_back)\b',
                'severity': ThreatLevel.CRITICAL,
                'description': 'Frame object inspection - potential sandbox escape',
                'mitigation': ['Remove frame inspection code']
            },
            'builtins_access': {
                'pattern': r'__builtins__\[',
                'severity': ThreatLevel.CRITICAL,
                'description': 'Direct access to builtins - sandbox escape attempt',
                'mitigation': ['Remove builtins access']
            },
            'import_override': {
                'pattern': r'__import__\s*=',
                'severity': ThreatLevel.CRITICAL,
                'description': 'Attempt to override import mechanism',
                'mitigation': ['Remove import mechanism override']
            },
            'class_attribute_access': {
                'pattern': r'\.__class__\.__bases__',
                'severity': ThreatLevel.HIGH,
                'description': 'Class hierarchy inspection - potential escape vector',
                'mitigation': ['Remove class hierarchy inspection']
            }
        }
    
    def _generate_validation_id(self, context: str) -> str:
        """Generate unique validation ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        return f"VAL_{timestamp}_{context_hash}"
    
    def _generate_threat_id(self, threat_type: str) -> str:
        """Generate unique threat ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        type_hash = hashlib.md5(threat_type.encode()).hexdigest()[:6]
        return f"THR_{type_hash}_{timestamp}"


class ValidationError(Exception):
    """Validation process error."""
    pass


# Export main classes
__all__ = [
    'ComprehensiveSafetyValidator',
    'ValidationReport',
    'SecurityThreat',
    'ValidationLevel',
    'ThreatLevel',
    'ValidationResult',
    'ValidationError'
]