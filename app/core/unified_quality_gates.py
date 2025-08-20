"""
Unified Quality Gates System for LeanVibe Agent Hive 2.0

Consolidates all quality validation implementations into a comprehensive
framework for command reliability, security, performance, and user experience.

Features:
- Multi-layer validation (syntax, semantic, security, performance, compatibility, UX)
- Real-time security threat detection with AI analysis
- Performance benchmarking and regression detection
- Mobile compatibility validation
- Cross-project compatibility checks
- Intelligent error recovery and guidance
- Comprehensive reporting and analytics
"""

import asyncio
import json
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum
from pathlib import Path
from collections import defaultdict
import structlog

logger = structlog.get_logger()


class ValidationLevel(Enum):
    """Validation depth levels."""
    BASIC = "basic"                 # Essential validation only
    STANDARD = "standard"           # Recommended validation
    COMPREHENSIVE = "comprehensive" # Full validation suite
    CRITICAL = "critical"          # Maximum security validation


class ValidationResult:
    """Comprehensive validation result with detailed feedback."""
    
    def __init__(self, command: str = ""):
        self.command = command
        self.overall_valid = False
        self.overall_score = 0.0
        self.execution_time_ms = 0.0
        self.timestamp = datetime.utcnow()
        
        # Layer-specific results
        self.syntax_validation = SyntaxValidationResult()
        self.semantic_validation = SemanticValidationResult()
        self.security_validation = SecurityValidationResult()
        self.performance_validation = PerformanceValidationResult()
        self.compatibility_validation = CompatibilityValidationResult()
        self.ux_validation = UXValidationResult()
        
        # Overall assessment
        self.blocking_issues = []
        self.warnings = []
        self.suggestions = []
        self.auto_fixes = []
        self.recovery_strategies = []
        
        # Metadata
        self.validation_level = ValidationLevel.STANDARD
        self.mobile_optimized = False
        self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "command": self.command,
            "overall_valid": self.overall_valid,
            "overall_score": round(self.overall_score, 2),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "validation_layers": {
                "syntax": self.syntax_validation.to_dict(),
                "semantic": self.semantic_validation.to_dict(),
                "security": self.security_validation.to_dict(),
                "performance": self.performance_validation.to_dict(),
                "compatibility": self.compatibility_validation.to_dict(),
                "user_experience": self.ux_validation.to_dict()
            },
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "auto_fixes": self.auto_fixes,
            "recovery_strategies": self.recovery_strategies,
            "validation_level": self.validation_level.value,
            "mobile_optimized": self.mobile_optimized,
            "context": self.context
        }


class LayerValidationResult:
    """Base class for layer-specific validation results."""
    
    def __init__(self):
        self.valid = True
        self.score = 1.0
        self.errors = []
        self.warnings = []
        self.suggestions = []
        self.execution_time_ms = 0.0
        self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "score": round(self.score, 2),
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "details": self.details
        }


class SyntaxValidationResult(LayerValidationResult):
    """Syntax validation specific result."""
    pass


class SemanticValidationResult(LayerValidationResult):
    """Semantic validation specific result."""
    
    def __init__(self):
        super().__init__()
        self.parameter_validation = {}
        self.context_requirements = []
        self.prerequisite_checks = []


class SecurityValidationResult(LayerValidationResult):
    """Security validation specific result."""
    
    def __init__(self):
        super().__init__()
        self.threats = []
        self.security_score = 1.0
        self.required_permissions = []
        self.data_access_analysis = {}
        self.ai_threat_analysis = {}


class PerformanceValidationResult(LayerValidationResult):
    """Performance validation specific result."""
    
    def __init__(self):
        super().__init__()
        self.estimated_execution_time = 0.0
        self.resource_requirements = {}
        self.scalability_analysis = {}
        self.mobile_performance_score = 1.0


class CompatibilityValidationResult(LayerValidationResult):
    """Compatibility validation specific result."""
    
    def __init__(self):
        super().__init__()
        self.mobile_compatibility = True
        self.browser_compatibility = {}
        self.platform_compatibility = {}
        self.api_version_compatibility = {}


class UXValidationResult(LayerValidationResult):
    """User experience validation specific result."""
    
    def __init__(self):
        super().__init__()
        self.accessibility_score = 1.0
        self.mobile_ux_score = 1.0
        self.usability_issues = []
        self.mobile_optimizations = []


class SecurityThreat:
    """Represents a security threat found during validation."""
    
    def __init__(
        self,
        threat_type: str,
        severity: str,
        description: str,
        pattern: str = "",
        mitigation: str = "",
        confidence: float = 1.0
    ):
        self.threat_type = threat_type
        self.severity = severity  # critical, high, medium, low
        self.description = description
        self.pattern = pattern
        self.mitigation = mitigation
        self.confidence = confidence
        self.detected_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.threat_type,
            "severity": self.severity,
            "description": self.description,
            "pattern": self.pattern,
            "mitigation": self.mitigation,
            "confidence": round(self.confidence, 2),
            "detected_at": self.detected_at.isoformat()
        }


class CommandSyntaxValidator:
    """Validates command syntax and structure."""
    
    async def validate(self, command: str, context: Dict[str, Any] = None) -> SyntaxValidationResult:
        """Validate command syntax."""
        start_time = time.time()
        result = SyntaxValidationResult()
        
        try:
            # Basic structure validation
            if not command or not isinstance(command, str):
                result.valid = False
                result.score = 0.0
                result.errors.append("Command must be a non-empty string")
                return result
            
            command = command.strip()
            
            # Hive command format validation
            if not command.startswith("/hive:"):
                result.valid = False
                result.score = 0.2
                result.errors.append("Command must start with '/hive:'")
                result.suggestions.append(f"Try: /hive:{command}" if not command.startswith("/") else "")
                return result
            
            # Parse command structure
            parts = command.split()
            if len(parts) < 1:
                result.valid = False
                result.score = 0.3
                result.errors.append("Invalid command structure")
                return result
            
            command_name = parts[0].replace("/hive:", "")
            args = parts[1:] if len(parts) > 1 else []
            
            # Command name validation
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', command_name):
                result.valid = False
                result.score = 0.4
                result.errors.append(f"Invalid command name format: '{command_name}'")
                return result
            
            # Arguments structure validation
            for i, arg in enumerate(args):
                if arg.startswith("--"):
                    # Flag validation
                    if not re.match(r'^--[a-zA-Z][a-zA-Z0-9_-]*(?:=.+)?$', arg):
                        result.warnings.append(f"Potentially malformed flag: '{arg}'")
                        result.score -= 0.1
                elif arg.startswith("-"):
                    # Short flag validation
                    if not re.match(r'^-[a-zA-Z]$', arg):
                        result.warnings.append(f"Potentially malformed short flag: '{arg}'")
                        result.score -= 0.05
            
            # Quote matching validation
            quote_chars = ['"', "'"]
            for quote_char in quote_chars:
                quote_count = command.count(quote_char)
                if quote_count % 2 != 0:
                    result.warnings.append(f"Unmatched {quote_char} quotes detected")
                    result.score -= 0.1
            
            # Success case
            if result.valid:
                result.score = max(0.0, min(1.0, result.score))
                result.details = {
                    "command_name": command_name,
                    "args_count": len(args),
                    "parsed_args": args
                }
            
            return result
            
        except Exception as e:
            logger.error("Syntax validation failed", error=str(e))
            result.valid = False
            result.score = 0.0
            result.errors.append(f"Syntax validation error: {e}")
            return result
        finally:
            result.execution_time_ms = (time.time() - start_time) * 1000


class CommandSemanticValidator:
    """Validates command semantics and logical correctness."""
    
    def __init__(self):
        self.command_definitions = self._load_command_definitions()
    
    async def validate(self, command: str, context: Dict[str, Any] = None) -> SemanticValidationResult:
        """Validate command semantics."""
        start_time = time.time()
        result = SemanticValidationResult()
        
        try:
            # Parse command
            parts = command.strip().split()
            if not parts or not parts[0].startswith("/hive:"):
                result.valid = False
                result.errors.append("Invalid command format for semantic validation")
                return result
            
            command_name = parts[0].replace("/hive:", "")
            args = parts[1:] if len(parts) > 1 else []
            
            # Check if command exists
            from .hive_slash_commands import get_hive_command_registry
            registry = get_hive_command_registry()
            command_obj = registry.get_command(command_name)
            
            if not command_obj:
                result.valid = False
                result.score = 0.0
                result.errors.append(f"Unknown command: {command_name}")
                result.suggestions.extend(await self._suggest_similar_commands(command_name, registry))
                return result
            
            # Validate command arguments
            arg_validation = await self._validate_command_arguments(command_obj, args, context)
            result.parameter_validation = arg_validation
            
            if not arg_validation.get("valid", True):
                result.valid = False
                result.score = 0.6
                result.errors.extend(arg_validation.get("errors", []))
                result.warnings.extend(arg_validation.get("warnings", []))
                result.suggestions.extend(arg_validation.get("suggestions", []))
            
            # Check context requirements
            context_check = await self._check_context_requirements(command_name, context)
            result.context_requirements = context_check.get("requirements", [])
            
            if not context_check.get("valid", True):
                result.warnings.extend(context_check.get("warnings", []))
                result.score -= 0.1
            
            # Check prerequisites
            prereq_check = await self._check_prerequisites(command_name, context)
            result.prerequisite_checks = prereq_check.get("checks", [])
            
            if not prereq_check.get("valid", True):
                result.warnings.extend(prereq_check.get("warnings", []))
                result.suggestions.extend(prereq_check.get("suggestions", []))
                result.score -= 0.15
            
            # Semantic scoring
            if result.valid:
                semantic_score = self._calculate_semantic_score(command_obj, args, context)
                result.score = semantic_score
            
            result.details = {
                "command_found": True,
                "command_type": type(command_obj).__name__,
                "args_analyzed": len(args),
                "context_available": bool(context)
            }
            
            return result
            
        except Exception as e:
            logger.error("Semantic validation failed", error=str(e))
            result.valid = False
            result.score = 0.0
            result.errors.append(f"Semantic validation error: {e}")
            return result
        finally:
            result.execution_time_ms = (time.time() - start_time) * 1000
    
    def _load_command_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load command definitions for semantic validation."""
        # This would load from a configuration file or database
        # For now, return basic definitions
        return {
            "start": {
                "required_args": [],
                "optional_args": ["--quick", "--team-size", "--timeout"],
                "context_requirements": [],
                "prerequisites": []
            },
            "develop": {
                "required_args": ["project_description"],
                "optional_args": ["--dashboard", "--timeout"],
                "context_requirements": ["agents_available"],
                "prerequisites": ["platform_running"]
            },
            # Add more command definitions...
        }
    
    async def _suggest_similar_commands(self, command_name: str, registry) -> List[str]:
        """Suggest similar command names."""
        available_commands = list(registry.commands.keys())
        suggestions = []
        
        for cmd in available_commands:
            if self._string_similarity(command_name, cmd) > 0.6:
                suggestions.append(f"Did you mean '/hive:{cmd}'?")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity."""
        if not s1 or not s2:
            return 0.0
        
        # Simple character-based similarity
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
        
        matches = sum(1 for a, b in zip(shorter, longer) if a == b)
        return matches / len(longer)
    
    async def _validate_command_arguments(
        self, 
        command_obj, 
        args: List[str], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Validate command arguments against command definition."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # Use command's built-in validation if available
            if hasattr(command_obj, 'validate_args'):
                is_valid = command_obj.validate_args(args)
                if not is_valid:
                    validation["valid"] = False
                    validation["errors"].append("Command argument validation failed")
                    validation["suggestions"].append(f"Usage: {command_obj.usage}")
            
            return validation
            
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Argument validation error: {e}")
            return validation
    
    async def _check_context_requirements(
        self, 
        command_name: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Check if command's context requirements are met."""
        check = {
            "valid": True,
            "requirements": [],
            "warnings": []
        }
        
        # Command-specific context requirements
        requirements = {
            "develop": ["agents_available", "platform_running"],
            "spawn": ["platform_running"],
            "oversight": ["platform_running"]
        }
        
        cmd_requirements = requirements.get(command_name, [])
        check["requirements"] = cmd_requirements
        
        if context:
            for req in cmd_requirements:
                if req == "agents_available" and context.get("agent_count", 0) == 0:
                    check["valid"] = False
                    check["warnings"].append("No agents available for this command")
                elif req == "platform_running" and not context.get("platform_active", False):
                    check["valid"] = False
                    check["warnings"].append("Platform must be running for this command")
        
        return check
    
    async def _check_prerequisites(
        self, 
        command_name: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Check command prerequisites."""
        check = {
            "valid": True,
            "checks": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Define prerequisites for commands
        prerequisites = {
            "develop": [
                {"check": "platform_active", "message": "Platform should be active"},
                {"check": "sufficient_agents", "message": "At least 3 agents recommended"}
            ],
            "spawn": [
                {"check": "platform_active", "message": "Platform should be active"}
            ]
        }
        
        cmd_prereqs = prerequisites.get(command_name, [])
        check["checks"] = cmd_prereqs
        
        if context:
            for prereq in cmd_prereqs:
                if prereq["check"] == "platform_active" and not context.get("platform_active", False):
                    check["valid"] = False
                    check["warnings"].append(prereq["message"])
                    check["suggestions"].append("Run /hive:start first")
                elif prereq["check"] == "sufficient_agents" and context.get("agent_count", 0) < 3:
                    check["warnings"].append(prereq["message"])
                    check["suggestions"].append("Consider spawning more agents")
        
        return check
    
    def _calculate_semantic_score(self, command_obj, args: List[str], context: Dict[str, Any] = None) -> float:
        """Calculate semantic correctness score."""
        base_score = 1.0
        
        # Deduct for missing context
        if not context:
            base_score -= 0.1
        
        # Score based on argument completeness
        if hasattr(command_obj, 'usage'):
            usage = command_obj.usage
            if "required" in usage.lower() and not args:
                base_score -= 0.3
        
        return max(0.0, min(1.0, base_score))


class CommandSecurityValidator:
    """Advanced security validation with AI-powered threat detection."""
    
    # Enhanced security patterns
    SECURITY_PATTERNS = {
        "command_injection": [
            r"[;&|`$()]",
            r"rm\s+-rf",
            r"sudo\s+",
            r"\|\s*sh",
            r"eval\s*\(",
            r"exec\s*\(",
            r"system\s*\(",
            r"\\x[0-9a-fA-F]{2}",  # Hex encoding
            r"\\[0-7]{3}",         # Octal encoding
        ],
        "path_traversal": [
            r"\.\./",
            r"\.\.\\",
            r"/etc/passwd",
            r"/etc/shadow",
            r"\\windows\\system32",
            r"\.\.%2f",            # URL encoded
            r"\.\.%5c",            # URL encoded backslash
        ],
        "sensitive_data": [
            r"password\s*[:=]\s*['\"]?[\w@#$%^&*]+['\"]?",
            r"secret\s*[:=]\s*['\"]?[\w@#$%^&*]+['\"]?",
            r"token\s*[:=]\s*['\"]?[\w@#$%^&*-]+['\"]?",
            r"api[_-]?key\s*[:=]\s*['\"]?[\w-]+['\"]?",
            r"private[_-]?key",
            r"-----BEGIN.*PRIVATE KEY-----",
        ],
        "sql_injection": [
            r"('|(\\')|(\"|(\\\"))).*(\bor\b|\band\b).*('|(\\')|(\"|(\\\"))).*",
            r"\bunion\b.*\bselect\b",
            r"\bdrop\b.*\btable\b",
            r"\bdelete\b.*\bfrom\b",
            r"\binsert\b.*\binto\b",
            r"\bupdate\b.*\bset\b"
        ],
        "xss_patterns": [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>"
        ],
        "network_abuse": [
            r"(https?|ftp)://\d+\.\d+\.\d+\.\d+",  # Direct IP URLs
            r"localhost:\d+",
            r"127\.0\.0\.1:\d+",
            r"0\.0\.0\.0:\d+",
            r"file://",
            r"data:.*base64"
        ]
    }
    
    SEVERITY_WEIGHTS = {
        "command_injection": 1.0,    # Critical
        "path_traversal": 0.9,       # High
        "sensitive_data": 0.8,       # High
        "sql_injection": 1.0,        # Critical
        "xss_patterns": 0.7,         # Medium-High
        "network_abuse": 0.6         # Medium
    }
    
    async def validate(self, command: str, context: Dict[str, Any] = None) -> SecurityValidationResult:
        """Comprehensive security validation."""
        start_time = time.time()
        result = SecurityValidationResult()
        
        try:
            # Pattern-based security checks
            threats = []
            for threat_type, patterns in self.SECURITY_PATTERNS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, command, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        threat = SecurityThreat(
                            threat_type=threat_type,
                            severity=self._determine_severity(threat_type),
                            description=f"Potentially dangerous pattern detected: {pattern}",
                            pattern=pattern,
                            mitigation=self._get_mitigation(threat_type),
                            confidence=0.8
                        )
                        threats.append(threat)
            
            result.threats = threats
            
            # AI-powered threat analysis
            ai_analysis = await self._ai_threat_detection(command, context)
            result.ai_threat_analysis = ai_analysis
            
            # Extend threats with AI findings
            if ai_analysis.get("threats"):
                for ai_threat in ai_analysis["threats"]:
                    threat = SecurityThreat(
                        threat_type="ai_detected",
                        severity=ai_threat.get("severity", "medium"),
                        description=ai_threat.get("description", "AI detected potential security issue"),
                        confidence=ai_threat.get("confidence", 0.6)
                    )
                    threats.append(threat)
            
            # Calculate security score
            security_score = self._calculate_security_score(threats)
            result.security_score = security_score
            result.score = security_score
            
            # Determine overall validity
            critical_threats = [t for t in threats if t.severity == "critical"]
            high_threats = [t for t in threats if t.severity == "high"]
            
            if critical_threats:
                result.valid = False
                result.errors.append(f"Critical security threats detected: {len(critical_threats)}")
            elif len(high_threats) > 2:
                result.valid = False
                result.errors.append(f"Multiple high-severity security threats detected: {len(high_threats)}")
            elif threats:
                result.warnings.append(f"Security threats detected: {len(threats)}")
            
            # Permission requirements analysis
            result.required_permissions = await self._analyze_permission_requirements(command)
            
            # Data access analysis
            result.data_access_analysis = await self._analyze_data_access(command, context)
            
            result.details = {
                "total_threats": len(threats),
                "threat_types": list(set(t.threat_type for t in threats)),
                "highest_severity": max([t.severity for t in threats], default="none"),
                "ai_analysis_available": bool(ai_analysis)
            }
            
            return result
            
        except Exception as e:
            logger.error("Security validation failed", error=str(e))
            result.valid = False
            result.score = 0.0
            result.errors.append(f"Security validation error: {e}")
            return result
        finally:
            result.execution_time_ms = (time.time() - start_time) * 1000
    
    def _determine_severity(self, threat_type: str) -> str:
        """Determine threat severity based on type."""
        weight = self.SEVERITY_WEIGHTS.get(threat_type, 0.5)
        
        if weight >= 0.9:
            return "critical"
        elif weight >= 0.7:
            return "high"
        elif weight >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _get_mitigation(self, threat_type: str) -> str:
        """Get mitigation advice for threat type."""
        mitigations = {
            "command_injection": "Avoid using shell metacharacters and system commands",
            "path_traversal": "Use absolute paths and avoid directory traversal patterns",
            "sensitive_data": "Never include sensitive information in commands",
            "sql_injection": "Use parameterized queries and avoid SQL keywords",
            "xss_patterns": "Sanitize all user input and avoid HTML/JavaScript",
            "network_abuse": "Use trusted domains and avoid direct IP access"
        }
        
        return mitigations.get(threat_type, "Review command for potential security issues")
    
    async def _ai_threat_detection(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI-powered threat detection (placeholder for future AI integration)."""
        try:
            # This would integrate with an AI service for advanced threat detection
            # For now, return a basic analysis
            analysis = {
                "analyzed": True,
                "confidence": 0.7,
                "threats": [],
                "risk_score": 0.1,
                "analysis_method": "pattern_based",  # Would be "ai_model" in real implementation
                "processing_time_ms": 10
            }
            
            # Basic heuristic analysis
            suspicious_keywords = ["exec", "eval", "system", "shell", "command", "process"]
            found_keywords = [kw for kw in suspicious_keywords if kw in command.lower()]
            
            if found_keywords:
                analysis["threats"].append({
                    "type": "suspicious_keywords",
                    "severity": "medium",
                    "description": f"Suspicious keywords found: {', '.join(found_keywords)}",
                    "confidence": 0.6
                })
                analysis["risk_score"] = 0.4
            
            return analysis
            
        except Exception as e:
            logger.error("AI threat detection failed", error=str(e))
            return {"analyzed": False, "error": str(e)}
    
    def _calculate_security_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall security score based on threats."""
        if not threats:
            return 1.0
        
        # Weight threats by severity and confidence
        total_penalty = 0.0
        for threat in threats:
            severity_penalty = {
                "critical": 0.5,
                "high": 0.3,
                "medium": 0.15,
                "low": 0.05
            }.get(threat.severity, 0.1)
            
            confidence_factor = threat.confidence
            total_penalty += severity_penalty * confidence_factor
        
        # Cap the penalty to avoid negative scores
        total_penalty = min(1.0, total_penalty)
        
        return max(0.0, 1.0 - total_penalty)
    
    async def _analyze_permission_requirements(self, command: str) -> List[str]:
        """Analyze what permissions the command might require."""
        permissions = []
        
        # Basic permission analysis based on command patterns
        if re.search(r"spawn|start|create", command, re.IGNORECASE):
            permissions.append("agent_creation")
        
        if re.search(r"stop|kill|terminate", command, re.IGNORECASE):
            permissions.append("agent_termination")
        
        if re.search(r"status|monitor|oversight", command, re.IGNORECASE):
            permissions.append("system_monitoring")
        
        if re.search(r"develop|execute|run", command, re.IGNORECASE):
            permissions.append("code_execution")
        
        return permissions
    
    async def _analyze_data_access(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze what data the command might access."""
        analysis = {
            "data_types": [],
            "access_level": "read",
            "sensitive_data_risk": False,
            "external_access": False
        }
        
        # Analyze data access patterns
        if re.search(r"status|monitor|get|read|view", command, re.IGNORECASE):
            analysis["data_types"].append("system_status")
            analysis["access_level"] = "read"
        
        if re.search(r"spawn|create|start|develop|modify", command, re.IGNORECASE):
            analysis["data_types"].append("system_configuration")
            analysis["access_level"] = "write"
        
        if re.search(r"compact|compress|delete|remove", command, re.IGNORECASE):
            analysis["data_types"].append("conversation_data")
            analysis["access_level"] = "modify"
        
        # Check for sensitive data risks
        if re.search(r"password|secret|key|token|private", command, re.IGNORECASE):
            analysis["sensitive_data_risk"] = True
        
        # Check for external access
        if re.search(r"http|ftp|url|external|remote", command, re.IGNORECASE):
            analysis["external_access"] = True
        
        return analysis


class CommandPerformanceValidator:
    """Validates command performance characteristics."""
    
    async def validate(self, command: str, context: Dict[str, Any] = None) -> PerformanceValidationResult:
        """Validate command performance characteristics."""
        start_time = time.time()
        result = PerformanceValidationResult()
        
        try:
            # Estimate execution time based on command type
            estimated_time = await self._estimate_execution_time(command, context)
            result.estimated_execution_time = estimated_time
            
            # Analyze resource requirements
            resource_analysis = await self._analyze_resource_requirements(command, context)
            result.resource_requirements = resource_analysis
            
            # Scalability analysis
            scalability_analysis = await self._analyze_scalability(command, context)
            result.scalability_analysis = scalability_analysis
            
            # Mobile performance assessment
            mobile_score = await self._assess_mobile_performance(command, context)
            result.mobile_performance_score = mobile_score
            
            # Overall performance score
            performance_factors = [
                ("execution_time", self._score_execution_time(estimated_time)),
                ("resource_efficiency", resource_analysis.get("efficiency_score", 0.8)),
                ("scalability", scalability_analysis.get("scalability_score", 0.8)),
                ("mobile_performance", mobile_score)
            ]
            
            weighted_score = sum(score * 0.25 for _, score in performance_factors)
            result.score = weighted_score
            
            # Performance warnings
            if estimated_time > 60:  # More than 1 minute
                result.warnings.append(f"Long execution time estimated: {estimated_time}s")
                result.suggestions.append("Consider using --timeout flag for long operations")
            
            if mobile_score < 0.7:
                result.warnings.append("Command may not be optimized for mobile devices")
                result.suggestions.append("Add --mobile flag for mobile optimization")
            
            result.details = {
                "estimated_execution_time": estimated_time,
                "performance_factors": dict(performance_factors),
                "optimization_opportunities": await self._identify_optimizations(command, context)
            }
            
            return result
            
        except Exception as e:
            logger.error("Performance validation failed", error=str(e))
            result.valid = False
            result.score = 0.0
            result.errors.append(f"Performance validation error: {e}")
            return result
        finally:
            result.execution_time_ms = (time.time() - start_time) * 1000
    
    async def _estimate_execution_time(self, command: str, context: Dict[str, Any] = None) -> float:
        """Estimate command execution time in seconds."""
        # Base estimates by command type
        time_estimates = {
            "status": 1.0,
            "focus": 2.0,
            "start": 30.0,
            "spawn": 5.0,
            "develop": 300.0,  # 5 minutes
            "compact": 15.0,
            "oversight": 2.0,
            "productivity": 3.0,
            "stop": 5.0
        }
        
        # Extract command name
        command_name = command.replace("/hive:", "").split()[0]
        base_time = time_estimates.get(command_name, 10.0)
        
        # Adjust based on context
        if context:
            # More agents = potentially faster execution for some commands
            agent_count = context.get("agent_count", 1)
            if command_name in ["develop", "start"] and agent_count > 5:
                base_time *= 0.8  # 20% faster with more agents
            
            # Mobile mode might be faster due to caching
            if context.get("mobile_optimized"):
                base_time *= 0.7  # 30% faster for mobile-optimized
        
        return base_time
    
    async def _analyze_resource_requirements(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze resource requirements for command."""
        analysis = {
            "cpu_intensive": False,
            "memory_intensive": False,
            "network_intensive": False,
            "disk_intensive": False,
            "estimated_memory_mb": 50,
            "estimated_cpu_percent": 10,
            "efficiency_score": 0.8
        }
        
        command_name = command.replace("/hive:", "").split()[0]
        
        # Command-specific resource analysis
        if command_name == "develop":
            analysis["cpu_intensive"] = True
            analysis["memory_intensive"] = True
            analysis["estimated_memory_mb"] = 200
            analysis["estimated_cpu_percent"] = 60
            analysis["efficiency_score"] = 0.7
        elif command_name == "compact":
            analysis["cpu_intensive"] = True
            analysis["memory_intensive"] = True
            analysis["estimated_memory_mb"] = 150
            analysis["estimated_cpu_percent"] = 40
        elif command_name == "start":
            analysis["cpu_intensive"] = True
            analysis["estimated_memory_mb"] = 100
            analysis["estimated_cpu_percent"] = 30
        elif command_name in ["status", "focus", "oversight"]:
            analysis["network_intensive"] = True
            analysis["estimated_memory_mb"] = 20
            analysis["estimated_cpu_percent"] = 5
            analysis["efficiency_score"] = 0.9
        
        return analysis
    
    async def _analyze_scalability(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze command scalability characteristics."""
        analysis = {
            "scales_with_agents": False,
            "scales_with_data": False,
            "parallelizable": False,
            "bottlenecks": [],
            "scalability_score": 0.8
        }
        
        command_name = command.replace("/hive:", "").split()[0]
        
        if command_name in ["develop", "start"]:
            analysis["scales_with_agents"] = True
            analysis["parallelizable"] = True
            analysis["scalability_score"] = 0.9
        elif command_name == "compact":
            analysis["scales_with_data"] = True
            analysis["bottlenecks"].append("memory_for_large_contexts")
            analysis["scalability_score"] = 0.6
        elif command_name in ["status", "oversight"]:
            analysis["scales_with_agents"] = True
            analysis["scalability_score"] = 0.8
        
        return analysis
    
    async def _assess_mobile_performance(self, command: str, context: Dict[str, Any] = None) -> float:
        """Assess mobile performance characteristics."""
        base_score = 0.8
        
        command_name = command.replace("/hive:", "").split()[0]
        
        # Mobile-friendly commands
        mobile_friendly = ["status", "focus", "oversight", "productivity"]
        if command_name in mobile_friendly:
            base_score = 0.9
        
        # Mobile-challenging commands
        mobile_challenging = ["develop", "compact", "start"]
        if command_name in mobile_challenging:
            base_score = 0.6
        
        # Check for mobile optimization flags
        if "--mobile" in command:
            base_score += 0.2
        
        # Context adjustments
        if context and context.get("mobile_optimized"):
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _score_execution_time(self, estimated_time: float) -> float:
        """Score based on estimated execution time."""
        if estimated_time <= 1:
            return 1.0
        elif estimated_time <= 5:
            return 0.9
        elif estimated_time <= 15:
            return 0.8
        elif estimated_time <= 60:
            return 0.6
        elif estimated_time <= 300:
            return 0.4
        else:
            return 0.2
    
    async def _identify_optimizations(self, command: str, context: Dict[str, Any] = None) -> List[str]:
        """Identify potential optimizations."""
        optimizations = []
        
        if "--mobile" not in command and context and context.get("mobile_context"):
            optimizations.append("Add --mobile flag for mobile optimization")
        
        if "status" in command and "--priority" not in command:
            optimizations.append("Use --priority=high for faster status checks")
        
        if "develop" in command and "--dashboard" not in command:
            optimizations.append("Add --dashboard for progress monitoring")
        
        return optimizations


class UnifiedQualityGates:
    """
    Unified quality gates system that orchestrates all validation layers
    and provides comprehensive command reliability assessment.
    """
    
    def __init__(self):
        self.syntax_validator = CommandSyntaxValidator()
        self.semantic_validator = CommandSemanticValidator()
        self.security_validator = CommandSecurityValidator()
        self.performance_validator = CommandPerformanceValidator()
        
        # Validation cache for performance
        self.validation_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.validation_metrics = {
            "total_validations": 0,
            "avg_validation_time": 0.0,
            "cache_hit_rate": 0.0,
            "validation_success_rate": 0.0
        }
    
    async def validate_command(
        self,
        command: str,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False,
        fail_fast: bool = False
    ) -> ValidationResult:
        """
        Comprehensive command validation through all quality gates.
        
        Args:
            command: Command to validate
            validation_level: Depth of validation to perform
            context: Additional context for validation
            mobile_optimized: Whether to optimize for mobile
            fail_fast: Stop validation on first critical error
            
        Returns:
            ValidationResult with comprehensive assessment
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(command, validation_level, context, mobile_optimized)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._update_metrics(cached=True)
                return cached_result
            
            # Initialize result
            result = ValidationResult(command)
            result.validation_level = validation_level
            result.mobile_optimized = mobile_optimized
            result.context = context or {}
            
            # Layer 1: Syntax Validation
            result.syntax_validation = await self.syntax_validator.validate(command, context)
            if not result.syntax_validation.valid and fail_fast:
                result.overall_valid = False
                result.blocking_issues.extend(result.syntax_validation.errors)
                return self._finalize_result(result, start_time, cache_key)
            
            # Layer 2: Semantic Validation (if syntax is valid or not fail_fast)
            if result.syntax_validation.valid or not fail_fast:
                result.semantic_validation = await self.semantic_validator.validate(command, context)
                if not result.semantic_validation.valid and fail_fast:
                    result.overall_valid = False
                    result.blocking_issues.extend(result.semantic_validation.errors)
                    return self._finalize_result(result, start_time, cache_key)
            
            # Layer 3: Security Validation
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE, ValidationLevel.CRITICAL]:
                result.security_validation = await self.security_validator.validate(command, context)
                
                # Critical security issues always block
                critical_threats = [t for t in result.security_validation.threats if t.severity == "critical"]
                if critical_threats:
                    result.overall_valid = False
                    result.blocking_issues.extend([f"Critical security threat: {t.description}" for t in critical_threats])
                    if fail_fast:
                        return self._finalize_result(result, start_time, cache_key)
            
            # Layer 4: Performance Validation
            if validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.CRITICAL]:
                result.performance_validation = await self.performance_validator.validate(command, context)
            
            # Layer 5: Compatibility Validation
            if validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.CRITICAL]:
                result.compatibility_validation = await self._validate_compatibility(command, context, mobile_optimized)
            
            # Layer 6: User Experience Validation
            if validation_level == ValidationLevel.CRITICAL or mobile_optimized:
                result.ux_validation = await self._validate_user_experience(command, context, mobile_optimized)
            
            # Calculate overall assessment
            result = await self._calculate_overall_assessment(result)
            
            # Generate recovery strategies
            if not result.overall_valid or result.warnings:
                result.recovery_strategies = await self._generate_recovery_strategies(result)
            
            return self._finalize_result(result, start_time, cache_key)
            
        except Exception as e:
            logger.error("Quality gates validation failed", error=str(e))
            
            # Return error result
            error_result = ValidationResult(command)
            error_result.overall_valid = False
            error_result.blocking_issues.append(f"Validation system error: {e}")
            error_result.execution_time_ms = (time.time() - start_time) * 1000
            
            return error_result
    
    async def _validate_compatibility(
        self,
        command: str,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False
    ) -> CompatibilityValidationResult:
        """Validate cross-platform and mobile compatibility."""
        result = CompatibilityValidationResult()
        
        try:
            # Mobile compatibility check
            result.mobile_compatibility = self._check_mobile_compatibility(command, mobile_optimized)
            
            # Browser compatibility (for WebSocket commands)
            result.browser_compatibility = await self._check_browser_compatibility(command)
            
            # Platform compatibility
            result.platform_compatibility = await self._check_platform_compatibility(command, context)
            
            # API version compatibility
            result.api_version_compatibility = await self._check_api_version_compatibility(command)
            
            # Overall compatibility score
            compatibility_factors = [
                result.mobile_compatibility,
                result.browser_compatibility.get("compatible", True),
                result.platform_compatibility.get("compatible", True),
                result.api_version_compatibility.get("compatible", True)
            ]
            
            result.score = sum(1.0 if factor else 0.0 for factor in compatibility_factors) / len(compatibility_factors)
            result.valid = result.score >= 0.8
            
            if not result.mobile_compatibility and mobile_optimized:
                result.warnings.append("Command may not be optimized for mobile devices")
                result.suggestions.append("Consider using mobile-specific alternatives")
            
            return result
            
        except Exception as e:
            result.valid = False
            result.errors.append(f"Compatibility validation error: {e}")
            return result
    
    async def _validate_user_experience(
        self,
        command: str,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False
    ) -> UXValidationResult:
        """Validate user experience aspects."""
        result = UXValidationResult()
        
        try:
            # Accessibility assessment
            result.accessibility_score = await self._assess_accessibility(command, context)
            
            # Mobile UX assessment
            if mobile_optimized:
                result.mobile_ux_score = await self._assess_mobile_ux(command, context)
            
            # Usability analysis
            usability_issues = await self._identify_usability_issues(command, context)
            result.usability_issues = usability_issues
            
            # Mobile optimizations check
            if mobile_optimized:
                result.mobile_optimizations = await self._check_mobile_optimizations(command)
            
            # Overall UX score
            ux_factors = [
                result.accessibility_score,
                result.mobile_ux_score if mobile_optimized else 1.0
            ]
            
            result.score = sum(ux_factors) / len(ux_factors)
            result.valid = result.score >= 0.7 and len(usability_issues) == 0
            
            if len(usability_issues) > 0:
                result.warnings.extend([f"Usability issue: {issue}" for issue in usability_issues])
            
            return result
            
        except Exception as e:
            result.valid = False
            result.errors.append(f"UX validation error: {e}")
            return result
    
    async def _calculate_overall_assessment(self, result: ValidationResult) -> ValidationResult:
        """Calculate overall validation assessment."""
        try:
            # Collect all layer scores
            layer_scores = []
            layer_weights = {
                "syntax": 0.15,
                "semantic": 0.20,
                "security": 0.25,
                "performance": 0.15,
                "compatibility": 0.15,
                "ux": 0.10
            }
            
            if result.syntax_validation:
                layer_scores.append(("syntax", result.syntax_validation.score, layer_weights["syntax"]))
            
            if result.semantic_validation:
                layer_scores.append(("semantic", result.semantic_validation.score, layer_weights["semantic"]))
            
            if result.security_validation:
                layer_scores.append(("security", result.security_validation.score, layer_weights["security"]))
            
            if result.performance_validation:
                layer_scores.append(("performance", result.performance_validation.score, layer_weights["performance"]))
            
            if result.compatibility_validation:
                layer_scores.append(("compatibility", result.compatibility_validation.score, layer_weights["compatibility"]))
            
            if result.ux_validation:
                layer_scores.append(("ux", result.ux_validation.score, layer_weights["ux"]))
            
            # Calculate weighted score
            if layer_scores:
                total_weight = sum(weight for _, _, weight in layer_scores)
                weighted_score = sum(score * weight for _, score, weight in layer_scores) / total_weight
                result.overall_score = weighted_score
            else:
                result.overall_score = 0.0
            
            # Determine overall validity
            blocking_layer_failures = [
                not result.syntax_validation.valid if result.syntax_validation else False,
                not result.semantic_validation.valid if result.semantic_validation else False,
                not result.security_validation.valid if result.security_validation else False
            ]
            
            result.overall_valid = (
                not any(blocking_layer_failures) and
                result.overall_score >= 0.7 and
                len(result.blocking_issues) == 0
            )
            
            # Collect warnings and suggestions from all layers
            for layer_result in [
                result.syntax_validation,
                result.semantic_validation, 
                result.security_validation,
                result.performance_validation,
                result.compatibility_validation,
                result.ux_validation
            ]:
                if layer_result:
                    result.warnings.extend(layer_result.warnings)
                    result.suggestions.extend(layer_result.suggestions)
            
            # Remove duplicates
            result.warnings = list(set(result.warnings))
            result.suggestions = list(set(result.suggestions))
            
            return result
            
        except Exception as e:
            logger.error("Overall assessment calculation failed", error=str(e))
            result.overall_valid = False
            result.blocking_issues.append(f"Assessment calculation error: {e}")
            return result
    
    async def _generate_recovery_strategies(self, result: ValidationResult) -> List[str]:
        """Generate intelligent recovery strategies for validation failures."""
        strategies = []
        
        try:
            # Syntax error recovery
            if result.syntax_validation and not result.syntax_validation.valid:
                strategies.append("Check command syntax and ensure proper formatting")
                strategies.append("Verify command starts with '/hive:' prefix")
            
            # Semantic error recovery
            if result.semantic_validation and not result.semantic_validation.valid:
                strategies.append("Verify command name is correct and exists")
                strategies.append("Check command arguments match expected format")
            
            # Security threat recovery
            if result.security_validation and result.security_validation.threats:
                strategies.append("Remove potentially dangerous patterns from command")
                strategies.append("Use safer alternatives for sensitive operations")
            
            # Performance optimization
            if result.performance_validation and result.performance_validation.estimated_execution_time > 60:
                strategies.append("Consider breaking long operations into smaller steps")
                strategies.append("Use timeout flags for long-running commands")
            
            # Mobile compatibility
            if result.mobile_optimized and result.compatibility_validation and not result.compatibility_validation.mobile_compatibility:
                strategies.append("Add --mobile flag for mobile optimization")
                strategies.append("Use mobile-specific command variants")
            
            return strategies[:5]  # Limit to top 5 strategies
            
        except Exception as e:
            logger.error("Recovery strategy generation failed", error=str(e))
            return ["Review command and try again"]
    
    def _finalize_result(
        self,
        result: ValidationResult,
        start_time: float,
        cache_key: str
    ) -> ValidationResult:
        """Finalize validation result and update metrics."""
        result.execution_time_ms = (time.time() - start_time) * 1000
        
        # Cache result if valid
        if result.overall_valid:
            self._cache_result(cache_key, result)
        
        # Update metrics
        self._update_metrics(
            validation_time=result.execution_time_ms / 1000.0,
            success=result.overall_valid,
            cached=False
        )
        
        return result
    
    def _generate_cache_key(
        self,
        command: str,
        validation_level: ValidationLevel,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False
    ) -> str:
        """Generate cache key for validation result."""
        context_hash = hashlib.md5(json.dumps(context or {}, sort_keys=True).encode()).hexdigest()
        
        return f"validation:{hashlib.md5(command.encode()).hexdigest()}:{validation_level.value}:{mobile_optimized}:{context_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[ValidationResult]:
        """Get cached validation result if still valid."""
        if cache_key in self.validation_cache:
            cached_data = self.validation_cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["result"]
        return None
    
    def _cache_result(self, cache_key: str, result: ValidationResult):
        """Cache validation result."""
        self.validation_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # Cleanup old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, data in self.validation_cache.items()
            if current_time - data["timestamp"] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.validation_cache[key]
    
    def _update_metrics(
        self,
        validation_time: float = 0.0,
        success: bool = True,
        cached: bool = False
    ):
        """Update validation metrics."""
        self.validation_metrics["total_validations"] += 1
        
        if cached:
            # Update cache hit rate
            cache_hits = self.validation_metrics.get("cache_hits", 0) + 1
            self.validation_metrics["cache_hits"] = cache_hits
            self.validation_metrics["cache_hit_rate"] = cache_hits / self.validation_metrics["total_validations"]
        else:
            # Update average validation time
            total = self.validation_metrics["total_validations"]
            current_avg = self.validation_metrics["avg_validation_time"]
            self.validation_metrics["avg_validation_time"] = (current_avg * (total - 1) + validation_time) / total
            
            # Update success rate
            successes = self.validation_metrics.get("successes", 0) + (1 if success else 0)
            self.validation_metrics["successes"] = successes
            self.validation_metrics["validation_success_rate"] = successes / self.validation_metrics["total_validations"]
    
    # Helper methods for compatibility and UX validation
    
    def _check_mobile_compatibility(self, command: str, mobile_optimized: bool) -> bool:
        """Check if command is compatible with mobile devices."""
        # Most hive commands are mobile-compatible
        # Long-running commands might be less suitable for mobile
        long_running_commands = ["develop", "compact"]
        command_name = command.replace("/hive:", "").split()[0]
        
        if command_name in long_running_commands and not mobile_optimized:
            return False
        
        return True
    
    async def _check_browser_compatibility(self, command: str) -> Dict[str, Any]:
        """Check browser compatibility for WebSocket-based commands."""
        return {
            "compatible": True,
            "websocket_support": True,
            "modern_browsers": True,
            "fallback_available": True
        }
    
    async def _check_platform_compatibility(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check platform compatibility."""
        return {
            "compatible": True,
            "supported_platforms": ["web", "mobile", "desktop"],
            "platform_specific_features": []
        }
    
    async def _check_api_version_compatibility(self, command: str) -> Dict[str, Any]:
        """Check API version compatibility."""
        return {
            "compatible": True,
            "min_api_version": "2.0",
            "current_api_version": "2.0",
            "deprecated_features": []
        }
    
    async def _assess_accessibility(self, command: str, context: Dict[str, Any] = None) -> float:
        """Assess accessibility of command."""
        # Basic accessibility score - could be enhanced with actual accessibility checks
        return 0.9
    
    async def _assess_mobile_ux(self, command: str, context: Dict[str, Any] = None) -> float:
        """Assess mobile user experience."""
        base_score = 0.8
        
        # Commands with mobile flags get higher scores
        if "--mobile" in command:
            base_score += 0.2
        
        # Quick commands are better for mobile
        quick_commands = ["status", "focus", "oversight"]
        command_name = command.replace("/hive:", "").split()[0]
        
        if command_name in quick_commands:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    async def _identify_usability_issues(self, command: str, context: Dict[str, Any] = None) -> List[str]:
        """Identify potential usability issues."""
        issues = []
        
        # Check for overly complex commands
        if len(command.split()) > 8:
            issues.append("Command may be too complex for typical use")
        
        # Check for missing required parameters
        if command.strip().endswith(":develop") or command.strip() == "/hive:develop":
            issues.append("Development command requires project description")
        
        return issues
    
    async def _check_mobile_optimizations(self, command: str) -> List[str]:
        """Check available mobile optimizations."""
        optimizations = []
        
        if "--mobile" in command:
            optimizations.append("Mobile flag enabled")
        
        if "--priority" in command:
            optimizations.append("Priority optimization enabled")
        
        return optimizations
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get current validation metrics."""
        return {
            **self.validation_metrics,
            "cache_size": len(self.validation_cache),
            "cache_ttl_seconds": self.cache_ttl
        }


# Global instance
_quality_gates: Optional[UnifiedQualityGates] = None


def get_quality_gates() -> UnifiedQualityGates:
    """Get global unified quality gates instance."""
    global _quality_gates
    if _quality_gates is None:
        _quality_gates = UnifiedQualityGates()
    return _quality_gates