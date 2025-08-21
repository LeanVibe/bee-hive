"""
Plugin Security Framework for LeanVibe Agent Hive 2.0 - Epic 2 Phase 2.1

Implements comprehensive security framework for plugin validation, resource isolation,
and runtime monitoring while preserving Epic 1 performance achievements.

Key Features:
- Plugin resource isolation and sandboxing
- Runtime security monitoring and enforcement
- Code analysis and vulnerability scanning
- Resource usage tracking and limits
- Security policy enforcement

Epic 1 Preservation:
- <30ms security validation for API response times
- Minimal memory overhead for security checks
- Non-blocking security operations
- Efficient resource monitoring
"""

import asyncio
import ast
import hashlib
import inspect
import resource
import sys
import threading
import time
import traceback
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import logging
import tempfile
import shutil

from .logging_service import get_component_logger
from .advanced_plugin_manager import PluginSecurityLevel, PluginSecurityPolicy, SecurityReport

logger = get_component_logger("plugin_security_framework")


class SecurityViolationType(Enum):
    """Types of security violations."""
    DANGEROUS_IMPORT = "dangerous_import"
    SYSTEM_ACCESS = "system_access"
    NETWORK_ACCESS = "network_access"
    FILE_ACCESS = "file_access"
    DYNAMIC_CODE = "dynamic_code"
    RESOURCE_LIMIT = "resource_limit"
    PERMISSION_DENIED = "permission_denied"


class ResourceType(Enum):
    """Types of monitored resources."""
    MEMORY = "memory"
    CPU_TIME = "cpu_time"
    FILE_DESCRIPTORS = "file_descriptors"
    NETWORK_CONNECTIONS = "network_connections"
    EXECUTION_TIME = "execution_time"


@dataclass
class SecurityViolation:
    """Security violation record."""
    violation_type: SecurityViolationType
    description: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "description": self.description,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


@dataclass
class ResourceUsage:
    """Resource usage measurement."""
    resource_type: ResourceType
    current_value: float
    max_value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def usage_percentage(self) -> float:
        """Calculate usage as percentage of limit."""
        if self.max_value <= 0:
            return 0.0
        return (self.current_value / self.max_value) * 100
    
    @property
    def is_over_limit(self) -> bool:
        """Check if current usage exceeds limit."""
        return self.current_value > self.max_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type.value,
            "current_value": self.current_value,
            "max_value": self.max_value,
            "unit": self.unit,
            "usage_percentage": round(self.usage_percentage, 2),
            "is_over_limit": self.is_over_limit,
            "timestamp": self.timestamp.isoformat()
        }


class SecurityContext:
    """Security context for plugin execution."""
    
    def __init__(
        self,
        plugin_id: str,
        security_policy: PluginSecurityPolicy,
        execution_id: Optional[str] = None
    ):
        self.plugin_id = plugin_id
        self.security_policy = security_policy
        self.execution_id = execution_id or f"exec_{hash(time.time())}"
        
        # Monitoring state
        self.start_time = datetime.utcnow()
        self.resource_usage: Dict[ResourceType, ResourceUsage] = {}
        self.violations: List[SecurityViolation] = []
        
        # Resource limits
        self._original_limits: Dict[str, Any] = {}
        self._monitoring_active = False
    
    def add_violation(self, violation: SecurityViolation) -> None:
        """Add a security violation."""
        self.violations.append(violation)
        logger.warning("Security violation detected",
                      plugin_id=self.plugin_id,
                      violation_type=violation.violation_type.value,
                      severity=violation.severity,
                      description=violation.description)
    
    def record_resource_usage(self, resource_usage: ResourceUsage) -> None:
        """Record resource usage measurement."""
        self.resource_usage[resource_usage.resource_type] = resource_usage
        
        # Check for violations
        if resource_usage.is_over_limit:
            violation = SecurityViolation(
                violation_type=SecurityViolationType.RESOURCE_LIMIT,
                description=f"{resource_usage.resource_type.value} limit exceeded: {resource_usage.current_value}{resource_usage.unit} > {resource_usage.max_value}{resource_usage.unit}",
                severity="high",
                details=resource_usage.to_dict()
            )
            self.add_violation(violation)
    
    def get_execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        return (datetime.utcnow() - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat(),
            "execution_time_ms": round(self.get_execution_time_ms(), 2),
            "violations": [v.to_dict() for v in self.violations],
            "resource_usage": {rt.value: ru.to_dict() for rt, ru in self.resource_usage.items()},
            "security_level": self.security_policy.security_level.value
        }


class PluginCodeAnalyzer:
    """Analyzes plugin code for security vulnerabilities."""
    
    def __init__(self):
        # Dangerous patterns and their severity
        self.dangerous_patterns = {
            # System access patterns
            "os.system": ("critical", "System command execution"),
            "subprocess": ("critical", "Subprocess execution"),
            "os.popen": ("critical", "Process creation"),
            "os.spawn": ("critical", "Process spawning"),
            
            # Dynamic code execution
            "exec(": ("high", "Dynamic code execution"),
            "eval(": ("high", "Code evaluation"),
            "compile(": ("medium", "Code compilation"),
            "__import__": ("medium", "Dynamic import"),
            
            # File system access
            "open(": ("medium", "File access"),
            "os.remove": ("high", "File deletion"),
            "os.rmdir": ("high", "Directory deletion"),
            "shutil.rmtree": ("critical", "Recursive deletion"),
            
            # Network access
            "socket": ("high", "Network socket access"),
            "urllib": ("medium", "HTTP requests"),
            "requests": ("medium", "HTTP requests"),
            "http.client": ("medium", "HTTP client"),
            
            # Dangerous modules
            "ctypes": ("critical", "Native code access"),
            "marshal": ("high", "Object serialization"),
            "pickle": ("high", "Object deserialization"),
            "dill": ("high", "Extended pickle")
        }
        
        # Allowed imports by security level
        self.allowed_imports = {
            PluginSecurityLevel.TRUSTED: set(),  # No restrictions
            PluginSecurityLevel.VERIFIED: {
                "asyncio", "datetime", "logging", "typing", "dataclasses",
                "enum", "json", "time", "uuid", "hashlib", "base64"
            },
            PluginSecurityLevel.SANDBOX: {
                "asyncio", "datetime", "logging", "typing", "dataclasses",
                "enum", "json", "time", "uuid"
            },
            PluginSecurityLevel.UNTRUSTED: {
                "typing", "datetime", "enum"
            }
        }
    
    async def analyze_source_code(
        self,
        source_code: str,
        security_level: PluginSecurityLevel,
        plugin_id: str
    ) -> List[SecurityViolation]:
        """Analyze source code for security violations."""
        violations = []
        
        try:
            # Parse AST for detailed analysis
            tree = ast.parse(source_code)
            violations.extend(await self._analyze_ast(tree, security_level, plugin_id))
            
        except SyntaxError as e:
            violation = SecurityViolation(
                violation_type=SecurityViolationType.DYNAMIC_CODE,
                description=f"Syntax error in plugin code: {str(e)}",
                severity="high",
                details={"syntax_error": str(e)}
            )
            violations.append(violation)
        
        # Pattern-based analysis
        violations.extend(await self._analyze_patterns(source_code, security_level, plugin_id))
        
        return violations
    
    async def _analyze_ast(
        self,
        tree: ast.AST,
        security_level: PluginSecurityLevel,
        plugin_id: str
    ) -> List[SecurityViolation]:
        """Analyze AST for security violations."""
        violations = []
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                violations.extend(await self._check_import_security(node, security_level, plugin_id))
            
            # Check function calls
            elif isinstance(node, ast.Call):
                violations.extend(await self._check_call_security(node, security_level, plugin_id))
            
            # Check attribute access
            elif isinstance(node, ast.Attribute):
                violations.extend(await self._check_attribute_security(node, security_level, plugin_id))
        
        return violations
    
    async def _check_import_security(
        self,
        node: Union[ast.Import, ast.ImportFrom],
        security_level: PluginSecurityLevel,
        plugin_id: str
    ) -> List[SecurityViolation]:
        """Check import statements for security violations."""
        violations = []
        
        if security_level == PluginSecurityLevel.TRUSTED:
            return violations  # No restrictions for trusted plugins
        
        allowed = self.allowed_imports.get(security_level, set())
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name not in allowed:
                    violation = SecurityViolation(
                        violation_type=SecurityViolationType.DANGEROUS_IMPORT,
                        description=f"Unauthorized import: {module_name}",
                        severity="medium",
                        details={"module": module_name, "security_level": security_level.value}
                    )
                    violations.append(violation)
        
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name and module_name not in allowed:
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.DANGEROUS_IMPORT,
                    description=f"Unauthorized import from: {module_name}",
                    severity="medium",
                    details={"module": module_name, "security_level": security_level.value}
                )
                violations.append(violation)
        
        return violations
    
    async def _check_call_security(
        self,
        node: ast.Call,
        security_level: PluginSecurityLevel,
        plugin_id: str
    ) -> List[SecurityViolation]:
        """Check function calls for security violations."""
        violations = []
        
        # Get function name
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = ast.unparse(node.func) if hasattr(ast, 'unparse') else str(node.func)
        
        # Check against dangerous patterns
        for pattern, (severity, description) in self.dangerous_patterns.items():
            if pattern in func_name:
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.SYSTEM_ACCESS,
                    description=f"Dangerous function call: {description} ({func_name})",
                    severity=severity,
                    details={"function": func_name, "pattern": pattern}
                )
                violations.append(violation)
        
        return violations
    
    async def _check_attribute_security(
        self,
        node: ast.Attribute,
        security_level: PluginSecurityLevel,
        plugin_id: str
    ) -> List[SecurityViolation]:
        """Check attribute access for security violations."""
        violations = []
        
        # Check for dangerous attribute access patterns
        attr_path = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
        
        dangerous_attrs = [
            "__globals__", "__builtins__", "__code__", "__dict__",
            "__class__.__bases__", "func_globals"
        ]
        
        for dangerous_attr in dangerous_attrs:
            if dangerous_attr in attr_path:
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.SYSTEM_ACCESS,
                    description=f"Dangerous attribute access: {attr_path}",
                    severity="high",
                    details={"attribute": attr_path}
                )
                violations.append(violation)
        
        return violations
    
    async def _analyze_patterns(
        self,
        source_code: str,
        security_level: PluginSecurityLevel,
        plugin_id: str
    ) -> List[SecurityViolation]:
        """Analyze source code using pattern matching."""
        violations = []
        
        # Simple string-based pattern matching
        for pattern, (severity, description) in self.dangerous_patterns.items():
            if pattern in source_code:
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.DANGEROUS_IMPORT,
                    description=f"Dangerous pattern detected: {description}",
                    severity=severity,
                    details={"pattern": pattern}
                )
                violations.append(violation)
        
        return violations


class ResourceMonitor:
    """Monitors plugin resource usage in real-time."""
    
    def __init__(self):
        self._monitoring_contexts: Dict[str, SecurityContext] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._lock = threading.Lock()
    
    def start_monitoring_context(self, context: SecurityContext) -> None:
        """Start monitoring a security context."""
        with self._lock:
            self._monitoring_contexts[context.execution_id] = context
            context._monitoring_active = True
            
            # Start monitoring thread if not already running
            if not self._monitoring_active:
                self._start_monitoring_thread()
    
    def stop_monitoring_context(self, execution_id: str) -> Optional[SecurityContext]:
        """Stop monitoring a security context."""
        with self._lock:
            context = self._monitoring_contexts.pop(execution_id, None)
            if context:
                context._monitoring_active = False
            return context
    
    def _start_monitoring_thread(self) -> None:
        """Start the background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PluginResourceMonitor"
        )
        self._monitoring_thread.start()
        logger.info("Plugin resource monitoring started")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                with self._lock:
                    contexts = list(self._monitoring_contexts.values())
                
                for context in contexts:
                    if context._monitoring_active:
                        self._measure_resource_usage(context)
                
                # Epic 1: Efficient monitoring interval
                time.sleep(1.0)  # 1 second intervals
                
            except Exception as e:
                logger.error("Error in resource monitoring loop", error=str(e))
                time.sleep(5.0)  # Wait longer on error
    
    def _measure_resource_usage(self, context: SecurityContext) -> None:
        """Measure resource usage for a context."""
        try:
            import psutil
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_usage = ResourceUsage(
                resource_type=ResourceType.MEMORY,
                current_value=memory_mb,
                max_value=context.security_policy.max_memory_mb,
                unit="MB"
            )
            context.record_resource_usage(memory_usage)
            
            # CPU time (simplified)
            cpu_percent = process.cpu_percent()
            cpu_usage = ResourceUsage(
                resource_type=ResourceType.CPU_TIME,
                current_value=cpu_percent,
                max_value=80.0,  # 80% CPU limit
                unit="%"
            )
            context.record_resource_usage(cpu_usage)
            
            # Execution time
            exec_time_ms = context.get_execution_time_ms()
            exec_time_usage = ResourceUsage(
                resource_type=ResourceType.EXECUTION_TIME,
                current_value=exec_time_ms,
                max_value=context.security_policy.max_cpu_time_ms,
                unit="ms"
            )
            context.record_resource_usage(exec_time_usage)
            
        except Exception as e:
            logger.debug("Failed to measure resource usage",
                        plugin_id=context.plugin_id,
                        error=str(e))
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        with self._lock:
            self._monitoring_contexts.clear()
        
        logger.info("Plugin resource monitoring stopped")


class PluginSecurityFramework:
    """
    Comprehensive plugin security framework.
    
    Epic 1 Optimizations:
    - <30ms security validation
    - Minimal memory overhead
    - Non-blocking operations
    - Efficient resource monitoring
    """
    
    def __init__(self):
        self.code_analyzer = PluginCodeAnalyzer()
        self.resource_monitor = ResourceMonitor()
        
        # Security state
        self._security_contexts: Dict[str, SecurityContext] = {}
        self._global_policies: Dict[PluginSecurityLevel, PluginSecurityPolicy] = {}
        
        # Epic 1: Performance tracking
        self._validation_times: List[float] = []
        
        self._initialize_default_policies()
        logger.info("PluginSecurityFramework initialized")
    
    def _initialize_default_policies(self) -> None:
        """Initialize default security policies."""
        self._global_policies = {
            PluginSecurityLevel.TRUSTED: PluginSecurityPolicy(
                security_level=PluginSecurityLevel.TRUSTED,
                max_memory_mb=200,
                max_cpu_time_ms=1000,
                allowed_imports=set(),  # No restrictions
                network_access=True,
                file_system_access=True
            ),
            PluginSecurityLevel.VERIFIED: PluginSecurityPolicy(
                security_level=PluginSecurityLevel.VERIFIED,
                max_memory_mb=100,
                max_cpu_time_ms=500,
                allowed_imports={"asyncio", "datetime", "logging", "typing", "json"},
                network_access=False,
                file_system_access=False
            ),
            PluginSecurityLevel.SANDBOX: PluginSecurityPolicy(
                security_level=PluginSecurityLevel.SANDBOX,
                max_memory_mb=50,  # Epic 1: Conservative limit
                max_cpu_time_ms=100,  # Epic 1: Response time preservation
                allowed_imports={"typing", "datetime", "enum"},
                network_access=False,
                file_system_access=False
            ),
            PluginSecurityLevel.UNTRUSTED: PluginSecurityPolicy(
                security_level=PluginSecurityLevel.UNTRUSTED,
                max_memory_mb=20,  # Epic 1: Very conservative
                max_cpu_time_ms=50,   # Epic 1: Strict time limit
                allowed_imports={"typing"},
                network_access=False,
                file_system_access=False
            )
        }
    
    async def validate_plugin_security(
        self,
        plugin_id: str,
        source_code: Optional[str] = None,
        source_path: Optional[Path] = None,
        security_level: PluginSecurityLevel = PluginSecurityLevel.SANDBOX
    ) -> SecurityReport:
        """
        Comprehensive plugin security validation.
        
        Epic 1: <30ms validation target
        """
        start_time = datetime.utcnow()
        
        try:
            violations = []
            warnings = []
            
            # Code analysis
            if source_code:
                code_violations = await self.code_analyzer.analyze_source_code(
                    source_code, security_level, plugin_id
                )
                violations.extend(code_violations)
            
            # File analysis
            if source_path:
                file_violations = await self._analyze_file_security(source_path, plugin_id)
                violations.extend(file_violations)
            
            # Determine final security level
            final_security_level = security_level
            if any(v.severity == "critical" for v in violations):
                final_security_level = PluginSecurityLevel.UNTRUSTED
            elif any(v.severity == "high" for v in violations):
                final_security_level = PluginSecurityLevel.SANDBOX
            
            # Create security report
            is_safe = len([v for v in violations if v.severity in ["critical", "high"]]) == 0
            
            report = SecurityReport(
                plugin_id=plugin_id,
                is_safe=is_safe,
                security_level=final_security_level,
                violations=[v.description for v in violations],
                warnings=[v.description for v in violations if v.severity in ["low", "medium"]],
                resource_usage={}
            )
            
            # Epic 1: Track validation performance
            validation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._validation_times.append(validation_time_ms)
            if len(self._validation_times) > 100:
                self._validation_times.pop(0)
            
            logger.debug("Plugin security validation completed",
                        plugin_id=plugin_id,
                        is_safe=is_safe,
                        violations_count=len(violations),
                        validation_time_ms=round(validation_time_ms, 2))
            
            return report
            
        except Exception as e:
            logger.error("Security validation failed",
                        plugin_id=plugin_id,
                        error=str(e))
            
            return SecurityReport(
                plugin_id=plugin_id,
                is_safe=False,
                security_level=PluginSecurityLevel.UNTRUSTED,
                violations=[f"Validation error: {str(e)}"]
            )
    
    async def _analyze_file_security(self, file_path: Path, plugin_id: str) -> List[SecurityViolation]:
        """Analyze file-based security issues."""
        violations = []
        
        try:
            if not file_path.exists():
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.FILE_ACCESS,
                    description="Plugin file does not exist",
                    severity="critical"
                )
                violations.append(violation)
                return violations
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > 1024 * 1024:  # 1MB limit
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.RESOURCE_LIMIT,
                    description=f"Plugin file too large: {file_size} bytes",
                    severity="medium"
                )
                violations.append(violation)
            
            # Check file permissions (basic check)
            if file_path.stat().st_mode & 0o077:  # World or group writable
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.PERMISSION_DENIED,
                    description="Plugin file has unsafe permissions",
                    severity="medium"
                )
                violations.append(violation)
            
        except Exception as e:
            violation = SecurityViolation(
                violation_type=SecurityViolationType.FILE_ACCESS,
                description=f"File analysis error: {str(e)}",
                severity="high"
            )
            violations.append(violation)
        
        return violations
    
    @asynccontextmanager
    async def secure_execution_context(
        self,
        plugin_id: str,
        security_policy: Optional[PluginSecurityPolicy] = None
    ):
        """Create a secure execution context for plugin operations."""
        
        # Use default policy if none provided
        if not security_policy:
            security_policy = self._global_policies[PluginSecurityLevel.SANDBOX]
        
        # Create security context
        context = SecurityContext(plugin_id, security_policy)
        self._security_contexts[context.execution_id] = context
        
        # Start monitoring
        self.resource_monitor.start_monitoring_context(context)
        
        try:
            yield context
        finally:
            # Stop monitoring and get final context
            final_context = self.resource_monitor.stop_monitoring_context(context.execution_id)
            if final_context:
                self._security_contexts[context.execution_id] = final_context
    
    def get_security_context(self, execution_id: str) -> Optional[SecurityContext]:
        """Get security context by execution ID."""
        return self._security_contexts.get(execution_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get security framework performance metrics."""
        if not self._validation_times:
            return {"validation_times": {"count": 0}}
        
        avg_time = sum(self._validation_times) / len(self._validation_times)
        max_time = max(self._validation_times)
        min_time = min(self._validation_times)
        
        return {
            "validation_times": {
                "count": len(self._validation_times),
                "avg_ms": round(avg_time, 2),
                "max_ms": round(max_time, 2),
                "min_ms": round(min_time, 2),
                "epic1_compliant": avg_time < 30  # Epic 1: <30ms target
            },
            "active_contexts": len(self._security_contexts),
            "monitoring_active": self.resource_monitor._monitoring_active
        }
    
    async def cleanup(self) -> None:
        """Cleanup security framework resources."""
        logger.info("Cleaning up PluginSecurityFramework")
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        
        # Clear contexts
        self._security_contexts.clear()
        self._validation_times.clear()
        
        logger.info("PluginSecurityFramework cleanup complete")


# Global instance for framework access
_security_framework: Optional[PluginSecurityFramework] = None


def get_plugin_security_framework() -> PluginSecurityFramework:
    """Get global plugin security framework instance."""
    global _security_framework
    if _security_framework is None:
        _security_framework = PluginSecurityFramework()
    return _security_framework