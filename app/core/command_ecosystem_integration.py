"""
LeanVibe Agent Hive 2.0 - Command Ecosystem Integration

This module integrates all the enhanced command system components and provides
backward compatibility, migration helpers, and unified access to the improved
command infrastructure.

Features:
- Unified access to all enhanced command components
- Backward compatibility with existing command implementations
- Automatic migration and upgrade helpers
- Performance monitoring and analytics
- Mobile optimization orchestration
- Quality gates integration
- Command discovery and validation
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import structlog

# Import enhanced command components
from .unified_compression_command import get_unified_compressor, UnifiedCompressionCommand
from .enhanced_command_discovery import get_command_discovery, IntelligentCommandDiscovery
from .unified_quality_gates import get_quality_gates, UnifiedQualityGates, ValidationLevel
from .hive_slash_commands import get_hive_command_registry, execute_hive_command

logger = structlog.get_logger()


class CommandEcosystemStatus(Enum):
    """Status of the command ecosystem components."""
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded" 
    ERROR = "error"


class EcosystemMetrics:
    """Comprehensive metrics for the command ecosystem."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.total_commands_executed = 0
        self.successful_commands = 0
        self.failed_commands = 0
        self.cached_commands = 0
        self.mobile_optimized_commands = 0
        self.avg_execution_time_ms = 0.0
        self.compression_operations = 0
        self.discovery_queries = 0
        self.validation_checks = 0
        self.quality_gate_passes = 0
        self.quality_gate_failures = 0
        
        # Performance tracking
        self.performance_history = []
        self.error_patterns = {}
        self.usage_patterns = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        success_rate = self.successful_commands / max(1, self.total_commands_executed)
        cache_hit_rate = self.cached_commands / max(1, self.total_commands_executed)
        mobile_usage_rate = self.mobile_optimized_commands / max(1, self.total_commands_executed)
        quality_gate_pass_rate = self.quality_gate_passes / max(1, self.validation_checks)
        
        return {
            "uptime_seconds": uptime_seconds,
            "total_commands_executed": self.total_commands_executed,
            "success_rate": round(success_rate, 3),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "mobile_usage_rate": round(mobile_usage_rate, 3),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "commands_per_minute": round(self.total_commands_executed / max(1, uptime_seconds / 60), 2),
            "compression_operations": self.compression_operations,
            "discovery_queries": self.discovery_queries,
            "validation_checks": self.validation_checks,
            "quality_gate_pass_rate": round(quality_gate_pass_rate, 3),
            "performance_grade": self._calculate_performance_grade(),
            "system_health": self._calculate_system_health()
        }
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade."""
        success_rate = self.successful_commands / max(1, self.total_commands_executed)
        
        if success_rate >= 0.95 and self.avg_execution_time_ms < 1000:
            return "A"
        elif success_rate >= 0.90 and self.avg_execution_time_ms < 2000:
            return "B"
        elif success_rate >= 0.80 and self.avg_execution_time_ms < 5000:
            return "C"
        else:
            return "D"
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health."""
        success_rate = self.successful_commands / max(1, self.total_commands_executed)
        quality_rate = self.quality_gate_passes / max(1, self.validation_checks)
        
        if success_rate >= 0.95 and quality_rate >= 0.90:
            return "excellent"
        elif success_rate >= 0.85 and quality_rate >= 0.80:
            return "good"
        elif success_rate >= 0.70 and quality_rate >= 0.70:
            return "fair"
        else:
            return "poor"


class CommandEcosystemIntegration:
    """
    Central integration point for the enhanced command ecosystem.
    
    This class orchestrates all enhanced command components and provides
    a unified interface for command execution, validation, discovery,
    and compression operations.
    """
    
    def __init__(self):
        self.status = CommandEcosystemStatus.INITIALIZING
        self.metrics = EcosystemMetrics()
        self.migration_status = {}
        self.feature_flags = {}
        
        # Component instances
        self._compressor: Optional[UnifiedCompressionCommand] = None
        self._discovery: Optional[IntelligentCommandDiscovery] = None
        self._quality_gates: Optional[UnifiedQualityGates] = None
        
        # Backward compatibility mappings
        self.legacy_command_mappings = self._initialize_legacy_mappings()
        self.migration_helpers = {}
        
        # Performance tracking
        self.performance_monitor = None
        self.analytics_enabled = True
    
    async def initialize(self) -> bool:
        """Initialize the enhanced command ecosystem."""
        try:
            logger.info("🚀 Initializing LeanVibe Agent Hive 2.0 Command Ecosystem")
            
            # Initialize core components
            await self._initialize_components()
            
            # Setup backward compatibility
            await self._setup_backward_compatibility()
            
            # Initialize performance monitoring
            await self._initialize_performance_monitoring()
            
            # Setup feature flags
            await self._setup_feature_flags()
            
            # Validate system readiness
            readiness_check = await self._validate_system_readiness()
            if not readiness_check["ready"]:
                logger.error("System readiness check failed", issues=readiness_check["issues"])
                self.status = CommandEcosystemStatus.DEGRADED
                return False
            
            self.status = CommandEcosystemStatus.READY
            logger.info("✅ Command ecosystem initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error("❌ Failed to initialize command ecosystem", error=str(e))
            self.status = CommandEcosystemStatus.ERROR
            return False
    
    async def execute_enhanced_command(
        self,
        command: str,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False,
        use_quality_gates: bool = True,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Dict[str, Any]:
        """
        Execute a command through the enhanced ecosystem with full validation,
        optimization, and analytics.
        """
        start_time = time.time()
        execution_id = f"cmd_{int(time.time() * 1000)}"
        
        try:
            # Record command attempt
            self.metrics.total_commands_executed += 1
            if mobile_optimized:
                self.metrics.mobile_optimized_commands += 1
            
            logger.info("🎯 Executing enhanced command", 
                       command=command, 
                       execution_id=execution_id,
                       mobile_optimized=mobile_optimized)
            
            # Phase 1: Quality Gates Validation (if enabled)
            validation_result = None
            if use_quality_gates and self._quality_gates:
                self.metrics.validation_checks += 1
                
                validation_result = await self._quality_gates.validate_command(
                    command=command,
                    validation_level=validation_level,
                    context=context,
                    mobile_optimized=mobile_optimized
                )
                
                if validation_result.overall_valid:
                    self.metrics.quality_gate_passes += 1
                else:
                    self.metrics.quality_gate_failures += 1
                    
                    # Return validation failure with recovery strategies
                    return {
                        "success": False,
                        "error": "Command failed quality gates validation",
                        "validation_result": validation_result.to_dict(),
                        "recovery_strategies": validation_result.recovery_strategies,
                        "execution_id": execution_id,
                        "execution_time_ms": (time.time() - start_time) * 1000
                    }
            
            # Phase 2: Command Discovery and Enhancement (if needed)
            enhanced_command = command
            if self._discovery:
                # Check if command can be enhanced for mobile
                if mobile_optimized and "--mobile" not in command:
                    mobile_validation = await self._discovery.validate_command(
                        command, context, mobile_optimized=True
                    )
                    if mobile_validation.mobile_compatible:
                        enhanced_command = self._add_mobile_optimization(command)
            
            # Phase 3: Execute Command
            execution_result = await self._execute_core_command(
                enhanced_command, context, mobile_optimized
            )
            
            # Phase 4: Post-processing and Enhancement
            if execution_result.get("success"):
                self.metrics.successful_commands += 1
                
                # Apply mobile optimizations to result
                if mobile_optimized:
                    execution_result = await self._apply_mobile_result_optimizations(
                        execution_result, context
                    )
                
                # Cache successful results if applicable
                if self._should_cache_result(enhanced_command, execution_result):
                    self.metrics.cached_commands += 1
                    await self._cache_command_result(enhanced_command, execution_result, context)
            else:
                self.metrics.failed_commands += 1
                
                # Attempt intelligent error recovery
                recovery_result = await self._attempt_error_recovery(
                    enhanced_command, execution_result, context
                )
                if recovery_result["recovered"]:
                    execution_result = recovery_result["result"]
                    execution_result["recovered"] = True
                    self.metrics.successful_commands += 1
                    self.metrics.failed_commands -= 1
            
            # Phase 5: Analytics and Learning
            execution_time = (time.time() - start_time) * 1000
            await self._record_execution_analytics(
                command, execution_result, execution_time, context, validation_result
            )
            
            # Update performance metrics
            self._update_performance_metrics(execution_time)
            
            # Add ecosystem metadata to result
            execution_result.update({
                "execution_id": execution_id,
                "ecosystem_version": "2.0",
                "enhanced_execution": True,
                "mobile_optimized": mobile_optimized,
                "execution_time_ms": execution_time,
                "quality_gates_used": use_quality_gates,
                "validation_level": validation_level.value if validation_level else None,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return execution_result
            
        except Exception as e:
            self.metrics.failed_commands += 1
            execution_time = (time.time() - start_time) * 1000
            
            logger.error("❌ Enhanced command execution failed", 
                        command=command, 
                        execution_id=execution_id, 
                        error=str(e))
            
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution_id,
                "execution_time_ms": execution_time,
                "ecosystem_error": True,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def compress_content(
        self,
        content: str,
        strategy: str = "adaptive",
        level: str = "standard",
        mobile_optimized: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Enhanced content compression using the unified compression system."""
        try:
            if not self._compressor:
                raise ValueError("Compression system not available")
            
            self.metrics.compression_operations += 1
            
            result = await self._compressor.compress(
                content=content,
                strategy=strategy,
                level=level,
                mobile_optimized=mobile_optimized,
                **kwargs
            )
            
            return result.to_dict()
            
        except Exception as e:
            logger.error("Enhanced compression failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "compression_attempted": True
            }
    
    async def discover_commands(
        self,
        user_intent: str,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Enhanced command discovery with AI-powered suggestions."""
        try:
            if not self._discovery:
                raise ValueError("Command discovery system not available")
            
            self.metrics.discovery_queries += 1
            
            suggestions = await self._discovery.discover_commands(
                user_intent=user_intent,
                context=context,
                mobile_optimized=mobile_optimized,
                limit=limit
            )
            
            return [suggestion.to_dict() for suggestion in suggestions]
            
        except Exception as e:
            logger.error("Enhanced command discovery failed", error=str(e))
            return []
    
    async def get_system_status(self, include_detailed_metrics: bool = False) -> Dict[str, Any]:
        """Get comprehensive ecosystem status and metrics."""
        try:
            status = {
                "ecosystem_status": self.status.value,
                "ecosystem_version": "2.0",
                "components": {
                    "compression": self._compressor is not None,
                    "discovery": self._discovery is not None,
                    "quality_gates": self._quality_gates is not None,
                    "command_registry": True  # Always available
                },
                "metrics": self.metrics.to_dict(),
                "backward_compatibility": {
                    "legacy_mappings_count": len(self.legacy_command_mappings),
                    "migration_status": self.migration_status
                },
                "feature_flags": self.feature_flags,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if include_detailed_metrics:
                # Add component-specific metrics
                if self._quality_gates:
                    status["detailed_metrics"] = {
                        "quality_gates": self._quality_gates.get_validation_metrics(),
                        "performance_history": self.metrics.performance_history[-10:],  # Last 10 entries
                        "error_patterns": dict(list(self.metrics.error_patterns.items())[:5])  # Top 5 patterns
                    }
            
            return status
            
        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            return {
                "ecosystem_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def migrate_legacy_command(self, legacy_command: str) -> Dict[str, Any]:
        """Migrate a legacy command to the enhanced format."""
        try:
            # Check if command needs migration
            if legacy_command in self.legacy_command_mappings:
                migration = self.legacy_command_mappings[legacy_command]
                
                return {
                    "migration_needed": True,
                    "legacy_command": legacy_command,
                    "enhanced_command": migration["enhanced_command"],
                    "migration_notes": migration.get("notes", []),
                    "breaking_changes": migration.get("breaking_changes", []),
                    "migration_guide": migration.get("guide", "")
                }
            else:
                return {
                    "migration_needed": False,
                    "command": legacy_command,
                    "message": "Command is already in enhanced format or not found"
                }
                
        except Exception as e:
            logger.error("Legacy command migration failed", error=str(e))
            return {
                "migration_needed": False,
                "error": str(e)
            }
    
    # Private Methods
    
    async def _initialize_components(self):
        """Initialize all enhanced command components."""
        try:
            # Initialize compression system
            self._compressor = get_unified_compressor()
            logger.info("✅ Unified compression system initialized")
            
            # Initialize command discovery
            self._discovery = get_command_discovery()
            logger.info("✅ Enhanced command discovery initialized")
            
            # Initialize quality gates
            self._quality_gates = get_quality_gates()
            logger.info("✅ Unified quality gates initialized")
            
        except Exception as e:
            logger.error("Component initialization failed", error=str(e))
            raise
    
    async def _setup_backward_compatibility(self):
        """Setup backward compatibility mappings and helpers."""
        try:
            # Initialize migration helpers
            self.migration_helpers = {
                "compress_context": self._migrate_context_compression,
                "compress_memory": self._migrate_memory_compression,
                "compress_conversation": self._migrate_conversation_compression
            }
            
            logger.info("✅ Backward compatibility configured")
            
        except Exception as e:
            logger.error("Backward compatibility setup failed", error=str(e))
            raise
    
    async def _initialize_performance_monitoring(self):
        """Initialize performance monitoring and analytics."""
        try:
            # Setup performance tracking
            self.performance_monitor = {
                "enabled": True,
                "sample_rate": 1.0,  # Track all executions
                "metrics_retention_hours": 24
            }
            
            logger.info("✅ Performance monitoring initialized")
            
        except Exception as e:
            logger.error("Performance monitoring initialization failed", error=str(e))
            raise
    
    async def _setup_feature_flags(self):
        """Setup feature flags for gradual rollout and experimentation."""
        self.feature_flags = {
            "enhanced_compression": True,
            "ai_command_discovery": True,
            "advanced_quality_gates": True,
            "mobile_optimizations": True,
            "intelligent_error_recovery": True,
            "performance_analytics": True,
            "automatic_migrations": True
        }
    
    async def _validate_system_readiness(self) -> Dict[str, Any]:
        """Validate that the system is ready for enhanced operations."""
        issues = []
        
        # Check component availability
        if not self._compressor:
            issues.append("Compression system not available")
        
        if not self._discovery:
            issues.append("Command discovery system not available")
        
        if not self._quality_gates:
            issues.append("Quality gates system not available")
        
        # Check command registry
        try:
            registry = get_hive_command_registry()
            if not registry or len(registry.commands) == 0:
                issues.append("Command registry is empty or unavailable")
        except Exception as e:
            issues.append(f"Command registry error: {e}")
        
        return {
            "ready": len(issues) == 0,
            "issues": issues,
            "component_count": sum(1 for comp in [self._compressor, self._discovery, self._quality_gates] if comp),
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_core_command(
        self, 
        command: str, 
        context: Dict[str, Any] = None, 
        mobile_optimized: bool = False
    ) -> Dict[str, Any]:
        """Execute command using the core command system."""
        try:
            # Use enhanced context
            enhanced_context = {
                **(context or {}),
                "ecosystem_version": "2.0",
                "mobile_optimized": mobile_optimized,
                "enhanced_execution": True
            }
            
            # Execute through core system
            result = await execute_hive_command(command, enhanced_context)
            return result
            
        except Exception as e:
            logger.error("Core command execution failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    def _add_mobile_optimization(self, command: str) -> str:
        """Add mobile optimization flags to command."""
        if "--mobile" not in command:
            command += " --mobile"
        return command
    
    async def _apply_mobile_result_optimizations(
        self, 
        result: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Apply mobile-specific optimizations to command results."""
        try:
            # Add mobile-specific metadata
            result["mobile_optimized"] = True
            
            # Truncate long text fields for mobile
            if "message" in result and len(str(result["message"])) > 200:
                result["message"] = str(result["message"])[:197] + "..."
                result["truncated"] = True
            
            # Add mobile quick actions if applicable
            if "recommendations" in result and isinstance(result["recommendations"], list):
                mobile_actions = []
                for rec in result["recommendations"][:3]:  # Limit to 3 for mobile
                    if isinstance(rec, dict) and "command" in rec:
                        mobile_actions.append({
                            "title": rec.get("title", rec.get("description", "Action"))[:30],
                            "command": rec["command"]
                        })
                
                if mobile_actions:
                    result["mobile_quick_actions"] = mobile_actions
            
            return result
            
        except Exception as e:
            logger.error("Mobile result optimization failed", error=str(e))
            return result
    
    def _should_cache_result(self, command: str, result: Dict[str, Any]) -> bool:
        """Determine if command result should be cached."""
        # Cache successful, non-state-changing commands
        cacheable_commands = ["status", "focus", "productivity", "help"]
        non_cacheable_commands = ["start", "spawn", "develop", "stop", "compact"]
        
        command_name = command.replace("/hive:", "").split()[0]
        
        if command_name in non_cacheable_commands:
            return False
        
        return (
            result.get("success", False) and
            command_name in cacheable_commands
        )
    
    async def _cache_command_result(
        self, 
        command: str, 
        result: Dict[str, Any], 
        context: Dict[str, Any] = None
    ):
        """Cache command result for future use."""
        try:
            # This would integrate with a caching system
            # For now, just log the cache operation
            logger.debug("Caching command result", command=command, cached=True)
            
        except Exception as e:
            logger.error("Result caching failed", error=str(e))
    
    async def _attempt_error_recovery(
        self, 
        command: str, 
        error_result: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Attempt intelligent error recovery."""
        try:
            # Basic error recovery strategies
            recovery_strategies = [
                self._retry_with_different_params,
                self._suggest_alternative_command,
                self._check_prerequisites_and_fix
            ]
            
            for strategy in recovery_strategies:
                try:
                    recovery_result = await strategy(command, error_result, context)
                    if recovery_result.get("success"):
                        return {
                            "recovered": True,
                            "result": recovery_result,
                            "recovery_method": strategy.__name__
                        }
                except Exception as e:
                    logger.debug("Recovery strategy failed", strategy=strategy.__name__, error=str(e))
            
            return {"recovered": False}
            
        except Exception as e:
            logger.error("Error recovery failed", error=str(e))
            return {"recovered": False}
    
    async def _record_execution_analytics(
        self,
        command: str,
        result: Dict[str, Any],
        execution_time: float,
        context: Dict[str, Any] = None,
        validation_result = None
    ):
        """Record detailed execution analytics for learning and improvement."""
        try:
            if not self.analytics_enabled:
                return
            
            analytics_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "command": command,
                "success": result.get("success", False),
                "execution_time_ms": execution_time,
                "mobile_optimized": context.get("mobile_optimized", False) if context else False,
                "cached": result.get("cached", False),
                "quality_gates_passed": validation_result.overall_valid if validation_result else None,
                "error_type": result.get("error", "") if not result.get("success") else None
            }
            
            # Add to performance history
            self.metrics.performance_history.append(analytics_record)
            
            # Keep only last 100 records
            if len(self.metrics.performance_history) > 100:
                self.metrics.performance_history = self.metrics.performance_history[-100:]
            
            # Track error patterns
            if not result.get("success") and result.get("error"):
                error_type = result.get("error", "unknown")
                self.metrics.error_patterns[error_type] = self.metrics.error_patterns.get(error_type, 0) + 1
            
            # Track usage patterns
            command_name = command.replace("/hive:", "").split()[0]
            self.metrics.usage_patterns[command_name] = self.metrics.usage_patterns.get(command_name, 0) + 1
            
        except Exception as e:
            logger.error("Analytics recording failed", error=str(e))
    
    def _update_performance_metrics(self, execution_time: float):
        """Update rolling performance metrics."""
        try:
            # Update average execution time
            total_commands = self.metrics.total_commands_executed
            current_avg = self.metrics.avg_execution_time_ms
            
            self.metrics.avg_execution_time_ms = (
                (current_avg * (total_commands - 1) + execution_time) / total_commands
            )
            
        except Exception as e:
            logger.error("Performance metrics update failed", error=str(e))
    
    def _initialize_legacy_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mappings for legacy command compatibility."""
        return {
            # Legacy compression command mappings
            "compress_context": {
                "enhanced_command": "/hive:compress --strategy=context",
                "notes": ["Use unified compression system", "Strategy auto-detection available"],
                "breaking_changes": ["Return format updated", "New parameters available"],
                "guide": "Replace with /hive:compress for automatic strategy selection"
            },
            "compress_memory": {
                "enhanced_command": "/hive:compress --strategy=memory", 
                "notes": ["Enhanced memory compression", "Mobile optimization support"],
                "breaking_changes": ["Enhanced metadata in response"],
                "guide": "Use /hive:compress --strategy=memory or let system auto-detect"
            },
            "compress_conversation": {
                "enhanced_command": "/hive:compress --strategy=conversation",
                "notes": ["Improved conversation analysis", "Better pattern recognition"],
                "breaking_changes": ["Response structure enhanced"],
                "guide": "Use /hive:compress --strategy=conversation"
            }
        }
    
    # Migration helper methods
    async def _migrate_context_compression(self, *args, **kwargs) -> Dict[str, Any]:
        """Migrate legacy context compression calls."""
        return await self.compress_content(strategy="context", *args, **kwargs)
    
    async def _migrate_memory_compression(self, *args, **kwargs) -> Dict[str, Any]:
        """Migrate legacy memory compression calls.""" 
        return await self.compress_content(strategy="memory", *args, **kwargs)
    
    async def _migrate_conversation_compression(self, *args, **kwargs) -> Dict[str, Any]:
        """Migrate legacy conversation compression calls."""
        return await self.compress_content(strategy="conversation", *args, **kwargs)
    
    # Error recovery strategy methods
    async def _retry_with_different_params(self, command: str, error_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retry command with modified parameters."""
        # This would implement intelligent parameter modification
        raise NotImplementedError("Recovery strategy not implemented")
    
    async def _suggest_alternative_command(self, command: str, error_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Suggest alternative command."""
        # This would use command discovery to suggest alternatives
        raise NotImplementedError("Recovery strategy not implemented")
    
    async def _check_prerequisites_and_fix(self, command: str, error_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check and fix command prerequisites."""
        # This would check system state and fix issues
        raise NotImplementedError("Recovery strategy not implemented")


# Global instance
_ecosystem_integration: Optional[CommandEcosystemIntegration] = None


async def get_ecosystem_integration() -> CommandEcosystemIntegration:
    """Get or create the global command ecosystem integration instance."""
    global _ecosystem_integration
    if _ecosystem_integration is None:
        _ecosystem_integration = CommandEcosystemIntegration()
        await _ecosystem_integration.initialize()
    return _ecosystem_integration


# Backward compatibility functions
async def enhanced_execute_hive_command(
    command: str, 
    context: Dict[str, Any] = None,
    mobile_optimized: bool = False,
    use_quality_gates: bool = True
) -> Dict[str, Any]:
    """Enhanced version of execute_hive_command with full ecosystem integration."""
    ecosystem = await get_ecosystem_integration()
    return await ecosystem.execute_enhanced_command(
        command=command,
        context=context,
        mobile_optimized=mobile_optimized,
        use_quality_gates=use_quality_gates
    )


# Alias for backward compatibility
async def execute_enhanced_command(command: str, **kwargs) -> Dict[str, Any]:
    """Alias for enhanced command execution."""
    return await enhanced_execute_hive_command(command, **kwargs)


# Legacy compression compatibility
async def compress_context_enhanced(content: str, **kwargs) -> Dict[str, Any]:
    """Enhanced context compression with backward compatibility."""
    ecosystem = await get_ecosystem_integration()
    return await ecosystem.compress_content(content, strategy="context", **kwargs)


async def compress_memory_enhanced(content: str, **kwargs) -> Dict[str, Any]:
    """Enhanced memory compression with backward compatibility."""
    ecosystem = await get_ecosystem_integration()
    return await ecosystem.compress_content(content, strategy="memory", **kwargs)


async def compress_conversation_enhanced(content: str, **kwargs) -> Dict[str, Any]:
    """Enhanced conversation compression with backward compatibility."""
    ecosystem = await get_ecosystem_integration()
    return await ecosystem.compress_content(content, strategy="conversation", **kwargs)


# Command discovery compatibility
async def discover_commands_enhanced(user_intent: str, **kwargs) -> List[Dict[str, Any]]:
    """Enhanced command discovery with backward compatibility."""
    ecosystem = await get_ecosystem_integration()
    return await ecosystem.discover_commands(user_intent, **kwargs)


# System status with ecosystem info
async def get_enhanced_system_status(include_detailed_metrics: bool = False) -> Dict[str, Any]:
    """Get enhanced system status including ecosystem information."""
    ecosystem = await get_ecosystem_integration()
    return await ecosystem.get_system_status(include_detailed_metrics)