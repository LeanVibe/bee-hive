"""
Comprehensive Configuration System for Error Handling - LeanVibe Agent Hive 2.0 - VS 3.3

Centralized configuration management for all error handling components:
- Environment-based configuration with validation
- Dynamic configuration updates with hot-reload
- Performance target definitions and monitoring
- Component-specific settings with inheritance
- Integration with existing config system
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Type, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
except ImportError:
    from pydantic import BaseSettings, Field, validator

logger = structlog.get_logger()


class ErrorHandlingEnvironment(Enum):
    """Environment types for error handling configuration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationSource(Enum):
    """Sources for configuration values."""
    ENVIRONMENT = "environment"
    CONFIG_FILE = "config_file"
    DATABASE = "database"
    DEFAULT = "default"


@dataclass
class PerformanceTargets:
    """Performance targets for error handling components."""
    
    # General targets
    max_processing_time_ms: float = 5.0
    availability_target: float = 0.9995  # 99.95%
    recovery_time_target_ms: float = 30000.0  # 30 seconds
    
    # Component-specific targets
    circuit_breaker_decision_time_ms: float = 1.0
    retry_policy_calculation_time_ms: float = 0.5
    graceful_degradation_time_ms: float = 2.0
    middleware_overhead_ms: float = 5.0
    
    # Workflow targets
    workflow_recovery_time_ms: float = 30000.0
    task_retry_timeout_ms: float = 300000.0  # 5 minutes
    batch_execution_timeout_ms: float = 1800000.0  # 30 minutes


@dataclass
class CircuitBreakerSettings:
    """Configuration settings for circuit breakers."""
    
    enabled: bool = True
    failure_threshold: int = 10
    success_threshold: int = 5
    timeout_seconds: int = 60
    monitoring_window_seconds: int = 300
    min_requests_threshold: int = 5
    failure_rate_threshold: float = 0.5
    half_open_max_requests: int = 3
    
    # Performance settings
    max_processing_time_ms: float = 1.0
    enable_adaptive_timeout: bool = True
    
    # Component-specific settings
    database_failure_threshold: int = 5
    agent_failure_threshold: int = 10
    semantic_memory_failure_threshold: int = 8
    workflow_failure_threshold: int = 15


@dataclass
class RetryPolicySettings:
    """Configuration settings for retry policies."""
    
    enabled: bool = True
    default_strategy: str = "exponential_backoff"
    max_attempts: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 30000
    max_duration_ms: int = 300000  # 5 minutes
    backoff_multiplier: float = 2.0
    jitter_type: str = "equal"
    jitter_factor: float = 0.1
    
    # Strategy-specific settings
    adaptive_success_threshold: float = 0.8
    adaptive_failure_threshold: float = 0.3
    adaptive_adjustment_factor: float = 1.5
    fibonacci_max_sequence: int = 20
    
    # Component-specific settings
    task_max_retries: int = 3
    batch_max_retries: int = 2
    workflow_max_retries: int = 1
    database_max_retries: int = 5
    network_max_retries: int = 4


@dataclass
class GracefulDegradationSettings:
    """Configuration settings for graceful degradation."""
    
    enabled: bool = True
    auto_recovery_enabled: bool = True
    recovery_check_interval_seconds: int = 30
    recovery_success_threshold: int = 3
    max_processing_time_ms: float = 2.0
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    
    # Service-specific settings
    service_timeouts: Dict[str, int] = field(default_factory=lambda: {
        "semantic_memory": 5000,
        "workflow_engine": 10000,
        "agent_communication": 3000,
        "database": 2000,
        "redis": 1000
    })
    
    # Degradation levels
    minimal_degradation_threshold: float = 0.1  # 10% error rate
    partial_degradation_threshold: float = 0.3  # 30% error rate
    full_degradation_threshold: float = 0.7     # 70% error rate


@dataclass
class ObservabilitySettings:
    """Configuration settings for error handling observability."""
    
    enabled: bool = True
    detailed_logging: bool = True
    emit_events: bool = True
    emit_metrics: bool = True
    
    # Event settings
    event_sampling_rate: float = 1.0
    critical_event_always_emit: bool = True
    max_event_payload_size: int = 50000
    
    # Logging settings
    log_level: str = "INFO"
    log_performance_metrics: bool = True
    log_recovery_attempts: bool = True
    log_circuit_breaker_changes: bool = True
    
    # Metrics settings
    metrics_collection_interval_seconds: int = 60
    metrics_retention_hours: int = 24
    enable_prometheus_metrics: bool = True


@dataclass
class WorkflowErrorHandlingSettings:
    """Configuration settings for workflow error handling."""
    
    enabled: bool = True
    max_task_retries: int = 3
    max_batch_retries: int = 2
    max_workflow_retries: int = 1
    
    # Timeout configurations
    task_timeout_ms: int = 300000      # 5 minutes
    batch_timeout_ms: int = 1800000    # 30 minutes
    workflow_timeout_ms: int = 3600000 # 1 hour
    
    # Recovery settings
    enable_graceful_degradation: bool = True
    enable_checkpoint_recovery: bool = True
    recovery_timeout_ms: int = 30000
    
    # Error classification settings
    classify_errors_automatically: bool = True
    enable_error_pattern_learning: bool = True
    error_frequency_threshold: int = 5
    error_frequency_window_minutes: int = 10


class ErrorHandlingConfiguration(BaseSettings):
    """
    Comprehensive error handling configuration with environment support.
    
    Uses Pydantic BaseSettings for automatic environment variable loading
    and validation with type checking.
    """
    
    # Environment settings
    environment: ErrorHandlingEnvironment = Field(
        default=ErrorHandlingEnvironment.DEVELOPMENT,
        env="ERROR_HANDLING_ENVIRONMENT"
    )
    
    debug_mode: bool = Field(default=False, env="ERROR_HANDLING_DEBUG")
    
    # Global settings
    enabled: bool = Field(default=True, env="ERROR_HANDLING_ENABLED")
    
    # Performance targets
    performance_targets: PerformanceTargets = Field(default_factory=PerformanceTargets)
    
    # Component configurations
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    retry_policy: RetryPolicySettings = Field(default_factory=RetryPolicySettings)
    graceful_degradation: GracefulDegradationSettings = Field(default_factory=GracefulDegradationSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    workflow_error_handling: WorkflowErrorHandlingSettings = Field(default_factory=WorkflowErrorHandlingSettings)
    
    # Middleware settings
    middleware_enabled: bool = Field(default=True, env="ERROR_HANDLING_MIDDLEWARE_ENABLED")
    middleware_order_priority: int = Field(default=100, env="ERROR_HANDLING_MIDDLEWARE_PRIORITY")
    
    # Integration settings
    integrate_with_observability: bool = Field(default=True, env="ERROR_HANDLING_INTEGRATE_OBSERVABILITY")
    integrate_with_metrics: bool = Field(default=True, env="ERROR_HANDLING_INTEGRATE_METRICS")
    
    # Configuration management
    config_file_path: Optional[str] = Field(default=None, env="ERROR_HANDLING_CONFIG_FILE")
    enable_hot_reload: bool = Field(default=False, env="ERROR_HANDLING_HOT_RELOAD")
    hot_reload_interval_seconds: int = Field(default=60, env="ERROR_HANDLING_HOT_RELOAD_INTERVAL")
    
    class Config:
        env_prefix = "ERROR_HANDLING_"
        case_sensitive = False
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            ErrorHandlingEnvironment: lambda v: v.value
        }
    
    @validator('performance_targets', pre=True)
    def validate_performance_targets(cls, v):
        """Validate performance targets based on environment."""
        if isinstance(v, dict):
            return PerformanceTargets(**v)
        return v
    
    @validator('circuit_breaker', pre=True)
    def validate_circuit_breaker(cls, v):
        """Validate circuit breaker settings."""
        if isinstance(v, dict):
            return CircuitBreakerSettings(**v)
        return v
    
    @validator('retry_policy', pre=True)
    def validate_retry_policy(cls, v):
        """Validate retry policy settings."""
        if isinstance(v, dict):
            return RetryPolicySettings(**v)
        return v
    
    @validator('graceful_degradation', pre=True)
    def validate_graceful_degradation(cls, v):
        """Validate graceful degradation settings."""
        if isinstance(v, dict):
            return GracefulDegradationSettings(**v)
        return v
    
    @validator('observability', pre=True)
    def validate_observability(cls, v):
        """Validate observability settings."""
        if isinstance(v, dict):
            return ObservabilitySettings(**v)
        return v
    
    @validator('workflow_error_handling', pre=True)
    def validate_workflow_error_handling(cls, v):
        """Validate workflow error handling settings."""
        if isinstance(v, dict):
            return WorkflowErrorHandlingSettings(**v)
        return v
    
    def apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides."""
        
        if self.environment == ErrorHandlingEnvironment.PRODUCTION:
            # Production optimizations
            self.observability.detailed_logging = False
            self.debug_mode = False
            self.performance_targets.max_processing_time_ms = 3.0
            self.circuit_breaker.failure_threshold = 15
            self.retry_policy.max_attempts = 5
            
        elif self.environment == ErrorHandlingEnvironment.TESTING:
            # Testing configurations
            self.circuit_breaker.timeout_seconds = 10
            self.retry_policy.base_delay_ms = 10
            self.graceful_degradation.cache_ttl_seconds = 60
            self.workflow_error_handling.recovery_timeout_ms = 5000
            
        elif self.environment == ErrorHandlingEnvironment.DEVELOPMENT:
            # Development configurations
            self.debug_mode = True
            self.observability.detailed_logging = True
            self.enable_hot_reload = True
            
        logger.info(
            "📝 Applied environment-specific configuration overrides",
            environment=self.environment.value,
            debug_mode=self.debug_mode,
            detailed_logging=self.observability.detailed_logging
        )
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component."""
        
        component_configs = {
            "circuit_breaker": asdict(self.circuit_breaker),
            "retry_policy": asdict(self.retry_policy),
            "graceful_degradation": asdict(self.graceful_degradation),
            "observability": asdict(self.observability),
            "workflow_error_handling": asdict(self.workflow_error_handling),
            "performance_targets": asdict(self.performance_targets)
        }
        
        if component_name not in component_configs:
            raise ValueError(f"Unknown component: {component_name}")
        
        return component_configs[component_name]
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues."""
        
        issues = {
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Validate performance targets
        if self.performance_targets.max_processing_time_ms > 10.0:
            issues["warnings"].append("Processing time target >10ms may impact user experience")
        
        if self.performance_targets.availability_target < 0.99:
            issues["errors"].append("Availability target <99% is too low for production")
        
        # Validate circuit breaker settings
        if self.circuit_breaker.failure_threshold < 3:
            issues["warnings"].append("Circuit breaker failure threshold <3 may be too sensitive")
        
        if self.circuit_breaker.timeout_seconds > 300:
            issues["warnings"].append("Circuit breaker timeout >5 minutes may be too long")
        
        # Validate retry settings
        if self.retry_policy.max_attempts > 10:
            issues["warnings"].append("Max retry attempts >10 may cause excessive delays")
        
        if self.retry_policy.max_delay_ms > 60000:
            issues["warnings"].append("Max retry delay >60s may impact user experience")
        
        # Validate workflow settings
        if self.workflow_error_handling.task_timeout_ms > 1800000:  # 30 minutes
            issues["warnings"].append("Task timeout >30 minutes may be excessive")
        
        # Environment-specific validations
        if self.environment == ErrorHandlingEnvironment.PRODUCTION:
            if self.debug_mode:
                issues["errors"].append("Debug mode should be disabled in production")
            
            if self.observability.detailed_logging:
                issues["recommendations"].append("Consider disabling detailed logging in production for performance")
        
        return issues
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary format."""
        return {
            "environment": self.environment.value,
            "enabled": self.enabled,
            "debug_mode": self.debug_mode,
            "performance_targets": asdict(self.performance_targets),
            "circuit_breaker": asdict(self.circuit_breaker),
            "retry_policy": asdict(self.retry_policy),
            "graceful_degradation": asdict(self.graceful_degradation),
            "observability": asdict(self.observability),
            "workflow_error_handling": asdict(self.workflow_error_handling),
            "middleware_enabled": self.middleware_enabled,
            "integration_settings": {
                "integrate_with_observability": self.integrate_with_observability,
                "integrate_with_metrics": self.integrate_with_metrics
            }
        }
    
    def export_to_json(self, indent: int = 2) -> str:
        """Export configuration to JSON format."""
        return json.dumps(self.export_to_dict(), indent=indent, default=str)
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to file."""
        with open(file_path, 'w') as f:
            f.write(self.export_to_json())
        
        logger.info(f"💾 Configuration saved to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ErrorHandlingConfiguration':
        """Load configuration from file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            logger.info(f"📂 Configuration loaded from {file_path}")
            return cls(**config_dict)
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration from {file_path}: {e}")
            raise
    
    @classmethod
    def create_default_config(cls, environment: ErrorHandlingEnvironment) -> 'ErrorHandlingConfiguration':
        """Create default configuration for specific environment."""
        
        config = cls(environment=environment)
        config.apply_environment_overrides()
        
        logger.info(f"🏗️ Created default configuration for {environment.value}")
        return config


class ConfigurationManager:
    """
    Manager for error handling configuration with hot-reload support.
    
    Features:
    - Configuration loading from multiple sources
    - Hot-reload with file watching
    - Validation and error reporting
    - Configuration versioning and rollback
    """
    
    def __init__(
        self,
        config: Optional[ErrorHandlingConfiguration] = None,
        enable_hot_reload: bool = False
    ):
        """Initialize configuration manager."""
        self.config = config or ErrorHandlingConfiguration()
        self.enable_hot_reload = enable_hot_reload
        
        # Configuration history for rollback
        self.config_history: List[Tuple[datetime, ErrorHandlingConfiguration]] = []
        self.max_history_size = 10
        
        # Hot-reload settings
        self._hot_reload_task: Optional[asyncio.Task] = None
        self._config_file_mtime: Optional[float] = None
        
        # Configuration change callbacks
        self._change_callbacks: List[Callable[[ErrorHandlingConfiguration], None]] = []
        
        logger.info(
            "⚙️ Configuration manager initialized",
            environment=self.config.environment.value,
            hot_reload=enable_hot_reload
        )
    
    def get_config(self) -> ErrorHandlingConfiguration:
        """Get current configuration."""
        return self.config
    
    def update_config(
        self,
        new_config: ErrorHandlingConfiguration,
        source: ConfigurationSource = ConfigurationSource.DEFAULT
    ) -> bool:
        """
        Update configuration with validation.
        
        Args:
            new_config: New configuration to apply
            source: Source of the configuration change
            
        Returns:
            True if update was successful
        """
        try:
            # Validate new configuration
            validation_issues = new_config.validate_configuration()
            
            if validation_issues["errors"]:
                logger.error(
                    "❌ Configuration validation failed",
                    errors=validation_issues["errors"],
                    source=source.value
                )
                return False
            
            if validation_issues["warnings"]:
                logger.warning(
                    "⚠️ Configuration validation warnings",
                    warnings=validation_issues["warnings"],
                    source=source.value
                )
            
            # Save current config to history
            self._save_to_history(self.config)
            
            # Apply new configuration
            old_config = self.config
            self.config = new_config
            
            # Notify callbacks
            for callback in self._change_callbacks:
                try:
                    callback(new_config)
                except Exception as callback_error:
                    logger.warning(f"Configuration change callback failed: {callback_error}")
            
            logger.info(
                "✅ Configuration updated successfully",
                source=source.value,
                warnings=len(validation_issues["warnings"]),
                recommendations=len(validation_issues["recommendations"])
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update configuration: {e}")
            return False
    
    def rollback_config(self, steps: int = 1) -> bool:
        """
        Rollback configuration to previous version.
        
        Args:
            steps: Number of steps to rollback
            
        Returns:
            True if rollback was successful
        """
        if len(self.config_history) < steps:
            logger.error(f"❌ Cannot rollback {steps} steps - only {len(self.config_history)} versions available")
            return False
        
        try:
            # Get previous configuration
            rollback_timestamp, rollback_config = self.config_history[-steps]
            
            # Apply rollback configuration
            self.config = rollback_config
            
            # Remove rolled back versions from history
            self.config_history = self.config_history[:-steps]
            
            logger.info(
                "🔄 Configuration rolled back successfully",
                steps=steps,
                rollback_timestamp=rollback_timestamp.isoformat()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to rollback configuration: {e}")
            return False
    
    def add_change_callback(self, callback: Callable[[ErrorHandlingConfiguration], None]) -> None:
        """Add callback for configuration changes."""
        self._change_callbacks.append(callback)
        logger.debug(f"➕ Added configuration change callback: {callback.__name__}")
    
    def remove_change_callback(self, callback: Callable[[ErrorHandlingConfiguration], None]) -> None:
        """Remove configuration change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
            logger.debug(f"➖ Removed configuration change callback: {callback.__name__}")
    
    async def start_hot_reload(self) -> None:
        """Start hot-reload monitoring if enabled."""
        if not self.enable_hot_reload or not self.config.config_file_path:
            return
        
        self._hot_reload_task = asyncio.create_task(self._hot_reload_loop())
        logger.info("🔥 Hot-reload monitoring started")
    
    async def stop_hot_reload(self) -> None:
        """Stop hot-reload monitoring."""
        if self._hot_reload_task:
            self._hot_reload_task.cancel()
            try:
                await self._hot_reload_task
            except asyncio.CancelledError:
                pass
            self._hot_reload_task = None
            
        logger.info("🛑 Hot-reload monitoring stopped")
    
    async def _hot_reload_loop(self) -> None:
        """Hot-reload monitoring loop."""
        if not self.config.config_file_path:
            return
        
        config_file = self.config.config_file_path
        
        try:
            # Get initial file modification time
            if os.path.exists(config_file):
                self._config_file_mtime = os.path.getmtime(config_file)
            
            while True:
                await asyncio.sleep(self.config.hot_reload_interval_seconds)
                
                if not os.path.exists(config_file):
                    continue
                
                current_mtime = os.path.getmtime(config_file)
                
                if self._config_file_mtime is None or current_mtime > self._config_file_mtime:
                    logger.info(f"🔄 Configuration file changed, reloading: {config_file}")
                    
                    try:
                        new_config = ErrorHandlingConfiguration.load_from_file(config_file)
                        
                        if self.update_config(new_config, ConfigurationSource.CONFIG_FILE):
                            self._config_file_mtime = current_mtime
                            logger.info("✅ Hot-reload completed successfully")
                        else:
                            logger.error("❌ Hot-reload failed - keeping current configuration")
                            
                    except Exception as e:
                        logger.error(f"❌ Hot-reload error: {e}")
                
        except asyncio.CancelledError:
            logger.info("🛑 Hot-reload loop cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ Hot-reload loop error: {e}")
    
    def _save_to_history(self, config: ErrorHandlingConfiguration) -> None:
        """Save configuration to history."""
        current_time = datetime.utcnow()
        self.config_history.append((current_time, config))
        
        # Limit history size
        if len(self.config_history) > self.max_history_size:
            self.config_history.pop(0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration manager status."""
        return {
            "current_environment": self.config.environment.value,
            "hot_reload_enabled": self.enable_hot_reload,
            "hot_reload_active": self._hot_reload_task is not None and not self._hot_reload_task.done(),
            "config_file_path": self.config.config_file_path,
            "history_size": len(self.config_history),
            "change_callbacks": len(self._change_callbacks),
            "last_update": self.config_history[-1][0].isoformat() if self.config_history else None
        }


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_error_handling_config() -> ErrorHandlingConfiguration:
    """Get current error handling configuration."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    
    return _config_manager.get_config()


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    
    return _config_manager


def initialize_error_handling_config(
    environment: Optional[ErrorHandlingEnvironment] = None,
    config_file_path: Optional[str] = None,
    enable_hot_reload: bool = False
) -> ConfigurationManager:
    """
    Initialize error handling configuration system.
    
    Args:
        environment: Target environment
        config_file_path: Path to configuration file
        enable_hot_reload: Enable hot-reload monitoring
        
    Returns:
        Configured configuration manager
    """
    global _config_manager
    
    # Load configuration
    if config_file_path and os.path.exists(config_file_path):
        config = ErrorHandlingConfiguration.load_from_file(config_file_path)
        config.config_file_path = config_file_path
    else:
        env = environment or ErrorHandlingEnvironment.DEVELOPMENT
        config = ErrorHandlingConfiguration.create_default_config(env)
        if config_file_path:
            config.config_file_path = config_file_path
    
    # Apply environment overrides
    config.apply_environment_overrides()
    
    # Create configuration manager
    _config_manager = ConfigurationManager(
        config=config,
        enable_hot_reload=enable_hot_reload
    )
    
    logger.info(
        "✅ Error handling configuration system initialized",
        environment=config.environment.value,
        config_file=config_file_path,
        hot_reload=enable_hot_reload
    )
    
    return _config_manager