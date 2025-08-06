"""
Hook Configuration System for LeanVibe Agent Hive 2.0 Observability

Centralized configuration for Claude Code integration hooks including
performance thresholds, error handling policies, and session management.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class HookPerformanceConfig:
    """Performance thresholds and monitoring configuration for hooks."""
    
    # Tool execution monitoring
    slow_tool_threshold_ms: int = 5000  # Alert for tools taking >5s
    very_slow_tool_threshold_ms: int = 15000  # Critical alert for tools taking >15s
    
    # Session lifecycle monitoring
    session_timeout_minutes: int = 120  # Alert if session inactive for >2h
    memory_usage_threshold_mb: int = 1000  # Alert if memory usage >1GB
    
    # Error rate monitoring
    error_rate_threshold_percent: float = 10.0  # Alert if error rate >10%
    consecutive_failures_threshold: int = 5  # Alert after 5 consecutive failures
    
    # Performance optimization
    batch_size: int = 100  # Batch event processing for efficiency
    buffer_size: int = 1000  # Buffer events before flushing
    flush_interval_seconds: int = 30  # Flush buffer every 30s


@dataclass
class HookSecurityConfig:
    """Security configuration for hook operations."""
    
    # Data sanitization
    max_payload_size: int = 100000  # Max event payload size (100KB)
    sanitize_sensitive_data: bool = True  # Remove sensitive data from payloads
    
    # Sensitive patterns to redact
    sensitive_patterns: List[str] = None
    
    def __post_init__(self):
        if self.sensitive_patterns is None:
            self.sensitive_patterns = [
                r"password[\"':\s]*[\"']([^\"']+)[\"']",
                r"token[\"':\s]*[\"']([^\"']+)[\"']",
                r"key[\"':\s]*[\"']([^\"']+)[\"']",
                r"secret[\"':\s]*[\"']([^\"']+)[\"']",
                r"api_key[\"':\s]*[\"']([^\"']+)[\"']",
            ]


@dataclass
class HookIntegrationConfig:
    """Configuration for integration with existing observability systems."""
    
    # Database connection
    use_database: bool = True  # Store events in database
    db_batch_size: int = 50  # Batch database inserts
    
    # Redis streams
    use_redis_streams: bool = True  # Publish events to Redis
    redis_stream_key: str = "agent_events"
    redis_max_len: int = 10000  # Max stream length
    
    # Prometheus metrics
    use_prometheus: bool = True  # Expose Prometheus metrics
    metrics_port: int = 9090
    
    # External webhooks
    webhook_urls: List[str] = None  # External systems to notify
    webhook_timeout_seconds: int = 5
    
    def __post_init__(self):
        if self.webhook_urls is None:
            self.webhook_urls = []


@dataclass
class HookSessionConfig:
    """Configuration for session lifecycle management."""
    
    # Session tracking
    auto_create_sessions: bool = True  # Auto-create session if not exists
    session_id_source: str = "env"  # env|uuid|timestamp
    
    # Sleep/wake cycle integration
    enable_sleep_wake_hooks: bool = True  # Monitor sleep/wake cycles
    consolidation_trigger_threshold: int = 1000  # Trigger consolidation after N events
    
    # Memory management
    context_threshold_percent: int = 85  # Trigger sleep at 85% context usage
    memory_cleanup_interval_minutes: int = 60  # Clean up old events every hour


class HookConfig:
    """
    Centralized configuration for all observability hooks.
    
    Provides environment-based configuration with sensible defaults
    for production, development, and testing environments.
    """
    
    def __init__(self):
        """Initialize hook configuration from environment variables."""
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Load configuration based on environment
        if self.environment == "production":
            self.performance = self._load_production_performance_config()
        elif self.environment == "testing":
            self.performance = self._load_testing_performance_config()
        else:
            self.performance = self._load_development_performance_config()
        
        # Load other configurations
        self.security = self._load_security_config()
        self.integration = self._load_integration_config()
        self.session = self._load_session_config()
        
        # Hook enablement flags
        self.enable_pre_tool_use = self._get_bool_env("ENABLE_PRE_TOOL_USE_HOOK", True)
        self.enable_post_tool_use = self._get_bool_env("ENABLE_POST_TOOL_USE_HOOK", True)
        self.enable_session_lifecycle = self._get_bool_env("ENABLE_SESSION_LIFECYCLE_HOOK", True)
        self.enable_error_hooks = self._get_bool_env("ENABLE_ERROR_HOOKS", True)
        
        # Hook script paths
        self.hooks_directory = Path(__file__).parent
        self.pre_tool_use_script = self.hooks_directory / "pre_tool_use.py"
        self.post_tool_use_script = self.hooks_directory / "post_tool_use.py"
        self.session_lifecycle_script = self.hooks_directory / "session_lifecycle.py"
        
        logger.info(
            "🔧 Hook configuration initialized",
            environment=self.environment,
            hooks_directory=str(self.hooks_directory),
            enable_pre_tool_use=self.enable_pre_tool_use,
            enable_post_tool_use=self.enable_post_tool_use,
            enable_session_lifecycle=self.enable_session_lifecycle
        )
    
    def _load_production_performance_config(self) -> HookPerformanceConfig:
        """Load production-optimized performance configuration."""
        return HookPerformanceConfig(
            slow_tool_threshold_ms=3000,  # Stricter in production
            very_slow_tool_threshold_ms=10000,
            session_timeout_minutes=60,  # Shorter timeout
            memory_usage_threshold_mb=2000,  # Higher threshold for production
            error_rate_threshold_percent=5.0,  # Stricter error rate
            batch_size=200,  # Larger batches for efficiency
            buffer_size=2000,
            flush_interval_seconds=15  # More frequent flushes
        )
    
    def _load_development_performance_config(self) -> HookPerformanceConfig:
        """Load development-friendly performance configuration."""
        return HookPerformanceConfig(
            slow_tool_threshold_ms=10000,  # More lenient for debugging
            very_slow_tool_threshold_ms=30000,
            session_timeout_minutes=240,  # Longer timeout for development
            memory_usage_threshold_mb=500,
            error_rate_threshold_percent=20.0,  # More lenient error rate
            batch_size=50,  # Smaller batches for quick feedback
            buffer_size=500,
            flush_interval_seconds=60
        )
    
    def _load_testing_performance_config(self) -> HookPerformanceConfig:
        """Load testing-optimized performance configuration."""
        return HookPerformanceConfig(
            slow_tool_threshold_ms=1000,  # Fast feedback in tests
            very_slow_tool_threshold_ms=5000,
            session_timeout_minutes=15,  # Short timeout for tests
            memory_usage_threshold_mb=100,
            error_rate_threshold_percent=1.0,  # Strict for tests
            batch_size=10,  # Small batches for test predictability
            buffer_size=100,
            flush_interval_seconds=5
        )
    
    def _load_security_config(self) -> HookSecurityConfig:
        """Load security configuration from environment."""
        return HookSecurityConfig(
            max_payload_size=self._get_int_env("HOOK_MAX_PAYLOAD_SIZE", 100000),
            sanitize_sensitive_data=self._get_bool_env("HOOK_SANITIZE_SENSITIVE_DATA", True)
        )
    
    def _load_integration_config(self) -> HookIntegrationConfig:
        """Load integration configuration from environment."""
        webhook_urls_str = os.getenv("HOOK_WEBHOOK_URLS", "")
        webhook_urls = [url.strip() for url in webhook_urls_str.split(",") if url.strip()]
        
        return HookIntegrationConfig(
            use_database=self._get_bool_env("HOOK_USE_DATABASE", True),
            db_batch_size=self._get_int_env("HOOK_DB_BATCH_SIZE", 50),
            use_redis_streams=self._get_bool_env("HOOK_USE_REDIS_STREAMS", True),
            redis_stream_key=os.getenv("HOOK_REDIS_STREAM_KEY", "agent_events"),
            use_prometheus=self._get_bool_env("HOOK_USE_PROMETHEUS", True),
            webhook_urls=webhook_urls,
            webhook_timeout_seconds=self._get_int_env("HOOK_WEBHOOK_TIMEOUT", 5)
        )
    
    def _load_session_config(self) -> HookSessionConfig:
        """Load session configuration from environment."""
        return HookSessionConfig(
            auto_create_sessions=self._get_bool_env("HOOK_AUTO_CREATE_SESSIONS", True),
            session_id_source=os.getenv("HOOK_SESSION_ID_SOURCE", "env"),
            enable_sleep_wake_hooks=self._get_bool_env("HOOK_ENABLE_SLEEP_WAKE", True),
            consolidation_trigger_threshold=self._get_int_env("HOOK_CONSOLIDATION_THRESHOLD", 1000),
            context_threshold_percent=self._get_int_env("HOOK_CONTEXT_THRESHOLD", 85)
        )
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(
                f"Invalid integer value for {key}: {value}, using default {default}"
            )
            return default
    
    def get_session_id(self) -> str:
        """Get session ID based on configured source."""
        if self.session.session_id_source == "env":
            session_id = os.getenv("CLAUDE_SESSION_ID")
            if session_id:
                # Validate that it's a proper UUID format
                try:
                    import uuid
                    uuid.UUID(session_id)
                    return session_id
                except ValueError:
                    # If not a valid UUID, generate a new one
                    pass
        
        # Fallback to UUID generation
        import uuid
        return str(uuid.uuid4())
    
    def get_agent_id(self) -> str:
        """Get agent ID from environment or generate one."""
        agent_id = os.getenv("CLAUDE_AGENT_ID")
        if agent_id:
            # Validate that it's a proper UUID format
            try:
                import uuid
                uuid.UUID(agent_id)
                return agent_id
            except ValueError:
                # If not a valid UUID, generate a new one
                pass
        
        # Generate a UUID-based agent ID
        import uuid
        return str(uuid.uuid4())
    
    def should_capture_event(self, event_type: str) -> bool:
        """Check if a specific event type should be captured."""
        if event_type == "PreToolUse":
            return self.enable_pre_tool_use
        elif event_type == "PostToolUse":
            return self.enable_post_tool_use
        elif event_type in ("SessionStart", "SessionEnd", "Sleep", "Wake"):
            return self.enable_session_lifecycle
        elif event_type in ("Error", "Exception"):
            return self.enable_error_hooks
        else:
            return True  # Default to capturing unknown events
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate configuration and return validation results."""
        results = {}
        
        # Check script files exist
        results["pre_tool_use_script_exists"] = self.pre_tool_use_script.exists()
        results["post_tool_use_script_exists"] = self.post_tool_use_script.exists()
        results["session_lifecycle_script_exists"] = self.session_lifecycle_script.exists()
        
        # Check thresholds are reasonable
        results["performance_thresholds_valid"] = (
            self.performance.slow_tool_threshold_ms > 0 and
            self.performance.very_slow_tool_threshold_ms > self.performance.slow_tool_threshold_ms
        )
        
        # Check batch sizes are reasonable
        results["batch_sizes_valid"] = (
            self.performance.batch_size > 0 and
            self.performance.buffer_size > self.performance.batch_size
        )
        
        # Check security configuration
        results["security_config_valid"] = (
            self.security.max_payload_size > 0 and
            len(self.security.sensitive_patterns) > 0
        )
        
        all_valid = all(results.values())
        
        logger.info(
            "🔍 Hook configuration validation completed",
            all_valid=all_valid,
            **results
        )
        
        return results


# Global configuration instance
_config: Optional[HookConfig] = None


def get_hook_config() -> HookConfig:
    """Get the global hook configuration instance."""
    global _config
    if _config is None:
        _config = HookConfig()
    return _config


def reload_hook_config() -> HookConfig:
    """Reload the global hook configuration instance."""
    global _config
    _config = HookConfig()
    return _config