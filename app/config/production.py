"""
Production Configuration System for Multi-CLI Agent Coordination

This module provides comprehensive production configuration management
for the Multi-CLI Agent Coordination System, including Redis/WebSocket
settings, security configurations, and environment-based overrides.

IMPLEMENTATION STATUS: PRODUCTION READY
- Environment-based configuration management
- Redis and WebSocket connection settings
- Security and authentication configurations
- Agent-specific settings and limits
- Performance tuning and monitoring settings
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import ssl

# ================================================================================
# Core Configuration Models
# ================================================================================

@dataclass
class RedisConfiguration:
    """Redis connection and behavior configuration."""
    # Connection settings - Using non-standard port to avoid conflicts
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6380"))  # Non-standard port (default 6379 + 1)
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # SSL/TLS settings
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    ssl_cert_reqs: Optional[str] = os.getenv("REDIS_SSL_CERT_REQS")
    ssl_ca_certs: Optional[str] = os.getenv("REDIS_SSL_CA_CERTS")
    ssl_certfile: Optional[str] = os.getenv("REDIS_SSL_CERTFILE")
    ssl_keyfile: Optional[str] = os.getenv("REDIS_SSL_KEYFILE")
    
    # Performance settings
    connection_pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "20"))
    retry_on_timeout: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    socket_keepalive: bool = os.getenv("REDIS_KEEPALIVE", "true").lower() == "true"
    
    # Message persistence settings
    message_ttl_hours: int = int(os.getenv("REDIS_MESSAGE_TTL_HOURS", "24"))
    max_message_size: int = int(os.getenv("REDIS_MAX_MESSAGE_SIZE", "1048576"))  # 1MB
    
    # Channel configuration
    channel_prefix: str = os.getenv("REDIS_CHANNEL_PREFIX", "cli_agents")
    pattern_subscriptions: bool = os.getenv("REDIS_PATTERN_SUBSCRIPTIONS", "true").lower() == "true"

@dataclass
class WebSocketConfiguration:
    """WebSocket server and client configuration."""
    # Server settings - Using non-standard port to avoid conflicts
    host: str = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
    port: int = int(os.getenv("WEBSOCKET_PORT", "8766"))  # Non-standard port (8765 + 1)
    
    # SSL/TLS settings
    ssl_enabled: bool = os.getenv("WEBSOCKET_SSL_ENABLED", "false").lower() == "true"
    ssl_certfile: Optional[str] = os.getenv("WEBSOCKET_SSL_CERTFILE")
    ssl_keyfile: Optional[str] = os.getenv("WEBSOCKET_SSL_KEYFILE")
    ssl_ca_certs: Optional[str] = os.getenv("WEBSOCKET_SSL_CA_CERTS")
    
    # Performance settings
    ping_interval: float = float(os.getenv("WEBSOCKET_PING_INTERVAL", "20.0"))
    ping_timeout: float = float(os.getenv("WEBSOCKET_PING_TIMEOUT", "20.0"))
    close_timeout: float = float(os.getenv("WEBSOCKET_CLOSE_TIMEOUT", "10.0"))
    max_size: int = int(os.getenv("WEBSOCKET_MAX_SIZE", "1000000"))  # 1MB
    max_queue: int = int(os.getenv("WEBSOCKET_MAX_QUEUE", "64"))
    compression: Optional[str] = os.getenv("WEBSOCKET_COMPRESSION", "deflate")
    
    # Connection management
    max_connections: int = int(os.getenv("WEBSOCKET_MAX_CONNECTIONS", "1000"))
    connection_timeout: float = float(os.getenv("WEBSOCKET_CONNECTION_TIMEOUT", "30.0"))
    heartbeat_interval: float = float(os.getenv("WEBSOCKET_HEARTBEAT_INTERVAL", "30.0"))

@dataclass
class AgentConfiguration:
    """Individual CLI agent configuration settings."""
    # Basic agent settings
    agent_type: str
    cli_path: str
    working_directory: Optional[str] = None
    
    # Resource limits
    max_concurrent_tasks: int = 3
    default_timeout: float = 300.0
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 50.0
    
    # Performance settings
    response_timeout_ms: int = 5000
    health_check_interval: int = 60
    auto_restart: bool = True
    
    # Security settings
    allowed_commands: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    sandbox_enabled: bool = True
    worktree_isolation: bool = True
    
    # Communication settings
    message_queue_size: int = 1000
    batch_size: int = 10
    priority_queue_enabled: bool = True

@dataclass
class SecurityConfiguration:
    """Security and authentication configuration."""
    # Authentication
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    api_key_required: bool = os.getenv("API_KEY_REQUIRED", "true").lower() == "true"
    
    # Rate limiting
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    
    # Security validation
    command_validation: bool = os.getenv("COMMAND_VALIDATION", "true").lower() == "true"
    path_traversal_protection: bool = os.getenv("PATH_TRAVERSAL_PROTECTION", "true").lower() == "true"
    dangerous_command_detection: bool = os.getenv("DANGEROUS_COMMAND_DETECTION", "true").lower() == "true"
    
    # Audit logging
    audit_logging_enabled: bool = os.getenv("AUDIT_LOGGING", "true").lower() == "true"
    audit_log_level: str = os.getenv("AUDIT_LOG_LEVEL", "INFO")
    sensitive_data_masking: bool = os.getenv("SENSITIVE_DATA_MASKING", "true").lower() == "true"
    
    # Network security
    allowed_hosts: List[str] = field(default_factory=lambda: os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(","))
    cors_enabled: bool = os.getenv("CORS_ENABLED", "false").lower() == "true"
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [])

@dataclass
class MonitoringConfiguration:
    """Monitoring and observability configuration."""
    # Prometheus metrics - Using non-standard port to avoid conflicts
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "9091"))  # Non-standard port (9090 + 1)
    metrics_path: str = os.getenv("METRICS_PATH", "/metrics")
    
    # Health checks
    health_check_enabled: bool = os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true"
    health_check_path: str = os.getenv("HEALTH_CHECK_PATH", "/health")
    health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    log_file: Optional[str] = os.getenv("LOG_FILE")
    
    # Performance monitoring
    performance_monitoring: bool = os.getenv("PERFORMANCE_MONITORING", "true").lower() == "true"
    response_time_tracking: bool = os.getenv("RESPONSE_TIME_TRACKING", "true").lower() == "true"
    resource_usage_tracking: bool = os.getenv("RESOURCE_USAGE_TRACKING", "true").lower() == "true"
    
    # Alerting
    alerting_enabled: bool = os.getenv("ALERTING_ENABLED", "false").lower() == "true"
    alert_webhook_url: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")
    alert_email: Optional[str] = os.getenv("ALERT_EMAIL")

@dataclass
class PerformanceConfiguration:
    """Performance tuning and optimization settings."""
    # Concurrency limits
    max_concurrent_agents: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "50"))
    max_concurrent_tasks_per_agent: int = int(os.getenv("MAX_CONCURRENT_TASKS_PER_AGENT", "3"))
    task_queue_size: int = int(os.getenv("TASK_QUEUE_SIZE", "10000"))
    
    # Resource management
    memory_limit_mb: int = int(os.getenv("SYSTEM_MEMORY_LIMIT_MB", "4096"))
    cpu_limit_percent: float = float(os.getenv("SYSTEM_CPU_LIMIT_PERCENT", "80.0"))
    disk_space_limit_gb: int = int(os.getenv("DISK_SPACE_LIMIT_GB", "50"))
    
    # Caching
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    
    # Connection pooling
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "20"))
    redis_pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "20"))
    websocket_pool_size: int = int(os.getenv("WEBSOCKET_POOL_SIZE", "100"))
    
    # Timeouts and retries
    default_task_timeout: float = float(os.getenv("DEFAULT_TASK_TIMEOUT", "300.0"))
    max_retry_attempts: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    retry_backoff_base: float = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))

# ================================================================================
# Production Configuration Class
# ================================================================================

class ProductionConfiguration:
    """
    Comprehensive production configuration management for Multi-CLI Agent Coordination System.
    
    Features:
    - Environment-based configuration loading
    - Runtime configuration updates
    - Configuration validation and defaults
    - Agent-specific configuration management
    - Security and performance optimization settings
    """
    
    def __init__(self, environment: str = None):
        """Initialize production configuration."""
        self.environment = environment or os.getenv("ENVIRONMENT", "production")
        
        # Load core configurations
        self.redis = RedisConfiguration()
        self.websocket = WebSocketConfiguration()
        self.security = SecurityConfiguration()
        self.monitoring = MonitoringConfiguration()
        self.performance = PerformanceConfiguration()
        
        # Agent configurations
        self.agents = self._load_agent_configurations()
        
        # Runtime settings
        self.runtime = {
            "startup_time": None,
            "config_version": "1.0.0",
            "deployment_id": os.getenv("DEPLOYMENT_ID", "default"),
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
            "maintenance_mode": False
        }
        
        self._validate_configuration()
    
    def _load_agent_configurations(self) -> Dict[str, AgentConfiguration]:
        """Load configurations for different CLI agents."""
        agents = {}
        
        # Claude Code Agent
        agents["claude_code"] = AgentConfiguration(
            agent_type="claude_code",
            cli_path=os.getenv("CLAUDE_CLI_PATH", "claude"),
            working_directory=os.getenv("CLAUDE_WORKING_DIR"),
            max_concurrent_tasks=int(os.getenv("CLAUDE_MAX_CONCURRENT_TASKS", "5")),
            default_timeout=float(os.getenv("CLAUDE_TIMEOUT", "600.0")),
            memory_limit_mb=int(os.getenv("CLAUDE_MEMORY_LIMIT", "1024")),
            allowed_commands=os.getenv("CLAUDE_ALLOWED_COMMANDS", "").split(",") if os.getenv("CLAUDE_ALLOWED_COMMANDS") else [],
            sandbox_enabled=os.getenv("CLAUDE_SANDBOX", "true").lower() == "true",
            worktree_isolation=os.getenv("CLAUDE_WORKTREE_ISOLATION", "true").lower() == "true"
        )
        
        # Cursor Agent
        agents["cursor"] = AgentConfiguration(
            agent_type="cursor",
            cli_path=os.getenv("CURSOR_CLI_PATH", "cursor"),
            working_directory=os.getenv("CURSOR_WORKING_DIR"),
            max_concurrent_tasks=int(os.getenv("CURSOR_MAX_CONCURRENT_TASKS", "3")),
            default_timeout=float(os.getenv("CURSOR_TIMEOUT", "300.0")),
            memory_limit_mb=int(os.getenv("CURSOR_MEMORY_LIMIT", "512")),
            allowed_commands=os.getenv("CURSOR_ALLOWED_COMMANDS", "").split(",") if os.getenv("CURSOR_ALLOWED_COMMANDS") else [],
            sandbox_enabled=os.getenv("CURSOR_SANDBOX", "true").lower() == "true"
        )
        
        # GitHub Copilot Agent
        agents["github_copilot"] = AgentConfiguration(
            agent_type="github_copilot",
            cli_path=os.getenv("GH_CLI_PATH", "gh"),
            working_directory=os.getenv("GH_WORKING_DIR"),
            max_concurrent_tasks=int(os.getenv("GH_MAX_CONCURRENT_TASKS", "2")),
            default_timeout=float(os.getenv("GH_TIMEOUT", "300.0")),
            memory_limit_mb=int(os.getenv("GH_MEMORY_LIMIT", "256")),
            allowed_commands=["gh copilot suggest", "gh copilot explain"],
            sandbox_enabled=os.getenv("GH_SANDBOX", "true").lower() == "true"
        )
        
        # Gemini CLI Agent
        agents["gemini_cli"] = AgentConfiguration(
            agent_type="gemini_cli",
            cli_path=os.getenv("GEMINI_CLI_PATH", "gemini"),
            working_directory=os.getenv("GEMINI_WORKING_DIR"),
            max_concurrent_tasks=int(os.getenv("GEMINI_MAX_CONCURRENT_TASKS", "2")),
            default_timeout=float(os.getenv("GEMINI_TIMEOUT", "300.0")),
            memory_limit_mb=int(os.getenv("GEMINI_MEMORY_LIMIT", "256")),
            allowed_commands=os.getenv("GEMINI_ALLOWED_COMMANDS", "").split(",") if os.getenv("GEMINI_ALLOWED_COMMANDS") else [],
            sandbox_enabled=os.getenv("GEMINI_SANDBOX", "true").lower() == "true"
        )
        
        return agents
    
    def _validate_configuration(self):
        """Validate configuration settings and apply security defaults."""
        # Validate Redis configuration
        if self.redis.connection_pool_size < 1:
            self.redis.connection_pool_size = 10
        if self.redis.connection_pool_size > 100:
            self.redis.connection_pool_size = 100
        
        # Validate WebSocket configuration
        if self.websocket.max_connections < 1:
            self.websocket.max_connections = 100
        if self.websocket.max_connections > 10000:
            self.websocket.max_connections = 10000
        
        # Validate security configuration
        if len(self.security.jwt_secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long for production")
        
        # Validate performance limits
        if self.performance.max_concurrent_agents > 200:
            self.performance.max_concurrent_agents = 200
        
        # Validate agent configurations
        for agent_name, agent_config in self.agents.items():
            if agent_config.max_concurrent_tasks > 10:
                agent_config.max_concurrent_tasks = 10
            if agent_config.memory_limit_mb > 2048:
                agent_config.memory_limit_mb = 2048
    
    def get_ssl_context(self, component: str = "websocket") -> Optional[ssl.SSLContext]:
        """Create SSL context for secure connections."""
        try:
            if component == "websocket" and self.websocket.ssl_enabled:
                if not (self.websocket.ssl_certfile and self.websocket.ssl_keyfile):
                    return None
                
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(
                    self.websocket.ssl_certfile,
                    self.websocket.ssl_keyfile
                )
                
                if self.websocket.ssl_ca_certs:
                    context.load_verify_locations(self.websocket.ssl_ca_certs)
                    context.verify_mode = ssl.CERT_REQUIRED
                
                return context
                
            elif component == "redis" and self.redis.ssl:
                context = ssl.create_default_context()
                
                if self.redis.ssl_ca_certs:
                    context.load_verify_locations(self.redis.ssl_ca_certs)
                
                if self.redis.ssl_certfile and self.redis.ssl_keyfile:
                    context.load_cert_chain(
                        self.redis.ssl_certfile,
                        self.redis.ssl_keyfile
                    )
                
                return context
                
        except Exception as e:
            import logging
            logging.error(f"Failed to create SSL context for {component}: {e}")
            return None
        
        return None
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        protocol = "rediss" if self.redis.ssl else "redis"
        
        url = f"{protocol}://"
        
        if self.redis.password:
            url += f":{self.redis.password}@"
        
        url += f"{self.redis.host}:{self.redis.port}/{self.redis.db}"
        
        return url
    
    def get_websocket_url(self) -> str:
        """Get WebSocket server URL."""
        protocol = "wss" if self.websocket.ssl_enabled else "ws"
        return f"{protocol}://{self.websocket.host}:{self.websocket.port}"
    
    def get_agent_config(self, agent_type: str) -> Optional[AgentConfiguration]:
        """Get configuration for specific agent type."""
        return self.agents.get(agent_type)
    
    def update_runtime_setting(self, key: str, value: Any):
        """Update runtime configuration setting."""
        if key in self.runtime:
            self.runtime[key] = value
            
            # Special handling for maintenance mode
            if key == "maintenance_mode":
                import logging
                if value:
                    logging.warning("System entering maintenance mode")
                else:
                    logging.info("System exiting maintenance mode")
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.runtime.get("debug_mode", False)
    
    def is_maintenance_mode(self) -> bool:
        """Check if maintenance mode is enabled."""
        return self.runtime.get("maintenance_mode", False)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration (excluding sensitive data)."""
        return {
            "environment": self.environment,
            "config_version": self.runtime["config_version"],
            "deployment_id": self.runtime["deployment_id"],
            "debug_mode": self.is_debug_mode(),
            "maintenance_mode": self.is_maintenance_mode(),
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "ssl_enabled": self.redis.ssl,
                "pool_size": self.redis.connection_pool_size
            },
            "websocket": {
                "host": self.websocket.host,
                "port": self.websocket.port,
                "ssl_enabled": self.websocket.ssl_enabled,
                "max_connections": self.websocket.max_connections
            },
            "security": {
                "api_key_required": self.security.api_key_required,
                "rate_limit_enabled": self.security.rate_limit_enabled,
                "command_validation": self.security.command_validation,
                "audit_logging": self.security.audit_logging_enabled
            },
            "performance": {
                "max_concurrent_agents": self.performance.max_concurrent_agents,
                "cache_enabled": self.performance.cache_enabled,
                "memory_limit_mb": self.performance.memory_limit_mb
            },
            "agents": {
                name: {
                    "cli_path": config.cli_path,
                    "max_concurrent_tasks": config.max_concurrent_tasks,
                    "sandbox_enabled": config.sandbox_enabled,
                    "worktree_isolation": config.worktree_isolation
                }
                for name, config in self.agents.items()
            },
            "monitoring": {
                "metrics_enabled": self.monitoring.metrics_enabled,
                "health_check_enabled": self.monitoring.health_check_enabled,
                "log_level": self.monitoring.log_level,
                "performance_monitoring": self.monitoring.performance_monitoring
            }
        }

# ================================================================================
# Configuration Factory and Environment Management
# ================================================================================

def create_production_config(
    environment: str = None,
    config_overrides: Dict[str, Any] = None
) -> ProductionConfiguration:
    """
    Factory function to create production configuration.
    
    Args:
        environment: Target environment (production, staging, development)
        config_overrides: Dictionary of configuration overrides
        
    Returns:
        ProductionConfiguration: Configured instance
    """
    config = ProductionConfiguration(environment)
    
    # Apply any configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config

def load_environment_variables() -> Dict[str, str]:
    """Load and validate required environment variables."""
    required_vars = [
        "REDIS_HOST",
        "WEBSOCKET_HOST",
        "JWT_SECRET_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {key: os.getenv(key, "") for key in required_vars}

def validate_production_readiness(config: ProductionConfiguration) -> List[str]:
    """
    Validate production readiness and return list of warnings/issues.
    
    Returns:
        List[str]: List of validation warnings or issues
    """
    issues = []
    
    # Security validations
    if config.security.jwt_secret_key == "your-secret-key-change-in-production":
        issues.append("CRITICAL: Default JWT secret key detected - change immediately")
    
    if not config.security.api_key_required:
        issues.append("WARNING: API key authentication disabled")
    
    if not config.security.rate_limit_enabled:
        issues.append("WARNING: Rate limiting disabled")
    
    # SSL/TLS validations
    if not config.websocket.ssl_enabled:
        issues.append("WARNING: WebSocket SSL/TLS disabled")
    
    if not config.redis.ssl:
        issues.append("WARNING: Redis SSL/TLS disabled")
    
    # Performance validations
    if config.performance.max_concurrent_agents > 100:
        issues.append("WARNING: Very high concurrent agent limit may impact performance")
    
    # Monitoring validations
    if not config.monitoring.metrics_enabled:
        issues.append("WARNING: Metrics collection disabled")
    
    if not config.monitoring.health_check_enabled:
        issues.append("WARNING: Health checks disabled")
    
    if config.monitoring.log_level == "DEBUG" and not config.is_debug_mode():
        issues.append("INFO: Debug logging enabled in production")
    
    return issues

# ================================================================================
# Global Configuration Instance
# ================================================================================

# Global configuration instance (initialized on first import)
_global_config: Optional[ProductionConfiguration] = None

def get_config() -> ProductionConfiguration:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = create_production_config()
    return _global_config

def reload_config(environment: str = None) -> ProductionConfiguration:
    """Reload the global configuration."""
    global _global_config
    _global_config = create_production_config(environment)
    return _global_config