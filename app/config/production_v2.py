"""
Production Configuration System v2.0 for LeanVibe Agent Hive 2.0

This module provides comprehensive production configuration management
for the LeanVibe Agent Hive 2.0 system, including enhanced support for:
- All CLI adapters (Claude Code, Cursor, GitHub Copilot, Gemini)
- Redis and WebSocket connection configurations
- Agent orchestration settings and limits
- Security configurations and API keys management
- Performance tuning parameters
- Monitoring and logging configurations
- Health check intervals and thresholds

IMPLEMENTATION STATUS: PRODUCTION READY v2.0
- Enhanced CLI adapter configuration support
- Comprehensive secrets management
- Advanced security configurations
- Production-optimized performance settings
- Complete monitoring and observability setup
"""

import os
import json
import logging
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    from pydantic import Field, field_validator, SecretStr
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import Field, validator as field_validator, SecretStr
    from pydantic import BaseSettings

logger = logging.getLogger(__name__)

# ================================================================================
# ENHANCED ENUMS AND TYPES
# ================================================================================

class Environment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class AdapterType(str, Enum):
    """Supported CLI adapter types."""
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    GITHUB_COPILOT = "github_copilot"
    GEMINI_CLI = "gemini_cli"
    OPENCODE = "opencode"

class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SecurityLevel(str, Enum):
    """Security configuration levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"

# ================================================================================
# CLI ADAPTER CONFIGURATIONS
# ================================================================================

@dataclass
class CLIAdapterConfiguration:
    """Base configuration for CLI adapters."""
    
    # Core adapter settings
    adapter_type: AdapterType
    cli_path: str
    enabled: bool = True
    working_directory: Optional[str] = None
    
    # Resource limits
    max_concurrent_tasks: int = 3
    default_timeout_seconds: float = 300.0
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 50.0
    
    # Performance settings
    response_timeout_ms: int = 5000
    health_check_interval_seconds: int = 60
    auto_restart: bool = True
    restart_delay_seconds: float = 5.0
    max_restart_attempts: int = 3
    
    # Security settings
    allowed_commands: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    sandbox_enabled: bool = True
    worktree_isolation: bool = True
    path_traversal_protection: bool = True
    
    # Communication settings
    message_queue_size: int = 1000
    batch_size: int = 10
    priority_queue_enabled: bool = True
    retry_attempts: int = 3
    retry_backoff_base: float = 2.0
    
    # Monitoring and observability
    metrics_enabled: bool = True
    performance_tracking: bool = True
    error_reporting: bool = True
    debug_logging: bool = False
    
    # Environment variables
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate adapter configuration and return any issues."""
        issues = []
        
        if not self.cli_path:
            issues.append(f"{self.adapter_type.value}: CLI path is required")
        
        if self.max_concurrent_tasks < 1:
            issues.append(f"{self.adapter_type.value}: max_concurrent_tasks must be >= 1")
        
        if self.memory_limit_mb < 64:
            issues.append(f"{self.adapter_type.value}: memory_limit_mb must be >= 64")
        
        if self.cpu_limit_percent < 1.0 or self.cpu_limit_percent > 100.0:
            issues.append(f"{self.adapter_type.value}: cpu_limit_percent must be between 1.0 and 100.0")
        
        return issues

@dataclass
class ClaudeCodeAdapterConfig(CLIAdapterConfiguration):
    """Configuration specific to Claude Code CLI adapter."""
    
    def __post_init__(self):
        self.adapter_type = AdapterType.CLAUDE_CODE
        if not self.cli_path:
            self.cli_path = "claude"
        
        # Claude Code specific defaults
        self.max_concurrent_tasks = 5
        self.default_timeout_seconds = 600.0
        self.memory_limit_mb = 1024
        self.sandbox_enabled = True
        self.worktree_isolation = True
        
        # Claude Code specific allowed commands
        if not self.allowed_commands:
            self.allowed_commands = [
                "read", "write", "edit", "bash", "grep", "ls", "search"
            ]
        
        # Claude Code specific environment variables
        self.environment_variables.update({
            "CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY", ""),
            "CLAUDE_MODEL": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            "CLAUDE_MAX_TOKENS": os.getenv("CLAUDE_MAX_TOKENS", "8192")
        })

@dataclass
class CursorAdapterConfig(CLIAdapterConfiguration):
    """Configuration specific to Cursor CLI adapter."""
    
    def __post_init__(self):
        self.adapter_type = AdapterType.CURSOR
        if not self.cli_path:
            self.cli_path = "cursor"
        
        # Cursor specific defaults
        self.max_concurrent_tasks = 3
        self.default_timeout_seconds = 300.0
        self.memory_limit_mb = 512
        self.sandbox_enabled = True
        
        # Cursor specific environment variables
        self.environment_variables.update({
            "CURSOR_API_KEY": os.getenv("CURSOR_API_KEY", ""),
            "CURSOR_MODEL": os.getenv("CURSOR_MODEL", "gpt-4")
        })

@dataclass
class GitHubCopilotAdapterConfig(CLIAdapterConfiguration):
    """Configuration specific to GitHub Copilot CLI adapter."""
    
    def __post_init__(self):
        self.adapter_type = AdapterType.GITHUB_COPILOT
        if not self.cli_path:
            self.cli_path = "gh"
        
        # GitHub Copilot specific defaults
        self.max_concurrent_tasks = 2
        self.default_timeout_seconds = 300.0
        self.memory_limit_mb = 256
        
        # GitHub Copilot specific allowed commands
        if not self.allowed_commands:
            self.allowed_commands = [
                "gh copilot suggest",
                "gh copilot explain",
                "gh auth status",
                "gh auth login"
            ]
        
        # GitHub Copilot specific environment variables
        self.environment_variables.update({
            "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", ""),
            "GH_TOKEN": os.getenv("GH_TOKEN", "")
        })

@dataclass
class GeminiCLIAdapterConfig(CLIAdapterConfiguration):
    """Configuration specific to Gemini CLI adapter."""
    
    def __post_init__(self):
        self.adapter_type = AdapterType.GEMINI_CLI
        if not self.cli_path:
            self.cli_path = "gemini"
        
        # Gemini specific defaults
        self.max_concurrent_tasks = 2
        self.default_timeout_seconds = 300.0
        self.memory_limit_mb = 256
        self.sandbox_enabled = True
        
        # Gemini specific environment variables
        self.environment_variables.update({
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-pro"),
            "GEMINI_PROJECT_ID": os.getenv("GEMINI_PROJECT_ID", "")
        })

# ================================================================================
# ENHANCED REDIS CONFIGURATION
# ================================================================================

@dataclass
class RedisConfiguration:
    """Enhanced Redis configuration with clustering and security support."""
    
    # Basic connection settings
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6380"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # SSL/TLS configuration
    ssl_enabled: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    ssl_cert_reqs: str = os.getenv("REDIS_SSL_CERT_REQS", "required")
    ssl_ca_certs: Optional[str] = os.getenv("REDIS_SSL_CA_CERTS")
    ssl_certfile: Optional[str] = os.getenv("REDIS_SSL_CERTFILE")
    ssl_keyfile: Optional[str] = os.getenv("REDIS_SSL_KEYFILE")
    ssl_check_hostname: bool = os.getenv("REDIS_SSL_CHECK_HOSTNAME", "true").lower() == "true"
    
    # Connection pool settings
    connection_pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "20"))
    max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "200"))
    connection_timeout: float = float(os.getenv("REDIS_CONNECTION_TIMEOUT", "5.0"))
    socket_timeout: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
    socket_keepalive: bool = os.getenv("REDIS_KEEPALIVE", "true").lower() == "true"
    retry_on_timeout: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    
    # Performance settings
    decode_responses: bool = True
    encoding: str = "utf-8"
    compression_enabled: bool = os.getenv("REDIS_COMPRESSION", "true").lower() == "true"
    compression_algorithm: str = os.getenv("REDIS_COMPRESSION_ALGORITHM", "zlib")
    compression_level: int = int(os.getenv("REDIS_COMPRESSION_LEVEL", "6"))
    
    # Stream settings
    stream_max_len: int = int(os.getenv("REDIS_STREAM_MAX_LEN", "10000"))
    consumer_group_timeout: int = int(os.getenv("REDIS_CONSUMER_GROUP_TIMEOUT", "30000"))
    stream_read_timeout: float = float(os.getenv("REDIS_STREAM_READ_TIMEOUT", "1.0"))
    
    # Message persistence
    message_ttl_hours: int = int(os.getenv("REDIS_MESSAGE_TTL_HOURS", "24"))
    max_message_size: int = int(os.getenv("REDIS_MAX_MESSAGE_SIZE", "1048576"))  # 1MB
    
    # Channel configuration
    channel_prefix: str = os.getenv("REDIS_CHANNEL_PREFIX", "leanvibe_agents")
    pattern_subscriptions: bool = os.getenv("REDIS_PATTERN_SUBSCRIPTIONS", "true").lower() == "true"
    
    # Clustering support
    cluster_enabled: bool = os.getenv("REDIS_CLUSTER_ENABLED", "false").lower() == "true"
    cluster_nodes: List[str] = field(default_factory=lambda: os.getenv("REDIS_CLUSTER_NODES", "").split(",") if os.getenv("REDIS_CLUSTER_NODES") else [])
    cluster_max_redirections: int = int(os.getenv("REDIS_CLUSTER_MAX_REDIRECTIONS", "16"))
    
    # Monitoring and health checks
    health_check_interval: int = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
    ping_timeout: float = float(os.getenv("REDIS_PING_TIMEOUT", "1.0"))
    
    def get_connection_url(self) -> str:
        """Generate Redis connection URL."""
        protocol = "rediss" if self.ssl_enabled else "redis"
        auth_part = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth_part}{self.host}:{self.port}/{self.db}"
    
    def validate(self) -> List[str]:
        """Validate Redis configuration."""
        issues = []
        
        if self.connection_pool_size < 1:
            issues.append("Redis connection pool size must be >= 1")
        
        if self.port < 1 or self.port > 65535:
            issues.append("Redis port must be between 1 and 65535")
        
        if self.cluster_enabled and not self.cluster_nodes:
            issues.append("Redis cluster nodes must be specified when clustering is enabled")
        
        return issues

# ================================================================================
# ENHANCED WEBSOCKET CONFIGURATION
# ================================================================================

@dataclass
class WebSocketConfiguration:
    """Enhanced WebSocket configuration with advanced features."""
    
    # Server settings
    host: str = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
    port: int = int(os.getenv("WEBSOCKET_PORT", "8766"))
    path: str = os.getenv("WEBSOCKET_PATH", "/ws")
    
    # SSL/TLS configuration
    ssl_enabled: bool = os.getenv("WEBSOCKET_SSL_ENABLED", "false").lower() == "true"
    ssl_certfile: Optional[str] = os.getenv("WEBSOCKET_SSL_CERTFILE")
    ssl_keyfile: Optional[str] = os.getenv("WEBSOCKET_SSL_KEYFILE")
    ssl_ca_certs: Optional[str] = os.getenv("WEBSOCKET_SSL_CA_CERTS")
    ssl_check_hostname: bool = os.getenv("WEBSOCKET_SSL_CHECK_HOSTNAME", "true").lower() == "true"
    
    # Connection management
    max_connections: int = int(os.getenv("WEBSOCKET_MAX_CONNECTIONS", "1000"))
    connection_timeout: float = float(os.getenv("WEBSOCKET_CONNECTION_TIMEOUT", "30.0"))
    idle_timeout: float = float(os.getenv("WEBSOCKET_IDLE_TIMEOUT", "300.0"))
    
    # Performance settings
    ping_interval: float = float(os.getenv("WEBSOCKET_PING_INTERVAL", "20.0"))
    ping_timeout: float = float(os.getenv("WEBSOCKET_PING_TIMEOUT", "20.0"))
    close_timeout: float = float(os.getenv("WEBSOCKET_CLOSE_TIMEOUT", "10.0"))
    max_size: int = int(os.getenv("WEBSOCKET_MAX_SIZE", "1000000"))  # 1MB
    max_queue: int = int(os.getenv("WEBSOCKET_MAX_QUEUE", "64"))
    
    # Compression settings
    compression: Optional[str] = os.getenv("WEBSOCKET_COMPRESSION", "deflate")
    compression_level: int = int(os.getenv("WEBSOCKET_COMPRESSION_LEVEL", "6"))
    compression_threshold: int = int(os.getenv("WEBSOCKET_COMPRESSION_THRESHOLD", "1024"))
    
    # Authentication and security
    auth_required: bool = os.getenv("WEBSOCKET_AUTH_REQUIRED", "true").lower() == "true"
    auth_timeout: float = float(os.getenv("WEBSOCKET_AUTH_TIMEOUT", "30.0"))
    rate_limit_enabled: bool = os.getenv("WEBSOCKET_RATE_LIMIT_ENABLED", "true").lower() == "true"
    rate_limit_per_second: int = int(os.getenv("WEBSOCKET_RATE_LIMIT_PER_SECOND", "10"))
    
    # Message handling
    message_queue_size: int = int(os.getenv("WEBSOCKET_MESSAGE_QUEUE_SIZE", "1000"))
    heartbeat_interval: float = float(os.getenv("WEBSOCKET_HEARTBEAT_INTERVAL", "30.0"))
    
    # Load balancing and clustering
    cluster_enabled: bool = os.getenv("WEBSOCKET_CLUSTER_ENABLED", "false").lower() == "true"
    cluster_nodes: List[str] = field(default_factory=lambda: os.getenv("WEBSOCKET_CLUSTER_NODES", "").split(",") if os.getenv("WEBSOCKET_CLUSTER_NODES") else [])
    sticky_sessions: bool = os.getenv("WEBSOCKET_STICKY_SESSIONS", "true").lower() == "true"
    
    def get_server_url(self) -> str:
        """Generate WebSocket server URL."""
        protocol = "wss" if self.ssl_enabled else "ws"
        return f"{protocol}://{self.host}:{self.port}{self.path}"
    
    def validate(self) -> List[str]:
        """Validate WebSocket configuration."""
        issues = []
        
        if self.port < 1 or self.port > 65535:
            issues.append("WebSocket port must be between 1 and 65535")
        
        if self.max_connections < 1:
            issues.append("WebSocket max connections must be >= 1")
        
        if self.ssl_enabled and not (self.ssl_certfile and self.ssl_keyfile):
            issues.append("WebSocket SSL certificate and key files are required when SSL is enabled")
        
        if self.cluster_enabled and not self.cluster_nodes:
            issues.append("WebSocket cluster nodes must be specified when clustering is enabled")
        
        return issues

# ================================================================================
# ENHANCED SECURITY CONFIGURATION
# ================================================================================

@dataclass
class SecurityConfiguration:
    """Enhanced security configuration with comprehensive protection features."""
    
    # Authentication settings
    jwt_secret_key: SecretStr = SecretStr(os.getenv("JWT_SECRET_KEY", ""))
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    jwt_refresh_enabled: bool = os.getenv("JWT_REFRESH_ENABLED", "true").lower() == "true"
    jwt_refresh_expiration_days: int = int(os.getenv("JWT_REFRESH_EXPIRATION_DAYS", "7"))
    
    # API Authentication
    api_key_required: bool = os.getenv("API_KEY_REQUIRED", "true").lower() == "true"
    api_key_header: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    api_keys: Set[str] = field(default_factory=lambda: set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set())
    
    # Rate limiting
    rate_limiting_enabled: bool = os.getenv("RATE_LIMITING_ENABLED", "true").lower() == "true"
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    rate_limit_per_day: int = int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))
    rate_limit_storage: str = os.getenv("RATE_LIMIT_STORAGE", "redis")  # redis, memory
    
    # Command security
    command_validation: bool = os.getenv("COMMAND_VALIDATION", "true").lower() == "true"
    path_traversal_protection: bool = os.getenv("PATH_TRAVERSAL_PROTECTION", "true").lower() == "true"
    dangerous_command_detection: bool = os.getenv("DANGEROUS_COMMAND_DETECTION", "true").lower() == "true"
    dangerous_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /", "sudo rm", "format", "fdisk", "mkfs",
        "dd if=", ":(){ :|:& };:", "chmod 777", "chown -R"
    ])
    
    # File system security
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
        ".md", ".txt", ".csv", ".xml", ".html", ".css", ".scss", ".less"
    })
    forbidden_file_extensions: Set[str] = field(default_factory=lambda: {
        ".exe", ".bat", ".cmd", ".com", ".scr", ".vbs", ".ps1", ".sh"
    })
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    
    # Input validation
    input_sanitization: bool = os.getenv("INPUT_SANITIZATION", "true").lower() == "true"
    xss_protection: bool = os.getenv("XSS_PROTECTION", "true").lower() == "true"
    sql_injection_protection: bool = os.getenv("SQL_INJECTION_PROTECTION", "true").lower() == "true"
    
    # Audit and logging
    audit_logging_enabled: bool = os.getenv("AUDIT_LOGGING", "true").lower() == "true"
    audit_log_level: LogLevel = LogLevel(os.getenv("AUDIT_LOG_LEVEL", "INFO"))
    sensitive_data_masking: bool = os.getenv("SENSITIVE_DATA_MASKING", "true").lower() == "true"
    audit_log_retention_days: int = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "90"))
    
    # Network security
    allowed_hosts: List[str] = field(default_factory=lambda: os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(","))
    cors_enabled: bool = os.getenv("CORS_ENABLED", "false").lower() == "true"
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_headers: List[str] = field(default_factory=lambda: ["Content-Type", "Authorization", "X-API-Key"])
    
    # Encryption settings
    encryption_at_rest: bool = os.getenv("ENCRYPTION_AT_REST", "true").lower() == "true"
    encryption_in_transit: bool = os.getenv("ENCRYPTION_IN_TRANSIT", "true").lower() == "true"
    encryption_algorithm: str = os.getenv("ENCRYPTION_ALGORITHM", "AES-256-GCM")
    encryption_key: Optional[SecretStr] = SecretStr(os.getenv("ENCRYPTION_KEY", "")) if os.getenv("ENCRYPTION_KEY") else None
    
    # Advanced security features
    security_level: SecurityLevel = SecurityLevel(os.getenv("SECURITY_LEVEL", "standard"))
    threat_detection_enabled: bool = os.getenv("THREAT_DETECTION", "true").lower() == "true"
    intrusion_detection: bool = os.getenv("INTRUSION_DETECTION", "false").lower() == "true"
    security_headers_enabled: bool = os.getenv("SECURITY_HEADERS", "true").lower() == "true"
    
    # Session security
    session_timeout_minutes: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
    session_cookie_secure: bool = os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true"
    session_cookie_httponly: bool = os.getenv("SESSION_COOKIE_HTTPONLY", "true").lower() == "true"
    session_cookie_samesite: str = os.getenv("SESSION_COOKIE_SAMESITE", "strict")
    
    def validate(self) -> List[str]:
        """Validate security configuration."""
        issues = []
        
        # JWT secret validation
        if not self.jwt_secret_key.get_secret_value():
            issues.append("JWT secret key is required")
        elif len(self.jwt_secret_key.get_secret_value()) < 32:
            issues.append("JWT secret key must be at least 32 characters long")
        
        # API key validation
        if self.api_key_required and not self.api_keys:
            issues.append("API keys are required when API key authentication is enabled")
        
        # Rate limiting validation
        if self.rate_limit_per_minute < 1:
            issues.append("Rate limit per minute must be >= 1")
        
        # Encryption validation
        if self.encryption_at_rest and not self.encryption_key:
            issues.append("Encryption key is required when encryption at rest is enabled")
        
        return issues

# ================================================================================
# ENHANCED MONITORING CONFIGURATION
# ================================================================================

@dataclass
class MonitoringConfiguration:
    """Enhanced monitoring and observability configuration."""
    
    # Prometheus metrics
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    metrics_host: str = os.getenv("METRICS_HOST", "0.0.0.0")
    metrics_port: int = int(os.getenv("METRICS_PORT", "9091"))
    metrics_path: str = os.getenv("METRICS_PATH", "/metrics")
    metrics_retention_days: int = int(os.getenv("METRICS_RETENTION_DAYS", "30"))
    
    # Custom metrics
    custom_metrics_enabled: bool = os.getenv("CUSTOM_METRICS_ENABLED", "true").lower() == "true"
    business_metrics_enabled: bool = os.getenv("BUSINESS_METRICS_ENABLED", "true").lower() == "true"
    
    # Health checks
    health_check_enabled: bool = os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true"
    health_check_path: str = os.getenv("HEALTH_CHECK_PATH", "/health")
    health_check_interval_seconds: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    health_check_timeout_seconds: float = float(os.getenv("HEALTH_CHECK_TIMEOUT", "5.0"))
    deep_health_check_enabled: bool = os.getenv("DEEP_HEALTH_CHECK_ENABLED", "false").lower() == "true"
    
    # Logging configuration
    log_level: LogLevel = LogLevel(os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = os.getenv("LOG_FORMAT", "json")  # json, text
    log_file: Optional[str] = os.getenv("LOG_FILE")
    log_rotation_enabled: bool = os.getenv("LOG_ROTATION_ENABLED", "true").lower() == "true"
    log_max_size_mb: int = int(os.getenv("LOG_MAX_SIZE_MB", "100"))
    log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "10"))
    
    # Structured logging
    structured_logging: bool = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
    log_correlation_id: bool = os.getenv("LOG_CORRELATION_ID", "true").lower() == "true"
    log_request_id: bool = os.getenv("LOG_REQUEST_ID", "true").lower() == "true"
    
    # Performance monitoring
    performance_monitoring: bool = os.getenv("PERFORMANCE_MONITORING", "true").lower() == "true"
    response_time_tracking: bool = os.getenv("RESPONSE_TIME_TRACKING", "true").lower() == "true"
    resource_usage_tracking: bool = os.getenv("RESOURCE_USAGE_TRACKING", "true").lower() == "true"
    slow_query_threshold_ms: float = float(os.getenv("SLOW_QUERY_THRESHOLD_MS", "1000.0"))
    
    # Distributed tracing
    tracing_enabled: bool = os.getenv("TRACING_ENABLED", "false").lower() == "true"
    tracing_endpoint: Optional[str] = os.getenv("TRACING_ENDPOINT")
    trace_sampling_rate: float = float(os.getenv("TRACE_SAMPLING_RATE", "0.1"))
    trace_export_timeout_seconds: int = int(os.getenv("TRACE_EXPORT_TIMEOUT", "30"))
    
    # Error tracking
    error_tracking_enabled: bool = os.getenv("ERROR_TRACKING_ENABLED", "true").lower() == "true"
    error_aggregation_window_seconds: int = int(os.getenv("ERROR_AGGREGATION_WINDOW", "300"))
    error_notification_threshold: int = int(os.getenv("ERROR_NOTIFICATION_THRESHOLD", "10"))
    
    # Alerting configuration
    alerting_enabled: bool = os.getenv("ALERTING_ENABLED", "false").lower() == "true"
    alert_webhook_url: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")
    alert_email: Optional[str] = os.getenv("ALERT_EMAIL")
    alert_slack_webhook: Optional[str] = os.getenv("ALERT_SLACK_WEBHOOK")
    alert_pagerduty_key: Optional[str] = os.getenv("ALERT_PAGERDUTY_KEY")
    
    # Dashboard and visualization
    dashboard_enabled: bool = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"
    grafana_enabled: bool = os.getenv("GRAFANA_ENABLED", "false").lower() == "true"
    grafana_url: Optional[str] = os.getenv("GRAFANA_URL")
    
    def validate(self) -> List[str]:
        """Validate monitoring configuration."""
        issues = []
        
        if self.metrics_port < 1 or self.metrics_port > 65535:
            issues.append("Metrics port must be between 1 and 65535")
        
        if self.health_check_interval_seconds < 1:
            issues.append("Health check interval must be >= 1 second")
        
        if self.trace_sampling_rate < 0.0 or self.trace_sampling_rate > 1.0:
            issues.append("Trace sampling rate must be between 0.0 and 1.0")
        
        return issues

# ================================================================================
# ORCHESTRATION CONFIGURATION
# ================================================================================

@dataclass
class OrchestrationConfiguration:
    """Configuration for agent orchestration and coordination."""
    
    # Core orchestration settings
    max_agents: int = int(os.getenv("MAX_AGENTS", "55"))
    agent_registration_timeout_seconds: float = float(os.getenv("AGENT_REGISTRATION_TIMEOUT", "100.0"))
    agent_heartbeat_interval_seconds: int = int(os.getenv("AGENT_HEARTBEAT_INTERVAL", "30"))
    agent_health_check_timeout_seconds: float = float(os.getenv("AGENT_HEALTH_CHECK_TIMEOUT", "10.0"))
    
    # Task management
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "200"))
    task_queue_size: int = int(os.getenv("TASK_QUEUE_SIZE", "10000"))
    task_timeout_seconds: float = float(os.getenv("TASK_TIMEOUT_SECONDS", "300.0"))
    task_retry_attempts: int = int(os.getenv("TASK_RETRY_ATTEMPTS", "3"))
    task_retry_backoff_base: float = float(os.getenv("TASK_RETRY_BACKOFF_BASE", "2.0"))
    
    # Load balancing
    load_balancing_enabled: bool = os.getenv("LOAD_BALANCING_ENABLED", "true").lower() == "true"
    load_balancing_algorithm: str = os.getenv("LOAD_BALANCING_ALGORITHM", "round_robin")  # round_robin, least_connections, weighted
    load_balancing_weights: Dict[str, float] = field(default_factory=dict)
    
    # Capacity management
    capacity_monitoring_enabled: bool = os.getenv("CAPACITY_MONITORING_ENABLED", "true").lower() == "true"
    auto_scaling_enabled: bool = os.getenv("AUTO_SCALING_ENABLED", "false").lower() == "true"
    scale_up_threshold: float = float(os.getenv("SCALE_UP_THRESHOLD", "0.8"))
    scale_down_threshold: float = float(os.getenv("SCALE_DOWN_THRESHOLD", "0.3"))
    min_agents: int = int(os.getenv("MIN_AGENTS", "1"))
    max_scaling_agents: int = int(os.getenv("MAX_SCALING_AGENTS", "100"))
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
    circuit_breaker_threshold: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
    circuit_breaker_timeout_seconds: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
    circuit_breaker_recovery_timeout_seconds: int = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30"))
    
    # Message routing
    message_routing_enabled: bool = os.getenv("MESSAGE_ROUTING_ENABLED", "true").lower() == "true"
    message_persistence_enabled: bool = os.getenv("MESSAGE_PERSISTENCE_ENABLED", "true").lower() == "true"
    message_compression_enabled: bool = os.getenv("MESSAGE_COMPRESSION_ENABLED", "true").lower() == "true"
    message_encryption_enabled: bool = os.getenv("MESSAGE_ENCRYPTION_ENABLED", "false").lower() == "true"
    
    # Dead letter queue
    dlq_enabled: bool = os.getenv("DLQ_ENABLED", "true").lower() == "true"
    dlq_max_retries: int = int(os.getenv("DLQ_MAX_RETRIES", "3"))
    dlq_retention_hours: int = int(os.getenv("DLQ_RETENTION_HOURS", "72"))
    dlq_processing_interval_seconds: int = int(os.getenv("DLQ_PROCESSING_INTERVAL", "300"))
    
    # Plugin system
    plugins_enabled: bool = os.getenv("PLUGINS_ENABLED", "true").lower() == "true"
    plugin_directory: str = os.getenv("PLUGIN_DIRECTORY", "app/core/orchestrator_plugins")
    plugin_auto_discovery: bool = os.getenv("PLUGIN_AUTO_DISCOVERY", "true").lower() == "true"
    
    def validate(self) -> List[str]:
        """Validate orchestration configuration."""
        issues = []
        
        if self.max_agents < 1:
            issues.append("Max agents must be >= 1")
        
        if self.max_concurrent_tasks < 1:
            issues.append("Max concurrent tasks must be >= 1")
        
        if self.min_agents > self.max_agents:
            issues.append("Min agents cannot be greater than max agents")
        
        if self.scale_up_threshold <= self.scale_down_threshold:
            issues.append("Scale up threshold must be greater than scale down threshold")
        
        if self.load_balancing_algorithm not in ["round_robin", "least_connections", "weighted"]:
            issues.append("Invalid load balancing algorithm")
        
        return issues