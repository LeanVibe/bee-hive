"""
Unified Configuration Service for LeanVibe Agent Hive 2.0

Centralized configuration management system that consolidates all scattered configuration 
patterns into a single, type-safe service with comprehensive validation and environment detection.

This service replaces and integrates:
- app/core/config.py (main application configuration)
- app/core/error_handling_config.py (error handling settings)
- app/config/semantic_memory_config.py (semantic memory configuration)
- app/core/sandbox/sandbox_config.py (sandbox environment configuration)
- Various module-specific configuration patterns

Key Features:
- Type-safe Pydantic settings with validation
- Environment-based configuration with intelligent detection
- Centralized environment variable management
- Configuration validation and hot-reload support
- Backwards compatibility with existing patterns
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union, Type
from pathlib import Path
from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging for this module
logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Deployment environment types with intelligent detection."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels for structured configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfiguration(BaseSettings):
    """Database configuration with connection pooling and performance settings."""
    
    url: str = Field(..., env="DATABASE_URL", description="Primary database connection URL")
    pool_size: int = Field(20, env="DATABASE_POOL_SIZE", description="Connection pool size")
    max_overflow: int = Field(30, env="DATABASE_MAX_OVERFLOW", description="Max pool overflow connections")
    pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT", description="Pool checkout timeout (seconds)")
    pool_recycle: int = Field(3600, env="DATABASE_POOL_RECYCLE", description="Connection recycle time (seconds)")
    pool_pre_ping: bool = Field(True, env="DATABASE_POOL_PRE_PING", description="Enable connection pre-ping")
    
    # pgvector specific settings
    embedding_dimensions: int = Field(1536, env="DATABASE_EMBEDDING_DIMENSIONS", description="Vector embedding dimensions")
    hnsw_m: int = Field(16, env="DATABASE_HNSW_M", description="HNSW index parameter M")
    hnsw_ef_construction: int = Field(64, env="DATABASE_HNSW_EF_CONSTRUCTION", description="HNSW construction parameter")
    hnsw_ef_search: int = Field(40, env="DATABASE_HNSW_EF_SEARCH", description="HNSW search parameter")
    
    @field_validator('url')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        valid_prefixes = (
            'postgresql://', 'postgresql+asyncpg://', 'postgresql+psycopg2://',
            'sqlite://', 'sqlite+aiosqlite://', 'mysql://', 'mysql+aiomysql://'
        )
        if not v.startswith(valid_prefixes):
            raise ValueError(f'Database URL must start with one of: {", ".join(valid_prefixes)}')
        return v

    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class RedisConfiguration(BaseSettings):
    """Redis configuration for caching and message streaming."""
    
    url: str = Field("redis://localhost:6379", env="REDIS_URL", description="Redis connection URL")
    max_connections: int = Field(200, env="REDIS_MAX_CONNECTIONS", description="Maximum Redis connections")
    connection_pool_size: int = Field(50, env="REDIS_CONNECTION_POOL_SIZE", description="Redis connection pool size")
    connection_timeout: float = Field(5.0, env="REDIS_CONNECTION_TIMEOUT", description="Redis connection timeout")
    retry_on_timeout: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT", description="Retry on connection timeout")
    health_check_interval: int = Field(30, env="REDIS_HEALTH_CHECK_INTERVAL", description="Health check interval")
    
    # Stream configuration
    stream_max_len: int = Field(10000, env="REDIS_STREAM_MAX_LEN", description="Maximum stream length")
    
    @field_validator('url')
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith(('redis://', 'rediss://')):
            raise ValueError('Redis URL must start with redis:// or rediss://')
        return v

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class SecurityConfiguration(BaseSettings):
    """Security configuration with JWT, authentication, and encryption settings."""
    
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY", description="Application secret key")
    jwt_secret_key: str = Field(default="dev-jwt-secret-key-change-in-production", env="JWT_SECRET_KEY", description="JWT signing secret key")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM", description="JWT signing algorithm")
    jwt_access_token_expire_minutes: int = Field(30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES", description="JWT token expiration")
    
    # API Security
    allowed_hosts: List[str] = Field(["*"], env="ALLOWED_HOSTS", description="Allowed host headers")
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8080", "http://localhost:5173"],
        env="CORS_ORIGINS",
        description="CORS allowed origins"
    )
    
    # Enterprise Security
    security_enabled: bool = Field(True, env="SECURITY_ENABLED", description="Enable security features")
    mfa_enabled: bool = Field(True, env="MFA_ENABLED", description="Enable multi-factor authentication")
    api_rate_limit_enabled: bool = Field(True, env="API_RATE_LIMIT_ENABLED", description="Enable API rate limiting")
    threat_detection_enabled: bool = Field(True, env="THREAT_DETECTION_ENABLED", description="Enable threat detection")
    
    # Secrets Management
    secrets_encryption_enabled: bool = Field(True, env="SECRETS_ENCRYPTION_ENABLED", description="Enable secrets encryption")
    secret_rotation_enabled: bool = Field(True, env="SECRET_ROTATION_ENABLED", description="Enable secret rotation")
    
    @field_validator('secret_key', 'jwt_secret_key')
    @classmethod
    def validate_secret_length(cls, v):
        """Validate secret key length."""
        if len(v) < 32:
            raise ValueError('Secret keys must be at least 32 characters long')
        return v
    
    @field_validator('cors_origins', 'allowed_hosts', mode='before')
    @classmethod
    def parse_list_values(cls, v):
        """Parse comma-separated string values into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v

    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class PerformanceConfiguration(BaseSettings):
    """Performance and monitoring configuration."""
    
    max_workers: int = Field(4, env="MAX_WORKERS", description="Maximum worker processes")
    request_timeout: int = Field(300, env="REQUEST_TIMEOUT", description="Request timeout (seconds)")
    max_request_size: int = Field(16 * 1024 * 1024, env="MAX_REQUEST_SIZE", description="Maximum request size (bytes)")
    
    # Metrics and Monitoring
    enable_metrics: bool = Field(True, env="ENABLE_METRICS", description="Enable metrics collection")
    metrics_port: int = Field(9090, env="METRICS_PORT", description="Prometheus metrics port")
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED", description="Enable Prometheus metrics")
    
    # Performance Targets
    target_throughput_msg_per_sec: int = Field(10000, env="TARGET_THROUGHPUT_MSG_PER_SEC", description="Target message throughput")
    target_p95_latency_ms: float = Field(200.0, env="TARGET_P95_LATENCY_MS", description="Target P95 latency")
    target_p99_latency_ms: float = Field(500.0, env="TARGET_P99_LATENCY_MS", description="Target P99 latency")
    target_success_rate: float = Field(0.999, env="TARGET_SUCCESS_RATE", description="Target success rate")

    model_config = SettingsConfigDict(env_prefix="PERFORMANCE_")


class ErrorHandlingConfiguration(BaseSettings):
    """Error handling and resilience configuration."""
    
    enabled: bool = Field(True, env="ERROR_HANDLING_ENABLED", description="Enable error handling")
    
    # Circuit Breaker Configuration
    circuit_breaker_enabled: bool = Field(True, env="CIRCUIT_BREAKER_ENABLED", description="Enable circuit breakers")
    circuit_breaker_failure_threshold: int = Field(10, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD", description="Failure threshold")
    circuit_breaker_timeout_seconds: int = Field(60, env="CIRCUIT_BREAKER_TIMEOUT_SECONDS", description="Circuit breaker timeout")
    circuit_breaker_success_threshold: int = Field(5, env="CIRCUIT_BREAKER_SUCCESS_THRESHOLD", description="Success threshold")
    
    # Retry Policy Configuration
    retry_max_attempts: int = Field(3, env="RETRY_MAX_ATTEMPTS", description="Maximum retry attempts")
    retry_base_delay_ms: int = Field(100, env="RETRY_BASE_DELAY_MS", description="Base retry delay")
    retry_max_delay_ms: int = Field(30000, env="RETRY_MAX_DELAY_MS", description="Maximum retry delay")
    retry_backoff_multiplier: float = Field(2.0, env="RETRY_BACKOFF_MULTIPLIER", description="Backoff multiplier")
    
    # Dead Letter Queue Configuration
    dlq_enabled: bool = Field(True, env="DLQ_ENABLED", description="Enable dead letter queue")
    dlq_max_retries: int = Field(3, env="DLQ_MAX_RETRIES", description="DLQ maximum retries")
    dlq_max_size: int = Field(100000, env="DLQ_MAX_SIZE", description="DLQ maximum size")
    dlq_ttl_hours: int = Field(72, env="DLQ_TTL_HOURS", description="DLQ time-to-live")

    model_config = SettingsConfigDict(env_prefix="ERROR_HANDLING_")


class ProjectIndexConfiguration(BaseSettings):
    """Project Index specific configuration."""
    
    max_file_size: int = Field(10 * 1024 * 1024, env="PROJECT_INDEX_MAX_FILE_SIZE", description="Maximum file size")
    supported_extensions: List[str] = Field([".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml"], 
                                          env="PROJECT_INDEX_EXTENSIONS", description="Supported file extensions")
    analysis_timeout: int = Field(300, env="PROJECT_INDEX_ANALYSIS_TIMEOUT", description="Analysis timeout")
    cache_ttl: int = Field(3600, env="PROJECT_INDEX_CACHE_TTL", description="Cache time-to-live")
    
    # Context Management
    context_max_tokens: int = Field(8000, env="PROJECT_INDEX_CONTEXT_MAX_TOKENS", description="Maximum context tokens")
    context_compression_threshold: float = Field(0.8, env="PROJECT_INDEX_COMPRESSION_THRESHOLD", description="Compression threshold")
    
    @field_validator('supported_extensions', mode='before')
    @classmethod
    def parse_extensions(cls, v):
        """Parse comma-separated extensions."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    model_config = SettingsConfigDict(env_prefix="PROJECT_INDEX_")


class SandboxConfiguration(BaseSettings):
    """Sandbox mode configuration for development and testing."""
    
    enabled: bool = Field(False, env="SANDBOX_MODE", description="Enable sandbox mode")
    demo_mode: bool = Field(False, env="SANDBOX_DEMO_MODE", description="Enable demo mode")
    auto_detected: bool = Field(False, description="Sandbox mode auto-detected")
    
    # Mock Services
    mock_anthropic: bool = Field(False, env="SANDBOX_MOCK_ANTHROPIC", description="Mock Anthropic API")
    mock_openai: bool = Field(False, env="SANDBOX_MOCK_OPENAI", description="Mock OpenAI API")
    mock_github: bool = Field(False, env="SANDBOX_MOCK_GITHUB", description="Mock GitHub API")
    
    # Demo Settings
    response_delay_min: float = Field(1.0, env="SANDBOX_RESPONSE_DELAY_MIN", description="Minimum response delay")
    response_delay_max: float = Field(4.0, env="SANDBOX_RESPONSE_DELAY_MAX", description="Maximum response delay")
    show_banner: bool = Field(True, env="SANDBOX_SHOW_BANNER", description="Show sandbox banner")

    model_config = SettingsConfigDict(env_prefix="SANDBOX_")


class ExternalServicesConfiguration(BaseSettings):
    """External service API configuration."""
    
    # Anthropic Claude API
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY", description="Anthropic API key")
    anthropic_model: str = Field("claude-3-5-sonnet-20241022", env="ANTHROPIC_MODEL", description="Default Claude model")
    anthropic_max_tokens: int = Field(4096, env="ANTHROPIC_MAX_TOKENS", description="Maximum tokens for Claude")
    
    # OpenAI API
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY", description="OpenAI API key")
    openai_model: Optional[str] = Field(None, env="OPENAI_MODEL", description="Default OpenAI model")
    openai_embedding_model: str = Field("text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL", description="Embedding model")
    openai_embedding_max_tokens: int = Field(8191, env="OPENAI_EMBEDDING_MAX_TOKENS", description="Max embedding tokens")
    openai_rate_limit_rpm: int = Field(3000, env="OPENAI_RATE_LIMIT_RPM", description="Rate limit (requests/min)")
    
    # GitHub Integration
    github_token: Optional[str] = Field(None, env="GITHUB_TOKEN", description="GitHub API token")
    github_app_id: Optional[str] = Field(None, env="GITHUB_APP_ID", description="GitHub App ID")
    github_private_key: Optional[str] = Field(None, env="GITHUB_PRIVATE_KEY", description="GitHub App private key")
    github_api_url: str = Field("https://api.github.com", env="GITHUB_API_URL", description="GitHub API URL")

    model_config = SettingsConfigDict(env_prefix="EXTERNAL_")


class ApplicationConfiguration(BaseSettings):
    """Main application configuration that aggregates all sub-configurations."""
    
    # Core Application Settings
    app_name: str = Field("LeanVibe Agent Hive 2.0", env="APP_NAME", description="Application name")
    version: str = Field("2.0", env="APP_VERSION", description="Application version")
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT", description="Deployment environment")
    debug: bool = Field(False, env="DEBUG", description="Debug mode")
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL", description="Logging level")
    api_prefix: str = Field("/api", env="API_PREFIX", description="API route prefix")
    
    # File Paths
    workspace_dir: Path = Field(Path("./workspaces"), env="WORKSPACE_DIR", description="Workspace directory")
    logs_dir: Path = Field(Path("./logs"), env="LOGS_DIR", description="Logs directory")
    checkpoints_dir: Path = Field(Path("./checkpoints"), env="CHECKPOINTS_DIR", description="Checkpoints directory")
    repositories_dir: Path = Field(Path("./repositories"), env="REPOSITORIES_DIR", description="Repositories directory")
    
    # Sub-configurations
    database: DatabaseConfiguration = DatabaseConfiguration()
    redis: RedisConfiguration = RedisConfiguration()
    security: SecurityConfiguration = SecurityConfiguration()
    performance: PerformanceConfiguration = PerformanceConfiguration()
    error_handling: ErrorHandlingConfiguration = ErrorHandlingConfiguration()
    project_index: ProjectIndexConfiguration = ProjectIndexConfiguration()
    sandbox: SandboxConfiguration = SandboxConfiguration()
    external_services: ExternalServicesConfiguration = ExternalServicesConfiguration()

    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v):
        """Validate and normalize environment value."""
        if isinstance(v, str):
            v = v.lower()
            # Map common variations
            if v in ['dev', 'develop']:
                return Environment.DEVELOPMENT
            elif v in ['test', 'tests']:
                return Environment.TESTING
            elif v in ['stage', 'staging']:
                return Environment.STAGING
            elif v in ['prod', 'production']:
                return Environment.PRODUCTION
        return v
    
    @field_validator('workspace_dir', 'logs_dir', 'checkpoints_dir', 'repositories_dir')
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @model_validator(mode='after')
    def configure_sandbox_mode(self):
        """Auto-configure sandbox mode based on missing API keys."""
        # Detect missing required API keys
        missing_keys = []
        if not self.external_services.anthropic_api_key:
            missing_keys.append("ANTHROPIC_API_KEY")
            self.sandbox.mock_anthropic = True
        
        if not self.external_services.openai_api_key:
            self.sandbox.mock_openai = True
        
        if not self.external_services.github_token:
            self.sandbox.mock_github = True
        
        # Auto-enable sandbox mode if critical keys are missing
        if missing_keys and not self.sandbox.enabled:
            self.sandbox.enabled = True
            self.sandbox.auto_detected = True
            logger.info(f"🏖️ Sandbox mode auto-enabled due to missing API keys: {', '.join(missing_keys)}")
        
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        validate_assignment=True,
        extra='ignore'
    )


class ConfigurationService:
    """
    Centralized configuration service providing unified access to all application settings.
    
    Features:
    - Singleton pattern for consistent configuration access
    - Environment-based configuration with intelligent detection
    - Type-safe configuration with Pydantic validation
    - Hot-reload support for development
    - Backwards compatibility with existing configuration patterns
    """
    
    _instance: Optional['ConfigurationService'] = None
    _config: Optional[ApplicationConfiguration] = None
    
    def __new__(cls) -> 'ConfigurationService':
        """Singleton implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration service."""
        if self._config is None:
            self._load_configuration()
    
    def _load_configuration(self):
        """Load and validate configuration from environment."""
        try:
            self._config = ApplicationConfiguration()
            
            # Apply environment-specific optimizations
            self._apply_environment_optimizations()
            
            # Log configuration summary
            self._log_configuration_summary()
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise
    
    def _apply_environment_optimizations(self):
        """Apply environment-specific configuration optimizations."""
        if not self._config:
            return
        
        env = self._config.environment
        
        if env == Environment.PRODUCTION:
            # Production optimizations
            self._config.debug = False
            self._config.log_level = LogLevel.INFO
            self._config.database.pool_size = max(self._config.database.pool_size, 50)
            self._config.performance.enable_metrics = True
            self._config.security.security_enabled = True
            
        elif env == Environment.DEVELOPMENT:
            # Development optimizations
            self._config.debug = True
            self._config.log_level = LogLevel.DEBUG
            self._config.database.pool_size = min(self._config.database.pool_size, 10)
            self._config.security.api_rate_limit_enabled = False
            
        elif env == Environment.TESTING:
            # Testing optimizations
            self._config.debug = False
            self._config.log_level = LogLevel.WARNING
            self._config.database.pool_size = 2
            self._config.redis.max_connections = 10
            self._config.performance.enable_metrics = False
            
        logger.info(f"✅ Applied {env.value} environment optimizations")
    
    def _log_configuration_summary(self):
        """Log configuration summary for debugging."""
        if not self._config:
            return
        
        summary = {
            "environment": self._config.environment.value,
            "debug": self._config.debug,
            "sandbox_mode": self._config.sandbox.enabled,
            "database_pool_size": self._config.database.pool_size,
            "redis_max_connections": self._config.redis.max_connections,
            "security_enabled": self._config.security.security_enabled,
        }
        
        logger.info(f"🚀 Configuration loaded successfully", extra=summary)
    
    @property
    def config(self) -> ApplicationConfiguration:
        """Get current configuration."""
        if self._config is None:
            self._load_configuration()
        return self._config
    
    def reload_configuration(self):
        """Reload configuration from environment variables."""
        self._config = None
        self._load_configuration()
        logger.info("🔄 Configuration reloaded from environment")
    
    # Environment Detection Methods
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.config.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.config.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.config.environment == Environment.TESTING
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.config.environment == Environment.STAGING
    
    # Service-specific Configuration Access
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return self.config.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        return self.config.redis.url
    
    def get_security_config(self) -> SecurityConfiguration:
        """Get security configuration."""
        return self.config.security
    
    def get_performance_config(self) -> PerformanceConfiguration:
        """Get performance configuration."""
        return self.config.performance
    
    def get_error_handling_config(self) -> ErrorHandlingConfiguration:
        """Get error handling configuration."""
        return self.config.error_handling
    
    def get_project_index_config(self) -> ProjectIndexConfiguration:
        """Get project index configuration."""
        return self.config.project_index
    
    def get_sandbox_config(self) -> SandboxConfiguration:
        """Get sandbox configuration."""
        return self.config.sandbox
    
    def get_external_services_config(self) -> ExternalServicesConfiguration:
        """Get external services configuration."""
        return self.config.external_services
    
    # Backwards Compatibility Methods
    def get_settings(self) -> ApplicationConfiguration:
        """Get settings (backwards compatibility with existing code)."""
        return self.config
    
    def get_anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key."""
        return self.config.external_services.anthropic_api_key
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.config.external_services.openai_api_key
    
    def get_github_token(self) -> Optional[str]:
        """Get GitHub token."""
        return self.config.external_services.github_token
    
    # Configuration Validation
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate current configuration and return issues."""
        issues = {
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Validate required keys for production
        if self.is_production():
            if not self.config.external_services.anthropic_api_key:
                issues["errors"].append("ANTHROPIC_API_KEY is required in production")
            
            if not self.config.security.secret_key:
                issues["errors"].append("SECRET_KEY is required in production")
            
            if self.config.debug:
                issues["warnings"].append("Debug mode should be disabled in production")
        
        # Validate database configuration
        if self.config.database.pool_size > 100:
            issues["warnings"].append("Large database pool size may consume excessive resources")
        
        # Validate performance targets
        if self.config.performance.target_p95_latency_ms > 1000:
            issues["warnings"].append("High P95 latency target may impact user experience")
        
        return issues
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration service status."""
        return {
            "service": "ConfigurationService",
            "version": "1.0.0",
            "environment": self.config.environment.value,
            "debug": self.config.debug,
            "sandbox_mode": self.config.sandbox.enabled,
            "loaded_at": "runtime",
            "validation_status": "ok" if not self.validate_configuration()["errors"] else "warnings"
        }


# Global configuration service instance
_config_service: Optional[ConfigurationService] = None


def get_configuration_service() -> ConfigurationService:
    """Get the global configuration service instance."""
    global _config_service
    
    if _config_service is None:
        _config_service = ConfigurationService()
    
    return _config_service


def get_config() -> ApplicationConfiguration:
    """Get application configuration (primary access method)."""
    return get_configuration_service().config


def get_settings() -> ApplicationConfiguration:
    """Get settings (backwards compatibility alias)."""
    return get_config()


# Backwards compatibility exports
settings = get_configuration_service()  # Lazy proxy for existing code


@lru_cache(maxsize=1)
def get_cached_config() -> ApplicationConfiguration:
    """Get cached configuration instance for performance."""
    return get_config()


# Configuration initialization function for explicit setup
def initialize_configuration(
    environment: Optional[Environment] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> ConfigurationService:
    """
    Initialize configuration service with optional overrides.
    
    Args:
        environment: Override environment detection
        config_overrides: Dictionary of configuration overrides
        
    Returns:
        Configured ConfigurationService instance
    """
    service = get_configuration_service()
    
    # Apply environment override
    if environment:
        os.environ["ENVIRONMENT"] = environment.value
        service.reload_configuration()
    
    # Apply configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(service.config, key):
                setattr(service.config, key, value)
        logger.info(f"Applied {len(config_overrides)} configuration overrides")
    
    # Validate final configuration
    validation_issues = service.validate_configuration()
    if validation_issues["errors"]:
        logger.error(f"❌ Configuration validation failed: {validation_issues['errors']}")
        raise ValueError(f"Configuration validation failed: {validation_issues['errors']}")
    
    if validation_issues["warnings"]:
        logger.warning(f"⚠️ Configuration warnings: {validation_issues['warnings']}")
    
    logger.info("✅ Configuration service initialized successfully")
    return service