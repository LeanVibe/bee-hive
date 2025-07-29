"""
Configuration Management for Semantic Memory Service

Centralized configuration management with environment-based settings,
validation, and performance tuning for all semantic memory components.

Features:
- Environment-based configuration loading
- Performance optimization settings
- Database and connection pool configuration
- OpenAI API and embedding service settings
- Monitoring and logging configuration
- Production vs development profiles
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration for pgvector operations."""
    # Connection settings
    connection_pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    
    # pgvector settings
    embedding_dimensions: int = 1536
    hnsw_m: int = 16  # Number of connections for HNSW
    hnsw_ef_construction: int = 64  # Size of dynamic candidate list
    hnsw_ef_search: int = 40  # Size of dynamic candidate list for search
    
    # Performance settings
    work_mem_mb: int = 256
    maintenance_work_mem_mb: int = 512
    max_parallel_workers: int = 4
    
    # Query optimization
    enable_seqscan: bool = False
    random_page_cost: float = 1.1
    effective_cache_size_mb: int = 4096


@dataclass
class EmbeddingConfig:
    """Configuration for OpenAI embedding service."""
    # OpenAI API settings
    model: str = "text-embedding-ada-002"
    api_key: Optional[str] = None
    organization: Optional[str] = None
    api_base: Optional[str] = None
    
    # Performance settings
    batch_size: int = 50
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30
    
    # Rate limiting
    rate_limit_rpm: int = 3000  # Requests per minute
    rate_limit_tpm: int = 1000000  # Tokens per minute
    
    # Retry configuration
    retry_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0
    
    # Caching
    cache_ttl_hours: int = 24
    cache_max_size: int = 10000
    enable_disk_cache: bool = False
    disk_cache_path: str = "/tmp/semantic_memory_cache"
    
    # Content processing
    max_tokens: int = 8191
    enable_preprocessing: bool = True
    enable_token_optimization: bool = True


@dataclass
class SearchConfig:
    """Configuration for semantic search operations."""
    # Default search parameters
    default_similarity_threshold: float = 0.7
    max_search_results: int = 100
    default_search_limit: int = 10
    
    # Performance targets
    target_p95_latency_ms: float = 200.0
    target_throughput_docs_per_sec: float = 500.0
    
    # Search optimization
    enable_reranking: bool = True
    reranking_model: Optional[str] = None
    enable_query_expansion: bool = False
    enable_result_caching: bool = True
    result_cache_ttl_seconds: int = 300  # 5 minutes
    
    # Filtering
    enable_agent_isolation: bool = True
    enable_workflow_scoping: bool = True
    enable_metadata_filtering: bool = True


@dataclass
class CompressionConfig:
    """Configuration for context compression algorithms."""
    # Default compression settings
    default_compression_method: str = "semantic_clustering"
    default_target_reduction: float = 0.7
    default_importance_threshold: float = 0.8
    
    # Algorithm parameters
    clustering_max_clusters: int = 10
    clustering_min_cluster_size: int = 2
    temporal_decay_days: int = 30
    hybrid_weight_importance: float = 0.7
    hybrid_weight_recency: float = 0.3
    
    # Performance settings
    max_documents_per_compression: int = 10000
    compression_timeout_seconds: int = 300  # 5 minutes


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and observability."""
    # Logging
    log_level: LogLevel = LogLevel.INFO
    enable_structured_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Metrics
    enable_prometheus_metrics: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Performance monitoring
    enable_performance_tracking: bool = True
    track_slow_queries_ms: float = 1000.0
    track_memory_usage: bool = True
    
    # Health checks
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    
    # Alerting
    enable_alerting: bool = False
    alert_webhook_url: Optional[str] = None
    alert_on_error_rate_percent: float = 5.0
    alert_on_latency_ms: float = 500.0


@dataclass
class SecurityConfig:
    """Security configuration for the service."""
    # API security
    enable_authentication: bool = True
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 1000
    rate_limit_requests_per_hour: int = 10000
    
    # CORS
    enable_cors: bool = True
    cors_allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allowed_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Input validation
    max_content_length: int = 100000  # characters
    max_batch_size: int = 100
    enable_content_filtering: bool = True


@dataclass
class SemanticMemoryConfig:
    """Main configuration class for the Semantic Memory Service."""
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Service settings
    service_name: str = "semantic-memory-service"
    service_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # Performance settings
    max_workers: int = 4
    worker_timeout_seconds: int = 300
    graceful_shutdown_timeout: int = 30
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_environment_overrides()
        self._optimize_for_environment()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate embedding dimensions
        if self.database.embedding_dimensions not in [1536, 3072]:
            logger.warning(f"Unusual embedding dimensions: {self.database.embedding_dimensions}")
        
        # Validate batch sizes
        if self.embedding.batch_size > 100:
            logger.warning(f"Large embedding batch size may cause timeouts: {self.embedding.batch_size}")
        
        # Validate rate limits
        if self.embedding.rate_limit_rpm > 5000:
            logger.warning(f"High OpenAI rate limit may exceed plan: {self.embedding.rate_limit_rpm}")
        
        # Validate search parameters
        if self.search.target_p95_latency_ms > 1000:
            logger.warning(f"High P95 latency target: {self.search.target_p95_latency_ms}ms")
    
    def _setup_environment_overrides(self):
        """Override configuration from environment variables."""
        # Database settings
        if pool_size := os.getenv("SEMANTIC_MEMORY_DB_POOL_SIZE"):
            self.database.connection_pool_size = int(pool_size)
        
        if max_overflow := os.getenv("SEMANTIC_MEMORY_DB_MAX_OVERFLOW"):
            self.database.max_overflow = int(max_overflow)
        
        # OpenAI settings
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.embedding.api_key = api_key
        
        if org := os.getenv("OPENAI_ORGANIZATION"):
            self.embedding.organization = org
        
        if api_base := os.getenv("OPENAI_API_BASE"):
            self.embedding.api_base = api_base
        
        # Performance settings
        if batch_size := os.getenv("SEMANTIC_MEMORY_BATCH_SIZE"):
            self.embedding.batch_size = int(batch_size)
        
        if concurrent_requests := os.getenv("SEMANTIC_MEMORY_MAX_CONCURRENT"):
            self.embedding.max_concurrent_requests = int(concurrent_requests)
        
        # Monitoring settings
        if log_level := os.getenv("SEMANTIC_MEMORY_LOG_LEVEL"):
            self.monitoring.log_level = LogLevel(log_level.upper())
        
        if metrics_port := os.getenv("SEMANTIC_MEMORY_METRICS_PORT"):
            self.monitoring.metrics_port = int(metrics_port)
        
        # Security settings
        if jwt_secret := os.getenv("SEMANTIC_MEMORY_JWT_SECRET"):
            self.security.jwt_secret_key = jwt_secret
        
        # Environment detection
        if env := os.getenv("ENVIRONMENT"):
            self.environment = Environment(env.lower())
    
    def _optimize_for_environment(self):
        """Optimize configuration based on environment."""
        if self.environment == Environment.PRODUCTION:
            # Production optimizations
            self.debug = False
            self.monitoring.log_level = LogLevel.INFO
            self.monitoring.enable_performance_tracking = True
            self.security.enable_authentication = True
            self.security.enable_rate_limiting = True
            
            # Higher performance targets
            self.search.target_p95_latency_ms = 150.0
            self.search.target_throughput_docs_per_sec = 750.0
            
            # Larger connection pools
            self.database.connection_pool_size = 50
            self.database.max_overflow = 100
            
            # More aggressive caching
            self.embedding.cache_ttl_hours = 48
            self.search.result_cache_ttl_seconds = 900  # 15 minutes
            
        elif self.environment == Environment.DEVELOPMENT:
            # Development optimizations
            self.debug = True
            self.monitoring.log_level = LogLevel.DEBUG
            self.security.enable_authentication = False
            self.security.enable_rate_limiting = False
            
            # Smaller pools for development
            self.database.connection_pool_size = 5
            self.database.max_overflow = 10
            
            # Shorter cache times
            self.embedding.cache_ttl_hours = 1
            self.search.result_cache_ttl_seconds = 60
            
        elif self.environment == Environment.TESTING:
            # Testing optimizations
            self.debug = True
            self.monitoring.log_level = LogLevel.WARNING
            self.security.enable_authentication = False
            self.security.enable_rate_limiting = False
            
            # Minimal resources for testing
            self.database.connection_pool_size = 2
            self.database.max_overflow = 5
            self.embedding.batch_size = 5
            self.embedding.max_concurrent_requests = 2
            
            # Disable caching for consistent test results
            self.embedding.cache_ttl_hours = 0
            self.search.result_cache_ttl_seconds = 0
    
    def get_database_url(self) -> str:
        """Get database URL with optimized parameters."""
        base_url = os.getenv("DATABASE_URL", "postgresql://localhost/leanvibe_dev")
        
        # Add connection parameters for performance
        params = [
            f"pool_size={self.database.connection_pool_size}",
            f"max_overflow={self.database.max_overflow}",
            f"pool_timeout={self.database.pool_timeout}",
            f"pool_recycle={self.database.pool_recycle}",
        ]
        
        separator = "&" if "?" in base_url else "?"
        return f"{base_url}{separator}{'&'.join(params)}"
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI client configuration."""
        config = {
            "api_key": self.embedding.api_key,
            "timeout": self.embedding.timeout_seconds,
        }
        
        if self.embedding.organization:
            config["organization"] = self.embedding.organization
        
        if self.embedding.api_base:
            config["api_base"] = self.embedding.api_base
        
        return config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.monitoring.log_format
                },
                "structured": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "level": self.monitoring.log_level.value,
                    "formatter": "structured" if self.monitoring.enable_structured_logging else "standard",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "semantic_memory": {
                    "handlers": ["default"],
                    "level": self.monitoring.log_level.value,
                    "propagate": False
                },
                "pgvector": {
                    "handlers": ["default"],
                    "level": self.monitoring.log_level.value,
                    "propagate": False
                },
                "openai": {
                    "handlers": ["default"],
                    "level": "WARNING",  # Reduce OpenAI noise
                    "propagate": False
                }
            },
            "root": {
                "level": self.monitoring.log_level.value,
                "handlers": ["default"]
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "database": {
                "connection_pool_size": self.database.connection_pool_size,
                "max_overflow": self.database.max_overflow,
                "embedding_dimensions": self.database.embedding_dimensions,
                "hnsw_m": self.database.hnsw_m,
                "hnsw_ef_construction": self.database.hnsw_ef_construction,
                "hnsw_ef_search": self.database.hnsw_ef_search
            },
            "embedding": {
                "model": self.embedding.model,
                "batch_size": self.embedding.batch_size,
                "max_concurrent_requests": self.embedding.max_concurrent_requests,
                "rate_limit_rpm": self.embedding.rate_limit_rpm,
                "rate_limit_tpm": self.embedding.rate_limit_tpm,
                "cache_ttl_hours": self.embedding.cache_ttl_hours
            },
            "search": {
                "default_similarity_threshold": self.search.default_similarity_threshold,
                "max_search_results": self.search.max_search_results,
                "target_p95_latency_ms": self.search.target_p95_latency_ms,
                "target_throughput_docs_per_sec": self.search.target_throughput_docs_per_sec
            },
            "monitoring": {
                "log_level": self.monitoring.log_level.value,
                "enable_prometheus_metrics": self.monitoring.enable_prometheus_metrics,
                "enable_performance_tracking": self.monitoring.enable_performance_tracking
            }
        }


# Global configuration instance
_config: Optional[SemanticMemoryConfig] = None

def get_semantic_memory_config() -> SemanticMemoryConfig:
    """Get the global semantic memory configuration."""
    global _config
    
    if _config is None:
        _config = SemanticMemoryConfig()
        logger.info(f"✅ Semantic Memory configuration loaded for {_config.environment.value} environment")
        
        if _config.debug:
            logger.debug(f"Configuration: {_config.to_dict()}")
    
    return _config

def reload_config() -> SemanticMemoryConfig:
    """Reload configuration from environment variables."""
    global _config
    
    _config = None
    return get_semantic_memory_config()

def update_config(**kwargs) -> SemanticMemoryConfig:
    """Update configuration with new values."""
    global _config
    
    config = get_semantic_memory_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.info(f"Configuration updated: {key} = {value}")
        else:
            logger.warning(f"Unknown configuration key: {key}")
    
    return config


# Configuration presets for different scenarios
DEVELOPMENT_PRESET = {
    "database": DatabaseConfig(
        connection_pool_size=5,
        max_overflow=10,
        hnsw_ef_search=20
    ),
    "embedding": EmbeddingConfig(
        batch_size=10,
        max_concurrent_requests=2,
        cache_ttl_hours=1
    ),
    "monitoring": MonitoringConfig(
        log_level=LogLevel.DEBUG,
        enable_performance_tracking=False
    )
}

PRODUCTION_PRESET = {
    "database": DatabaseConfig(
        connection_pool_size=50,
        max_overflow=100,
        hnsw_ef_search=64,
        work_mem_mb=512,
        maintenance_work_mem_mb=1024
    ),
    "embedding": EmbeddingConfig(
        batch_size=50,
        max_concurrent_requests=20,
        cache_ttl_hours=48,
        enable_disk_cache=True
    ),
    "search": SearchConfig(
        target_p95_latency_ms=150.0,
        target_throughput_docs_per_sec=1000.0,
        enable_result_caching=True
    ),
    "monitoring": MonitoringConfig(
        log_level=LogLevel.INFO,
        enable_prometheus_metrics=True,
        enable_performance_tracking=True,
        enable_alerting=True
    )
}

TESTING_PRESET = {
    "database": DatabaseConfig(
        connection_pool_size=2,
        max_overflow=5
    ),
    "embedding": EmbeddingConfig(
        batch_size=5,
        max_concurrent_requests=1,
        cache_ttl_hours=0  # Disable caching for consistent tests
    ),
    "monitoring": MonitoringConfig(
        log_level=LogLevel.WARNING,
        enable_performance_tracking=False
    ),
    "security": SecurityConfig(
        enable_authentication=False,
        enable_rate_limiting=False
    )
}