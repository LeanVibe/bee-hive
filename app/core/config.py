"""
Configuration management for LeanVibe Agent Hive 2.0

Uses Pydantic settings for type-safe configuration with environment variable support.
Designed for multi-agent coordination with security and observability in mind.
"""

import os
from typing import List, Optional
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    APP_NAME: str = "LeanVibe Agent Hive 2.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_STREAM_MAX_LEN: int = Field(default=10000, env="REDIS_STREAM_MAX_LEN")
    
    # Redis Performance Optimization
    REDIS_CONNECTION_POOL_SIZE: int = Field(default=50, env="REDIS_CONNECTION_POOL_SIZE")
    REDIS_MAX_CONNECTIONS: int = Field(default=200, env="REDIS_MAX_CONNECTIONS")
    REDIS_CONNECTION_TIMEOUT: float = Field(default=5.0, env="REDIS_CONNECTION_TIMEOUT")
    
    # Message Batching Configuration
    MESSAGE_BATCH_SIZE: int = Field(default=100, env="MESSAGE_BATCH_SIZE")
    MESSAGE_BATCH_WAIT_MS: int = Field(default=50, env="MESSAGE_BATCH_WAIT_MS")
    ADAPTIVE_BATCHING_ENABLED: bool = Field(default=True, env="ADAPTIVE_BATCHING_ENABLED")
    BATCH_TARGET_LATENCY_MS: float = Field(default=100.0, env="BATCH_TARGET_LATENCY_MS")
    
    # Payload Compression
    COMPRESSION_ENABLED: bool = Field(default=True, env="COMPRESSION_ENABLED")
    COMPRESSION_ALGORITHM: str = Field(default="zlib", env="COMPRESSION_ALGORITHM")  # none, gzip, zlib
    COMPRESSION_LEVEL: int = Field(default=6, env="COMPRESSION_LEVEL")  # 1-9
    COMPRESSION_MIN_SIZE: int = Field(default=1024, env="COMPRESSION_MIN_SIZE")  # bytes
    
    # Dead Letter Queue Configuration
    DLQ_MAX_RETRIES: int = Field(default=3, env="DLQ_MAX_RETRIES")
    DLQ_INITIAL_RETRY_DELAY_MS: int = Field(default=1000, env="DLQ_INITIAL_RETRY_DELAY_MS")
    DLQ_MAX_RETRY_DELAY_MS: int = Field(default=60000, env="DLQ_MAX_RETRY_DELAY_MS")
    DLQ_MAX_SIZE: int = Field(default=100000, env="DLQ_MAX_SIZE")
    DLQ_TTL_HOURS: int = Field(default=72, env="DLQ_TTL_HOURS")
    DLQ_POLICY: str = Field(default="exponential_backoff", env="DLQ_POLICY")  # immediate, exponential_backoff, linear_backoff, circuit_breaker
    
    # Back-pressure Management
    BACKPRESSURE_ENABLED: bool = Field(default=True, env="BACKPRESSURE_ENABLED")
    BACKPRESSURE_WARNING_LAG: int = Field(default=1000, env="BACKPRESSURE_WARNING_LAG")
    BACKPRESSURE_CRITICAL_LAG: int = Field(default=5000, env="BACKPRESSURE_CRITICAL_LAG")
    BACKPRESSURE_EMERGENCY_LAG: int = Field(default=10000, env="BACKPRESSURE_EMERGENCY_LAG")
    CONSUMER_MIN_COUNT: int = Field(default=1, env="CONSUMER_MIN_COUNT")
    CONSUMER_MAX_COUNT: int = Field(default=10, env="CONSUMER_MAX_COUNT")
    CONSUMER_SCALE_UP_THRESHOLD: float = Field(default=0.8, env="CONSUMER_SCALE_UP_THRESHOLD")
    CONSUMER_SCALE_DOWN_THRESHOLD: float = Field(default=0.3, env="CONSUMER_SCALE_DOWN_THRESHOLD")
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_ENABLED: bool = Field(default=True, env="CIRCUIT_BREAKER_ENABLED")
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=10, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = Field(default=60, env="CIRCUIT_BREAKER_TIMEOUT_SECONDS")
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = Field(default=5, env="CIRCUIT_BREAKER_SUCCESS_THRESHOLD")
    
    # Stream Monitoring
    STREAM_MONITORING_ENABLED: bool = Field(default=True, env="STREAM_MONITORING_ENABLED")
    STREAM_MONITORING_INTERVAL: int = Field(default=5, env="STREAM_MONITORING_INTERVAL")  # seconds
    STREAM_METRICS_RETENTION_HOURS: int = Field(default=24, env="STREAM_METRICS_RETENTION_HOURS")
    PROMETHEUS_METRICS_ENABLED: bool = Field(default=True, env="PROMETHEUS_METRICS_ENABLED")
    
    # Performance Targets
    TARGET_THROUGHPUT_MSG_PER_SEC: int = Field(default=10000, env="TARGET_THROUGHPUT_MSG_PER_SEC")
    TARGET_P95_LATENCY_MS: float = Field(default=200.0, env="TARGET_P95_LATENCY_MS")
    TARGET_P99_LATENCY_MS: float = Field(default=500.0, env="TARGET_P99_LATENCY_MS")
    TARGET_SUCCESS_RATE: float = Field(default=0.999, env="TARGET_SUCCESS_RATE")  # 99.9%
    
    # Load Testing Configuration
    LOAD_TEST_ENABLED: bool = Field(default=False, env="LOAD_TEST_ENABLED")
    LOAD_TEST_DURATION_MINUTES: int = Field(default=10, env="LOAD_TEST_DURATION_MINUTES")
    LOAD_TEST_CONCURRENT_PRODUCERS: int = Field(default=50, env="LOAD_TEST_CONCURRENT_PRODUCERS")
    LOAD_TEST_CONCURRENT_CONSUMERS: int = Field(default=25, env="LOAD_TEST_CONCURRENT_CONSUMERS")
    
    # Anthropic Claude API
    ANTHROPIC_API_KEY: str = Field(..., env="ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = Field(default="claude-3-5-sonnet-20241022", env="ANTHROPIC_MODEL")
    ANTHROPIC_MAX_TOKENS: int = Field(default=4096, env="ANTHROPIC_MAX_TOKENS")
    
    # GitHub Integration
    GITHUB_TOKEN: str = Field(..., env="GITHUB_TOKEN")
    GITHUB_APP_ID: Optional[str] = Field(None, env="GITHUB_APP_ID")
    GITHUB_PRIVATE_KEY: Optional[str] = Field(None, env="GITHUB_PRIVATE_KEY")
    WORK_TREES_BASE_PATH: str = Field(default="/tmp/agent-workspaces", env="WORK_TREES_BASE_PATH")
    BASE_URL: str = Field(..., env="BASE_URL")  # For webhook URLs
    
    # OpenAI API (for embeddings)
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL")
    OPENAI_EMBEDDING_MAX_TOKENS: int = Field(default=8191, env="OPENAI_EMBEDDING_MAX_TOKENS")
    OPENAI_EMBEDDING_CACHE_TTL: int = Field(default=3600, env="OPENAI_EMBEDDING_CACHE_TTL")
    OPENAI_EMBEDDING_MAX_RETRIES: int = Field(default=3, env="OPENAI_EMBEDDING_MAX_RETRIES")
    OPENAI_EMBEDDING_BASE_DELAY: float = Field(default=1.0, env="OPENAI_EMBEDDING_BASE_DELAY")
    OPENAI_EMBEDDING_MAX_DELAY: float = Field(default=60.0, env="OPENAI_EMBEDDING_MAX_DELAY")
    OPENAI_EMBEDDING_RATE_LIMIT_RPM: int = Field(default=3000, env="OPENAI_EMBEDDING_RATE_LIMIT_RPM")
    OPENAI_EMBEDDING_BATCH_SIZE: int = Field(default=100, env="OPENAI_EMBEDDING_BATCH_SIZE")
    
    # Security
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRE_MINUTES: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Multi-Agent Configuration
    MAX_CONCURRENT_AGENTS: int = Field(default=50, env="MAX_CONCURRENT_AGENTS")
    AGENT_HEARTBEAT_INTERVAL: int = Field(default=30, env="AGENT_HEARTBEAT_INTERVAL")  # seconds
    AGENT_TIMEOUT: int = Field(default=300, env="AGENT_TIMEOUT")  # seconds
    
    # Context Management
    CONTEXT_EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002", env="CONTEXT_EMBEDDING_MODEL")
    CONTEXT_MAX_TOKENS: int = Field(default=8000, env="CONTEXT_MAX_TOKENS")
    CONTEXT_COMPRESSION_THRESHOLD: float = Field(default=0.8, env="CONTEXT_COMPRESSION_THRESHOLD")
    
    # Sleep-Wake Cycles
    SLEEP_CYCLE_INTERVAL: int = Field(default=3600, env="SLEEP_CYCLE_INTERVAL")  # seconds
    CONSOLIDATION_THRESHOLD: float = Field(default=0.85, env="CONSOLIDATION_THRESHOLD")
    
    # Hook Lifecycle System Configuration
    HOOK_BATCH_SIZE: int = Field(default=100, env="HOOK_BATCH_SIZE")
    HOOK_FLUSH_INTERVAL_MS: int = Field(default=1000, env="HOOK_FLUSH_INTERVAL_MS")
    HOOK_PERFORMANCE_THRESHOLD_MS: float = Field(default=50.0, env="HOOK_PERFORMANCE_THRESHOLD_MS")
    HOOK_MAX_PAYLOAD_SIZE: int = Field(default=100000, env="HOOK_MAX_PAYLOAD_SIZE")
    HOOK_SECURITY_VALIDATION_ENABLED: bool = Field(default=True, env="HOOK_SECURITY_VALIDATION_ENABLED")
    HOOK_EVENT_AGGREGATION_ENABLED: bool = Field(default=True, env="HOOK_EVENT_AGGREGATION_ENABLED")
    HOOK_WEBSOCKET_STREAMING_ENABLED: bool = Field(default=True, env="HOOK_WEBSOCKET_STREAMING_ENABLED")
    HOOK_REDIS_STREAMING_ENABLED: bool = Field(default=True, env="HOOK_REDIS_STREAMING_ENABLED")
    HOOK_DANGEROUS_COMMAND_BLOCKING: bool = Field(default=True, env="HOOK_DANGEROUS_COMMAND_BLOCKING")
    
    # Self-Modification
    SELF_MODIFICATION_ENABLED: bool = Field(default=True, env="SELF_MODIFICATION_ENABLED")
    SAFE_MODIFICATION_ONLY: bool = Field(default=True, env="SAFE_MODIFICATION_ONLY")
    
    # tmux Integration
    TMUX_SESSION_NAME: str = Field(default="agent-hive", env="TMUX_SESSION_NAME")
    TMUX_AUTO_CREATE: bool = Field(default=True, env="TMUX_AUTO_CREATE")
    
    # Observability
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    GRAFANA_PORT: int = Field(default=3000, env="GRAFANA_PORT")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    
    # External Tools Configuration
    GITHUB_TOKEN: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    GITHUB_API_URL: str = Field(default="https://api.github.com", env="GITHUB_API_URL")
    DOCKER_HOST: Optional[str] = Field(default=None, env="DOCKER_HOST")
    DOCKER_REGISTRY: str = Field(default="docker.io", env="DOCKER_REGISTRY")
    
    # Git Configuration
    GIT_DEFAULT_BRANCH: str = Field(default="main", env="GIT_DEFAULT_BRANCH")
    GIT_USER_NAME: Optional[str] = Field(default=None, env="GIT_USER_NAME")
    GIT_USER_EMAIL: Optional[str] = Field(default=None, env="GIT_USER_EMAIL")
    
    # CI/CD Configuration  
    CI_CD_ENABLED: bool = Field(default=True, env="CI_CD_ENABLED")
    DEPLOYMENT_TIMEOUT: int = Field(default=1800, env="DEPLOYMENT_TIMEOUT")  # 30 minutes
    
    # Security for External Tools
    EXTERNAL_TOOLS_SECURITY_LEVEL: str = Field(default="moderate", env="EXTERNAL_TOOLS_SECURITY_LEVEL")
    ALLOW_SYSTEM_COMMANDS: bool = Field(default=False, env="ALLOW_SYSTEM_COMMANDS")
    
    # File Paths
    WORKSPACE_DIR: Path = Field(default=Path("./workspaces"), env="WORKSPACE_DIR")
    LOGS_DIR: Path = Field(default=Path("./logs"), env="LOGS_DIR")
    CHECKPOINTS_DIR: Path = Field(default=Path("./checkpoints"), env="CHECKPOINTS_DIR")
    REPOSITORIES_DIR: Path = Field(default=Path("./repositories"), env="REPOSITORIES_DIR")
    
    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("ALLOWED_HOSTS", mode="after") 
    @classmethod
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @field_validator("WORKSPACE_DIR", "LOGS_DIR", "CHECKPOINTS_DIR", "REPOSITORIES_DIR")
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings