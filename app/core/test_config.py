"""
Test-specific configuration for LeanVibe Agent Hive 2.0

Provides isolated, clean configuration for testing without pydantic validation conflicts.
"""

from typing import List, Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class TestSettings(BaseSettings):
    """Minimal test configuration without validation conflicts."""
    
    # Application
    APP_NAME: str = "LeanVibe Agent Hive 2.0 Test"
    ENVIRONMENT: str = "testing"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    SECRET_KEY: str = "test-secret-key-for-testing-purposes-only"
    
    # JWT Authentication
    JWT_SECRET_KEY: str = "test-jwt-secret-key-for-testing-purposes-only"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///:memory:"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/1"
    REDIS_STREAM_MAX_LEN: int = 1000
    
    # Message Processing
    MAX_MESSAGE_SIZE_BYTES: int = 1024*512  # 512KB for testing
    
    # Orchestrator Configuration
    USE_SIMPLE_ORCHESTRATOR: bool = True
    MAX_CONCURRENT_AGENTS: int = 3  # Lower for testing
    ORCHESTRATOR_TYPE: str = "simple"
    
    # Performance optimizations (reduced for testing)
    REDIS_CONNECTION_POOL_SIZE: int = 5
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_CONNECTION_TIMEOUT: float = 2.0
    
    # Message Batching Configuration
    MESSAGE_BATCH_SIZE: int = 10
    MESSAGE_BATCH_WAIT_MS: int = 10
    ADAPTIVE_BATCHING_ENABLED: bool = False
    BATCH_TARGET_LATENCY_MS: float = 50.0
    
    # Payload Compression (disabled for testing)
    COMPRESSION_ENABLED: bool = False
    COMPRESSION_ALGORITHM: str = "none"
    
    # DLQ Configuration (minimal for testing)
    DLQ_MAX_RETRIES: int = 1
    DLQ_INITIAL_RETRY_DELAY_MS: int = 100
    DLQ_MAX_RETRY_DELAY_MS: int = 1000
    DLQ_MAX_SIZE: int = 100
    DLQ_TTL_HOURS: int = 1
    DLQ_POLICY: str = "immediate"
    
    # Back-pressure Management (disabled for testing)
    BACKPRESSURE_ENABLED: bool = False
    BACKPRESSURE_WARNING_LAG: int = 100
    BACKPRESSURE_CRITICAL_LAG: int = 500
    BACKPRESSURE_EMERGENCY_LAG: int = 1000
    CONSUMER_MIN_COUNT: int = 1
    CONSUMER_MAX_COUNT: int = 3
    CONSUMER_SCALE_UP_THRESHOLD: float = 0.8
    CONSUMER_SCALE_DOWN_THRESHOLD: float = 0.3
    
    # Circuit Breaker Configuration (disabled for testing)
    CIRCUIT_BREAKER_ENABLED: bool = False
    
    # Stream Monitoring (minimal for testing)
    STREAM_MONITORING_ENABLED: bool = False
    PROMETHEUS_METRICS_ENABLED: bool = False
    
    # Performance Targets (relaxed for testing)
    TARGET_THROUGHPUT_MSG_PER_SEC: int = 100
    TARGET_P95_LATENCY_MS: float = 100.0
    TARGET_P99_LATENCY_MS: float = 200.0
    TARGET_SUCCESS_RATE: float = 0.95
    
    # Load Testing (disabled)
    LOAD_TEST_ENABLED: bool = False
    
    # Sandbox Mode
    SANDBOX_MODE: bool = True
    SANDBOX_DEMO_MODE: bool = True
    
    # API Keys (test values)
    ANTHROPIC_API_KEY: Optional[str] = "test-anthropic-key"
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
    ANTHROPIC_MAX_TOKENS: int = 1000
    
    # GitHub Integration (mocked for testing)
    GITHUB_TOKEN: Optional[str] = "test-github-token"
    GITHUB_USERNAME: Optional[str] = "test-user"
    GITHUB_REPO_OWNER: Optional[str] = "test-owner"
    GITHUB_REPO_NAME: Optional[str] = "test-repo"
    
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 18080
    API_PREFIX: str = "/api"
    
    # Security
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # Directories (use temp directories for testing)
    WORKSPACE_DIR: Path = Path("/tmp/test_workspace")
    LOGS_DIR: Path = Path("/tmp/test_logs")
    CHECKPOINTS_DIR: Path = Path("/tmp/test_checkpoints")
    REPOSITORIES_DIR: Path = Path("/tmp/test_repositories")
    
    # Additional test-specific settings
    TEST_WITH_POSTGRES: bool = False
    TEST_WITH_REDIS: bool = False
    SKIP_DB_MIGRATION: bool = True
    MOCK_EXTERNAL_APIS: bool = True
    
    model_config = {
        "case_sensitive": True,
        "env_prefix": "",
        "validate_assignment": True,
    }


def get_test_settings() -> TestSettings:
    """Get test settings instance."""
    return TestSettings()


# Create global test settings instance with error handling
def create_test_settings():
    """Create test settings with fallback for pydantic issues."""
    try:
        return TestSettings()
    except Exception:
        # Fallback to simple namespace if pydantic validation fails
        from types import SimpleNamespace
        settings = SimpleNamespace()
        settings.APP_NAME = "LeanVibe Agent Hive 2.0 Test"
        settings.ENVIRONMENT = "testing"
        settings.DEBUG = True
        settings.LOG_LEVEL = "DEBUG"
        settings.SECRET_KEY = "test-secret-key-for-testing-purposes-only"
        settings.JWT_SECRET_KEY = "test-jwt-secret-key-for-testing-purposes-only"
        settings.JWT_ALGORITHM = "HS256"
        settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
        settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
        settings.DATABASE_POOL_SIZE = 5
        settings.DATABASE_MAX_OVERFLOW = 10
        settings.REDIS_URL = "redis://localhost:6379/1"
        settings.REDIS_STREAM_MAX_LEN = 1000
        settings.MAX_MESSAGE_SIZE_BYTES = 1024*512
        settings.USE_SIMPLE_ORCHESTRATOR = True
        settings.MAX_CONCURRENT_AGENTS = 3
        settings.ORCHESTRATOR_TYPE = "simple"
        # Add all other required fields
        return settings

test_settings = create_test_settings()