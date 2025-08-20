import asyncio
"""
Production Configuration Extension for LeanVibe Agent Hive 2.0
Enhanced configuration for new CLI adapters and real-time communication

This module extends the unified configuration system with production-ready settings
for the newly implemented components:
- Cursor CLI Adapter
- GitHub Copilot CLI Adapter  
- Gemini CLI Adapter
- Redis/WebSocket real-time communication
- Production deployment settings
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from .unified_config import Environment, UnifiedConfig


class AdapterType(str, Enum):
    """Supported CLI adapter types."""
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    GITHUB_COPILOT = "github_copilot"
    GEMINI_CLI = "gemini_cli"


@dataclass
class CLIAdapterConfig:
    """Base configuration for CLI adapters."""
    enabled: bool = True
    cli_path: Optional[str] = None
    working_directory: str = "/tmp/agent_workspace"
    timeout_seconds: int = 300
    max_concurrent_tasks: int = 5
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "memory_mb": 1024,
        "cpu_percent": 50,
        "disk_mb": 500
    })
    security_settings: Dict[str, Any] = field(default_factory=lambda: {
        "allow_file_access": True,
        "allowed_directories": ["/tmp", "/var/tmp"],
        "forbidden_commands": ["rm -rf", "dd", "mkfs", "fdisk"],
        "path_traversal_protection": True
    })


@dataclass
class CursorAdapterConfig(CLIAdapterConfig):
    """Cursor CLI adapter specific configuration."""
    cli_path: str = "cursor"
    ai_assist_enabled: bool = True
    smart_completions: bool = True
    project_scope_analysis: bool = True
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "code_implementation": 0.98,
        "refactoring": 0.95,
        "ui_development": 0.92,
        "code_analysis": 0.92,
        "debugging": 0.90
    })


@dataclass
class GitHubCopilotAdapterConfig(CLIAdapterConfig):
    """GitHub Copilot CLI adapter specific configuration."""
    cli_path: str = "gh"
    max_concurrent_tasks: int = 2  # Lower due to API rate limits
    api_settings: Dict[str, Any] = field(default_factory=lambda: {
        "rate_limit_per_minute": 30,
        "request_timeout": 30,
        "max_retries": 3,
        "backoff_factor": 2
    })
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "code_implementation": 0.85,
        "testing": 0.80,
        "code_review": 0.75,
        "debugging": 0.70,
        "documentation": 0.80
    })


@dataclass
class GeminiAdapterConfig(CLIAdapterConfig):
    """Gemini CLI adapter specific configuration."""
    api_key_env: str = "GEMINI_API_KEY"
    model_name: str = "gemini-pro"
    deep_thinking_mode: bool = True
    multimodal_analysis: bool = True
    token_allocation: Dict[str, int] = field(default_factory=lambda: {
        "simple_tasks": 4000,
        "complex_tasks": 6000,
        "max_tokens": 8000
    })
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "testing": 0.90,
        "code_analysis": 0.90,
        "security_analysis": 0.88,
        "architecture_design": 0.85,
        "performance_optimization": 0.80
    })


@dataclass
class AdapterConfigs:
    """Container for all CLI adapter configurations."""
    cursor: CursorAdapterConfig = field(default_factory=CursorAdapterConfig)
    github_copilot: GitHubCopilotAdapterConfig = field(default_factory=GitHubCopilotAdapterConfig)
    gemini: GeminiAdapterConfig = field(default_factory=GeminiAdapterConfig)


@dataclass
class RealTimeCommunicationConfig:
    """Configuration for real-time communication systems."""
    
    # Redis configuration
    redis_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    redis_connection_pool: Dict[str, Any] = field(default_factory=lambda: {
        "max_connections": 100,
        "retry_on_timeout": True,
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "socket_keepalive": True,
        "socket_keepalive_options": {}
    })
    
    # WebSocket configuration
    websocket_enabled: bool = True
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    websocket_settings: Dict[str, Any] = field(default_factory=lambda: {
        "max_size": 1024 * 1024,  # 1MB
        "compression": "deflate",
        "heartbeat_interval": 30,
        "close_timeout": 10
    })
    
    # Message routing
    message_routing: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_retries": 3,
        "retry_delay": 1.0,
        "dead_letter_queue": True,
        "message_ttl": 3600  # 1 hour
    })
    
    # Performance settings
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 100,
        "flush_interval": 5,
        "connection_timeout": 30,
        "max_concurrent_connections": 1000
    })


@dataclass
class SecurityConfig:
    """Enhanced security configuration for production."""
    
    # Authentication
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30
    
    # API keys management
    api_keys: Dict[str, str] = field(default_factory=dict)
    encryption_enabled: bool = True
    encryption_key: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    
    # Rate limiting
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "requests_per_minute": 100,
        "burst_size": 20,
        "cleanup_interval": 60
    })
    
    # Security monitoring
    security_monitoring: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "log_failed_attempts": True,
        "max_failed_attempts": 5,
        "lockout_duration": 300  # 5 minutes
    })


@dataclass
class PerformanceConfig:
    """Performance optimization settings."""
    
    # Agent orchestration
    orchestration: Dict[str, Any] = field(default_factory=lambda: {
        "max_concurrent_agents": 50,
        "task_timeout": 300,
        "agent_registration_timeout": 10,
        "health_check_interval": 30
    })
    
    # Resource limits
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": 2048,
        "max_cpu_percent": 80,
        "max_disk_mb": 1024,
        "max_network_connections": 100
    })
    
    # Caching
    caching: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "cache_ttl": 3600,
        "max_cache_size": "100MB",
        "cache_compression": True
    })


class ProductionSettings(BaseSettings):
    """Production environment settings with validation."""
    
    # Environment
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: str = "INFO"
    
    # Core application
    app_name: str = "LeanVibe Agent Hive 2.0"
    version: str = "2.0.0"
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = 50
    database_max_overflow: int = 100
    
    # Redis
    redis_url: str = Field(..., env="REDIS_URL")
    
    # External services
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    
    class Config:
        env_file = ".env.production"
        env_file_encoding = "utf-8"
        case_sensitive = True


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    
    # Core settings
    settings: ProductionSettings = field(default_factory=ProductionSettings)
    
    # Adapter configurations
    adapters: AdapterConfigs = field(default_factory=AdapterConfigs)
    
    # Real-time communication
    realtime: RealTimeCommunicationConfig = field(default_factory=RealTimeCommunicationConfig)
    
    # Security
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Performance
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "settings": self.settings.dict(),
            "adapters": {
                "cursor": self.adapters.cursor.__dict__,
                "github_copilot": self.adapters.github_copilot.__dict__,
                "gemini": self.adapters.gemini.__dict__
            },
            "realtime": self.realtime.__dict__,
            "security": self.security.__dict__,
            "performance": self.performance.__dict__
        }
    
    @classmethod
    def from_environment(cls, env: Environment = Environment.PRODUCTION) -> "ProductionConfig":
        """Create configuration from environment."""
        settings = ProductionSettings()
        
        # Adjust settings based on environment
        if env == Environment.DEVELOPMENT:
            settings.debug = True
            settings.log_level = "DEBUG"
        elif env == Environment.STAGING:
            settings.log_level = "INFO"
        
        return cls(settings=settings)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate required environment variables
        required_vars = ["DATABASE_URL", "REDIS_URL", "JWT_SECRET_KEY"]
        for var in required_vars:
            if not os.getenv(var):
                issues.append(f"Missing required environment variable: {var}")
        
        # Validate adapter CLI paths
        for adapter_name, adapter_config in [
            ("cursor", self.adapters.cursor),
            ("github_copilot", self.adapters.github_copilot),
            ("gemini", self.adapters.gemini)
        ]:
            if adapter_config.enabled and adapter_config.cli_path:
                from shutil import which
                if not which(adapter_config.cli_path):
                    issues.append(f"{adapter_name} CLI not found: {adapter_config.cli_path}")
        
        # Validate ports are available
        import socket
        ports_to_check = [
            self.realtime.websocket_port,
            self.settings.prometheus_port
        ]
        
        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    issues.append(f"Port {port} is already in use")
            finally:
                sock.close()
        
        return issues


def create_production_config(
    environment: Environment = Environment.PRODUCTION,
    config_overrides: Optional[Dict[str, Any]] = None
) -> ProductionConfig:
    """Factory function to create production configuration."""
    config = ProductionConfig.from_environment(environment)
    
    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Validate configuration
    issues = config.validate()
    if issues:
        import warnings
        for issue in issues:
            warnings.warn(f"Configuration issue: {issue}")
    
    return config


def get_adapter_config(
    adapter_type: AdapterType,
    config: Optional[ProductionConfig] = None
) -> CLIAdapterConfig:
    """Get configuration for specific adapter type."""
    if config is None:
        config = create_production_config()
    
    adapter_map = {
        AdapterType.CURSOR: config.adapters.cursor,
        AdapterType.GITHUB_COPILOT: config.adapters.github_copilot,
        AdapterType.GEMINI_CLI: config.adapters.gemini
    }
    
    return adapter_map.get(adapter_type)


# Example configuration files
EXAMPLE_ENV_PRODUCTION = """
# Production Environment Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/leanhive_prod

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-here
ENCRYPTION_KEY=your-encryption-key-here

# API Keys
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key

# Monitoring
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
"""

EXAMPLE_ENV_DEVELOPMENT = """
# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database (SQLite for development)
DATABASE_URL=sqlite:///./dev.db

# Redis
REDIS_URL=redis://localhost:6379/1

# Security (development keys - change in production!)
JWT_SECRET_KEY=development-jwt-secret-key
ENCRYPTION_KEY=development-encryption-key

# API Keys (optional for development)
# ANTHROPIC_API_KEY=your-dev-anthropic-key
# OPENAI_API_KEY=your-dev-openai-key
# GEMINI_API_KEY=your-dev-gemini-key
"""

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class ProductionConfigScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            # Example usage and validation
            config = create_production_config(Environment.DEVELOPMENT)
            self.logger.info("Development configuration created successfully")
            self.logger.info(f"Configuration validation issues: {config.validate()}")
            
            return {"status": "completed"}
    
    script_main(ProductionConfigScript)