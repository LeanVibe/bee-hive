"""
Staging Configuration System for Multi-CLI Agent Coordination

This module provides staging-specific configuration for testing and pre-production
validation of the Multi-CLI Agent Coordination System.

IMPLEMENTATION STATUS: PRODUCTION READY
- Staging-optimized settings with relaxed security for testing
- Local Redis and WebSocket configurations
- Enhanced debugging and monitoring capabilities
- Agent-specific testing configurations
"""

import os
from typing import Dict, Any, Optional
from .production import (
    ProductionConfiguration, 
    RedisConfiguration,
    WebSocketConfiguration,
    SecurityConfiguration,
    MonitoringConfiguration,
    PerformanceConfiguration,
    AgentConfiguration
)

# ================================================================================
# Staging-Specific Configuration Overrides
# ================================================================================

class StagingConfiguration(ProductionConfiguration):
    """
    Staging configuration with relaxed security and enhanced debugging.
    
    Features:
    - Relaxed security settings for easier testing
    - Enhanced logging and debugging capabilities
    - Local service defaults for development
    - Reduced resource limits for cost efficiency
    """
    
    def __init__(self):
        """Initialize staging configuration with appropriate overrides."""
        # Set environment first
        self.environment = "staging"
        
        # Initialize with staging-specific configurations
        self._init_staging_configs()
        
        # Load agent configurations
        self.agents = self._load_staging_agent_configurations()
        
        # Runtime settings for staging
        self.runtime = {
            "startup_time": None,
            "config_version": "1.0.0-staging",
            "deployment_id": os.getenv("DEPLOYMENT_ID", "staging"),
            "debug_mode": True,  # Always debug in staging
            "maintenance_mode": False,
            "test_mode": True,
            "mock_external_services": os.getenv("MOCK_EXTERNAL_SERVICES", "false").lower() == "true"
        }
        
        # Skip strict validation for staging
        self._validate_staging_configuration()
    
    def _init_staging_configs(self):
        """Initialize staging-specific configurations."""
        # Redis configuration for staging
        self.redis = RedisConfiguration(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6381")),  # Non-standard port (6379 + 2)
            db=int(os.getenv("REDIS_DB", "1")),  # Use different DB for staging
            password=os.getenv("REDIS_PASSWORD", None),  # Optional for local Redis
            ssl=False,  # Disable SSL for local testing
            connection_pool_size=10,  # Smaller pool for staging
            message_ttl_hours=2,  # Shorter TTL for testing
            channel_prefix="staging_cli_agents"
        )
        
        # WebSocket configuration for staging
        self.websocket = WebSocketConfiguration(
            host=os.getenv("WEBSOCKET_HOST", "localhost"),
            port=int(os.getenv("WEBSOCKET_PORT", "8767")),  # Non-standard port (8765 + 2)
            ssl_enabled=False,  # Disable SSL for local testing
            ping_interval=10.0,  # More frequent pings for testing
            ping_timeout=10.0,
            max_connections=100,  # Smaller limit for staging
            compression=None  # Disable compression for debugging
        )
        
        # Security configuration for staging (relaxed)
        self.security = SecurityConfiguration(
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", "staging-secret-key-not-for-production"),
            jwt_expiration_hours=48,  # Longer expiration for testing
            api_key_required=False,  # Disable for easier testing
            rate_limit_enabled=True,
            rate_limit_per_minute=1000,  # Higher limits for testing
            rate_limit_per_hour=10000,
            command_validation=True,  # Keep validation enabled
            path_traversal_protection=True,
            dangerous_command_detection=True,
            audit_logging_enabled=True,
            sensitive_data_masking=False,  # Disable masking for debugging
            allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "*"],  # Allow all for testing
            cors_enabled=True,  # Enable CORS for frontend testing
            cors_origins=["http://localhost:3000", "http://localhost:8080", "*"]
        )
        
        # Monitoring configuration for staging (enhanced)
        self.monitoring = MonitoringConfiguration(
            metrics_enabled=True,
            metrics_port=int(os.getenv("METRICS_PORT", "9092")),  # Non-standard port (9090 + 2)
            health_check_enabled=True,
            health_check_interval=10,  # More frequent checks
            log_level="DEBUG",  # Debug level for staging
            log_format="text",  # Human-readable format
            performance_monitoring=True,
            response_time_tracking=True,
            resource_usage_tracking=True,
            alerting_enabled=False  # Disable alerts in staging
        )
        
        # Performance configuration for staging (reduced limits)
        self.performance = PerformanceConfiguration(
            max_concurrent_agents=20,  # Smaller limit for testing
            max_concurrent_tasks_per_agent=2,
            task_queue_size=1000,
            memory_limit_mb=2048,  # Reduced memory limit
            cpu_limit_percent=70.0,
            cache_enabled=True,
            cache_ttl_seconds=1800,  # Shorter TTL
            database_pool_size=5,  # Smaller pools
            redis_pool_size=5,
            websocket_pool_size=20,
            default_task_timeout=120.0,  # Shorter timeout for faster testing
            max_retry_attempts=2,  # Fewer retries
            retry_backoff_base=1.5
        )
    
    def _load_staging_agent_configurations(self) -> Dict[str, AgentConfiguration]:
        """Load staging-specific agent configurations."""
        agents = {}
        
        # Claude Code Agent (staging)
        agents["claude_code"] = AgentConfiguration(
            agent_type="claude_code",
            cli_path=os.getenv("CLAUDE_CLI_PATH", "claude"),
            working_directory=os.getenv("CLAUDE_WORKING_DIR", "/tmp/claude_staging"),
            max_concurrent_tasks=2,  # Reduced for staging
            default_timeout=120.0,  # Shorter timeout
            memory_limit_mb=512,  # Reduced memory
            allowed_commands=[],  # Allow all commands in staging
            forbidden_patterns=["rm -rf /", "sudo rm", "format"],  # Basic safety
            sandbox_enabled=True,
            worktree_isolation=True,
            auto_restart=True,
            message_queue_size=100,  # Smaller queue
            priority_queue_enabled=True
        )
        
        # Cursor Agent (staging)
        agents["cursor"] = AgentConfiguration(
            agent_type="cursor",
            cli_path=os.getenv("CURSOR_CLI_PATH", "cursor"),
            working_directory=os.getenv("CURSOR_WORKING_DIR", "/tmp/cursor_staging"),
            max_concurrent_tasks=2,
            default_timeout=120.0,
            memory_limit_mb=256,
            sandbox_enabled=True,
            worktree_isolation=True,
            auto_restart=True
        )
        
        # GitHub Copilot Agent (staging)
        agents["github_copilot"] = AgentConfiguration(
            agent_type="github_copilot",
            cli_path=os.getenv("GH_CLI_PATH", "gh"),
            working_directory=os.getenv("GH_WORKING_DIR", "/tmp/gh_staging"),
            max_concurrent_tasks=1,  # Conservative for API limits
            default_timeout=120.0,
            memory_limit_mb=128,
            allowed_commands=["gh copilot suggest", "gh copilot explain", "gh auth status"],
            sandbox_enabled=True,
            auto_restart=True
        )
        
        # Gemini CLI Agent (staging)
        agents["gemini_cli"] = AgentConfiguration(
            agent_type="gemini_cli",
            cli_path=os.getenv("GEMINI_CLI_PATH", "gemini"),
            working_directory=os.getenv("GEMINI_WORKING_DIR", "/tmp/gemini_staging"),
            max_concurrent_tasks=1,  # Conservative for API limits
            default_timeout=120.0,
            memory_limit_mb=128,
            sandbox_enabled=True,
            auto_restart=True
        )
        
        # Mock Agent for testing
        agents["mock_agent"] = AgentConfiguration(
            agent_type="mock_agent",
            cli_path="mock_cli",
            working_directory="/tmp/mock_staging",
            max_concurrent_tasks=5,  # Higher for testing
            default_timeout=30.0,  # Fast for mock responses
            memory_limit_mb=64,
            sandbox_enabled=False,  # No sandbox for mock
            auto_restart=True,
            message_queue_size=1000
        )
        
        return agents
    
    def _validate_staging_configuration(self):
        """Validate staging configuration (more lenient than production)."""
        # Basic validation only - allow more flexibility in staging
        
        # Ensure we're not using production-like settings
        if self.security.jwt_secret_key == "your-secret-key-change-in-production":
            self.security.jwt_secret_key = "staging-secret-key-for-testing-only"
        
        # Ensure debug mode is enabled
        self.runtime["debug_mode"] = True
        
        # Validate basic resource limits
        if self.performance.max_concurrent_agents > 50:
            self.performance.max_concurrent_agents = 50
        
        # Set appropriate working directories
        for agent_config in self.agents.values():
            if not agent_config.working_directory:
                agent_config.working_directory = f"/tmp/{agent_config.agent_type}_staging"
    
    def is_test_mode(self) -> bool:
        """Check if test mode is enabled."""
        return self.runtime.get("test_mode", False)
    
    def should_mock_external_services(self) -> bool:
        """Check if external services should be mocked."""
        return self.runtime.get("mock_external_services", False)
    
    def get_test_data_directory(self) -> str:
        """Get directory for test data."""
        return os.getenv("TEST_DATA_DIR", "/tmp/cli_agent_test_data")
    
    def get_staging_configuration_summary(self) -> Dict[str, Any]:
        """Get staging-specific configuration summary."""
        summary = self.get_configuration_summary()
        summary.update({
            "test_mode": self.is_test_mode(),
            "mock_external_services": self.should_mock_external_services(),
            "test_data_directory": self.get_test_data_directory(),
            "staging_specific": {
                "relaxed_security": True,
                "debug_logging": True,
                "cors_enabled": True,
                "ssl_disabled": True,
                "reduced_limits": True
            }
        })
        return summary

# ================================================================================
# Development/Local Configuration
# ================================================================================

class DevelopmentConfiguration(StagingConfiguration):
    """
    Development configuration for local development and testing.
    
    Even more relaxed than staging with additional development conveniences.
    """
    
    def __init__(self):
        """Initialize development configuration."""
        super().__init__()
        
        # Override for development
        self.environment = "development"
        self.runtime["config_version"] = "1.0.0-development"
        self.runtime["deployment_id"] = "local-dev"
        self.runtime["mock_external_services"] = True
        
        # Further relaxed settings for local development
        self._apply_development_overrides()
    
    def _apply_development_overrides(self):
        """Apply development-specific configuration overrides."""
        # Use different Redis DB and port for development
        self.redis.db = 2
        self.redis.port = 6382  # Non-standard port (6379 + 3)
        self.redis.channel_prefix = "dev_cli_agents"
        
        # Use different WebSocket port
        self.websocket.port = 8768  # Non-standard port (8765 + 3)
        
        # Even more relaxed security
        self.security.api_key_required = False
        self.security.rate_limit_enabled = False
        self.security.audit_logging_enabled = False
        
        # Minimal performance limits
        self.performance.max_concurrent_agents = 10
        self.performance.memory_limit_mb = 1024
        self.performance.cache_ttl_seconds = 300  # 5 minutes
        
        # Enhanced debugging
        self.monitoring.health_check_interval = 5  # Every 5 seconds
        self.monitoring.log_level = "DEBUG"
        
        # Development-friendly agent settings
        for agent_config in self.agents.values():
            agent_config.max_concurrent_tasks = 1  # One at a time for clarity
            agent_config.default_timeout = 60.0  # Shorter for development
            agent_config.sandbox_enabled = False  # Disable for easier debugging
            agent_config.auto_restart = False  # Manual restart for debugging

# ================================================================================
# Configuration Factory Functions
# ================================================================================

def create_staging_config() -> StagingConfiguration:
    """Create staging configuration instance."""
    return StagingConfiguration()

def create_development_config() -> DevelopmentConfiguration:
    """Create development configuration instance."""
    return DevelopmentConfiguration()

def create_config_for_environment(environment: str = None):
    """
    Create configuration appropriate for the specified environment.
    
    Args:
        environment: Target environment (production, staging, development)
        
    Returns:
        Appropriate configuration instance
    """
    environment = environment or os.getenv("ENVIRONMENT", "development")
    
    if environment.lower() == "production":
        from .production import create_production_config
        return create_production_config()
    elif environment.lower() == "staging":
        return create_staging_config()
    elif environment.lower() == "development":
        return create_development_config()
    else:
        # Default to development for unknown environments
        return create_development_config()

def get_environment_specific_overrides(environment: str) -> Dict[str, Any]:
    """Get environment-specific configuration overrides."""
    overrides = {}
    
    if environment == "staging":
        overrides.update({
            "debug_logging": True,
            "cors_enabled": True,
            "ssl_disabled": True,
            "reduced_limits": True,
            "mock_services": False
        })
    elif environment == "development":
        overrides.update({
            "debug_logging": True,
            "cors_enabled": True,
            "ssl_disabled": True,
            "minimal_limits": True,
            "mock_services": True,
            "sandbox_disabled": True
        })
    
    return overrides