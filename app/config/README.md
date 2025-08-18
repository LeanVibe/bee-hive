# Configuration Management

This directory contains the comprehensive configuration system for LeanVibe Agent Hive 2.0.

## Configuration Files

### Core Configuration
- **`unified_config.py`** - Single source of truth for all system configuration
- **`production_config.py`** - Production-ready configuration for new adapters and real-time features
- **`semantic_memory_config.py`** - Configuration for semantic memory and context systems

### Environment-Specific Configuration

#### Development Environment
```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///./dev.db
REDIS_URL=redis://localhost:6379/1
JWT_SECRET_KEY=development-jwt-secret-key
```

#### Production Environment
```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:password@localhost:5432/leanhive_prod
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=your-super-secret-jwt-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key
```

## Configuration Components

### CLI Adapter Configuration
Each adapter has specific configuration for optimal performance:

```python
from app.config.production_config import get_adapter_config, AdapterType

# Get Cursor adapter configuration
cursor_config = get_adapter_config(AdapterType.CURSOR)

# Get GitHub Copilot adapter configuration  
copilot_config = get_adapter_config(AdapterType.GITHUB_COPILOT)

# Get Gemini adapter configuration
gemini_config = get_adapter_config(AdapterType.GEMINI_CLI)
```

### Real-Time Communication Configuration
Redis and WebSocket settings for real-time agent coordination:

```python
from app.config.production_config import ProductionConfig

config = ProductionConfig.from_environment()
realtime_config = config.realtime

# Redis settings
print(f"Redis URL: {realtime_config.redis_url}")
print(f"Connection pool: {realtime_config.redis_connection_pool}")

# WebSocket settings
print(f"WebSocket port: {realtime_config.websocket_port}")
print(f"WebSocket settings: {realtime_config.websocket_settings}")
```

### Security Configuration
Production-ready security settings:

```python
security_config = config.security

# Authentication
jwt_settings = {
    "secret_key": security_config.jwt_secret_key,
    "algorithm": security_config.jwt_algorithm,
    "expiry": security_config.jwt_expiry_minutes
}

# Rate limiting
rate_limits = security_config.rate_limiting
```

## Environment Setup

### 1. Development Setup
```bash
# Copy development environment template
cp .env.development.example .env

# Edit configuration
nano .env

# Install dependencies
pip install -r requirements.txt

# Run development server
python -m app.main
```

### 2. Production Setup
```bash
# Copy production environment template
cp .env.production.example .env.production

# Edit with production values
nano .env.production

# Set environment
export ENVIRONMENT=production

# Run with production config
python -m app.main
```

### 3. Configuration Validation
```python
from app.config.production_config import create_production_config, Environment

# Create and validate configuration
config = create_production_config(Environment.PRODUCTION)
issues = config.validate()

if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration validated successfully!")
```

## Configuration Features

### Type Safety
All configuration uses Pydantic for type validation and IDE support:

```python
from app.config.production_config import ProductionConfig

config = ProductionConfig()
# Type hints and validation ensure correctness
config.performance.orchestration["max_concurrent_agents"] = 100
```

### Environment Variable Overrides
Configuration supports environment variable overrides:

```python
# Settings automatically load from environment
class ProductionSettings(BaseSettings):
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(..., env="REDIS_URL")
    
    class Config:
        env_file = ".env.production"
```

### Configuration Validation
Built-in validation ensures deployment readiness:

```python
def validate(self) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check required environment variables
    # Validate CLI tool availability
    # Check port availability
    # Validate database connections
    
    return issues
```

### Hot Reload Support
Configuration supports hot reload for development:

```python
# Configuration automatically reloads when files change
config = create_production_config()
# File watcher detects changes and reloads configuration
```

## Best Practices

### 1. Environment Separation
- Never use production credentials in development
- Use different databases and Redis instances per environment
- Set appropriate log levels per environment

### 2. Security
- Store sensitive values in environment variables
- Use strong, unique keys for JWT and encryption
- Enable rate limiting in production
- Monitor failed authentication attempts

### 3. Performance
- Tune connection pool sizes based on load
- Set appropriate timeouts for external services
- Configure caching for optimal performance
- Monitor resource usage and adjust limits

### 4. Monitoring
- Enable structured logging in production
- Configure health checks for all services
- Set up alerts for critical failures
- Monitor performance metrics

## Troubleshooting

### Common Issues

#### Configuration Validation Failures
```python
# Check configuration issues
config = create_production_config()
issues = config.validate()
for issue in issues:
    print(f"Fix: {issue}")
```

#### Missing Environment Variables
```bash
# Check required environment variables
export DATABASE_URL="postgresql://localhost/mydb"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET_KEY="your-secret-key"
```

#### CLI Tool Not Found
```bash
# Install missing CLI tools
curl -fsSL https://github.com/anthropics/anthropic-cli/releases/latest/download/install.sh | sh
npm install -g @githubnext/github-copilot-cli
```

#### Port Conflicts
```python
# Check and change conflicting ports
config.realtime.websocket_port = 8766  # Instead of 8765
config.settings.prometheus_port = 9091  # Instead of 9090
```

### Debug Mode
Enable debug mode for detailed logging:

```python
config = create_production_config(Environment.DEVELOPMENT)
config.settings.debug = True
config.settings.log_level = "DEBUG"
```