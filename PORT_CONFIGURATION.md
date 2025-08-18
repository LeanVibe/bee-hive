# Port Configuration for Multi-CLI Agent Coordination System

## 🔌 **Non-Standard Port Strategy**

To avoid conflicts with other projects running on the same machine, all services use **non-standard ports** by default:

## 📊 **Port Allocation by Environment**

| Service | Standard Port | Production | Staging | Development | Testing |
|---------|---------------|------------|---------|-------------|---------|
| **Redis** | 6379 | 6380 | 6381 | 6382 | 6383-6390 |
| **WebSocket** | 8765 | 8766 | 8767 | 8768 | 8769-8780 |
| **Prometheus** | 9090 | 9091 | 9092 | 9093 | 9094-9100 |

## 🛠️ **Environment Configuration**

### **Production Environment**
```bash
export REDIS_PORT="6380"          # Redis (default 6379 + 1)
export WEBSOCKET_PORT="8766"      # WebSocket (default 8765 + 1) 
export METRICS_PORT="9091"        # Prometheus (default 9090 + 1)
```

### **Staging Environment**
```bash
export REDIS_PORT="6381"          # Redis (default 6379 + 2)
export WEBSOCKET_PORT="8767"      # WebSocket (default 8765 + 2)
export METRICS_PORT="9092"        # Prometheus (default 9090 + 2)
```

### **Development Environment**
```bash
export REDIS_PORT="6382"          # Redis (default 6379 + 3)
export WEBSOCKET_PORT="8768"      # WebSocket (default 8765 + 3)
export METRICS_PORT="9093"        # Prometheus (default 9090 + 3)
```

### **Testing Environment**
```bash
export REDIS_PORT="6383"          # Redis test instances (6383-6390)
export WEBSOCKET_PORT="8769"      # WebSocket test instances (8769-8780)
export METRICS_PORT="9094"        # Prometheus test instances (9094-9100)
```

## 🔧 **Easy Port Customization**

All ports are easily configurable through environment variables:

```bash
# Custom Redis setup
export REDIS_HOST="my-redis-server.com"
export REDIS_PORT="6400"          # Custom port
export REDIS_PASSWORD="secure-password"

# Custom WebSocket setup
export WEBSOCKET_HOST="0.0.0.0"
export WEBSOCKET_PORT="9000"      # Custom port
export WEBSOCKET_SSL_ENABLED="true"

# Custom Prometheus setup
export METRICS_PORT="8080"        # Custom port
export METRICS_PATH="/custom-metrics"
```

## 🐳 **Docker Compose Configuration**

Example `docker-compose.yml` for development:

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6382:6379"  # Map container 6379 to host 6382
    command: redis-server --requirepass mypassword
  
  cli-coordinator:
    build: .
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379       # Container internal port
      - REDIS_PASSWORD=mypassword
      - WEBSOCKET_PORT=8768
      - METRICS_PORT=9093
    ports:
      - "8768:8768"           # WebSocket
      - "9093:9093"           # Prometheus metrics
    depends_on:
      - redis
```

## 🧪 **Testing Port Management**

For parallel test execution, ports are automatically assigned:

```python
# Test configuration with dynamic port assignment
def get_test_config(test_id: str):
    base_redis_port = 6383
    base_websocket_port = 8769
    
    return {
        "redis_port": base_redis_port + hash(test_id) % 10,
        "websocket_port": base_websocket_port + hash(test_id) % 10,
        "metrics_port": 9094 + hash(test_id) % 10
    }
```

## 🔍 **Port Conflict Detection**

Built-in port availability checking:

```python
import socket

def is_port_available(host: str, port: int) -> bool:
    """Check if port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False

def find_available_port(base_port: int, max_attempts: int = 100) -> int:
    """Find next available port starting from base_port."""
    for i in range(max_attempts):
        port = base_port + i
        if is_port_available("localhost", port):
            return port
    raise RuntimeError(f"No available port found starting from {base_port}")
```

## 🚦 **Port Health Monitoring**

```python
async def check_service_health():
    """Check if all services are running on configured ports."""
    services = {
        "Redis": (config.redis.host, config.redis.port),
        "WebSocket": (config.websocket.host, config.websocket.port),
        "Prometheus": ("localhost", config.monitoring.metrics_port)
    }
    
    for service, (host, port) in services.items():
        if not is_port_available(host, port):
            print(f"✅ {service} running on {host}:{port}")
        else:
            print(f"❌ {service} not running on {host}:{port}")
```

## 🔧 **Quick Start Commands**

```bash
# Check port availability
netstat -an | grep :6380    # Redis production
netstat -an | grep :8766    # WebSocket production

# Kill services on specific ports (if needed)
lsof -ti:6380 | xargs kill  # Kill Redis on production port
lsof -ti:8766 | xargs kill  # Kill WebSocket on production port

# Start services with custom ports
REDIS_PORT=6400 WEBSOCKET_PORT=9000 python -m app.main

# Test port configuration
python -c "
from app.config.production import get_config
config = get_config()
print(f'Redis: {config.redis.host}:{config.redis.port}')
print(f'WebSocket: {config.websocket.host}:{config.websocket.port}')
print(f'Metrics: localhost:{config.monitoring.metrics_port}')
"
```

## ⚠️ **Important Notes**

1. **Firewall Rules**: Ensure firewall allows traffic on custom ports
2. **Load Balancers**: Update load balancer configs when changing ports
3. **Monitoring**: Update monitoring tools to check custom ports
4. **Documentation**: Keep port documentation up-to-date in deployment guides
5. **Environment Sync**: Ensure all team members use same port configurations

## 🎯 **Benefits of Non-Standard Ports**

- ✅ **Avoid Conflicts**: No conflicts with Redis (6379), WebSocket (8765), Prometheus (9090)
- ✅ **Easy Identification**: Non-standard ports make services easily identifiable
- ✅ **Environment Isolation**: Different ports for prod/staging/dev prevent cross-environment issues
- ✅ **Parallel Testing**: Multiple test instances can run simultaneously
- ✅ **Security**: Less obvious ports reduce automated attack vectors
- ✅ **Flexibility**: Easy to change ports without affecting other services