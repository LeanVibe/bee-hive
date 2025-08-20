# 🔌 LeanVibe Agent Hive - Port Configuration

## 🎯 **Non-Standard Port Strategy**

LeanVibe Agent Hive uses **non-standard ports** to avoid conflicts with common development tools and system services. This ensures smooth operation alongside:

- React/Next.js dev servers (port 3000)
- Vite dev servers (port 5173) 
- Django dev servers (port 8000)
- System PostgreSQL (port 5432)
- System Redis (port 6379)
- Docker services
- Other development tools

## 📋 **Port Mapping Reference**

### **Core Services**
| Service | Standard Port | Agent Hive Port | Purpose |
|---------|---------------|-----------------|---------|
| **FastAPI Backend** | 8000 | **18080** | Main API server |
| **PWA Development** | 5173 | **18443** | Frontend dev server |
| **PWA Preview** | 4173 | **18444** | Production preview |
| **PostgreSQL** | 5432 | **15432** | Database server |
| **Redis** | 6379 | **16379** | Cache & messaging |

### **Monitoring & Observability**
| Service | Standard Port | Agent Hive Port | Purpose |
|---------|---------------|-----------------|---------|
| **Prometheus Metrics** | 9090 | **19090** | Monitoring |
| **Health Checks** | 8080 | **18081** | Status monitoring |
| **WebSocket** | 8080 | **18082** | Real-time updates |

### **Development & Debugging**
| Service | Standard Port | Agent Hive Port | Purpose |
|---------|---------------|-----------------|---------|
| **Hot Reload** | 8000 | **18083** | Development server |
| **Debug Interface** | 8080 | **18084** | Profiling & debugging |
| **API Docs** | 8080 | **18085** | Separate docs server |

### **Testing & Mocking**
| Service | Standard Port | Agent Hive Port | Purpose |
|---------|---------------|-----------------|---------|
| **Mock Anthropic** | - | **18090** | Testing without real AI API |
| **Mock OpenAI** | - | **18091** | Testing without real AI API |

## 🚀 **Quick Access URLs**

### **Development URLs**
```bash
# Main services
API Server:        http://localhost:18080
API Documentation: http://localhost:18080/docs
API Health:        http://localhost:18080/health
PWA Dashboard:     http://localhost:18443

# Database connections
PostgreSQL:        postgresql://user:pass@localhost:15432/db
Redis:             redis://localhost:16379/0
```

### **CLI Commands**
```bash
# System management
hive start          # Starts API on port 18080
hive dashboard      # Opens PWA on port 18443  
hive doctor         # Checks all port status
hive status         # System health on port 18080

# Service management
docker-compose up postgres redis  # Uses ports 15432, 16379
cd mobile-pwa && npm run dev      # Starts PWA on port 18443
```

## ⚙️ **Configuration Files**

### **Environment Variables**
The port configuration is defined in multiple files:

**Primary Configuration: `.env`**
```env
# Core service ports
API_PORT=18080
PWA_DEV_PORT=18443
PWA_PREVIEW_PORT=18444
PROMETHEUS_PORT=19090

# Database ports  
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:15432/db
REDIS_URL=redis://localhost:16379/0

# CORS origins
CORS_ORIGINS=http://localhost:18080,http://localhost:18443,http://localhost:18444
```

**Detailed Configuration: `.env.ports`**
- Complete port mapping for all services
- Port range reservations (18000-18999)
- Conflict detection settings
- Enterprise feature ports

**Docker Compose: `docker-compose.yml`**
```yaml
# PostgreSQL
ports:
  - "${POSTGRES_PORT:-15432}:5432"

# Redis  
ports:
  - "${REDIS_PORT:-16379}:6379"
```

**PWA Configuration: `mobile-pwa/vite.config.ts`**
```typescript
server: {
  port: Number(process.env.PWA_DEV_PORT || 18443),
  proxy: {
    '/api': {
      target: 'http://localhost:18080'
    }
  }
}
```

## 🔧 **Port Conflict Resolution**

### **Automatic Detection**
The CLI includes automatic port conflict detection:

```bash
hive doctor  # Shows port status for all services
```

**Sample Output:**
```
🔌 Port Status:
  Port 18080 (API Server): 🟢 Available
  Port 15432 (PostgreSQL): 🔴 In use  
  Port 16379 (Redis): 🟢 Available
  Port 18443 (PWA Dev Server): 🟢 Available
```

### **Manual Port Changes**
To customize ports, update the `.env` file:

```env
# Change API port if 18080 conflicts
API_PORT=18088

# Change database port if 15432 conflicts  
POSTGRES_PORT=15433
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:15433/db
```

Then restart services:
```bash
hive stop
hive start
```

### **Port Range Strategy**
Agent Hive reserves the **18000-18999** port range:

- **18000-18099**: Core services (API, PWA, databases)
- **18100-18199**: Monitoring & observability
- **18200-18299**: Enterprise features
- **18300-18399**: Development tools
- **18400-18499**: Testing & mocking
- **18500-18599**: Service discovery
- **18600-18699**: Dynamic allocation
- **18700-18999**: Reserved for future expansion

## 🛡️ **Security Considerations**

### **Development vs Production**
```env
# Development - localhost only
BIND_HOST=127.0.0.1

# Production - configure firewall rules
ENABLE_PORT_FIREWALL=true
```

### **Firewall Configuration**
For production deployments:

```bash
# Allow Agent Hive port range
sudo ufw allow 18000:18999/tcp
sudo ufw allow 15432/tcp  # PostgreSQL
sudo ufw allow 16379/tcp  # Redis
```

### **Network Security**
- All services bind to localhost by default
- CORS configured for specific ports only
- No services exposed externally without explicit configuration

## 🚨 **Troubleshooting**

### **Common Port Conflicts**
| Error | Cause | Solution |
|-------|-------|----------|
| `Address already in use: 18080` | API port conflict | Change `API_PORT` in `.env` |
| `Connection refused: 15432` | PostgreSQL not running | `docker-compose up postgres` |
| `Redis connection failed: 16379` | Redis not running | `docker-compose up redis` |
| `PWA dev server failed: 18443` | PWA port conflict | Change `PWA_DEV_PORT` in `.env` |

### **Diagnostic Commands**
```bash
# Check all port status
hive doctor

# Check specific port
netstat -an | grep 18080
lsof -i :18080

# Test connectivity  
curl http://localhost:18080/health
telnet localhost 15432
```

### **Service Dependencies**
Services must start in the correct order:
1. **PostgreSQL** (port 15432) - Database
2. **Redis** (port 16379) - Cache/messaging  
3. **FastAPI** (port 18080) - Backend API
4. **PWA** (port 18443) - Frontend dashboard

```bash
# Correct startup sequence
docker-compose up -d postgres redis  # Start databases
hive start                           # Start API server
cd mobile-pwa && npm run dev         # Start PWA (optional)
```

## 📱 **Mobile & Remote Access**

### **Network Configuration**
For mobile access on the same network:

```env
# Bind to all interfaces (development only)
API_HOST=0.0.0.0

# Update CORS for network access
CORS_ORIGINS=http://localhost:18080,http://192.168.1.100:18080,http://10.0.0.100:18080
```

### **Access URLs**
- **Local**: `http://localhost:18443`
- **Network**: `http://YOUR-IP:18443`
- **Tailscale**: `http://YOUR-TAILSCALE-IP:18443`

## 💡 **Best Practices**

1. **Use the CLI**: `hive start` handles all port configuration automatically
2. **Check conflicts**: Run `hive doctor` before starting services
3. **Environment consistency**: Use `.env` file for all port configuration
4. **Documentation**: Update port mappings when adding new services
5. **Testing**: Validate port changes with `hive status` and `hive dashboard`

---

**The non-standard port strategy ensures LeanVibe Agent Hive can run alongside any development environment without conflicts.**