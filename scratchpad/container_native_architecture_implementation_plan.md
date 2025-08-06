# Container-Native Architecture Implementation Plan

## CRITICAL ARCHITECTURAL GAP ANALYSIS

### Current State (Hybrid Architecture)
**What Works:**
- Docker services: PostgreSQL, Redis, FastAPI API
- Host-based agents: Claude Code CLI in tmux sessions
- Agent spawning via tmux session management
- Direct file system access for code modification
- Working orchestrator with task delegation

**Critical Limitations:**
- **Not container-native**: Agents run on host, not in Docker
- **No Kubernetes support**: Cannot deploy to production container orchestrators
- **Limited scalability**: tmux sessions don't scale horizontally
- **Host dependency**: Requires specific host setup (Claude Code CLI, tmux)
- **Security concerns**: Agents have full host access

### Target State (Container-Native Production)
**Requirements from Core Specifications:**
- Full container-native microservices
- Kubernetes deployment capability
- 50+ concurrent agents support
- <10 second agent spawn time
- <500ms orchestration latency
- Auto-scaling policies
- Production-grade security isolation

## PHASE 1: AGENT CONTAINERIZATION STRATEGY

### 1.1 Agent Runtime Architecture

Instead of tmux + Claude Code CLI, we need:

```python
# New Agent Runtime Architecture
class ContainerizedAgent:
    """Claude API-based agent running in Docker container"""
    
    def __init__(self, agent_type: str, anthropic_client: AsyncAnthropic):
        self.agent_type = agent_type
        self.anthropic_client = anthropic_client
        self.message_broker = get_message_broker()
        self.workspace = "/app/workspace"  # Container volume
    
    async def run(self):
        """Main agent loop - containerized version"""
        while True:
            # Get task from Redis (same as current)
            task = await self.get_next_task()
            
            # Execute via Claude API (not CLI)
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": task.prompt}],
                max_tokens=4000
            )
            
            # Process tools/actions within container
            await self.process_claude_response(response, task)
            
            # Store result (same as current)
            await self.store_result(task.id, response.content[0].text)
```

### 1.2 Docker Images for Agent Types

**Base Agent Image (Dockerfile.agent-base):**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-agent.txt /app/
RUN pip install -r /app/requirements-agent.txt

# Create workspace for agent operations
RUN mkdir -p /app/workspace
WORKDIR /app

# Copy agent runtime
COPY app/agents/ /app/agents/
COPY app/core/agent_runtime.py /app/

# Non-root user for security
RUN useradd -m -s /bin/bash agent
USER agent

ENTRYPOINT ["python", "agents/runtime.py"]
```

**Specialized Agent Images:**
```dockerfile
# Dockerfile.agent-architect
FROM leanvibe/agent-base:latest
ENV AGENT_TYPE=architect
ENV AGENT_CAPABILITIES="system_design,architecture_review,technical_planning"

# Dockerfile.agent-developer  
FROM leanvibe/agent-base:latest
ENV AGENT_TYPE=developer
ENV AGENT_CAPABILITIES="code_generation,bug_fixing,feature_implementation"

# Dockerfile.agent-qa
FROM leanvibe/agent-base:latest
ENV AGENT_TYPE=qa
ENV AGENT_CAPABILITIES="test_creation,validation,quality_assurance"
```

### 1.3 Agent-Container Communication Bridge

**Current Challenge:** tmux session management
**Solution:** Container orchestration via Docker API

```python
# app/core/container_orchestrator.py
import docker
from typing import Dict, List
import asyncio

class ContainerAgentOrchestrator:
    """Manages containerized agents instead of tmux sessions"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.running_agents: Dict[str, docker.models.containers.Container] = {}
    
    async def spawn_agent(self, agent_type: str) -> str:
        """Spawn agent in Docker container instead of tmux session"""
        agent_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"
        
        # Create container with proper networking
        container = self.docker_client.containers.run(
            image=f"leanvibe/agent-{agent_type}:latest",
            name=agent_id,
            environment={
                "AGENT_ID": agent_id,
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                "DATABASE_URL": settings.database_url,
                "REDIS_URL": settings.redis_url
            },
            volumes={
                "agent_workspace": {"bind": "/app/workspace", "mode": "rw"}
            },
            network="leanvibe_network",  # Same network as other services
            detach=True,
            auto_remove=False  # Keep for debugging
        )
        
        self.running_agents[agent_id] = container
        return agent_id
    
    async def stop_agent(self, agent_id: str):
        """Stop and remove agent container"""
        if agent_id in self.running_agents:
            container = self.running_agents[agent_id]
            container.stop()
            container.remove()
            del self.running_agents[agent_id]
    
    async def get_agent_logs(self, agent_id: str) -> str:
        """Get agent container logs (replaces tmux log viewing)"""
        if agent_id in self.running_agents:
            container = self.running_agents[agent_id]
            return container.logs().decode('utf-8')
        return ""
```

## PHASE 2: KUBERNETES PRODUCTION ARCHITECTURE

### 2.1 Agent Deployment Manifests

**Agent Deployment Template:**
```yaml
# k8s/agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-{agent_type}
  namespace: leanvibe-hive
spec:
  replicas: 3  # Auto-scaled based on load
  selector:
    matchLabels:
      app: agent
      type: {agent_type}
  template:
    metadata:
      labels:
        app: agent
        type: {agent_type}
    spec:
      containers:
      - name: agent
        image: leanvibe/agent-{agent_type}:latest
        env:
        - name: AGENT_TYPE
          value: "{agent_type}"
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-api-key
              key: key
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: database-config
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: workspace
          mountPath: /app/workspace
      volumes:
      - name: workspace
        emptyDir: {}
```

**Auto-Scaling Configuration:**
```yaml
# k8s/agent-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-{agent_type}-hpa
  namespace: leanvibe-hive
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-{agent_type}
  minReplicas: 1
  maxReplicas: 20  # Support for 50+ concurrent agents across all types
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: agent_queue_length  # Custom metric from Redis queue
      target:
        type: AverageValue
        averageValue: "5"
```

### 2.2 Service Mesh and Load Balancing

**Agent Service Definition:**
```yaml
# k8s/agent-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: agent-{agent_type}-service
  namespace: leanvibe-hive
spec:
  selector:
    app: agent
    type: {agent_type}
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

## PHASE 3: MIGRATION STRATEGY

### 3.1 Blue-Green Deployment Approach

**Step 1: Parallel Infrastructure**
- Deploy containerized agents alongside existing tmux agents
- Route small percentage of tasks to containerized agents
- Monitor performance and reliability

**Step 2: Gradual Migration**
- Increase percentage of tasks to containerized agents
- Implement feature parity validation
- Performance benchmarking

**Step 3: Complete Cutover**
- Route all tasks to containerized agents
- Shutdown tmux-based agents
- Remove tmux orchestration code

### 3.2 Feature Parity Validation

**Current Features to Preserve:**
```python
# Migration validation checklist
FEATURE_PARITY_TESTS = [
    "agent_spawning_time_<10s",
    "task_delegation_working",
    "inter_agent_communication", 
    "context_sharing_redis",
    "file_modification_capabilities",
    "error_handling_resilience",
    "performance_monitoring",
    "health_checks"
]
```

## PHASE 4: IMPLEMENTATION DELIVERABLES

### 4.1 Week 1-2: Container Migration

**Deliverable 1: Agent Runtime Container**
- Base Docker image with Claude API integration
- Specialized images for each agent type
- Container orchestration replacing tmux

**Deliverable 2: Development Environment**
- Updated docker-compose.yml with agent containers
- Local development workflow preservation
- Debugging capabilities (container logs vs tmux attach)

### 4.2 Week 3-4: Kubernetes Production

**Deliverable 3: K8s Manifests**
- Complete Kubernetes deployment configurations
- Auto-scaling policies and resource limits
- Service mesh and load balancing

**Deliverable 4: Production Infrastructure**
- CI/CD pipelines for agent image builds
- Production deployment scripts
- Monitoring and alerting setup

## SUCCESS METRICS VALIDATION

### Performance Targets (from PRD Requirements):
- ✅ Agent spawn time: <10 seconds (vs current tmux limitation)
- ✅ Concurrent agent capacity: 50+ agents (vs tmux ~10 practical limit)
- ✅ Orchestration latency: <500ms (Redis messaging preserved)
- ✅ System reliability: <0.1% orchestrator failure rate

### Production Readiness Checklist:
- [ ] Container images built and tested
- [ ] Kubernetes manifests validated
- [ ] Auto-scaling tested under load
- [ ] Migration path validated
- [ ] Security isolation implemented
- [ ] Monitoring and alerting configured
- [ ] Documentation updated

## RISK MITIGATION

### High Risk: Agent Communication Failures
**Current:** tmux session management
**Risk:** Container networking issues
**Mitigation:** Comprehensive network testing, circuit breakers

### Medium Risk: Performance Regression  
**Current:** Direct host execution
**Risk:** Container overhead
**Mitigation:** Performance benchmarking, resource optimization

### Low Risk: Development Experience
**Current:** tmux attach for debugging
**Risk:** Less intuitive container debugging
**Mitigation:** Container log streaming, exec access

This implementation plan bridges the critical architectural gap between the current hybrid system and the container-native production architecture specified in the core requirements, ensuring LeanVibe Agent Hive 2.0 can scale to enterprise production deployments.