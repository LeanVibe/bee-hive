# CLAUDE.md - API Layer & Web Services

## 🎯 **Context: REST API & WebSocket Services**

You are working in the **API service layer** of LeanVibe Agent Hive 2.0. This directory contains FastAPI endpoints, WebSocket handlers, and HTTP service implementations that provide the primary interface between clients and the hive system.

## 🚀 **API Architecture Overview**

### **Current Status: 219 Routes Discovered**
- **API v1 routes**: 139 discovered (`/api/v1/*`)
- **Dashboard routes**: 43 identified (`/dashboard/*`) 
- **Health endpoints**: 16 various health checks
- **WebSocket routes**: 8 real-time communication endpoints

### **Critical Implementation Gap**
- **Working Endpoints**: Health, metrics, status, basic dashboard APIs
- **Missing**: Many `/api/v1/agents` and `/api/v1/tasks` endpoints from PRD
- **Challenge**: Middleware dependencies causing startup issues

## 🎨 **Development Standards**

### **FastAPI Best Practices**
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

# Router organization pattern
router = APIRouter(
    prefix="/api/v1/agents",
    tags=["agents"],
    responses={404: {"description": "Not found"}}
)

# Request/Response models
class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., regex="^(backend-engineer|qa-test-guardian|devops-deployer|general-purpose)$")
    capabilities: List[str] = Field(default_factory=list)

class AgentResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    created_at: datetime
    
# Endpoint implementation
@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: AgentCreateRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    current_user: User = Depends(get_current_user)
):
    """Create new agent with proper error handling"""
    try:
        agent = await orchestrator.create_agent(
            name=request.name,
            type=request.type,
            capabilities=request.capabilities,
            created_by=current_user.id
        )
        return AgentResponse.from_agent(agent)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### **Error Handling Standards**
```python
# Consistent error response format
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "request_id": request.state.request_id
        }
    )

# Custom exceptions for business logic
class AgentNotFoundError(Exception):
    pass

class AgentCapacityExceededError(Exception):
    pass
```

### **WebSocket Implementation Patterns**
```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set

class ConnectionManager:
    """WebSocket connection management"""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
    async def broadcast_agent_update(self, agent_id: str, update_data: dict):
        """Broadcast agent status updates to subscribed clients"""
        for client_id, subscriptions in self.subscriptions.items():
            if agent_id in subscriptions:
                websocket = self.active_connections.get(client_id)
                if websocket:
                    await websocket.send_json({
                        "type": "agent_update",
                        "agent_id": agent_id,
                        "data": update_data
                    })

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            message = await websocket.receive_json()
            await handle_websocket_message(client_id, message)
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
```

## 📊 **Testing Requirements**

### **API Testing Standards**
- **Unit Tests**: Individual endpoint behavior validation
- **Integration Tests**: Database and orchestrator interaction
- **Contract Tests**: Request/response schema validation
- **Performance Tests**: Response time <200ms for 95th percentile

### **Test Organization**
```python
# tests/api/test_agents_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_orchestrator():
    orchestrator = Mock()
    orchestrator.create_agent = AsyncMock()
    return orchestrator

def test_create_agent_success(client: TestClient, mock_orchestrator):
    """Test successful agent creation"""
    mock_orchestrator.create_agent.return_value = Agent(
        id="agent-123",
        name="test-agent",
        type="backend-engineer"
    )
    
    response = client.post("/api/v1/agents", json={
        "name": "test-agent",
        "type": "backend-engineer"
    })
    
    assert response.status_code == 201
    assert response.json()["name"] == "test-agent"

def test_create_agent_validation_error(client: TestClient):
    """Test validation error handling"""
    response = client.post("/api/v1/agents", json={
        "name": "",  # Invalid: empty name
        "type": "invalid-type"  # Invalid: unknown type
    })
    
    assert response.status_code == 422
    assert "Validation Error" in response.json()["error"]
```

## 🔒 **Security & Authentication**

### **Authentication Patterns**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException
import jwt

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and validate user from JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
            
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/v1/agents")
@limiter.limit("10/minute")
async def list_agents(request: Request):
    """List agents with rate limiting"""
    pass
```

## 📈 **Performance & Monitoring**

### **Response Time Requirements**
- **Health Endpoints**: <50ms
- **Simple CRUD**: <200ms
- **Complex Queries**: <500ms
- **WebSocket Messages**: <100ms

### **Monitoring Integration**
```python
from prometheus_client import Counter, Histogram
from fastapi import Request, Response
import time

# Metrics collection
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.observe(process_time)
    
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## 🎯 **Epic Integration**

### **Epic 1: Agent Orchestration API**
Priority endpoints to implement:
- `POST /api/v1/agents` - Agent creation
- `GET /api/v1/agents/{id}` - Agent details  
- `PUT /api/v1/agents/{id}/status` - Agent status control
- `DELETE /api/v1/agents/{id}` - Agent cleanup

### **Epic 2: Task Management API**
- `POST /api/v1/tasks` - Task creation and assignment
- `GET /api/v1/tasks/{id}/status` - Task progress tracking
- `PUT /api/v1/tasks/{id}/priority` - Task prioritization

### **Epic 3: System Health API**
- Enhanced monitoring endpoints with detailed metrics
- Security audit endpoints for compliance
- Performance analytics endpoints

## ⚠️ **Critical Issues to Address**

### **Middleware Dependencies**
Current Issue: Middleware startup dependencies on Redis/external services
Solution Needed: Environment-aware dependency injection

### **Route Implementation Gap**
Current Issue: Many PRD-specified endpoints not implemented
Priority: Focus on core agent and task management endpoints

### **WebSocket Reliability**
Current Issue: Connection management and error recovery needs improvement
Priority: Implement robust connection lifecycle management

## ✅ **Success Criteria**

Your work in `/app/api/` is successful when:
- **Complete Coverage**: All PRD-specified endpoints implemented
- **Performance**: <200ms response times for 95th percentile
- **Reliability**: Proper error handling and graceful degradation
- **Security**: Authentication, authorization, and rate limiting
- **Testing**: Comprehensive test coverage with contract validation

Focus on **implementing missing endpoints** and **resolving middleware issues** to support Epic 1 orchestration requirements.