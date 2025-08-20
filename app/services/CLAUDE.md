# CLAUDE.md - Service Layer Guidelines

## 🎯 **Context: Business Logic & Service Layer**

You are working in the **service layer** of LeanVibe Agent Hive 2.0. This directory contains business logic implementations, service orchestration, and domain-specific operations that coordinate between the API layer and core system components.

## ✅ **Existing Service Architecture (DO NOT REBUILD)**

### **Core Service Categories Already Implemented**
- **Agent Services**: Agent lifecycle management, registration, status tracking
- **Task Services**: Task creation, assignment, execution monitoring
- **Communication Services**: Inter-agent messaging, notification handling
- **Configuration Services**: System configuration, environment management
- **Monitoring Services**: Health checks, metrics collection, alerting

### **Service Design Patterns Already Established**
- **Dependency Injection**: Clean separation of concerns
- **Async Operations**: Non-blocking service calls
- **Error Handling**: Consistent error propagation and logging
- **Caching Layer**: Redis integration for performance optimization
- **Transaction Management**: Database consistency and rollback handling

## 🔧 **Development Guidelines**

### **Enhancement Strategy (NOT Replacement)**
When improving service functionality:

1. **FIRST**: Analyze existing service implementations and patterns
2. **ENHANCE** existing services with AI-powered capabilities
3. **INTEGRATE** with enhanced core systems from `/app/core/`
4. **MAINTAIN** clean architecture and service boundaries

### **Service Integration with Enhanced Systems**
```python
# Pattern for enhancing existing services
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from app.core.command_ecosystem_integration import get_ecosystem_integration
from app.core.unified_quality_gates import QualityGateValidator
import structlog

logger = structlog.get_logger(__name__)

class EnhancedServiceBase(ABC):
    """Base class for all enhanced services."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.ecosystem_integration: Optional[Any] = None
        self.quality_validator = QualityGateValidator()
    
    async def initialize_enhanced_features(self):
        """Initialize AI-powered enhancements."""
        try:
            self.ecosystem_integration = await get_ecosystem_integration()
            self.logger.info("Enhanced features initialized")
        except Exception as e:
            self.logger.warning(f"Enhanced features unavailable: {e}")
    
    async def execute_with_quality_gates(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute service operation with quality gate validation."""
        # Pre-execution validation
        validation_result = await self.quality_validator.validate_pre_execution(
            service=self.__class__.__name__,
            operation=operation,
            parameters=kwargs
        )
        
        if not validation_result.passed:
            raise ServiceValidationError(f"Quality gate failed: {validation_result.message}")
        
        # Execute operation
        result = await self._execute_operation(operation, **kwargs)
        
        # Post-execution validation
        await self.quality_validator.validate_post_execution(
            service=self.__class__.__name__,
            operation=operation,
            result=result
        )
        
        return result
    
    @abstractmethod
    async def _execute_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Service-specific operation implementation."""
        pass
```

### **Enhanced Agent Service Pattern**
```python
from app.services.base_agent_service import BaseAgentService
from app.models.agent import Agent, AgentStatus
from typing import List, Optional

class EnhancedAgentService(BaseAgentService):
    """Enhanced agent service with AI-powered capabilities."""
    
    async def create_agent_with_intelligence(
        self, 
        spec: AgentSpec, 
        use_enhanced_creation: bool = False
    ) -> Agent:
        """Create agent with optional AI-powered optimization."""
        
        if use_enhanced_creation and self.ecosystem_integration:
            # Use AI to optimize agent configuration
            optimized_spec = await self.ecosystem_integration.optimize_agent_spec(spec)
            spec = optimized_spec
        
        # Use existing agent creation logic
        agent = await self.create_agent(spec)
        
        # Enhanced monitoring setup
        await self._setup_enhanced_monitoring(agent)
        
        return agent
    
    async def get_agent_insights(self, agent_id: str) -> Dict[str, Any]:
        """Get AI-powered insights about agent performance."""
        agent = await self.get_agent(agent_id)
        
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")
        
        insights = {
            "performance_metrics": await self._get_performance_metrics(agent),
            "task_efficiency": await self._analyze_task_efficiency(agent),
            "resource_utilization": await self._get_resource_utilization(agent)
        }
        
        # Add AI insights if available
        if self.ecosystem_integration:
            ai_insights = await self.ecosystem_integration.analyze_agent_performance(agent_id)
            insights["ai_recommendations"] = ai_insights
        
        return insights
    
    async def _setup_enhanced_monitoring(self, agent: Agent):
        """Setup enhanced monitoring for the agent."""
        # Implementation for enhanced monitoring
        pass
```

### **Service Error Handling Standards**
```python
class ServiceError(Exception):
    """Base exception for service layer errors."""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.error_code = error_code or "SERVICE_ERROR"
        self.details = details or {}
        self.timestamp = datetime.utcnow()

class ServiceValidationError(ServiceError):
    """Service validation specific error."""
    pass

class ServiceTimeoutError(ServiceError):
    """Service timeout specific error."""
    pass

def handle_service_errors(func):
    """Decorator for consistent service error handling."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ServiceError:
            # Re-raise service errors as-is
            raise
        except Exception as e:
            # Convert other errors to service errors
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise ServiceError(
                message=f"Service operation failed: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"function": func.__name__, "args": args, "kwargs": kwargs}
            )
    return wrapper
```

## 🔄 **Service Lifecycle Management**

### **Service Registration and Discovery**
```python
from typing import Dict, Type
from app.core.unified_managers.base_manager import BaseManager

class ServiceRegistry:
    """Registry for service discovery and lifecycle management."""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.service_health: Dict[str, bool] = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def register_service(self, service_name: str, service_instance: Any):
        """Register a service with the registry."""
        self.services[service_name] = service_instance
        await self._initialize_service(service_name, service_instance)
        self.logger.info(f"Service '{service_name}' registered successfully")
    
    async def get_service(self, service_name: str) -> Any:
        """Get service instance by name."""
        if service_name not in self.services:
            raise ServiceNotFoundError(f"Service '{service_name}' not found")
        
        service = self.services[service_name]
        if not await self._check_service_health(service_name):
            await self._restart_service(service_name)
        
        return service
    
    async def health_check_all_services(self) -> Dict[str, bool]:
        """Perform health check on all registered services."""
        health_status = {}
        for service_name, service in self.services.items():
            try:
                health = await service.health_check()
                health_status[service_name] = health.healthy
                self.service_health[service_name] = health.healthy
            except Exception as e:
                self.logger.error(f"Health check failed for {service_name}: {e}")
                health_status[service_name] = False
                self.service_health[service_name] = False
        
        return health_status

# Global service registry instance
service_registry = ServiceRegistry()
```

### **Service Configuration Management**
```python
from pydantic import BaseModel
from typing import Optional, Any
import os

class ServiceConfig(BaseModel):
    """Base configuration for all services."""
    name: str
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    timeout: int = 30
    retry_attempts: int = 3
    cache_ttl: int = 300
    enhanced_features_enabled: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class DatabaseServiceConfig(ServiceConfig):
    """Database service specific configuration."""
    database_url: str
    connection_pool_size: int = 10
    connection_timeout: int = 5
    query_timeout: int = 30

class CacheServiceConfig(ServiceConfig):
    """Cache service specific configuration."""
    redis_url: str
    max_connections: int = 20
    default_ttl: int = 3600

def load_service_config(service_name: str, config_class: Type[ServiceConfig]) -> ServiceConfig:
    """Load configuration for a specific service."""
    config_file = f"config/{service_name}.env"
    if os.path.exists(config_file):
        return config_class(_env_file=config_file)
    return config_class()
```

## 🧪 **Testing Requirements**

### **Service Testing Standards**
```python
# tests/services/test_enhanced_agent_service.py
import pytest
from unittest.mock import AsyncMock, Mock, patch
from app.services.enhanced_agent_service import EnhancedAgentService
from app.models.agent import Agent, AgentSpec
from app.core.exceptions import ServiceValidationError

@pytest.fixture
async def agent_service():
    """Create enhanced agent service for testing."""
    config = ServiceConfig(name="test_agent_service")
    service = EnhancedAgentService(config)
    await service.initialize_enhanced_features()
    return service

@pytest.mark.asyncio
async def test_create_agent_with_intelligence(agent_service):
    """Test AI-enhanced agent creation."""
    spec = AgentSpec(
        name="test-agent",
        type="backend-engineer",
        capabilities=["python", "fastapi"]
    )
    
    with patch.object(agent_service, 'ecosystem_integration') as mock_ecosystem:
        mock_ecosystem.optimize_agent_spec.return_value = spec
        
        agent = await agent_service.create_agent_with_intelligence(
            spec, 
            use_enhanced_creation=True
        )
        
        assert agent.name == "test-agent"
        assert agent.type == "backend-engineer"
        mock_ecosystem.optimize_agent_spec.assert_called_once_with(spec)

@pytest.mark.asyncio
async def test_service_error_handling(agent_service):
    """Test service error handling."""
    with pytest.raises(ServiceValidationError):
        await agent_service.execute_with_quality_gates(
            "invalid_operation",
            invalid_param="test"
        )

@pytest.mark.asyncio
async def test_service_health_check(agent_service):
    """Test service health checking."""
    health = await agent_service.health_check()
    assert health.healthy is True
    assert health.service_name == "EnhancedAgentService"
```

### **Integration Testing**
```python
# tests/services/test_service_integration.py
@pytest.mark.asyncio
async def test_service_to_service_communication():
    """Test communication between services."""
    agent_service = await service_registry.get_service("agent_service")
    task_service = await service_registry.get_service("task_service")
    
    # Create agent
    agent = await agent_service.create_agent(test_spec)
    
    # Assign task to agent
    task = await task_service.assign_task(agent.id, test_task_spec)
    
    assert task.agent_id == agent.id
    assert task.status == "assigned"

@pytest.mark.asyncio
async def test_enhanced_features_integration():
    """Test integration with enhanced core systems."""
    service = await service_registry.get_service("agent_service")
    
    if service.ecosystem_integration:
        insights = await service.get_agent_insights("test-agent-id")
        assert "ai_recommendations" in insights
```

## 🔗 **Integration Points**

### **Core System Integration** (`/app/core/`)
- Enhanced command ecosystem integration
- Unified quality gates for service validation
- Manager layer coordination and orchestration

### **API Layer Integration** (`/app/api/`)
- Service dependency injection into FastAPI endpoints
- Error handling and response formatting
- Async operation coordination

### **Database Integration** (`/app/models/`)
- ORM operations and transaction management
- Data validation and consistency
- Cache layer coordination

## ⚠️ **Critical Guidelines**

### **DO NOT Rebuild Existing Services**
- All basic service functionality exists and works well
- Focus on **enhancement** and **AI integration**
- Add quality gates to existing service operations
- Improve error handling and monitoring

### **Service Architecture Principles**
- **Single Responsibility**: Each service has one focused domain
- **Loose Coupling**: Services interact through well-defined interfaces
- **High Cohesion**: Related functionality grouped together
- **Async by Default**: All service operations are async

### **Performance Requirements**
- Service operations complete in <500ms for standard operations
- Complex operations (AI features) complete in <2 seconds
- Memory usage <100MB per service instance
- Graceful degradation when external dependencies fail

## 📋 **Enhancement Priorities**

### **High Priority**
1. **AI-powered insights** integration into existing services
2. **Quality gate validation** for all service operations
3. **Enhanced monitoring** and health checking
4. **Error handling** improvements and standardization

### **Medium Priority**
5. **Service mesh** communication patterns
6. **Advanced caching** strategies and optimization
7. **Transaction management** improvements
8. **Configuration management** enhancements

### **Low Priority**
9. **Service discovery** automation
10. **Load balancing** and scaling capabilities
11. **Circuit breaker** patterns for resilience
12. **Event sourcing** for audit and replay

## 🎯 **Success Criteria**

Your service enhancements are successful when:
- **Existing functionality** is preserved and enhanced
- **AI capabilities** integrate seamlessly with business logic
- **Quality gates** ensure operation reliability
- **Error handling** is consistent and informative
- **Performance** meets or exceeds current standards
- **Integration** with all system layers is robust

Focus on **enhancing existing service foundation** with AI-powered capabilities and improved reliability patterns.