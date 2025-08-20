# CLAUDE.md - Data Models & Database Layer

## 🎯 **Context: Data Models & ORM Layer**

You are working in the **data model and database layer** of LeanVibe Agent Hive 2.0. This directory contains SQLAlchemy models, Pydantic schemas, database migrations, and data access patterns that define the system's data architecture and persistence layer.

## ✅ **Existing Data Architecture (DO NOT REBUILD)**

### **Core Data Models Already Implemented**
- **Agent Models**: Agent entities, capabilities, status tracking, lifecycle states
- **Task Models**: Task definitions, assignments, dependencies, execution history
- **Communication Models**: Messages, events, notifications, inter-agent communication
- **System Models**: Configuration, metrics, audit logs, health status
- **User Models**: Authentication, authorization, roles, permissions

### **Database Patterns Already Established**
- **SQLAlchemy ORM**: Modern async ORM with relationship management
- **Pydantic Schemas**: Request/response validation and serialization
- **Alembic Migrations**: Version-controlled database schema evolution
- **Connection Pooling**: PostgreSQL connection management and optimization
- **Caching Layer**: Redis integration for frequently accessed data

## 🔧 **Development Guidelines**

### **Enhancement Strategy (NOT Replacement)**
When improving data models:

1. **FIRST**: Review existing models and database schema
2. **ENHANCE** existing models with AI-powered metadata
3. **INTEGRATE** with enhanced core systems for intelligent data operations
4. **MAINTAIN** database integrity and migration compatibility

### **Model Enhancement with AI Integration**
```python
# Pattern for enhancing existing models
from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any, List
from app.core.command_ecosystem_integration import get_ecosystem_integration
import structlog

logger = structlog.get_logger(__name__)

class EnhancedModelMixin:
    """Mixin for AI-enhanced model capabilities."""
    
    # AI metadata storage
    ai_metadata = Column(JSON, default=dict)
    enhanced_features_enabled = Column(Boolean, default=False)
    last_ai_analysis = Column(DateTime, nullable=True)
    
    async def generate_ai_insights(self) -> Dict[str, Any]:
        """Generate AI insights for this model instance."""
        try:
            ecosystem = await get_ecosystem_integration()
            insights = await ecosystem.analyze_model_data(
                model_type=self.__class__.__name__,
                model_data=self.to_dict(),
                include_predictions=True
            )
            
            # Update AI metadata
            self.ai_metadata = {
                **self.ai_metadata,
                "last_insights": insights,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            return insights
        except Exception as e:
            logger.error(f"Failed to generate AI insights for {self.__class__.__name__}: {e}")
            return {"error": str(e)}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for AI processing."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }

class EnhancedAgent(Base, EnhancedModelMixin):
    """Enhanced Agent model with AI capabilities."""
    __tablename__ = "agents"
    
    # Existing agent fields (DO NOT MODIFY)
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    capabilities = Column(JSON, default=list)
    
    # Enhanced fields for AI integration
    performance_score = Column(Integer, default=0)
    learning_metrics = Column(JSON, default=dict)
    optimization_suggestions = Column(JSON, default=list)
    
    @hybrid_property
    def efficiency_rating(self) -> float:
        """Calculate agent efficiency based on AI analysis."""
        if not self.ai_metadata.get("last_insights"):
            return 0.0
        
        insights = self.ai_metadata["last_insights"]
        return insights.get("efficiency_score", 0.0)
    
    async def update_performance_metrics(self) -> None:
        """Update performance metrics using AI analysis."""
        insights = await self.generate_ai_insights()
        
        if "performance_metrics" in insights:
            metrics = insights["performance_metrics"]
            self.performance_score = metrics.get("overall_score", 0)
            self.learning_metrics = metrics.get("learning_data", {})
            self.optimization_suggestions = metrics.get("optimizations", [])
```

### **Enhanced Pydantic Schemas**
```python
from pydantic import BaseModel, validator, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class AgentStatus(str, Enum):
    """Agent status enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class EnhancedAgentSchema(BaseModel):
    """Enhanced Pydantic schema for Agent with AI features."""
    
    # Core fields (existing)
    id: str
    name: str
    type: str
    status: AgentStatus
    created_at: datetime
    capabilities: List[str] = Field(default_factory=list)
    
    # Enhanced fields
    performance_score: int = Field(default=0, ge=0, le=100)
    learning_metrics: Dict[str, Any] = Field(default_factory=dict)
    optimization_suggestions: List[str] = Field(default_factory=list)
    ai_metadata: Dict[str, Any] = Field(default_factory=dict)
    enhanced_features_enabled: bool = False
    
    # Computed fields
    efficiency_rating: Optional[float] = None
    ai_insights_available: bool = False
    
    @validator('name')
    def validate_agent_name(cls, v):
        """Validate agent name follows naming conventions."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Agent name cannot be empty')
        if len(v) > 100:
            raise ValueError('Agent name cannot exceed 100 characters')
        return v.strip()
    
    @validator('type')
    def validate_agent_type(cls, v):
        """Validate agent type against supported types."""
        supported_types = [
            'backend-engineer', 'qa-test-guardian', 
            'devops-deployer', 'general-purpose'
        ]
        if v not in supported_types:
            raise ValueError(f'Agent type must be one of: {supported_types}')
        return v
    
    @validator('ai_metadata')
    def validate_ai_metadata(cls, v):
        """Ensure AI metadata contains required fields."""
        if v and 'analysis_timestamp' in v:
            # Validate timestamp format
            try:
                datetime.fromisoformat(v['analysis_timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                raise ValueError('Invalid analysis_timestamp format in ai_metadata')
        return v
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AgentCreateRequest(BaseModel):
    """Request schema for creating new agents."""
    name: str = Field(..., min_length=1, max_length=100)
    type: str
    capabilities: List[str] = Field(default_factory=list)
    enhanced_features_enabled: bool = Field(default=True)
    
    @validator('type')
    def validate_agent_type(cls, v):
        supported_types = [
            'backend-engineer', 'qa-test-guardian', 
            'devops-deployer', 'general-purpose'
        ]
        if v not in supported_types:
            raise ValueError(f'Agent type must be one of: {supported_types}')
        return v

class AgentUpdateRequest(BaseModel):
    """Request schema for updating agents."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    status: Optional[AgentStatus] = None
    capabilities: Optional[List[str]] = None
    enhanced_features_enabled: Optional[bool] = None
```

### **Database Migration Patterns**
```python
# alembic/versions/xxx_add_ai_enhancements.py
"""Add AI enhancement fields to existing models

Revision ID: xxx
Revises: previous_revision
Create Date: 2024-08-20 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'xxx'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade():
    # Add AI enhancement fields to agents table
    op.add_column('agents', sa.Column('performance_score', sa.Integer(), default=0))
    op.add_column('agents', sa.Column('learning_metrics', postgresql.JSON(), default={}))
    op.add_column('agents', sa.Column('optimization_suggestions', postgresql.JSON(), default=[]))
    op.add_column('agents', sa.Column('ai_metadata', postgresql.JSON(), default={}))
    op.add_column('agents', sa.Column('enhanced_features_enabled', sa.Boolean(), default=False))
    op.add_column('agents', sa.Column('last_ai_analysis', sa.DateTime(), nullable=True))
    
    # Add indexes for performance
    op.create_index('idx_agents_performance_score', 'agents', ['performance_score'])
    op.create_index('idx_agents_enhanced_features', 'agents', ['enhanced_features_enabled'])
    
    # Add AI enhancement fields to tasks table
    op.add_column('tasks', sa.Column('ai_complexity_score', sa.Float(), default=0.0))
    op.add_column('tasks', sa.Column('predicted_duration', sa.Integer(), nullable=True))
    op.add_column('tasks', sa.Column('ai_optimization_applied', sa.Boolean(), default=False))

def downgrade():
    # Remove AI enhancement fields
    op.drop_column('tasks', 'ai_optimization_applied')
    op.drop_column('tasks', 'predicted_duration')
    op.drop_column('tasks', 'ai_complexity_score')
    
    op.drop_index('idx_agents_enhanced_features')
    op.drop_index('idx_agents_performance_score')
    
    op.drop_column('agents', 'last_ai_analysis')
    op.drop_column('agents', 'enhanced_features_enabled')
    op.drop_column('agents', 'ai_metadata')
    op.drop_column('agents', 'optimization_suggestions')
    op.drop_column('agents', 'learning_metrics')
    op.drop_column('agents', 'performance_score')
```

## 🗄️ **Database Operations & Query Patterns**

### **Enhanced Database Access Layer**
```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Optional, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class EnhancedDatabaseManager:
    """Enhanced database manager with AI-powered optimizations."""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def get_agent_with_insights(self, agent_id: str) -> Optional[EnhancedAgent]:
        """Get agent with AI insights if available."""
        async with self.async_session() as session:
            agent = await session.get(EnhancedAgent, agent_id)
            if agent and agent.enhanced_features_enabled:
                # Refresh AI insights if stale
                if await self._should_refresh_insights(agent):
                    await agent.generate_ai_insights()
                    await session.commit()
            return agent
    
    async def get_optimized_agent_assignments(
        self, 
        task_requirements: Dict[str, Any]
    ) -> List[EnhancedAgent]:
        """Get agent assignments optimized by AI analysis."""
        async with self.async_session() as session:
            # Query agents with enhanced features
            query = session.query(EnhancedAgent).filter(
                EnhancedAgent.enhanced_features_enabled == True,
                EnhancedAgent.status == AgentStatus.ACTIVE
            )
            
            agents = await query.all()
            
            # Use AI to score and rank agents for the task
            if agents:
                ecosystem = await get_ecosystem_integration()
                scored_agents = await ecosystem.score_agents_for_task(
                    agents=[agent.to_dict() for agent in agents],
                    task_requirements=task_requirements
                )
                
                # Sort agents by AI score
                return sorted(agents, key=lambda a: scored_agents.get(a.id, 0), reverse=True)
            
            return agents
    
    async def _should_refresh_insights(self, agent: EnhancedAgent) -> bool:
        """Determine if AI insights need refreshing."""
        if not agent.last_ai_analysis:
            return True
        
        # Refresh if older than 1 hour
        time_since_analysis = datetime.utcnow() - agent.last_ai_analysis
        return time_since_analysis.total_seconds() > 3600
```

## 🧪 **Testing Requirements**

### **Model Testing Standards**
```python
# tests/models/test_enhanced_agent_model.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.agent import EnhancedAgent, AgentStatus
from app.models.schemas import EnhancedAgentSchema, AgentCreateRequest
import asyncio

@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

def test_enhanced_agent_creation(db_session):
    """Test enhanced agent model creation."""
    agent = EnhancedAgent(
        id="test-agent-123",
        name="Test Agent",
        type="backend-engineer",
        status=AgentStatus.INITIALIZING,
        enhanced_features_enabled=True
    )
    
    db_session.add(agent)
    db_session.commit()
    
    retrieved_agent = db_session.query(EnhancedAgent).filter_by(id="test-agent-123").first()
    assert retrieved_agent.name == "Test Agent"
    assert retrieved_agent.enhanced_features_enabled is True

@pytest.mark.asyncio
async def test_ai_insights_generation():
    """Test AI insights generation for models."""
    agent = EnhancedAgent(
        id="test-agent-123",
        name="Test Agent",
        type="backend-engineer",
        status=AgentStatus.ACTIVE,
        enhanced_features_enabled=True
    )
    
    # Mock ecosystem integration
    with patch('app.core.command_ecosystem_integration.get_ecosystem_integration') as mock_ecosystem:
        mock_ecosystem.return_value.analyze_model_data.return_value = {
            "efficiency_score": 85.5,
            "performance_metrics": {"overall_score": 85}
        }
        
        insights = await agent.generate_ai_insights()
        
        assert insights["efficiency_score"] == 85.5
        assert agent.ai_metadata["last_insights"] == insights

def test_agent_schema_validation():
    """Test Pydantic schema validation."""
    # Valid agent data
    valid_data = {
        "id": "agent-123",
        "name": "Valid Agent",
        "type": "backend-engineer",
        "status": "active",
        "created_at": "2024-08-20T12:00:00"
    }
    
    schema = EnhancedAgentSchema(**valid_data)
    assert schema.name == "Valid Agent"
    assert schema.type == "backend-engineer"
    
    # Invalid agent type
    invalid_data = valid_data.copy()
    invalid_data["type"] = "invalid-type"
    
    with pytest.raises(ValueError, match="Agent type must be one of"):
        EnhancedAgentSchema(**invalid_data)
```

### **Migration Testing**
```python
# tests/migrations/test_ai_enhancement_migration.py
def test_upgrade_migration(db_connection):
    """Test migration upgrade adds required columns."""
    # Run migration
    alembic_upgrade(db_connection, "xxx")  # AI enhancement revision
    
    # Check columns exist
    inspector = inspect(db_connection)
    columns = [col['name'] for col in inspector.get_columns('agents')]
    
    assert 'performance_score' in columns
    assert 'ai_metadata' in columns
    assert 'enhanced_features_enabled' in columns

def test_downgrade_migration(db_connection):
    """Test migration downgrade removes columns correctly."""
    # Run upgrade then downgrade
    alembic_upgrade(db_connection, "xxx")
    alembic_downgrade(db_connection, "previous_revision")
    
    # Check columns are removed
    inspector = inspect(db_connection)
    columns = [col['name'] for col in inspector.get_columns('agents')]
    
    assert 'performance_score' not in columns
    assert 'ai_metadata' not in columns
```

## 🔗 **Integration Points**

### **Service Layer Integration** (`/app/services/`)
- Model instances with enhanced AI capabilities
- Database operations with intelligent optimization
- Schema validation and error handling

### **API Layer Integration** (`/app/api/`)
- Pydantic schema serialization/deserialization
- Request validation and response formatting
- Database session management

### **Core System Integration** (`/app/core/`)
- AI-powered model analysis and insights
- Enhanced data processing integration
- Quality gate validation for data operations

## ⚠️ **Critical Guidelines**

### **DO NOT Rebuild Existing Models**
- All basic data models exist and work well
- Focus on **enhancement** with AI metadata and capabilities
- Add computed properties and intelligent operations
- Maintain backward compatibility with existing schema

### **Database Integrity Requirements**
- All migrations must be reversible
- Foreign key constraints must be preserved
- Indexes should be optimized for enhanced queries
- Data validation must be comprehensive

### **Performance Requirements**
- Database queries complete in <100ms for simple operations
- Complex AI-enhanced queries complete in <500ms
- Connection pool maintains <50 active connections
- Cache hit ratio >90% for frequently accessed data

## 📋 **Enhancement Priorities**

### **High Priority**
1. **AI metadata integration** into existing models
2. **Enhanced query patterns** with intelligent optimization
3. **Schema validation** improvements with AI validation
4. **Migration framework** for AI enhancement rollout

### **Medium Priority**
5. **Performance optimization** for AI-enhanced queries
6. **Data analytics** models for system insights
7. **Audit trail** enhancements with AI analysis
8. **Cache integration** optimization

### **Low Priority**
9. **Advanced indexing** strategies for AI metadata
10. **Data archival** and lifecycle management
11. **Multi-tenant** data isolation patterns
12. **Real-time synchronization** for distributed deployments

## 🎯 **Success Criteria**

Your model enhancements are successful when:
- **Existing functionality** is preserved and enhanced
- **AI capabilities** integrate seamlessly with data operations
- **Schema validation** is comprehensive and reliable
- **Database performance** meets or exceeds current standards
- **Migration compatibility** is maintained
- **Integration** with all system layers is robust

Focus on **enhancing existing data foundation** with AI-powered capabilities while maintaining data integrity and performance.