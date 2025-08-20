# LeanVibe Agent Hive 2.0 - Project Management System

## Overview

The Project Management System provides a comprehensive, hierarchical approach to organizing and managing development work within LeanVibe Agent Hive 2.0. Built with agent-centric workflows and intelligent task routing, it enables sophisticated project coordination across multiple AI agents.

## Architecture

### Hierarchy Structure
```
Projects (Strategic Level)
    ├── Epics (Feature Level)  
        ├── PRDs (Requirements Level)
            ├── Tasks (Implementation Level)
```

### Key Components

1. **Models Layer** (`app/models/project_management.py`)
   - Project, Epic, PRD, Task entities with full hierarchy support
   - Short ID system for human-friendly references
   - Rich metadata and configuration options

2. **Kanban State Machine** (`app/core/kanban_state_machine.py`)
   - Universal workflow states: Backlog → Ready → In Progress → Review → Done
   - Automatic state transitions and validation
   - WIP limits and bottleneck detection
   - Comprehensive metrics and analytics

3. **Service Layer** (`app/services/project_management_service.py`)
   - High-level business logic and operations
   - Template-based project creation
   - Intelligent task generation and assignment
   - Workload analysis and balancing

4. **CLI Interface** (`app/cli/project_management_commands.py`)
   - Complete command-line interface
   - Rich formatting and interactive features
   - Short ID integration for easy navigation

5. **Orchestrator Integration** (`app/core/project_management_orchestrator_integration.py`)
   - Seamless integration with existing SimpleOrchestrator
   - Bidirectional synchronization with legacy tasks
   - Event-driven workflow automation

## Features

### 🎯 Hierarchical Project Organization

**Projects** serve as top-level containers for strategic initiatives:
- Status tracking (Planning, Active, On Hold, Completed, Archived)
- Stakeholder management and ownership
- Success criteria and objectives
- Timeline and milestone tracking

**Epics** represent major features or initiatives within projects:
- User story management and acceptance criteria
- Business value tracking and technical notes
- Story point estimation and dependency management
- Priority-based ordering

**PRDs (Product Requirements Documents)** contain detailed specifications:
- Functional and technical requirements
- User flows and acceptance criteria
- Approval workflow with reviewers
- Version control and change tracking
- Complexity scoring (1-10 scale)

**Tasks** are atomic units of work implementing PRD requirements:
- Rich task types (Feature Development, Bug Fix, Architecture, etc.)
- Capability-based requirements matching
- Effort estimation and time tracking
- Quality gates and acceptance criteria

### 🔄 Universal Kanban Workflow

All entities follow a consistent Kanban workflow:

```
BACKLOG → READY → IN_PROGRESS → REVIEW → DONE
    ↓        ↓         ↓          ↓
CANCELLED ←  BLOCKED  ←  ────────  ←
```

**State Machine Features:**
- Validation rules preventing invalid transitions
- Automatic timestamp tracking for cycle time analysis
- Event-driven automation and notifications
- WIP limits with violation alerts
- Historical state tracking for audit trails

### 🤖 Intelligent Agent Integration

**Smart Task Assignment:**
- Capability matching between tasks and agents
- Workload balancing across available agents  
- Multiple assignment strategies (balanced, capability-match, round-robin)
- Real-time workload analysis and recommendations

**Workflow Automation:**
- Auto-assignment when tasks move to In Progress
- Dependent task activation upon completion
- Parent entity progress updates (PRD → Epic → Project)
- Bottleneck detection and resolution suggestions

### 📊 Comprehensive Analytics

**Project Metrics:**
- Completion percentages across hierarchy levels
- Cycle time analysis and throughput measurement
- State distribution and bottleneck identification
- Agent productivity and workload analysis

**Workflow Insights:**
- Average completion times by task type
- Velocity tracking (completed items per day)
- WIP limit violations and flow efficiency
- Trend analysis and predictive insights

### 🔗 Short ID System

Human-friendly identifiers for easy CLI navigation:
- **PRJ-A7B2** for projects
- **EPC-X3Y9** for epics  
- **PRD-M4N8** for PRDs
- **TSK-K2L6** for tasks

Features:
- Collision-resistant generation with retry logic
- Partial matching (TSK-K2 matches TSK-K2L6)
- Database triggers for automatic generation
- Cross-reference lookup and resolution

## CLI Usage

### Project Management Commands

```bash
# Project operations
hive project create "New Web Platform" --description "Modern web platform" --tags "web,platform"
hive project list --status active --state in_progress
hive project show PRJ-A7B2 --detailed

# Epic management
hive epic create PRJ-A7B2 "User Authentication" --priority high --story-points 13
hive epic list --project PRJ-A7B2 --state ready
hive epic show EPC-X3Y9

# PRD operations  
hive prd create EPC-X3Y9 "Authentication API Specification" --complexity 7 --effort-days 5
hive prd list --epic EPC-X3Y9 --status approved
hive prd show PRD-M4N8

# Task management
hive task create PRD-M4N8 "Implement JWT token service" --task-type feature_development --effort-minutes 240
hive task list --assignee AGT-Z9X8 --state in_progress --priority high
hive task show TSK-K2L6
```

### Kanban Board Operations

```bash
# View boards
hive board show task --project PRJ-A7B2
hive board show prd --epic EPC-X3Y9

# Move items
hive board move TSK-K2L6 in_progress --reason "Starting development"
hive board move PRD-M4N8 review --reason "Ready for approval"

# Bulk operations
hive board move-bulk TSK-K2L6,TSK-K2L7,TSK-K2L8 ready --reason "Dependencies complete"
```

### Workflow and Analytics

```bash
# Workflow metrics
hive metrics workflow task --days 30
hive metrics workflow project --days 90

# Agent workload
hive metrics workload AGT-Z9X8
hive metrics workload --all --overloaded-only

# Auto-advance workflow
hive workflow auto-advance --limit 50
hive workflow rebalance --max-workload 0.8
```

## Database Schema

### Core Tables

```sql
-- Projects table
CREATE TABLE projects (
    id UUID PRIMARY KEY,
    short_id VARCHAR(20) UNIQUE,
    name VARCHAR(255) NOT NULL,
    status projectstatus DEFAULT 'planning',
    kanban_state kanban_state_project DEFAULT 'backlog',
    objectives JSON,
    stakeholders VARCHAR[],
    owner_agent_id UUID REFERENCES agents(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Epics table  
CREATE TABLE epics (
    id UUID PRIMARY KEY,
    short_id VARCHAR(20) UNIQUE,
    name VARCHAR(255) NOT NULL,
    project_id UUID REFERENCES projects(id),
    status epicstatus DEFAULT 'draft',
    kanban_state kanban_state_epic DEFAULT 'backlog',
    priority epic_priority DEFAULT 'MEDIUM',
    estimated_story_points INTEGER,
    dependencies UUID[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- PRDs table
CREATE TABLE prds (
    id UUID PRIMARY KEY,
    short_id VARCHAR(20) UNIQUE, 
    title VARCHAR(255) NOT NULL,
    epic_id UUID REFERENCES epics(id),
    status prdstatus DEFAULT 'draft',
    kanban_state kanban_state_prd DEFAULT 'backlog',
    requirements JSON,
    technical_requirements JSON,
    complexity_score INTEGER CHECK (complexity_score BETWEEN 1 AND 10),
    reviewers UUID[],
    approved_by UUID[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Enhanced tasks table
CREATE TABLE tasks (
    id UUID PRIMARY KEY,
    short_id VARCHAR(20) UNIQUE,
    title VARCHAR(255) NOT NULL,
    prd_id UUID REFERENCES prds(id), 
    task_type tasktype,
    kanban_state kanban_state_task DEFAULT 'backlog',
    priority taskpriority DEFAULT 'MEDIUM',
    assigned_agent_id UUID REFERENCES agents(id),
    estimated_effort_minutes INTEGER,
    actual_start TIMESTAMP,
    actual_completion TIMESTAMP,
    state_history JSON,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Indexes and Performance

```sql
-- Composite indexes for common queries
CREATE INDEX ix_projects_status_kanban ON projects(status, kanban_state);
CREATE INDEX ix_tasks_prd_kanban ON tasks(prd_id, kanban_state);  
CREATE INDEX ix_tasks_assigned_kanban ON tasks(assigned_agent_id, kanban_state);

-- Short ID indexes
CREATE UNIQUE INDEX ix_projects_short_id ON projects(short_id);
CREATE UNIQUE INDEX ix_tasks_short_id ON tasks(short_id);

-- Timeline indexes  
CREATE INDEX ix_tasks_priority_due ON tasks(priority, due_date);
CREATE INDEX ix_projects_timeline ON projects(start_date, target_end_date);
```

## Integration with SimpleOrchestrator

### Bidirectional Synchronization

The system seamlessly integrates with the existing SimpleOrchestrator:

```python
# Create integration
integration = ProjectManagementOrchestratorIntegration(db_session, orchestrator)
integration.initialize_integration()

# Legacy task migration
legacy_task = LegacyTask(...)
project_task = integration.create_project_task_from_legacy(
    legacy_task, prd_id, auto_assign=True
)

# Sync assignment changes
integration.sync_task_assignment(project_task, new_agent_id, sync_to_legacy=True)

# Intelligent routing
best_agent = integration.intelligent_task_routing(
    project_task, RoutingStrategy.CAPABILITY_MATCH
)
```

### Event-Driven Automation

```python
# Workflow transition handling
def handle_task_completion(task, old_state, new_state):
    if new_state == KanbanState.DONE:
        # Auto-start dependent tasks
        # Update parent PRD progress  
        # Notify stakeholders
        # Update agent workload metrics
```

### Workload Management

```python
# Get comprehensive workload analysis
workload = integration.get_agent_project_workload(agent_id)

# Automatic rebalancing
results = integration.rebalance_workloads(
    max_workload_score=0.8,
    min_workload_score=0.2  
)
```

## Setup and Installation

### 1. Database Migration

```bash
# Run migration to add project management tables
alembic upgrade head

# The migration creates:
# - Project hierarchy tables
# - Enhanced task table with new columns
# - Kanban state enums
# - Short ID system with triggers
# - Performance indexes
```

### 2. Initialize Demo Data

```bash
# Run setup script to create demo project
python scripts/setup_project_management.py

# This creates:
# - 5 demo agents with different specializations
# - Complete project hierarchy with realistic data
# - Tasks in various workflow states
# - Sample progress simulation
```

### 3. Verify Installation

```bash
# Test CLI commands
hive project list
hive board show task --project PRJ-A7B2
hive metrics workflow task

# Check integration health
python -c "
from app.core.project_management_orchestrator_integration import *
from app.core.database import get_db_session
session = next(get_db_session())
integration = ProjectManagementOrchestratorIntegration(session, None)
print(integration.get_integration_health())
"
```

## Configuration

### Environment Variables

```bash
# Database configuration
DATABASE_URL=postgresql://user:pass@localhost/beehive
DATABASE_POOL_SIZE=10

# Short ID configuration
SHORT_ID_PREFIX_LENGTH=3
SHORT_ID_CODE_LENGTH=4
SHORT_ID_MAX_RETRIES=5

# Workflow configuration  
WIP_LIMIT_TASKS_IN_PROGRESS=50
WIP_LIMIT_TASKS_REVIEW=20
AUTO_ADVANCE_ENABLED=true
```

### Customization Options

```python
# Custom WIP limits
wip_limits = {
    "Task": {
        KanbanState.IN_PROGRESS: 25,  # Lower limit for focused work
        KanbanState.REVIEW: 10
    }
}

# Custom transition rules
custom_rules = [
    TransitionRule(
        from_state=KanbanState.REVIEW,
        to_state=KanbanState.DONE,
        conditions=[custom_quality_gate_check],
        description="Complete with quality gate"
    )
]

# Template customization
templates = {
    "ai_project": [
        {"name": "Data Collection", "priority": TaskPriority.HIGH},
        {"name": "Model Training", "priority": TaskPriority.HIGH}, 
        {"name": "Evaluation & Testing", "priority": TaskPriority.MEDIUM}
    ]
}
```

## API Reference

### REST Endpoints

```python
# Project operations
POST   /api/v1/projects                    # Create project
GET    /api/v1/projects                    # List projects
GET    /api/v1/projects/{id}               # Get project details
PUT    /api/v1/projects/{id}               # Update project
DELETE /api/v1/projects/{id}               # Delete project

# Kanban operations
POST   /api/v1/kanban/transitions          # Transition entity state
GET    /api/v1/kanban/board/{entity_type}  # Get board view
GET    /api/v1/kanban/metrics/{entity_type} # Get workflow metrics

# Agent operations
GET    /api/v1/agents/{id}/workload        # Get agent workload
POST   /api/v1/agents/rebalance            # Rebalance workloads
GET    /api/v1/agents/{id}/recommendations # Get task recommendations
```

### Service Classes

```python
from app.services.project_management_service import ProjectManagementService

# Create service
service = ProjectManagementService(db_session)

# Project operations
project, epics = service.create_project_with_initial_structure(
    name="New Project",
    template_type="web_application"
)

# Task generation
tasks = service.auto_generate_implementation_tasks(
    prd_id, strategy="comprehensive"
)

# Smart assignment
assignments = service.smart_task_assignment(
    task_ids, strategy="capability_match"
)

# Analytics
stats = service.get_project_hierarchy_stats(project_id)
metrics = service.get_productivity_metrics(days=30, agent_id=agent_id)
```

## Best Practices

### 1. Project Structure

- **Keep projects focused** - 3-5 epics maximum per project
- **Define clear success criteria** - Measurable objectives and outcomes
- **Assign ownership** - Every project/epic/PRD should have an owner
- **Use templates** - Standardize common project types

### 2. Epic and PRD Management

- **Size epics appropriately** - 2-3 weeks of work maximum
- **Document requirements thoroughly** - Clear, testable acceptance criteria
- **Manage dependencies explicitly** - Track and resolve blockers early
- **Version PRDs** - Track changes and maintain approval audit trail

### 3. Task Organization

- **Break down work** - Tasks should be 2-4 hours maximum
- **Assign required capabilities** - Enable intelligent routing
- **Set realistic estimates** - Use historical data for accuracy
- **Define quality gates** - Clear completion criteria

### 4. Workflow Optimization

- **Monitor WIP limits** - Prevent context switching and bottlenecks
- **Track cycle times** - Identify process improvement opportunities
- **Use automation** - Let the system handle routine transitions
- **Regular reviews** - Weekly workflow retrospectives

### 5. Agent Coordination

- **Balance workloads** - Prevent overloading high-performers
- **Match capabilities** - Assign tasks to agents with right skills
- **Track productivity** - Monitor and optimize agent utilization
- **Provide feedback** - Use metrics to improve task routing

## Troubleshooting

### Common Issues

**Short ID Conflicts:**
```bash
# Check for duplicates
SELECT short_id, COUNT(*) FROM short_id_registry GROUP BY short_id HAVING COUNT(*) > 1;

# Regenerate IDs for entities
UPDATE tasks SET short_id = NULL WHERE id = 'task-uuid';
# Trigger will regenerate automatically
```

**Migration Issues:**
```bash
# Check migration status
alembic current

# Rollback if needed
alembic downgrade -1

# Re-run migration
alembic upgrade head
```

**Performance Issues:**
```bash
# Check missing indexes
EXPLAIN ANALYZE SELECT * FROM tasks WHERE kanban_state = 'in_progress';

# Update table statistics
ANALYZE projects, epics, prds, tasks;

# Check slow queries
SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;
```

### Health Checks

```python
# System health
from app.core.project_management_orchestrator_integration import *

integration = create_project_management_integration(db_session, orchestrator)
health = integration.get_integration_health()

print("System Status:", "OK" if health["initialized"] else "ERROR")
print("Entity Counts:", health["entity_counts"])
print("Components:", health["components"])
```

## Roadmap

### Phase 1 Completed ✅
- Core hierarchy models (Projects → Epics → PRDs → Tasks)
- Universal Kanban state machine with automation
- Short ID system for human-friendly navigation
- CLI interface with rich formatting
- SimpleOrchestrator integration
- Comprehensive test suite

### Phase 2 (Next Quarter)
- **Advanced Analytics Dashboard**
  - Real-time workflow visualization
  - Predictive analytics and trend analysis
  - Custom reporting and export

- **Template System Enhancement**  
  - User-defined project templates
  - Industry-specific templates (AI/ML, Web Dev, DevOps)
  - Template sharing and marketplace

- **Advanced Automation**
  - Custom workflow rules engine
  - Integration with external tools (GitHub, Jira)
  - Slack/Teams notifications

### Phase 3 (Future)
- **Machine Learning Integration**
  - Predictive task routing based on historical data
  - Automatic effort estimation using ML models
  - Anomaly detection for workflow bottlenecks

- **Multi-tenant Support**
  - Organization-level isolation
  - Role-based permissions system
  - Cross-organization collaboration

- **Mobile Application**
  - Native mobile app for task management
  - Offline support with sync
  - Push notifications for workflow events

---

## Support

For questions, issues, or contributions:

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Use GitHub issues for bug reports and feature requests  
- **Development**: See `CONTRIBUTING.md` for development setup
- **Testing**: Run `pytest tests/test_project_management_system.py -v`

The Project Management System represents a significant evolution in AI agent coordination, providing the structure and intelligence needed for complex, multi-agent development workflows.