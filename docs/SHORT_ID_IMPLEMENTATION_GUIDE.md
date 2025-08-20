# Human-Friendly Short ID System Implementation Guide

## LeanVibe Agent Hive 2.0 - Short ID System

This guide provides comprehensive documentation for implementing and using the human-friendly short ID system in LeanVibe Agent Hive 2.0.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation](#implementation)
4. [Database Migration](#database-migration)
5. [CLI Integration](#cli-integration)
6. [API Usage](#api-usage)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

The Short ID system provides human-friendly identifiers alongside existing UUIDs to improve CLI usability and human interaction with the system.

### Key Features

- **Human-friendly format**: `TSK-A7B2` instead of `123e4567-e89b-12d3-a456-426614174000`
- **Hierarchical prefixes**: Different entity types have distinct prefixes
- **Collision-resistant**: Uses cryptographic hashing to minimize collisions
- **Partial matching**: Support for Git-style partial ID resolution
- **Database performance**: Maintains UUID backing for optimal database operations
- **Migration-friendly**: Can be added to existing systems without breaking changes

### Entity Types and Prefixes

| Entity Type | Prefix | Example | Description |
|-------------|---------|---------|-------------|
| Project | PRJ | PRJ-X2Y8 | Project indexes |
| Epic | EPC | EPC-M9K3 | Development epics |
| PRD | PRD | PRD-Q5R7 | Product Requirements Documents |
| Task | TSK | TSK-A7B2 | Individual tasks |
| Agent | AGT | AGT-M4K9 | AI agents |
| Workflow | WFL | WFL-P6N4 | Workflow definitions |
| File | FIL | FIL-H8T5 | File entries |
| Dependency | DEP | DEP-L2W9 | Code dependencies |
| Snapshot | SNP | SNP-B4G7 | Index snapshots |
| Session | SES | SES-K3J8 | Analysis sessions |
| Debt | DBT | DBT-F9C2 | Technical debt items |
| Plan | PLN | PLN-R7V4 | Remediation plans |

## Architecture

### Components

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Commands      │    │  Short ID        │    │   Database      │
│                     │────│  Generator       │────│                 │
│ hive task TSK-A7B2  │    │                  │    │ UUIDs + Short   │
└─────────────────────┘    └──────────────────┘    └─────────────────┘
                                   │
                           ┌───────────────────┐
                           │  Model Mixins     │
                           │                   │
                           │ - ShortIdMixin    │
                           │ - Resolution      │
                           │ - Validation      │
                           └───────────────────┘
```

### ID Format

```
PREFIX-CODE
│      │
│      └── 4-character Base32 code (human-friendly alphabet)
│
└── 3-character entity type prefix
```

Example: `TSK-A7B2`
- `TSK`: Task entity type
- `A7B2`: Unique 4-character code

### Base32 Alphabet

Uses Crockford's Base32 variant excluding confusing characters:
- **Included**: `23456789ABCDEFGHJKMNPQRSTVWXYZ`
- **Excluded**: `0`, `1`, `I`, `O` (to avoid confusion)

## Implementation

### 1. Model Integration

Add short ID support to existing models using the mixin:

```python
from app.models.short_id_mixin import ShortIdMixin
from app.core.short_id_generator import EntityType

class Task(Base, ShortIdMixin):
    __tablename__ = 'tasks'
    
    # Define entity type for this model
    ENTITY_TYPE = EntityType.TASK
    
    # Existing columns
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    # ... other columns
    
    # Short ID columns are added automatically by mixin
```

### 2. Generating Short IDs

#### Automatic Generation (Recommended)

Short IDs are generated automatically via database triggers:

```python
# Create new task - short ID generated automatically
task = Task(title="Fix authentication bug")
session.add(task)
session.commit()

print(task.short_id)  # Output: TSK-A7B2
```

#### Manual Generation

```python
from app.core.short_id_generator import generate_short_id, EntityType

# Generate manually
short_id, uuid_obj = generate_short_id(EntityType.TASK)
print(f"Generated: {short_id} -> {uuid_obj}")

# Generate for existing entity
task = Task(title="New task")
session.add(task)
session.flush()  # Get the UUID

task.generate_short_id()
print(task.short_id)  # Output: TSK-B3K9
```

### 3. Resolving IDs

#### Basic Resolution

```python
# Find by exact short ID
task = Task.find_by_short_id("TSK-A7B2", session)

# Find by partial ID
matches = Task.find_by_partial_short_id("TSK-A7", session)

# Resolve any ID format
task = Task.resolve_id_input("TSK-A7B2", session)  # Short ID
task = Task.resolve_id_input("A7B2", session)      # Partial
task = Task.resolve_id_input(uuid_obj, session)    # UUID
```

#### Smart Resolution with Error Handling

```python
try:
    task = Task.resolve_id_input(user_input, session)
    if task:
        print(f"Found task: {task.title}")
    else:
        print("Task not found")
except ValueError as e:
    print(f"Ambiguous ID: {e}")
    # Handle multiple matches
```

## Database Migration

### 1. Run Migration

```bash
# Apply migration
alembic upgrade head

# Check migration status  
alembic current

# Rollback if needed
alembic downgrade -1
```

### 2. Migration Components

The migration adds:

1. **New tables**:
   - `short_id_mappings`: Central mapping table

2. **New columns** (added to existing tables):
   - `short_id`: Human-friendly identifier
   - `short_id_generated_at`: Generation timestamp

3. **Database functions**:
   - `generate_short_id()`: Generate unique short IDs
   - `auto_generate_short_id()`: Trigger function

4. **Triggers**: Automatic generation on INSERT

5. **Indexes**: Performance optimization

### 3. Generate IDs for Existing Data

```python
from app.models.short_id_mixin import bulk_generate_short_ids
from app.models.task import Task

# Generate short IDs for all existing tasks
count = bulk_generate_short_ids(Task, session)
print(f"Generated {count} short IDs")
```

## CLI Integration

### 1. Basic Commands

```bash
# Show specific task
hive task show TSK-A7B2

# List tasks with filter
hive task list --filter TSK-A7

# Scale agent
hive agent scale AGT-M4K9 5

# Project status
hive project status PRJ-X2Y8
```

### 2. Partial ID Matching

```bash
# These are equivalent if unique:
hive task show TSK-A7B2
hive task show TSK-A7
hive task show A7B2
hive task show A7
```

### 3. Ambiguity Resolution

When multiple matches exist, the CLI will prompt:

```bash
$ hive task show A7

Multiple matches found for 'A7':
┌────────┬──────────┬─────────────┬────────────────────────────────────┐
│ Option │ Short ID │ Entity Type │ UUID                               │
├────────┼──────────┼─────────────┼────────────────────────────────────┤
│ 1      │ TSK-A7B2 │ TASK        │ 123e4567-e89b-12d3-a456-426614174000 │
│ 2      │ TSK-A7K9 │ TASK        │ 987fcdeb-51a2-43d7-b123-987654321000 │
└────────┴──────────┴─────────────┴────────────────────────────────────┘

Select option (number) or 'q' to quit: 1
```

### 4. Short ID Management Commands

```bash
# Generate test IDs
hive short-id generate task --count 10

# Resolve partial ID
hive short-id resolve A7B

# Validate ID format
hive short-id validate TSK-A7B2

# List all short IDs
hive short-id list --entity-type task
```

## API Usage

### 1. REST API Endpoints

Both UUIDs and short IDs are accepted in API endpoints:

```bash
# Using UUID
GET /api/tasks/123e4567-e89b-12d3-a456-426614174000

# Using short ID  
GET /api/tasks/TSK-A7B2

# Both return the same result
```

### 2. JSON Responses

API responses include both identifiers:

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "short_id": "TSK-A7B2", 
  "title": "Fix authentication bug",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 3. Creating Resources

Short IDs are generated automatically:

```bash
POST /api/tasks
{
  "title": "New task",
  "description": "Task description"
}

# Response includes generated short_id
{
  "id": "456e7890-e89b-12d3-a456-426614174001",
  "short_id": "TSK-B3K9",
  "title": "New task",
  "status": "pending"
}
```

## Best Practices

### 1. Model Design

```python
class YourModel(Base, ShortIdMixin):
    __tablename__ = 'your_table'
    
    # Always define entity type
    ENTITY_TYPE = EntityType.YOUR_TYPE
    
    # UUID primary key
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    
    # Business logic columns
    # ...
    
    def __str__(self):
        # Use display ID for string representation
        return f"{self.__class__.__name__}({self.get_display_id()})"
```

### 2. CLI Command Design

```python
@click.command()
@click.argument('task_id')
@resolve_id_argument(EntityType.TASK)
def show_task(task_id, resolved_task_id_short, resolved_task_id_uuid):
    """Show task details."""
    # Use resolved IDs
    task = session.query(Task).filter(
        Task.id == resolved_task_id_uuid
    ).first()
    
    if task:
        console.print(f"Task: {task.title}")
        console.print(f"ID: {task.short_id}")
    else:
        console.print("[red]Task not found[/red]")
```

### 3. Error Handling

```python
def resolve_user_input(user_input: str, model_class, session: Session):
    """Safely resolve user input to entity."""
    try:
        entity = model_class.resolve_id_input(user_input, session)
        if entity:
            return entity
        else:
            raise ValueError(f"No {model_class.__name__} found with ID '{user_input}'")
    
    except ValueError as e:
        if "Ambiguous" in str(e):
            # Handle ambiguous input
            matches = model_class.find_by_partial_short_id(user_input, session)
            return handle_ambiguous_matches(matches, user_input)
        else:
            raise
```

### 4. Testing

```python
def test_short_id_generation():
    """Test short ID generation."""
    task = Task(title="Test task")
    session.add(task)
    session.commit()
    
    # Check short ID was generated
    assert task.short_id is not None
    assert task.short_id.startswith("TSK-")
    assert len(task.short_id) == 8  # TSK-XXXX
    
    # Validate format
    assert validate_short_id_format(task.short_id)

def test_partial_id_resolution():
    """Test partial ID matching."""
    # Create test tasks
    task1 = Task(title="Task 1")
    task1.short_id = "TSK-A7B2"
    task2 = Task(title="Task 2") 
    task2.short_id = "TSK-A7K9"
    
    session.add_all([task1, task2])
    session.commit()
    
    # Test exact match
    result = Task.find_by_short_id("TSK-A7B2", session)
    assert result == task1
    
    # Test partial match (unique)
    matches = Task.find_by_partial_short_id("TSK-A7B", session)
    assert len(matches) == 1
    assert matches[0] == task1
    
    # Test partial match (ambiguous)
    matches = Task.find_by_partial_short_id("TSK-A7", session)
    assert len(matches) == 2
```

## Troubleshooting

### Common Issues

#### 1. Short ID Not Generated

**Problem**: New entities don't get short IDs automatically.

**Solutions**:
```python
# Check if ENTITY_TYPE is defined
assert hasattr(YourModel, 'ENTITY_TYPE')
assert YourModel.ENTITY_TYPE is not None

# Check database triggers
# Run: SELECT * FROM information_schema.triggers WHERE trigger_name LIKE '%short_id%';

# Manual generation fallback
entity.generate_short_id()
session.commit()
```

#### 2. Duplicate Short IDs

**Problem**: Multiple entities have the same short ID.

**Solutions**:
```python
# Check for duplicates
from app.models.short_id_mixin import validate_model_short_ids

results = validate_model_short_ids(Task, session)
if results['duplicate_short_ids']:
    print("Found duplicates:", results['duplicate_short_ids'])

# Fix duplicates
for duplicate in results['duplicate_short_ids']:
    entities = session.query(Task).filter(
        Task.short_id == duplicate['short_id']
    ).all()
    
    # Keep first, regenerate others
    for entity in entities[1:]:
        entity.generate_short_id(force=True)
        
session.commit()
```

#### 3. Migration Failures

**Problem**: Migration fails on existing data.

**Solutions**:
```bash
# Check current schema
alembic current

# Show migration SQL without applying
alembic upgrade head --sql

# Apply migration in stages
alembic upgrade short_id_system

# Manual data fixes if needed
```

#### 4. Performance Issues

**Problem**: Short ID lookups are slow.

**Solutions**:
```sql
-- Check index usage
EXPLAIN ANALYZE SELECT * FROM tasks WHERE short_id = 'TSK-A7B2';

-- Verify indexes exist
SELECT schemaname, tablename, indexname, indexdef 
FROM pg_indexes 
WHERE indexname LIKE '%short_id%';

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_tasks_short_id ON tasks (short_id);
```

#### 5. CLI Resolution Errors

**Problem**: CLI can't resolve partial IDs.

**Solutions**:
```python
# Debug resolution
from app.cli.short_id_commands import ShortIdResolver

resolver = ShortIdResolver()
result = resolver.resolve("A7B", EntityType.TASK)

if not result.success:
    print(f"Error: {result.error}")
    if result.matches:
        print(f"Matches: {[m[0] for m in result.matches]}")
```

### Performance Monitoring

```python
from app.core.short_id_generator import get_generator

generator = get_generator()
stats = generator.get_stats()

print(f"Generated: {stats.generated_count}")
print(f"Collisions: {stats.collision_count}")
print(f"Avg time: {stats.average_generation_time_ms:.2f}ms")
```

### Validation Tools

```bash
# Validate all short IDs
hive short-id validate-all

# Check for collisions
hive short-id check-duplicates

# Performance test
hive short-id benchmark --count 1000
```

## Migration from UUIDs

### Gradual Migration Strategy

1. **Phase 1**: Add short ID columns (nullable)
2. **Phase 2**: Generate short IDs for existing data
3. **Phase 3**: Update CLI to accept both formats
4. **Phase 4**: Update APIs to return both formats
5. **Phase 5**: Make short IDs the primary display format
6. **Phase 6**: (Optional) Make short ID required

### Backward Compatibility

- All existing UUID-based operations continue to work
- APIs accept both UUID and short ID formats
- Database constraints ensure data integrity
- CLI provides intelligent resolution

This implementation provides a smooth transition path while maintaining system reliability and performance.