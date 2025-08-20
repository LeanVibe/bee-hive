"""
Short ID System Usage Examples for LeanVibe Agent Hive 2.0

This module demonstrates practical usage patterns for the human-friendly
short ID system, showing real-world integration scenarios.
"""

import uuid
from typing import List, Optional
from datetime import datetime, timedelta

from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID

from app.core.short_id_generator import (
    ShortIdGenerator, EntityType, generate_short_id, 
    resolve_short_id, validate_short_id_format
)
from app.models.short_id_mixin import ShortIdMixin, bulk_generate_short_ids
from app.cli.short_id_commands import ShortIdResolver, IdResolutionStrategy

# Example database setup
Base = declarative_base()
engine = create_engine('sqlite:///example.db')  # Using SQLite for simplicity
Session = sessionmaker(bind=engine)


# ============================================================================
# Example 1: Basic Model Integration
# ============================================================================

class ExampleTask(Base, ShortIdMixin):
    """Example task model with short ID support."""
    
    __tablename__ = 'example_tasks'
    
    # Required: Define the entity type for short ID generation
    ENTITY_TYPE = EntityType.TASK
    
    # Standard model fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default='pending')
    priority = Column(Integer, default=3)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Short ID columns are added automatically by the mixin
    
    def __str__(self):
        """Human-friendly string representation using short ID."""
        return f"Task({self.get_display_id()}): {self.title}"


class ExampleAgent(Base, ShortIdMixin):
    """Example agent model with short ID support."""
    
    __tablename__ = 'example_agents'
    ENTITY_TYPE = EntityType.AGENT
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    type = Column(String(50), default='worker')
    status = Column(String(20), default='idle')
    last_active = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# Example 2: CLI Command Integration
# ============================================================================

def demonstrate_cli_resolution():
    """Demonstrate CLI-style ID resolution."""
    
    session = Session()
    
    # Create test data
    task1 = ExampleTask(
        title="Fix authentication bug",
        description="Users can't log in properly"
    )
    task2 = ExampleTask(
        title="Add new feature",
        description="Implement user dashboard"
    )
    
    session.add_all([task1, task2])
    session.commit()
    
    print(f"Created tasks:")
    print(f"  {task1}")  # Output: Task(TSK-A7B2): Fix authentication bug
    print(f"  {task2}")  # Output: Task(TSK-K9M3): Add new feature
    
    # CLI-style resolution examples
    resolver = ShortIdResolver()
    
    # Test exact resolution
    result = resolver.resolve(task1.short_id, EntityType.TASK)
    if result.success:
        print(f"\n✓ Exact match: {result.short_id}")
    
    # Test partial resolution (unique)
    partial = task1.short_id[:6]  # TSK-A7
    result = resolver.resolve(partial, EntityType.TASK, IdResolutionStrategy.PARTIAL_ALLOW)
    if result.success:
        print(f"✓ Partial match: {partial} -> {result.short_id}")
    
    # Test ambiguous resolution
    if len(task1.short_id) > 5 and len(task2.short_id) > 5:
        # Create an artificially ambiguous case for demo
        print(f"\n--- Ambiguity Handling Demo ---")
        print(f"Available tasks: {task1.short_id}, {task2.short_id}")
        print(f"If user inputs partial ID that matches multiple tasks,")
        print(f"the system will prompt for clarification.")
    
    session.close()


# ============================================================================
# Example 3: API Integration Patterns
# ============================================================================

class TaskAPI:
    """Example API class showing short ID integration."""
    
    def __init__(self, session):
        self.session = session
    
    def get_task(self, task_id: str) -> Optional[ExampleTask]:
        """
        Get task by ID (supports both UUID and short ID).
        
        Args:
            task_id: Either UUID string or short ID
            
        Returns:
            Task instance or None if not found
        """
        try:
            # Try resolving as either UUID or short ID
            task = ExampleTask.resolve_id_input(task_id, self.session)
            return task
        except ValueError as e:
            if "Ambiguous" in str(e):
                # In a real API, you might return multiple options
                # or require the client to be more specific
                raise ValueError(f"Ambiguous task ID: {task_id}")
            return None
    
    def create_task(self, title: str, description: str = None) -> dict:
        """Create new task and return JSON representation."""
        
        task = ExampleTask(
            title=title,
            description=description
        )
        
        self.session.add(task)
        self.session.commit()
        
        # Short ID is generated automatically
        return {
            'id': str(task.id),
            'short_id': task.short_id,
            'title': task.title,
            'description': task.description,
            'status': task.status,
            'created_at': task.created_at.isoformat()
        }
    
    def search_tasks(self, query: str) -> List[dict]:
        """Search tasks by partial short ID or title."""
        
        results = []
        
        # Try partial short ID match first
        if validate_short_id_format(query) or '-' in query:
            matches = ExampleTask.find_by_partial_short_id(query, self.session)
            results.extend(matches)
        
        # Also search by title
        title_matches = self.session.query(ExampleTask).filter(
            ExampleTask.title.ilike(f'%{query}%')
        ).all()
        
        # Combine and deduplicate
        all_tasks = list(set(results + title_matches))
        
        return [task.to_dict_with_short_id() for task in all_tasks]


# ============================================================================
# Example 4: Batch Operations and Migration
# ============================================================================

def demonstrate_batch_operations():
    """Show batch operations for migration scenarios."""
    
    session = Session()
    
    # Create some tasks without short IDs (simulating existing data)
    tasks = []
    for i in range(50):
        task = ExampleTask(
            title=f"Batch task {i+1}",
            description=f"Description for task {i+1}"
        )
        # Bypass automatic short ID generation for demo
        task.short_id = None
        tasks.append(task)
    
    session.add_all(tasks)
    session.commit()
    
    print(f"Created {len(tasks)} tasks without short IDs")
    
    # Bulk generate short IDs
    print("Generating short IDs in bulk...")
    generated_count = bulk_generate_short_ids(ExampleTask, session, batch_size=10)
    print(f"Generated {generated_count} short IDs")
    
    # Verify all tasks now have short IDs
    tasks_with_ids = session.query(ExampleTask).filter(
        ExampleTask.short_id.isnot(None)
    ).count()
    
    print(f"Tasks with short IDs: {tasks_with_ids}")
    
    session.close()


# ============================================================================
# Example 5: Performance and Validation
# ============================================================================

def demonstrate_performance_monitoring():
    """Show performance monitoring capabilities."""
    
    from app.core.short_id_generator import get_generator
    
    generator = get_generator()
    
    print("Generating 1000 short IDs for performance test...")
    
    start_time = datetime.now()
    
    # Generate many IDs quickly
    for i in range(1000):
        short_id, uuid_obj = generate_short_id(EntityType.TASK)
    
    elapsed = datetime.now() - start_time
    
    # Get statistics
    stats = generator.get_stats()
    
    print(f"\nPerformance Statistics:")
    print(f"  Generated: {stats.generated_count} IDs")
    print(f"  Collisions: {stats.collision_count}")
    print(f"  Cache hits: {stats.cache_hits}")
    print(f"  Cache misses: {stats.cache_misses}")
    print(f"  Average generation time: {stats.average_generation_time_ms:.2f}ms")
    print(f"  Total test time: {elapsed.total_seconds():.2f}s")
    print(f"  IDs per second: {1000/elapsed.total_seconds():.1f}")


def demonstrate_validation():
    """Show validation capabilities."""
    
    session = Session()
    
    # Create some test tasks
    tasks = [
        ExampleTask(title="Valid task 1"),
        ExampleTask(title="Valid task 2"),
        ExampleTask(title="Valid task 3")
    ]
    
    session.add_all(tasks)
    session.commit()
    
    # Validate all short IDs
    from app.models.short_id_mixin import validate_model_short_ids
    
    results = validate_model_short_ids(ExampleTask, session)
    
    print(f"\nValidation Results:")
    print(f"  Total entities: {results['total_entities']}")
    print(f"  Entities with short ID: {results['entities_with_short_id']}")
    print(f"  Valid short IDs: {results['valid_short_ids']}")
    print(f"  Invalid short IDs: {results['invalid_short_ids']}")
    
    if results['invalid_entities']:
        print(f"  Invalid entities: {results['invalid_entities']}")
    
    if results['duplicate_short_ids']:
        print(f"  Duplicate short IDs: {results['duplicate_short_ids']}")
    
    session.close()


# ============================================================================
# Example 6: Custom CLI Commands
# ============================================================================

def create_custom_cli_command():
    """Example of creating custom CLI commands with short ID support."""
    
    import click
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    @click.group()
    def task_commands():
        """Custom task management commands."""
        pass
    
    @task_commands.command()
    @click.argument('task_id')
    def show(task_id: str):
        """Show task details by ID (supports short IDs and partials)."""
        
        session = Session()
        
        try:
            task = ExampleTask.resolve_id_input(task_id, session)
            
            if task:
                # Create rich table for display
                table = Table(title=f"Task Details: {task.short_id}")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Short ID", task.short_id)
                table.add_row("UUID", str(task.id))
                table.add_row("Title", task.title)
                table.add_row("Status", task.status)
                table.add_row("Priority", str(task.priority))
                table.add_row("Created", task.created_at.strftime("%Y-%m-%d %H:%M:%S"))
                
                console.print(table)
            else:
                console.print(f"[red]Task not found: {task_id}[/red]")
                
        except ValueError as e:
            if "Ambiguous" in str(e):
                console.print(f"[yellow]Ambiguous ID: {task_id}[/yellow]")
                
                # Show matches
                matches = ExampleTask.find_by_partial_short_id(task_id, session)
                
                match_table = Table(title="Multiple matches found")
                match_table.add_column("Short ID", style="cyan")
                match_table.add_column("Title", style="green")
                match_table.add_column("Status", style="yellow")
                
                for match in matches[:10]:  # Limit to 10
                    match_table.add_row(match.short_id, match.title, match.status)
                
                console.print(match_table)
                console.print("\n[dim]Use a more specific ID to select one[/dim]")
            else:
                console.print(f"[red]Error: {e}[/red]")
        
        finally:
            session.close()
    
    @task_commands.command()
    @click.option('--status', help='Filter by status')
    @click.option('--limit', default=20, help='Limit number of results')
    def list(status: Optional[str], limit: int):
        """List tasks with short IDs."""
        
        session = Session()
        
        query = session.query(ExampleTask)
        
        if status:
            query = query.filter(ExampleTask.status == status)
        
        tasks = query.limit(limit).all()
        
        if tasks:
            table = Table(title=f"Tasks ({len(tasks)} found)")
            table.add_column("Short ID", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Priority", style="magenta")
            
            for task in tasks:
                table.add_row(
                    task.short_id,
                    task.title[:50] + "..." if len(task.title) > 50 else task.title,
                    task.status,
                    str(task.priority)
                )
            
            console.print(table)
        else:
            console.print("[yellow]No tasks found[/yellow]")
        
        session.close()
    
    return task_commands


# ============================================================================
# Example 7: Integration Testing
# ============================================================================

def run_integration_tests():
    """Run comprehensive integration tests."""
    
    print("Running Short ID Integration Tests...")
    
    # Setup
    Base.metadata.create_all(engine)
    session = Session()
    
    try:
        # Test 1: Basic generation
        print("\n1. Testing basic ID generation...")
        task = ExampleTask(title="Test task")
        session.add(task)
        session.commit()
        
        assert task.short_id is not None
        assert task.short_id.startswith("TSK-")
        assert len(task.short_id) == 8
        print(f"   ✓ Generated: {task.short_id}")
        
        # Test 2: Resolution
        print("\n2. Testing ID resolution...")
        found_task = ExampleTask.find_by_short_id(task.short_id, session)
        assert found_task.id == task.id
        print(f"   ✓ Resolved: {task.short_id} -> {found_task.title}")
        
        # Test 3: Partial matching
        print("\n3. Testing partial matching...")
        partial = task.short_id[:6]  # TSK-XX
        matches = ExampleTask.find_by_partial_short_id(partial, session)
        assert len(matches) >= 1
        assert task in matches
        print(f"   ✓ Partial match: {partial} -> {len(matches)} results")
        
        # Test 4: API simulation
        print("\n4. Testing API integration...")
        api = TaskAPI(session)
        
        # Test get by short ID
        retrieved = api.get_task(task.short_id)
        assert retrieved.id == task.id
        
        # Test get by UUID
        retrieved = api.get_task(str(task.id))
        assert retrieved.id == task.id
        
        print(f"   ✓ API retrieval works for both UUID and short ID")
        
        # Test 5: Batch operations
        print("\n5. Testing batch operations...")
        initial_count = session.query(ExampleTask).count()
        
        # Create batch without short IDs
        batch_tasks = []
        for i in range(10):
            batch_task = ExampleTask(title=f"Batch test {i}")
            batch_task.short_id = None  # Simulate existing data
            batch_tasks.append(batch_task)
        
        session.add_all(batch_tasks)
        session.commit()
        
        # Generate short IDs in batch
        generated = bulk_generate_short_ids(ExampleTask, session, batch_size=5)
        assert generated == 10
        print(f"   ✓ Bulk generated {generated} short IDs")
        
        # Test 6: Validation
        print("\n6. Testing validation...")
        from app.models.short_id_mixin import validate_model_short_ids
        
        results = validate_model_short_ids(ExampleTask, session)
        assert results['invalid_short_ids'] == 0
        assert results['duplicate_short_ids'] == []
        print(f"   ✓ Validation passed: {results['valid_short_ids']} valid IDs")
        
        print("\n✅ All integration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    
    finally:
        session.close()


# ============================================================================
# Main Demo Runner
# ============================================================================

if __name__ == "__main__":
    print("LeanVibe Agent Hive 2.0 - Short ID System Examples")
    print("=" * 60)
    
    # Create database tables
    Base.metadata.create_all(engine)
    
    try:
        # Run demonstrations
        print("\n📋 Example 1: CLI Resolution")
        demonstrate_cli_resolution()
        
        print("\n📈 Example 2: Batch Operations")
        demonstrate_batch_operations()
        
        print("\n⚡ Example 3: Performance Monitoring")
        demonstrate_performance_monitoring()
        
        print("\n✅ Example 4: Validation")
        demonstrate_validation()
        
        print("\n🧪 Example 5: Integration Tests")
        run_integration_tests()
        
        print("\n🎯 Example 6: Custom CLI Command Available")
        print("    Run: python -c \"from examples.short_id_examples import create_custom_cli_command; create_custom_cli_command()()\"")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Short ID system examples completed!")
    print("\nNext steps:")
    print("1. Review the implementation guide: docs/SHORT_ID_IMPLEMENTATION_GUIDE.md")
    print("2. Run database migration: alembic upgrade head")
    print("3. Update your models to inherit from ShortIdMixin")
    print("4. Test CLI commands with short IDs")
    print("5. Update API endpoints to accept short IDs")