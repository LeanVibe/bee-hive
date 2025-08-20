"""
CLI Commands for Project Management in LeanVibe Agent Hive 2.0

Provides comprehensive CLI interface for managing the project hierarchy:
Projects → Epics → PRDs → Tasks with Kanban workflow operations.
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from sqlalchemy.orm import Session

from ..core.database import get_db_session
from ..core.kanban_state_machine import KanbanStateMachine, KanbanState
from ..models.project_management import (
    Project, Epic, PRD, ProjectTask as Task, ProjectStatus, EpicStatus, 
    PRDStatus, TaskType, TaskPriority
)
from ..core.short_id_generator import get_generator, EntityType

console = Console()

# Create project management command group
@click.group(name='project')
@click.pass_context
def project_management(ctx):
    """Project management commands for hierarchical task organization."""
    # Ensure database session is available
    if not hasattr(ctx, 'obj'):
        ctx.obj = {}
    
    try:
        ctx.obj['db_session'] = next(get_db_session())
        ctx.obj['kanban_machine'] = KanbanStateMachine(ctx.obj['db_session'])
    except Exception as e:
        console.print(f"[red]Failed to initialize database: {e}[/red]")
        ctx.exit(1)


# Project Commands
@project_management.group(name='project')
def project_commands():
    """Project-level operations."""
    pass


@project_commands.command('create')
@click.argument('name')
@click.option('--description', '-d', help='Project description')
@click.option('--owner', '-o', help='Owner agent ID or short ID')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--target-date', help='Target end date (YYYY-MM-DD)')
@click.option('--tags', help='Comma-separated tags')
@click.pass_context
def create_project(ctx, name: str, description: Optional[str], owner: Optional[str], 
                  start_date: Optional[str], target_date: Optional[str], tags: Optional[str]):
    """Create a new project."""
    db_session: Session = ctx.obj['db_session']
    
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        target_dt = datetime.strptime(target_date, '%Y-%m-%d') if target_date else None
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
        
        # Create project
        project = Project(
            name=name,
            description=description,
            start_date=start_dt,
            target_end_date=target_dt,
            tags=tag_list
        )
        
        # Set owner if provided
        if owner:
            # TODO: Resolve agent ID from short ID or UUID
            pass
        
        # Generate short ID
        project.ensure_short_id(db_session)
        
        db_session.add(project)
        db_session.commit()
        
        console.print(f"[green]✓[/green] Created project [bold]{project.get_display_id()}[/bold]: {name}")
        _display_project_details(project)
        
    except Exception as e:
        db_session.rollback()
        console.print(f"[red]Failed to create project: {e}[/red]")


@project_commands.command('list')
@click.option('--status', type=click.Choice([s.value for s in ProjectStatus]), help='Filter by status')
@click.option('--state', type=click.Choice([s.value for s in KanbanState]), help='Filter by kanban state')
@click.option('--limit', default=20, help='Maximum number of projects to show')
@click.pass_context
def list_projects(ctx, status: Optional[str], state: Optional[str], limit: int):
    """List projects with optional filtering."""
    db_session: Session = ctx.obj['db_session']
    
    query = db_session.query(Project)
    
    if status:
        query = query.filter(Project.status == ProjectStatus(status))
    if state:
        query = query.filter(Project.kanban_state == KanbanState(state))
    
    projects = query.limit(limit).all()
    
    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        return
    
    table = Table(title="Projects")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("State", style="blue")
    table.add_column("Progress", justify="right")
    table.add_column("Epics", justify="right")
    table.add_column("Created")
    
    for project in projects:
        completion = f"{project.get_completion_percentage():.1f}%"
        epic_count = len(project.epics)
        
        table.add_row(
            project.get_display_id(),
            project.name,
            project.status.value,
            project.kanban_state.value,
            completion,
            str(epic_count),
            project.created_at.strftime('%Y-%m-%d')
        )
    
    console.print(table)


@project_commands.command('show')
@click.argument('project_id')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
@click.pass_context
def show_project(ctx, project_id: str, detailed: bool):
    """Show detailed project information."""
    db_session: Session = ctx.obj['db_session']
    
    project = Project.resolve_id_input(project_id, db_session)
    if not project:
        console.print(f"[red]Project not found: {project_id}[/red]")
        return
    
    _display_project_details(project, detailed)


# Epic Commands
@project_management.group(name='epic')
def epic_commands():
    """Epic-level operations."""
    pass


@epic_commands.command('create')
@click.argument('project_id')
@click.argument('name')
@click.option('--description', '-d', help='Epic description')
@click.option('--priority', type=click.Choice([p.name.lower() for p in TaskPriority]), 
              default='medium', help='Epic priority')
@click.option('--story-points', type=int, help='Estimated story points')
@click.pass_context
def create_epic(ctx, project_id: str, name: str, description: Optional[str], 
                priority: str, story_points: Optional[int]):
    """Create a new epic within a project."""
    db_session: Session = ctx.obj['db_session']
    
    # Find project
    project = Project.resolve_id_input(project_id, db_session)
    if not project:
        console.print(f"[red]Project not found: {project_id}[/red]")
        return
    
    try:
        priority_enum = TaskPriority[priority.upper()]
        
        epic = Epic(
            name=name,
            description=description,
            project_id=project.id,
            priority=priority_enum,
            estimated_story_points=story_points
        )
        
        epic.ensure_short_id(db_session)
        
        db_session.add(epic)
        db_session.commit()
        
        console.print(f"[green]✓[/green] Created epic [bold]{epic.get_display_id()}[/bold]: {name}")
        console.print(f"    Project: [cyan]{project.get_display_id()}[/cyan] {project.name}")
        
    except Exception as e:
        db_session.rollback()
        console.print(f"[red]Failed to create epic: {e}[/red]")


@epic_commands.command('list')
@click.option('--project', help='Filter by project ID')
@click.option('--status', type=click.Choice([s.value for s in EpicStatus]), help='Filter by status')
@click.option('--state', type=click.Choice([s.value for s in KanbanState]), help='Filter by kanban state')
@click.option('--limit', default=20, help='Maximum number of epics to show')
@click.pass_context
def list_epics(ctx, project: Optional[str], status: Optional[str], state: Optional[str], limit: int):
    """List epics with optional filtering."""
    db_session: Session = ctx.obj['db_session']
    
    query = db_session.query(Epic)
    
    if project:
        proj = Project.resolve_id_input(project, db_session)
        if proj:
            query = query.filter(Epic.project_id == proj.id)
        else:
            console.print(f"[red]Project not found: {project}[/red]")
            return
    
    if status:
        query = query.filter(Epic.status == EpicStatus(status))
    if state:
        query = query.filter(Epic.kanban_state == KanbanState(state))
    
    epics = query.limit(limit).all()
    
    if not epics:
        console.print("[yellow]No epics found.[/yellow]")
        return
    
    table = Table(title="Epics")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Project", style="dim")
    table.add_column("Status")
    table.add_column("State", style="blue")
    table.add_column("Priority")
    table.add_column("PRDs", justify="right")
    
    for epic in epics:
        prd_count = len(epic.prds)
        
        table.add_row(
            epic.get_display_id(),
            epic.name,
            epic.project.get_display_id() if epic.project else "N/A",
            epic.status.value,
            epic.kanban_state.value,
            epic.priority.name,
            str(prd_count)
        )
    
    console.print(table)


# PRD Commands
@project_management.group(name='prd')
def prd_commands():
    """PRD (Product Requirements Document) operations."""
    pass


@prd_commands.command('create')
@click.argument('epic_id')
@click.argument('title')
@click.option('--description', '-d', help='PRD description')
@click.option('--priority', type=click.Choice([p.name.lower() for p in TaskPriority]), 
              default='medium', help='PRD priority')
@click.option('--complexity', type=click.IntRange(1, 10), help='Complexity score (1-10)')
@click.option('--effort-days', type=int, help='Estimated effort in days')
@click.pass_context
def create_prd(ctx, epic_id: str, title: str, description: Optional[str], 
               priority: str, complexity: Optional[int], effort_days: Optional[int]):
    """Create a new PRD within an epic."""
    db_session: Session = ctx.obj['db_session']
    
    # Find epic
    epic = Epic.resolve_id_input(epic_id, db_session)
    if not epic:
        console.print(f"[red]Epic not found: {epic_id}[/red]")
        return
    
    try:
        priority_enum = TaskPriority[priority.upper()]
        
        prd = PRD(
            title=title,
            description=description,
            epic_id=epic.id,
            priority=priority_enum,
            complexity_score=complexity,
            estimated_effort_days=effort_days
        )
        
        prd.ensure_short_id(db_session)
        
        db_session.add(prd)
        db_session.commit()
        
        console.print(f"[green]✓[/green] Created PRD [bold]{prd.get_display_id()}[/bold]: {title}")
        console.print(f"    Epic: [cyan]{epic.get_display_id()}[/cyan] {epic.name}")
        
    except Exception as e:
        db_session.rollback()
        console.print(f"[red]Failed to create PRD: {e}[/red]")


@prd_commands.command('list')
@click.option('--epic', help='Filter by epic ID')
@click.option('--status', type=click.Choice([s.value for s in PRDStatus]), help='Filter by status')
@click.option('--state', type=click.Choice([s.value for s in KanbanState]), help='Filter by kanban state')
@click.option('--limit', default=20, help='Maximum number of PRDs to show')
@click.pass_context
def list_prds(ctx, epic: Optional[str], status: Optional[str], state: Optional[str], limit: int):
    """List PRDs with optional filtering."""
    db_session: Session = ctx.obj['db_session']
    
    query = db_session.query(PRD)
    
    if epic:
        ep = Epic.resolve_id_input(epic, db_session)
        if ep:
            query = query.filter(PRD.epic_id == ep.id)
        else:
            console.print(f"[red]Epic not found: {epic}[/red]")
            return
    
    if status:
        query = query.filter(PRD.status == PRDStatus(status))
    if state:
        query = query.filter(PRD.kanban_state == KanbanState(state))
    
    prds = query.limit(limit).all()
    
    if not prds:
        console.print("[yellow]No PRDs found.[/yellow]")
        return
    
    table = Table(title="PRDs")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="bold")
    table.add_column("Epic", style="dim")
    table.add_column("Status")
    table.add_column("State", style="blue")
    table.add_column("Priority")
    table.add_column("Complexity", justify="right")
    table.add_column("Tasks", justify="right")
    
    for prd in prds:
        task_count = len(prd.tasks)
        complexity_str = str(prd.complexity_score) if prd.complexity_score else "N/A"
        
        table.add_row(
            prd.get_display_id(),
            prd.title,
            prd.epic.get_display_id() if prd.epic else "N/A",
            prd.status.value,
            prd.kanban_state.value,
            prd.priority.name,
            complexity_str,
            str(task_count)
        )
    
    console.print(table)


# Task Commands
@project_management.group(name='task')
def task_commands():
    """Task-level operations."""
    pass


@task_commands.command('create')
@click.argument('prd_id')
@click.argument('title')
@click.option('--description', '-d', help='Task description')
@click.option('--task-type', type=click.Choice([t.value for t in TaskType]), 
              default='feature_development', help='Task type')
@click.option('--priority', type=click.Choice([p.name.lower() for p in TaskPriority]), 
              default='medium', help='Task priority')
@click.option('--effort-minutes', type=int, help='Estimated effort in minutes')
@click.option('--assignee', help='Assigned agent ID or short ID')
@click.pass_context
def create_task(ctx, prd_id: str, title: str, description: Optional[str], 
                task_type: str, priority: str, effort_minutes: Optional[int], 
                assignee: Optional[str]):
    """Create a new task within a PRD."""
    db_session: Session = ctx.obj['db_session']
    
    # Find PRD
    prd = PRD.resolve_id_input(prd_id, db_session)
    if not prd:
        console.print(f"[red]PRD not found: {prd_id}[/red]")
        return
    
    try:
        task_type_enum = TaskType(task_type)
        priority_enum = TaskPriority[priority.upper()]
        
        task = Task(
            title=title,
            description=description,
            prd_id=prd.id,
            task_type=task_type_enum,
            priority=priority_enum,
            estimated_effort_minutes=effort_minutes
        )
        
        # Set assignee if provided
        if assignee:
            # TODO: Resolve agent ID from short ID or UUID
            pass
        
        task.ensure_short_id(db_session)
        
        db_session.add(task)
        db_session.commit()
        
        console.print(f"[green]✓[/green] Created task [bold]{task.get_display_id()}[/bold]: {title}")
        console.print(f"    PRD: [cyan]{prd.get_display_id()}[/cyan] {prd.title}")
        
    except Exception as e:
        db_session.rollback()
        console.print(f"[red]Failed to create task: {e}[/red]")


@task_commands.command('list')
@click.option('--prd', help='Filter by PRD ID')
@click.option('--assignee', help='Filter by assigned agent')
@click.option('--task-type', type=click.Choice([t.value for t in TaskType]), help='Filter by task type')
@click.option('--state', type=click.Choice([s.value for s in KanbanState]), help='Filter by kanban state')
@click.option('--priority', type=click.Choice([p.name.lower() for p in TaskPriority]), help='Filter by priority')
@click.option('--limit', default=20, help='Maximum number of tasks to show')
@click.pass_context
def list_tasks(ctx, prd: Optional[str], assignee: Optional[str], task_type: Optional[str], 
               state: Optional[str], priority: Optional[str], limit: int):
    """List tasks with optional filtering."""
    db_session: Session = ctx.obj['db_session']
    
    query = db_session.query(Task)
    
    if prd:
        pr = PRD.resolve_id_input(prd, db_session)
        if pr:
            query = query.filter(Task.prd_id == pr.id)
        else:
            console.print(f"[red]PRD not found: {prd}[/red]")
            return
    
    if assignee:
        # TODO: Resolve agent ID from short ID or UUID
        pass
    
    if task_type:
        query = query.filter(Task.task_type == TaskType(task_type))
    if state:
        query = query.filter(Task.kanban_state == KanbanState(state))
    if priority:
        query = query.filter(Task.priority == TaskPriority[priority.upper()])
    
    tasks = query.limit(limit).all()
    
    if not tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        return
    
    table = Table(title="Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="bold")
    table.add_column("PRD", style="dim")
    table.add_column("Type")
    table.add_column("State", style="blue")
    table.add_column("Priority")
    table.add_column("Effort", justify="right")
    table.add_column("Assignee", style="green")
    
    for task in tasks:
        effort_str = f"{task.estimated_effort_minutes}m" if task.estimated_effort_minutes else "N/A"
        assignee_str = str(task.assigned_agent_id)[:8] + "..." if task.assigned_agent_id else "Unassigned"
        
        table.add_row(
            task.get_display_id(),
            task.title,
            task.prd.get_display_id() if task.prd else "N/A",
            task.task_type.value,
            task.kanban_state.value,
            task.priority.name,
            effort_str,
            assignee_str
        )
    
    console.print(table)


# Kanban Board Commands
@project_management.group(name='board')
def board_commands():
    """Kanban board operations."""
    pass


@board_commands.command('show')
@click.argument('entity_type', type=click.Choice(['project', 'epic', 'prd', 'task']))
@click.option('--project', help='Filter by project')
@click.option('--epic', help='Filter by epic')
@click.option('--prd', help='Filter by PRD')
@click.pass_context
def show_board(ctx, entity_type: str, project: Optional[str], epic: Optional[str], prd: Optional[str]):
    """Show kanban board for entity type."""
    db_session: Session = ctx.obj['db_session']
    kanban_machine: KanbanStateMachine = ctx.obj['kanban_machine']
    
    entity_type_cap = entity_type.capitalize()
    
    # Get entities by state
    board_data = {}
    for state in KanbanState:
        entities = kanban_machine.get_entities_by_state(entity_type_cap, state)
        
        # Apply filters
        if project:
            proj = Project.resolve_id_input(project, db_session)
            if proj:
                entities = [e for e in entities if _entity_belongs_to_project(e, proj)]
        
        if epic:
            ep = Epic.resolve_id_input(epic, db_session)
            if ep:
                entities = [e for e in entities if _entity_belongs_to_epic(e, ep)]
        
        if prd:
            pr = PRD.resolve_id_input(prd, db_session)
            if pr:
                entities = [e for e in entities if _entity_belongs_to_prd(e, pr)]
        
        board_data[state] = entities
    
    _display_kanban_board(entity_type_cap, board_data)


@board_commands.command('move')
@click.argument('entity_id')
@click.argument('new_state', type=click.Choice([s.value for s in KanbanState]))
@click.option('--reason', help='Reason for the transition')
@click.option('--force', is_flag=True, help='Force transition ignoring rules')
@click.pass_context
def move_entity(ctx, entity_id: str, new_state: str, reason: Optional[str], force: bool):
    """Move an entity to a new kanban state."""
    db_session: Session = ctx.obj['db_session']
    kanban_machine: KanbanStateMachine = ctx.obj['kanban_machine']
    
    # Try to find entity across all types
    entity = None
    entity_type = None
    
    for model_class, type_name in [(Project, "Project"), (Epic, "Epic"), (PRD, "PRD"), (Task, "Task")]:
        try:
            found = model_class.resolve_id_input(entity_id, db_session)
            if found:
                entity = found
                entity_type = type_name
                break
        except:
            continue
    
    if not entity:
        console.print(f"[red]Entity not found: {entity_id}[/red]")
        return
    
    new_state_enum = KanbanState(new_state)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Moving entity...", total=1)
        
        result = kanban_machine.transition_entity_state(
            entity, new_state_enum, reason=reason, force=force
        )
        
        progress.update(task, advance=1)
    
    if result.success:
        console.print(f"[green]✓[/green] Moved {entity_type} [bold]{entity.get_display_id()}[/bold] "
                     f"from [yellow]{result.old_state.value}[/yellow] to [blue]{result.new_state.value}[/blue]")
        
        if result.auto_actions_performed:
            console.print("[dim]Auto-actions performed:[/dim]")
            for action in result.auto_actions_performed:
                console.print(f"  • {action}")
    else:
        console.print(f"[red]✗[/red] Failed to move entity:")
        for error in result.errors:
            console.print(f"  • [red]{error}[/red]")


# Metrics and Analytics Commands
@project_management.group(name='metrics')
def metrics_commands():
    """Project metrics and analytics."""
    pass


@metrics_commands.command('workflow')
@click.argument('entity_type', type=click.Choice(['project', 'epic', 'prd', 'task']))
@click.option('--days', default=30, help='Number of days to include in metrics')
@click.pass_context
def workflow_metrics(ctx, entity_type: str, days: int):
    """Show workflow metrics for entity type."""
    kanban_machine: KanbanStateMachine = ctx.obj['kanban_machine']
    
    entity_type_cap = entity_type.capitalize()
    metrics = kanban_machine.get_workflow_metrics(entity_type_cap, days)
    
    _display_workflow_metrics(metrics)


# Utility functions

def _display_project_details(project: Project, detailed: bool = False):
    """Display detailed project information."""
    panel_content = f"""
[bold]{project.name}[/bold]
ID: [cyan]{project.get_display_id()}[/cyan]
Status: {project.status.value}
Kanban State: [blue]{project.kanban_state.value}[/blue]
Progress: {project.get_completion_percentage():.1f}%
Created: {project.created_at.strftime('%Y-%m-%d %H:%M')}
"""
    
    if project.description:
        panel_content += f"\nDescription: {project.description}"
    
    if project.start_date:
        panel_content += f"\nStart Date: {project.start_date.strftime('%Y-%m-%d')}"
    
    if project.target_end_date:
        panel_content += f"\nTarget Date: {project.target_end_date.strftime('%Y-%m-%d')}"
    
    if project.tags:
        panel_content += f"\nTags: {', '.join(project.tags)}"
    
    console.print(Panel(panel_content, title="Project Details", expand=False))
    
    if detailed and project.epics:
        table = Table(title="Epics")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Status")
        table.add_column("State", style="blue")
        table.add_column("PRDs", justify="right")
        
        for epic in project.epics:
            prd_count = len(epic.prds)
            table.add_row(
                epic.get_display_id(),
                epic.name,
                epic.status.value,
                epic.kanban_state.value,
                str(prd_count)
            )
        
        console.print(table)


def _display_kanban_board(entity_type: str, board_data: Dict[KanbanState, List]):
    """Display a kanban board visualization."""
    console.print(f"\n[bold]{entity_type} Kanban Board[/bold]\n")
    
    # Create columns for each state
    tables = {}
    for state in KanbanState:
        table = Table(title=state.value.replace('_', ' ').title(), 
                     title_style="bold blue", show_header=False)
        table.add_column("Items", style="dim")
        
        entities = board_data.get(state, [])
        for entity in entities[:10]:  # Limit to 10 items per column
            display_name = f"{entity.get_display_id()}: {getattr(entity, 'name', getattr(entity, 'title', 'Unknown'))[:30]}"
            table.add_row(display_name)
        
        if len(entities) > 10:
            table.add_row(f"... and {len(entities) - 10} more")
        
        tables[state] = table
    
    # Display in rows of 3 columns
    states = list(KanbanState)
    for i in range(0, len(states), 3):
        row_states = states[i:i+3]
        row_tables = [tables[state] for state in row_states]
        
        # Print tables side by side (simplified - rich doesn't directly support this)
        for table in row_tables:
            console.print(table)
        console.print()


def _display_workflow_metrics(metrics):
    """Display workflow metrics in a formatted way."""
    panel_content = f"""
[bold]{metrics.entity_type} Workflow Metrics[/bold]
Total Entities: {metrics.total_entities}
Average Cycle Time: {metrics.average_cycle_time_days:.1f} days (if available)
Throughput: {metrics.throughput_per_day:.2f} items/day
"""
    
    if metrics.wip_limits_violated:
        panel_content += f"\n[red]WIP Limits Violated:[/red]\n"
        for violation in metrics.wip_limits_violated:
            panel_content += f"  • {violation}\n"
    
    if metrics.bottleneck_states:
        panel_content += f"\n[yellow]Bottleneck States:[/yellow]\n"
        for state in metrics.bottleneck_states:
            panel_content += f"  • {state.value}\n"
    
    console.print(Panel(panel_content, title="Metrics", expand=False))
    
    # State distribution table
    table = Table(title="State Distribution")
    table.add_column("State", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    
    total = sum(metrics.state_counts.values()) or 1
    for state, count in metrics.state_counts.items():
        percentage = (count / total) * 100
        table.add_row(
            state.value,
            str(count),
            f"{percentage:.1f}%"
        )
    
    console.print(table)


def _entity_belongs_to_project(entity, project: Project) -> bool:
    """Check if entity belongs to a project."""
    if hasattr(entity, 'project_id'):
        return entity.project_id == project.id
    elif hasattr(entity, 'epic') and entity.epic:
        return entity.epic.project_id == project.id
    elif hasattr(entity, 'prd') and entity.prd and entity.prd.epic:
        return entity.prd.epic.project_id == project.id
    return False


def _entity_belongs_to_epic(entity, epic: Epic) -> bool:
    """Check if entity belongs to an epic."""
    if hasattr(entity, 'epic_id'):
        return entity.epic_id == epic.id
    elif hasattr(entity, 'prd') and entity.prd:
        return entity.prd.epic_id == epic.id
    return False


def _entity_belongs_to_prd(entity, prd: PRD) -> bool:
    """Check if entity belongs to a PRD."""
    if hasattr(entity, 'prd_id'):
        return entity.prd_id == prd.id
    return False


# Add the project management group to the main CLI
def register_project_management_commands(main_cli_group):
    """Register project management commands with the main CLI group."""
    main_cli_group.add_command(project_management)