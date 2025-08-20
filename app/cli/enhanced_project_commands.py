"""
Enhanced Project Management CLI Commands with SimpleOrchestrator Integration

Extends the basic project management commands with real agent execution capabilities
through SimpleOrchestrator integration.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlalchemy.orm import Session

try:
    from ..core.database import get_db_session
    from ..core.simple_orchestrator import SimpleOrchestrator
    from ..core.configuration_service import ConfigurationService
    from ..core.project_task_execution_bridge import ProjectTaskExecutionBridge
    from ..core.project_management_orchestrator_integration import ProjectManagementOrchestratorIntegration
    from ..models.project_management import ProjectTask, Project, Epic, PRD
    from ..core.enhanced_agent_launcher import AgentLauncherType
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    # Create dummy types for CLI definition
    class AgentLauncherType:
        CLAUDE_CODE = "claude-code"
        TMUX_SESSION = "tmux-session"

console = Console()

class EnhancedProjectCLI:
    """Enhanced project CLI with orchestrator integration."""
    
    def __init__(self):
        self.orchestrator = None
        self.execution_bridge = None
        self.pm_integration = None
        self.db_session = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure all components are initialized."""
        if self._initialized:
            return True
        
        try:
            # Initialize configuration
            config_service = ConfigurationService()
            
            # Initialize database
            self.db_session = next(get_db_session())
            
            # Initialize orchestrator
            self.orchestrator = SimpleOrchestrator(config_service)
            
            # Initialize PM integration
            self.pm_integration = ProjectManagementOrchestratorIntegration(
                self.orchestrator, self.db_session
            )
            
            # Initialize execution bridge
            self.execution_bridge = ProjectTaskExecutionBridge(
                self.orchestrator, self.pm_integration, self.db_session
            )
            
            self._initialized = True
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to initialize enhanced project CLI: {e}[/red]")
            return False

# Create global instance
enhanced_cli = EnhancedProjectCLI()

@click.group()
def enhanced_project():
    """Enhanced project management with agent execution."""
    pass

@enhanced_project.command('execute-task')
@click.argument('task_id')
@click.option('--agent-type', type=click.Choice(['claude-code', 'tmux-session']), 
              help='Preferred agent type')
@click.option('--auto-spawn/--no-auto-spawn', default=True, 
              help='Auto-spawn agent if none available')
@click.option('--watch', '-w', is_flag=True, help='Watch execution progress')
def execute_task(task_id: str, agent_type: Optional[str], auto_spawn: bool, watch: bool):
    """Execute a project task through SimpleOrchestrator."""
    if not DEPENDENCIES_AVAILABLE:
        console.print("[red]Enhanced project commands require full system setup[/red]")
        console.print("💡 Run 'hive doctor' to check system status")
        return
    
    async def _execute():
        if not await enhanced_cli._ensure_initialized():
            return
        
        console.print(f"🚀 [bold blue]Executing task {task_id}...[/bold blue]")
        
        try:
            # Parse task ID
            if task_id.startswith('TSK-'):
                # Short ID - need to resolve to UUID
                from ..core.short_id_generator import resolve_short_id
                try:
                    _, task_uuid = resolve_short_id(task_id)
                except ValueError:
                    console.print(f"[red]Invalid task ID: {task_id}[/red]")
                    return
            else:
                # Assume UUID
                task_uuid = uuid.UUID(task_id)
            
            # Convert agent type if provided
            preferred_agent = None
            if agent_type:
                preferred_agent = AgentLauncherType(agent_type)
            
            # Execute task
            result = await enhanced_cli.execution_bridge.execute_project_task(
                task_uuid, auto_spawn, preferred_agent
            )
            
            if result.success:
                console.print(f"✅ [green]Task execution started successfully[/green]")
                console.print(f"   Agent ID: [cyan]{result.agent_id}[/cyan]")
                if result.session_name:
                    console.print(f"   Session: [dim]{result.session_name}[/dim]")
                
                if watch:
                    await _watch_task_execution(task_uuid)
            else:
                console.print(f"❌ [red]Task execution failed: {result.error_message}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error executing task: {e}[/red]")
    
    asyncio.run(_execute())

@enhanced_project.command('monitor-task')
@click.argument('task_id')
@click.option('--refresh', '-r', default=5, help='Refresh interval in seconds')
def monitor_task(task_id: str, refresh: int):
    """Monitor project task execution in real-time."""
    if not DEPENDENCIES_AVAILABLE:
        console.print("[red]Enhanced project commands require full system setup[/red]")
        return
        
    async def _monitor():
        if not await enhanced_cli._ensure_initialized():
            return
        
        try:
            # Parse task ID
            if task_id.startswith('TSK-'):
                from ..core.short_id_generator import resolve_short_id
                _, task_uuid = resolve_short_id(task_id)
            else:
                task_uuid = uuid.UUID(task_id)
            
            console.print(f"👁️ [bold]Monitoring task {task_id} (Press Ctrl+C to stop)[/bold]")
            
            while True:
                try:
                    status = await enhanced_cli.execution_bridge.monitor_task_execution(task_uuid)
                    
                    # Clear screen and show status
                    click.clear()
                    console.print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if status["status"] == "not_found":
                        console.print("[yellow]Task execution not tracked[/yellow]")
                        break
                    elif status["status"] == "no_agent":
                        console.print("[yellow]Task not assigned to agent[/yellow]")
                    else:
                        # Show detailed status
                        _display_execution_status(status)
                    
                    await asyncio.sleep(refresh)
                    
                except KeyboardInterrupt:
                    console.print("\\n🛑 Monitoring stopped")
                    break
                    
        except Exception as e:
            console.print(f"[red]Error monitoring task: {e}[/red]")
    
    asyncio.run(_monitor())

@enhanced_project.command('complete-task')
@click.argument('task_id')
@click.option('--success/--failed', default=True, help='Task completion status')
@click.option('--notes', help='Completion notes')
def complete_task(task_id: str, success: bool, notes: Optional[str]):
    """Mark a project task as completed."""
    if not DEPENDENCIES_AVAILABLE:
        console.print("[red]Enhanced project commands require full system setup[/red]")
        return
        
    async def _complete():
        if not await enhanced_cli._ensure_initialized():
            return
        
        try:
            # Parse task ID
            if task_id.startswith('TSK-'):
                from ..core.short_id_generator import resolve_short_id
                _, task_uuid = resolve_short_id(task_id)
            else:
                task_uuid = uuid.UUID(task_id)
            
            # Prepare results
            results = {}
            if notes:
                results['completion_notes'] = notes
            
            # Complete task
            completed = await enhanced_cli.execution_bridge.complete_task_execution(
                task_uuid, success, results
            )
            
            if completed:
                status_text = "completed" if success else "failed"
                console.print(f"✅ [green]Task {task_id} marked as {status_text}[/green]")
            else:
                console.print(f"❌ [red]Failed to update task completion status[/red]")
                
        except Exception as e:
            console.print(f"[red]Error completing task: {e}[/red]")
    
    asyncio.run(_complete())

@enhanced_project.command('execution-summary')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed execution info')
def execution_summary(detailed: bool):
    """Show summary of all active task executions."""
    if not DEPENDENCIES_AVAILABLE:
        console.print("[red]Enhanced project commands require full system setup[/red]")
        return
        
    async def _summary():
        if not await enhanced_cli._ensure_initialized():
            return
        
        try:
            summary = await enhanced_cli.execution_bridge.get_execution_summary()
            
            console.print("🎯 [bold]Task Execution Summary[/bold]")
            
            # Summary stats
            stats_panel = f"""
[bold]Active Executions:[/bold] {summary['total_active_executions']}
[green]Successful:[/green] {summary['successful_executions']}
[red]Failed:[/red] {summary['failed_executions']}
"""
            
            console.print(Panel(stats_panel.strip(), title="Statistics", border_style="blue"))
            
            if summary['executions'] and detailed:
                # Detailed execution table
                table = Table(title="Active Executions")
                table.add_column("Task ID", style="cyan")
                table.add_column("Status")
                table.add_column("Agent ID", style="green")
                table.add_column("Session", style="dim")
                table.add_column("Error", style="red")
                
                for task_id, execution in summary['executions'].items():
                    status = "✅ Success" if execution['success'] else "❌ Failed"
                    agent_id = execution['agent_id'] or "N/A"
                    session = execution['session_name'] or "N/A"
                    error = "Yes" if execution['has_error'] else "No"
                    
                    table.add_row(task_id[:8] + "...", status, agent_id, session, error)
                
                console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error getting execution summary: {e}[/red]")
    
    asyncio.run(_summary())

@enhanced_project.command('auto-assign-tasks')
@click.option('--project', help='Project ID to process')
@click.option('--epic', help='Epic ID to process')
@click.option('--prd', help='PRD ID to process')
@click.option('--dry-run', is_flag=True, help='Show what would be assigned without executing')
def auto_assign_tasks(project: Optional[str], epic: Optional[str], 
                     prd: Optional[str], dry_run: bool):
    """Auto-assign tasks to appropriate agents based on capabilities."""
    if not DEPENDENCIES_AVAILABLE:
        console.print("[red]Enhanced project commands require full system setup[/red]")
        return
        
    async def _auto_assign():
        if not await enhanced_cli._ensure_initialized():
            return
        
        try:
            console.print("🤖 [bold blue]Auto-assigning tasks to agents...[/bold blue]")
            
            # Get tasks to process
            query = enhanced_cli.db_session.query(ProjectTask).filter(
                ProjectTask.assigned_agent_id.is_(None)  # Only unassigned tasks
            )
            
            # Apply filters
            if prd:
                if prd.startswith('PRD-'):
                    from ..core.short_id_generator import resolve_short_id
                    _, prd_uuid = resolve_short_id(prd)
                else:
                    prd_uuid = uuid.UUID(prd)
                query = query.filter(ProjectTask.prd_id == prd_uuid)
                
            elif epic:
                if epic.startswith('EPC-'):
                    from ..core.short_id_generator import resolve_short_id
                    _, epic_uuid = resolve_short_id(epic)
                else:
                    epic_uuid = uuid.UUID(epic)
                query = query.join(PRD).filter(PRD.epic_id == epic_uuid)
                
            elif project:
                if project.startswith('PRJ-'):
                    from ..core.short_id_generator import resolve_short_id
                    _, project_uuid = resolve_short_id(project)
                else:
                    project_uuid = uuid.UUID(project)
                query = query.join(PRD).join(Epic).filter(Epic.project_id == project_uuid)
            
            tasks = query.limit(20).all()  # Limit for safety
            
            if not tasks:
                console.print("[yellow]No unassigned tasks found[/yellow]")
                return
            
            console.print(f"Found {len(tasks)} unassigned tasks")
            
            # Process each task
            assigned_count = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task_progress = progress.add_task("Processing tasks...", total=len(tasks))
                
                for task in tasks:
                    task_display = task.get_display_id() if hasattr(task, 'get_display_id') else str(task.id)[:8]
                    
                    if dry_run:
                        console.print(f"[dim]Would assign: {task_display} - {task.title}[/dim]")
                    else:
                        # Execute task with auto-spawn
                        result = await enhanced_cli.execution_bridge.execute_project_task(
                            task.id, auto_spawn_agent=True
                        )
                        
                        if result.success:
                            console.print(f"✅ Assigned {task_display} to {result.agent_id}")
                            assigned_count += 1
                        else:
                            console.print(f"❌ Failed to assign {task_display}: {result.error_message}")
                    
                    progress.advance(task_progress)
            
            if not dry_run:
                console.print(f"\\n🎉 [green]Successfully assigned {assigned_count}/{len(tasks)} tasks[/green]")
            
        except Exception as e:
            console.print(f"[red]Error in auto-assignment: {e}[/red]")
    
    asyncio.run(_auto_assign())

async def _watch_task_execution(task_uuid: uuid.UUID):
    """Watch task execution progress."""
    console.print("\\n👁️ [dim]Watching execution (Press Ctrl+C to stop)...[/dim]")
    
    try:
        for i in range(10):  # Watch for up to 50 seconds
            await asyncio.sleep(5)
            
            status = await enhanced_cli.execution_bridge.monitor_task_execution(task_uuid)
            
            if status["status"] == "active":
                agent_status = status.get("agent_status", {})
                console.print(f"   📊 Agent Status: {agent_status.get('status', 'Unknown')}")
            else:
                console.print(f"   ℹ️ Status: {status['status']}")
            
    except KeyboardInterrupt:
        console.print("\\n🛑 Stopped watching")

def _display_execution_status(status: Dict[str, Any]):
    """Display detailed execution status."""
    execution = status.get("execution", {})
    agent_status = status.get("agent_status", {})
    
    # Create status panel
    status_content = f"""
[bold]Execution Status:[/bold] {status['status']}
[bold]Agent ID:[/bold] {execution.get('agent_id', 'N/A')}
[bold]Session:[/bold] {execution.get('session_name', 'N/A')}
[bold]Agent Status:[/bold] {agent_status.get('status', 'Unknown')}
[bold]Last Updated:[/bold] {status.get('last_updated', 'N/A')}
"""
    
    if execution.get('error_message'):
        status_content += f"[red][bold]Error:[/bold] {execution['error_message']}[/red]"
    
    console.print(Panel(status_content.strip(), title="Task Execution Status", border_style="blue"))
    
    # Show agent details if available
    if agent_status:
        agent_table = Table(title="Agent Details")
        agent_table.add_column("Property", style="cyan")
        agent_table.add_column("Value")
        
        for key, value in agent_status.items():
            if key not in ['status']:  # Already shown above
                agent_table.add_row(key.replace('_', ' ').title(), str(value))
        
        if agent_table.row_count > 0:
            console.print(agent_table)

# Export the command group
enhanced_project_commands = enhanced_project