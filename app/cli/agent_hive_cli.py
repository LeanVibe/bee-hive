"""
AgentHiveCLI - Main CLI Class for LeanVibe Agent Hive 2.0

This module provides the AgentHiveCLI class that integrates with the existing Unix-style
CLI architecture while providing the expected class-based interface for imports.

Epic C Phase 2: CLI Restoration and Integration
"""

import asyncio
import sys
from typing import List, Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Import existing CLI commands
from .unix_commands import (
    hive_status, hive_get, hive_logs, hive_create, hive_delete,
    hive_scale, hive_config, hive_init, hive_metrics, hive_debug, 
    hive_doctor, hive_version, hive_help
)
from .main import COMMAND_REGISTRY

console = Console()


class AgentHiveCLI:
    """
    Main CLI class for LeanVibe Agent Hive 2.0.
    
    This class provides a unified interface to all CLI functionality while
    integrating with the existing Unix-style command architecture.
    
    Features:
    - Agent management commands
    - Task management commands  
    - System health and monitoring
    - Integration with new API endpoints
    - Unix philosophy: focused, composable, pipeable
    """
    
    def __init__(self):
        """Initialize AgentHiveCLI with connection to API endpoints."""
        self.console = Console()
        self.api_base = "http://localhost:8000"
        
    def execute_command(self, command: str, args: List[str] = None) -> int:
        """
        Execute a CLI command with given arguments.
        
        Args:
            command: Command name (e.g., 'status', 'get', 'create')
            args: List of command arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        if args is None:
            args = []
            
        try:
            if command in COMMAND_REGISTRY:
                cmd_func = COMMAND_REGISTRY[command]
                # Convert args to click context if needed
                ctx = click.Context(cmd_func)
                ctx.params = self._parse_args_to_params(args)
                return cmd_func.callback(ctx.params) or 0
            else:
                self.console.print(f"[red]Unknown command: {command}[/red]")
                self.console.print("Run 'hive help' for available commands")
                return 1
        except Exception as e:
            self.console.print(f"[red]Command failed: {e}[/red]")
            return 1
    
    def _parse_args_to_params(self, args: List[str]) -> Dict[str, Any]:
        """Parse command-line arguments to parameter dictionary."""
        params = {}
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith('--'):
                # Long option
                key = arg[2:]
                if i + 1 < len(args) and not args[i + 1].startswith('-'):
                    params[key] = args[i + 1]
                    i += 2
                else:
                    params[key] = True
                    i += 1
            elif arg.startswith('-'):
                # Short option
                key = arg[1:]
                if i + 1 < len(args) and not args[i + 1].startswith('-'):
                    params[key] = args[i + 1]
                    i += 2
                else:
                    params[key] = True
                    i += 1
            else:
                # Positional argument
                if 'resource' not in params:
                    params['resource'] = arg
                i += 1
        return params
    
    # Agent Management Methods
    def create_agent(self, name: str, agent_type: str, capabilities: List[str] = None) -> Dict[str, Any]:
        """
        Create a new agent using the API endpoints.
        
        Args:
            name: Agent name
            agent_type: Agent type (backend-engineer, qa-test-guardian, etc.)
            capabilities: List of agent capabilities
            
        Returns:
            API response with agent details
        """
        import requests
        
        try:
            data = {
                "name": name,
                "type": agent_type,
                "capabilities": capabilities or []
            }
            
            response = requests.post(f"{self.api_base}/api/v1/agents", json=data)
            
            if response.status_code == 201:
                agent_data = response.json()
                self.console.print(f"[green]✅ Agent created successfully[/green]")
                self.console.print(f"Agent ID: {agent_data.get('id')}")
                self.console.print(f"Name: {agent_data.get('name')}")
                self.console.print(f"Type: {agent_data.get('type')}")
                return agent_data
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                self.console.print(f"[red]❌ Failed to create agent: {error_msg}[/red]")
                return {"error": error_msg}
                
        except requests.exceptions.ConnectionError:
            self.console.print(f"[red]❌ Cannot connect to API at {self.api_base}[/red]")
            self.console.print("Run 'hive doctor' to diagnose connection issues")
            return {"error": "API connection failed"}
        except Exception as e:
            self.console.print(f"[red]❌ Agent creation failed: {e}[/red]")
            return {"error": str(e)}
    
    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent details by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            API response with agent details
        """
        import requests
        
        try:
            response = requests.get(f"{self.api_base}/api/v1/agents/{agent_id}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                self.console.print(f"[yellow]⚠️  Agent not found: {agent_id}[/yellow]")
                return {"error": "Agent not found"}
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                self.console.print(f"[red]❌ Failed to get agent: {error_msg}[/red]")
                return {"error": error_msg}
                
        except Exception as e:
            self.console.print(f"[red]❌ Failed to get agent: {e}[/red]")
            return {"error": str(e)}
    
    def list_agents(self, status: str = None) -> List[Dict[str, Any]]:
        """
        List all agents with optional status filtering.
        
        Args:
            status: Optional status filter (active, inactive, etc.)
            
        Returns:
            List of agent details
        """
        import requests
        
        try:
            params = {}
            if status:
                params['status'] = status
                
            response = requests.get(f"{self.api_base}/api/v1/agents", params=params)
            
            if response.status_code == 200:
                data = response.json()
                agents = data.get('agents', [])
                
                # Display agents in table format
                table = Table(title="Agents")
                table.add_column("ID", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Type", style="blue")
                table.add_column("Status", style="yellow")
                table.add_column("Created", style="magenta")
                
                for agent in agents:
                    table.add_row(
                        agent.get('id', '')[:8] + '...' if agent.get('id') else '',
                        agent.get('name', ''),
                        agent.get('type', ''),
                        agent.get('status', ''),
                        agent.get('created_at', '')[:19] if agent.get('created_at') else ''
                    )
                
                self.console.print(table)
                return agents
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                self.console.print(f"[red]❌ Failed to list agents: {error_msg}[/red]")
                return []
                
        except Exception as e:
            self.console.print(f"[red]❌ Failed to list agents: {e}[/red]")
            return []
    
    # Task Management Methods
    def create_task(self, description: str, agent_id: str = None, priority: str = "medium") -> Dict[str, Any]:
        """
        Create a new task using the API endpoints.
        
        Args:
            description: Task description
            agent_id: Optional agent ID to assign task to
            priority: Task priority (low, medium, high, critical)
            
        Returns:
            API response with task details
        """
        import requests
        
        try:
            data = {
                "description": description,
                "priority": priority
            }
            
            if agent_id:
                data["agent_id"] = agent_id
                
            response = requests.post(f"{self.api_base}/api/v1/tasks", json=data)
            
            if response.status_code == 201:
                task_data = response.json()
                self.console.print(f"[green]✅ Task created successfully[/green]")
                self.console.print(f"Task ID: {task_data.get('id')}")
                self.console.print(f"Description: {task_data.get('description')}")
                self.console.print(f"Priority: {task_data.get('priority')}")
                if task_data.get('agent_id'):
                    self.console.print(f"Assigned to: {task_data.get('agent_id')}")
                return task_data
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                self.console.print(f"[red]❌ Failed to create task: {error_msg}[/red]")
                return {"error": error_msg}
                
        except Exception as e:
            self.console.print(f"[red]❌ Task creation failed: {e}[/red]")
            return {"error": str(e)}
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status and progress.
        
        Args:
            task_id: Task identifier
            
        Returns:
            API response with task status
        """
        import requests
        
        try:
            response = requests.get(f"{self.api_base}/api/v1/tasks/{task_id}/status")
            
            if response.status_code == 200:
                task_data = response.json()
                
                # Display task status
                status = task_data.get('status', 'unknown')
                progress = task_data.get('progress', 0)
                
                status_color = {
                    'pending': 'yellow',
                    'in_progress': 'blue', 
                    'completed': 'green',
                    'failed': 'red',
                    'cancelled': 'magenta'
                }.get(status, 'white')
                
                self.console.print(f"Task Status: [{status_color}]{status}[/{status_color}]")
                self.console.print(f"Progress: {progress}%")
                
                return task_data
            elif response.status_code == 404:
                self.console.print(f"[yellow]⚠️  Task not found: {task_id}[/yellow]")
                return {"error": "Task not found"}
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                self.console.print(f"[red]❌ Failed to get task status: {error_msg}[/red]")
                return {"error": error_msg}
                
        except Exception as e:
            self.console.print(f"[red]❌ Failed to get task status: {e}[/red]")
            return {"error": str(e)}
    
    # System Methods
    def system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        try:
            # Use existing hive status command
            return self.execute_command('status', ['--format', 'json'])
        except Exception as e:
            self.console.print(f"[red]❌ System health check failed: {e}[/red]")
            return {"error": str(e)}
    
    def system_stats(self) -> Dict[str, Any]:
        """Get system statistics and metrics."""
        try:
            # Use existing hive metrics command
            return self.execute_command('metrics', ['--format', 'json'])
        except Exception as e:
            self.console.print(f"[red]❌ System stats failed: {e}[/red]")
            return {"error": str(e)}


# Export the main class for imports
__all__ = ['AgentHiveCLI']


# CLI command group for click integration
@click.group()
def agent_hive_cli():
    """LeanVibe Agent Hive 2.0 CLI"""
    pass


@agent_hive_cli.command()
@click.option('--name', required=True, help='Agent name')
@click.option('--type', required=True, help='Agent type')
@click.option('--capabilities', multiple=True, help='Agent capabilities')
def create_agent(name: str, type: str, capabilities: tuple):
    """Create a new agent"""
    cli = AgentHiveCLI()
    cli.create_agent(name, type, list(capabilities))


@agent_hive_cli.command()
@click.argument('agent_id')
def get_agent(agent_id: str):
    """Get agent details by ID"""
    cli = AgentHiveCLI()
    cli.get_agent(agent_id)


@agent_hive_cli.command()
@click.option('--status', help='Filter by status')
def list_agents(status: str):
    """List all agents"""
    cli = AgentHiveCLI()
    cli.list_agents(status)


@agent_hive_cli.command()
@click.option('--description', required=True, help='Task description')
@click.option('--agent-id', help='Agent ID to assign task to')
@click.option('--priority', default='medium', help='Task priority')
def create_task(description: str, agent_id: str, priority: str):
    """Create a new task"""
    cli = AgentHiveCLI()
    cli.create_task(description, agent_id, priority)


@agent_hive_cli.command()
@click.argument('task_id')
def task_status(task_id: str):
    """Get task status"""
    cli = AgentHiveCLI()
    cli.get_task_status(task_id)


@agent_hive_cli.command()
def health():
    """System health check"""
    cli = AgentHiveCLI()
    cli.system_health()


if __name__ == '__main__':
    agent_hive_cli()