"""
Unix Philosophy CLI Commands for LeanVibe Agent Hive 2.0

Following the Unix philosophy: "Do one thing and do it well"
Each command is focused, composable, and follows consistent patterns.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import click
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

# Common configuration and utilities
@dataclass
class HiveContext:
    """Shared context for all hive commands."""
    api_base: str = "http://localhost:8000"
    config_dir: Path = Path.home() / ".config" / "agent-hive"
    
    def __post_init__(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def api_call(self, endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
        """Make API call with consistent error handling."""
        try:
            url = f"{self.api_base}/{endpoint.lstrip('/')}"
            if method == "GET":
                response = requests.get(url, timeout=5)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=5)
            else:
                response = requests.request(method, url, json=data, timeout=5)
            
            if response.status_code == 200:
                return response.json() if response.content else {}
            return None
        except Exception:
            return None

# Global context instance
ctx = HiveContext()


# Core System Commands (kubectl-style)

@click.command()
@click.option('--format', '-f', type=click.Choice(['json', 'table', 'wide']), default='table')
@click.option('--watch', '-w', is_flag=True, help='Watch for changes')
def hive_status(format, watch):
    """Get system status and health information."""
    if watch:
        import time
        while True:
            click.clear()
            _show_status(format)
            time.sleep(2)
    else:
        _show_status(format)

def _show_status(format_type):
    """Internal status display function."""
    status = ctx.api_call("status")
    health = ctx.api_call("health")
    
    if format_type == 'json':
        click.echo(json.dumps({"status": status, "health": health}, indent=2))
    elif format_type == 'table':
        if status:
            table = Table(title="Agent Hive Status")
            table.add_column("Component")
            table.add_column("Status")
            table.add_column("Uptime")
            
            for component, info in status.items():
                if isinstance(info, dict):
                    table.add_row(component, str(info.get('status', 'unknown')), str(info.get('uptime', 'unknown')))
            
            console.print(table)
        else:
            console.print("[red]System not responding[/red]")


@click.command()
@click.option('--output', '-o', type=click.Choice(['json', 'yaml', 'table']), default='table')
@click.argument('resource', required=False, default='agents')
def hive_get(output, resource):
    """Get resources (agents, tasks, workflows)."""
    endpoint_map = {
        'agents': 'debug-agents',
        'tasks': 'api/tasks/active',
        'workflows': 'api/workflows/active',
        'metrics': 'metrics'
    }
    
    endpoint = endpoint_map.get(resource, resource)
    data = ctx.api_call(endpoint)
    
    if not data:
        console.print(f"[red]Failed to get {resource}[/red]")
        sys.exit(1)
    
    if output == 'json':
        click.echo(json.dumps(data, indent=2))
    elif output == 'table' and resource == 'agents':
        _display_agents_table(data)
    else:
        click.echo(json.dumps(data, indent=2))

def _display_agents_table(agents_data):
    """Display agents in table format."""
    if not agents_data or 'agents' not in agents_data:
        console.print("No agents found")
        return
    
    table = Table(title="Active Agents")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Tasks", justify="right")
    table.add_column("Uptime")
    
    for agent in agents_data['agents']:
        table.add_row(
            agent.get('id', 'unknown')[:12] + '...',
            agent.get('status', 'unknown'),
            str(agent.get('active_tasks', 0)),
            agent.get('uptime', 'unknown')
        )
    
    console.print(table)


@click.command()
@click.option('--lines', '-n', type=int, default=50, help='Number of lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Filter by log level')
@click.argument('component', required=False, default='all')
def hive_logs(lines, follow, level, component):
    """View system logs (docker logs style)."""
    params = {'lines': lines}
    if level:
        params['level'] = level
    if component != 'all':
        params['component'] = component
    
    if follow:
        # Implement log following
        console.print(f"Following logs for {component}...")
        # This would connect to a streaming endpoint
        while True:
            logs = ctx.api_call("logs", method="GET")
            if logs and 'entries' in logs:
                for entry in logs['entries'][-5:]:  # Show last 5 new entries
                    _format_log_entry(entry)
            time.sleep(1)
    else:
        logs = ctx.api_call("logs")
        if logs and 'entries' in logs:
            for entry in logs['entries'][-lines:]:
                _format_log_entry(entry)

def _format_log_entry(entry):
    """Format a single log entry."""
    timestamp = entry.get('timestamp', '')
    level = entry.get('level', 'INFO')
    message = entry.get('message', '')
    component = entry.get('component', 'system')
    
    level_colors = {
        'DEBUG': 'dim',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red'
    }
    
    color = level_colors.get(level, 'white')
    console.print(f"[{color}]{timestamp} [{level}] {component}: {message}[/{color}]")


@click.command()
@click.argument('agent_id')
@click.option('--reason', default='user_request', help='Reason for termination')
def hive_kill(agent_id, reason):
    """Terminate a specific agent."""
    data = {'reason': reason}
    result = ctx.api_call(f"api/agents/{agent_id}/terminate", method="POST", data=data)
    
    if result:
        console.print(f"[green]Agent {agent_id} terminated successfully[/green]")
    else:
        console.print(f"[red]Failed to terminate agent {agent_id}[/red]")
        sys.exit(1)


# Resource Management Commands

@click.command()
@click.option('--count', '-c', type=int, default=1, help='Number of agents to create')
@click.option('--type', '-t', default='general', help='Agent type')
@click.option('--config', '-f', help='Config file path')
def hive_create(count, type, config):
    """Create new agents or resources."""
    agent_config = {'type': type, 'count': count}
    
    if config:
        config_path = Path(config)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    agent_config.update(json.load(f))
            except Exception as e:
                console.print(f"[red]Error reading config: {e}[/red]")
                sys.exit(1)
    
    result = ctx.api_call("api/agents/create", method="POST", data=agent_config)
    
    if result:
        console.print(f"[green]Created {count} agent(s) successfully[/green]")
        if 'agents' in result:
            for agent in result['agents']:
                console.print(f"  - {agent['id']}: {agent['status']}")
    else:
        console.print("[red]Failed to create agents[/red]")
        sys.exit(1)


@click.command()
@click.argument('resource_type')
@click.argument('resource_name')
def hive_delete(resource_type, resource_name):
    """Delete resources (agents, tasks, workflows)."""
    endpoint_map = {
        'agent': f'api/agents/{resource_name}',
        'task': f'api/tasks/{resource_name}', 
        'workflow': f'api/workflows/{resource_name}'
    }
    
    endpoint = endpoint_map.get(resource_type)
    if not endpoint:
        console.print(f"[red]Unknown resource type: {resource_type}[/red]")
        sys.exit(1)
    
    if click.confirm(f"Delete {resource_type} '{resource_name}'?"):
        result = ctx.api_call(endpoint, method="DELETE")
        if result:
            console.print(f"[green]{resource_type} '{resource_name}' deleted[/green]")
        else:
            console.print(f"[red]Failed to delete {resource_type}[/red]")
            sys.exit(1)


@click.command()
@click.argument('agent_id')
@click.argument('replicas', type=int)
def hive_scale(agent_id, replicas):
    """Scale agent instances up or down."""
    data = {'replicas': replicas}
    result = ctx.api_call(f"api/agents/{agent_id}/scale", method="POST", data=data)
    
    if result:
        console.print(f"[green]Scaled agent {agent_id} to {replicas} replicas[/green]")
    else:
        console.print(f"[red]Failed to scale agent {agent_id}[/red]")
        sys.exit(1)


# Configuration and Setup Commands

@click.command()
@click.argument('key', required=False)
@click.argument('value', required=False)
@click.option('--unset', is_flag=True, help='Unset a configuration key')
@click.option('--list', '-l', is_flag=True, help='List all configuration')
def hive_config(key, value, unset, list):
    """Manage configuration (git config style)."""
    config_file = ctx.config_dir / "config.json"
    
    # Load existing config
    config = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
        except Exception:
            pass
    
    if list or (not key and not value):
        # List all configuration
        for k, v in config.items():
            console.print(f"{k}={v}")
    elif unset and key:
        # Unset configuration
        if key in config:
            del config[key]
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            console.print(f"Unset {key}")
        else:
            console.print(f"Key '{key}' not found")
    elif key and value:
        # Set configuration
        config[key] = value
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        console.print(f"Set {key}={value}")
    elif key and not value:
        # Get configuration
        if key in config:
            console.print(config[key])
        else:
            console.print(f"Key '{key}' not found")
            sys.exit(1)


@click.command()
@click.option('--quick', is_flag=True, help='Quick setup with defaults')
@click.option('--reset', is_flag=True, help='Reset to default configuration')
def hive_init(quick, reset):
    """Initialize or reset system configuration."""
    config_file = ctx.config_dir / "config.json"
    
    if reset or not config_file.exists():
        default_config = {
            "api_base": "http://localhost:8000",
            "auto_start": True,
            "max_agents": 10,
            "log_level": "INFO",
            "workspace_dir": str(ctx.config_dir / "workspaces")
        }
        
        if not quick and not reset:
            # Interactive setup
            default_config["api_base"] = click.prompt("API Base URL", default=default_config["api_base"])
            default_config["max_agents"] = click.prompt("Max Agents", type=int, default=default_config["max_agents"])
            default_config["log_level"] = click.prompt("Log Level", default=default_config["log_level"])
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        console.print("[green]Configuration initialized[/green]")
    else:
        console.print("Configuration already exists. Use --reset to overwrite.")


# Monitoring and Debug Commands

@click.command()
@click.option('--watch', '-w', is_flag=True, help='Watch metrics continuously')
@click.option('--format', type=click.Choice(['json', 'prometheus']), default='json')
def hive_metrics(watch, format):
    """Display system metrics."""
    if watch:
        while True:
            click.clear()
            _show_metrics(format)
            time.sleep(5)
    else:
        _show_metrics(format)

def _show_metrics(format_type):
    """Internal metrics display function."""
    metrics = ctx.api_call("metrics")
    
    if format_type == 'json':
        click.echo(json.dumps(metrics, indent=2))
    elif format_type == 'prometheus':
        if metrics:
            for key, value in metrics.items():
                console.print(f"{key} {value}")
    else:
        if metrics:
            table = Table(title="System Metrics")
            table.add_column("Metric")
            table.add_column("Value")
            
            for key, value in metrics.items():
                table.add_row(key, str(value))
            
            console.print(table)


@click.command()
@click.argument('component', required=False, default='system')
def hive_debug(component):
    """Debug system or component issues."""
    debug_info = ctx.api_call(f"debug/{component}")
    
    if debug_info:
        console.print(Panel(json.dumps(debug_info, indent=2), title=f"Debug: {component}"))
    else:
        console.print(f"[red]No debug information available for {component}[/red]")


@click.command()
@click.option('--fix', is_flag=True, help='Attempt to fix issues automatically')
def hive_doctor(fix):
    """Diagnose system health and suggest fixes."""
    health_check = ctx.api_call("health")
    
    if not health_check:
        console.print("[red]System not responding[/red]")
        console.print("\nSuggested fixes:")
        console.print("1. Check if services are running")
        console.print("2. Verify network connectivity") 
        console.print("3. Check configuration files")
        return
    
    issues = []
    if health_check.get('database_status') != 'healthy':
        issues.append("Database connection issues")
    if health_check.get('redis_status') != 'healthy':
        issues.append("Redis connection issues")
    if health_check.get('agent_count', 0) == 0:
        issues.append("No active agents")
    
    if not issues:
        console.print("[green]System is healthy[/green]")
    else:
        console.print("[yellow]Issues found:[/yellow]")
        for issue in issues:
            console.print(f"  - {issue}")
        
        if fix:
            console.print("\n[blue]Attempting automatic fixes...[/blue]")
            # Implement automatic fix attempts
            result = ctx.api_call("system/auto-fix", method="POST")
            if result and result.get('success'):
                console.print("[green]Issues resolved[/green]")
            else:
                console.print("[red]Automatic fixes failed[/red]")


# Utility Commands

@click.command()
@click.option('--format', type=click.Choice(['json', 'table']), default='table')
def hive_version(format):
    """Show version information."""
    version_info = {
        "version": "2.0.0",
        "build": "unix-cli",
        "api_version": "v2"
    }
    
    # Try to get server version
    server_info = ctx.api_call("version")
    if server_info:
        version_info.update(server_info)
    
    if format == 'json':
        click.echo(json.dumps(version_info, indent=2))
    else:
        table = Table(title="Version Information")
        table.add_column("Component")
        table.add_column("Version")
        
        for key, value in version_info.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)


@click.command()
@click.argument('command_name', required=False)
def hive_help(command_name):
    """Show help for commands."""
    if command_name:
        # Show specific command help
        commands = {
            'status': 'Get system status and health information',
            'get': 'Get resources (agents, tasks, workflows)',
            'logs': 'View system logs',
            'kill': 'Terminate a specific agent',
            'create': 'Create new agents or resources',
            'delete': 'Delete resources',
            'scale': 'Scale agent instances',
            'config': 'Manage configuration',
            'init': 'Initialize system configuration',
            'metrics': 'Display system metrics',
            'debug': 'Debug system issues',
            'doctor': 'Diagnose system health',
            'version': 'Show version information'
        }
        
        if command_name in commands:
            console.print(f"[bold]{command_name}[/bold]: {commands[command_name]}")
        else:
            console.print(f"Unknown command: {command_name}")
    else:
        # Show all commands
        console.print("[bold]Available Commands:[/bold]")
        commands = [
            ("status", "System status and health"),
            ("get", "Get resources"),
            ("logs", "View logs"),
            ("create", "Create resources"),
            ("delete", "Delete resources"),
            ("scale", "Scale agents"),
            ("config", "Configuration management"),
            ("metrics", "System metrics"),
            ("debug", "Debug utilities"),
            ("doctor", "Health diagnostics")
        ]
        
        table = Table()
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        
        for cmd, desc in commands:
            table.add_row(f"hive {cmd}", desc)
        
        console.print(table)


# Export all commands for registration
unix_commands = [
    hive_status,
    hive_get,
    hive_logs, 
    hive_kill,
    hive_create,
    hive_delete,
    hive_scale,
    hive_config,
    hive_init,
    hive_metrics,
    hive_debug,
    hive_doctor,
    hive_version,
    hive_help
]