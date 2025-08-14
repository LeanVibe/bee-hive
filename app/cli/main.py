"""
Main CLI entry point and command dispatcher for Unix-style Agent Hive commands.

This module provides both individual command access and a unified 'hive' command
that follows kubectl/docker patterns for enterprise CLI tools.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .unix_commands import (
    hive_status, hive_get, hive_logs, hive_kill, hive_create, hive_delete,
    hive_scale, hive_config, hive_init, hive_metrics, hive_debug, 
    hive_doctor, hive_version, hive_help
)

console = Console()

# Command registry mapping command names to functions
COMMAND_REGISTRY = {
    'status': hive_status,
    'get': hive_get,
    'logs': hive_logs,
    'kill': hive_kill,
    'create': hive_create,
    'delete': hive_delete,
    'scale': hive_scale,
    'config': hive_config,
    'init': hive_init,
    'metrics': hive_metrics,
    'debug': hive_debug,
    'doctor': hive_doctor,
    'version': hive_version,
    'help': hive_help
}


@click.group(invoke_without_command=True, no_args_is_help=False)
@click.option('--version', is_flag=True, help='Show version information')
@click.pass_context
def hive_cli(ctx, version):
    """
    LeanVibe Agent Hive - Unix Philosophy CLI Toolkit
    
    Enterprise-grade multi-agent orchestration with kubectl/docker-style commands.
    Each command follows Unix principles: focused, composable, and pipeable.
    
    Examples:
    
      # System status (kubectl style)
      hive status --watch
      hive get agents -o json
      
      # Resource management (docker style)
      hive create --count 3 --type worker
      hive scale agent-123 5
      hive kill agent-456
      
      # Configuration (git style)  
      hive config api.base http://localhost:8000
      hive config --list
      
      # Monitoring and debugging
      hive logs -f -n 100
      hive metrics --watch
      hive doctor --fix
    """
    if version:
        ctx.invoke(hive_version)
        return
    
    if ctx.invoked_subcommand is None:
        # Show overview if no subcommand
        _show_overview()


def _show_overview():
    """Show system overview and command suggestions."""
    console.print("[bold blue]LeanVibe Agent Hive - Unix CLI Toolkit[/bold blue]\n")
    
    # System status summary
    from .unix_commands import ctx
    status = ctx.api_call("health")
    
    if status:
        console.print("[green]✓[/green] System is running")
        console.print(f"  API: {ctx.api_base}")
        
        # Show quick stats
        debug_info = ctx.api_call("debug-agents")
        if debug_info and 'agents' in debug_info:
            agent_count = len(debug_info['agents'])
            console.print(f"  Agents: {agent_count} active")
    else:
        console.print("[red]✗[/red] System not responding")
        console.print("  Run 'hive doctor' to diagnose issues")
    
    console.print("\n[bold]Common Commands:[/bold]")
    
    # Quick command reference
    quick_commands = [
        ("hive status", "System status and health"),
        ("hive get agents", "List all agents"),
        ("hive logs -f", "Follow system logs"),
        ("hive create --count 2", "Create new agents"),
        ("hive doctor", "Diagnose system issues"),
    ]
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    
    for cmd, desc in quick_commands:
        table.add_row(cmd, desc)
    
    console.print(table)
    console.print("\nRun 'hive --help' for all commands or 'hive help' for detailed usage.")


# Register all commands as subcommands
for name, func in COMMAND_REGISTRY.items():
    # Clone the command and add it to the group
    cmd = click.Command(
        name=name,
        callback=func.callback,
        params=func.params,
        help=func.help,
        short_help=func.short_help
    )
    hive_cli.add_command(cmd)


def install_unix_commands():
    """
    Install individual Unix-style commands.
    
    This creates individual executables for each command following Unix philosophy:
    hive-status, hive-get, hive-logs, etc.
    """
    import os
    import stat
    
    # Get installation directory (prefer ~/.local/bin, fallback to /usr/local/bin)
    install_dirs = [
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin")
    ]
    
    install_dir = None
    for dir_path in install_dirs:
        if dir_path.exists() or dir_path.parent.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            if os.access(dir_path.parent, os.W_OK):
                install_dir = dir_path
                break
    
    if not install_dir:
        console.print("[red]Error: No writable installation directory found[/red]")
        console.print("Please ensure ~/.local/bin exists and is writable")
        sys.exit(1)
    
    # Python executable
    python_exe = sys.executable
    
    # Create individual command scripts
    commands_installed = []
    
    for command_name in COMMAND_REGISTRY.keys():
        script_name = f"hive-{command_name}"
        script_path = install_dir / script_name
        
        # Create wrapper script
        script_content = f"""#!/bin/bash
# LeanVibe Agent Hive Unix Command: {command_name}
# Auto-generated wrapper for hive {command_name}

exec "{python_exe}" -c "
import sys
sys.path.insert(0, '{Path(__file__).parent.parent.absolute()}')
from app.cli.main import hive_cli
sys.argv = ['hive'] + ['{command_name}'] + sys.argv[1:]
hive_cli()
" "$@"
"""
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
            commands_installed.append(script_name)
            
        except Exception as e:
            console.print(f"[red]Failed to install {script_name}: {e}[/red]")
    
    if commands_installed:
        console.print(f"[green]Installed {len(commands_installed)} Unix commands to {install_dir}[/green]")
        console.print("\nAvailable commands:")
        
        # Show installed commands in columns
        from rich.columns import Columns
        command_list = [f"[cyan]{cmd}[/cyan]" for cmd in sorted(commands_installed)]
        console.print(Columns(command_list, equal=True, expand=False))
        
        console.print(f"\nEnsure {install_dir} is in your PATH:")
        console.print(f"  export PATH=\"{install_dir}:$PATH\"")
        
        # Test installation
        test_cmd = install_dir / "hive-status"
        if test_cmd.exists():
            console.print(f"\nTest installation:")
            console.print(f"  {test_cmd} --help")
    else:
        console.print("[red]No commands were installed successfully[/red]")


def uninstall_unix_commands():
    """Remove individual Unix-style commands."""
    install_dirs = [
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin")
    ]
    
    commands_removed = []
    
    for install_dir in install_dirs:
        if not install_dir.exists():
            continue
            
        for command_name in COMMAND_REGISTRY.keys():
            script_name = f"hive-{command_name}"
            script_path = install_dir / script_name
            
            if script_path.exists():
                try:
                    script_path.unlink()
                    commands_removed.append(str(script_path))
                except Exception as e:
                    console.print(f"[red]Failed to remove {script_path}: {e}[/red]")
    
    if commands_removed:
        console.print(f"[green]Removed {len(commands_removed)} Unix commands[/green]")
    else:
        console.print("No Unix commands found to remove")


# Individual command entry points for direct execution
def hive_status_main():
    """Entry point for hive-status command."""
    hive_status(standalone_mode=False)

def hive_get_main():
    """Entry point for hive-get command."""
    hive_get(standalone_mode=False)

def hive_logs_main():
    """Entry point for hive-logs command."""
    hive_logs(standalone_mode=False)

# ... (similar entry points for other commands)


if __name__ == '__main__':
    hive_cli()