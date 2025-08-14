"""
Unix CLI Setup and Installation Script

This script handles the installation and configuration of Unix-style
Agent Hive commands following kubectl/docker patterns.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


class UnixCLIInstaller:
    """Handles Unix CLI installation and setup."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.install_dirs = [
            Path.home() / ".local" / "bin",
            Path("/usr/local/bin")
        ]
        self.commands = [
            'status', 'get', 'logs', 'kill', 'create', 'delete',
            'scale', 'config', 'init', 'metrics', 'debug', 
            'doctor', 'version', 'help'
        ]
    
    def detect_install_location(self) -> Path:
        """Detect the best installation location."""
        for install_dir in self.install_dirs:
            if install_dir.exists() or install_dir.parent.exists():
                try:
                    install_dir.mkdir(parents=True, exist_ok=True)
                    if os.access(install_dir, os.W_OK):
                        return install_dir
                except PermissionError:
                    continue
        
        # Fallback: create local bin
        local_bin = Path.home() / ".local" / "bin"
        local_bin.mkdir(parents=True, exist_ok=True)
        return local_bin
    
    def create_wrapper_script(self, command: str, install_dir: Path) -> bool:
        """Create a wrapper script for a Unix command."""
        script_name = f"hive-{command}"
        script_path = install_dir / script_name
        
        python_exe = sys.executable
        cli_module = self.project_root / "app" / "cli" / "main.py"
        
        # Create bash wrapper script
        script_content = f"""#!/bin/bash
# LeanVibe Agent Hive Unix Command: {command}
# Auto-generated wrapper for enterprise CLI toolkit

# Set up Python path
export PYTHONPATH="{self.project_root}:$PYTHONPATH"

# Execute the command
exec "{python_exe}" -c "
import sys
sys.path.insert(0, '{self.project_root}')
from app.cli.main import hive_cli
sys.argv = ['hive', '{command}'] + sys.argv[1:]
try:
    hive_cli()
except KeyboardInterrupt:
    sys.exit(130)
except Exception as e:
    print(f'Error: {{e}}', file=sys.stderr)
    sys.exit(1)
" "$@"
"""
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to create {script_name}: {e}[/red]")
            return False
    
    def install_commands(self) -> bool:
        """Install all Unix-style commands."""
        install_dir = self.detect_install_location()
        
        console.print(f"[blue]Installing Unix commands to {install_dir}[/blue]")
        
        installed_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Installing commands...", total=len(self.commands))
            
            for command in self.commands:
                progress.update(task, description=f"Installing hive-{command}...")
                
                if self.create_wrapper_script(command, install_dir):
                    installed_count += 1
                
                progress.advance(task)
        
        if installed_count > 0:
            console.print(f"[green]✓ Successfully installed {installed_count} commands[/green]")
            self._post_install_setup(install_dir)
            return True
        else:
            console.print("[red]✗ Installation failed[/red]")
            return False
    
    def _post_install_setup(self, install_dir: Path):
        """Handle post-installation setup."""
        # Check if install_dir is in PATH
        path_env = os.environ.get('PATH', '')
        if str(install_dir) not in path_env:
            console.print(Panel(
                f"""[yellow]PATH Setup Required[/yellow]
                
Add this to your shell configuration file (~/.bashrc, ~/.zshrc, etc.):

    export PATH="{install_dir}:$PATH"

Then reload your shell or run:

    source ~/.bashrc  # or ~/.zshrc
""",
                title="Installation Complete",
                border_style="yellow"
            ))
        
        # Test installation
        test_command = install_dir / "hive-status"
        if test_command.exists():
            console.print(f"\n[green]Test your installation:[/green]")
            console.print(f"  {test_command} --help")
            console.print(f"  hive-get agents")
            console.print(f"  hive-doctor")
    
    def uninstall_commands(self) -> bool:
        """Remove all Unix-style commands."""
        removed_count = 0
        
        for install_dir in self.install_dirs:
            if not install_dir.exists():
                continue
            
            for command in self.commands:
                script_path = install_dir / f"hive-{command}"
                if script_path.exists():
                    try:
                        script_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        console.print(f"[red]Failed to remove {script_path}: {e}[/red]")
        
        if removed_count > 0:
            console.print(f"[green]✓ Removed {removed_count} commands[/green]")
            return True
        else:
            console.print("[yellow]No commands found to remove[/yellow]")
            return False
    
    def verify_installation(self) -> Dict[str, Any]:
        """Verify Unix CLI installation."""
        verification = {
            'installed_commands': [],
            'missing_commands': [],
            'path_configured': False,
            'total_commands': len(self.commands)
        }
        
        # Check each command
        for command in self.commands:
            script_name = f"hive-{command}"
            found = False
            
            for install_dir in self.install_dirs:
                script_path = install_dir / script_name
                if script_path.exists() and os.access(script_path, os.X_OK):
                    verification['installed_commands'].append(str(script_path))
                    found = True
                    break
            
            if not found:
                verification['missing_commands'].append(script_name)
        
        # Check PATH configuration
        path_env = os.environ.get('PATH', '')
        for install_dir in self.install_dirs:
            if str(install_dir) in path_env:
                verification['path_configured'] = True
                break
        
        return verification


@click.group()
def unix_cli_setup():
    """Unix CLI setup and management commands."""
    pass


@unix_cli_setup.command()
@click.option('--force', is_flag=True, help='Force reinstallation')
def install(force):
    """Install Unix-style CLI commands."""
    installer = UnixCLIInstaller()
    
    if not force:
        # Check if already installed
        verification = installer.verify_installation()
        if verification['installed_commands']:
            console.print("[yellow]Commands already installed. Use --force to reinstall.[/yellow]")
            return
    
    success = installer.install_commands()
    if success:
        console.print("\n[bold green]Unix CLI installation complete![/bold green]")
        console.print("\nAvailable commands:")
        
        # Show command overview
        from rich.columns import Columns
        command_list = [f"[cyan]hive-{cmd}[/cyan]" for cmd in installer.commands]
        console.print(Columns(command_list, equal=True, expand=False))


@unix_cli_setup.command()
def uninstall():
    """Remove Unix-style CLI commands."""
    installer = UnixCLIInstaller()
    
    if click.confirm("Remove all Unix CLI commands?"):
        installer.uninstall_commands()


@unix_cli_setup.command()
def verify():
    """Verify Unix CLI installation."""
    installer = UnixCLIInstaller()
    verification = installer.verify_installation()
    
    console.print("[bold]Unix CLI Installation Status[/bold]\n")
    
    # Installation summary
    installed_count = len(verification['installed_commands'])
    missing_count = len(verification['missing_commands'])
    total_count = verification['total_commands']
    
    if installed_count == total_count:
        console.print(f"[green]✓ All {total_count} commands installed[/green]")
    else:
        console.print(f"[yellow]⚠ {installed_count}/{total_count} commands installed[/yellow]")
        if missing_count > 0:
            console.print(f"[red]Missing: {', '.join(verification['missing_commands'])}[/red]")
    
    # PATH configuration
    if verification['path_configured']:
        console.print("[green]✓ PATH configured correctly[/green]")
    else:
        console.print("[red]✗ Installation directory not in PATH[/red]")
    
    # Show installed commands
    if verification['installed_commands']:
        console.print(f"\n[bold]Installed Commands:[/bold]")
        for cmd_path in verification['installed_commands']:
            console.print(f"  [cyan]{Path(cmd_path).name}[/cyan] → {cmd_path}")


@unix_cli_setup.command() 
def demo():
    """Demonstrate Unix CLI commands."""
    console.print("[bold blue]Unix CLI Demo - LeanVibe Agent Hive[/bold blue]\n")
    
    demo_commands = [
        ("hive-status", "System status and health check"),
        ("hive-get agents", "List all active agents"),
        ("hive-config --list", "Show current configuration"),
        ("hive-metrics", "Display system metrics"),
        ("hive-doctor", "Diagnose system health"),
        ("hive-help", "Show all available commands")
    ]
    
    console.print("Try these commands:")
    
    from rich.table import Table
    table = Table()
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    
    for cmd, desc in demo_commands:
        table.add_row(cmd, desc)
    
    console.print(table)
    
    console.print(f"\n[dim]Note: Commands follow Unix philosophy - focused, composable, and pipeable[/dim]")


if __name__ == '__main__':
    unix_cli_setup()