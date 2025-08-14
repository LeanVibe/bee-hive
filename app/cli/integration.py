"""
Integration layer between existing CLI and new Unix-style commands.

This module provides backward compatibility while enabling the new Unix philosophy CLI.
"""

import click
from rich.console import Console
from rich.panel import Panel

from .main import hive_cli, install_unix_commands, uninstall_unix_commands
from .setup import UnixCLIInstaller

console = Console()


def integrate_with_dx_cli():
    """Add Unix CLI commands to the existing dx_cli structure."""
    
    # Add unix-cli group to existing lv command
    @click.group(name='unix')
    def unix_group():
        """Unix philosophy CLI commands (kubectl/docker style)."""
        pass
    
    # Add installation commands
    @unix_group.command()
    @click.option('--force', is_flag=True, help='Force reinstallation')
    def install(force):
        """Install Unix-style individual commands."""
        installer = UnixCLIInstaller()
        
        console.print("[blue]Installing Unix-style CLI commands...[/blue]")
        success = installer.install_commands()
        
        if success:
            console.print(Panel(
                """[green]Unix CLI Installation Complete![/green]

The following individual commands are now available:

• hive-status    - System status and health
• hive-get       - Get resources (agents, tasks, workflows)  
• hive-logs      - View system logs
• hive-create    - Create new agents/resources
• hive-delete    - Delete resources
• hive-scale     - Scale agent instances
• hive-config    - Configuration management
• hive-metrics   - System metrics
• hive-debug     - Debug utilities
• hive-doctor    - Health diagnostics

These commands follow Unix philosophy: focused, composable, and pipeable.
Perfect for automation scripts and power users.
""",
                title="Unix CLI Ready",
                border_style="green"
            ))
        else:
            console.print("[red]Installation failed. Check permissions and try again.[/red]")
    
    @unix_group.command()
    def uninstall():
        """Remove Unix-style individual commands."""
        installer = UnixCLIInstaller()
        
        if click.confirm("Remove all Unix CLI commands?"):
            installer.uninstall_commands()
    
    @unix_group.command()
    def verify():
        """Verify Unix CLI installation."""
        installer = UnixCLIInstaller()
        verification = installer.verify_installation()
        
        installed_count = len(verification['installed_commands'])
        total_count = verification['total_commands']
        
        if installed_count == total_count:
            console.print(f"[green]✓ All {total_count} Unix commands installed and working[/green]")
        else:
            console.print(f"[yellow]⚠ {installed_count}/{total_count} commands available[/yellow]")
            
            if verification['missing_commands']:
                console.print("[red]Missing commands:[/red]")
                for cmd in verification['missing_commands']:
                    console.print(f"  - {cmd}")
        
        if not verification['path_configured']:
            console.print("[yellow]⚠ Installation directory may not be in PATH[/yellow]")
    
    @unix_group.command()
    def demo():
        """Show demo of Unix CLI commands."""
        console.print("[bold blue]Unix CLI Demo Commands[/bold blue]\n")
        
        demo_commands = [
            ("# System monitoring", ""),
            ("hive-status --watch", "Watch system status"),
            ("hive-get agents -o json | jq '.agents[].id'", "Get agent IDs (pipeable)"),
            ("hive-logs -f -n 100", "Follow last 100 log entries"),
            ("", ""),
            ("# Resource management", ""), 
            ("hive-create --count 3 --type worker", "Create 3 worker agents"),
            ("hive-scale agent-123 5", "Scale agent to 5 replicas"),
            ("hive-delete agent agent-456", "Delete specific agent"),
            ("", ""),
            ("# Configuration", ""),
            ("hive-config api.base http://localhost:8000", "Set API base URL"),
            ("hive-config --list | grep api", "List API-related config"),
            ("", ""),
            ("# Debugging", ""),
            ("hive-doctor --fix", "Auto-diagnose and fix issues"),
            ("hive-debug system", "Debug system component"),
            ("hive-metrics --format prometheus > metrics.txt", "Export metrics"),
        ]
        
        for cmd, desc in demo_commands:
            if cmd.startswith("#"):
                console.print(f"\n[bold]{cmd}[/bold]")
            elif cmd == "":
                continue
            else:
                console.print(f"  [cyan]{cmd}[/cyan]")
                if desc:
                    console.print(f"    {desc}")
        
        console.print("\n[dim]These commands are designed for automation, scripting, and power users.[/dim]")
    
    return unix_group


def add_compatibility_commands():
    """Add compatibility layer for existing lv commands."""
    
    # Map existing lv commands to new Unix equivalents
    command_mapping = {
        'start': 'hive-status',
        'develop': 'hive-create',
        'debug': 'hive-debug', 
        'status': 'hive-status',
        'dashboard': 'hive-get agents',
        'health': 'hive-doctor',
        'logs': 'hive-logs',
        'reset': 'hive-config --reset',
    }
    
    console.print("[blue]Command Compatibility Mapping:[/blue]")
    
    from rich.table import Table
    table = Table()
    table.add_column("Legacy Command", style="yellow")
    table.add_column("Unix Equivalent", style="cyan")
    
    for old, new in command_mapping.items():
        table.add_row(f"lv {old}", new)
    
    console.print(table)
    
    return command_mapping


def create_migration_guide():
    """Create migration guide for users moving from lv to Unix commands."""
    
    migration_guide = """
# Migration Guide: lv → Unix CLI Commands

## Philosophy Change

**Before (Monolithic):**
```bash
lv start --dashboard          # One command, many options
lv develop "My Project" --timeout=300
lv debug                      # General debugging
```

**After (Unix Philosophy):**
```bash
hive-status --watch          # Focused status monitoring
hive-create --count 2        # Focused agent creation  
hive-debug system            # Focused debugging
```

## Key Advantages

1. **Composability**: Commands can be chained and piped
   ```bash
   hive-get agents -o json | jq '.agents[].id' | xargs -I {} hive-scale {} 3
   ```

2. **Automation-Friendly**: Each command has consistent exit codes and output
   ```bash
   if hive-doctor --quiet; then
       hive-create --count 5
   fi
   ```

3. **Performance**: Faster startup, focused functionality
   ```bash
   time hive-status    # ~0.1s vs lv status ~0.5s
   ```

4. **Discoverability**: Standard Unix patterns
   ```bash
   hive-*           # Tab completion shows all commands
   man hive-status  # Standard documentation
   ```

## Migration Commands

| Legacy `lv` Command | Unix Equivalent | Notes |
|-------------------|-----------------|-------|
| `lv start` | `hive-status` | Status checking |
| `lv develop PROJECT` | `hive-create --type dev` | Agent creation |
| `lv debug` | `hive-debug system` | System debugging |
| `lv status` | `hive-status` | System status |
| `lv dashboard` | `hive-get agents` | Resource listing |
| `lv health` | `hive-doctor` | Health checking |
| `lv logs` | `hive-logs -f` | Log viewing |
| `lv reset` | `hive-config --reset` | Configuration reset |

## Compatibility Layer

Both systems can coexist:
- Keep using `lv` for interactive development
- Use `hive-*` commands for scripts and automation
- Gradually migrate to Unix commands for consistency

## Installation

```bash
# Install Unix commands
lv unix install

# Verify installation  
lv unix verify

# See demo
lv unix demo
```
"""
    
    return migration_guide


def show_unix_cli_benefits():
    """Show benefits of Unix CLI approach."""
    
    console.print(Panel(
        """[bold green]Unix Philosophy Benefits[/bold green]

🎯 [bold]Focused Commands[/bold]
   Each command does one thing exceptionally well
   
🔗 [bold]Composable[/bold] 
   Commands work together through pipes and scripts
   
⚡ [bold]Performance[/bold]
   Faster startup, lighter resource usage
   
🛠️ [bold]Automation-Friendly[/bold]
   Consistent interfaces, predictable exit codes
   
📈 [bold]Scalable[/bold]
   Easy to add new commands without breaking existing ones
   
🎨 [bold]Familiar[/bold]
   Following kubectl/docker patterns that developers know

[dim]Perfect for DevOps, CI/CD, and power users[/dim]
""",
        title="Why Unix CLI?",
        border_style="green"
    ))


if __name__ == '__main__':
    # Demo the integration
    unix_group = integrate_with_dx_cli()
    show_unix_cli_benefits()
    console.print("\n" + create_migration_guide())