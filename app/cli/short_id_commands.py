"""
CLI Integration for Human-Friendly Short IDs in LeanVibe Agent Hive 2.0

This module provides CLI commands and utilities for working with short IDs
in a human-friendly way. Supports partial matching, intelligent completion,
and seamless integration with existing Hive commands.

Examples:
    hive task show TSK-A7B2
    hive task list --filter TSK-A7
    hive agent scale AGT-M4K9 5
    hive project status PRJ-X2Y8
"""

import sys
import uuid
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from ..core.short_id_generator import (
    ShortIdGenerator, EntityType, generate_short_id, resolve_short_id,
    validate_short_id_format, get_generator
)

console = Console()


class IdResolutionStrategy(Enum):
    """Strategy for resolving ambiguous ID inputs."""
    EXACT_MATCH = "exact"          # Only exact matches
    PARTIAL_ALLOW = "partial"      # Allow partial matches if unique
    PARTIAL_CONFIRM = "confirm"    # Confirm partial matches with user
    PARTIAL_LIST = "list"          # List all partial matches


@dataclass
class IdResolutionResult:
    """Result of ID resolution attempt."""
    success: bool
    short_id: Optional[str] = None
    uuid: Optional[uuid.UUID] = None
    matches: List[Tuple[str, uuid.UUID]] = None
    error: Optional[str] = None
    
    @property
    def is_ambiguous(self) -> bool:
        """Check if result has multiple matches."""
        return self.matches and len(self.matches) > 1
    
    @property
    def is_unique(self) -> bool:
        """Check if result has exactly one match."""
        return self.matches and len(self.matches) == 1


class ShortIdResolver:
    """
    Smart resolver for handling various ID input formats in CLI commands.
    
    Handles:
    - Full short IDs: TSK-A7B2
    - Partial short IDs: TSK-A7, A7B2
    - UUIDs: 123e4567-e89b-12d3-a456-426614174000
    - Intelligent matching and disambiguation
    """
    
    def __init__(self, generator: ShortIdGenerator = None):
        """
        Initialize the resolver.
        
        Args:
            generator: Optional generator instance
        """
        self.generator = generator or get_generator()
        self.console = Console()
    
    def resolve(self, id_input: str, 
                entity_type: Optional[EntityType] = None,
                strategy: IdResolutionStrategy = IdResolutionStrategy.PARTIAL_CONFIRM) -> IdResolutionResult:
        """
        Resolve an ID input to canonical short ID and UUID.
        
        Args:
            id_input: User input (partial ID, full ID, or UUID)
            entity_type: Optional entity type filter
            strategy: Resolution strategy for ambiguous matches
            
        Returns:
            IdResolutionResult with resolution outcome
        """
        if not id_input or not id_input.strip():
            return IdResolutionResult(success=False, error="Empty ID input")
        
        id_input = id_input.strip().upper()
        
        # Try exact resolution first
        try:
            if self._is_uuid_format(id_input):
                # Handle UUID input
                uuid_obj = uuid.UUID(id_input.lower())
                short_id, resolved_uuid = resolve_short_id(uuid_obj)
                return IdResolutionResult(
                    success=True,
                    short_id=short_id,
                    uuid=resolved_uuid,
                    matches=[(short_id, resolved_uuid)]
                )
            
            elif validate_short_id_format(id_input):
                # Handle full short ID
                short_id, uuid_obj = resolve_short_id(id_input)
                return IdResolutionResult(
                    success=True,
                    short_id=short_id,
                    uuid=uuid_obj,
                    matches=[(short_id, uuid_obj)]
                )
                
        except ValueError:
            # Not an exact match, try partial matching
            pass
        
        # Try partial matching
        matches = self._find_partial_matches(id_input, entity_type)
        
        if not matches:
            return IdResolutionResult(
                success=False,
                error=f"No matches found for '{id_input}'"
            )
        
        # Handle matches based on strategy
        return self._handle_matches(matches, id_input, strategy)
    
    def _is_uuid_format(self, value: str) -> bool:
        """Check if string looks like a UUID."""
        try:
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False
    
    def _find_partial_matches(self, partial_id: str, 
                             entity_type: Optional[EntityType] = None) -> List[Tuple[str, uuid.UUID]]:
        """
        Find partial matches for an ID fragment.
        
        Args:
            partial_id: Partial ID to match
            entity_type: Optional entity type filter
            
        Returns:
            List of matching (short_id, uuid) tuples
        """
        matches = []
        
        # Direct partial match
        direct_matches = self.generator.partial_match(partial_id, entity_type)
        matches.extend(direct_matches)
        
        # If no direct matches and no separator, try with entity prefixes
        if not matches and "-" not in partial_id:
            if entity_type:
                # Try with specific entity type
                prefixed = f"{entity_type.value}-{partial_id}"
                prefixed_matches = self.generator.partial_match(prefixed, entity_type)
                matches.extend(prefixed_matches)
            else:
                # Try with all entity types
                for et in EntityType:
                    prefixed = f"{et.value}-{partial_id}"
                    prefixed_matches = self.generator.partial_match(prefixed, et)
                    matches.extend(prefixed_matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            if match[0] not in seen:
                seen.add(match[0])
                unique_matches.append(match)
        
        return unique_matches
    
    def _handle_matches(self, matches: List[Tuple[str, uuid.UUID]], 
                       original_input: str,
                       strategy: IdResolutionStrategy) -> IdResolutionResult:
        """
        Handle multiple matches based on resolution strategy.
        
        Args:
            matches: List of matching (short_id, uuid) tuples
            original_input: Original user input
            strategy: Resolution strategy
            
        Returns:
            IdResolutionResult with appropriate handling
        """
        if len(matches) == 1:
            # Single match - always return it
            short_id, uuid_obj = matches[0]
            return IdResolutionResult(
                success=True,
                short_id=short_id,
                uuid=uuid_obj,
                matches=matches
            )
        
        # Multiple matches - handle by strategy
        if strategy == IdResolutionStrategy.EXACT_MATCH:
            return IdResolutionResult(
                success=False,
                error=f"Ambiguous ID '{original_input}' - use exact ID",
                matches=matches
            )
        
        elif strategy == IdResolutionStrategy.PARTIAL_LIST:
            return IdResolutionResult(
                success=False,
                error=f"Multiple matches for '{original_input}'",
                matches=matches
            )
        
        elif strategy == IdResolutionStrategy.PARTIAL_CONFIRM:
            # Interactive confirmation
            selected = self._interactive_select(matches, original_input)
            if selected:
                return IdResolutionResult(
                    success=True,
                    short_id=selected[0],
                    uuid=selected[1],
                    matches=[selected]
                )
            else:
                return IdResolutionResult(
                    success=False,
                    error="No selection made",
                    matches=matches
                )
        
        elif strategy == IdResolutionStrategy.PARTIAL_ALLOW:
            # Just take the first match
            short_id, uuid_obj = matches[0]
            return IdResolutionResult(
                success=True,
                short_id=short_id,
                uuid=uuid_obj,
                matches=matches
            )
        
        return IdResolutionResult(
            success=False,
            error="Unknown resolution strategy",
            matches=matches
        )
    
    def _interactive_select(self, matches: List[Tuple[str, uuid.UUID]], 
                           original_input: str) -> Optional[Tuple[str, uuid.UUID]]:
        """
        Interactive selection for ambiguous matches.
        
        Args:
            matches: List of matching options
            original_input: Original user input
            
        Returns:
            Selected (short_id, uuid) tuple or None if cancelled
        """
        if not matches:
            return None
        
        self.console.print(f"\n[yellow]Multiple matches found for '{original_input}':[/yellow]")
        
        # Create selection table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Option", style="green", width=8)
        table.add_column("Short ID", style="cyan", width=12)
        table.add_column("Entity Type", style="magenta", width=12)
        table.add_column("UUID", style="dim", width=36)
        
        for i, (short_id, uuid_obj) in enumerate(matches, 1):
            entity_type = self.generator.extract_entity_type(short_id)
            entity_name = entity_type.name if entity_type else "Unknown"
            table.add_row(str(i), short_id, entity_name, str(uuid_obj))
        
        self.console.print(table)
        
        # Get user selection
        while True:
            try:
                choice = click.prompt(
                    "\nSelect option (number) or 'q' to quit",
                    type=str,
                    default="q"
                ).strip().lower()
                
                if choice == 'q':
                    return None
                
                option_num = int(choice)
                if 1 <= option_num <= len(matches):
                    return matches[option_num - 1]
                else:
                    self.console.print("[red]Invalid option. Please try again.[/red]")
                    
            except (ValueError, KeyboardInterrupt):
                return None


# CLI Command Decorators and Utilities

def resolve_id_argument(entity_type: Optional[EntityType] = None,
                       strategy: IdResolutionStrategy = IdResolutionStrategy.PARTIAL_CONFIRM):
    """
    Decorator to automatically resolve ID arguments in CLI commands.
    
    Args:
        entity_type: Optional entity type constraint
        strategy: Resolution strategy for ambiguous IDs
        
    Usage:
        @click.command()
        @click.argument('task_id')
        @resolve_id_argument(EntityType.TASK)
        def show_task(task_id, resolved_short_id, resolved_uuid):
            # resolved_short_id and resolved_uuid are automatically added
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Find ID arguments in kwargs
            id_args = {}
            for key, value in kwargs.items():
                if key.endswith('_id') and isinstance(value, str):
                    # Resolve this ID
                    resolver = ShortIdResolver()
                    result = resolver.resolve(value, entity_type, strategy)
                    
                    if not result.success:
                        console.print(f"[red]Error resolving ID '{value}': {result.error}[/red]")
                        
                        if result.matches:
                            # Show available matches
                            table = Table(title="Available matches")
                            table.add_column("Short ID", style="cyan")
                            table.add_column("Entity Type", style="magenta")
                            
                            for short_id, _ in result.matches[:10]:  # Limit to 10
                                et = get_generator().extract_entity_type(short_id)
                                entity_name = et.name if et else "Unknown"
                                table.add_row(short_id, entity_name)
                            
                            console.print(table)
                        
                        sys.exit(1)
                    
                    # Add resolved values to kwargs
                    kwargs[f'resolved_{key}_short'] = result.short_id
                    kwargs[f'resolved_{key}_uuid'] = result.uuid
                    
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def format_id_table(items: List[Dict[str, Any]], id_field: str = 'id', 
                   short_id_field: str = 'short_id') -> Table:
    """
    Format a table with both UUIDs and short IDs for display.
    
    Args:
        items: List of dictionaries with entity data
        id_field: Field name containing UUID
        short_id_field: Field name containing short ID
        
    Returns:
        Rich Table object ready for display
    """
    table = Table(show_header=True, header_style="bold blue")
    
    if items:
        # Add columns based on first item
        first_item = items[0]
        
        # Always show short ID first if available
        if short_id_field in first_item:
            table.add_column("ID", style="cyan", width=12)
        
        # Add other columns
        for key in first_item.keys():
            if key not in [id_field, short_id_field]:
                table.add_column(key.replace('_', ' ').title(), width=20)
        
        # Add UUID last (dimmed)
        if id_field in first_item:
            table.add_column("UUID", style="dim", width=36)
        
        # Add rows
        for item in items:
            row_data = []
            
            # Short ID first
            if short_id_field in item:
                row_data.append(str(item[short_id_field] or "N/A"))
            
            # Other fields
            for key, value in item.items():
                if key not in [id_field, short_id_field]:
                    row_data.append(str(value) if value is not None else "N/A")
            
            # UUID last
            if id_field in item:
                row_data.append(str(item[id_field]))
            
            table.add_row(*row_data)
    
    return table


# CLI Commands for Short ID Management

@click.group()
def short_id():
    """Manage short IDs in the hive system."""
    pass


@short_id.command()
@click.argument('entity_type', type=click.Choice([et.name.lower() for et in EntityType]))
@click.option('--count', '-c', default=1, help='Number of IDs to generate')
@click.option('--format', '-f', type=click.Choice(['table', 'csv', 'json']), 
              default='table', help='Output format')
def generate(entity_type: str, count: int, format: str):
    """Generate new short IDs for testing or pre-allocation."""
    
    et = EntityType[entity_type.upper()]
    generator = get_generator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Generating {count} {entity_type} IDs...", total=count)
        
        results = []
        for i in range(count):
            short_id, uuid_obj = generate_short_id(et)
            results.append({
                'short_id': short_id,
                'uuid': str(uuid_obj),
                'entity_type': entity_type
            })
            progress.advance(task)
    
    # Output results
    if format == 'table':
        table = format_id_table(results, 'uuid', 'short_id')
        table.title = f"Generated {count} {entity_type} IDs"
        console.print(table)
        
    elif format == 'csv':
        import csv
        import sys
        writer = csv.DictWriter(sys.stdout, fieldnames=['short_id', 'uuid', 'entity_type'])
        writer.writeheader()
        writer.writerows(results)
        
    elif format == 'json':
        import json
        print(json.dumps(results, indent=2))


@short_id.command()
@click.argument('id_input')
@click.option('--entity-type', '-t', type=click.Choice([et.name.lower() for et in EntityType]),
              help='Filter by entity type')
def resolve(id_input: str, entity_type: Optional[str]):
    """Resolve a partial or full ID to canonical form."""
    
    et = EntityType[entity_type.upper()] if entity_type else None
    resolver = ShortIdResolver()
    
    result = resolver.resolve(id_input, et, IdResolutionStrategy.PARTIAL_LIST)
    
    if result.success:
        console.print(f"[green]✓[/green] Resolved '{id_input}' to [cyan]{result.short_id}[/cyan]")
        console.print(f"  UUID: {result.uuid}")
        
        entity_type = get_generator().extract_entity_type(result.short_id)
        if entity_type:
            console.print(f"  Type: {entity_type.name}")
            
    else:
        console.print(f"[red]✗[/red] {result.error}")
        
        if result.matches:
            table = Table(title=f"Partial matches for '{id_input}'")
            table.add_column("Short ID", style="cyan")
            table.add_column("Entity Type", style="magenta")
            table.add_column("UUID", style="dim")
            
            for short_id, uuid_obj in result.matches:
                et = get_generator().extract_entity_type(short_id)
                entity_name = et.name if et else "Unknown"
                table.add_row(short_id, entity_name, str(uuid_obj))
            
            console.print(table)


@short_id.command()
@click.option('--entity-type', '-t', type=click.Choice([et.name.lower() for et in EntityType]),
              help='Filter by entity type')
@click.option('--limit', '-l', default=50, help='Limit number of results')
def list(entity_type: Optional[str], limit: int):
    """List all short IDs in the system."""
    
    generator = get_generator()
    stats = generator.get_stats()
    
    console.print(Panel(
        f"[bold]Short ID Statistics[/bold]\n\n"
        f"Generated: {stats.generated_count}\n"
        f"Collisions: {stats.collision_count}\n"
        f"Cache Hits: {stats.cache_hits}\n"
        f"Cache Misses: {stats.cache_misses}\n"
        f"Avg Generation Time: {stats.average_generation_time_ms:.2f}ms",
        title="System Stats",
        border_style="blue"
    ))
    
    # This would query the database in a real implementation
    console.print(f"\n[dim]Database listing not yet implemented[/dim]")


@short_id.command()
@click.argument('short_id')
def validate(short_id: str):
    """Validate short ID format."""
    
    is_valid = validate_short_id_format(short_id)
    
    if is_valid:
        console.print(f"[green]✓[/green] '{short_id}' has valid format")
        
        entity_type = get_generator().extract_entity_type(short_id)
        if entity_type:
            console.print(f"  Entity Type: {entity_type.name} ({entity_type.value})")
        
        # Parse components
        parts = short_id.split('-')
        if len(parts) == 2:
            console.print(f"  Prefix: {parts[0]}")
            console.print(f"  Code: {parts[1]}")
    else:
        console.print(f"[red]✗[/red] '{short_id}' has invalid format")
        console.print("  Expected format: PREFIX-CODE (e.g., TSK-A7B2)")


if __name__ == '__main__':
    short_id()