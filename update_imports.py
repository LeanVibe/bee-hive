#!/usr/bin/env python3
"""
Update Imports Script for Epic 1 Phase 1.2 Consolidation

This script systematically updates all import statements across the codebase
to use the new unified managers instead of the scattered individual files.

Import Mapping Rules:
- agent_spawner -> agent_manager (AgentManager)
- messaging_service -> communication_manager (CommunicationManager)
- context_compression -> context_manager_unified (ContextManagerUnified)
- workflow_engine -> workflow_manager (WorkflowManager)
- performance_optimizer -> resource_manager (ResourceManager)  
- security_audit -> security_manager (SecurityManager)
- database/redis -> storage_manager (StorageManager)

Usage:
    python update_imports.py [--dry-run] [--file PATH]
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

# Import mapping rules: old_module -> (new_module, new_class, additional_imports)
IMPORT_MAPPINGS = {
    'agent_spawner': ('agent_manager', 'AgentManager', ['AgentSpec', 'AgentRole']),
    'agent_registry': ('agent_manager', 'AgentManager', ['AgentRegistration']),
    'agent_lifecycle_manager': ('agent_manager', 'AgentManager', ['AgentLifecycle']),
    'messaging_service': ('communication_manager', 'CommunicationManager', ['Message', 'MessageType', 'MessagePriority']),
    'redis_pubsub_manager': ('communication_manager', 'CommunicationManager', ['MessageBroker']),
    'context_compression': ('context_manager_unified', 'ContextManagerUnified', ['ContextCompressor']),
    'context_lifecycle_manager': ('context_manager_unified', 'ContextManagerUnified', ['ContextLifecycle']),
    'workflow_engine': ('workflow_manager', 'WorkflowManager', ['WorkflowEngine', 'WorkflowDefinition']),
    'task_scheduler': ('workflow_manager', 'WorkflowManager', ['TaskScheduler']),
    'performance_optimizer': ('resource_manager', 'ResourceManager', ['PerformanceOptimizer']),
    'resource_optimizer': ('resource_manager', 'ResourceManager', ['ResourceAllocation']),
    'security_audit': ('security_manager', 'SecurityManager', ['SecurityAuditor']),
    'auth': ('security_manager', 'SecurityManager', ['AuthManager']),
    'database': ('storage_manager', 'StorageManager', ['DatabaseManager']),
    'redis': ('storage_manager', 'StorageManager', ['CacheManager']),
}

# Function name mappings for commonly used functions
FUNCTION_MAPPINGS = {
    'get_messaging_service': 'CommunicationManager',
    'get_agent_manager': 'AgentManager', 
    'get_active_agents_status': 'list_agents',
    'spawn_agent': 'spawn_agent',
    'spawn_development_team': 'spawn_agent_team',
    'compress_context': 'compress_context',
    'decompress_context': 'decompress_context',
    'execute_workflow': 'execute_workflow',
    'schedule_task': 'schedule_task',
    'optimize_performance': 'optimize_performance',
    'log_security_event': 'log_security_event',
    'authenticate': 'authenticate',
    'get_session': 'get_database_session',
    'get_redis': 'get_redis_client',
}

# Patterns to match different import styles
IMPORT_PATTERNS = [
    r'from\s+\.{1,2}(\w+)\s+import\s+([^#\n]+)',  # from .module import items
    r'import\s+\.{1,2}(\w+)(?:\s+as\s+(\w+))?',    # import .module [as alias]
]

def parse_import_line(line: str) -> Tuple[str, str, List[str]]:
    """Parse an import line and extract module and imported items."""
    line = line.strip()
    
    # Handle "from .module import items"
    from_match = re.match(r'from\s+\.{1,2}(\w+)\s+import\s+([^#\n]+)', line)
    if from_match:
        module = from_match.group(1)
        imports_str = from_match.group(2)
        items = [item.strip() for item in imports_str.split(',')]
        return 'from', module, items
    
    # Handle "import .module [as alias]"
    import_match = re.match(r'import\s+\.{1,2}(\w+)(?:\s+as\s+(\w+))?', line)
    if import_match:
        module = import_match.group(1)
        alias = import_match.group(2)
        return 'import', module, [alias] if alias else []
    
    return '', '', []

def update_import_line(line: str) -> str:
    """Update a single import line to use unified managers."""
    original_line = line
    import_type, module, items = parse_import_line(line)
    
    if not module or module not in IMPORT_MAPPINGS:
        return line
    
    new_module, main_class, additional_imports = IMPORT_MAPPINGS[module]
    
    if import_type == 'from':
        # Update "from .old_module import items" to "from .new_module import NewClass"
        updated_items = []
        
        for item in items:
            item = item.strip()
            if item in FUNCTION_MAPPINGS:
                # Map old function names to new class
                updated_items.append(main_class)
            elif item.startswith('(') or item.endswith(')'):
                # Handle multiline imports - keep the parentheses
                updated_items.append(item)
            else:
                # Keep the item or map to new equivalent
                updated_items.append(item)
        
        # Add the main class if not already present
        if main_class not in updated_items:
            updated_items.insert(0, main_class)
        
        # Add additional imports that are commonly needed
        for add_import in additional_imports:
            if add_import not in updated_items:
                updated_items.append(add_import)
        
        new_line = f"from .{new_module} import {', '.join(updated_items)}"
        
    elif import_type == 'import':
        # Update "import .old_module" to "from .new_module import NewClass"
        new_line = f"from .{new_module} import {main_class}"
    
    else:
        return line
    
    return new_line

def update_function_calls(content: str) -> str:
    """Update function calls to use new unified manager methods."""
    for old_func, new_ref in FUNCTION_MAPPINGS.items():
        if old_func in content:
            if new_ref.endswith('Manager'):
                # Replace function calls with manager instantiation and method calls
                # This is a simple replacement - complex cases may need manual review
                content = content.replace(f"{old_func}()", f"{new_ref}()")
            else:
                # Direct method name replacement
                content = content.replace(old_func, new_ref)
    
    return content

def update_file_imports(file_path: Path, dry_run: bool = False) -> bool:
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated_lines = []
        changes_made = False
        
        for line in lines:
            original_line = line
            
            # Check if this is an import line that needs updating
            stripped = line.strip()
            if stripped.startswith(('from .', 'import .')) and any(module in stripped for module in IMPORT_MAPPINGS.keys()):
                updated_line = update_import_line(line)
                if updated_line != original_line:
                    changes_made = True
                    print(f"  {file_path.name}: {original_line.strip()} -> {updated_line.strip()}")
                updated_lines.append(updated_line + '\n' if not updated_line.endswith('\n') else updated_line)
            else:
                updated_lines.append(original_line)
        
        if changes_made and not dry_run:
            # Backup original file
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
        
        return changes_made
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory and subdirectories."""
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files

def main():
    parser = argparse.ArgumentParser(description="Update imports to use unified managers")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be changed without making changes")
    parser.add_argument('--file', type=str, help="Update specific file instead of entire codebase")
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help="Exclude files matching patterns")
    
    args = parser.parse_args()
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 1
        files_to_update = [file_path]
    else:
        # Find all Python files in the project
        project_root = Path(__file__).parent
        files_to_update = find_python_files(project_root)
    
    # Filter out excluded files
    if args.exclude:
        excluded_patterns = args.exclude
        files_to_update = [
            f for f in files_to_update 
            if not any(pattern in str(f) for pattern in excluded_patterns)
        ]
    
    print(f"{'DRY RUN: ' if args.dry_run else ''}Updating imports in {len(files_to_update)} files...")
    print("=" * 60)
    
    total_updated = 0
    
    for file_path in files_to_update:
        if update_file_imports(file_path, args.dry_run):
            total_updated += 1
    
    print("=" * 60)
    print(f"{'Would update' if args.dry_run else 'Updated'} imports in {total_updated} files")
    
    if args.dry_run:
        print("\nRun without --dry-run to apply changes")
    else:
        print("\nBackup files created with .backup extension")
        print("Original files can be restored if needed")
    
    return 0

if __name__ == '__main__':
    exit(main())