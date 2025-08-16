#!/usr/bin/env python3
"""
Logger Migration Script for Epic 1 - Logging Service Consolidation

This script systematically replaces 340+ logger instances across the codebase
with centralized logging service calls.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

def find_files_with_logger_patterns(root_dir: str) -> List[Path]:
    """Find all Python files with logger patterns to replace."""
    files_to_migrate = []
    
    for file_path in Path(root_dir).rglob("*.py"):
        if file_path.name == "logging_service.py":
            continue  # Skip our logging service file
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for patterns we need to replace
            if (
                'structlog.get_logger' in content or
                'logging.getLogger' in content or
                'import structlog' in content
            ):
                files_to_migrate.append(file_path)
        except (UnicodeDecodeError, PermissionError):
            continue
            
    return files_to_migrate

def analyze_file_imports(file_path: Path) -> Dict[str, bool]:
    """Analyze what needs to be replaced in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {
        'has_structlog_import': 'import structlog' in content,
        'has_structlog_logger': 'structlog.get_logger' in content,
        'has_logging_logger': 'logging.getLogger' in content,
        'has_structlog_configure': 'structlog.configure' in content,
    }

def generate_replacement_plan(root_dir: str) -> Tuple[List[Path], Dict[str, int]]:
    """Generate a plan for replacing logger instances."""
    files = find_files_with_logger_patterns(root_dir)
    stats = {
        'total_files': len(files),
        'structlog_imports': 0,
        'structlog_loggers': 0,
        'logging_loggers': 0,
        'structlog_configs': 0,
    }
    
    priority_files = []
    regular_files = []
    
    for file_path in files:
        analysis = analyze_file_imports(file_path)
        
        # Update statistics
        if analysis['has_structlog_import']:
            stats['structlog_imports'] += 1
        if analysis['has_structlog_logger']:
            stats['structlog_loggers'] += 1
        if analysis['has_logging_logger']:
            stats['logging_loggers'] += 1
        if analysis['has_structlog_configure']:
            stats['structlog_configs'] += 1
            
        # Prioritize core files
        if any(part in str(file_path) for part in ['core/', 'main.py', 'orchestrator']):
            priority_files.append(file_path)
        else:
            regular_files.append(file_path)
    
    return priority_files + regular_files, stats

def migrate_file(file_path: Path, dry_run: bool = True) -> bool:
    """Migrate a single file to use centralized logging service."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        content = original_content
        changes_made = False
        
        # Determine appropriate import based on file location
        rel_path = str(file_path)
        
        if '/app/core/' in rel_path:
            logging_import = "from .logging_service import get_logger, get_component_logger"
        elif '/app/' in rel_path:
            # Calculate relative path to core
            depth = rel_path.count('/') - rel_path[:rel_path.find('/app/')].count('/') - 2
            if depth > 1:
                prefix = "../" * (depth - 1)
                logging_import = f"from {prefix}core.logging_service import get_logger, get_component_logger"
            else:
                logging_import = "from ..core.logging_service import get_logger, get_component_logger"
        else:
            logging_import = "from app.core.logging_service import get_logger, get_component_logger"
        
        # Replace structlog import with logging service import
        if 'import structlog\n' in content:
            content = content.replace('import structlog\n', logging_import + '\n')
            changes_made = True
        
        # Replace logger initialization patterns
        patterns = [
            (r'logger = structlog\.get_logger\(\)', 'logger = get_logger(__name__)'),
            (r'logger = structlog\.get_logger\(__name__\)', 'logger = get_logger(__name__)'),
            (r'logger = structlog\.get_logger\("([^"]+)"\)', r'logger = get_logger("\1")'),
            (r'logger = structlog\.get_logger\(([^)]+)\)', r'logger = get_logger(\1)'),
            (r'logger = logging\.getLogger\(\)', 'logger = get_logger(__name__)'),
            (r'logger = logging\.getLogger\(__name__\)', 'logger = get_logger(__name__)'),
            (r'logger = logging\.getLogger\("([^"]+)"\)', r'logger = get_logger("\1")'),
            (r'logger = logging\.getLogger\(([^)]+)\)', r'logger = get_logger(\1)'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                changes_made = True
        
        # Remove structlog.configure calls (should only be in our centralized service now)
        configure_pattern = r'structlog\.configure\([^}]+\}\s*\)'
        if re.search(configure_pattern, content, re.DOTALL):
            content = re.sub(configure_pattern, '# Logging configured via centralized logging service', content, flags=re.DOTALL)
            changes_made = True
        
        if changes_made and not dry_run:
            # Create backup
            backup_path = str(file_path) + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        return changes_made
        
    except Exception as e:
        print(f"Error migrating {file_path}: {e}")
        return False

if __name__ == "__main__":
    # Generate migration plan
    app_dir = "app"
    files, stats = generate_replacement_plan(app_dir)
    
    print("=== LOGGER MIGRATION ANALYSIS ===")
    print(f"Total files to migrate: {stats['total_files']}")
    print(f"Files with structlog imports: {stats['structlog_imports']}")
    print(f"Files with structlog loggers: {stats['structlog_loggers']}")
    print(f"Files with logging loggers: {stats['logging_loggers']}")
    print(f"Files with structlog configs: {stats['structlog_configs']}")
    print()
    
    print("=== MIGRATION PLAN ===")
    print("Priority files (core infrastructure):")
    for i, file_path in enumerate(files[:10]):  # Show first 10
        print(f"  {i+1}. {file_path}")
    
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    print()
    print("Ready to execute migration? This will:")
    print("1. Create .backup files for all modified files")
    print("2. Replace logger patterns with centralized service calls")
    print("3. Update import statements")
    print("4. Remove duplicate structlog.configure calls")
    
    # For now, just run in dry-run mode for analysis
    print("\n=== DRY RUN RESULTS ===")
    migrated_count = 0
    for file_path in files[:5]:  # Test first 5 files
        if migrate_file(file_path, dry_run=True):
            migrated_count += 1
            print(f"✅ Would migrate: {file_path}")
        else:
            print(f"⏭️  No changes needed: {file_path}")
    
    print(f"\nDry run complete. {migrated_count} files would be modified.")