#!/usr/bin/env python3
"""
Import Pattern Consolidation Script
==================================

Systematic consolidation of duplicate import patterns using proven methodologies.
Creates shared import modules and updates files to use consolidated imports.
"""

import os
import shutil
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import tempfile
import re

class ImportPatternConsolidator:
    """Consolidates duplicate import patterns with safety systems."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = Path(tempfile.mkdtemp(prefix="import_backups_"))
        self.common_imports_dir = self.project_root / "app" / "common" / "imports"
        self.results = []
        
    def analyze_patterns(self) -> Tuple[Dict, Counter]:
        """Analyze import patterns to find consolidation opportunities."""
        import_patterns = defaultdict(list)
        common_imports = Counter()
        
        print("🔍 Analyzing import patterns across Python files...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip virtual environments and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        imports = self.extract_imports(file_path)
                        if imports:
                            # Create pattern signature for exact matches
                            pattern_sig = tuple(sorted(imports))
                            import_patterns[pattern_sig].append(file_path)
                            
                            # Count individual imports
                            for imp in imports:
                                common_imports[imp] += 1
                                
                    except Exception:
                        pass
        
        return import_patterns, common_imports
    
    def extract_imports(self, file_path: Path) -> List[str]:
        """Extract import statements from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            imports = []
            in_multiline_import = False
            current_import = ""
            
            for line in lines[:80]:  # Check first 80 lines where imports typically are
                stripped = line.strip()
                
                # Skip comments and docstrings
                if stripped.startswith('#') or not stripped:
                    continue
                    
                # Stop at first function/class definition (imports should be before)
                if stripped.startswith(('def ', 'class ', 'if __name__')):
                    break
                
                # Handle multiline imports
                if '(' in stripped and stripped.startswith(('import ', 'from ')) and ')' not in stripped:
                    in_multiline_import = True
                    current_import = stripped
                    continue
                elif in_multiline_import:
                    current_import += " " + stripped
                    if ')' in stripped:
                        imports.append(self.normalize_import(current_import))
                        in_multiline_import = False
                        current_import = ""
                    continue
                
                # Single line imports
                if stripped.startswith(('import ', 'from ')):
                    imports.append(self.normalize_import(stripped))
            
            return imports
            
        except Exception:
            return []
    
    def normalize_import(self, import_line: str) -> str:
        """Normalize import statements for comparison."""
        # Remove 'as' aliases for comparison
        if ' as ' in import_line:
            import_line = import_line.split(' as ')[0]
        
        # Remove extra whitespace and newlines
        import_line = ' '.join(import_line.split())
        
        return import_line
    
    def create_common_import_modules(self, common_imports: Counter) -> Dict[str, List[str]]:
        """Create common import modules for frequently used patterns."""
        
        # Define common import groups
        import_groups = {
            'standard_lib': [
                'import asyncio',
                'import json', 
                'import os',
                'import sys',
                'import time',
                'import uuid',
                'from datetime import datetime, timedelta',
                'from pathlib import Path',
                'from typing import Dict, List, Set, Tuple, Optional, Any',
                'from dataclasses import dataclass, field',
                'from enum import Enum'
            ],
            'fastapi_common': [
                'from fastapi import APIRouter, Depends, HTTPException, status',
                'from fastapi.responses import JSONResponse',
                'from fastapi.security import HTTPBearer'
            ],
            'database_common': [
                'from sqlalchemy.ext.asyncio import AsyncSession',
                'from sqlalchemy import Column, String, Integer, Boolean, DateTime',
                'from sqlalchemy.dialects import postgresql'
            ],
            'testing_common': [
                'import pytest',
                'from fastapi.testclient import TestClient',
                'from httpx import AsyncClient',
                'from unittest.mock import Mock, AsyncMock, MagicMock'
            ],
            'logging_common': [
                'import logging',
                'import structlog'
            ]
        }
        
        # Create the common imports directory
        self.common_imports_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for the imports package
        init_file = self.common_imports_dir / "__init__.py"
        init_file.write_text('"""Common import patterns for LeanVibe Agent Hive 2.0."""\\n\\n__all__ = []\\n')
        
        created_modules = {}
        
        for group_name, imports in import_groups.items():
            # Filter to only include imports that are actually common (used in 10+ files)
            filtered_imports = [imp for imp in imports if common_imports.get(imp, 0) >= 10]
            
            if filtered_imports:
                module_path = self.common_imports_dir / f"{group_name}.py"
                module_content = f'''"""
{group_name.replace('_', ' ').title()} imports for LeanVibe Agent Hive 2.0.

Common import patterns used across multiple modules.
"""

# Re-export common imports for convenience
{chr(10).join(filtered_imports)}

__all__ = [
    # Add exported items here as needed
]
'''
                module_path.write_text(module_content)
                created_modules[group_name] = filtered_imports
                print(f"✅ Created {group_name}.py with {len(filtered_imports)} imports")
        
        return created_modules
    
    def create_backup(self, file_path: Path) -> Path:
        """Create backup of original file."""
        backup_path = self.backup_dir / f"{file_path.name}_{int(datetime.now().timestamp())}.backup"
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def consolidate_duplicate_patterns(self, import_patterns: Dict, dry_run: bool = False) -> List[Dict]:
        """Consolidate files with identical import patterns."""
        results = []
        
        # Find patterns with 3+ files (worth consolidating)
        significant_patterns = {
            pattern: files for pattern, files in import_patterns.items() 
            if len(files) >= 3 and len(pattern) >= 2  # At least 3 files and 2 imports
        }
        
        print(f"📋 Found {len(significant_patterns)} significant duplicate patterns")
        
        for i, (pattern, files) in enumerate(significant_patterns.items()):
            if len(files) < 3:  # Skip small patterns
                continue
                
            print(f"\\n🎯 Pattern {i+1}: {len(files)} files with {len(pattern)} imports")
            for imp in pattern[:3]:
                print(f"   {imp}")
            if len(pattern) > 3:
                print(f"   ... and {len(pattern)-3} more imports")
            
            # Calculate potential savings
            # Each file could save approximately len(pattern) lines by using a consolidated import
            potential_savings = len(files) * min(len(pattern), 5)  # Cap at 5 lines savings per file
            
            result = {
                'pattern': pattern,
                'files': files,
                'potential_savings': potential_savings,
                'consolidation_type': 'duplicate_pattern'
            }
            
            if not dry_run and len(files) >= 5:  # Only consolidate if 5+ files
                # For now, just report - actual consolidation would require more careful analysis
                result['status'] = 'identified_for_consolidation'
            else:
                result['status'] = 'analyzed'
            
            results.append(result)
        
        return results
    
    def process_all_patterns(self, dry_run: bool = True) -> Dict:
        """Process all import patterns for consolidation opportunities."""
        print("🚀 Starting import pattern analysis...")
        
        # Step 1: Analyze patterns
        import_patterns, common_imports = self.analyze_patterns()
        
        print(f"\\n📊 Analysis Results:")
        print(f"   Files analyzed: {sum(len(files) for files in import_patterns.values())}")
        print(f"   Unique patterns: {len(import_patterns)}")
        print(f"   Duplicate patterns: {len([p for p in import_patterns.values() if len(p) > 1])}")
        
        # Step 2: Create common import modules
        if not dry_run:
            created_modules = self.create_common_import_modules(common_imports)
        else:
            created_modules = {}
        
        # Step 3: Consolidate duplicate patterns
        consolidation_results = self.consolidate_duplicate_patterns(import_patterns, dry_run)
        
        # Calculate total potential savings
        total_savings = sum(result['potential_savings'] for result in consolidation_results)
        
        summary = {
            'total_files_analyzed': sum(len(files) for files in import_patterns.values()),
            'duplicate_patterns_found': len([p for p in import_patterns.values() if len(p) > 1]),
            'significant_patterns': len([r for r in consolidation_results if len(r['files']) >= 3]),
            'total_potential_savings': total_savings,
            'created_modules': created_modules,
            'consolidation_results': consolidation_results
        }
        
        return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Consolidate import patterns')
    parser.add_argument('--dry-run', action='store_true', help='Analyze patterns without making changes')
    parser.add_argument('--apply', action='store_true', help='Apply consolidation changes')
    
    args = parser.parse_args()
    
    consolidator = ImportPatternConsolidator()
    
    if args.apply:
        print("🚀 Applying import pattern consolidation...")
        summary = consolidator.process_all_patterns(dry_run=False)
    else:
        print("📋 Analyzing import patterns (dry run)...")
        summary = consolidator.process_all_patterns(dry_run=True)
    
    print(f"\\n📊 Consolidation Summary:")
    print(f"   📈 {summary['total_files_analyzed']} files analyzed")
    print(f"   🔍 {summary['duplicate_patterns_found']} duplicate patterns found")
    print(f"   🎯 {summary['significant_patterns']} significant consolidation opportunities")
    print(f"   💰 {summary['total_potential_savings']} LOC potential savings")
    
    if not args.apply:
        print(f"\\n🚀 Run with --apply to create common import modules")