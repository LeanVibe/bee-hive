#!/usr/bin/env python3
"""
Init File Standardization Script
===============================

Systematic standardization of __init__.py files using proven Track 1 methodology.
Applies consistent templates while preserving unique functionality.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional
import tempfile

class InitFileStandardizer:
    """Standardizes __init__.py files with safety systems."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = Path(tempfile.mkdtemp(prefix="init_backups_"))
        self.results = []
        
    def find_init_files(self) -> List[Path]:
        """Find all __init__.py files excluding virtual environments."""
        init_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip virtual environments and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__', 'node_modules']]
            
            for file in files:
                if file == '__init__.py':
                    init_files.append(Path(root) / file)
        
        return sorted(init_files)
    
    def analyze_init_file(self, file_path: Path) -> Dict:
        """Analyze an init file to determine its type and content."""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            analysis = {
                'file_path': file_path,
                'content': content,
                'lines': lines,
                'has_version': any('__version__' in line for line in lines),
                'has_author': any('__author__' in line for line in lines),
                'has_imports': any(line.startswith(('from ', 'import ')) for line in lines),
                'has_all': any('__all__' in line for line in lines),
                'has_router': any('router' in line.lower() for line in lines),
                'line_count': len(lines),
                'type': 'minimal'  # Default
            }
            
            # Classify init file type
            if analysis['has_version'] and analysis['has_author']:
                analysis['type'] = 'metadata'
            elif analysis['has_router'] or analysis['has_imports']:
                analysis['type'] = 'complex'
            elif analysis['line_count'] <= 2:
                analysis['type'] = 'minimal'
            
            return analysis
            
        except Exception as e:
            return {'file_path': file_path, 'error': str(e)}
    
    def create_backup(self, file_path: Path) -> Path:
        """Create backup of original file."""
        backup_path = self.backup_dir / f"{file_path.name}_{int(datetime.now().timestamp())}.backup"
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def standardize_minimal(self, analysis: Dict) -> str:
        """Create standardized minimal init file."""
        module_name = analysis['file_path'].parent.name.replace('_', ' ').title()
        return f'''"""
{module_name} module for LeanVibe Agent Hive 2.0.

This module provides {module_name.lower()} functionality.
"""

__all__ = []
'''
    
    def standardize_metadata(self, analysis: Dict) -> str:
        """Create standardized metadata init file."""
        module_name = analysis['file_path'].parent.name.replace('_', ' ').title()
        
        # Preserve existing version if present
        version = "2.0.0"  # Default
        for line in analysis['lines']:
            if '__version__' in line:
                if '"' in line:
                    version = line.split('"')[1]
                elif "'" in line:
                    version = line.split("'")[1]
        
        return f'''"""
{module_name} module for LeanVibe Agent Hive 2.0.

This module provides {module_name.lower()} functionality.
"""

__version__ = "{version}"
__author__ = "LeanVibe Agent Hive Team"
__email__ = "dev@leanvibe.com"

__all__ = []
'''
    
    def process_file(self, file_path: Path, dry_run: bool = False) -> Dict:
        """Process a single init file."""
        try:
            analysis = self.analyze_init_file(file_path)
            
            if 'error' in analysis:
                return {'file_path': file_path, 'status': 'error', 'message': analysis['error']}
            
            # Skip complex files that need manual review
            if analysis['type'] == 'complex':
                return {'file_path': file_path, 'status': 'skipped', 'message': 'Complex file - manual review required'}
            
            # Generate standardized content
            if analysis['type'] == 'metadata' or analysis['has_version']:
                new_content = self.standardize_metadata(analysis)
            else:
                new_content = self.standardize_minimal(analysis)
            
            # Calculate savings
            original_lines = len(analysis['lines'])
            new_lines = len([line for line in new_content.split('\n') if line.strip()])
            loc_savings = max(0, original_lines - new_lines)
            
            if dry_run:
                return {
                    'file_path': file_path,
                    'status': 'would_process',
                    'type': analysis['type'],
                    'original_lines': original_lines,
                    'new_lines': new_lines,
                    'loc_savings': loc_savings
                }
            
            # Create backup and apply changes
            backup_path = self.create_backup(file_path)
            file_path.write_text(new_content, encoding='utf-8')
            
            return {
                'file_path': file_path,
                'status': 'success',
                'type': analysis['type'],
                'original_lines': original_lines,
                'new_lines': new_lines,
                'loc_savings': loc_savings,
                'backup_path': backup_path
            }
            
        except Exception as e:
            return {'file_path': file_path, 'status': 'error', 'message': str(e)}
    
    def process_all_files(self, dry_run: bool = False) -> Dict:
        """Process all init files with comprehensive reporting."""
        init_files = self.find_init_files()
        print(f"🔍 Found {len(init_files)} __init__.py files to analyze")
        
        results = []
        total_loc_savings = 0
        
        for file_path in init_files:
            result = self.process_file(file_path, dry_run=dry_run)
            results.append(result)
            
            if result['status'] in ['success', 'would_process']:
                total_loc_savings += result.get('loc_savings', 0)
            
            # Print progress
            status_emoji = {
                'success': '✅',
                'would_process': '📋',
                'skipped': '⚠️',
                'error': '❌'
            }
            emoji = status_emoji.get(result['status'], '❓')
            print(f"{emoji} {file_path.relative_to(self.project_root)}: {result['status']}")
        
        # Summary
        summary = {
            'total_files': len(init_files),
            'successful': len([r for r in results if r['status'] == 'success']),
            'would_process': len([r for r in results if r['status'] == 'would_process']),
            'skipped': len([r for r in results if r['status'] == 'skipped']),
            'errors': len([r for r in results if r['status'] == 'error']),
            'total_loc_savings': total_loc_savings,
            'results': results
        }
        
        return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Standardize __init__.py files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without applying')
    parser.add_argument('--apply', action='store_true', help='Apply standardization changes')
    
    args = parser.parse_args()
    
    standardizer = InitFileStandardizer()
    
    if args.apply:
        print("🚀 Applying init file standardization...")
        summary = standardizer.process_all_files(dry_run=False)
        print(f"\n✅ Standardization complete:")
        print(f"   📊 {summary['successful']}/{summary['total_files']} files processed")
        print(f"   📈 {summary['total_loc_savings']} LOC savings achieved")
        print(f"   ⚠️ {summary['skipped']} complex files skipped")
        print(f"   💾 Backups saved to: {standardizer.backup_dir}")
    else:
        print("📋 Analyzing init files (dry run)...")
        summary = standardizer.process_all_files(dry_run=True)
        print(f"\n📊 Analysis complete:")
        print(f"   📈 Would process {summary['would_process']}/{summary['total_files']} files")
        print(f"   💰 Potential {summary['total_loc_savings']} LOC savings")
        print(f"   ⚠️ {summary['skipped']} complex files need manual review")
        print(f"\n🚀 Run with --apply to execute standardization")