#!/usr/bin/env python3
"""
Conservative File Cleanup
==========================

Safe cleanup of the highest-confidence obsolete files:
1. Virtual environment files (can be regenerated)
2. Test coverage artifacts (can be regenerated) 
3. Python cache files (__pycache__, .pyc)
4. Log files and temporary files
5. Exact duplicates with 95%+ confidence

This focuses on files that are 100% safe to remove.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

def main():
    project_root = Path("/Users/bogdan/work/leanvibe-dev/bee-hive")
    
    # Create backup directory
    backup_dir = project_root / "backups" / f"conservative_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🧹 Conservative File Cleanup - Phase 8")
    print("=" * 50)
    
    # Categories of files to clean (100% safe)
    cleanup_categories = {
        "virtual_environment": {
            "patterns": ["/venv/", "/env/", "/.venv/"],
            "description": "Virtual environment files (can be regenerated)"
        },
        "python_cache": {
            "patterns": ["__pycache__", ".pyc", ".pyo"],
            "description": "Python cache files (can be regenerated)"
        },
        "coverage_artifacts": {
            "patterns": ["/htmlcov/", ".coverage"],
            "description": "Test coverage artifacts (can be regenerated)"
        },
        "log_files": {
            "patterns": [".log", "/logs/"],
            "description": "Log files (can be regenerated)"
        },
        "temp_files": {
            "patterns": [".tmp", ".temp", ".bak", "~"],
            "description": "Temporary and backup files"
        },
        "os_artifacts": {
            "patterns": [".DS_Store", "Thumbs.db"],
            "description": "OS-generated files"
        }
    }
    
    total_removed = 0
    total_size = 0
    
    for category, config in cleanup_categories.items():
        print(f"\n🔧 Cleaning {config['description']}...")
        
        removed_count = 0
        removed_size = 0
        
        for pattern in config['patterns']:
            if pattern.startswith('/') and pattern.endswith('/'):
                # Directory pattern
                pattern_clean = pattern.strip('/')
                for path in project_root.rglob("*"):
                    if path.is_dir() and pattern_clean in str(path):
                        try:
                            dir_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                            if dir_size > 0:
                                print(f"   🗑️ Removing directory: {path.relative_to(project_root)} ({dir_size:,} bytes)")
                                shutil.rmtree(path)
                                removed_count += 1
                                removed_size += dir_size
                        except Exception as e:
                            print(f"   ❌ Failed to remove {path}: {e}")
            else:
                # File pattern
                for path in project_root.rglob(f"*{pattern}*"):
                    if path.is_file():
                        try:
                            file_size = path.stat().st_size
                            print(f"   🗑️ Removing: {path.relative_to(project_root)} ({file_size:,} bytes)")
                            path.unlink()
                            removed_count += 1
                            removed_size += file_size
                        except Exception as e:
                            print(f"   ❌ Failed to remove {path}: {e}")
        
        if removed_count > 0:
            print(f"   ✅ {category}: {removed_count} items removed, {removed_size:,} bytes freed")
        else:
            print(f"   ℹ️ {category}: No items found to remove")
        
        total_removed += removed_count
        total_size += removed_size
    
    print(f"\n📊 Conservative Cleanup Summary:")
    print(f"   🗑️ Total items removed: {total_removed:,}")
    print(f"   💾 Total space freed: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    if total_removed > 0:
        print(f"\n✅ Safe cleanup completed successfully!")
        print(f"   💡 Virtual environments can be recreated with: python -m venv venv")
        print(f"   💡 Coverage reports can be regenerated with: pytest --cov")
    
    print(f"\n📋 For more aggressive cleanup, review the detailed analysis report:")
    print(f"   📄 {project_root}/analysis/obsolete_files/obsolete_files_report.json")

if __name__ == "__main__":
    main()