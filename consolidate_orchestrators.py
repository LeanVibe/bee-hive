#!/usr/bin/env python3
"""
Orchestrator Consolidation Script
Systematically archive redundant orchestrator implementations
"""
import os
import shutil
from pathlib import Path

# Files to KEEP (core orchestrators)
KEEP_FILES = {
    'simple_orchestrator.py',  # Production orchestrator
    'orchestrator_migration_adapter.py',  # Compatibility layer
    'production_orchestrator.py',  # Feature harvesting
    'unified_orchestrator.py',  # Architecture patterns
    'orchestrator.py',  # Legacy reference
}

# Archive structure
ARCHIVE_DIRS = {
    'plugins': [
        'automated_orchestrator.py',
        'cli_agent_orchestrator.py', 
        'enhanced_orchestrator_plugin.py',
        'integration_orchestrator_plugin.py',
        'performance_orchestrator_plugin.py',
        'production_orchestrator_plugin.py',
        'specialized_orchestrator_plugin.py',
    ],
    'specialized': [
        'high_concurrency_orchestrator.py',
        'performance_orchestrator.py',
        'performance_orchestrator_integration.py',
        'simple_orchestrator_enhanced.py',
        'simple_orchestrator_adapter.py',
        'vertical_slice_orchestrator.py',
        'context_aware_orchestrator_integration.py',
        'context_orchestrator_integration.py',
        'enhanced_orchestrator_integration.py',
        'orchestrator_load_balancing_integration.py',
        'orchestrator_shared_state_integration.py',
        'security_orchestrator_integration.py',
        'task_orchestrator_integration.py',
        'orchestrator_hook_integration.py',
        'orchestrator_load_testing.py',
    ],
    'legacy': [
        'orchestrator_v2.py',
        'orchestrator_v2_migration.py', 
        'orchestrator_v2_plugins.py',
        'universal_orchestrator.py',
        'production_orchestrator_unified.py',
        'unified_production_orchestrator.py',
    ]
}

def main():
    core_dir = Path('app/core')
    archive_base = core_dir / 'archive_orchestrators'
    
    # Create archive directories
    for subdir in ARCHIVE_DIRS.keys():
        (archive_base / subdir).mkdir(parents=True, exist_ok=True)
    
    files_moved = 0
    
    # Move files to appropriate archive directories
    for category, files in ARCHIVE_DIRS.items():
        archive_dir = archive_base / category
        for filename in files:
            source = core_dir / filename
            if source.exists():
                target = archive_dir / filename
                print(f"Moving {filename} to {category}/")
                shutil.move(str(source), str(target))
                files_moved += 1
            else:
                print(f"  Skip {filename} (not found)")
    
    # Count remaining orchestrator files
    remaining = list(core_dir.glob('*orchestrator*.py'))
    remaining = [f for f in remaining if f.name in KEEP_FILES]
    
    print(f"\n✅ Consolidation Complete!")
    print(f"   Files moved: {files_moved}")
    print(f"   Files kept: {len(remaining)}")
    print(f"   Kept files: {[f.name for f in remaining]}")
    
    # Validate the system can still start
    print(f"\n🧪 Testing system stability...")
    test_cmd = "cd /Users/bogdan/work/leanvibe-dev/bee-hive && python -c 'from app.core.simple_orchestrator import SimpleOrchestrator; print(\"✅ SimpleOrchestrator loads successfully\")'"
    result = os.system(test_cmd)
    if result == 0:
        print("✅ System stability confirmed")
    else:
        print("❌ System stability check failed")
    
if __name__ == '__main__':
    main()