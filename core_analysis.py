#!/usr/bin/env python3
"""
Core System Analysis Script
Analyzes the 313+ files in app/core/ to categorize functionality and identify consolidation opportunities
"""

import os
import re
from collections import defaultdict
from pathlib import Path

# Define functional categories and their patterns
FUNCTIONAL_CATEGORIES = {
    'Agent Management': [
        'agent_', 'spawner', 'registry', 'lifecycle', 'persona', 'identity',
        'load_balancer', 'capability', 'workflow_tracker'
    ],
    'Communication & Messaging': [
        'communication', 'messaging', 'message_', 'pubsub', 'coordination',
        'realtime', 'transcript', 'websocket', 'streams'
    ],
    'Orchestration': [
        'orchestrator', 'automation_engine', 'coordinator', 'workflow_engine',
        'task_router', 'scheduler', 'executor'
    ],
    'Security & Authentication': [
        'security', 'auth', 'oauth', 'rbac', 'mfa', 'webauthn', 'api_key',
        'secret', 'compliance', 'audit', 'threat', 'validator'
    ],
    'Performance & Monitoring': [
        'performance', 'monitoring', 'metrics', 'prometheus', 'benchmarks',
        'health', 'alert', 'observability', 'analytics'
    ],
    'Context & Memory': [
        'context', 'memory', 'semantic', 'embedding', 'vector', 'consolidation',
        'compression', 'knowledge', 'search', 'cache'
    ],
    'External Integration': [
        'github', 'external_tools', 'webhook', 'api_', 'cli_', 'enterprise',
        'pilot', 'customer'
    ],
    'Infrastructure': [
        'database', 'redis', 'container', 'config', 'tmux', 'circuit_breaker',
        'dead_letter', 'dlq', 'backpressure', 'recovery'
    ],
    'Workflow & Task Management': [
        'workflow', 'task_', 'batch_executor', 'queue', 'distributor',
        'execution_engine', 'state_manager'
    ],
    'Self-Modification': [
        'self_modification', 'self_improvement', 'self_optimization',
        'evolutionary', 'meta_learning'
    ]
}

# Orchestrator categorization patterns
ORCHESTRATOR_TYPES = {
    'Core Production': [
        'orchestrator.py', 'production_orchestrator.py', 
        'unified_production_orchestrator.py'
    ],
    'Specialized Orchestrators': [
        'automated_orchestrator.py', 'performance_orchestrator.py',
        'cli_agent_orchestrator.py', 'container_orchestrator.py'
    ],
    'Integration Orchestrators': [
        'context_aware_orchestrator_integration.py',
        'context_orchestrator_integration.py',
        'enhanced_orchestrator_integration.py',
        'orchestrator_hook_integration.py',
        'security_orchestrator_integration.py'
    ],
    'Enterprise/Demo': [
        'enterprise_demo_orchestrator.py', 'pilot_infrastructure_orchestrator.py',
        'vertical_slice_orchestrator.py'
    ],
    'Testing/Performance': [
        'high_concurrency_orchestrator.py', 'orchestrator_load_testing.py',
        'orchestrator_load_balancing_integration.py',
        'orchestrator_shared_state_integration.py'
    ]
}

def analyze_core_directory():
    """Analyze all files in app/core/ directory"""
    core_path = Path('/Users/bogdan/work/leanvibe-dev/bee-hive/app/core')
    
    # Get all Python files
    py_files = list(core_path.glob('*.py'))
    py_files = [f for f in py_files if f.name not in ['__init__.py', 'CLAUDE.md']]
    
    print(f"📊 CORE SYSTEM ANALYSIS")
    print(f"=" * 50)
    print(f"Total Python files: {len(py_files)}")
    print()
    
    # Categorize by functionality
    categorized_files = defaultdict(list)
    uncategorized_files = []
    
    for file_path in py_files:
        filename = file_path.name
        categorized = False
        
        for category, patterns in FUNCTIONAL_CATEGORIES.items():
            for pattern in patterns:
                if pattern in filename.lower():
                    categorized_files[category].append(filename)
                    categorized = True
                    break
            if categorized:
                break
        
        if not categorized:
            uncategorized_files.append(filename)
    
    # Print functional categorization
    print("🎯 FUNCTIONAL CATEGORIZATION")
    print("=" * 30)
    total_categorized = 0
    for category, files in categorized_files.items():
        print(f"{category:25} : {len(files):3d} files")
        total_categorized += len(files)
    
    print(f"{'Uncategorized':25} : {len(uncategorized_files):3d} files")
    print(f"{'TOTAL':25} : {total_categorized + len(uncategorized_files):3d} files")
    print()
    
    # Orchestrator analysis
    print("🎻 ORCHESTRATOR ANALYSIS")
    print("=" * 25)
    orchestrator_files = [f for f in py_files if 'orchestrator' in f.name.lower()]
    print(f"Total orchestrator files: {len(orchestrator_files)}")
    
    for category, patterns in ORCHESTRATOR_TYPES.items():
        matching_files = []
        for pattern in patterns:
            for orch_file in orchestrator_files:
                if orch_file.name == pattern:
                    matching_files.append(orch_file.name)
        if matching_files:
            print(f"{category:25} : {len(matching_files)} files")
            for file in matching_files:
                print(f"  - {file}")
    print()
    
    # Detailed breakdown for top categories
    print("📋 DETAILED BREAKDOWN")
    print("=" * 20)
    for category, files in sorted(categorized_files.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"\n{category} ({len(files)} files):")
        for file in sorted(files)[:10]:  # Show first 10
            print(f"  - {file}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    
    if uncategorized_files:
        print(f"\nUncategorized ({len(uncategorized_files)} files):")
        for file in sorted(uncategorized_files)[:15]:  # Show first 15
            print(f"  - {file}")
        if len(uncategorized_files) > 15:
            print(f"  ... and {len(uncategorized_files) - 15} more")
    
    return {
        'total_files': len(py_files),
        'categorized': categorized_files,
        'uncategorized': uncategorized_files,
        'orchestrators': orchestrator_files
    }

def identify_duplication_patterns():
    """Identify potential duplication patterns"""
    print("\n🔍 DUPLICATION ANALYSIS")
    print("=" * 25)
    
    core_path = Path('/Users/bogdan/work/leanvibe-dev/bee-hive/app/core')
    py_files = list(core_path.glob('*.py'))
    
    # Group by common prefixes/suffixes
    prefix_groups = defaultdict(list)
    suffix_groups = defaultdict(list)
    
    for file_path in py_files:
        filename = file_path.stem  # Without .py extension
        
        # Group by common prefixes (first 2 words)
        words = filename.split('_')
        if len(words) >= 2:
            prefix = '_'.join(words[:2])
            prefix_groups[prefix].append(filename)
        
        # Group by common suffixes (last 2 words)
        if len(words) >= 2:
            suffix = '_'.join(words[-2:])
            suffix_groups[suffix].append(filename)
    
    # Show groups with multiple files (potential duplicates)
    print("🔗 COMMON PREFIXES (potential related functionality):")
    for prefix, files in sorted(prefix_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(files) > 2:  # Show groups with 3+ files
            print(f"  {prefix}_* ({len(files)} files): {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")
    
    print("\n🔗 COMMON SUFFIXES (potential duplicated patterns):")
    for suffix, files in sorted(suffix_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(files) > 2:  # Show groups with 3+ files
            print(f"  *_{suffix} ({len(files)} files): {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")

if __name__ == "__main__":
    results = analyze_core_directory()
    identify_duplication_patterns()