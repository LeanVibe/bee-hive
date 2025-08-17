#!/usr/bin/env python3
"""
Core File Analysis Script for Epic 1 Phase 1.2
Analyzes 338 core files and categorizes them by domain for consolidation.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def analyze_core_files(core_dir: str) -> Dict[str, List[str]]:
    """Analyze and categorize core files by domain."""
    
    # Domain patterns for categorization
    domain_patterns = {
        'agent': [
            'agent_', 'ai_architect_agent', 'ai_task_worker', 'code_intelligence_agent',
            'self_optimization_agent', 'multi_agent', 'cross_agent', 'cli_agent',
            'real_agent', 'enhanced_agent'
        ],
        'context': [
            'context_', 'enhanced_context', 'sleep_wake_context', 'workflow_context'
        ],
        'performance': [
            'performance_', 'resource_optimizer', 'capacity_', 'load_', 'benchmark',
            'metrics_', 'monitoring', 'observability_performance', 'vs_2_1_performance',
            'integrated_system_performance'
        ],
        'communication': [
            'communication', 'messaging', 'redis', 'pubsub', 'websocket', 'streams',
            'coordination_', 'team_coordination', 'enhanced_redis', 'optimized_redis'
        ],
        'security': [
            'security', 'auth', 'oauth', 'rbac', 'mfa', 'jwt', 'api_security',
            'production_api_security', 'enterprise_security', 'github_security',
            'webauthn', 'secret_manager', 'audit', 'compliance', 'threat_detection'
        ],
        'workflow': [
            'workflow_', 'task_', 'execution_engine', 'scheduler', 'distributor',
            'intelligent_task', 'enhanced_intelligent_task', 'batch_executor',
            'intelligent_workflow', 'enhanced_workflow'
        ],
        'storage': [
            'database', 'storage', 'cache', 'pgvector', 'embedding', 'vector_search',
            'index_management', 'hybrid_search', 'memory_aware_vector'
        ],
        'orchestration': [
            'orchestrator', 'coordination', 'enhanced_orchestrator', 'production_orchestrator',
            'development_orchestrator', 'unified_orchestrator', 'automated_orchestrator',
            'cli_agent_orchestrator', 'container_orchestrator', 'pilot_infrastructure',
            'global_deployment', 'vertical_slice'
        ]
    }
    
    # Get all Python files in core directory
    core_path = Path(core_dir)
    py_files = [f.name for f in core_path.glob("*.py") if f.is_file()]
    
    # Categorize files
    categorized = defaultdict(list)
    uncategorized = []
    
    for file in py_files:
        # Skip __init__.py and other special files
        if file.startswith('__') or file in ['config.py']:
            continue
            
        categorized_flag = False
        
        for domain, patterns in domain_patterns.items():
            for pattern in patterns:
                if pattern in file:
                    categorized[domain].append(file)
                    categorized_flag = True
                    break
            if categorized_flag:
                break
        
        if not categorized_flag:
            uncategorized.append(file)
    
    # Add uncategorized to a special category
    if uncategorized:
        categorized['uncategorized'] = uncategorized
    
    return dict(categorized)

def analyze_file_dependencies(core_dir: str, files: List[str]) -> Dict[str, Set[str]]:
    """Analyze import dependencies between files."""
    dependencies = {}
    core_path = Path(core_dir)
    
    for file in files:
        file_path = core_path / file
        if not file_path.exists():
            continue
            
        deps = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find imports from other core files
            import_patterns = [
                r'from\s+(?:app\.)?core\.(\w+)\s+import',
                r'import\s+(?:app\.)?core\.(\w+)',
                r'from\s+\.(\w+)\s+import'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if f"{match}.py" in files:
                        deps.add(f"{match}.py")
                        
        except Exception as e:
            print(f"Error analyzing {file}: {e}")
            
        dependencies[file] = deps
    
    return dependencies

def generate_consolidation_report(core_dir: str) -> str:
    """Generate comprehensive consolidation analysis report."""
    
    categorized = analyze_core_files(core_dir)
    all_files = []
    for files in categorized.values():
        all_files.extend(files)
    
    dependencies = analyze_file_dependencies(core_dir, all_files)
    
    report = []
    report.append("# Epic 1 Phase 1.2 - Core File Consolidation Analysis")
    report.append("=" * 60)
    report.append(f"\n**Total Core Files**: {len(all_files)}")
    report.append(f"**Target Reduction**: {len(all_files)} → 70 files (80% reduction)")
    report.append("")
    
    # Domain breakdown
    report.append("## Domain Categorization")
    report.append("-" * 30)
    
    for domain, files in categorized.items():
        report.append(f"\n### {domain.upper()} DOMAIN ({len(files)} files)")
        report.append("Files to consolidate:")
        for file in sorted(files):
            # Show dependency count
            dep_count = len(dependencies.get(file, set()))
            report.append(f"  - {file} ({dep_count} dependencies)")
    
    # Consolidation mapping
    report.append("\n## Consolidation Mapping")
    report.append("-" * 30)
    
    consolidation_map = {
        'agent': 'agent_manager.py',
        'context': 'context_manager.py', 
        'performance': 'resource_manager.py (+ performance plugins)',
        'communication': 'communication_manager.py',
        'security': 'security_manager.py (+ enterprise plugins)',
        'workflow': 'workflow_manager.py',
        'storage': 'storage_manager.py',
        'orchestration': 'Keep existing orchestrator.py (Phase 1.1 complete)'
    }
    
    for domain, target in consolidation_map.items():
        if domain in categorized:
            file_count = len(categorized[domain])
            report.append(f"\n**{domain.upper()}**: {file_count} files → {target}")
            
    # High-priority dependencies
    report.append("\n## Critical Dependencies Analysis")
    report.append("-" * 30)
    
    high_dep_files = [(f, len(deps)) for f, deps in dependencies.items() if len(deps) > 5]
    high_dep_files.sort(key=lambda x: x[1], reverse=True)
    
    report.append("\nFiles with >5 dependencies (consolidation priority):")
    for file, dep_count in high_dep_files[:10]:
        report.append(f"  - {file}: {dep_count} dependencies")
    
    # Implementation recommendations
    report.append("\n## Implementation Strategy")
    report.append("-" * 30)
    report.append("""
1. **Start with lowest dependency files** to avoid circular reference issues
2. **Create manager interfaces first** with dependency injection patterns
3. **Implement plugin architecture** for specialized functionality
4. **Maintain compatibility layer** during transition
5. **Update imports incrementally** across codebase
    """)
    
    return "\n".join(report)

if __name__ == "__main__":
    core_directory = "/Users/bogdan/work/leanvibe-dev/bee-hive/app/core"
    
    print("Analyzing core files for consolidation...")
    report = generate_consolidation_report(core_directory)
    
    # Save report
    with open("core_consolidation_analysis.md", "w") as f:
        f.write(report)
    
    print("Analysis complete! Report saved to core_consolidation_analysis.md")
    print(f"Found {len([f for files in analyze_core_files(core_directory).values() for f in files])} files to analyze")