#!/usr/bin/env python3
"""
Manager Dependency Analysis for LeanVibe Agent Hive
Maps dependencies between manager classes to understand consolidation impact.
"""

import os
import re
import ast
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque
import json

class DependencyMapper:
    """Maps dependencies between manager classes."""
    
    def __init__(self, core_path: str):
        self.core_path = core_path
        self.managers = {}  # manager_name -> file_path
        self.dependencies = defaultdict(set)  # source -> set of dependencies
        self.reverse_dependencies = defaultdict(set)  # target -> set of dependents
        
    def find_manager_files(self):
        """Find all manager files."""
        for root, dirs, files in os.walk(self.core_path):
            if '__pycache__' in root:
                continue
            
            for file in files:
                if 'manager' in file.lower() and file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # Extract manager class names from file
                    manager_names = self.extract_manager_classes(file_path)
                    for name in manager_names:
                        self.managers[name] = file_path
    
    def extract_manager_classes(self, file_path: str) -> List[str]:
        """Extract manager class names from a file."""
        manager_names = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and 'Manager' in node.name:
                    manager_names.append(node.name)
        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return manager_names
    
    def analyze_dependencies(self):
        """Analyze dependencies between manager files."""
        for manager_name, file_path in self.managers.items():
            deps = self.extract_file_dependencies(file_path)
            
            # Filter to only include other managers
            manager_deps = set()
            for dep in deps:
                # Check if this dependency refers to another manager
                for other_manager in self.managers.keys():
                    if other_manager.lower() in dep.lower() or dep in other_manager:
                        manager_deps.add(other_manager)
            
            self.dependencies[manager_name] = manager_deps
            
            # Build reverse dependencies
            for dep in manager_deps:
                self.reverse_dependencies[dep].add(manager_name)
    
    def extract_file_dependencies(self, file_path: str) -> Set[str]:
        """Extract import dependencies from a file."""
        dependencies = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for imports
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.add(node.module)
                        # Also add specific imports
                        for alias in node.names:
                            dependencies.add(alias.name)
            
            # Also look for string-based references
            for line in content.split('\n'):
                # Look for manager references in strings and comments
                if 'manager' in line.lower():
                    # Extract potential manager names
                    words = re.findall(r'\b\w*[Mm]anager\w*\b', line)
                    dependencies.update(words)
        
        except Exception as e:
            print(f"Error analyzing dependencies in {file_path}: {e}")
        
        return dependencies
    
    def find_dependency_clusters(self) -> List[Set[str]]:
        """Find clusters of highly interdependent managers."""
        clusters = []
        visited = set()
        
        for manager in self.managers.keys():
            if manager in visited:
                continue
            
            # Find connected components using BFS
            cluster = set()
            queue = deque([manager])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.add(current)
                
                # Add dependencies and dependents
                for dep in self.dependencies.get(current, set()):
                    if dep not in visited and dep in self.managers:
                        queue.append(dep)
                
                for dep in self.reverse_dependencies.get(current, set()):
                    if dep not in visited and dep in self.managers:
                        queue.append(dep)
            
            if len(cluster) > 1:  # Only include clusters with multiple managers
                clusters.append(cluster)
        
        return clusters
    
    def calculate_consolidation_impact(self) -> Dict[str, any]:
        """Calculate the impact of consolidating managers."""
        impact = {
            'high_coupling': [],
            'low_coupling': [],
            'circular_dependencies': [],
            'consolidation_groups': []
        }
        
        # Find high and low coupling managers
        for manager, deps in self.dependencies.items():
            coupling_score = len(deps) + len(self.reverse_dependencies.get(manager, set()))
            
            if coupling_score > 5:
                impact['high_coupling'].append((manager, coupling_score))
            elif coupling_score <= 1:
                impact['low_coupling'].append((manager, coupling_score))
        
        # Find circular dependencies
        impact['circular_dependencies'] = self.find_circular_dependencies()
        
        # Find consolidation groups
        clusters = self.find_dependency_clusters()
        impact['consolidation_groups'] = [list(cluster) for cluster in clusters]
        
        return impact
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependency chains."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for dep in self.dependencies.get(node, set()):
                if dep in self.managers:
                    dfs(dep, path + [node])
            
            rec_stack.remove(node)
        
        for manager in self.managers.keys():
            if manager not in visited:
                dfs(manager, [])
        
        return cycles
    
    def generate_dependency_report(self) -> str:
        """Generate a comprehensive dependency analysis report."""
        self.find_manager_files()
        self.analyze_dependencies()
        impact = self.calculate_consolidation_impact()
        
        report = []
        report.append("# Manager Dependency Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary Statistics
        report.append("## Summary Statistics")
        report.append(f"- **Total Managers**: {len(self.managers)}")
        report.append(f"- **Total Dependencies**: {sum(len(deps) for deps in self.dependencies.values())}")
        report.append(f"- **High Coupling Managers**: {len(impact['high_coupling'])}")
        report.append(f"- **Low Coupling Managers**: {len(impact['low_coupling'])}")
        report.append(f"- **Circular Dependencies**: {len(impact['circular_dependencies'])}")
        report.append(f"- **Consolidation Groups**: {len(impact['consolidation_groups'])}")
        report.append("")
        
        # High Coupling Analysis
        if impact['high_coupling']:
            report.append("## High Coupling Managers")
            report.append("*Managers with many dependencies (consolidation candidates)*")
            report.append("")
            
            for manager, score in sorted(impact['high_coupling'], key=lambda x: x[1], reverse=True):
                deps = list(self.dependencies.get(manager, set()))
                reverse_deps = list(self.reverse_dependencies.get(manager, set()))
                
                report.append(f"### {manager} (Coupling Score: {score})")
                report.append(f"- **Dependencies**: {', '.join(deps[:5])}")
                if len(deps) > 5:
                    report.append(f"  ... and {len(deps) - 5} more")
                report.append(f"- **Dependents**: {', '.join(reverse_deps[:5])}")
                if len(reverse_deps) > 5:
                    report.append(f"  ... and {len(reverse_deps) - 5} more")
                report.append("")
        
        # Low Coupling Analysis
        if impact['low_coupling']:
            report.append("## Low Coupling Managers")
            report.append("*Independent managers (easy consolidation targets)*")
            report.append("")
            
            for manager, score in impact['low_coupling']:
                report.append(f"- **{manager}**: {score} dependencies")
            report.append("")
        
        # Circular Dependencies
        if impact['circular_dependencies']:
            report.append("## Circular Dependencies")
            report.append("*Dependency cycles that need resolution*")
            report.append("")
            
            for i, cycle in enumerate(impact['circular_dependencies'], 1):
                report.append(f"### Cycle {i}")
                report.append(f"**Path**: {' → '.join(cycle)}")
                report.append("")
        
        # Consolidation Groups
        if impact['consolidation_groups']:
            report.append("## Recommended Consolidation Groups")
            report.append("*Clusters of interdependent managers*")
            report.append("")
            
            for i, group in enumerate(impact['consolidation_groups'], 1):
                if len(group) > 2:  # Only show meaningful groups
                    report.append(f"### Group {i} ({len(group)} managers)")
                    report.append(f"**Managers**: {', '.join(sorted(group))}")
                    
                    # Calculate internal vs external dependencies
                    internal_deps = 0
                    external_deps = 0
                    
                    for manager in group:
                        for dep in self.dependencies.get(manager, set()):
                            if dep in group:
                                internal_deps += 1
                            else:
                                external_deps += 1
                    
                    cohesion = internal_deps / (internal_deps + external_deps) if (internal_deps + external_deps) > 0 else 0
                    report.append(f"**Cohesion Score**: {cohesion:.2%}")
                    report.append("")
        
        # Consolidation Recommendations
        report.append("## Consolidation Recommendations")
        report.append("")
        
        # Prioritize by coupling and grouping
        priorities = []
        
        # High priority: Large consolidation groups with high cohesion
        for group in impact['consolidation_groups']:
            if len(group) >= 3:
                priorities.append({
                    'type': 'group_consolidation',
                    'managers': group,
                    'priority': 'HIGH',
                    'reason': 'Large interdependent cluster'
                })
        
        # Medium priority: High coupling managers
        for manager, score in impact['high_coupling']:
            if score > 8:
                priorities.append({
                    'type': 'hub_consolidation',
                    'managers': [manager],
                    'priority': 'MEDIUM',
                    'reason': f'High coupling hub (score: {score})'
                })
        
        # Low priority: Independent managers
        low_coupling_managers = [m for m, s in impact['low_coupling']]
        if len(low_coupling_managers) >= 3:
            priorities.append({
                'type': 'independent_consolidation',
                'managers': low_coupling_managers,
                'priority': 'LOW',
                'reason': 'Independent managers with minimal dependencies'
            })
        
        for i, priority in enumerate(priorities, 1):
            report.append(f"### Priority {i}: {priority['priority']}")
            report.append(f"**Type**: {priority['type'].replace('_', ' ').title()}")
            report.append(f"**Reason**: {priority['reason']}")
            report.append(f"**Managers**: {', '.join(priority['managers'])}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    mapper = DependencyMapper("/Users/bogdan/work/leanvibe-dev/bee-hive/app/core")
    report = mapper.generate_dependency_report()
    print(report)