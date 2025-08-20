import asyncio
#!/usr/bin/env python3
"""
Dependency Analysis Tool for Manager Consolidation
Analyzes import dependencies and usage patterns between managers.
"""

import os
import re
import ast
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass
class DependencyInfo:
    """Information about manager dependencies."""
    manager_name: str
    file_path: str
    imports_from_core: List[str]
    imports_managers: List[str]
    used_by_managers: List[str]
    external_dependencies: List[str]
    complexity_score: int

class DependencyAnalyzer:
    """Analyzes dependencies between manager classes."""
    
    def __init__(self, core_path: str):
        self.core_path = core_path
        self.manager_files = []
        self.dependencies = {}
        self.usage_matrix = defaultdict(set)
        self.core_imports = defaultdict(set)
        
    def find_manager_files(self) -> List[str]:
        """Find all manager-related files."""
        manager_files = []
        for root, dirs, files in os.walk(self.core_path):
            if '__pycache__' in root:
                continue
            for file in files:
                if (('manager' in file.lower() or 'management' in file.lower() or 'coordinator' in file.lower()) 
                    and file.endswith('.py')):
                    manager_files.append(os.path.join(root, file))
        return manager_files
    
    def analyze_file_dependencies(self, file_path: str) -> DependencyInfo:
        """Analyze dependencies for a single manager file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            imports_from_core = []
            imports_managers = []
            external_dependencies = []
            
            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'core' in alias.name:
                            imports_from_core.append(alias.name)
                        elif 'manager' in alias.name.lower():
                            imports_managers.append(alias.name)
                        else:
                            external_dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if 'core' in node.module:
                            imports_from_core.append(node.module)
                        elif 'manager' in node.module.lower():
                            imports_managers.append(node.module)
                        else:
                            external_dependencies.append(node.module)
            
            # Calculate complexity score based on various factors
            class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            method_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            line_count = len(content.splitlines())
            
            complexity_score = (class_count * 10) + (method_count * 2) + (line_count // 10)
            
            # Extract manager name from file
            manager_name = os.path.basename(file_path).replace('.py', '')
            
            return DependencyInfo(
                manager_name=manager_name,
                file_path=file_path,
                imports_from_core=imports_from_core,
                imports_managers=imports_managers,
                used_by_managers=[],  # Will be populated later
                external_dependencies=external_dependencies,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def build_usage_matrix(self):
        """Build matrix of which managers use which other managers."""
        for file_path in self.manager_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                current_manager = os.path.basename(file_path).replace('.py', '')
                
                # Look for references to other manager files
                for other_file in self.manager_files:
                    if other_file == file_path:
                        continue
                    
                    other_manager = os.path.basename(other_file).replace('.py', '')
                    
                    # Check if other manager is imported or referenced
                    if (other_manager.lower() in content.lower() or 
                        any(other_manager.replace('_', '').lower() in line.lower() 
                            for line in content.split('\n'))):
                        self.usage_matrix[current_manager].add(other_manager)
            
            except Exception as e:
                print(f"Error building usage matrix for {file_path}: {e}")
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies between managers."""
        def has_path(start, end, visited=None):
            if visited is None:
                visited = set()
            if start == end:
                return True
            if start in visited:
                return False
            visited.add(start)
            
            for neighbor in self.usage_matrix.get(start, []):
                if has_path(neighbor, end, visited.copy()):
                    return True
            return False
        
        cycles = []
        for manager in self.usage_matrix:
            for dependency in self.usage_matrix[manager]:
                if has_path(dependency, manager):
                    cycle = [manager, dependency]
                    if cycle not in cycles and cycle[::-1] not in cycles:
                        cycles.append(cycle)
        
        return cycles
    
    def identify_core_clusters(self) -> Dict[str, List[str]]:
        """Identify clusters of tightly coupled managers."""
        # Use simple clustering based on shared dependencies
        clusters = defaultdict(list)
        
        for manager, deps in self.usage_matrix.items():
            # Find other managers with similar dependencies
            similar_managers = []
            for other_manager, other_deps in self.usage_matrix.items():
                if manager != other_manager:
                    shared_deps = len(deps & other_deps)
                    if shared_deps > 0:
                        similar_managers.append((other_manager, shared_deps))
            
            # Sort by similarity and group
            similar_managers.sort(key=lambda x: x[1], reverse=True)
            
            # Assign to cluster based on strongest dependencies
            if similar_managers:
                cluster_key = f"cluster_{len(clusters)}"
                clusters[cluster_key].append(manager)
                for sim_manager, _ in similar_managers[:3]:  # Top 3 similar
                    if sim_manager not in [m for cluster in clusters.values() for m in cluster]:
                        clusters[cluster_key].append(sim_manager)
        
        return dict(clusters)
    
    def generate_dependency_report(self) -> str:
        """Generate comprehensive dependency analysis report."""
        self.manager_files = self.find_manager_files()
        dependencies = []
        
        for file_path in self.manager_files:
            dep_info = self.analyze_file_dependencies(file_path)
            if dep_info:
                dependencies.append(dep_info)
                self.dependencies[dep_info.manager_name] = dep_info
        
        self.build_usage_matrix()
        cycles = self.detect_circular_dependencies()
        clusters = self.identify_core_clusters()
        
        report = []
        report.append("# Manager Dependency Analysis")
        report.append("=" * 40)
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Total Manager Files**: {len(self.manager_files)}")
        report.append(f"- **Circular Dependencies**: {len(cycles)}")
        report.append(f"- **Dependency Clusters**: {len(clusters)}")
        report.append("")
        
        # High-level dependency patterns
        report.append("## High-Level Dependency Patterns")
        
        # Most depended upon managers
        dependency_counts = defaultdict(int)
        for manager, deps in self.usage_matrix.items():
            for dep in deps:
                dependency_counts[dep] += 1
        
        report.append("### Most Depended Upon Managers")
        for manager, count in sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"- **{manager}**: Used by {count} other managers")
        report.append("")
        
        # Most complex managers
        report.append("### Most Complex Managers")
        for dep in sorted(dependencies, key=lambda x: x.complexity_score, reverse=True)[:10]:
            report.append(f"- **{dep.manager_name}**: Complexity score {dep.complexity_score}")
        report.append("")
        
        # Circular dependencies
        if cycles:
            report.append("### Circular Dependencies")
            for cycle in cycles:
                report.append(f"- {' -> '.join(cycle)} -> {cycle[0]}")
            report.append("")
        
        # Dependency clusters
        report.append("### Dependency Clusters")
        for cluster_name, managers in clusters.items():
            if len(managers) > 1:
                report.append(f"**{cluster_name.title()}**: {', '.join(managers)}")
        report.append("")
        
        # Detailed manager analysis
        report.append("## Detailed Manager Analysis")
        
        for dep in sorted(dependencies, key=lambda x: x.complexity_score, reverse=True):
            report.append(f"### {dep.manager_name}")
            report.append(f"- **File**: `{os.path.relpath(dep.file_path, self.core_path)}`")
            report.append(f"- **Complexity Score**: {dep.complexity_score}")
            report.append(f"- **Core Imports**: {len(dep.imports_from_core)}")
            report.append(f"- **Manager Dependencies**: {len(dep.imports_managers)}")
            report.append(f"- **External Dependencies**: {len(dep.external_dependencies)}")
            
            if dep.imports_managers:
                report.append(f"- **Depends On**: {', '.join(dep.imports_managers[:5])}")
            
            used_by = [m for m, deps in self.usage_matrix.items() if dep.manager_name in deps]
            if used_by:
                report.append(f"- **Used By**: {', '.join(used_by[:5])}")
            
            report.append("")
        
        return "\n".join(report)
    
    def analyze(self) -> str:
        """Run complete dependency analysis."""
        return self.generate_dependency_report()

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class DependencyAnalyzerScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            analyzer = DependencyAnalyzer("/Users/bogdan/work/leanvibe-dev/bee-hive/app/core")
            report = analyzer.analyze()
            self.logger.info(report)
            
            return {"status": "completed"}
    
    script_main(DependencyAnalyzerScript)