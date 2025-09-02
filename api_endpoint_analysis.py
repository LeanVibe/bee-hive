#!/usr/bin/env python3
"""
Epic 4 Phase 1: Enhanced API Endpoint Analysis
Deep inspection of actual API endpoints, routes, and patterns for precise consolidation design.
"""

import os
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class DetectedEndpoint:
    """Detected API endpoint with full context."""
    method: str
    path: str
    function_name: str
    file_path: str
    line_number: int
    parameters: List[str]
    response_models: List[str]
    dependencies: List[str]
    middleware: List[str]
    security_schemes: List[str]
    tags: List[str]

class EnhancedAPIAnalyzer:
    """Enhanced API analyzer for actual endpoint detection."""
    
    def __init__(self, root_path: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.root_path = Path(root_path)
        self.endpoints: List[DetectedEndpoint] = []
        self.route_patterns = [
            # FastAPI patterns
            r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            # Flask patterns  
            r'@app\.route\(["\']([^"\']+)["\'].*methods=\[([^\]]+)\]',
            r'@bp\.route\(["\']([^"\']+)["\'].*methods=\[([^\]]+)\]'
        ]
    
    def analyze_endpoints_in_file(self, file_path: Path) -> List[DetectedEndpoint]:
        """Analyze actual endpoints in a specific file."""
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for detailed analysis
            tree = ast.parse(content)
            
            # Also use regex for route detection
            endpoints.extend(self._extract_endpoints_with_regex(content, file_path))
            endpoints.extend(self._extract_endpoints_with_ast(tree, file_path))
            
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
        
        return endpoints
    
    def _extract_endpoints_with_regex(self, content: str, file_path: Path) -> List[DetectedEndpoint]:
        """Extract endpoints using regex patterns."""
        endpoints = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in self.route_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if len(match) == 2:
                        method, path = match
                        if isinstance(method, str) and '|' in method:
                            # Handle multiple methods
                            methods = method.split('|')
                            for m in methods:
                                endpoints.append(DetectedEndpoint(
                                    method=m.strip().upper(),
                                    path=path,
                                    function_name=self._find_function_name(lines, i),
                                    file_path=str(file_path),
                                    line_number=i + 1,
                                    parameters=[],
                                    response_models=[],
                                    dependencies=[],
                                    middleware=[],
                                    security_schemes=[],
                                    tags=[]
                                ))
                        else:
                            endpoints.append(DetectedEndpoint(
                                method=method.upper(),
                                path=path,
                                function_name=self._find_function_name(lines, i),
                                file_path=str(file_path),
                                line_number=i + 1,
                                parameters=[],
                                response_models=[],
                                dependencies=[],
                                middleware=[],
                                security_schemes=[],
                                tags=[]
                            ))
        
        return endpoints
    
    def _extract_endpoints_with_ast(self, tree: ast.AST, file_path: Path) -> List[DetectedEndpoint]:
        """Extract endpoints using AST analysis."""
        endpoints = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Look for FastAPI/Flask decorators
                for decorator in node.decorator_list:
                    endpoint = self._analyze_decorator(decorator, node, file_path)
                    if endpoint:
                        endpoints.append(endpoint)
        
        return endpoints
    
    def _analyze_decorator(self, decorator: ast.expr, func_node: ast.FunctionDef, file_path: Path) -> Optional[DetectedEndpoint]:
        """Analyze decorator for API endpoint patterns."""
        try:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    # router.get(), app.post(), etc.
                    attr_name = decorator.func.attr.lower()
                    if attr_name in ['get', 'post', 'put', 'delete', 'patch']:
                        path = self._extract_path_from_call(decorator)
                        if path:
                            return DetectedEndpoint(
                                method=attr_name.upper(),
                                path=path,
                                function_name=func_node.name,
                                file_path=str(file_path),
                                line_number=func_node.lineno,
                                parameters=[arg.arg for arg in func_node.args.args if arg.arg != 'self'],
                                response_models=self._extract_response_models(decorator),
                                dependencies=self._extract_dependencies(decorator),
                                middleware=[],
                                security_schemes=[],
                                tags=self._extract_tags(decorator)
                            )
        except Exception:
            pass
        
        return None
    
    def _extract_path_from_call(self, call: ast.Call) -> Optional[str]:
        """Extract path from decorator call."""
        if call.args and isinstance(call.args[0], ast.Constant):
            return call.args[0].value
        elif call.args and isinstance(call.args[0], ast.Str):  # Python < 3.8
            return call.args[0].s
        return None
    
    def _extract_response_models(self, call: ast.Call) -> List[str]:
        """Extract response models from decorator."""
        models = []
        for keyword in call.keywords:
            if keyword.arg == 'response_model':
                if isinstance(keyword.value, ast.Name):
                    models.append(keyword.value.id)
        return models
    
    def _extract_dependencies(self, call: ast.Call) -> List[str]:
        """Extract dependencies from decorator."""
        deps = []
        for keyword in call.keywords:
            if keyword.arg == 'dependencies':
                if isinstance(keyword.value, ast.List):
                    for item in keyword.value.elts:
                        if isinstance(item, ast.Name):
                            deps.append(item.id)
        return deps
    
    def _extract_tags(self, call: ast.Call) -> List[str]:
        """Extract tags from decorator."""
        tags = []
        for keyword in call.keywords:
            if keyword.arg == 'tags':
                if isinstance(keyword.value, ast.List):
                    for item in keyword.value.elts:
                        if isinstance(item, ast.Constant):
                            tags.append(item.value)
                        elif isinstance(item, ast.Str):  # Python < 3.8
                            tags.append(item.s)
        return tags
    
    def _find_function_name(self, lines: List[str], decorator_line: int) -> str:
        """Find function name following a decorator."""
        for i in range(decorator_line + 1, min(len(lines), decorator_line + 5)):
            line = lines[i].strip()
            if line.startswith('def '):
                match = re.match(r'def\s+(\w+)', line)
                if match:
                    return match.group(1)
        return "unknown_function"

def analyze_sample_api_files():
    """Analyze sample API files to understand endpoint patterns."""
    
    analyzer = EnhancedAPIAnalyzer()
    sample_files = [
        "/Users/bogdan/work/leanvibe-dev/bee-hive/app/api/main.py",
        "/Users/bogdan/work/leanvibe-dev/bee-hive/app/api/routes.py",
        "/Users/bogdan/work/leanvibe-dev/bee-hive/app/api/agent_coordination.py",
        "/Users/bogdan/work/leanvibe-dev/bee-hive/app/api/dashboard_monitoring.py",
        "/Users/bogdan/work/leanvibe-dev/bee-hive/app/api_v2/routers/agents.py"
    ]
    
    all_endpoints = []
    
    print("🔍 Analyzing sample API files for endpoint patterns...")
    
    for file_path in sample_files:
        path_obj = Path(file_path)
        if path_obj.exists():
            print(f"  📄 Analyzing: {path_obj.name}")
            endpoints = analyzer.analyze_endpoints_in_file(path_obj)
            all_endpoints.extend(endpoints)
            print(f"     Found {len(endpoints)} endpoints")
    
    # Generate analysis report
    report = {
        'total_endpoints_found': len(all_endpoints),
        'endpoints_by_method': {},
        'endpoints_by_file': {},
        'common_patterns': [],
        'consolidation_insights': []
    }
    
    # Group by method
    method_counts = defaultdict(int)
    for endpoint in all_endpoints:
        method_counts[endpoint.method] += 1
    report['endpoints_by_method'] = dict(method_counts)
    
    # Group by file
    file_counts = defaultdict(int)
    for endpoint in all_endpoints:
        file_name = Path(endpoint.file_path).name
        file_counts[file_name] += 1
    report['endpoints_by_file'] = dict(file_counts)
    
    # Common path patterns
    paths = [endpoint.path for endpoint in all_endpoints]
    report['common_patterns'] = list(set(paths))
    
    # Sample endpoints for analysis
    report['sample_endpoints'] = [
        {
            'method': ep.method,
            'path': ep.path,
            'function': ep.function_name,
            'file': Path(ep.file_path).name
        } for ep in all_endpoints[:10]
    ]
    
    print(f"\n📊 ENDPOINT ANALYSIS RESULTS:")
    print(f"  🔗 Total endpoints found: {len(all_endpoints)}")
    print(f"  📊 Methods: {dict(method_counts)}")
    print(f"  📁 Files analyzed: {len(sample_files)}")
    
    return report

def main():
    """Execute enhanced API endpoint analysis."""
    print("="*80)
    print("🚀 EPIC 4 PHASE 1: Enhanced API Endpoint Analysis")
    print("="*80)
    
    # Analyze sample files for endpoint patterns
    endpoint_report = analyze_sample_api_files()
    
    # Save report
    report_path = Path("/Users/bogdan/work/leanvibe-dev/bee-hive/epic4_endpoint_analysis.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(endpoint_report, f, indent=2, default=str)
    
    print(f"\n💾 Endpoint analysis saved to: {report_path}")
    return endpoint_report

if __name__ == '__main__':
    main()