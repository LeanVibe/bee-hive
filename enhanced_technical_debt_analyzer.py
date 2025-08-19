#!/usr/bin/env python3
"""
Enhanced Technical Debt Analyzer for LeanVibe Agent Hive 2.0

Post-Epic 1 Phase 3: Advanced debt detection using project index integration.
Builds upon the existing analyzer with sophisticated duplicate detection,
semantic analysis, and AI-driven consolidation recommendations.

Features:
- AST-based duplicate code detection
- Semantic similarity analysis
- Import dependency graph analysis
- Architecture pattern recognition
- ROI-based prioritization
- Integration with project indexing system
"""

import ast
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import difflib
import networkx as nx

# Add project root to path
sys.path.insert(0, '/Users/bogdan/work/leanvibe-dev/bee-hive')


@dataclass
class CodeClone:
    """Represents a detected code clone/duplicate."""
    clone_type: str  # "exact", "structural", "semantic"
    similarity_score: float  # 0.0 to 1.0
    files: List[str]
    line_ranges: List[Tuple[int, int]]
    estimated_consolidation_savings: int  # lines of code
    consolidation_complexity: str  # "low", "medium", "high"
    recommended_action: str
    

@dataclass
class ArchitecturalDebt:
    """Represents architectural-level technical debt."""
    debt_category: str
    severity: str  # "critical", "high", "medium", "low"
    pattern_name: str
    affected_components: List[str]
    root_cause: str
    business_impact: str
    technical_impact: str
    remediation_strategy: str
    effort_estimate: str
    roi_score: float  # Return on investment score


@dataclass
class DependencyIssue:
    """Represents dependency-related technical debt."""
    issue_type: str  # "circular", "excessive_coupling", "unused", "outdated"
    severity: str
    components: List[str]
    dependency_graph: Dict[str, List[str]]
    impact_analysis: str
    resolution_steps: List[str]


class EnhancedTechnicalDebtAnalyzer:
    """Advanced technical debt analyzer with semantic understanding."""
    
    def __init__(self, project_root: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.project_root = Path(project_root)
        self.analysis_timestamp = datetime.utcnow()
        
        # Analysis results
        self.code_clones: List[CodeClone] = []
        self.architectural_debts: List[ArchitecturalDebt] = []
        self.dependency_issues: List[DependencyIssue] = []
        self.file_metrics: Dict[str, Dict] = {}
        self.dependency_graph = nx.DiGraph()
        
        # Analysis configuration
        self.similarity_thresholds = {
            "exact_match": 0.95,
            "structural_similarity": 0.80,
            "semantic_similarity": 0.70
        }
        
        self.architectural_patterns = {
            "orchestrator": {
                "keywords": ["orchestrat", "coordinat", "delegat", "schedul", "dispatch"],
                "anti_patterns": ["god_object", "excessive_responsibility", "tight_coupling"]
            },
            "manager": {
                "keywords": ["manag", "lifecycl", "resourc", "memor", "context"],
                "anti_patterns": ["excessive_managers", "manager_proliferation", "unclear_boundaries"]
            },
            "engine": {
                "keywords": ["engine", "process", "execut", "handl", "work"],
                "anti_patterns": ["duplicate_engines", "performance_overhead", "resource_conflicts"]
            },
            "service": {
                "keywords": ["service", "provider", "client", "interface", "protocol"],
                "anti_patterns": ["service_proliferation", "inconsistent_interfaces", "protocol_duplication"]
            }
        }
        
        print(f"🔬 Enhanced Technical Debt Analyzer initialized for: {project_root}")
        print(f"⏰ Analysis timestamp: {self.analysis_timestamp}")

    async def analyze_post_epic1_debt(self):
        """Main analysis entry point, focusing on post-Epic 1 Phase 3 debt."""
        print("\n🚀 Post-Epic 1 Phase 3: Enhanced Technical Debt Analysis")
        print("=" * 70)
        
        analysis_start = time.time()
        
        # Phase 1: File discovery and basic metrics
        print("\n📁 Phase 1: File discovery and basic metrics collection...")
        await self._collect_file_metrics()
        
        # Phase 2: Dependency graph construction
        print("\n🕸️  Phase 2: Building dependency graph...")
        await self._build_dependency_graph()
        
        # Phase 3: Code clone detection
        print("\n👯 Phase 3: Advanced code clone detection...")
        await self._detect_code_clones()
        
        # Phase 4: Architectural debt analysis
        print("\n🏗️  Phase 4: Architectural pattern analysis...")
        await self._analyze_architectural_debt()
        
        # Phase 5: Dependency issue analysis
        print("\n🔗 Phase 5: Dependency issue analysis...")
        await self._analyze_dependency_issues()
        
        # Phase 6: ROI calculation and prioritization
        print("\n💰 Phase 6: ROI calculation and prioritization...")
        await self._calculate_roi_scores()
        
        analysis_time = time.time() - analysis_start
        
        # Generate comprehensive report
        report = await self._generate_enhanced_report(analysis_time)
        
        return report
    
    async def _collect_file_metrics(self):
        """Collect comprehensive metrics for each Python file."""
        python_files = list(self.project_root.rglob("*.py"))
        
        # Filter out files that should be skipped
        python_files = [f for f in python_files if not self._should_skip_file(f)]
        
        print(f"   📊 Analyzing {len(python_files)} Python files...")
        
        for file_path in python_files:
            try:
                metrics = await self._analyze_file_comprehensive(file_path)
                if metrics:
                    rel_path = str(file_path.relative_to(self.project_root))
                    self.file_metrics[rel_path] = metrics
            except Exception as e:
                print(f"   ⚠️  Error analyzing {file_path}: {e}")
        
        print(f"   ✅ Collected metrics for {len(self.file_metrics)} files")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Enhanced file filtering."""
        skip_patterns = [
            "__pycache__", ".git", "migrations", "node_modules", 
            ".pytest_cache", ".backup", "test_", "tests/",
            ".pyc", "venv", ".env"
        ]
        
        str_path = str(file_path)
        return any(pattern in str_path for pattern in skip_patterns)
    
    async def _analyze_file_comprehensive(self, file_path: Path) -> Optional[Dict]:
        """Comprehensive file analysis with AST parsing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return {
                    "error": f"Syntax error: {e}",
                    "total_lines": len(lines),
                    "parseable": False
                }
            
            # Comprehensive metrics
            metrics = {
                "total_lines": len(lines),
                "non_empty_lines": len([l for l in lines if l.strip()]),
                "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
                "docstring_lines": 0,
                "classes": [],
                "functions": [],
                "imports": [],
                "complexity_indicators": {},
                "architectural_patterns": [],
                "potential_clones": [],
                "ast_hash": "",
                "parseable": True
            }
            
            # AST-based analysis
            await self._analyze_ast_comprehensive(tree, content, metrics)
            
            # Pattern recognition
            await self._identify_architectural_patterns(content, metrics)
            
            # Generate AST hash for structural comparison
            metrics["ast_hash"] = self._generate_ast_hash(tree)
            
            return metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "total_lines": 0,
                "parseable": False
            }
    
    async def _analyze_ast_comprehensive(self, tree: ast.AST, content: str, metrics: Dict):
        """Comprehensive AST analysis."""
        
        class ComprehensiveVisitor(ast.NodeVisitor):
            def __init__(self, metrics):
                self.metrics = metrics
                self.nesting_depth = 0
                self.max_nesting = 0
                self.function_complexity = {}
                
            def visit_ClassDef(self, node):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [],
                    "base_classes": [base.id if hasattr(base, 'id') else str(base) for base in node.bases],
                    "decorators": [dec.id if hasattr(dec, 'id') else str(dec) for dec in node.decorator_list]
                }
                
                # Count methods
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        class_info["methods"].append(child.name)
                
                # Check for docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    self.metrics["docstring_lines"] += len(docstring.splitlines())
                
                self.metrics["classes"].append(class_info)
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args_count": len(node.args.args),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "decorators": [dec.id if hasattr(dec, 'id') else str(dec) for dec in node.decorator_list],
                    "returns": hasattr(node, 'returns') and node.returns is not None
                }
                
                # Calculate cyclomatic complexity (simplified)
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                
                func_info["complexity"] = complexity
                self.function_complexity[node.name] = complexity
                
                # Check for docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    self.metrics["docstring_lines"] += len(docstring.splitlines())
                
                self.metrics["functions"].append(func_info)
                self.generic_visit(node)
                
            def visit_Import(self, node):
                for alias in node.names:
                    self.metrics["imports"].append({
                        "name": alias.name,
                        "alias": alias.asname,
                        "type": "import"
                    })
                    
            def visit_ImportFrom(self, node):
                module = node.module or ""
                for alias in node.names:
                    self.metrics["imports"].append({
                        "name": f"{module}.{alias.name}",
                        "alias": alias.asname,
                        "module": module,
                        "type": "from_import"
                    })
                    
            def visit_If(self, node):
                self.nesting_depth += 1
                self.max_nesting = max(self.max_nesting, self.nesting_depth)
                self.generic_visit(node)
                self.nesting_depth -= 1
                
            def visit_For(self, node):
                self.nesting_depth += 1
                self.max_nesting = max(self.max_nesting, self.nesting_depth)
                self.generic_visit(node)
                self.nesting_depth -= 1
                
            def visit_While(self, node):
                self.nesting_depth += 1
                self.max_nesting = max(self.max_nesting, self.nesting_depth)
                self.generic_visit(node)
                self.nesting_depth -= 1
        
        visitor = ComprehensiveVisitor(metrics)
        visitor.visit(tree)
        
        # Store complexity indicators
        metrics["complexity_indicators"] = {
            "max_nesting_depth": visitor.max_nesting,
            "average_function_complexity": sum(visitor.function_complexity.values()) / len(visitor.function_complexity) if visitor.function_complexity else 0,
            "high_complexity_functions": [name for name, complexity in visitor.function_complexity.items() if complexity > 10]
        }
    
    async def _identify_architectural_patterns(self, content: str, metrics: Dict):
        """Identify architectural patterns in the file."""
        content_lower = content.lower()
        
        for pattern_name, pattern_config in self.architectural_patterns.items():
            keyword_matches = sum(1 for keyword in pattern_config["keywords"] if keyword in content_lower)
            
            if keyword_matches >= 2:  # Threshold for pattern recognition
                pattern_strength = keyword_matches / len(pattern_config["keywords"])
                
                metrics["architectural_patterns"].append({
                    "pattern": pattern_name,
                    "strength": pattern_strength,
                    "keywords_matched": keyword_matches
                })
    
    def _generate_ast_hash(self, tree: ast.AST) -> str:
        """Generate a hash representing the AST structure."""
        # Simplified AST structure extraction
        structure_elements = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                structure_elements.append(f"class:{node.name}")
            elif isinstance(node, ast.FunctionDef):
                structure_elements.append(f"func:{node.name}:{len(node.args.args)}")
            elif isinstance(node, ast.If):
                structure_elements.append("if")
            elif isinstance(node, ast.For):
                structure_elements.append("for")
            elif isinstance(node, ast.While):
                structure_elements.append("while")
        
        structure_string = "|".join(structure_elements)
        return hashlib.md5(structure_string.encode()).hexdigest()
    
    async def _build_dependency_graph(self):
        """Build a comprehensive dependency graph."""
        print("   🔗 Building dependency relationships...")
        
        for file_path, metrics in self.file_metrics.items():
            if not metrics.get("parseable", False):
                continue
                
            # Add node to graph
            self.dependency_graph.add_node(file_path, **{
                "lines": metrics.get("total_lines", 0),
                "classes": len(metrics.get("classes", [])),
                "functions": len(metrics.get("functions", []))
            })
            
            # Add dependency edges
            for import_info in metrics.get("imports", []):
                import_name = import_info["name"]
                
                # Try to resolve to actual file
                resolved_file = self._resolve_import_to_file(import_name)
                if resolved_file and resolved_file in self.file_metrics:
                    self.dependency_graph.add_edge(file_path, resolved_file, 
                                                 import_type=import_info["type"])
        
        print(f"   📊 Dependency graph: {len(self.dependency_graph.nodes)} nodes, {len(self.dependency_graph.edges)} edges")
    
    def _resolve_import_to_file(self, import_name: str) -> Optional[str]:
        """Attempt to resolve an import to its file path."""
        # Simple heuristic-based resolution
        for file_path in self.file_metrics.keys():
            file_name = Path(file_path).stem
            
            # Direct name match
            if import_name.endswith(file_name):
                return file_path
            
            # Module path match
            if import_name.replace('.', '/') in file_path:
                return file_path
        
        return None
    
    async def _detect_code_clones(self):
        """Advanced code clone detection."""
        print("   🔍 Detecting code clones with multiple strategies...")
        
        # Strategy 1: Exact hash matching
        await self._detect_exact_clones()
        
        # Strategy 2: Structural similarity
        await self._detect_structural_clones()
        
        # Strategy 3: Function-level similarity
        await self._detect_function_clones()
        
        print(f"   📋 Found {len(self.code_clones)} clone clusters")
    
    async def _detect_exact_clones(self):
        """Detect exact code clones using AST hashes."""
        hash_to_files = defaultdict(list)
        
        for file_path, metrics in self.file_metrics.items():
            if not metrics.get("parseable", False):
                continue
                
            ast_hash = metrics.get("ast_hash", "")
            if ast_hash and metrics.get("total_lines", 0) > 20:  # Only consider substantial files
                hash_to_files[ast_hash].append(file_path)
        
        for ast_hash, files in hash_to_files.items():
            if len(files) > 1:
                total_lines = sum(self.file_metrics[f].get("total_lines", 0) for f in files)
                
                self.code_clones.append(CodeClone(
                    clone_type="exact",
                    similarity_score=1.0,
                    files=files,
                    line_ranges=[(1, self.file_metrics[f].get("total_lines", 0)) for f in files],
                    estimated_consolidation_savings=total_lines - max(self.file_metrics[f].get("total_lines", 0) for f in files),
                    consolidation_complexity="low",
                    recommended_action=f"Exact duplicates - consolidate into single file and update imports"
                ))
    
    async def _detect_structural_clones(self):
        """Detect structurally similar code."""
        # Group files by similar patterns and complexity
        pattern_groups = defaultdict(list)
        
        for file_path, metrics in self.file_metrics.items():
            if not metrics.get("parseable", False):
                continue
                
            # Create a signature based on classes, functions, and complexity
            signature = {
                "class_count": len(metrics.get("classes", [])),
                "function_count": len(metrics.get("functions", [])),
                "max_nesting": metrics.get("complexity_indicators", {}).get("max_nesting_depth", 0),
                "patterns": tuple(sorted([p["pattern"] for p in metrics.get("architectural_patterns", [])]))
            }
            
            signature_key = (
                signature["class_count"],
                signature["function_count"], 
                signature["max_nesting"],
                signature["patterns"]
            )
            
            pattern_groups[signature_key].append(file_path)
        
        # Find groups with similar structure
        for signature_key, files in pattern_groups.items():
            if len(files) > 1 and signature_key[0] > 0:  # Must have classes or significant structure
                # Calculate structural similarity
                similarity_scores = []
                for i, file1 in enumerate(files):
                    for file2 in files[i+1:]:
                        similarity = self._calculate_structural_similarity(file1, file2)
                        similarity_scores.append(similarity)
                
                avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                
                if avg_similarity >= self.similarity_thresholds["structural_similarity"]:
                    total_lines = sum(self.file_metrics[f].get("total_lines", 0) for f in files)
                    
                    self.code_clones.append(CodeClone(
                        clone_type="structural",
                        similarity_score=avg_similarity,
                        files=files,
                        line_ranges=[(1, self.file_metrics[f].get("total_lines", 0)) for f in files],
                        estimated_consolidation_savings=int(total_lines * 0.6),  # Conservative estimate
                        consolidation_complexity="medium",
                        recommended_action=f"Extract common structure into base class or mixin"
                    ))
    
    def _calculate_structural_similarity(self, file1: str, file2: str) -> float:
        """Calculate structural similarity between two files."""
        metrics1 = self.file_metrics[file1]
        metrics2 = self.file_metrics[file2]
        
        # Compare class names
        classes1 = set(c["name"] for c in metrics1.get("classes", []))
        classes2 = set(c["name"] for c in metrics2.get("classes", []))
        class_similarity = len(classes1 & classes2) / max(len(classes1 | classes2), 1)
        
        # Compare function names
        functions1 = set(f["name"] for f in metrics1.get("functions", []))
        functions2 = set(f["name"] for f in metrics2.get("functions", []))
        function_similarity = len(functions1 & functions2) / max(len(functions1 | functions2), 1)
        
        # Compare import patterns
        imports1 = set(imp["name"] for imp in metrics1.get("imports", []))
        imports2 = set(imp["name"] for imp in metrics2.get("imports", []))
        import_similarity = len(imports1 & imports2) / max(len(imports1 | imports2), 1)
        
        # Weighted average
        return (class_similarity * 0.4 + function_similarity * 0.4 + import_similarity * 0.2)
    
    async def _detect_function_clones(self):
        """Detect duplicate functions across files."""
        function_signatures = defaultdict(list)
        
        for file_path, metrics in self.file_metrics.items():
            if not metrics.get("parseable", False):
                continue
                
            for func in metrics.get("functions", []):
                # Create function signature
                signature = (
                    func["name"],
                    func["args_count"], 
                    func.get("complexity", 0),
                    func.get("is_async", False)
                )
                
                function_signatures[signature].append((file_path, func))
        
        # Find duplicate function signatures
        for signature, occurrences in function_signatures.items():
            if len(occurrences) > 1:
                files = [occ[0] for occ in occurrences]
                
                # Estimate lines for these functions (rough approximation)
                estimated_lines = max(20, signature[2] * 5)  # Based on complexity
                total_lines = estimated_lines * len(occurrences)
                savings = total_lines - estimated_lines
                
                self.code_clones.append(CodeClone(
                    clone_type="functional",
                    similarity_score=0.9,  # High similarity based on signature match
                    files=files,
                    line_ranges=[(occ[1]["line"], occ[1]["line"] + estimated_lines) for occ in occurrences],
                    estimated_consolidation_savings=savings,
                    consolidation_complexity="medium",
                    recommended_action=f"Extract function '{signature[0]}' into shared utility module"
                ))
    
    async def _analyze_architectural_debt(self):
        """Analyze architectural-level technical debt."""
        print("   🏗️  Analyzing architectural debt patterns...")
        
        # Analyze orchestrator pattern debt
        await self._analyze_orchestrator_debt()
        
        # Analyze manager proliferation debt  
        await self._analyze_manager_debt()
        
        # Analyze engine redundancy debt
        await self._analyze_engine_debt()
        
        # Analyze service proliferation debt
        await self._analyze_service_debt()
        
        print(f"   📊 Identified {len(self.architectural_debts)} architectural debt issues")
    
    async def _analyze_orchestrator_debt(self):
        """Analyze orchestrator-related architectural debt."""
        orchestrator_files = [
            f for f, metrics in self.file_metrics.items() 
            if any(pattern["pattern"] == "orchestrator" for pattern in metrics.get("architectural_patterns", []))
        ]
        
        if len(orchestrator_files) > 5:  # Post-Epic 1, this should be much lower
            total_lines = sum(self.file_metrics[f].get("total_lines", 0) for f in orchestrator_files)
            
            # This should be drastically reduced post-Epic 1 Phase 3
            if len(orchestrator_files) > 10:  # Still too many
                severity = "high"
                impact = "Significant orchestrator redundancy remains after Epic 1 Phase 3 consolidation"
            else:
                severity = "medium"
                impact = "Some orchestrator patterns could still be consolidated"
            
            self.architectural_debts.append(ArchitecturalDebt(
                debt_category="Architecture",
                severity=severity,
                pattern_name="Orchestrator Pattern Debt",
                affected_components=orchestrator_files,
                root_cause="Incomplete consolidation or new orchestrator patterns introduced",
                business_impact=impact,
                technical_impact=f"Maintenance overhead across {len(orchestrator_files)} files",
                remediation_strategy="Further consolidation into the legacy compatibility plugin pattern",
                effort_estimate="2-4 weeks",
                roi_score=0.8 if severity == "high" else 0.6
            ))
    
    async def _analyze_manager_debt(self):
        """Analyze manager-related architectural debt."""
        manager_files = [
            f for f, metrics in self.file_metrics.items()
            if any(pattern["pattern"] == "manager" for pattern in metrics.get("architectural_patterns", []))
        ]
        
        if len(manager_files) > 20:  # Threshold for excessive managers
            # Group by domain to identify consolidation opportunities
            domain_groups = self._group_managers_by_domain(manager_files)
            
            for domain, files in domain_groups.items():
                if len(files) > 3:  # More than 3 managers in one domain
                    total_lines = sum(self.file_metrics[f].get("total_lines", 0) for f in files)
                    
                    self.architectural_debts.append(ArchitecturalDebt(
                        debt_category="Code Organization",
                        severity="medium",
                        pattern_name=f"{domain.title()} Manager Proliferation",
                        affected_components=files,
                        root_cause="Lack of unified manager pattern for domain",
                        business_impact="Increased maintenance burden and testing complexity",
                        technical_impact=f"Duplicate functionality across {len(files)} manager classes",
                        remediation_strategy=f"Consolidate into unified {domain} manager with plugin architecture",
                        effort_estimate="3-5 weeks",
                        roi_score=0.7
                    ))
    
    def _group_managers_by_domain(self, manager_files: List[str]) -> Dict[str, List[str]]:
        """Group manager files by functional domain."""
        domain_groups = defaultdict(list)
        
        domain_keywords = {
            "memory": ["memory", "context", "cache", "storage"],
            "communication": ["message", "event", "websocket", "redis", "pubsub"],
            "security": ["auth", "security", "permission", "access", "jwt"],
            "workflow": ["task", "job", "workflow", "execution", "pipeline"],
            "agent": ["agent", "lifecycle", "spawn", "coordinate"],
            "performance": ["performance", "metric", "monitor", "benchmark"],
            "configuration": ["config", "setting", "feature", "flag"]
        }
        
        for file_path in manager_files:
            file_name = Path(file_path).stem.lower()
            
            # Determine domain based on file name and imports
            assigned_domain = "general"
            for domain, keywords in domain_keywords.items():
                if any(keyword in file_name for keyword in keywords):
                    assigned_domain = domain
                    break
            
            domain_groups[assigned_domain].append(file_path)
        
        return dict(domain_groups)
    
    async def _analyze_engine_debt(self):
        """Analyze engine-related architectural debt."""
        engine_files = [
            f for f, metrics in self.file_metrics.items()
            if any(pattern["pattern"] == "engine" for pattern in metrics.get("architectural_patterns", []))
        ]
        
        if len(engine_files) > 15:  # Too many engines
            total_lines = sum(self.file_metrics[f].get("total_lines", 0) for f in engine_files)
            
            self.architectural_debts.append(ArchitecturalDebt(
                debt_category="Performance",
                severity="high",
                pattern_name="Engine Pattern Proliferation",
                affected_components=engine_files,
                root_cause="Lack of unified engine architecture with specialization",
                business_impact="Resource inefficiency and performance degradation",
                technical_impact=f"Duplicate processing logic across {len(engine_files)} engines",
                remediation_strategy="Consolidate into specialized high-performance engines with plugin system",
                effort_estimate="4-6 weeks",
                roi_score=0.75
            ))
    
    async def _analyze_service_debt(self):
        """Analyze service-related architectural debt."""
        service_files = [
            f for f, metrics in self.file_metrics.items()
            if any(pattern["pattern"] == "service" for pattern in metrics.get("architectural_patterns", []))
        ]
        
        if len(service_files) > 25:  # Too many services
            # Analyze for protocol duplication
            protocol_patterns = defaultdict(list)
            
            for file_path in service_files:
                metrics = self.file_metrics[file_path]
                # Group by similar import patterns (protocols)
                import_signature = tuple(sorted(imp["name"] for imp in metrics.get("imports", [])[:5]))
                protocol_patterns[import_signature].append(file_path)
            
            for signature, files in protocol_patterns.items():
                if len(files) > 3:  # Multiple files with same protocol pattern
                    total_lines = sum(self.file_metrics[f].get("total_lines", 0) for f in files)
                    
                    self.architectural_debts.append(ArchitecturalDebt(
                        debt_category="Integration",
                        severity="medium",
                        pattern_name="Service Protocol Duplication",
                        affected_components=files,
                        root_cause="Inconsistent service interface patterns",
                        business_impact="Integration complexity and maintenance overhead",
                        technical_impact=f"Duplicate protocol implementation across {len(files)} services",
                        remediation_strategy="Standardize on unified service interface pattern",
                        effort_estimate="2-4 weeks",
                        roi_score=0.65
                    ))
    
    async def _analyze_dependency_issues(self):
        """Analyze dependency-related issues."""
        print("   🔗 Analyzing dependency issues...")
        
        # Detect circular dependencies
        await self._detect_circular_dependencies()
        
        # Detect excessive coupling
        await self._detect_excessive_coupling()
        
        # Detect unused dependencies
        await self._detect_unused_dependencies()
        
        print(f"   📊 Found {len(self.dependency_issues)} dependency issues")
    
    async def _detect_circular_dependencies(self):
        """Detect circular dependency chains."""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            
            for cycle in cycles:
                if len(cycle) > 1:  # Actual cycle, not self-loop
                    cycle_description = " → ".join(cycle) + " → " + cycle[0]
                    
                    self.dependency_issues.append(DependencyIssue(
                        issue_type="circular",
                        severity="high",
                        components=cycle,
                        dependency_graph={node: list(self.dependency_graph.successors(node)) for node in cycle},
                        impact_analysis=f"Circular dependency prevents clean module organization and testing",
                        resolution_steps=[
                            "Identify the core abstraction causing the cycle",
                            "Extract interface or base class to break the cycle",
                            "Use dependency injection to resolve circular references",
                            "Refactor imports to be more specific and avoid module-level cycles"
                        ]
                    ))
            
        except Exception as e:
            print(f"   ⚠️  Error detecting cycles: {e}")
    
    async def _detect_excessive_coupling(self):
        """Detect components with excessive dependencies."""
        # Calculate in-degree and out-degree for each node
        for node in self.dependency_graph.nodes():
            in_degree = self.dependency_graph.in_degree(node)
            out_degree = self.dependency_graph.out_degree(node)
            
            # Thresholds for excessive coupling
            if in_degree > 10:  # Too many dependencies on this module
                self.dependency_issues.append(DependencyIssue(
                    issue_type="excessive_coupling",
                    severity="medium",
                    components=[node],
                    dependency_graph={node: list(self.dependency_graph.predecessors(node))},
                    impact_analysis=f"Module has {in_degree} dependents - changes will have wide impact",
                    resolution_steps=[
                        "Split module into smaller, focused components",
                        "Extract stable interfaces to reduce coupling",
                        "Use event-driven patterns to decouple components"
                    ]
                ))
            
            if out_degree > 15:  # This module depends on too many others
                self.dependency_issues.append(DependencyIssue(
                    issue_type="excessive_coupling",
                    severity="medium", 
                    components=[node],
                    dependency_graph={node: list(self.dependency_graph.successors(node))},
                    impact_analysis=f"Module depends on {out_degree} other modules - high complexity",
                    resolution_steps=[
                        "Apply dependency injection to reduce direct dependencies",
                        "Use facade pattern to simplify external dependencies",
                        "Extract common dependencies into shared utilities"
                    ]
                ))
    
    async def _detect_unused_dependencies(self):
        """Detect potentially unused dependencies."""
        # Simple heuristic: files with imports but no incoming dependencies
        isolated_nodes = [node for node in self.dependency_graph.nodes() 
                         if self.dependency_graph.in_degree(node) == 0 
                         and self.dependency_graph.out_degree(node) > 0]
        
        if isolated_nodes:
            self.dependency_issues.append(DependencyIssue(
                issue_type="unused",
                severity="low",
                components=isolated_nodes,
                dependency_graph={},
                impact_analysis="Potentially unused modules consuming resources and maintenance effort",
                resolution_steps=[
                    "Verify if modules are actually unused or just not captured in static analysis",
                    "Check for dynamic imports or runtime usage",
                    "Remove truly unused modules to reduce codebase size"
                ]
            ))
    
    async def _calculate_roi_scores(self):
        """Calculate ROI scores for prioritization."""
        print("   💰 Calculating ROI scores for prioritization...")
        
        # ROI factors
        severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.3}
        effort_weights = {"1-2 weeks": 1.0, "2-4 weeks": 0.8, "3-5 weeks": 0.6, "4-6 weeks": 0.4, "6-8 weeks": 0.2}
        
        # Calculate ROI for code clones
        for clone in self.code_clones:
            # Benefit: lines saved
            benefit = clone.estimated_consolidation_savings
            
            # Cost: complexity of consolidation
            complexity_cost = {"low": 1.0, "medium": 2.0, "high": 4.0}[clone.consolidation_complexity]
            
            roi = benefit / complexity_cost if complexity_cost > 0 else 0
            clone.roi_score = roi
        
        # ROI scores are already calculated for architectural debts during analysis
        
        # Sort by ROI for prioritization
        self.code_clones.sort(key=lambda x: x.roi_score, reverse=True)
        self.architectural_debts.sort(key=lambda x: x.roi_score, reverse=True)
    
    async def _generate_enhanced_report(self, analysis_time: float) -> Dict[str, Any]:
        """Generate comprehensive enhanced report."""
        print("   📊 Generating enhanced technical debt report...")
        
        total_files = len(self.file_metrics)
        total_lines = sum(m.get("total_lines", 0) for m in self.file_metrics.values())
        parseable_files = sum(1 for m in self.file_metrics.values() if m.get("parseable", False))
        
        # Calculate potential savings
        clone_savings = sum(clone.estimated_consolidation_savings for clone in self.code_clones)
        
        # Categorize issues by severity
        critical_debts = [d for d in self.architectural_debts if d.severity == "critical"]
        high_debts = [d for d in self.architectural_debts if d.severity == "high"] 
        medium_debts = [d for d in self.architectural_debts if d.severity == "medium"]
        low_debts = [d for d in self.architectural_debts if d.severity == "low"]
        
        report = {
            "analysis_metadata": {
                "timestamp": self.analysis_timestamp.isoformat(),
                "analysis_time_seconds": round(analysis_time, 2),
                "analyzer_version": "Enhanced v2.0 (Post-Epic 1 Phase 3)",
                "project_root": str(self.project_root)
            },
            "codebase_metrics": {
                "total_files_discovered": total_files,
                "parseable_files": parseable_files,
                "unparseable_files": total_files - parseable_files,
                "total_lines_of_code": total_lines,
                "average_file_size": round(total_lines / total_files if total_files > 0 else 0, 1),
                "dependency_graph_nodes": len(self.dependency_graph.nodes),
                "dependency_graph_edges": len(self.dependency_graph.edges)
            },
            "code_clone_analysis": {
                "total_clone_clusters": len(self.code_clones),
                "clone_types": {
                    "exact": len([c for c in self.code_clones if c.clone_type == "exact"]),
                    "structural": len([c for c in self.code_clones if c.clone_type == "structural"]),
                    "functional": len([c for c in self.code_clones if c.clone_type == "functional"])
                },
                "estimated_consolidation_savings": clone_savings,
                "top_consolidation_opportunities": [
                    {
                        "type": clone.clone_type,
                        "files": clone.files,
                        "savings": clone.estimated_consolidation_savings,
                        "complexity": clone.consolidation_complexity,
                        "roi_score": round(clone.roi_score, 2)
                    }
                    for clone in self.code_clones[:5]  # Top 5
                ]
            },
            "architectural_debt": {
                "total_debt_issues": len(self.architectural_debts),
                "by_severity": {
                    "critical": len(critical_debts),
                    "high": len(high_debts),
                    "medium": len(medium_debts),
                    "low": len(low_debts)
                },
                "by_category": dict(Counter(debt.debt_category for debt in self.architectural_debts)),
                "high_priority_issues": [
                    {
                        "category": debt.debt_category,
                        "pattern": debt.pattern_name,
                        "severity": debt.severity,
                        "components_affected": len(debt.affected_components),
                        "business_impact": debt.business_impact,
                        "remediation": debt.remediation_strategy,
                        "effort": debt.effort_estimate,
                        "roi_score": round(debt.roi_score, 2)
                    }
                    for debt in (critical_debts + high_debts)
                ]
            },
            "dependency_analysis": {
                "total_dependency_issues": len(self.dependency_issues),
                "circular_dependencies": len([d for d in self.dependency_issues if d.issue_type == "circular"]),
                "excessive_coupling": len([d for d in self.dependency_issues if d.issue_type == "excessive_coupling"]),
                "unused_components": len([d for d in self.dependency_issues if d.issue_type == "unused"]),
                "critical_dependency_issues": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity,
                        "components": issue.components,
                        "impact": issue.impact_analysis
                    }
                    for issue in self.dependency_issues if issue.severity in ["critical", "high"]
                ]
            },
            "prioritized_recommendations": {
                "immediate_action": [],
                "short_term": [],
                "medium_term": [],
                "long_term": []
            }
        }
        
        # Generate prioritized recommendations
        await self._generate_prioritized_recommendations(report)
        
        return report
    
    async def _generate_prioritized_recommendations(self, report: Dict[str, Any]):
        """Generate prioritized recommendations based on ROI analysis."""
        recommendations = report["prioritized_recommendations"]
        
        # Immediate action (High ROI, low effort)
        high_roi_clones = [c for c in self.code_clones if c.roi_score > 100 and c.consolidation_complexity == "low"]
        for clone in high_roi_clones[:3]:
            recommendations["immediate_action"].append({
                "type": "Code Clone Elimination",
                "description": f"{clone.clone_type.title()} clone across {len(clone.files)} files",
                "files": clone.files,
                "effort": "1-2 weeks",
                "savings": f"{clone.estimated_consolidation_savings} LOC",
                "roi_score": round(clone.roi_score, 2)
            })
        
        # Short-term (High-impact architectural debt)
        critical_arch_debt = [d for d in self.architectural_debts if d.severity in ["critical", "high"] and d.roi_score > 0.7]
        for debt in critical_arch_debt[:3]:
            recommendations["short_term"].append({
                "type": "Architectural Debt Remediation",
                "description": debt.pattern_name,
                "category": debt.debt_category,
                "components": len(debt.affected_components),
                "effort": debt.effort_estimate,
                "strategy": debt.remediation_strategy,
                "roi_score": round(debt.roi_score, 2)
            })
        
        # Medium-term (Structural improvements)
        structural_clones = [c for c in self.code_clones if c.clone_type == "structural" and c.roi_score > 50]
        for clone in structural_clones[:3]:
            recommendations["medium_term"].append({
                "type": "Structural Consolidation",
                "description": f"Structural similarity across {len(clone.files)} files",
                "files": clone.files,
                "effort": "3-4 weeks",
                "approach": clone.recommended_action,
                "roi_score": round(clone.roi_score, 2)
            })
        
        # Long-term (Dependency and infrastructure improvements)
        for issue in self.dependency_issues:
            if issue.severity in ["high", "medium"]:
                recommendations["long_term"].append({
                    "type": "Dependency Improvement",
                    "description": f"{issue.issue_type.replace('_', ' ').title()} issue",
                    "components": issue.components,
                    "effort": "2-6 weeks",
                    "resolution": issue.resolution_steps[0] if issue.resolution_steps else "Review and refactor"
                })


async def main():
    """Main execution function."""
    print("🔬 Enhanced Technical Debt Analyzer - LeanVibe Agent Hive 2.0")
    print("Post-Epic 1 Phase 3: Advanced Debt Detection & Prioritization")
    print("=" * 70)
    
    analyzer = EnhancedTechnicalDebtAnalyzer()
    
    try:
        report = await analyzer.analyze_post_epic1_debt()
        
        # Save detailed report
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = f"enhanced_debt_analysis_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print executive summary
        print(f"\n📊 ENHANCED ANALYSIS SUMMARY")
        print(f"=" * 50)
        print(f"Files Analyzed: {report['codebase_metrics']['parseable_files']}/{report['codebase_metrics']['total_files_discovered']}")
        print(f"Total LOC: {report['codebase_metrics']['total_lines_of_code']:,}")
        print(f"Analysis Time: {report['analysis_metadata']['analysis_time_seconds']}s")
        
        print(f"\n🎯 DEBT DETECTION RESULTS")
        print(f"Code Clone Clusters: {report['code_clone_analysis']['total_clone_clusters']}")
        print(f"Architectural Debt Issues: {report['architectural_debt']['total_debt_issues']}")
        print(f"Dependency Issues: {report['dependency_analysis']['total_dependency_issues']}")
        
        print(f"\n⚡ SEVERITY BREAKDOWN")
        arch_debt = report['architectural_debt']['by_severity']
        for severity, count in arch_debt.items():
            if count > 0:
                print(f"   {severity.title()}: {count}")
        
        print(f"\n💰 CONSOLIDATION OPPORTUNITIES")
        print(f"Estimated LOC Savings: {report['code_clone_analysis']['estimated_consolidation_savings']:,}")
        
        print(f"\n🚀 TOP RECOMMENDATIONS")
        immediate = report['prioritized_recommendations']['immediate_action']
        for i, rec in enumerate(immediate[:3], 1):
            print(f"   {i}. {rec['description']} (ROI: {rec.get('roi_score', 'N/A')})")
        
        print(f"\n📄 Detailed report saved: {report_file}")
        print(f"✅ Enhanced technical debt analysis complete!")
        
        return report
        
    except Exception as e:
        print(f"❌ Enhanced analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())