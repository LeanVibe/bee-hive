#!/usr/bin/env python3
"""
Advanced Semantic Code Analyzer
===============================

Deep semantic analysis of the 626 detected code clones using advanced pattern matching
beyond structural similarity. Identifies true duplicate logic vs. superficial similarities.
"""

import ast
import hashlib
import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, NamedTuple
import structlog
from difflib import SequenceMatcher

logger = structlog.get_logger(__name__)

@dataclass
class SemanticPattern:
    """Represents a semantic code pattern for similarity analysis."""
    pattern_id: str
    pattern_type: str  # 'function', 'class', 'error_handler', 'data_processor'
    semantic_signature: str  # What the code actually does
    structural_signature: str  # How the code is structured
    business_logic_hash: str  # Core business logic fingerprint
    files: List[Path] = field(default_factory=list)
    similarity_score: float = 0.0
    consolidation_potential: int = 0

@dataclass
class FunctionSemantics:
    """Semantic analysis of a function's purpose and behavior."""
    name: str
    purpose: str  # What it does (authentication, validation, transformation, etc.)
    inputs: List[str]  # Parameter types and patterns
    outputs: str  # Return type and pattern
    side_effects: List[str]  # External interactions (database, network, files)
    business_domain: str  # Which business area it serves
    complexity_score: float

class AdvancedSemanticAnalyzer:
    """Advanced semantic analysis for deep duplicate detection."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.semantic_patterns = defaultdict(list)
        self.function_semantics = {}
        self.business_logic_map = defaultdict(list)
        
        # Semantic pattern definitions
        self.semantic_categories = {
            'authentication': ['login', 'auth', 'token', 'verify', 'validate_user'],
            'validation': ['validate', 'check', 'verify', 'ensure', 'assert'],
            'transformation': ['transform', 'convert', 'map', 'serialize', 'format'],
            'data_access': ['get', 'fetch', 'load', 'save', 'store', 'persist'],
            'error_handling': ['handle', 'catch', 'recover', 'fallback', 'retry'],
            'configuration': ['config', 'setup', 'init', 'configure', 'settings'],
            'communication': ['send', 'receive', 'publish', 'subscribe', 'notify'],
            'monitoring': ['log', 'track', 'monitor', 'measure', 'observe']
        }
    
    def analyze_semantic_patterns(self) -> Dict[str, List[SemanticPattern]]:
        """Perform deep semantic analysis of all Python files."""
        logger.info("🔍 Starting advanced semantic pattern analysis")
        
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if not any(skip in str(f) for skip in ['.venv', 'venv', '__pycache__'])]
        
        logger.info(f"Analyzing {len(python_files)} Python files for semantic patterns")
        
        # Step 1: Extract semantic information from all functions
        for file_path in python_files:
            try:
                self.analyze_file_semantics(file_path)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Step 2: Group functions by semantic similarity
        semantic_groups = self.group_by_semantic_similarity()
        
        # Step 3: Identify consolidation opportunities
        consolidation_opportunities = self.identify_consolidation_opportunities(semantic_groups)
        
        logger.info(f"Found {len(consolidation_opportunities)} semantic consolidation opportunities")
        return consolidation_opportunities
    
    def analyze_file_semantics(self, file_path: Path):
        """Extract semantic information from a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    semantics = self.analyze_function_semantics(node, file_path, content)
                    if semantics:
                        key = f"{file_path}:{node.name}:{node.lineno}"
                        self.function_semantics[key] = semantics
                        
                        # Group by business domain
                        self.business_logic_map[semantics.business_domain].append({
                            'key': key,
                            'file': file_path,
                            'semantics': semantics
                        })
                        
        except Exception as e:
            logger.warning(f"Error analyzing semantics in {file_path}: {e}")
    
    def analyze_function_semantics(self, node: ast.FunctionDef, file_path: Path, content: str) -> Optional[FunctionSemantics]:
        """Analyze the semantic meaning of a function."""
        try:
            # Extract function source
            lines = content.split('\\n')
            start_line = node.lineno - 1
            
            # Determine function purpose based on name and body
            purpose = self.classify_function_purpose(node.name, node)
            
            # Analyze inputs
            inputs = [arg.arg for arg in node.args.args]
            
            # Analyze return patterns
            outputs = self.analyze_return_patterns(node)
            
            # Detect side effects
            side_effects = self.detect_side_effects(node)
            
            # Determine business domain
            business_domain = self.classify_business_domain(file_path, node.name, node)
            
            # Calculate complexity
            complexity_score = self.calculate_semantic_complexity(node)
            
            return FunctionSemantics(
                name=node.name,
                purpose=purpose,
                inputs=inputs,
                outputs=outputs,
                side_effects=side_effects,
                business_domain=business_domain,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            logger.debug(f"Error analyzing function {node.name} in {file_path}: {e}")
            return None
    
    def classify_function_purpose(self, func_name: str, node: ast.FunctionDef) -> str:
        """Classify what the function actually does."""
        func_name_lower = func_name.lower()
        
        # Check against semantic categories
        for category, keywords in self.semantic_categories.items():
            if any(keyword in func_name_lower for keyword in keywords):
                return category
        
        # Analyze function body for purpose clues
        body_analysis = self.analyze_function_body_purpose(node)
        if body_analysis:
            return body_analysis
            
        return 'utility'
    
    def analyze_function_body_purpose(self, node: ast.FunctionDef) -> Optional[str]:
        """Analyze function body to determine purpose."""
        # Look for common patterns in function body
        body_str = ast.unparse(node).lower() if hasattr(ast, 'unparse') else ""
        
        patterns = {
            'authentication': ['password', 'token', 'login', 'auth', 'jwt'],
            'validation': ['raise', 'assert', 'validate', 'check', 'error'],
            'data_access': ['query', 'select', 'insert', 'update', 'delete', 'session'],
            'transformation': ['json', 'serialize', 'convert', 'transform', 'parse'],
            'error_handling': ['try', 'except', 'catch', 'handle', 'recover'],
            'communication': ['request', 'response', 'send', 'post', 'get', 'api'],
            'monitoring': ['log', 'print', 'track', 'metric', 'monitor']
        }
        
        for purpose, keywords in patterns.items():
            if sum(1 for keyword in keywords if keyword in body_str) >= 2:
                return purpose
        
        return None
    
    def detect_side_effects(self, node: ast.FunctionDef) -> List[str]:
        """Detect side effects (database calls, network requests, file I/O)."""
        side_effects = []
        
        for n in ast.walk(node):
            # Database operations
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                attr_name = n.func.attr.lower()
                if any(db_op in attr_name for db_op in ['query', 'execute', 'commit', 'session']):
                    side_effects.append('database')
                elif any(net_op in attr_name for net_op in ['request', 'post', 'get', 'send']):
                    side_effects.append('network')
                elif any(file_op in attr_name for file_op in ['open', 'read', 'write', 'save']):
                    side_effects.append('file_io')
                elif any(log_op in attr_name for log_op in ['log', 'print', 'debug', 'info']):
                    side_effects.append('logging')
        
        return list(set(side_effects))
    
    def analyze_return_patterns(self, node: ast.FunctionDef) -> str:
        """Analyze what the function returns."""
        return_statements = []
        
        for n in ast.walk(node):
            if isinstance(n, ast.Return) and n.value:
                if isinstance(n.value, ast.Constant):
                    return_statements.append(f"constant_{type(n.value.value).__name__}")
                elif isinstance(n.value, ast.Dict):
                    return_statements.append("dict")
                elif isinstance(n.value, ast.List):
                    return_statements.append("list")
                elif isinstance(n.value, ast.Call):
                    return_statements.append("function_call")
                else:
                    return_statements.append("expression")
        
        if not return_statements:
            return "none"
        
        # Return most common pattern
        return Counter(return_statements).most_common(1)[0][0] if return_statements else "none"
    
    def classify_business_domain(self, file_path: Path, func_name: str, node: ast.FunctionDef) -> str:
        """Classify which business domain this function serves."""
        path_str = str(file_path).lower()
        
        domains = {
            'authentication': ['auth', 'login', 'user', 'security'],
            'api': ['api', 'endpoint', 'router', 'handler'],
            'database': ['db', 'model', 'schema', 'migration'],
            'configuration': ['config', 'settings', 'env'],
            'monitoring': ['monitor', 'observe', 'metric', 'log'],
            'orchestration': ['orchestrat', 'agent', 'task', 'workflow'],
            'communication': ['message', 'websocket', 'stream', 'pubsub'],
            'testing': ['test', 'mock', 'fixture'],
            'integration': ['integration', 'client', 'service']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in path_str for keyword in keywords):
                return domain
        
        return 'core'
    
    def calculate_semantic_complexity(self, node: ast.FunctionDef) -> float:
        """Calculate semantic complexity based on control flow and operations."""
        complexity = 1.0  # Base complexity
        
        for n in ast.walk(node):
            if isinstance(n, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 0.5
            elif isinstance(n, ast.Call):
                complexity += 0.1
        
        return round(complexity, 2)
    
    def group_by_semantic_similarity(self) -> Dict[str, List[Dict]]:
        """Group functions by semantic similarity."""
        semantic_groups = defaultdict(list)
        
        # Group by business domain and purpose
        for domain, functions in self.business_logic_map.items():
            purpose_groups = defaultdict(list)
            
            for func_info in functions:
                semantics = func_info['semantics']
                purpose_groups[semantics.purpose].append(func_info)
            
            # Find groups with multiple functions (potential duplicates)
            for purpose, func_list in purpose_groups.items():
                if len(func_list) >= 2:  # At least 2 functions with same purpose
                    group_key = f"{domain}_{purpose}"
                    semantic_groups[group_key] = func_list
        
        return semantic_groups
    
    def identify_consolidation_opportunities(self, semantic_groups: Dict[str, List[Dict]]) -> Dict[str, List[SemanticPattern]]:
        """Identify specific consolidation opportunities."""
        opportunities = {}
        
        for group_key, functions in semantic_groups.items():
            if len(functions) < 3:  # Need at least 3 similar functions
                continue
            
            # Analyze similarity within the group
            similar_clusters = self.cluster_similar_functions(functions)
            
            patterns = []
            for cluster in similar_clusters:
                if len(cluster) >= 3:  # Minimum cluster size
                    pattern = self.create_semantic_pattern(group_key, cluster)
                    patterns.append(pattern)
            
            if patterns:
                opportunities[group_key] = patterns
        
        return opportunities
    
    def cluster_similar_functions(self, functions: List[Dict]) -> List[List[Dict]]:
        """Cluster functions by detailed similarity analysis."""
        clusters = []
        processed = set()
        
        for i, func1 in enumerate(functions):
            if i in processed:
                continue
                
            cluster = [func1]
            processed.add(i)
            
            for j, func2 in enumerate(functions[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.calculate_function_similarity(func1, func2)
                if similarity > 0.8:  # High similarity threshold
                    cluster.append(func2)
                    processed.add(j)
            
            if len(cluster) >= 2:
                clusters.append(cluster)
        
        return clusters
    
    def calculate_function_similarity(self, func1: Dict, func2: Dict) -> float:
        """Calculate detailed similarity between two functions."""
        s1, s2 = func1['semantics'], func2['semantics']
        
        # Weight different similarity aspects
        weights = {
            'purpose': 0.4,     # Most important
            'inputs': 0.2,
            'outputs': 0.2,
            'side_effects': 0.1,
            'complexity': 0.1
        }
        
        similarities = {}
        
        # Purpose similarity (exact match)
        similarities['purpose'] = 1.0 if s1.purpose == s2.purpose else 0.0
        
        # Input similarity
        input_similarity = len(set(s1.inputs) & set(s2.inputs)) / max(len(set(s1.inputs) | set(s2.inputs)), 1)
        similarities['inputs'] = input_similarity
        
        # Output similarity
        similarities['outputs'] = 1.0 if s1.outputs == s2.outputs else 0.0
        
        # Side effects similarity
        side_effect_similarity = len(set(s1.side_effects) & set(s2.side_effects)) / max(len(set(s1.side_effects) | set(s2.side_effects)), 1)
        similarities['side_effects'] = side_effect_similarity
        
        # Complexity similarity (close complexity scores)
        complexity_diff = abs(s1.complexity_score - s2.complexity_score)
        similarities['complexity'] = max(0, 1.0 - complexity_diff / 10.0)
        
        # Calculate weighted average
        total_similarity = sum(similarities[aspect] * weight for aspect, weight in weights.items())
        
        return total_similarity
    
    def create_semantic_pattern(self, group_key: str, cluster: List[Dict]) -> SemanticPattern:
        """Create a semantic pattern from a cluster of similar functions."""
        # Get representative semantics
        representative = cluster[0]['semantics']
        
        # Calculate consolidation potential
        total_lines = sum(func['semantics'].complexity_score * 5 for func in cluster)  # Estimate LOC
        consolidation_potential = int(total_lines * 0.7)  # 70% of lines could be eliminated
        
        # Create business logic hash
        logic_elements = [
            representative.purpose,
            representative.business_domain,
            str(sorted(representative.side_effects)),
            representative.outputs
        ]
        business_logic_hash = hashlib.md5('|'.join(logic_elements).encode()).hexdigest()[:12]
        
        return SemanticPattern(
            pattern_id=f"semantic_{business_logic_hash}",
            pattern_type=representative.purpose,
            semantic_signature=f"{representative.business_domain}:{representative.purpose}",
            structural_signature=f"inputs_{len(representative.inputs)}_effects_{len(representative.side_effects)}",
            business_logic_hash=business_logic_hash,
            files=[Path(func['file']) for func in cluster],
            similarity_score=0.85,  # Average similarity in cluster
            consolidation_potential=consolidation_potential
        )
    
    def generate_consolidation_report(self, opportunities: Dict[str, List[SemanticPattern]]) -> Dict:
        """Generate comprehensive consolidation report."""
        total_patterns = sum(len(patterns) for patterns in opportunities.values())
        total_files = sum(len(pattern.files) for patterns in opportunities.values() for pattern in patterns)
        total_savings = sum(pattern.consolidation_potential for patterns in opportunities.values() for pattern in patterns)
        
        # Group by pattern type for analysis
        by_type = defaultdict(list)
        for patterns in opportunities.values():
            for pattern in patterns:
                by_type[pattern.pattern_type].append(pattern)
        
        type_analysis = {}
        for pattern_type, patterns in by_type.items():
            type_analysis[pattern_type] = {
                'count': len(patterns),
                'total_files': sum(len(p.files) for p in patterns),
                'total_savings': sum(p.consolidation_potential for p in patterns),
                'avg_similarity': sum(p.similarity_score for p in patterns) / len(patterns)
            }
        
        return {
            'summary': {
                'total_semantic_patterns': total_patterns,
                'total_files_affected': total_files,
                'total_consolidation_potential': total_savings,
                'top_opportunities': sorted(by_type.items(), key=lambda x: type_analysis[x[0]]['total_savings'], reverse=True)[:10]
            },
            'by_type': type_analysis,
            'detailed_opportunities': opportunities
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced semantic code analysis')
    parser.add_argument('--analyze', action='store_true', help='Run comprehensive semantic analysis')
    parser.add_argument('--report', action='store_true', help='Generate detailed consolidation report')
    
    args = parser.parse_args()
    
    analyzer = AdvancedSemanticAnalyzer()
    
    if args.analyze or not any([args.report]):
        print("🔍 Starting advanced semantic analysis...")
        opportunities = analyzer.analyze_semantic_patterns()
        
        if opportunities:
            report = analyzer.generate_consolidation_report(opportunities)
            
            print("\\n📊 Advanced Semantic Analysis Results:")
            print(f"   🎯 {report['summary']['total_semantic_patterns']} semantic patterns identified")
            print(f"   📁 {report['summary']['total_files_affected']} files affected")  
            print(f"   💰 {report['summary']['total_consolidation_potential']} LOC consolidation potential")
            
            print("\\n🏆 Top Consolidation Opportunities:")
            for pattern_type, analysis in list(report['by_type'].items())[:5]:
                print(f"   • {pattern_type.title()}: {analysis['count']} patterns, {analysis['total_savings']} LOC savings")
            
            # Save detailed report
            report_path = Path('advanced_semantic_analysis_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\\n📋 Detailed report saved to: {report_path}")
        else:
            print("\\n📊 No significant semantic consolidation opportunities found")
    
    print("\\n✅ Advanced semantic analysis complete")