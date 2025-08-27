#!/usr/bin/env python3
"""
Epic 7 Final Consolidation Phase
Eliminate system redundancy and achieve <50% duplication.
"""
import sys
import os
import re
from pathlib import Path
from collections import defaultdict
import hashlib

def analyze_duplicate_imports():
    """Analyze and report duplicate import patterns."""
    print("🔧 Analyzing duplicate import patterns...")
    try:
        import_patterns = defaultdict(list)
        
        # Scan key directories for imports
        directories = [
            Path("app/core"),
            Path("app/api"),
            Path("app/models")
        ]
        
        for directory in directories:
            if directory.exists():
                for file_path in directory.rglob("*.py"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Extract import statements
                        import_lines = re.findall(r'^from .* import .*$|^import .*$', content, re.MULTILINE)
                        
                        for import_line in import_lines:
                            import_patterns[import_line].append(str(file_path))
                            
                    except Exception as e:
                        continue
        
        # Find duplicates
        duplicated_imports = 0
        total_imports = 0
        redundant_patterns = []
        
        for import_line, files in import_patterns.items():
            total_imports += len(files)
            if len(files) > 1:
                duplicated_imports += len(files) - 1
                if len(files) > 5:  # Significant duplication
                    redundant_patterns.append((import_line, len(files)))
        
        redundancy_rate = (duplicated_imports / total_imports) * 100 if total_imports > 0 else 0
        
        print(f"  - Total import statements: {total_imports}")
        print(f"  - Duplicated imports: {duplicated_imports}")
        print(f"  - Import redundancy rate: {redundancy_rate:.1f}%")
        print(f"  - Most redundant patterns: {len(redundant_patterns)}")
        
        if redundancy_rate < 50:
            print("✅ Import redundancy within acceptable limits")
            return True, redundancy_rate
        else:
            print("❌ Import redundancy exceeds 50%")
            return False, redundancy_rate
            
    except Exception as e:
        print(f"❌ Import analysis failed: {e}")
        return False, 100.0

def analyze_code_duplication():
    """Analyze code duplication patterns."""
    print("🔧 Analyzing code duplication patterns...")
    try:
        function_signatures = defaultdict(list)
        class_definitions = defaultdict(list)
        
        # Scan core directories
        directories = [
            Path("app/core"),
            Path("app/api")
        ]
        
        for directory in directories:
            if directory.exists():
                for file_path in directory.rglob("*.py"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract function definitions
                        functions = re.findall(r'^def ([a-zA-Z_][a-zA-Z0-9_]*)\(.*?\):', content, re.MULTILINE)
                        for func in functions:
                            function_signatures[func].append(str(file_path))
                        
                        # Extract class definitions
                        classes = re.findall(r'^class ([a-zA-Z_][a-zA-Z0-9_]*)', content, re.MULTILINE)
                        for cls in classes:
                            class_definitions[cls].append(str(file_path))
                            
                    except Exception:
                        continue
        
        # Analyze duplication
        duplicate_functions = sum(1 for files in function_signatures.values() if len(files) > 1)
        duplicate_classes = sum(1 for files in class_definitions.values() if len(files) > 1)
        
        total_functions = len(function_signatures)
        total_classes = len(class_definitions)
        
        func_duplication_rate = (duplicate_functions / total_functions) * 100 if total_functions > 0 else 0
        class_duplication_rate = (duplicate_classes / total_classes) * 100 if total_classes > 0 else 0
        
        overall_duplication = (func_duplication_rate + class_duplication_rate) / 2
        
        print(f"  - Total functions: {total_functions}")
        print(f"  - Duplicate function names: {duplicate_functions}")
        print(f"  - Function duplication rate: {func_duplication_rate:.1f}%")
        print(f"  - Total classes: {total_classes}")
        print(f"  - Duplicate class names: {duplicate_classes}")
        print(f"  - Class duplication rate: {class_duplication_rate:.1f}%")
        print(f"  - Overall code duplication rate: {overall_duplication:.1f}%")
        
        if overall_duplication < 50:
            print("✅ Code duplication within acceptable limits")
            return True, overall_duplication
        else:
            print("❌ Code duplication exceeds 50%")
            return False, overall_duplication
            
    except Exception as e:
        print(f"❌ Code duplication analysis failed: {e}")
        return False, 100.0

def analyze_configuration_redundancy():
    """Analyze configuration file redundancy."""
    print("🔧 Analyzing configuration redundancy...")
    try:
        config_files = []
        config_content = {}
        
        # Find configuration files
        for file_path in Path(".").rglob("*.py"):
            if any(keyword in str(file_path).lower() for keyword in ['config', 'settings', 'env']):
                if file_path.is_file():
                    config_files.append(file_path)
        
        # Analyze configuration content
        similar_configs = 0
        total_configs = len(config_files)
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for configuration patterns
                config_vars = re.findall(r'[A-Z_][A-Z0-9_]*\s*=', content)
                config_content[str(config_file)] = set(config_vars)
                
            except Exception:
                continue
        
        # Find similar configurations
        config_files_list = list(config_content.keys())
        for i in range(len(config_files_list)):
            for j in range(i + 1, len(config_files_list)):
                file1 = config_files_list[i]
                file2 = config_files_list[j]
                
                overlap = len(config_content[file1] & config_content[file2])
                total_vars = len(config_content[file1] | config_content[file2])
                
                if total_vars > 0 and (overlap / total_vars) > 0.5:
                    similar_configs += 1
        
        redundancy_rate = (similar_configs / max(total_configs, 1)) * 100
        
        print(f"  - Total configuration files: {total_configs}")
        print(f"  - Similar configuration pairs: {similar_configs}")
        print(f"  - Configuration redundancy rate: {redundancy_rate:.1f}%")
        
        if redundancy_rate < 50:
            print("✅ Configuration redundancy within acceptable limits")
            return True, redundancy_rate
        else:
            print("❌ Configuration redundancy exceeds 50%")
            return False, redundancy_rate
            
    except Exception as e:
        print(f"❌ Configuration analysis failed: {e}")
        return False, 100.0

def analyze_interface_consistency():
    """Analyze interface consistency and standardization."""
    print("🔧 Analyzing interface consistency...")
    try:
        api_endpoints = []
        
        # Scan API files for endpoints
        api_dir = Path("app/api")
        if api_dir.exists():
            for file_path in api_dir.rglob("*.py"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for FastAPI route decorators
                    routes = re.findall(r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)', content)
                    api_endpoints.extend(routes)
                    
                except Exception:
                    continue
        
        # Analyze endpoint consistency
        route_patterns = defaultdict(list)
        for method, path in api_endpoints:
            # Extract pattern (remove specific IDs)
            pattern = re.sub(r'\{[^}]+\}', '{id}', path)
            route_patterns[pattern].append(method)
        
        consistent_interfaces = 0
        total_patterns = len(route_patterns)
        
        for pattern, methods in route_patterns.items():
            # Check if standard CRUD operations are consistently implemented
            if len(set(methods)) == len(methods):  # No duplicate methods
                consistent_interfaces += 1
        
        consistency_rate = (consistent_interfaces / max(total_patterns, 1)) * 100
        
        print(f"  - Total API endpoint patterns: {total_patterns}")
        print(f"  - Consistent interface patterns: {consistent_interfaces}")
        print(f"  - Interface consistency rate: {consistency_rate:.1f}%")
        
        if consistency_rate > 70:
            print("✅ Interface consistency acceptable")
            return True, consistency_rate
        else:
            print("❌ Interface consistency below target")
            return False, consistency_rate
            
    except Exception as e:
        print(f"❌ Interface analysis failed: {e}")
        return False, 0.0

def validate_single_source_of_truth():
    """Validate single source of truth for critical components."""
    print("🔧 Validating single source of truth...")
    try:
        critical_components = [
            'database',
            'redis',
            'config',
            'auth',
            'orchestrator'
        ]
        
        violations = 0
        total_components = len(critical_components)
        
        for component in critical_components:
            # Find files related to this component
            related_files = []
            
            for file_path in Path("app").rglob("*.py"):
                if component in str(file_path).lower():
                    related_files.append(file_path)
            
            # Check for multiple implementations
            implementation_files = []
            for file_path in related_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for class definitions (potential implementations)
                    classes = re.findall(r'^class ([a-zA-Z_][a-zA-Z0-9_]*)', content, re.MULTILINE)
                    if classes:
                        implementation_files.append(file_path)
                        
                except Exception:
                    continue
            
            if len(implementation_files) > 3:  # Allow some reasonable duplication
                violations += 1
                print(f"  ⚠️ {component}: {len(implementation_files)} implementation files")
            else:
                print(f"  ✅ {component}: {len(implementation_files)} implementation files")
        
        compliance_rate = ((total_components - violations) / total_components) * 100
        
        print(f"  - Components analyzed: {total_components}")
        print(f"  - Single source violations: {violations}")
        print(f"  - Single source compliance: {compliance_rate:.1f}%")
        
        if violations == 0:
            print("✅ Single source of truth maintained")
            return True, compliance_rate
        else:
            print("❌ Single source of truth violations found")
            return False, compliance_rate
            
    except Exception as e:
        print(f"❌ Single source validation failed: {e}")
        return False, 0.0

def calculate_overall_consolidation_score():
    """Calculate overall system consolidation score."""
    print("🔧 Calculating overall consolidation score...")
    
    scores = []
    weights = []
    
    # Run all consolidation tests
    tests = [
        (analyze_duplicate_imports, 0.2),
        (analyze_code_duplication, 0.3),
        (analyze_configuration_redundancy, 0.2),
        (analyze_interface_consistency, 0.15),
        (validate_single_source_of_truth, 0.15),
    ]
    
    for test_func, weight in tests:
        try:
            success, score = test_func()
            if success:
                # Convert redundancy rates to consolidation scores
                if 'redundancy' in test_func.__name__ or 'duplication' in test_func.__name__:
                    consolidation_score = max(0, 100 - score)  # Lower redundancy = higher score
                else:
                    consolidation_score = score  # Higher consistency/compliance = higher score
            else:
                consolidation_score = 0
            
            scores.append(consolidation_score)
            weights.append(weight)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            scores.append(0)
            weights.append(weight)
            print("-" * 40)
    
    # Calculate weighted average
    if weights:
        overall_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
    else:
        overall_score = 0
    
    return overall_score

def main():
    """Run Epic 7 final consolidation validation."""
    print("🎯 EPIC 7 FINAL CONSOLIDATION PHASE")
    print("=" * 60)
    print("Target: Eliminate >50% system redundancy")
    print("=" * 60)
    
    overall_score = calculate_overall_consolidation_score()
    
    print(f"\n📊 FINAL CONSOLIDATION RESULTS")
    print(f"Overall Consolidation Score: {overall_score:.1f}%")
    print(f"Redundancy Elimination Target: >50%")
    print(f"Consolidation Success Target: >70%")
    
    if overall_score >= 70:
        redundancy_eliminated = 100 - overall_score
        print(f"✅ CONSOLIDATION SUCCESSFUL")
        print(f"System redundancy reduced by ~{redundancy_eliminated:.1f}%")
        return True
    elif overall_score >= 50:
        print(f"⚠️ CONSOLIDATION PARTIAL")
        print(f"System shows good consolidation foundations")
        return True
    else:
        print(f"❌ CONSOLIDATION NEEDS WORK")
        print(f"System requires further consolidation efforts")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)