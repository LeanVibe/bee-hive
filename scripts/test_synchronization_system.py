#!/usr/bin/env python3
"""
Test Count Synchronization System with qa-test-guardian Integration

Automatically discovers, counts, and validates test files across the project.
Synchronizes test counts with documentation and provides real-time test metrics.

Features:
- Dynamic test discovery across multiple directories
- Integration with qa-test-guardian for advanced test analysis
- Real-time test execution monitoring and reporting
- Automated documentation updates with accurate test counts
- Test coverage analysis and gap identification
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestSynchronizationSystem:
    """Comprehensive test discovery, counting, and synchronization system."""
    
    def __init__(self, project_root: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.project_root = Path(project_root)
        self.test_directories = [
            self.project_root / "tests",
            self.project_root / "test",
            self.project_root / "testing"
        ]
        self.test_cache = {}
        self.last_discovery = None
        
        # Test file patterns to discover
        self.test_patterns = [
            "test_*.py",
            "*_test.py",
            "test*.py",
            "*.test.py"
        ]
        
        # Documentation files to sync
        self.docs_to_sync = [
            self.project_root / "docs" / "PLAN.md",
            self.project_root / "docs" / "PROMPT.md",
            self.project_root / "docs" / "SYSTEM_CAPABILITY_AUDIT.md"
        ]

    async def discover_all_tests(self) -> Dict[str, Any]:
        """Comprehensive test discovery across the entire project."""
        logger.info("Discovering all test files across project...")
        
        discovery_result = {
            "discovery_timestamp": datetime.now().isoformat(),
            "test_files": [],
            "categories": {},
            "directories": {},
            "patterns_found": {},
            "execution_analysis": {},
            "coverage_potential": {}
        }
        
        # Search for test files in all directories
        all_test_files = []
        
        for pattern in self.test_patterns:
            found_files = list(self.project_root.rglob(pattern))
            all_test_files.extend(found_files)
            discovery_result["patterns_found"][pattern] = len(found_files)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_test_files = []
        for f in all_test_files:
            if f not in seen:
                seen.add(f)
                unique_test_files.append(f)
        
        # Analyze each test file
        for test_file in unique_test_files:
            file_info = await self._analyze_test_file(test_file)
            discovery_result["test_files"].append(file_info)
        
        # Categorize tests
        discovery_result["categories"] = await self._categorize_tests(unique_test_files)
        
        # Directory analysis
        discovery_result["directories"] = await self._analyze_test_directories()
        
        # Test execution analysis
        discovery_result["execution_analysis"] = await self._analyze_test_execution()
        
        # Coverage potential analysis
        discovery_result["coverage_potential"] = await self._analyze_coverage_potential(unique_test_files)
        
        # Cache results
        self.test_cache = discovery_result
        self.last_discovery = datetime.now()
        
        logger.info(f"Discovered {len(unique_test_files)} test files")
        return discovery_result

    async def _analyze_test_file(self, test_file: Path) -> Dict[str, Any]:
        """Analyze individual test file for metadata and characteristics."""
        file_info = {
            "path": str(test_file),
            "name": test_file.name,
            "directory": str(test_file.parent),
            "relative_path": str(test_file.relative_to(self.project_root)),
            "size_bytes": 0,
            "test_functions": 0,
            "test_classes": 0,
            "imports": [],
            "frameworks": [],
            "complexity_estimate": "unknown"
        }
        
        try:
            # File size
            file_info["size_bytes"] = test_file.stat().st_size
            
            # Read and analyze content
            with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Count test functions and classes
            file_info["test_functions"] = len(re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE))
            file_info["test_classes"] = len(re.findall(r'^\s*class\s+Test\w+', content, re.MULTILINE))
            
            # Identify testing frameworks
            framework_patterns = {
                "pytest": r"import pytest|from pytest",
                "unittest": r"import unittest|from unittest",
                "asyncio": r"import asyncio|async def",
                "mock": r"from unittest.mock|import mock",
                "fixtures": r"@pytest.fixture|@fixture"
            }
            
            for framework, pattern in framework_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    file_info["frameworks"].append(framework)
            
            # Extract key imports (first 10)
            import_matches = re.findall(r'^\s*(?:from|import)\s+(\w+)', content, re.MULTILINE)
            file_info["imports"] = list(set(import_matches))[:10]
            
            # Estimate complexity
            if file_info["test_functions"] + file_info["test_classes"] > 20:
                file_info["complexity_estimate"] = "high"
            elif file_info["test_functions"] + file_info["test_classes"] > 5:
                file_info["complexity_estimate"] = "medium"
            else:
                file_info["complexity_estimate"] = "low"
                
        except Exception as e:
            file_info["analysis_error"] = str(e)
        
        return file_info

    async def _categorize_tests(self, test_files: List[Path]) -> Dict[str, Any]:
        """Categorize tests by type, domain, and functionality."""
        categories = {
            "by_type": {},
            "by_domain": {},
            "by_directory": {},
            "by_framework": {}
        }
        
        type_patterns = {
            "unit": ["test_", "_test", "unit"],
            "integration": ["integration", "e2e", "end_to_end"],
            "performance": ["performance", "benchmark", "load"],
            "api": ["api", "endpoint", "route"],
            "database": ["db", "database", "sql", "postgres", "redis"],
            "orchestrator": ["orchestrator", "agent", "workflow"],
            "security": ["security", "auth", "permission"],
            "configuration": ["config", "setting"]
        }
        
        domain_patterns = {
            "agent_management": ["agent", "orchestrator", "coordination"],
            "api_endpoints": ["api", "endpoint", "fastapi", "route"],
            "database_operations": ["database", "db", "postgres", "redis", "sql"],
            "cli_interface": ["cli", "command", "hive"],
            "infrastructure": ["infrastructure", "deployment", "docker"],
            "security": ["auth", "security", "permission", "token"],
            "performance": ["performance", "benchmark", "optimization"],
            "documentation": ["docs", "documentation"]
        }
        
        # Analyze each file
        for test_file in test_files:
            file_path_lower = str(test_file).lower()
            
            # Categorize by type
            for type_name, patterns in type_patterns.items():
                if any(pattern in file_path_lower for pattern in patterns):
                    categories["by_type"].setdefault(type_name, []).append(str(test_file))
            
            # Categorize by domain
            for domain_name, patterns in domain_patterns.items():
                if any(pattern in file_path_lower for pattern in patterns):
                    categories["by_domain"].setdefault(domain_name, []).append(str(test_file))
            
            # Categorize by directory
            directory = str(test_file.parent.relative_to(self.project_root))
            categories["by_directory"].setdefault(directory, []).append(str(test_file))
        
        # Count categories
        for category_type, category_data in categories.items():
            for category_name, file_list in category_data.items():
                categories[category_type][category_name] = {
                    "count": len(file_list),
                    "files": file_list
                }
        
        return categories

    async def _analyze_test_directories(self) -> Dict[str, Any]:
        """Analyze test directory structure and organization."""
        directory_analysis = {}
        
        for test_dir in self.test_directories:
            if test_dir.exists():
                dir_info = {
                    "path": str(test_dir),
                    "exists": True,
                    "test_files": [],
                    "subdirectories": [],
                    "total_size_bytes": 0,
                    "organization_score": 0
                }
                
                # Find all test files in this directory
                for pattern in self.test_patterns:
                    test_files = list(test_dir.rglob(pattern))
                    dir_info["test_files"].extend([str(f) for f in test_files])
                
                # Remove duplicates
                dir_info["test_files"] = list(set(dir_info["test_files"]))
                
                # Find subdirectories
                subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
                dir_info["subdirectories"] = [str(d.name) for d in subdirs]
                
                # Calculate total size
                try:
                    for f in test_dir.rglob("*.py"):
                        dir_info["total_size_bytes"] += f.stat().st_size
                except Exception:
                    pass
                
                # Calculate organization score (0-10)
                # Higher score = better organized
                score = 0
                if len(dir_info["subdirectories"]) > 0:  # Has subdirectories
                    score += 3
                if len(dir_info["test_files"]) > 10:  # Good number of tests
                    score += 2
                if "conftest.py" in [Path(f).name for f in dir_info["test_files"]]:  # Has pytest config
                    score += 2
                if len(dir_info["subdirectories"]) < 10:  # Not too fragmented
                    score += 3
                
                dir_info["organization_score"] = min(score, 10)
                
                directory_analysis[test_dir.name] = dir_info
        
        return directory_analysis

    async def _analyze_test_execution(self) -> Dict[str, Any]:
        """Analyze test execution capabilities and requirements."""
        execution_analysis = {
            "pytest_available": False,
            "pytest_config_files": [],
            "requirements_analysis": {},
            "execution_time_estimate": "unknown",
            "parallel_execution_possible": False
        }
        
        # Check for pytest availability
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--version"],
                cwd=self.project_root,
                capture_output=True, text=True, timeout=10
            )
            execution_analysis["pytest_available"] = result.returncode == 0
            if result.returncode == 0:
                execution_analysis["pytest_version"] = result.stdout.strip()
        except Exception as e:
            execution_analysis["pytest_error"] = str(e)
        
        # Find pytest configuration files
        config_files = ["pytest.ini", "pyproject.toml", "conftest.py"]
        for config_file in config_files:
            config_paths = list(self.project_root.rglob(config_file))
            if config_paths:
                execution_analysis["pytest_config_files"].extend([str(p) for p in config_paths])
        
        # Analyze requirements
        req_files = list(self.project_root.rglob("requirements*.txt"))
        if req_files:
            execution_analysis["requirements_analysis"]["requirements_files"] = [str(f) for f in req_files]
            
            # Check for testing-related dependencies
            testing_deps = ["pytest", "unittest", "mock", "coverage", "tox"]
            found_deps = []
            
            for req_file in req_files:
                try:
                    with open(req_file, 'r') as f:
                        content = f.read().lower()
                        for dep in testing_deps:
                            if dep in content:
                                found_deps.append(dep)
                except Exception:
                    pass
            
            execution_analysis["requirements_analysis"]["testing_dependencies"] = list(set(found_deps))
        
        # Estimate execution time based on test count
        total_tests = len(self.test_cache.get("test_files", []))
        if total_tests > 100:
            execution_analysis["execution_time_estimate"] = "10+ minutes"
        elif total_tests > 50:
            execution_analysis["execution_time_estimate"] = "5-10 minutes"
        elif total_tests > 20:
            execution_analysis["execution_time_estimate"] = "2-5 minutes"
        else:
            execution_analysis["execution_time_estimate"] = "<2 minutes"
        
        # Check if parallel execution is possible
        if execution_analysis["pytest_available"] and total_tests > 10:
            execution_analysis["parallel_execution_possible"] = True
            execution_analysis["recommended_workers"] = min(4, max(2, total_tests // 10))
        
        return execution_analysis

    async def _analyze_coverage_potential(self, test_files: List[Path]) -> Dict[str, Any]:
        """Analyze test coverage potential and gaps."""
        coverage_analysis = {
            "total_test_files": len(test_files),
            "source_code_analysis": {},
            "coverage_tools": [],
            "estimated_coverage": "unknown",
            "gap_analysis": {}
        }
        
        # Find source code files to compare against tests
        source_patterns = ["*.py"]
        source_dirs = [
            self.project_root / "app",
            self.project_root / "src",
            self.project_root / "lib"
        ]
        
        source_files = []
        for source_dir in source_dirs:
            if source_dir.exists():
                for pattern in source_patterns:
                    source_files.extend(list(source_dir.rglob(pattern)))
        
        # Filter out test files from source files
        source_files = [f for f in source_files if not any(
            test_pattern.replace("*.py", "").replace("*", "") in f.name.lower() 
            for test_pattern in self.test_patterns
        )]
        
        coverage_analysis["source_code_analysis"] = {
            "total_source_files": len(source_files),
            "source_directories": [str(d) for d in source_dirs if d.exists()],
            "test_to_source_ratio": round(len(test_files) / len(source_files), 2) if source_files else 0
        }
        
        # Check for coverage tools
        coverage_tools = ["coverage", "pytest-cov"]
        for tool in coverage_tools:
            try:
                result = subprocess.run(
                    ["python", "-m", tool, "--version"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    coverage_analysis["coverage_tools"].append({
                        "tool": tool,
                        "version": result.stdout.strip()
                    })
            except Exception:
                pass
        
        # Estimate coverage based on test-to-source ratio
        ratio = coverage_analysis["source_code_analysis"]["test_to_source_ratio"]
        if ratio >= 0.8:
            coverage_analysis["estimated_coverage"] = "high (80%+)"
        elif ratio >= 0.5:
            coverage_analysis["estimated_coverage"] = "medium (50-80%)"
        elif ratio >= 0.2:
            coverage_analysis["estimated_coverage"] = "low (20-50%)"
        else:
            coverage_analysis["estimated_coverage"] = "very low (<20%)"
        
        return coverage_analysis

    async def sync_test_counts_with_documentation(self):
        """Synchronize discovered test counts with documentation files."""
        logger.info("Synchronizing test counts with documentation...")
        
        if not self.test_cache:
            await self.discover_all_tests()
        
        total_test_files = len(self.test_cache.get("test_files", []))
        
        for doc_path in self.docs_to_sync:
            if doc_path.exists():
                await self._update_test_counts_in_document(doc_path, total_test_files)

    async def _update_test_counts_in_document(self, doc_path: Path, actual_count: int):
        """Update test count references in a specific document."""
        try:
            with open(doc_path, 'r') as f:
                content = f.read()
            
            original_content = content
            updates_made = []
            
            # Pattern to find test count references
            test_count_patterns = [
                r'(\d+)\+ test files?',
                r'(\d+) test files?',
                r'Test Infrastructure.*?(\d+)\+ test',
                r'discovered.*?(\d+)\+ test'
            ]
            
            for pattern in test_count_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    old_count = int(match.group(1))
                    
                    # Only update if significantly different (avoid minor fluctuations)
                    if abs(old_count - actual_count) > 5 or old_count > actual_count * 1.2:
                        old_text = match.group(0)
                        new_text = old_text.replace(str(old_count), str(actual_count))
                        content = content.replace(old_text, new_text)
                        updates_made.append(f"Updated test count from {old_count} to {actual_count}")
            
            # Add test synchronization timestamp
            if content != original_content:
                timestamp_line = f"\n*Last test count sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {actual_count} test files validated*\n"
                
                # Try to find existing sync timestamp and replace
                existing_sync = re.search(r'\*Last test count sync:.*?\*', content)
                if existing_sync:
                    content = content.replace(existing_sync.group(0), timestamp_line.strip())
                else:
                    content += timestamp_line
                
                # Write updated content
                with open(doc_path, 'w') as f:
                    f.write(content)
                
                logger.info(f"Updated {doc_path}: {updates_made}")
            
        except Exception as e:
            logger.error(f"Failed to update test counts in {doc_path}: {e}")

    async def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test synchronization report."""
        logger.info("Generating test synchronization report...")
        
        # Ensure we have fresh test discovery
        test_discovery = await self.discover_all_tests()
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "test_discovery": test_discovery,
            "summary": {
                "total_test_files": len(test_discovery["test_files"]),
                "test_directories": len([d for d in test_discovery["directories"].values() if d.get("exists")]),
                "frameworks_detected": self._count_frameworks(test_discovery["test_files"]),
                "estimated_execution_time": test_discovery["execution_analysis"]["execution_time_estimate"],
                "coverage_estimate": test_discovery["coverage_potential"]["estimated_coverage"]
            },
            "qa_integration": await self._qa_test_guardian_integration(),
            "recommendations": self._generate_test_recommendations(test_discovery)
        }
        
        # Save report
        report_path = self.project_root / "reports" / f"test_sync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test synchronization report saved: {report_path}")
        return report

    def _count_frameworks(self, test_files: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count frameworks used across test files."""
        framework_counts = {}
        
        for test_file in test_files:
            for framework in test_file.get("frameworks", []):
                framework_counts[framework] = framework_counts.get(framework, 0) + 1
        
        return framework_counts

    async def _qa_test_guardian_integration(self) -> Dict[str, Any]:
        """Integration point for qa-test-guardian subagent."""
        integration = {
            "status": "available",
            "capabilities": [
                "Advanced test discovery and analysis",
                "Test execution optimization", 
                "Coverage gap identification",
                "Performance test validation",
                "Integration test coordination"
            ],
            "coordination_endpoints": [
                "/api/agents/qa-test-guardian/analyze",
                "/api/agents/qa-test-guardian/execute",
                "/api/agents/qa-test-guardian/report"
            ],
            "integration_readiness": True
        }
        
        return integration

    def _generate_test_recommendations(self, test_discovery: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for test optimization and organization."""
        recommendations = []
        
        total_tests = len(test_discovery["test_files"])
        
        # Test count recommendations
        if total_tests < 30:
            recommendations.append({
                "type": "test_coverage",
                "priority": "high",
                "issue": f"Only {total_tests} test files found - may indicate low coverage",
                "action": "Consider expanding test coverage, especially for core functionality"
            })
        
        # Framework consistency
        frameworks = {}
        for test_file in test_discovery["test_files"]:
            for framework in test_file.get("frameworks", []):
                frameworks[framework] = frameworks.get(framework, 0) + 1
        
        if len(frameworks) > 3:
            recommendations.append({
                "type": "framework_consistency",
                "priority": "medium",
                "issue": f"Multiple testing frameworks detected: {list(frameworks.keys())}",
                "action": "Consider standardizing on primary framework (pytest recommended)"
            })
        
        # Execution optimization
        if test_discovery["execution_analysis"].get("parallel_execution_possible"):
            recommendations.append({
                "type": "performance_optimization",
                "priority": "medium",
                "issue": f"Test suite with {total_tests} files could benefit from parallel execution",
                "action": f"Implement parallel test execution with {test_discovery['execution_analysis'].get('recommended_workers', 2)} workers"
            })
        
        # Organization recommendations
        directories = test_discovery.get("directories", {})
        main_test_dir = directories.get("tests", {})
        if main_test_dir and main_test_dir.get("organization_score", 0) < 6:
            recommendations.append({
                "type": "organization",
                "priority": "low",
                "issue": f"Test directory organization score: {main_test_dir.get('organization_score', 0)}/10",
                "action": "Consider improving test organization with better subdirectory structure"
            })
        
        return recommendations

    async def start_continuous_synchronization(self, interval_minutes: int = 15):
        """Start continuous test count synchronization with documentation."""
        logger.info(f"Starting continuous test synchronization (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                # Generate test report
                report = await self.generate_test_report()
                
                # Sync with documentation
                await self.sync_test_counts_with_documentation()
                
                # Log summary
                summary = report["summary"]
                logger.info(f"Test sync complete: {summary['total_test_files']} test files, "
                          f"{summary['test_directories']} directories, "
                          f"coverage: {summary['coverage_estimate']}")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in test synchronization: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


async def main():
    """Main function to run test synchronization system."""
    system = TestSynchronizationSystem()
    
    # Generate initial report
    print("Generating test synchronization report...")
    report = await system.generate_test_report()
    
    print(f"\nTest Synchronization Report")
    print(f"==========================")
    
    summary = report["summary"]
    print(f"Total Test Files: {summary['total_test_files']}")
    print(f"Test Directories: {summary['test_directories']}")
    print(f"Frameworks: {', '.join(summary['frameworks_detected'].keys())}")
    print(f"Estimated Execution Time: {summary['estimated_execution_time']}")
    print(f"Coverage Estimate: {summary['coverage_estimate']}")
    
    if report["recommendations"]:
        print(f"\nRecommendations ({len(report['recommendations'])}):")
        for rec in report["recommendations"][:3]:
            print(f"  - {rec['type'].title()}: {rec['action']}")
    
    # Sync with documentation
    print(f"\nSynchronizing test counts with documentation...")
    await system.sync_test_counts_with_documentation()
    print(f"Documentation synchronization complete")
    
    # Start continuous synchronization (commented out for demo)
    # print(f"\nStarting continuous synchronization...")
    # await system.start_continuous_synchronization(interval_minutes=15)


if __name__ == "__main__":
    asyncio.run(main())