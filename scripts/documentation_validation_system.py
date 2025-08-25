#!/usr/bin/env python3
"""
Automated Documentation Validation System for LeanVibe Agent Hive 2.0

This system prevents documentation drift by continuously validating key documentation
files (PLAN.md, PROMPT.md, etc.) against actual system state.

Features:
- Real-time port/service discovery and validation
- Test count synchronization with qa-test-guardian
- System state tracking and documentation accuracy monitoring
- Automated documentation corrections and alerts
- Living documentation patterns preventing future inaccuracies
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentationValidationSystem:
    """Comprehensive automated documentation validation and maintenance system."""
    
    def __init__(self, project_root: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.project_root = Path(project_root)
        self.docs_path = self.project_root / "docs"
        self.validation_config = self._load_validation_config()
        self.system_state_cache = {}
        self.last_validation = None
        
        # Critical documents requiring validation
        self.critical_docs = [
            self.docs_path / "PLAN.md",
            self.docs_path / "PROMPT.md",
            self.docs_path / "SYSTEM_CAPABILITY_AUDIT.md",
            self.project_root / "README.md"
        ]
        
    def _load_validation_config(self) -> Dict[str, Any]:
        """Load configuration for documentation validation rules."""
        config_path = self.project_root / "config" / "documentation_validation.yaml"
        
        default_config = {
            "validation_frequency_minutes": 30,
            "accuracy_threshold": 95.0,
            "max_drift_tolerance_hours": 4,
            "critical_patterns": {
                "ports": {
                    "postgresql": 5432,
                    "redis": 6379,
                    "api_server": 18080
                },
                "system_state": {
                    "functionality_percentage": r"(\d{1,2})% functional",
                    "test_count": r"(\d+)\+ test[s]? files?"
                },
                "infrastructure": [
                    "PostgreSQL",
                    "Redis", 
                    "FastAPI",
                    "tmux sessions"
                ]
            },
            "auto_correction": {
                "enabled": True,
                "require_approval": False,
                "backup_before_correction": True
            }
        }
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                
        return default_config

    async def discover_system_state(self) -> Dict[str, Any]:
        """Discover actual system state including ports, services, and test counts."""
        logger.info("Discovering actual system state...")
        
        system_state = {
            "timestamp": datetime.now().isoformat(),
            "infrastructure": await self._check_infrastructure(),
            "test_counts": await self._count_tests(),
            "service_ports": await self._discover_service_ports(),
            "functionality_status": await self._assess_functionality(),
            "agent_capabilities": await self._inventory_agent_capabilities()
        }
        
        self.system_state_cache = system_state
        return system_state

    async def _check_infrastructure(self) -> Dict[str, Any]:
        """Check infrastructure component status."""
        infrastructure = {}
        
        # Check PostgreSQL
        try:
            result = subprocess.run(
                ["pg_isready", "-h", "localhost", "-p", "5432"],
                capture_output=True, text=True, timeout=10
            )
            infrastructure["postgresql"] = {
                "status": "operational" if result.returncode == 0 else "down",
                "port": 5432,
                "validated": True
            }
        except Exception as e:
            infrastructure["postgresql"] = {
                "status": "unknown",
                "port": 5432,
                "error": str(e)
            }
        
        # Check Redis
        try:
            result = subprocess.run(
                ["redis-cli", "-p", "6379", "ping"],
                capture_output=True, text=True, timeout=10
            )
            infrastructure["redis"] = {
                "status": "operational" if "PONG" in result.stdout else "down",
                "port": 6379,
                "validated": True
            }
        except Exception as e:
            infrastructure["redis"] = {
                "status": "unknown", 
                "port": 6379,
                "error": str(e)
            }
            
        return infrastructure

    async def _count_tests(self) -> Dict[str, Any]:
        """Count actual test files and validate test infrastructure."""
        test_counts = {}
        
        # Count test files in tests directory
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            test_counts["total_test_files"] = len(test_files)
            test_counts["test_files_path"] = str(tests_dir)
            
            # Check for specific test categories
            categories = {
                "integration": len(list(tests_dir.glob("*integration*.py"))),
                "performance": len(list(tests_dir.glob("*performance*.py"))),
                "orchestrator": len(list(tests_dir.glob("*orchestrator*.py"))),
                "api": len(list(tests_dir.glob("*api*.py")))
            }
            test_counts["categories"] = categories
            
        # Try to run a simple test discovery to validate infrastructure
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--collect-only", "-q"],
                cwd=self.project_root,
                capture_output=True, text=True, timeout=30
            )
            test_counts["pytest_discovery"] = {
                "status": "success" if result.returncode == 0 else "failed",
                "output_sample": result.stdout[:500]
            }
        except Exception as e:
            test_counts["pytest_discovery"] = {
                "status": "error",
                "error": str(e)
            }
            
        return test_counts

    async def _discover_service_ports(self) -> Dict[str, Any]:
        """Discover actual service ports in use."""
        ports = {}
        
        # Check common ports for the application
        common_ports = [5432, 6379, 18080, 8000, 3000]
        
        for port in common_ports:
            try:
                result = subprocess.run(
                    ["lsof", "-i", f":{port}"],
                    capture_output=True, text=True, timeout=5
                )
                ports[port] = {
                    "in_use": len(result.stdout.strip()) > 0,
                    "process_info": result.stdout.strip().split('\n')[0] if result.stdout.strip() else None
                }
            except Exception as e:
                ports[port] = {"error": str(e)}
                
        return ports

    async def _assess_functionality(self) -> Dict[str, Any]:
        """Assess actual system functionality level."""
        functionality = {
            "assessment_method": "automated_validation",
            "timestamp": datetime.now().isoformat()
        }
        
        # Check CLI functionality
        try:
            result = subprocess.run(
                ["python", "hive.py", "--version"],
                cwd=self.project_root,
                capture_output=True, text=True, timeout=15
            )
            functionality["cli_responsive"] = result.returncode == 0
            functionality["cli_output"] = result.stdout.strip()
        except Exception as e:
            functionality["cli_responsive"] = False
            functionality["cli_error"] = str(e)
        
        # Estimate functionality percentage based on validated components
        working_components = 0
        total_components = 5  # Infrastructure, CLI, Tests, API, Agent Management
        
        if self.system_state_cache.get("infrastructure", {}).get("postgresql", {}).get("status") == "operational":
            working_components += 1
        if self.system_state_cache.get("infrastructure", {}).get("redis", {}).get("status") == "operational":
            working_components += 1
        if functionality.get("cli_responsive"):
            working_components += 1
        if self.system_state_cache.get("test_counts", {}).get("total_test_files", 0) > 50:
            working_components += 1
        # API check would require more complex validation
        
        functionality["estimated_percentage"] = round((working_components / total_components) * 100)
        functionality["components_working"] = working_components
        functionality["components_total"] = total_components
        
        return functionality

    async def _inventory_agent_capabilities(self) -> Dict[str, Any]:
        """Inventory available agent capabilities and deployment options."""
        capabilities = {}
        
        # Check for agent-related files and configurations
        agent_dirs = [
            self.project_root / "app" / "agents",
            self.project_root / "app" / "core"
        ]
        
        agent_files = []
        for agent_dir in agent_dirs:
            if agent_dir.exists():
                agent_files.extend(list(agent_dir.glob("*agent*.py")))
                agent_files.extend(list(agent_dir.glob("*orchestrator*.py")))
        
        capabilities["agent_implementation_files"] = len(agent_files)
        capabilities["agent_files_found"] = [str(f.name) for f in agent_files[:10]]  # Sample
        
        # Check for Docker agent configurations
        docker_files = list(self.project_root.glob("Dockerfile.agent-*"))
        capabilities["docker_agent_configs"] = len(docker_files)
        
        return capabilities

    async def validate_document_accuracy(self, doc_path: Path) -> Dict[str, Any]:
        """Validate a specific document against actual system state."""
        logger.info(f"Validating document accuracy: {doc_path}")
        
        if not doc_path.exists():
            return {"status": "error", "message": f"Document not found: {doc_path}"}
        
        with open(doc_path, 'r') as f:
            content = f.read()
        
        validation_results = {
            "document": str(doc_path),
            "timestamp": datetime.now().isoformat(),
            "accuracy_issues": [],
            "suggestions": [],
            "overall_accuracy": 100.0
        }
        
        # Check port references
        port_issues = await self._validate_port_references(content)
        validation_results["accuracy_issues"].extend(port_issues)
        
        # Check system state claims
        state_issues = await self._validate_system_state_claims(content)
        validation_results["accuracy_issues"].extend(state_issues)
        
        # Check test count accuracy
        test_issues = await self._validate_test_count_claims(content)
        validation_results["accuracy_issues"].extend(test_issues)
        
        # Calculate overall accuracy
        if validation_results["accuracy_issues"]:
            # Reduce accuracy based on number and severity of issues
            severity_weights = {"critical": 20, "high": 15, "medium": 10, "low": 5}
            total_deduction = sum(
                severity_weights.get(issue.get("severity", "medium"), 10)
                for issue in validation_results["accuracy_issues"]
            )
            validation_results["overall_accuracy"] = max(0, 100 - total_deduction)
        
        return validation_results

    async def _validate_port_references(self, content: str) -> List[Dict[str, Any]]:
        """Validate port references in documentation."""
        issues = []
        
        # Check for incorrect port references
        incorrect_ports = {
            "15432": "Should be 5432 for PostgreSQL",
            "16379": "Should be 6379 for Redis"
        }
        
        for incorrect_port, correction in incorrect_ports.items():
            if incorrect_port in content:
                issues.append({
                    "type": "incorrect_port",
                    "severity": "high",
                    "message": f"Found incorrect port {incorrect_port}. {correction}",
                    "suggested_fix": f"Replace {incorrect_port} with correct port"
                })
        
        return issues

    async def _validate_system_state_claims(self, content: str) -> List[Dict[str, Any]]:
        """Validate system state claims against actual functionality."""
        issues = []
        
        # Extract functionality percentage claims
        functionality_matches = re.findall(r"(\d{1,3})% functional", content)
        
        if functionality_matches:
            claimed_percentage = int(functionality_matches[0])
            actual_percentage = self.system_state_cache.get("functionality_status", {}).get("estimated_percentage", 85)
            
            # Allow 10% tolerance
            if abs(claimed_percentage - actual_percentage) > 10:
                issues.append({
                    "type": "functionality_mismatch",
                    "severity": "medium", 
                    "message": f"Document claims {claimed_percentage}% functional, validation suggests {actual_percentage}%",
                    "suggested_fix": f"Update to reflect validated {actual_percentage}% functionality"
                })
        
        return issues

    async def _validate_test_count_claims(self, content: str) -> List[Dict[str, Any]]:
        """Validate test count claims against actual test files."""
        issues = []
        
        # Extract test count claims
        test_matches = re.findall(r"(\d+)\+ test", content)
        
        if test_matches:
            claimed_count = int(test_matches[0])
            actual_count = self.system_state_cache.get("test_counts", {}).get("total_test_files", 0)
            
            # Check if claim is reasonable (within 20% of actual)
            if claimed_count > actual_count * 1.2 or claimed_count < actual_count * 0.8:
                issues.append({
                    "type": "test_count_mismatch",
                    "severity": "medium",
                    "message": f"Document claims {claimed_count}+ tests, actual count is {actual_count}",
                    "suggested_fix": f"Update to reflect actual {actual_count} test files"
                })
        
        return issues

    async def generate_living_documentation_report(self) -> Dict[str, Any]:
        """Generate comprehensive living documentation status report."""
        logger.info("Generating living documentation report...")
        
        # Discover current system state
        system_state = await self.discover_system_state()
        
        # Validate all critical documents
        document_validations = {}
        for doc_path in self.critical_docs:
            if doc_path.exists():
                validation = await self.validate_document_accuracy(doc_path)
                document_validations[str(doc_path)] = validation
        
        # Generate consolidation analysis
        consolidation_analysis = await self._analyze_documentation_consolidation()
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "system_state": system_state,
            "document_validations": document_validations,
            "consolidation_analysis": consolidation_analysis,
            "overall_accuracy": self._calculate_overall_accuracy(document_validations),
            "recommendations": self._generate_recommendations(document_validations, consolidation_analysis)
        }
        
        # Save report
        report_path = self.project_root / "reports" / f"living_documentation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Living documentation report saved: {report_path}")
        return report

    async def _analyze_documentation_consolidation(self) -> Dict[str, Any]:
        """Analyze documentation ecosystem for consolidation opportunities."""
        analysis = {
            "total_docs": 0,
            "categories": {},
            "redundancy_analysis": [],
            "consolidation_opportunities": []
        }
        
        # Count documentation files
        doc_patterns = ["*.md", "*.rst", "*.txt"]
        all_docs = []
        
        for pattern in doc_patterns:
            all_docs.extend(list(self.project_root.rglob(pattern)))
        
        analysis["total_docs"] = len(all_docs)
        
        # Categorize documents
        categories = {
            "strategic": ["PLAN", "PROMPT", "ROADMAP", "STRATEGY"],
            "technical": ["API", "ARCHITECTURE", "TECHNICAL", "IMPLEMENTATION"],
            "operational": ["DEPLOYMENT", "OPERATION", "RUNBOOK", "GUIDE"],
            "testing": ["TEST", "VALIDATION", "BENCHMARK"],
            "documentation": ["README", "GETTING_STARTED", "DOCS"]
        }
        
        for category, keywords in categories.items():
            category_docs = []
            for doc in all_docs:
                if any(keyword in doc.name.upper() for keyword in keywords):
                    category_docs.append(str(doc))
            analysis["categories"][category] = {
                "count": len(category_docs),
                "files": category_docs[:10]  # Sample
            }
        
        # Identify potential redundancy (simplified heuristic)
        doc_names = [doc.stem.upper() for doc in all_docs]
        similar_groups = {}
        
        for name in doc_names:
            base_name = re.sub(r'[_\-\d]+', '', name)
            if base_name not in similar_groups:
                similar_groups[base_name] = []
            similar_groups[base_name].append(name)
        
        for base_name, variants in similar_groups.items():
            if len(variants) > 2:  # More than 2 similar named files
                analysis["redundancy_analysis"].append({
                    "base_name": base_name,
                    "variants": variants,
                    "potential_redundancy": len(variants)
                })
        
        return analysis

    def _calculate_overall_accuracy(self, document_validations: Dict[str, Any]) -> float:
        """Calculate overall documentation accuracy score."""
        if not document_validations:
            return 0.0
        
        accuracies = [
            validation.get("overall_accuracy", 0.0)
            for validation in document_validations.values()
        ]
        
        return sum(accuracies) / len(accuracies)

    def _generate_recommendations(self, validations: Dict[str, Any], consolidation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Accuracy recommendations
        for doc_path, validation in validations.items():
            if validation.get("overall_accuracy", 100) < 90:
                recommendations.append({
                    "type": "accuracy_improvement",
                    "priority": "high",
                    "document": doc_path,
                    "action": "Review and correct identified accuracy issues",
                    "issues_count": len(validation.get("accuracy_issues", []))
                })
        
        # Consolidation recommendations
        total_docs = consolidation.get("total_docs", 0)
        if total_docs > 200:
            recommendations.append({
                "type": "consolidation",
                "priority": "medium",
                "action": f"Consider consolidating {total_docs} documentation files",
                "potential_reduction": f"Target 50-70% reduction to ~{total_docs // 3} core documents"
            })
        
        redundancy_count = len(consolidation.get("redundancy_analysis", []))
        if redundancy_count > 10:
            recommendations.append({
                "type": "redundancy_reduction", 
                "priority": "medium",
                "action": f"Address {redundancy_count} potential redundant document groups",
                "focus": "Merge or eliminate duplicate content"
            })
        
        return recommendations

    async def start_continuous_monitoring(self, interval_minutes: int = 30):
        """Start continuous documentation monitoring and validation."""
        logger.info(f"Starting continuous documentation monitoring (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                # Generate living documentation report
                report = await self.generate_living_documentation_report()
                
                # Check if any documents need attention
                overall_accuracy = report.get("overall_accuracy", 100)
                
                if overall_accuracy < self.validation_config.get("accuracy_threshold", 95):
                    logger.warning(f"Documentation accuracy below threshold: {overall_accuracy:.1f}%")
                    
                    # Trigger automated corrections if enabled
                    if self.validation_config.get("auto_correction", {}).get("enabled", False):
                        await self._attempt_automated_corrections(report)
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _attempt_automated_corrections(self, report: Dict[str, Any]):
        """Attempt automated corrections for common documentation issues."""
        logger.info("Attempting automated documentation corrections...")
        
        for doc_path, validation in report.get("document_validations", {}).items():
            if validation.get("overall_accuracy", 100) < 90:
                await self._correct_document_issues(doc_path, validation)

    async def _correct_document_issues(self, doc_path: str, validation: Dict[str, Any]):
        """Apply automated corrections to a specific document."""
        try:
            with open(doc_path, 'r') as f:
                content = f.read()
            
            original_content = content
            corrections_made = []
            
            # Apply corrections for known issues
            for issue in validation.get("accuracy_issues", []):
                if issue["type"] == "incorrect_port":
                    # Fix port references
                    content = content.replace("15432", "5432")
                    content = content.replace("16379", "6379")
                    corrections_made.append("Fixed incorrect port references")
            
            # Only write if changes were made
            if content != original_content:
                # Backup original if configured
                if self.validation_config.get("auto_correction", {}).get("backup_before_correction", True):
                    backup_path = f"{doc_path}.backup_{int(time.time())}"
                    with open(backup_path, 'w') as f:
                        f.write(original_content)
                    logger.info(f"Created backup: {backup_path}")
                
                # Write corrected content
                with open(doc_path, 'w') as f:
                    f.write(content)
                
                logger.info(f"Applied automated corrections to {doc_path}: {corrections_made}")
            
        except Exception as e:
            logger.error(f"Failed to apply corrections to {doc_path}: {e}")


async def main():
    """Main function to run documentation validation system."""
    system = DocumentationValidationSystem()
    
    # Generate initial report
    print("Generating initial living documentation report...")
    report = await system.generate_living_documentation_report()
    
    print(f"\nLiving Documentation System Report")
    print(f"==================================")
    print(f"System State: {report['system_state']['functionality_status']['estimated_percentage']}% functional")
    print(f"Infrastructure: PostgreSQL {report['system_state']['infrastructure']['postgresql']['status']}, Redis {report['system_state']['infrastructure']['redis']['status']}")
    print(f"Test Files: {report['system_state']['test_counts']['total_test_files']} discovered")
    print(f"Documentation Files: {report['consolidation_analysis']['total_docs']} total")
    print(f"Overall Accuracy: {report['overall_accuracy']:.1f}%")
    print(f"Recommendations: {len(report['recommendations'])}")
    
    for rec in report['recommendations'][:3]:  # Show top 3 recommendations
        print(f"  - {rec['type'].title()}: {rec['action']}")
    
    # Start continuous monitoring (commented out for demo)
    # print(f"\nStarting continuous monitoring...")
    # await system.start_continuous_monitoring(interval_minutes=30)


if __name__ == "__main__":
    asyncio.run(main())