#!/usr/bin/env python3
"""
Master Documentation System Orchestrator

Coordinates all documentation maintenance systems to provide a unified
interface for preventing documentation drift and ensuring accuracy.

Features:
- Unified command interface for all documentation systems
- Coordinated execution of validation, monitoring, and consolidation
- Real-time accuracy dashboard
- Automated quality gate enforcement
- Developer-friendly CLI with comprehensive reporting
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MasterDocumentationSystem:
    """Master orchestrator for all documentation maintenance systems."""
    
    def __init__(self, project_root: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.project_root = Path(project_root)
        
        # Import all subsystems dynamically to avoid import errors
        self.subsystems = {}
        
    def _get_subsystem(self, system_name: str):
        """Lazy load subsystems to avoid import issues."""
        if system_name not in self.subsystems:
            try:
                if system_name == "validation":
                    from .documentation_validation_system import DocumentationValidationSystem
                    self.subsystems[system_name] = DocumentationValidationSystem(str(self.project_root))
                elif system_name == "infrastructure":
                    from .infrastructure_monitoring_system import InfrastructureMonitoringSystem  
                    self.subsystems[system_name] = InfrastructureMonitoringSystem(str(self.project_root))
                elif system_name == "test_sync":
                    from .test_synchronization_system import TestSynchronizationSystem
                    self.subsystems[system_name] = TestSynchronizationSystem(str(self.project_root))
                elif system_name == "living_docs":
                    from .living_documentation_framework import LivingDocumentationFramework
                    self.subsystems[system_name] = LivingDocumentationFramework(str(self.project_root))
                elif system_name == "consolidation":
                    from .documentation_consolidation_analyzer import DocumentationConsolidationAnalyzer
                    self.subsystems[system_name] = DocumentationConsolidationAnalyzer(str(self.project_root))
                elif system_name == "quality_gates":
                    from .documentation_quality_gates import DocumentationQualityGates
                    self.subsystems[system_name] = DocumentationQualityGates(str(self.project_root))
            except ImportError as e:
                logger.warning(f"Could not import {system_name}: {e}")
                return None
        
        return self.subsystems.get(system_name)

    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check across all documentation systems."""
        print("🔍 Running Comprehensive Documentation Health Check")
        print("=" * 55)
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "systems": {},
            "summary": {},
            "recommendations": []
        }
        
        # Test basic system availability
        print("1. Checking system availability...")
        
        systems_status = {}
        for system_name in ["validation", "infrastructure", "test_sync", "living_docs", "consolidation", "quality_gates"]:
            try:
                system = self._get_subsystem(system_name)
                systems_status[system_name] = "available" if system else "unavailable"
                print(f"   {system_name}: {'✅' if system else '❌'}")
            except Exception as e:
                systems_status[system_name] = f"error: {str(e)}"
                print(f"   {system_name}: ❌ ({str(e)[:50]}...)")
        
        health_report["systems"] = systems_status
        
        # Basic file system checks
        print("\n2. Checking critical documentation files...")
        
        critical_files = [
            self.project_root / "docs" / "PLAN.md",
            self.project_root / "docs" / "PROMPT.md",
            self.project_root / "README.md"
        ]
        
        file_status = {}
        for file_path in critical_files:
            exists = file_path.exists()
            file_status[str(file_path.name)] = exists
            print(f"   {file_path.name}: {'✅' if exists else '❌'}")
        
        health_report["critical_files"] = file_status
        
        # Quick configuration check
        print("\n3. Checking configuration files...")
        
        config_files = [
            self.project_root / "config" / "documentation_validation.yaml",
            self.project_root / "config" / "living_documentation.yaml",
            self.project_root / "config" / "documentation_quality_gates.yaml"
        ]
        
        config_status = {}
        for config_path in config_files:
            exists = config_path.exists()
            config_status[str(config_path.name)] = exists
            print(f"   {config_path.name}: {'✅' if exists else '❌'}")
        
        health_report["configuration"] = config_status
        
        # Generate summary
        available_systems = sum(1 for status in systems_status.values() if status == "available")
        total_systems = len(systems_status)
        available_files = sum(1 for exists in file_status.values() if exists)
        total_files = len(file_status)
        available_configs = sum(1 for exists in config_status.values() if exists)
        total_configs = len(config_status)
        
        health_report["summary"] = {
            "systems_available": f"{available_systems}/{total_systems}",
            "critical_files_present": f"{available_files}/{total_files}",
            "configurations_present": f"{available_configs}/{total_configs}",
            "overall_health_percentage": round(
                ((available_systems / total_systems) + 
                 (available_files / total_files) + 
                 (available_configs / total_configs)) / 3 * 100, 1
            )
        }
        
        overall_health = health_report["summary"]["overall_health_percentage"]
        if overall_health >= 90:
            health_report["overall_status"] = "excellent"
        elif overall_health >= 75:
            health_report["overall_status"] = "good"
        elif overall_health >= 50:
            health_report["overall_status"] = "fair"
        else:
            health_report["overall_status"] = "poor"
        
        print(f"\n📊 Overall Health: {overall_health}% ({health_report['overall_status'].title()})")
        
        # Generate recommendations
        if available_systems < total_systems:
            health_report["recommendations"].append(
                "Some documentation systems are unavailable - check dependencies"
            )
        
        if available_files < total_files:
            health_report["recommendations"].append(
                "Critical documentation files are missing - verify file locations"
            )
        
        if available_configs < total_configs:
            health_report["recommendations"].append(
                "Configuration files are missing - run system installation"
            )
        
        return health_report

    async def quick_validation(self) -> Dict[str, Any]:
        """Run quick validation of critical documents."""
        print("⚡ Quick Documentation Validation")
        print("=" * 35)
        
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "documents_checked": [],
            "accuracy_scores": {},
            "critical_issues": [],
            "overall_accuracy": 0.0
        }
        
        # Check if validation system is available
        validation_system = self._get_subsystem("validation")
        if not validation_system:
            print("❌ Validation system not available")
            return validation_result
        
        # Quick check of critical documents
        critical_docs = [
            self.project_root / "docs" / "PLAN.md",
            self.project_root / "docs" / "PROMPT.md"
        ]
        
        accuracies = []
        
        for doc_path in critical_docs:
            if doc_path.exists():
                print(f"Checking {doc_path.name}...")
                try:
                    # Simple validation check
                    with open(doc_path, 'r') as f:
                        content = f.read()
                    
                    # Basic accuracy heuristics
                    issues = []
                    
                    # Check for incorrect ports
                    if "15432" in content:
                        issues.append("Incorrect PostgreSQL port (should be 5432)")
                    if "16379" in content:
                        issues.append("Incorrect Redis port (should be 6379)")
                    
                    # Check for unrealistic percentages
                    import re
                    percentages = re.findall(r'(\d{3})% functional', content)
                    if percentages and any(int(p) > 100 for p in percentages):
                        issues.append("Unrealistic functionality percentage found")
                    
                    # Calculate simple accuracy score
                    accuracy = 100.0 - (len(issues) * 15.0)  # 15% penalty per issue
                    accuracy = max(0.0, accuracy)
                    
                    accuracies.append(accuracy)
                    validation_result["documents_checked"].append(str(doc_path.name))
                    validation_result["accuracy_scores"][str(doc_path.name)] = accuracy
                    validation_result["critical_issues"].extend(issues)
                    
                    status_icon = "✅" if accuracy >= 90 else "⚠️" if accuracy >= 75 else "❌"
                    print(f"   {doc_path.name}: {status_icon} {accuracy:.1f}%")
                    
                    if issues:
                        for issue in issues:
                            print(f"     - {issue}")
                    
                except Exception as e:
                    print(f"   {doc_path.name}: ❌ Error - {str(e)}")
            else:
                print(f"   {doc_path.name}: ❌ File not found")
        
        if accuracies:
            validation_result["overall_accuracy"] = sum(accuracies) / len(accuracies)
        
        print(f"\n📊 Overall Accuracy: {validation_result['overall_accuracy']:.1f}%")
        if validation_result["critical_issues"]:
            print(f"🚨 Critical Issues: {len(validation_result['critical_issues'])}")
        
        return validation_result

    async def install_documentation_systems(self) -> Dict[str, Any]:
        """Install and configure all documentation systems."""
        print("🚀 Installing Documentation Maintenance Systems")
        print("=" * 48)
        
        installation_result = {
            "timestamp": datetime.now().isoformat(),
            "systems_installed": [],
            "configurations_created": [],
            "errors": []
        }
        
        # Create directories
        print("1. Creating system directories...")
        
        directories = [
            self.project_root / "config",
            self.project_root / "scripts",
            self.project_root / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"   ✅ {directory.name}/")
        
        # Install quality gates
        print("\n2. Installing documentation quality gates...")
        
        try:
            quality_gates = self._get_subsystem("quality_gates")
            if quality_gates:
                await quality_gates.install_quality_gates()
                installation_result["systems_installed"].append("quality_gates")
                print("   ✅ Quality gates installed")
            else:
                print("   ❌ Quality gates system not available")
        except Exception as e:
            error_msg = f"Quality gates installation failed: {str(e)}"
            installation_result["errors"].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        # Initialize living documentation
        print("\n3. Initializing living documentation framework...")
        
        try:
            living_docs = self._get_subsystem("living_docs")
            if living_docs:
                await living_docs.initialize_living_documentation()
                installation_result["systems_installed"].append("living_docs")
                print("   ✅ Living documentation framework initialized")
            else:
                print("   ❌ Living documentation system not available")
        except Exception as e:
            error_msg = f"Living documentation initialization failed: {str(e)}"
            installation_result["errors"].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        print(f"\n📦 Installation Summary:")
        print(f"   Systems Installed: {len(installation_result['systems_installed'])}")
        print(f"   Errors: {len(installation_result['errors'])}")
        
        if installation_result["errors"]:
            print(f"\n🚨 Installation Errors:")
            for error in installation_result["errors"]:
                print(f"   - {error}")
        
        return installation_result

    async def generate_status_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive status dashboard."""
        print("📊 Documentation Systems Status Dashboard")
        print("=" * 42)
        
        # Run health check
        health_report = await self.run_comprehensive_health_check()
        
        print(f"\n⚡ Quick Validation")
        print("-" * 18)
        validation_report = await self.quick_validation()
        
        # Combine reports
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "health_check": health_report,
            "validation_check": validation_report,
            "recommendations": []
        }
        
        # Generate consolidated recommendations
        overall_health = health_report["summary"]["overall_health_percentage"]
        overall_accuracy = validation_report["overall_accuracy"]
        
        if overall_health < 75:
            dashboard["recommendations"].append("🔧 Run system installation to fix missing components")
        
        if overall_accuracy < 90:
            dashboard["recommendations"].append("📝 Run documentation validation and auto-fix")
        
        if validation_report["critical_issues"]:
            dashboard["recommendations"].append("🚨 Address critical documentation issues immediately")
        
        print(f"\n🎯 Recommendations:")
        for rec in dashboard["recommendations"]:
            print(f"   {rec}")
        
        # Save dashboard report
        report_path = self.project_root / "reports" / f"documentation_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        print(f"\n💾 Dashboard saved: {report_path}")
        
        return dashboard


async def main():
    """Main CLI interface for master documentation system."""
    system = MasterDocumentationSystem()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "health":
            await system.run_comprehensive_health_check()
            
        elif command == "validate":
            await system.quick_validation()
            
        elif command == "install":
            await system.install_documentation_systems()
            
        elif command == "dashboard":
            await system.generate_status_dashboard()
            
        elif command == "help":
            print("Master Documentation System")
            print("===========================")
            print("Commands:")
            print("  health       - Run comprehensive health check")
            print("  validate     - Quick validation of critical documents")
            print("  install      - Install all documentation systems")
            print("  dashboard    - Generate status dashboard")
            print("  help         - Show this help")
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for available commands")
    
    else:
        # Default: show dashboard
        await system.generate_status_dashboard()


if __name__ == "__main__":
    asyncio.run(main())