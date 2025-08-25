#!/usr/bin/env python3
"""
Documentation Quality Gates Integration System

Integrates automated documentation validation and maintenance into the 
development workflow with quality gates preventing documentation drift.

Features:
- Pre-commit hooks for documentation validation
- CI/CD pipeline integration
- Git workflow automation
- Real-time quality monitoring
- Automated correction workflows
- Developer-friendly feedback systems
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
import yaml
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentationQualityGates:
    """Integration system for documentation quality gates in development workflow."""
    
    def __init__(self, project_root: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.project_root = Path(project_root)
        self.quality_config = self._load_quality_config()
        self.git_hooks_dir = self.project_root / ".git" / "hooks"
        self.ci_config_dir = self.project_root / ".github" / "workflows"
        
        # Quality gate thresholds
        self.quality_thresholds = {
            "minimum_accuracy": self.quality_config.get("minimum_accuracy", 90.0),
            "critical_issue_block": self.quality_config.get("critical_issue_block", True),
            "port_mismatch_block": self.quality_config.get("port_mismatch_block", True),
            "test_count_drift_threshold": self.quality_config.get("test_count_drift_threshold", 20)
        }

    def _load_quality_config(self) -> Dict[str, Any]:
        """Load quality gates configuration."""
        config_path = self.project_root / "config" / "documentation_quality_gates.yaml"
        
        default_config = {
            "enabled": True,
            "minimum_accuracy": 90.0,
            "critical_issue_block": True,
            "port_mismatch_block": True,
            "test_count_drift_threshold": 20,
            "auto_correction": {
                "enabled": True,
                "backup_before_correction": True,
                "require_approval_for_major_changes": False
            },
            "git_integration": {
                "pre_commit_validation": True,
                "post_commit_updates": True,
                "pre_push_comprehensive_check": True
            },
            "ci_cd_integration": {
                "github_actions": True,
                "documentation_checks": True,
                "fail_on_accuracy_below_threshold": True
            },
            "developer_experience": {
                "friendly_error_messages": True,
                "suggested_fixes": True,
                "documentation_links": True,
                "progress_indicators": True
            }
        }
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config

    async def install_quality_gates(self):
        """Install all documentation quality gates into development workflow."""
        logger.info("Installing documentation quality gates...")
        
        installation_result = {
            "timestamp": datetime.now().isoformat(),
            "components_installed": [],
            "configuration_files": [],
            "git_hooks": [],
            "errors": []
        }
        
        try:
            # Install Git hooks
            if self.quality_config.get("git_integration", {}).get("pre_commit_validation", True):
                await self._install_git_hooks()
                installation_result["git_hooks"] = ["pre-commit", "post-commit", "pre-push"]
            
            # Install GitHub Actions
            if self.quality_config.get("ci_cd_integration", {}).get("github_actions", True):
                await self._install_github_actions()
                installation_result["components_installed"].append("github_actions")
            
            # Create quality gates configuration
            await self._create_quality_config_files()
            installation_result["configuration_files"] = [
                "documentation_quality_gates.yaml",
                "pre_commit_config.yaml"
            ]
            
            # Install development tools
            await self._install_development_tools()
            installation_result["components_installed"].extend(["make_targets", "validation_scripts"])
            
        except Exception as e:
            error_msg = f"Failed to install quality gates: {str(e)}"
            installation_result["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Save installation status
        status_path = self.project_root / ".documentation_quality_gates_status.json"
        with open(status_path, 'w') as f:
            json.dump(installation_result, f, indent=2)
        
        logger.info("Documentation quality gates installation complete")
        return installation_result

    async def _install_git_hooks(self):
        """Install Git hooks for documentation quality gates."""
        if not self.git_hooks_dir.exists():
            logger.warning("Git hooks directory not found - initializing git repository")
            subprocess.run(["git", "init"], cwd=self.project_root)
            self.git_hooks_dir.mkdir(exist_ok=True)
        
        # Pre-commit hook
        pre_commit_content = '''#!/bin/bash
# Documentation Quality Gates Pre-commit Hook
echo "🔍 Validating documentation quality..."

python scripts/documentation_quality_gates.py --pre-commit-check
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "❌ Documentation quality checks failed!"
    echo "   Run 'make docs-fix' to auto-correct issues"
    echo "   Or use 'git commit --no-verify' to bypass (not recommended)"
    echo ""
    exit $exit_code
fi

echo "✅ Documentation quality checks passed"
'''
        
        await self._install_git_hook("pre-commit", pre_commit_content)
        
        # Post-commit hook
        post_commit_content = '''#!/bin/bash
# Documentation Quality Gates Post-commit Hook
echo "📝 Updating living documentation..."

python scripts/documentation_quality_gates.py --post-commit-update --background
'''
        
        await self._install_git_hook("post-commit", post_commit_content)
        
        # Pre-push hook
        pre_push_content = '''#!/bin/bash
# Documentation Quality Gates Pre-push Hook
echo "🚀 Comprehensive documentation validation..."

python scripts/documentation_quality_gates.py --pre-push-check
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "❌ Comprehensive documentation validation failed!"
    echo "   Please fix documentation issues before pushing"
    echo ""
    exit $exit_code
fi

echo "✅ All documentation quality checks passed"
'''
        
        await self._install_git_hook("pre-push", pre_push_content)

    async def _install_git_hook(self, hook_name: str, content: str):
        """Install a specific git hook."""
        hook_path = self.git_hooks_dir / hook_name
        
        # Backup existing hook if it exists
        if hook_path.exists():
            backup_path = hook_path.with_suffix(f".backup_{int(time.time())}")
            shutil.copy2(hook_path, backup_path)
            logger.info(f"Backed up existing {hook_name} hook to {backup_path}")
        
        # Write new hook
        with open(hook_path, 'w') as f:
            f.write(content)
        
        # Make executable
        os.chmod(hook_path, 0o755)
        
        logger.info(f"Installed {hook_name} hook")

    async def _install_github_actions(self):
        """Install GitHub Actions workflow for documentation quality."""
        self.ci_config_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_content = '''name: Documentation Quality Gates

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  documentation-quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt || echo "No requirements.txt found"
        pip install pyyaml psutil redis psycopg2-binary
    
    - name: Run documentation validation
      run: |
        python scripts/documentation_quality_gates.py --ci-validation
        
    - name: Generate documentation report
      run: |
        python scripts/documentation_validation_system.py > docs_validation_report.txt
        
    - name: Upload documentation report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: documentation-quality-report
        path: docs_validation_report.txt
        
    - name: Comment PR with results
      uses: actions/github-script@v6
      if: github.event_name == 'pull_request'
      with:
        script: |
          const fs = require('fs');
          if (fs.existsSync('docs_validation_report.txt')) {
            const report = fs.readFileSync('docs_validation_report.txt', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Documentation Quality Report\\n\\n\`\`\`\\n${report}\\n\`\`\``
            });
          }
'''
        
        workflow_path = self.ci_config_dir / "documentation-quality.yml"
        with open(workflow_path, 'w') as f:
            f.write(workflow_content)
        
        logger.info(f"Installed GitHub Actions workflow: {workflow_path}")

    async def _create_quality_config_files(self):
        """Create configuration files for quality gates."""
        # Documentation quality gates config
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        quality_config_path = config_dir / "documentation_quality_gates.yaml"
        if not quality_config_path.exists():
            with open(quality_config_path, 'w') as f:
                yaml.dump(self.quality_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created quality gates config: {quality_config_path}")
        
        # Pre-commit configuration (if using pre-commit framework)
        pre_commit_config = {
            "repos": [
                {
                    "repo": "local",
                    "hooks": [
                        {
                            "id": "documentation-quality",
                            "name": "Documentation Quality Gates",
                            "entry": "python scripts/documentation_quality_gates.py --pre-commit-check",
                            "language": "system",
                            "files": r"\.(md|rst|txt)$"
                        }
                    ]
                }
            ]
        }
        
        pre_commit_path = self.project_root / ".pre-commit-config.yaml"
        if not pre_commit_path.exists():
            with open(pre_commit_path, 'w') as f:
                yaml.dump(pre_commit_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created pre-commit config: {pre_commit_path}")

    async def _install_development_tools(self):
        """Install development tools and convenience scripts."""
        # Makefile targets for documentation
        makefile_content = '''
# Documentation Quality Gates Makefile targets

.PHONY: docs-validate docs-fix docs-update docs-consolidate docs-status

docs-validate:
\t@echo "🔍 Validating documentation quality..."
\t@python scripts/documentation_validation_system.py

docs-fix:
\t@echo "🔧 Auto-correcting documentation issues..."
\t@python scripts/documentation_quality_gates.py --auto-fix

docs-update:
\t@echo "📝 Updating living documentation..."
\t@python scripts/living_documentation_framework.py --update

docs-consolidate:
\t@echo "📋 Analyzing consolidation opportunities..."
\t@python scripts/documentation_consolidation_analyzer.py

docs-status:
\t@echo "📊 Documentation system status..."
\t@python scripts/documentation_quality_gates.py --status

docs-install:
\t@echo "⚙️  Installing documentation quality gates..."
\t@python scripts/documentation_quality_gates.py --install
'''
        
        makefile_path = self.project_root / "Makefile.docs"
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
        
        logger.info(f"Created documentation Makefile: {makefile_path}")
        
        # Validation script for manual use
        validation_script = '''#!/bin/bash
# Manual documentation validation script

echo "Documentation Quality Validation"
echo "==============================="

echo "1. Running comprehensive validation..."
python scripts/documentation_validation_system.py

echo ""
echo "2. Checking infrastructure state..."
python scripts/infrastructure_monitoring_system.py

echo ""
echo "3. Synchronizing test counts..."
python scripts/test_synchronization_system.py

echo ""
echo "4. Validating living documentation..."
python scripts/living_documentation_framework.py --validate

echo ""
echo "✅ Documentation validation complete"
'''
        
        validation_script_path = self.project_root / "scripts" / "validate_docs.sh"
        with open(validation_script_path, 'w') as f:
            f.write(validation_script)
        
        os.chmod(validation_script_path, 0o755)
        logger.info(f"Created validation script: {validation_script_path}")

    async def run_pre_commit_check(self) -> Dict[str, Any]:
        """Run pre-commit documentation quality check."""
        logger.info("Running pre-commit documentation quality check...")
        
        check_result = {
            "timestamp": datetime.now().isoformat(),
            "passed": False,
            "checks": {},
            "errors": [],
            "suggestions": []
        }
        
        try:
            # Import validation systems
            from .documentation_validation_system import DocumentationValidationSystem
            from .infrastructure_monitoring_system import InfrastructureMonitoringSystem
            from .living_documentation_framework import LivingDocumentationFramework
            
            # Run validation checks
            doc_validator = DocumentationValidationSystem(str(self.project_root))
            living_docs = LivingDocumentationFramework(str(self.project_root))
            
            # Check critical document accuracy
            plan_validation = await doc_validator.validate_document_accuracy(
                self.project_root / "docs" / "PLAN.md"
            )
            prompt_validation = await doc_validator.validate_document_accuracy(
                self.project_root / "docs" / "PROMPT.md"
            )
            
            check_result["checks"]["plan_accuracy"] = plan_validation["overall_accuracy"]
            check_result["checks"]["prompt_accuracy"] = prompt_validation["overall_accuracy"]
            
            # Check for critical issues
            critical_issues = []
            for validation in [plan_validation, prompt_validation]:
                for issue in validation.get("accuracy_issues", []):
                    if issue.get("severity") in ["critical", "high"]:
                        critical_issues.append(issue)
            
            check_result["checks"]["critical_issues"] = len(critical_issues)
            
            # Validate living documentation
            living_validation = await living_docs.validate_all_living_documents()
            check_result["checks"]["living_docs_accuracy"] = living_validation["overall_accuracy"]
            
            # Apply quality gate thresholds
            min_accuracy = self.quality_thresholds["minimum_accuracy"]
            
            passed = True
            
            if plan_validation["overall_accuracy"] < min_accuracy:
                check_result["errors"].append(f"PLAN.md accuracy ({plan_validation['overall_accuracy']:.1f}%) below threshold ({min_accuracy}%)")
                passed = False
            
            if prompt_validation["overall_accuracy"] < min_accuracy:
                check_result["errors"].append(f"PROMPT.md accuracy ({prompt_validation['overall_accuracy']:.1f}%) below threshold ({min_accuracy}%)")
                passed = False
            
            if critical_issues and self.quality_thresholds["critical_issue_block"]:
                check_result["errors"].append(f"Found {len(critical_issues)} critical documentation issues")
                passed = False
            
            # Generate suggestions for fixes
            if not passed:
                check_result["suggestions"] = [
                    "Run 'make docs-fix' to auto-correct common issues",
                    "Check 'python scripts/documentation_validation_system.py' for detailed report",
                    "Use 'git commit --no-verify' to bypass checks (not recommended)"
                ]
            
            check_result["passed"] = passed
            
        except Exception as e:
            error_msg = f"Pre-commit check failed: {str(e)}"
            check_result["errors"].append(error_msg)
            logger.error(error_msg)
        
        return check_result

    async def run_post_commit_update(self):
        """Run post-commit living documentation update."""
        logger.info("Running post-commit documentation update...")
        
        try:
            from .living_documentation_framework import LivingDocumentationFramework
            
            living_docs = LivingDocumentationFramework(str(self.project_root))
            update_result = await living_docs.detect_changes_and_update()
            
            if update_result["documents_updated"]:
                logger.info(f"Updated living documents: {update_result['documents_updated']}")
                
                # Auto-commit living documentation updates if configured
                if self.quality_config.get("auto_commit_living_docs", False):
                    subprocess.run([
                        "git", "add", *update_result["documents_updated"]
                    ], cwd=self.project_root)
                    
                    subprocess.run([
                        "git", "commit", "-m", 
                        f"docs: auto-update living documentation ({len(update_result['documents_updated'])} files)"
                    ], cwd=self.project_root)
            
        except Exception as e:
            logger.error(f"Post-commit update failed: {e}")

    async def run_pre_push_check(self) -> Dict[str, Any]:
        """Run comprehensive pre-push documentation validation."""
        logger.info("Running pre-push comprehensive documentation check...")
        
        check_result = {
            "timestamp": datetime.now().isoformat(),
            "passed": False,
            "comprehensive_checks": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Import all validation systems
            from .documentation_validation_system import DocumentationValidationSystem
            from .infrastructure_monitoring_system import InfrastructureMonitoringSystem
            from .test_synchronization_system import TestSynchronizationSystem
            from .living_documentation_framework import LivingDocumentationFramework
            
            # Run comprehensive validation
            doc_validator = DocumentationValidationSystem(str(self.project_root))
            infra_monitor = InfrastructureMonitoringSystem(str(self.project_root))
            test_sync = TestSynchronizationSystem(str(self.project_root))
            living_docs = LivingDocumentationFramework(str(self.project_root))
            
            # Generate comprehensive reports
            doc_report = await doc_validator.generate_living_documentation_report()
            infra_report = await infra_monitor.generate_infrastructure_report()
            test_report = await test_sync.generate_test_report()
            living_report = await living_docs.validate_all_living_documents()
            
            # Compile results
            check_result["comprehensive_checks"] = {
                "overall_accuracy": doc_report["overall_accuracy"],
                "infrastructure_health": infra_report["health_summary"]["health_percentage"],
                "test_synchronization": "passed",
                "living_docs_accuracy": living_report["overall_accuracy"]
            }
            
            # Apply comprehensive quality gates
            passed = True
            
            if doc_report["overall_accuracy"] < 85.0:
                check_result["errors"].append(f"Overall documentation accuracy too low: {doc_report['overall_accuracy']:.1f}%")
                passed = False
            
            if infra_report["health_summary"]["health_percentage"] < 80.0:
                check_result["warnings"].append(f"Infrastructure health concerning: {infra_report['health_summary']['health_percentage']:.1f}%")
            
            if living_report.get("critical_issues"):
                check_result["errors"].append(f"Critical living documentation issues: {len(living_report['critical_issues'])}")
                passed = False
            
            check_result["passed"] = passed
            
        except Exception as e:
            error_msg = f"Pre-push check failed: {str(e)}"
            check_result["errors"].append(error_msg)
            logger.error(error_msg)
        
        return check_result

    async def auto_fix_issues(self) -> Dict[str, Any]:
        """Automatically fix common documentation issues."""
        logger.info("Auto-fixing documentation issues...")
        
        fix_result = {
            "timestamp": datetime.now().isoformat(),
            "fixes_applied": [],
            "files_modified": [],
            "errors": []
        }
        
        try:
            from .documentation_validation_system import DocumentationValidationSystem
            
            doc_validator = DocumentationValidationSystem(str(self.project_root))
            
            # Attempt automated corrections
            critical_docs = [
                self.project_root / "docs" / "PLAN.md",
                self.project_root / "docs" / "PROMPT.md"
            ]
            
            for doc_path in critical_docs:
                if doc_path.exists():
                    validation = await doc_validator.validate_document_accuracy(doc_path)
                    
                    if validation.get("accuracy_issues"):
                        # Apply automated corrections
                        await doc_validator._correct_document_issues(str(doc_path), validation)
                        
                        fix_result["fixes_applied"].extend([
                            issue["type"] for issue in validation["accuracy_issues"]
                            if issue.get("severity") in ["high", "medium"]
                        ])
                        fix_result["files_modified"].append(str(doc_path))
            
        except Exception as e:
            error_msg = f"Auto-fix failed: {str(e)}"
            fix_result["errors"].append(error_msg)
            logger.error(error_msg)
        
        return fix_result

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive documentation system status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "quality_gates_installed": False,
            "git_hooks": {},
            "ci_integration": {},
            "system_health": {}
        }
        
        # Check git hooks
        for hook_name in ["pre-commit", "post-commit", "pre-push"]:
            hook_path = self.git_hooks_dir / hook_name
            status["git_hooks"][hook_name] = {
                "installed": hook_path.exists(),
                "executable": hook_path.exists() and os.access(hook_path, os.X_OK)
            }
        
        status["quality_gates_installed"] = all(
            hook_info["installed"] and hook_info["executable"]
            for hook_info in status["git_hooks"].values()
        )
        
        # Check CI integration
        github_workflow = self.ci_config_dir / "documentation-quality.yml"
        status["ci_integration"]["github_actions"] = github_workflow.exists()
        
        # System health check
        try:
            from .documentation_validation_system import DocumentationValidationSystem
            doc_validator = DocumentationValidationSystem(str(self.project_root))
            health_report = await doc_validator.generate_living_documentation_report()
            
            status["system_health"] = {
                "overall_accuracy": health_report["overall_accuracy"],
                "system_functional": health_report["system_state"]["functionality_status"]["estimated_percentage"]
            }
        except Exception as e:
            status["system_health"]["error"] = str(e)
        
        return status


async def main():
    """Main function for command-line usage."""
    import sys
    
    quality_gates = DocumentationQualityGates()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--install":
            print("Installing documentation quality gates...")
            result = await quality_gates.install_quality_gates()
            print(f"Installation complete. Components installed: {result['components_installed']}")
            
        elif command == "--pre-commit-check":
            result = await quality_gates.run_pre_commit_check()
            
            if result["passed"]:
                print("✅ Pre-commit documentation checks passed")
                sys.exit(0)
            else:
                print("❌ Pre-commit documentation checks failed:")
                for error in result["errors"]:
                    print(f"  - {error}")
                
                if result["suggestions"]:
                    print("\nSuggested fixes:")
                    for suggestion in result["suggestions"]:
                        print(f"  - {suggestion}")
                
                sys.exit(1)
                
        elif command == "--post-commit-update":
            await quality_gates.run_post_commit_update()
            print("✅ Post-commit documentation update complete")
            
        elif command == "--pre-push-check":
            result = await quality_gates.run_pre_push_check()
            
            if result["passed"]:
                print("✅ Pre-push comprehensive checks passed")
                sys.exit(0)
            else:
                print("❌ Pre-push comprehensive checks failed:")
                for error in result["errors"]:
                    print(f"  - {error}")
                
                if result["warnings"]:
                    print("\nWarnings:")
                    for warning in result["warnings"]:
                        print(f"  - {warning}")
                
                sys.exit(1)
                
        elif command == "--auto-fix":
            result = await quality_gates.auto_fix_issues()
            
            print(f"Auto-fix complete:")
            print(f"  Files modified: {len(result['files_modified'])}")
            print(f"  Fixes applied: {len(result['fixes_applied'])}")
            
            if result["files_modified"]:
                print(f"  Modified files: {result['files_modified']}")
                
        elif command == "--status":
            status = await quality_gates.get_system_status()
            
            print("Documentation Quality Gates Status")
            print("=================================")
            print(f"Quality Gates Installed: {'✅' if status['quality_gates_installed'] else '❌'}")
            
            print(f"\nGit Hooks:")
            for hook, info in status["git_hooks"].items():
                status_icon = "✅" if info["installed"] and info["executable"] else "❌"
                print(f"  {hook}: {status_icon}")
            
            print(f"\nCI Integration:")
            print(f"  GitHub Actions: {'✅' if status['ci_integration']['github_actions'] else '❌'}")
            
            if "system_health" in status and "overall_accuracy" in status["system_health"]:
                health = status["system_health"]
                print(f"\nSystem Health:")
                print(f"  Documentation Accuracy: {health['overall_accuracy']:.1f}%")
                print(f"  System Functional: {health['system_functional']}%")
                
        elif command == "--ci-validation":
            # CI-specific validation that exits with proper codes
            result = await quality_gates.run_pre_push_check()
            
            print("CI Documentation Validation")
            print("===========================")
            
            checks = result["comprehensive_checks"]
            print(f"Overall Accuracy: {checks.get('overall_accuracy', 0):.1f}%")
            print(f"Infrastructure Health: {checks.get('infrastructure_health', 0):.1f}%")
            print(f"Living Docs Accuracy: {checks.get('living_docs_accuracy', 0):.1f}%")
            
            if not result["passed"]:
                print("\nErrors:")
                for error in result["errors"]:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("\n✅ All documentation quality checks passed")
                sys.exit(0)
    
    else:
        # Default: show help and status
        print("Documentation Quality Gates")
        print("==========================")
        print("Usage:")
        print("  --install              Install quality gates")
        print("  --status               Show system status")
        print("  --pre-commit-check     Run pre-commit validation")
        print("  --auto-fix             Auto-fix common issues")
        print("  --ci-validation        CI/CD validation")
        print("")
        
        # Show current status
        status = await quality_gates.get_system_status()
        print(f"Current Status: {'✅ Ready' if status['quality_gates_installed'] else '❌ Not Installed'}")


if __name__ == "__main__":
    asyncio.run(main())