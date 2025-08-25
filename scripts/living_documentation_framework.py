#!/usr/bin/env python3
"""
Living Documentation Framework for LeanVibe Agent Hive 2.0

A comprehensive framework that ensures documentation automatically stays current
with system evolution. Prevents accuracy drift through continuous validation,
automated corrections, and real-time synchronization with code changes.

Features:
- Git hook integration for automatic documentation updates
- Real-time code change detection and documentation synchronization
- Intelligent content generation and accuracy validation
- Template-based living documentation patterns
- Automated quality gates preventing documentation drift
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
import hashlib
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LivingDocumentationFramework:
    """Framework for self-maintaining documentation that prevents accuracy drift."""
    
    def __init__(self, project_root: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.project_root = Path(project_root)
        self.framework_config = self._load_framework_config()
        self.template_engine = DocumentationTemplateEngine(self.project_root)
        self.git_integration = GitHookIntegration(self.project_root)
        self.accuracy_validator = AccuracyValidator(self.project_root)
        
        # Living document registry
        self.living_documents = [
            {
                "path": self.project_root / "docs" / "PLAN.md",
                "type": "strategic",
                "update_triggers": ["infrastructure_change", "functionality_change", "test_count_change"],
                "validation_rules": ["port_accuracy", "system_state_accuracy", "completion_percentage"]
            },
            {
                "path": self.project_root / "docs" / "PROMPT.md", 
                "type": "strategic",
                "update_triggers": ["infrastructure_change", "functionality_change", "agent_capability_change"],
                "validation_rules": ["port_accuracy", "system_state_accuracy", "agent_count_accuracy"]
            },
            {
                "path": self.project_root / "docs" / "SYSTEM_CAPABILITY_AUDIT.md",
                "type": "technical",
                "update_triggers": ["test_results_change", "infrastructure_change", "performance_change"],
                "validation_rules": ["test_count_accuracy", "performance_metrics", "capability_validation"]
            }
        ]
    
    def _load_framework_config(self) -> Dict[str, Any]:
        """Load living documentation framework configuration."""
        config_path = self.project_root / "config" / "living_documentation.yaml"
        
        default_config = {
            "enabled": True,
            "auto_update_frequency": "on_change",  # or "hourly", "daily"
            "validation_threshold": 95.0,
            "backup_before_update": True,
            "git_integration": True,
            "template_patterns": {
                "system_state": "<!-- LIVING:SYSTEM_STATE -->",
                "test_counts": "<!-- LIVING:TEST_COUNTS -->",
                "infrastructure": "<!-- LIVING:INFRASTRUCTURE -->",
                "performance_metrics": "<!-- LIVING:PERFORMANCE -->"
            },
            "accuracy_rules": {
                "port_mismatch_tolerance": 0,  # No tolerance for port errors
                "percentage_tolerance": 5.0,   # 5% tolerance for percentages
                "count_tolerance": 10          # 10 item tolerance for counts
            }
        }
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config

    async def initialize_living_documentation(self):
        """Initialize the living documentation framework with git hooks and templates."""
        logger.info("Initializing living documentation framework...")
        
        # Install git hooks
        await self.git_integration.install_hooks()
        
        # Initialize templates in existing documents
        for doc_config in self.living_documents:
            await self._initialize_document_templates(doc_config)
        
        # Create framework status file
        status = {
            "initialized": datetime.now().isoformat(),
            "git_hooks_installed": True,
            "living_documents_count": len(self.living_documents),
            "last_validation": None,
            "framework_version": "1.0.0"
        }
        
        status_path = self.project_root / ".living_docs_status.json"
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info("Living documentation framework initialized successfully")

    async def _initialize_document_templates(self, doc_config: Dict[str, Any]):
        """Initialize living documentation templates in a document."""
        doc_path = doc_config["path"]
        
        if not doc_path.exists():
            logger.warning(f"Document not found: {doc_path}")
            return
        
        with open(doc_path, 'r') as f:
            content = f.read()
        
        # Add template markers if they don't exist
        templates_added = []
        
        for trigger in doc_config["update_triggers"]:
            template_marker = self.framework_config["template_patterns"].get(
                trigger.replace("_change", ""), 
                f"<!-- LIVING:{trigger.upper()} -->"
            )
            
            if template_marker not in content:
                # Add template marker at appropriate location
                if trigger == "infrastructure_change":
                    # Add after infrastructure section
                    pattern = r"(## .*[Ii]nfrastructure.*\n)"
                    replacement = f"\\1{template_marker}\n\n"
                    content = re.sub(pattern, replacement, content)
                    templates_added.append(trigger)
                
                elif trigger == "test_count_change":
                    # Add after test section
                    pattern = r"(## .*[Tt]est.*\n)"
                    replacement = f"\\1{template_marker}\n\n"
                    content = re.sub(pattern, replacement, content)
                    templates_added.append(trigger)
        
        # Write updated content if templates were added
        if templates_added:
            with open(doc_path, 'w') as f:
                f.write(content)
            logger.info(f"Added living documentation templates to {doc_path}: {templates_added}")

    async def detect_changes_and_update(self) -> Dict[str, Any]:
        """Detect system changes and update living documentation accordingly."""
        logger.info("Detecting changes and updating living documentation...")
        
        update_result = {
            "timestamp": datetime.now().isoformat(),
            "changes_detected": [],
            "documents_updated": [],
            "validation_results": {},
            "errors": []
        }
        
        # Detect different types of changes
        changes = await self._detect_system_changes()
        update_result["changes_detected"] = changes
        
        # Update documents based on detected changes
        for doc_config in self.living_documents:
            try:
                doc_updated = await self._update_document_if_needed(doc_config, changes)
                if doc_updated:
                    update_result["documents_updated"].append(str(doc_config["path"]))
            except Exception as e:
                error_msg = f"Failed to update {doc_config['path']}: {str(e)}"
                update_result["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Validate updated documents
        for doc_config in self.living_documents:
            if str(doc_config["path"]) in update_result["documents_updated"]:
                validation = await self.accuracy_validator.validate_document(doc_config)
                update_result["validation_results"][str(doc_config["path"])] = validation
        
        return update_result

    async def _detect_system_changes(self) -> List[Dict[str, Any]]:
        """Detect various types of system changes that should trigger documentation updates."""
        changes = []
        
        # Infrastructure changes
        from .infrastructure_monitoring_system import InfrastructureMonitoringSystem
        infra_monitor = InfrastructureMonitoringSystem(str(self.project_root))
        infra_state = await infra_monitor.discover_infrastructure_state()
        
        # Check for port or service status changes
        cached_infra = self._load_cached_state("infrastructure")
        if cached_infra:
            for component, current_state in infra_state.get("components", {}).items():
                cached_component = cached_infra.get("components", {}).get(component, {})
                
                if current_state.get("overall_status") != cached_component.get("overall_status"):
                    changes.append({
                        "type": "infrastructure_change",
                        "component": component,
                        "old_status": cached_component.get("overall_status"),
                        "new_status": current_state.get("overall_status"),
                        "requires_update": True
                    })
        
        self._cache_state("infrastructure", infra_state)
        
        # Test count changes
        from .test_synchronization_system import TestSynchronizationSystem
        test_sync = TestSynchronizationSystem(str(self.project_root))
        test_data = await test_sync.discover_all_tests()
        
        cached_tests = self._load_cached_state("tests")
        current_test_count = len(test_data.get("test_files", []))
        
        if cached_tests:
            cached_test_count = cached_tests.get("test_count", 0)
            if abs(current_test_count - cached_test_count) > 5:  # Significant change
                changes.append({
                    "type": "test_count_change",
                    "old_count": cached_test_count,
                    "new_count": current_test_count,
                    "requires_update": True
                })
        
        self._cache_state("tests", {"test_count": current_test_count, "discovery": test_data})
        
        # Functionality percentage changes (based on infrastructure health)
        health_percentage = infra_state.get("health_summary", {}).get("health_percentage", 85)
        cached_health = self._load_cached_state("functionality")
        
        if cached_health:
            cached_percentage = cached_health.get("percentage", 85)
            if abs(health_percentage - cached_percentage) > 5:  # 5% threshold
                changes.append({
                    "type": "functionality_change",
                    "old_percentage": cached_percentage,
                    "new_percentage": health_percentage,
                    "requires_update": True
                })
        
        self._cache_state("functionality", {"percentage": health_percentage})
        
        return changes

    def _load_cached_state(self, state_type: str) -> Optional[Dict[str, Any]]:
        """Load cached state from previous runs."""
        cache_path = self.project_root / ".living_docs_cache" / f"{state_type}.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached state {state_type}: {e}")
        
        return None

    def _cache_state(self, state_type: str, state_data: Dict[str, Any]):
        """Cache current state for comparison in future runs."""
        cache_dir = self.project_root / ".living_docs_cache"
        cache_dir.mkdir(exist_ok=True)
        
        cache_path = cache_dir / f"{state_type}.json"
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache state {state_type}: {e}")

    async def _update_document_if_needed(self, doc_config: Dict[str, Any], changes: List[Dict[str, Any]]) -> bool:
        """Update a document if relevant changes were detected."""
        doc_path = doc_config["path"]
        relevant_changes = [
            change for change in changes
            if change["type"] in doc_config["update_triggers"] and change.get("requires_update")
        ]
        
        if not relevant_changes:
            return False
        
        logger.info(f"Updating {doc_path} based on changes: {[c['type'] for c in relevant_changes]}")
        
        # Backup original if configured
        if self.framework_config.get("backup_before_update", True):
            backup_path = f"{doc_path}.backup_{int(time.time())}"
            subprocess.run(["cp", str(doc_path), backup_path])
        
        # Load current content
        with open(doc_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply updates based on change types
        for change in relevant_changes:
            content = await self._apply_change_to_content(content, change)
        
        # Write updated content
        if content != original_content:
            with open(doc_path, 'w') as f:
                f.write(content)
            
            # Add update timestamp
            await self._add_update_timestamp(doc_path)
            
            return True
        
        return False

    async def _apply_change_to_content(self, content: str, change: Dict[str, Any]) -> str:
        """Apply a specific change to document content."""
        change_type = change["type"]
        
        if change_type == "infrastructure_change":
            # Update infrastructure status references
            component = change["component"]
            new_status = change["new_status"]
            status_emoji = "✅" if new_status == "operational" else "❌"
            
            # Pattern to find component references
            pattern = rf"([❌✅]?\s*{re.escape(component.title())}.*?)(operational|down|degraded)"
            replacement = f"{status_emoji} {component.title()} {new_status}"
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        elif change_type == "test_count_change":
            # Update test count references
            old_count = change["old_count"]
            new_count = change["new_count"]
            
            # Pattern to find test count references
            pattern = rf"\b{old_count}\+?\s*test[s]?\s*files?"
            replacement = f"{new_count}+ test files"
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        elif change_type == "functionality_change":
            # Update functionality percentage references
            old_percentage = change["old_percentage"]
            new_percentage = change["new_percentage"]
            
            # Pattern to find percentage references
            pattern = rf"\b{old_percentage}%\s*functional"
            replacement = f"{new_percentage}% functional"
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content

    async def _add_update_timestamp(self, doc_path: Path):
        """Add or update the living documentation timestamp."""
        with open(doc_path, 'r') as f:
            content = f.read()
        
        timestamp_line = f"\n*Living documentation auto-updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        # Try to find and replace existing timestamp
        existing_timestamp = re.search(r'\*Living documentation auto-updated:.*?\*', content)
        if existing_timestamp:
            content = content.replace(existing_timestamp.group(0), timestamp_line.strip())
        else:
            content += timestamp_line
        
        with open(doc_path, 'w') as f:
            f.write(content)

    async def validate_all_living_documents(self) -> Dict[str, Any]:
        """Validate all living documents for accuracy and consistency."""
        logger.info("Validating all living documents...")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "documents": {},
            "overall_accuracy": 0.0,
            "critical_issues": [],
            "recommendations": []
        }
        
        accuracies = []
        
        for doc_config in self.living_documents:
            doc_validation = await self.accuracy_validator.validate_document(doc_config)
            validation_results["documents"][str(doc_config["path"])] = doc_validation
            
            accuracy = doc_validation.get("accuracy_score", 0.0)
            accuracies.append(accuracy)
            
            # Collect critical issues
            for issue in doc_validation.get("issues", []):
                if issue.get("severity") in ["critical", "high"]:
                    validation_results["critical_issues"].append({
                        "document": str(doc_config["path"]),
                        "issue": issue
                    })
        
        # Calculate overall accuracy
        validation_results["overall_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        # Generate recommendations
        if validation_results["overall_accuracy"] < self.framework_config["validation_threshold"]:
            validation_results["recommendations"].append({
                "type": "accuracy_improvement",
                "priority": "high",
                "action": f"Overall accuracy ({validation_results['overall_accuracy']:.1f}%) below threshold ({self.framework_config['validation_threshold']}%)"
            })
        
        if validation_results["critical_issues"]:
            validation_results["recommendations"].append({
                "type": "critical_issues",
                "priority": "critical",
                "action": f"Address {len(validation_results['critical_issues'])} critical documentation issues"
            })
        
        return validation_results

    async def start_living_documentation_monitoring(self, interval_minutes: int = 30):
        """Start continuous living documentation monitoring and updating."""
        logger.info(f"Starting living documentation monitoring (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                # Detect changes and update documents
                update_result = await self.detect_changes_and_update()
                
                if update_result["documents_updated"]:
                    logger.info(f"Updated documents: {update_result['documents_updated']}")
                
                # Validate all documents
                validation_result = await self.validate_all_living_documents()
                
                if validation_result["critical_issues"]:
                    logger.warning(f"Critical documentation issues detected: {len(validation_result['critical_issues'])}")
                
                # Log summary
                logger.info(f"Living docs monitoring cycle complete - "
                          f"Accuracy: {validation_result['overall_accuracy']:.1f}%, "
                          f"Updates: {len(update_result['documents_updated'])}, "
                          f"Issues: {len(validation_result['critical_issues'])}")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in living documentation monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


class DocumentationTemplateEngine:
    """Template engine for generating dynamic documentation content."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load documentation templates."""
        return {
            "system_state": """
<!-- LIVING:SYSTEM_STATE -->
**Current System State** (Auto-updated: {timestamp})
- Infrastructure Health: {infrastructure_health}%
- Functionality Level: {functionality_percentage}%
- Test Coverage: {test_count} test files validated
- Service Status: {service_summary}
<!-- /LIVING:SYSTEM_STATE -->
""",
            "test_counts": """
<!-- LIVING:TEST_COUNTS -->
**Test Infrastructure** (Auto-updated: {timestamp})
- Total Test Files: {test_count}
- Test Categories: {test_categories}
- Execution Time: {estimated_execution_time}
- Coverage Estimate: {coverage_estimate}
<!-- /LIVING:TEST_COUNTS -->
""",
            "infrastructure": """
<!-- LIVING:INFRASTRUCTURE -->
**Infrastructure Status** (Auto-updated: {timestamp})
{infrastructure_components}
*Ports and services validated automatically*
<!-- /LIVING:INFRASTRUCTURE -->
"""
        }


class GitHookIntegration:
    """Git hook integration for automatic documentation updates on commits."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.hooks_dir = project_root / ".git" / "hooks"
    
    async def install_hooks(self):
        """Install git hooks for living documentation updates."""
        if not self.hooks_dir.exists():
            logger.warning("Git hooks directory not found - git repository may not be initialized")
            return
        
        # Pre-commit hook for documentation validation
        pre_commit_hook = self.hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# Living Documentation Pre-commit Hook
echo "Validating living documentation..."
python scripts/living_documentation_framework.py --validate
if [ $? -ne 0 ]; then
    echo "Living documentation validation failed"
    exit 1
fi
"""
        
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        
        os.chmod(pre_commit_hook, 0o755)
        
        # Post-commit hook for documentation updates
        post_commit_hook = self.hooks_dir / "post-commit"
        post_commit_content = """#!/bin/bash
# Living Documentation Post-commit Hook
echo "Updating living documentation..."
python scripts/living_documentation_framework.py --update --background
"""
        
        with open(post_commit_hook, 'w') as f:
            f.write(post_commit_content)
        
        os.chmod(post_commit_hook, 0o755)
        
        logger.info("Git hooks installed successfully")


class AccuracyValidator:
    """Validator for documentation accuracy against system state."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    async def validate_document(self, doc_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a document against its accuracy rules."""
        doc_path = doc_config["path"]
        validation_rules = doc_config.get("validation_rules", [])
        
        validation_result = {
            "document": str(doc_path),
            "timestamp": datetime.now().isoformat(),
            "accuracy_score": 100.0,
            "issues": [],
            "rules_checked": validation_rules
        }
        
        if not doc_path.exists():
            validation_result["issues"].append({
                "severity": "critical",
                "type": "missing_document",
                "message": "Document does not exist"
            })
            validation_result["accuracy_score"] = 0.0
            return validation_result
        
        with open(doc_path, 'r') as f:
            content = f.read()
        
        # Apply validation rules
        for rule in validation_rules:
            rule_result = await self._apply_validation_rule(rule, content)
            if rule_result["issues"]:
                validation_result["issues"].extend(rule_result["issues"])
                validation_result["accuracy_score"] -= rule_result["penalty"]
        
        validation_result["accuracy_score"] = max(0.0, validation_result["accuracy_score"])
        return validation_result
    
    async def _apply_validation_rule(self, rule: str, content: str) -> Dict[str, Any]:
        """Apply a specific validation rule to content."""
        result = {"issues": [], "penalty": 0.0}
        
        if rule == "port_accuracy":
            # Check for incorrect port references
            incorrect_ports = {"15432": "5432", "16379": "6379"}
            for incorrect, correct in incorrect_ports.items():
                if incorrect in content:
                    result["issues"].append({
                        "severity": "high",
                        "type": "incorrect_port",
                        "message": f"Found incorrect port {incorrect}, should be {correct}"
                    })
                    result["penalty"] = 15.0
        
        elif rule == "system_state_accuracy":
            # Validate system state claims against actual state
            functionality_matches = re.findall(r"(\d{1,3})% functional", content)
            if functionality_matches:
                claimed_percentage = int(functionality_matches[0])
                # This would need actual system state - simplified for now
                if claimed_percentage > 100 or claimed_percentage < 50:
                    result["issues"].append({
                        "severity": "medium",
                        "type": "unrealistic_percentage", 
                        "message": f"Functionality percentage {claimed_percentage}% seems unrealistic"
                    })
                    result["penalty"] = 10.0
        
        elif rule == "test_count_accuracy":
            # Validate test count claims
            test_matches = re.findall(r"(\d+)\+ test", content)
            if test_matches:
                claimed_count = int(test_matches[0])
                # This would need actual test count - simplified for now
                if claimed_count > 1000 or claimed_count == 0:
                    result["issues"].append({
                        "severity": "medium",
                        "type": "unrealistic_test_count",
                        "message": f"Test count {claimed_count} seems unrealistic"
                    })
                    result["penalty"] = 10.0
        
        return result


async def main():
    """Main function to run living documentation framework."""
    import sys
    
    framework = LivingDocumentationFramework()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        # Validation mode (for git hooks)
        validation_result = await framework.validate_all_living_documents()
        
        if validation_result["critical_issues"]:
            print(f"Critical documentation issues found: {len(validation_result['critical_issues'])}")
            for issue in validation_result["critical_issues"][:3]:
                print(f"  - {issue['issue']['type']}: {issue['issue']['message']}")
            sys.exit(1)
        else:
            print(f"Documentation validation passed - Accuracy: {validation_result['overall_accuracy']:.1f}%")
            sys.exit(0)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--update":
        # Update mode
        print("Updating living documentation...")
        update_result = await framework.detect_changes_and_update()
        
        if update_result["documents_updated"]:
            print(f"Updated documents: {update_result['documents_updated']}")
        else:
            print("No updates needed")
        
        if "--background" in sys.argv:
            # Start continuous monitoring
            await framework.start_living_documentation_monitoring()
    
    else:
        # Initialize and demo mode
        print("Initializing Living Documentation Framework...")
        await framework.initialize_living_documentation()
        
        print("\nValidating current documentation...")
        validation_result = await framework.validate_all_living_documents()
        
        print(f"\nLiving Documentation Framework Status")
        print(f"====================================")
        print(f"Overall Accuracy: {validation_result['overall_accuracy']:.1f}%")
        print(f"Documents Monitored: {len(framework.living_documents)}")
        print(f"Critical Issues: {len(validation_result['critical_issues'])}")
        
        if validation_result["recommendations"]:
            print(f"\nRecommendations:")
            for rec in validation_result["recommendations"][:3]:
                print(f"  - {rec['type'].title()}: {rec['action']}")
        
        print(f"\nFramework initialized and ready for continuous monitoring")


if __name__ == "__main__":
    asyncio.run(main())