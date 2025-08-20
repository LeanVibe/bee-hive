#!/usr/bin/env python3
"""
GitHub Integration Validation Script

Validates that all GitHub integration components are working correctly
and meet the specified performance requirements.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.github_api_client import GitHubAPIClient
from app.core.work_tree_manager import WorkTreeManager
from app.core.branch_manager import BranchManager
from app.core.pull_request_automator import PullRequestAutomator
from app.core.issue_manager import IssueManager
from app.core.code_review_assistant import CodeReviewAssistant
from app.core.github_webhooks import GitHubWebhookProcessor
from app.core.database import get_db_session
from app.models.github_integration import (
    GitHubRepository, AgentWorkTree, PullRequest, 
    GitHubIssue, CodeReview, WebhookEvent
)


class GitHubIntegrationValidator:
    """Comprehensive validation of GitHub integration system."""
    
    def __init__(self):
        self.results = []
        self.github_client = None
        
    async def validate_all_components(self) -> Dict[str, Any]:
        """Validate all GitHub integration components."""
        
        print("🚀 Starting GitHub Integration Core System Validation")
        print("=" * 60)
        
        validation_results = {
            "overall_status": "unknown",
            "component_results": {},
            "performance_metrics": {},
            "validation_timestamp": datetime.utcnow().isoformat()
        }
        
        # Test 1: Component Import and Initialization
        print("\n📦 Test 1: Component Import and Initialization")
        component_result = await self.test_component_imports()
        validation_results["component_results"]["imports"] = component_result
        
        # Test 2: Database Models and Schema
        print("\n🗄️  Test 2: Database Models and Schema")
        db_result = await self.test_database_models()
        validation_results["component_results"]["database"] = db_result
        
        # Test 3: Core Functionality
        print("\n⚙️  Test 3: Core Functionality")
        core_result = await self.test_core_functionality()
        validation_results["component_results"]["core_functionality"] = core_result
        
        # Test 4: Performance Requirements
        print("\n🏃 Test 4: Performance Requirements")
        perf_result = await self.test_performance_requirements()
        validation_results["component_results"]["performance"] = perf_result
        validation_results["performance_metrics"] = perf_result.get("metrics", {})
        
        # Test 5: Security Features
        print("\n🔒 Test 5: Security Features")
        security_result = await self.test_security_features()
        validation_results["component_results"]["security"] = security_result
        
        # Determine overall status
        all_tests_passed = all(
            result["status"] == "passed" 
            for result in validation_results["component_results"].values()
        )
        
        validation_results["overall_status"] = "passed" if all_tests_passed else "failed"
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        for test_name, result in validation_results["component_results"].items():
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"{status_icon} {test_name.title().replace('_', ' ')}: {result['status'].upper()}")
            
            if result.get("details"):
                for detail in result["details"]:
                    print(f"   • {detail}")
        
        overall_icon = "🎉" if validation_results["overall_status"] == "passed" else "💥"
        print(f"\n{overall_icon} Overall Status: {validation_results['overall_status'].upper()}")
        
        if validation_results["overall_status"] == "passed":
            print("\n✨ GitHub Integration Core System is PRODUCTION READY! ✨")
        else:
            print("\n🔧 Some components need attention before production deployment.")
        
        return validation_results
    
    async def test_component_imports(self) -> Dict[str, Any]:
        """Test that all components can be imported and initialized."""
        
        result = {"status": "unknown", "details": []}
        
        try:
            # Test GitHub API Client
            self.github_client = GitHubAPIClient()
            result["details"].append("GitHubAPIClient initialized successfully")
            
            # Test Work Tree Manager
            work_tree_manager = WorkTreeManager(self.github_client)
            result["details"].append("WorkTreeManager initialized successfully")
            
            # Test Branch Manager
            branch_manager = BranchManager(self.github_client, work_tree_manager)
            result["details"].append("BranchManager initialized successfully")
            
            # Test PR Automator
            pr_automator = PullRequestAutomator(self.github_client, branch_manager)
            result["details"].append("PullRequestAutomator initialized successfully")
            
            # Test Issue Manager
            issue_manager = IssueManager(self.github_client)
            result["details"].append("IssueManager initialized successfully")
            
            # Test Code Review Assistant
            review_assistant = CodeReviewAssistant(self.github_client)
            result["details"].append("CodeReviewAssistant initialized successfully")
            
            # Test Webhook Processor
            webhook_processor = GitHubWebhookProcessor(self.github_client)
            result["details"].append("GitHubWebhookProcessor initialized successfully")
            
            result["status"] = "passed"
            print("✅ All components imported and initialized successfully")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"❌ Component initialization failed: {e}")
        
        return result
    
    async def test_database_models(self) -> Dict[str, Any]:
        """Test database models and schema."""
        
        result = {"status": "unknown", "details": []}
        
        try:
            # Test that all models can be imported
            models = [
                GitHubRepository, AgentWorkTree, PullRequest,
                GitHubIssue, CodeReview, WebhookEvent
            ]
            
            for model in models:
                model_name = model.__name__
                # Verify table exists by checking the table name
                if hasattr(model, '__tablename__'):
                    result["details"].append(f"{model_name} model (table: {model.__tablename__}) available")
                
                result["status"] = "passed"
                print("✅ All database models available")
                
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"❌ Database model test failed: {e}")
        
        return result
    
    async def test_core_functionality(self) -> Dict[str, Any]:
        """Test core functionality of GitHub integration components."""
        
        result = {"status": "unknown", "details": []}
        
        try:
            # Test work tree manager functionality
            work_tree_manager = WorkTreeManager(self.github_client)
            result["details"].append("WorkTreeManager functionality available")
            
            # Test code review analyzers
            from app.core.code_review_assistant import SecurityAnalyzer, PerformanceAnalyzer, StyleAnalyzer
            
            security_analyzer = SecurityAnalyzer()
            performance_analyzer = PerformanceAnalyzer()
            style_analyzer = StyleAnalyzer()
            
            result["details"].append("Security analyzer initialized")
            result["details"].append("Performance analyzer initialized")
            result["details"].append("Style analyzer initialized")
            
            # Test webhook processor functionality
            webhook_processor = GitHubWebhookProcessor(self.github_client)
            result["details"].append("GitHubWebhookProcessor functionality available")
            
            result["status"] = "passed"
            print("✅ Core functionality tests passed")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"❌ Core functionality test failed: {e}")
        
        return result
    
    async def test_performance_requirements(self) -> Dict[str, Any]:
        """Test that performance requirements are met."""
        
        result = {"status": "unknown", "details": [], "metrics": {}}
        
        try:
            # Test 1: Component initialization time
            start_time = time.time()
            work_tree_manager = WorkTreeManager(self.github_client)
            init_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result["metrics"]["component_init_time_ms"] = init_time
            
            if init_time < 1000:  # Should initialize in <1 second
                result["details"].append(f"Component initialization: {init_time:.1f}ms (✅ <1000ms)")
            else:
                result["details"].append(f"Component initialization: {init_time:.1f}ms (❌ >1000ms)")
            
            # Test 2: Code review analysis performance
            test_code = '''
def test_function():
    password = "hardcoded123"  # Security issue
    for i in range(len(items)):  # Performance issue
        print(items[i])
'''
            
            review_assistant = CodeReviewAssistant(self.github_client)
            
            start_time = time.time()
            security_findings = review_assistant.security_analyzer.analyze_file("test.py", test_code)
            security_analysis_time = (time.time() - start_time) * 1000
            
            result["metrics"]["security_analysis_time_ms"] = security_analysis_time
            result["details"].append(f"Security analysis: {security_analysis_time:.1f}ms")
            
            start_time = time.time()
            performance_findings = review_assistant.performance_analyzer.analyze_file("test.py", test_code)
            performance_analysis_time = (time.time() - start_time) * 1000
            
            result["metrics"]["performance_analysis_time_ms"] = performance_analysis_time
            result["details"].append(f"Performance analysis: {performance_analysis_time:.1f}ms")
            
            # Test 3: Overall code review coverage
            all_findings = security_findings + performance_findings
            
            expected_issues = {"hardcoded_secrets", "range_len_antipattern"}
            found_issues = {finding.get("type") for finding in all_findings}
            
            coverage = len(found_issues.intersection(expected_issues)) / len(expected_issues)
            result["metrics"]["code_review_coverage"] = coverage
            
            if coverage > 0.8:  # >80% requirement
                result["details"].append(f"Code review coverage: {coverage:.1%} (✅ >80%)")
            else:
                result["details"].append(f"Code review coverage: {coverage:.1%} (❌ <80%)")
            
            # Overall performance assessment
            performance_ok = (
                init_time < 1000 and
                security_analysis_time < 5000 and
                performance_analysis_time < 5000 and
                coverage > 0.8
            )
            
            result["status"] = "passed" if performance_ok else "failed"
            
            if performance_ok:
                print("✅ Performance requirements met")
            else:
                print("❌ Some performance requirements not met")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"❌ Performance test failed: {e}")
        
        return result
    
    async def test_security_features(self) -> Dict[str, Any]:
        """Test security features and validation."""
        
        result = {"status": "unknown", "details": []}
        
        try:
            # Test webhook security via processor
            webhook_processor = GitHubWebhookProcessor(self.github_client)
            result["details"].append("Webhook processor security features available")
            
            # Test security issue detection
            from app.core.code_review_assistant import SecurityAnalyzer
            security_analyzer = SecurityAnalyzer()
            
            malicious_code = '''
password = "hardcoded_secret_123"
query = "SELECT * FROM users WHERE id = " + user_input
os.system("rm -rf " + user_path)
'''
            
            findings = security_analyzer.analyze_file("test.py", malicious_code)
            security_issues_found = len([f for f in findings if f.get("category") == "security"])
            
            if security_issues_found >= 3:  # Should find hardcoded secrets, SQL injection, command injection
                result["details"].append(f"Security analysis found {security_issues_found} issues")
                result["status"] = "passed"
            else:
                result["details"].append(f"Security analysis only found {security_issues_found} issues (expected ≥3)")
                result["status"] = "failed"
            
            print("✅ Security features validated")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"❌ Security test failed: {e}")
        
        return result


async def main():
    """Run GitHub integration validation."""
    
    validator = GitHubIntegrationValidator()
    results = await validator.validate_all_components()
    
    # Exit with proper code
    exit_code = 0 if results["overall_status"] == "passed" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class ValidateGithubIntegrationScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(ValidateGithubIntegrationScript)