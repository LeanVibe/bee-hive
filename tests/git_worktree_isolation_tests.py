#!/usr/bin/env python3
"""
Git Worktree Isolation Testing Suite

Comprehensive testing for git worktree isolation system that ensures proper
security boundaries and prevents CLI agents from accessing files outside
their assigned worktrees.

This suite validates:
- Path restriction enforcement
- Security boundary testing
- Worktree management operations
- Concurrent access protection
- Cleanup and recovery procedures
"""

import os
import tempfile
import shutil
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
import pytest
import git
from dataclasses import dataclass
import uuid
import json
import stat
from contextlib import asynccontextmanager
import time
import threading

@dataclass
class SecurityViolationAttempt:
    """Represents a security violation attempt to test isolation."""
    attempt_type: str
    description: str
    target_path: str
    source_worktree: str
    expected_result: str  # "blocked", "allowed", "error"

@dataclass
class WorktreeTestContext:
    """Test context for worktree isolation testing."""
    test_id: str
    base_repo_path: Path
    worktrees: Dict[str, Path]
    security_attempts: List[SecurityViolationAttempt]
    cleanup_required: bool = True

class WorktreeSecurityTester:
    """Tests git worktree security and isolation."""
    
    def __init__(self):
        self.temp_base_dir = None
        self.test_contexts = []
        self.violation_log = []
    
    async def setup_test_environment(self) -> Path:
        """Setup isolated test environment with base repository."""
        self.temp_base_dir = tempfile.mkdtemp(prefix="worktree_isolation_test_")
        base_repo = Path(self.temp_base_dir) / "base_repo"
        base_repo.mkdir()
        
        # Initialize git repository
        repo = git.Repo.init(base_repo)
        
        # Create initial file structure
        test_files = {
            "README.md": "# Test Repository for Worktree Isolation",
            "src/main.py": "def main():\n    print('Hello World')",
            "src/utils.py": "def utility_function():\n    pass",
            "tests/test_main.py": "import unittest\n\nclass TestMain(unittest.TestCase):\n    pass",
            "sensitive/secrets.txt": "SECRET_KEY=very_secret_value",
            "sensitive/config.json": '{"database_url": "postgresql://secret"}',
            "docs/README.md": "# Documentation",
            ".env": "SECRET_TOKEN=supersecret123"
        }
        
        # Create directory structure and files
        for file_path, content in test_files.items():
            full_path = base_repo / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        # Add and commit files
        repo.index.add(list(test_files.keys()))
        repo.index.commit("Initial commit with test structure")
        
        return base_repo
    
    async def create_isolated_worktree(self, base_repo: Path, agent_id: str, 
                                     allowed_paths: List[str] = None,
                                     restricted_paths: List[str] = None) -> Path:
        """Create an isolated worktree for an agent with path restrictions."""
        branch_name = f"agent_{agent_id}_{uuid.uuid4().hex[:8]}"
        worktree_path = Path(self.temp_base_dir) / f"worktree_{agent_id}"
        
        repo = git.Repo(base_repo)
        
        # Create new branch
        new_branch = repo.create_head(branch_name)
        
        # Create worktree
        repo.git.worktree('add', str(worktree_path), branch_name)
        
        # Apply path restrictions (simulation for testing)
        await self._apply_path_restrictions(worktree_path, allowed_paths, restricted_paths)
        
        return worktree_path
    
    async def _apply_path_restrictions(self, worktree_path: Path, 
                                     allowed_paths: List[str] = None,
                                     restricted_paths: List[str] = None):
        """Apply path restrictions to worktree (simulation for testing)."""
        # Create .worktree_config file to track restrictions
        config = {
            "worktree_path": str(worktree_path),
            "allowed_paths": allowed_paths or [],
            "restricted_paths": restricted_paths or [],
            "creation_time": time.time()
        }
        
        config_file = worktree_path / ".worktree_config"
        config_file.write_text(json.dumps(config, indent=2))
        
        # Make restricted paths read-only (if they exist)
        if restricted_paths:
            for restricted_path in restricted_paths:
                full_path = worktree_path / restricted_path
                if full_path.exists():
                    if full_path.is_file():
                        full_path.chmod(stat.S_IRUSR | stat.S_IRGRP)
                    elif full_path.is_dir():
                        os.chmod(full_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IXUSR | stat.S_IXGRP)
    
    def validate_path_access(self, worktree_path: Path, target_path: str) -> Dict[str, Any]:
        """Validate whether a path should be accessible from the worktree."""
        config_file = worktree_path / ".worktree_config"
        
        if not config_file.exists():
            return {"allowed": True, "reason": "No restrictions configured"}
        
        try:
            config = json.loads(config_file.read_text())
            allowed_paths = config.get("allowed_paths", [])
            restricted_paths = config.get("restricted_paths", [])
            
            # Check if path is explicitly restricted
            for restricted in restricted_paths:
                if target_path.startswith(restricted):
                    return {
                        "allowed": False, 
                        "reason": f"Path '{target_path}' is in restricted path '{restricted}'"
                    }
            
            # Check if path is in allowed list (if allowed list exists)
            if allowed_paths:
                for allowed in allowed_paths:
                    if target_path.startswith(allowed):
                        return {
                            "allowed": True,
                            "reason": f"Path '{target_path}' is in allowed path '{allowed}'"
                        }
                return {
                    "allowed": False,
                    "reason": f"Path '{target_path}' not in allowed paths"
                }
            
            return {"allowed": True, "reason": "No specific restrictions"}
            
        except Exception as e:
            return {"allowed": False, "reason": f"Error validating path: {str(e)}"}
    
    async def test_path_traversal_attacks(self, worktree_path: Path) -> List[Dict[str, Any]]:
        """Test various path traversal attack attempts."""
        attack_attempts = [
            # Relative path traversals
            "../sensitive/secrets.txt",
            "../../base_repo/sensitive/secrets.txt",
            "../../../etc/passwd",
            "..\\..\\sensitive\\config.json",  # Windows-style
            
            # Absolute path attempts
            "/etc/passwd",
            "/tmp/test_file",
            str(Path(self.temp_base_dir) / "base_repo" / "sensitive" / "secrets.txt"),
            
            # Symlink attempts
            "symlink_to_secrets",
            
            # Special characters
            "file%2e%2e%2fsensitive%2fsecrets.txt",
            "file..\\sensitive\\secrets.txt",
            
            # Double encoding
            "file%252e%252e%252fsensitive%252fsecrets.txt"
        ]
        
        results = []
        
        for attack_path in attack_attempts:
            try:
                # Simulate file access attempt
                validation_result = self.validate_path_access(worktree_path, attack_path)
                
                # Attempt to read file if validation allows
                if validation_result["allowed"]:
                    try:
                        full_path = worktree_path / attack_path
                        if full_path.exists():
                            content = full_path.read_text()
                            access_result = "succeeded"
                            security_violation = True
                        else:
                            access_result = "file_not_found"
                            security_violation = False
                    except PermissionError:
                        access_result = "permission_denied"
                        security_violation = False
                    except Exception as e:
                        access_result = f"error: {str(e)}"
                        security_violation = False
                else:
                    access_result = "blocked_by_validation"
                    security_violation = False
                
                result = {
                    "attack_path": attack_path,
                    "validation_result": validation_result,
                    "access_result": access_result,
                    "security_violation": security_violation,
                    "timestamp": time.time()
                }
                
                results.append(result)
                
                if security_violation:
                    self.violation_log.append({
                        "type": "path_traversal",
                        "worktree": str(worktree_path),
                        "attack_path": attack_path,
                        "severity": "high"
                    })
            
            except Exception as e:
                results.append({
                    "attack_path": attack_path,
                    "access_result": f"test_error: {str(e)}",
                    "security_violation": False,
                    "timestamp": time.time()
                })
        
        return results
    
    async def test_symlink_attacks(self, worktree_path: Path) -> List[Dict[str, Any]]:
        """Test symlink-based security attacks."""
        results = []
        
        # Test cases for symlink attacks
        symlink_attacks = [
            {
                "name": "symlink_to_sensitive_file",
                "target": "../sensitive/secrets.txt",
                "description": "Symlink pointing to sensitive file outside allowed paths"
            },
            {
                "name": "symlink_to_root",
                "target": "/",
                "description": "Symlink pointing to root directory"
            },
            {
                "name": "symlink_to_other_worktree",
                "target": "../worktree_other_agent",
                "description": "Symlink pointing to another agent's worktree"
            },
            {
                "name": "symlink_to_temp",
                "target": "/tmp",
                "description": "Symlink pointing to temp directory"
            }
        ]
        
        for attack in symlink_attacks:
            try:
                symlink_path = worktree_path / attack["name"]
                
                # Create symlink
                try:
                    os.symlink(attack["target"], symlink_path)
                    symlink_created = True
                except OSError as e:
                    symlink_created = False
                    creation_error = str(e)
                
                if symlink_created:
                    # Test access through symlink
                    try:
                        if symlink_path.exists():
                            # Try to read through symlink
                            content = symlink_path.read_text()
                            access_result = "succeeded"
                            security_violation = True
                        else:
                            access_result = "symlink_broken"
                            security_violation = False
                    except PermissionError:
                        access_result = "permission_denied"
                        security_violation = False
                    except Exception as e:
                        access_result = f"access_error: {str(e)}"
                        security_violation = False
                    
                    # Cleanup symlink
                    try:
                        symlink_path.unlink()
                    except:
                        pass
                else:
                    access_result = f"symlink_creation_failed: {creation_error}"
                    security_violation = False
                
                result = {
                    "attack_name": attack["name"],
                    "attack_description": attack["description"],
                    "target": attack["target"],
                    "symlink_created": symlink_created,
                    "access_result": access_result,
                    "security_violation": security_violation,
                    "timestamp": time.time()
                }
                
                results.append(result)
                
                if security_violation:
                    self.violation_log.append({
                        "type": "symlink_attack",
                        "worktree": str(worktree_path),
                        "attack": attack["name"],
                        "severity": "high"
                    })
            
            except Exception as e:
                results.append({
                    "attack_name": attack["name"],
                    "access_result": f"test_error: {str(e)}",
                    "security_violation": False,
                    "timestamp": time.time()
                })
        
        return results
    
    async def test_concurrent_worktree_access(self, worktrees: Dict[str, Path]) -> Dict[str, Any]:
        """Test concurrent access to multiple worktrees."""
        results = {
            "concurrent_operations": [],
            "race_conditions": [],
            "integrity_violations": []
        }
        
        async def worker_task(agent_id: str, worktree_path: Path, operations: List[str]):
            """Simulated worker performing operations in worktree."""
            worker_results = []
            
            for operation in operations:
                try:
                    if operation == "create_file":
                        test_file = worktree_path / f"agent_{agent_id}_file.txt"
                        test_file.write_text(f"Created by agent {agent_id} at {time.time()}")
                        worker_results.append({"operation": operation, "status": "success"})
                    
                    elif operation == "modify_shared_file":
                        shared_file = worktree_path / "src" / "main.py"
                        if shared_file.exists():
                            current_content = shared_file.read_text()
                            new_content = current_content + f"\n# Modified by agent {agent_id}"
                            shared_file.write_text(new_content)
                            worker_results.append({"operation": operation, "status": "success"})
                        else:
                            worker_results.append({"operation": operation, "status": "file_not_found"})
                    
                    elif operation == "access_other_worktree":
                        # Try to access another agent's worktree
                        other_worktrees = [path for aid, path in worktrees.items() if aid != agent_id]
                        if other_worktrees:
                            other_worktree = other_worktrees[0]
                            try:
                                other_file = other_worktree / "README.md"
                                content = other_file.read_text()
                                worker_results.append({
                                    "operation": operation, 
                                    "status": "security_violation",
                                    "details": "Accessed other worktree successfully"
                                })
                            except Exception as e:
                                worker_results.append({
                                    "operation": operation,
                                    "status": "blocked",
                                    "details": str(e)
                                })
                        else:
                            worker_results.append({"operation": operation, "status": "no_other_worktrees"})
                    
                    # Add small delay to simulate work
                    await asyncio.sleep(0.1)
                
                except Exception as e:
                    worker_results.append({
                        "operation": operation,
                        "status": "error",
                        "error": str(e)
                    })
            
            return {"agent_id": agent_id, "results": worker_results}
        
        # Define operations for each agent
        operations_per_agent = {
            agent_id: ["create_file", "modify_shared_file", "access_other_worktree"]
            for agent_id in worktrees.keys()
        }
        
        # Run concurrent operations
        tasks = [
            worker_task(agent_id, worktree_path, operations_per_agent[agent_id])
            for agent_id, worktree_path in worktrees.items()
        ]
        
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in concurrent_results:
            if isinstance(result, Exception):
                results["concurrent_operations"].append({
                    "status": "error",
                    "error": str(result)
                })
            else:
                results["concurrent_operations"].append(result)
                
                # Check for security violations
                for op_result in result["results"]:
                    if op_result.get("status") == "security_violation":
                        results["integrity_violations"].append({
                            "agent_id": result["agent_id"],
                            "operation": op_result["operation"],
                            "details": op_result.get("details")
                        })
        
        return results
    
    async def test_worktree_cleanup_integrity(self, base_repo: Path, worktrees: Dict[str, Path]) -> Dict[str, Any]:
        """Test worktree cleanup and integrity verification."""
        cleanup_results = {
            "worktrees_cleaned": 0,
            "cleanup_errors": [],
            "orphaned_files": [],
            "branch_cleanup": [],
            "integrity_check": {"passed": True, "issues": []}
        }
        
        repo = git.Repo(base_repo)
        
        for agent_id, worktree_path in worktrees.items():
            try:
                # Check if worktree exists
                if worktree_path.exists():
                    # Get branch name from config
                    config_file = worktree_path / ".worktree_config"
                    branch_name = f"agent_{agent_id}_*"  # Pattern match
                    
                    # Remove worktree
                    try:
                        repo.git.worktree('remove', str(worktree_path), '--force')
                        cleanup_results["worktrees_cleaned"] += 1
                    except Exception as e:
                        cleanup_results["cleanup_errors"].append({
                            "agent_id": agent_id,
                            "error": f"Worktree removal failed: {str(e)}"
                        })
                    
                    # Check for orphaned files
                    if worktree_path.exists():
                        cleanup_results["orphaned_files"].append(str(worktree_path))
                    
                    # Clean up branch
                    try:
                        # Find and delete branches for this agent
                        for branch in repo.heads:
                            if f"agent_{agent_id}" in branch.name:
                                repo.delete_head(branch, force=True)
                                cleanup_results["branch_cleanup"].append({
                                    "agent_id": agent_id,
                                    "branch": branch.name,
                                    "status": "deleted"
                                })
                    except Exception as e:
                        cleanup_results["branch_cleanup"].append({
                            "agent_id": agent_id,
                            "error": f"Branch cleanup failed: {str(e)}"
                        })
            
            except Exception as e:
                cleanup_results["cleanup_errors"].append({
                    "agent_id": agent_id,
                    "error": f"General cleanup error: {str(e)}"
                })
        
        # Integrity check
        try:
            # Check repository status
            if repo.is_dirty():
                cleanup_results["integrity_check"]["issues"].append("Repository has uncommitted changes")
                cleanup_results["integrity_check"]["passed"] = False
            
            # Check for remaining worktrees
            try:
                worktree_list = repo.git.worktree('list').split('\n')
                if len(worktree_list) > 1:  # More than just the main worktree
                    cleanup_results["integrity_check"]["issues"].append(f"Found {len(worktree_list)-1} remaining worktrees")
                    cleanup_results["integrity_check"]["passed"] = False
            except Exception as e:
                cleanup_results["integrity_check"]["issues"].append(f"Could not check worktree list: {str(e)}")
        
        except Exception as e:
            cleanup_results["integrity_check"]["issues"].append(f"Integrity check failed: {str(e)}")
            cleanup_results["integrity_check"]["passed"] = False
        
        return cleanup_results
    
    async def run_comprehensive_isolation_tests(self) -> Dict[str, Any]:
        """Run comprehensive worktree isolation tests."""
        test_results = {
            "test_suite": "Git Worktree Isolation",
            "start_time": time.time(),
            "tests_executed": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "security_violations": 0,
            "detailed_results": {}
        }
        
        try:
            # Setup test environment
            base_repo = await self.setup_test_environment()
            test_results["base_repo"] = str(base_repo)
            
            # Create test worktrees with different restriction levels
            worktrees = {}
            
            # Agent 1: Full access (no restrictions)
            worktrees["agent_full"] = await self.create_isolated_worktree(
                base_repo, "agent_full"
            )
            
            # Agent 2: Limited access (src directory only)
            worktrees["agent_limited"] = await self.create_isolated_worktree(
                base_repo, "agent_limited", 
                allowed_paths=["src/", "tests/"]
            )
            
            # Agent 3: Restricted access (cannot access sensitive/)
            worktrees["agent_restricted"] = await self.create_isolated_worktree(
                base_repo, "agent_restricted",
                restricted_paths=["sensitive/", ".env"]
            )
            
            test_results["worktrees_created"] = len(worktrees)
            
            # Test 1: Path traversal attacks
            print("🔍 Testing path traversal attacks...")
            for agent_id, worktree_path in worktrees.items():
                traversal_results = await self.test_path_traversal_attacks(worktree_path)
                test_results["detailed_results"][f"path_traversal_{agent_id}"] = traversal_results
                
                violations = sum(1 for r in traversal_results if r.get("security_violation", False))
                test_results["security_violations"] += violations
                
                test_results["tests_executed"] += 1
                if violations == 0:
                    test_results["tests_passed"] += 1
                else:
                    test_results["tests_failed"] += 1
            
            # Test 2: Symlink attacks
            print("🔗 Testing symlink attacks...")
            for agent_id, worktree_path in worktrees.items():
                symlink_results = await self.test_symlink_attacks(worktree_path)
                test_results["detailed_results"][f"symlink_attacks_{agent_id}"] = symlink_results
                
                violations = sum(1 for r in symlink_results if r.get("security_violation", False))
                test_results["security_violations"] += violations
                
                test_results["tests_executed"] += 1
                if violations == 0:
                    test_results["tests_passed"] += 1
                else:
                    test_results["tests_failed"] += 1
            
            # Test 3: Concurrent access
            print("⚡ Testing concurrent access...")
            concurrent_results = await self.test_concurrent_worktree_access(worktrees)
            test_results["detailed_results"]["concurrent_access"] = concurrent_results
            
            violations = len(concurrent_results.get("integrity_violations", []))
            test_results["security_violations"] += violations
            
            test_results["tests_executed"] += 1
            if violations == 0:
                test_results["tests_passed"] += 1
            else:
                test_results["tests_failed"] += 1
            
            # Test 4: Cleanup integrity
            print("🧹 Testing cleanup integrity...")
            cleanup_results = await self.test_worktree_cleanup_integrity(base_repo, worktrees)
            test_results["detailed_results"]["cleanup_integrity"] = cleanup_results
            
            test_results["tests_executed"] += 1
            if cleanup_results["integrity_check"]["passed"]:
                test_results["tests_passed"] += 1
            else:
                test_results["tests_failed"] += 1
        
        except Exception as e:
            test_results["fatal_error"] = str(e)
            test_results["tests_failed"] = test_results["tests_executed"]
        
        finally:
            test_results["end_time"] = time.time()
            test_results["duration"] = test_results["end_time"] - test_results["start_time"]
            test_results["violation_log"] = self.violation_log
            
            # Cleanup test environment
            if self.temp_base_dir and Path(self.temp_base_dir).exists():
                shutil.rmtree(self.temp_base_dir, ignore_errors=True)
        
        return test_results
    
    def generate_security_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a security report from test results."""
        report = []
        report.append("🔒 GIT WORKTREE ISOLATION SECURITY REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        report.append("📊 SUMMARY")
        report.append("-" * 20)
        report.append(f"Tests Executed: {test_results['tests_executed']}")
        report.append(f"Tests Passed: {test_results['tests_passed']}")
        report.append(f"Tests Failed: {test_results['tests_failed']}")
        report.append(f"Security Violations: {test_results['security_violations']}")
        report.append(f"Test Duration: {test_results['duration']:.2f}s")
        report.append("")
        
        # Security Status
        if test_results['security_violations'] == 0:
            report.append("✅ SECURITY STATUS: SECURE")
            report.append("No security violations detected.")
        else:
            report.append("❌ SECURITY STATUS: VIOLATIONS DETECTED")
            report.append(f"Found {test_results['security_violations']} security violations.")
        
        report.append("")
        
        # Detailed findings
        if test_results['violation_log']:
            report.append("🚨 SECURITY VIOLATIONS")
            report.append("-" * 30)
            for violation in test_results['violation_log']:
                report.append(f"- Type: {violation['type']}")
                report.append(f"  Severity: {violation['severity']}")
                report.append(f"  Worktree: {violation['worktree']}")
                if 'attack_path' in violation:
                    report.append(f"  Attack Path: {violation['attack_path']}")
                report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 25)
        if test_results['security_violations'] > 0:
            report.append("1. Implement stronger path validation")
            report.append("2. Add symlink detection and blocking")
            report.append("3. Enhance file system permissions")
            report.append("4. Add audit logging for access attempts")
        else:
            report.append("1. Continue regular security testing")
            report.append("2. Monitor for new attack vectors")
            report.append("3. Review and update isolation policies")
        
        return "\n".join(report)

# Pytest integration
@pytest.fixture
async def worktree_tester():
    """Pytest fixture for worktree isolation testing."""
    tester = WorktreeSecurityTester()
    yield tester
    # Cleanup is handled by the tester itself

@pytest.mark.asyncio
async def test_path_traversal_protection(worktree_tester):
    """Test path traversal attack protection."""
    base_repo = await worktree_tester.setup_test_environment()
    worktree = await worktree_tester.create_isolated_worktree(
        base_repo, "test_agent", restricted_paths=["sensitive/"]
    )
    
    results = await worktree_tester.test_path_traversal_attacks(worktree)
    
    # Should block access to sensitive paths
    violations = [r for r in results if r.get("security_violation", False)]
    assert len(violations) == 0, f"Found {len(violations)} security violations"

@pytest.mark.asyncio 
async def test_symlink_protection(worktree_tester):
    """Test symlink attack protection."""
    base_repo = await worktree_tester.setup_test_environment()
    worktree = await worktree_tester.create_isolated_worktree(
        base_repo, "test_agent", restricted_paths=["sensitive/"]
    )
    
    results = await worktree_tester.test_symlink_attacks(worktree)
    
    # Should prevent symlink-based attacks
    violations = [r for r in results if r.get("security_violation", False)]
    assert len(violations) == 0, f"Found {len(violations)} symlink vulnerabilities"

if __name__ == "__main__":
    async def main():
        """Run worktree isolation tests standalone."""
        print("🔒 Git Worktree Isolation Testing Suite")
        print("=" * 50)
        
        tester = WorktreeSecurityTester()
        
        try:
            results = await tester.run_comprehensive_isolation_tests()
            
            # Generate and display report
            report = tester.generate_security_report(results)
            print(report)
            
            # Save detailed results
            with open('worktree_isolation_test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📄 Detailed results saved to: worktree_isolation_test_results.json")
            
            # Exit with appropriate code
            exit_code = 0 if results['security_violations'] == 0 else 1
            return exit_code
            
        except Exception as e:
            print(f"❌ Test suite error: {str(e)}")
            return 1
    
    exit_code = asyncio.run(main())
    exit(exit_code)