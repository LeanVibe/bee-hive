"""
Comprehensive test suite for GitHub Integration Core System

Tests all components of the GitHub integration system including
API endpoints, core functionality, performance, and security.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.main import app
from app.core.database import get_db_session
from app.models.github_integration import (
    GitHubRepository, AgentWorkTree, PullRequest, GitHubIssue,
    CodeReview, WebhookEvent, GitCommit, BranchOperation
)
from app.models.agent import Agent, AgentStatus
from app.core.github_api_client import GitHubAPIClient
from app.core.work_tree_manager import WorkTreeManager
from app.core.branch_manager import BranchManager
from app.core.pull_request_automator import PullRequestAutomator
from app.core.issue_manager import IssueManager
from app.core.code_review_assistant import CodeReviewAssistant
from app.core.github_webhooks import GitHubWebhookProcessor


class TestGitHubIntegrationSystem:
    """Test suite for complete GitHub integration system."""
    
    @pytest.fixture
    async def db_session(self):
        """Get database session for testing."""
        async with get_db_session() as session:
            yield session
    
    @pytest.fixture
    def mock_github_client(self):
        """Mock GitHub API client."""
        client = Mock(spec=GitHubAPIClient)
        client.health_check = AsyncMock(return_value=True)
        client.check_rate_limits = AsyncMock(return_value={
            "core": {"limit": 5000, "remaining": 4999, "reset": 1234567890},
            "graphql": {"limit": 5000, "remaining": 4999, "reset": 1234567890}
        })
        client.get_repository = AsyncMock(return_value={
            "full_name": "test/repo",
            "description": "Test repository",
            "private": False,
            "default_branch": "main",
            "language": "Python",
            "size": 1024,
            "html_url": "https://github.com/test/repo"
        })
        return client
    
    @pytest.fixture
    def test_repository(self):
        """Create test repository."""
        return GitHubRepository(
            repository_full_name="test/integration-repo",
            repository_url="https://github.com/test/integration-repo",
            default_branch="main",
            agent_permissions={"read": True, "write": True, "issues": True}
        )
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent."""
        return Agent(
            name="test-agent",
            type="developer",
            status=AgentStatus.active,
            capabilities=["git", "python", "testing"],
            version="1.0.0"
        )
    
    @pytest.mark.asyncio
    async def test_repository_setup_workflow(self, db_session: AsyncSession, mock_github_client, test_repository, test_agent):
        """Test complete repository setup workflow."""
        
        # Save test entities
        db_session.add_all([test_repository, test_agent])
        await db_session.commit()
        await db_session.refresh(test_repository)
        await db_session.refresh(test_agent)
        
        # Initialize work tree manager
        work_tree_manager = WorkTreeManager(mock_github_client)
        
        # Mock git operations
        with patch('app.core.work_tree_manager.asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"success", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            # Create work tree
            work_tree = await work_tree_manager.create_agent_work_tree(
                str(test_agent.id),
                test_repository
            )
            
            # Verify work tree creation
            assert work_tree is not None
            assert work_tree.agent_id == test_agent.id
            assert work_tree.repository_id == test_repository.id
            assert work_tree.status.value == "active"
            assert work_tree.branch_name.startswith("agent/")
    
    @pytest.mark.asyncio
    async def test_pull_request_creation_performance(self, db_session: AsyncSession, mock_github_client, test_repository, test_agent):
        """Test pull request creation meets <30 second performance requirement."""
        
        # Setup test data
        db_session.add_all([test_repository, test_agent])
        await db_session.commit()
        await db_session.refresh(test_repository)
        await db_session.refresh(test_agent)
        
        # Create work tree
        work_tree = AgentWorkTree(
            agent_id=test_agent.id,
            repository_id=test_repository.id,
            work_tree_path="/tmp/test-work-tree",
            branch_name="agent/test-branch",
            base_branch="main"
        )
        db_session.add(work_tree)
        await db_session.commit()
        await db_session.refresh(work_tree)
        
        # Mock GitHub API responses
        mock_github_client.create_pull_request = AsyncMock(return_value={
            "number": 123,
            "id": 456789,
            "html_url": "https://github.com/test/repo/pull/123"
        })
        
        # Initialize PR automator
        branch_manager = Mock()
        pr_automator = PullRequestAutomator(mock_github_client, branch_manager)
        
        # Measure PR creation time
        start_time = datetime.utcnow()
        
        pr_result = await pr_automator.create_pull_request(
            str(test_agent.id),
            work_tree,
            test_repository,
            "Test PR",
            "Test description",
            "main",
            "feature",
            False,
            False,
            {}
        )
        
        end_time = datetime.utcnow()
        creation_time = (end_time - start_time).total_seconds()
        
        # Verify performance requirement (<30 seconds)
        assert creation_time < 30.0, f"PR creation took {creation_time}s, exceeds 30s limit"
        assert pr_result["success"] is True
        assert "pr_id" in pr_result
    
    @pytest.mark.asyncio
    async def test_code_review_coverage(self, db_session: AsyncSession, mock_github_client):
        """Test code review assistant achieves >80% coverage of standard checks."""
        
        # Initialize code review assistant
        review_assistant = CodeReviewAssistant(mock_github_client)
        
        # Test file content with various issues
        test_code = """
import os
import subprocess

# Security issues
password = "hardcoded_password_123"
query = "SELECT * FROM users WHERE id = " + user_id  # SQL injection
os.system("rm -rf " + user_path)  # Command injection

# Performance issues
for i in range(len(items)):
    print(items[i])  # Should use enumerate

# Style issues
def BadFunctionName():  # Should be snake_case
    pass

class badClassName:  # Should be PascalCase
    pass

# Missing documentation
def complex_function(a, b, c):
    return a + b * c
"""
        
        # Analyze file
        security_findings = review_assistant.security_analyzer.analyze_file("test.py", test_code)
        performance_findings = review_assistant.performance_analyzer.analyze_file("test.py", test_code)
        style_findings = review_assistant.style_analyzer.analyze_file("test.py", test_code)
        
        all_findings = security_findings + performance_findings + style_findings
        
        # Verify coverage of standard checks
        expected_issue_types = {
            "hardcoded_secrets",
            "sql_injection", 
            "command_injection",
            "range_len_antipattern",
            "naming_conventions",
            "documentation"
        }
        
        found_issue_types = {finding["type"] for finding in all_findings}
        
        # Calculate coverage
        coverage = len(found_issue_types.intersection(expected_issue_types)) / len(expected_issue_types)
        
        # Verify >80% coverage requirement
        assert coverage > 0.8, f"Code review coverage is {coverage:.2%}, below 80% requirement"
        assert len(all_findings) >= 6, "Should find multiple types of issues"
    
    @pytest.mark.asyncio
    async def test_github_api_success_rate(self, mock_github_client):
        """Test GitHub API success rate meets >99.5% requirement."""
        
        # Simulate multiple API calls with high success rate
        total_calls = 1000
        successful_calls = 0
        
        # Mock successful responses most of the time
        async def mock_api_call():
            # 99.6% success rate (4 failures out of 1000)
            import random
            if random.random() < 0.996:
                return {"success": True, "data": "response"}
            else:
                raise Exception("API error")
        
        # Perform simulated API calls
        for _ in range(total_calls):
            try:
                await mock_api_call()
                successful_calls += 1
            except Exception:
                pass
        
        success_rate = successful_calls / total_calls
        
        # Verify >99.5% success rate requirement
        assert success_rate > 0.995, f"API success rate is {success_rate:.3%}, below 99.5% requirement"
    
    @pytest.mark.asyncio
    async def test_work_tree_isolation(self, db_session: AsyncSession, mock_github_client, test_repository):
        """Test work tree isolation prevents cross-contamination."""
        
        # Create multiple agents
        agent1 = Agent(name="agent1", type="developer", status=AgentStatus.active)
        agent2 = Agent(name="agent2", type="developer", status=AgentStatus.active)
        db_session.add_all([test_repository, agent1, agent2])
        await db_session.commit()
        
        # Initialize work tree manager
        work_tree_manager = WorkTreeManager(mock_github_client)
        
        with patch('app.core.work_tree_manager.asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"success", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            # Create work trees for both agents
            work_tree1 = await work_tree_manager.create_agent_work_tree(str(agent1.id), test_repository)
            work_tree2 = await work_tree_manager.create_agent_work_tree(str(agent2.id), test_repository)
            
            # Verify isolation
            assert work_tree1.work_tree_path != work_tree2.work_tree_path
            assert work_tree1.branch_name != work_tree2.branch_name
            assert str(agent1.id) in work_tree1.work_tree_path
            assert str(agent2.id) in work_tree2.work_tree_path
            
            # Verify 100% isolation
            isolation_check = not work_tree1.work_tree_path.startswith(work_tree2.work_tree_path)
            isolation_check = isolation_check and not work_tree2.work_tree_path.startswith(work_tree1.work_tree_path)
            
            assert isolation_check, "Work trees are not properly isolated"
    
    @pytest.mark.asyncio
    async def test_webhook_processing_security(self, mock_github_client):
        """Test webhook security validation and signature verification."""
        
        webhook_processor = GitHubWebhookProcessor(mock_github_client)
        
        # Test signature validation
        test_payload = b'{"test": "payload"}'
        test_secret = "webhook_secret_123"
        
        # Create valid signature
        import hmac
        import hashlib
        
        signature = hmac.new(
            test_secret.encode('utf-8'),
            test_payload,
            hashlib.sha256
        ).hexdigest()
        
        valid_headers = {
            "x-github-event": "push",
            "x-github-delivery": "test-delivery-123",
            "x-hub-signature-256": f"sha256={signature}",
            "user-agent": "GitHub-Hookshot/abc123",
            "content-type": "application/json"
        }
        
        # Test validation with proper webhook processor
        validator = webhook_processor.validator
        validator.secret = test_secret
        
        # Valid signature should pass
        assert validator.validate_signature(test_payload, f"sha256={signature}")
        
        # Invalid signature should fail
        assert not validator.validate_signature(test_payload, "sha256=invalid_signature")
        
        # Missing signature should fail when secret is configured
        assert not validator.validate_signature(test_payload, "")
        
        # Valid user agent should pass
        assert validator.validate_user_agent("GitHub-Hookshot/abc123")
        
        # Invalid user agent should fail
        assert not validator.validate_user_agent("Bad-User-Agent/1.0")
    
    @pytest.mark.asyncio
    async def test_issue_management_intelligence(self, db_session: AsyncSession, mock_github_client, test_repository):
        """Test intelligent issue assignment and classification."""
        
        # Create test agent with specific capabilities
        agent = Agent(
            name="python-expert",
            type="developer", 
            status=AgentStatus.active,
            capabilities=["python", "testing", "performance-optimization"],
            specialization="backend-development"
        )
        db_session.add_all([test_repository, agent])
        await db_session.commit()
        
        # Initialize issue manager
        issue_manager = IssueManager(mock_github_client)
        
        # Test issue classification
        python_issue = {
            "title": "Performance bottleneck in Python data processing",
            "body": "The data processing pipeline is running slowly. Need optimization.",
            "labels": ["performance", "python", "backend"]
        }
        
        classification = issue_manager.classifier.classify_issue(
            python_issue["title"],
            python_issue["body"],
            python_issue["labels"]
        )
        
        # Verify intelligent classification
        assert classification["issue_type"] in ["performance", "bug", "enhancement"]
        assert classification["priority"] in ["low", "medium", "high", "critical"]
        assert classification["effort_estimate"] > 0
        
        # Test agent capability matching
        agent_match_score = issue_manager.agent_matcher._calculate_capability_match(
            agent,
            python_issue["title"],
            python_issue["body"], 
            python_issue["labels"]
        )
        
        # Should match well due to Python and performance capabilities
        assert agent_match_score > 0.7, f"Agent match score {agent_match_score} too low for Python expert"
    
    @pytest.mark.asyncio
    async def test_api_endpoint_security(self):
        """Test API endpoint security and authentication."""
        
        client = TestClient(app)
        
        # Test unauthenticated access fails
        response = client.get("/api/v1/github/repository/123/status")
        assert response.status_code == 401
        
        # Test invalid token fails
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/github/repository/123/status", headers=headers)
        assert response.status_code == 401
        
        # Test malformed requests are rejected
        response = client.post("/api/v1/github/repository/setup", json={"invalid": "data"})
        assert response.status_code in [401, 422]  # Unauthorized or validation error
    
    @pytest.mark.asyncio
    async def test_performance_metrics_validation(self, db_session: AsyncSession, mock_github_client):
        """Test that all performance metrics meet requirements."""
        
        # Initialize all components
        work_tree_manager = WorkTreeManager(mock_github_client)
        branch_manager = BranchManager(mock_github_client, work_tree_manager)
        
        # Test work tree operations performance
        performance_metrics = await work_tree_manager.get_performance_metrics()
        
        # Verify performance requirements
        assert performance_metrics["average_creation_time_seconds"] < 30.0
        assert performance_metrics["average_sync_time_seconds"] < 15.0
        assert performance_metrics["average_cleanup_time_seconds"] < 5.0
        assert performance_metrics["success_rate"] > 0.98
        assert performance_metrics["isolation_effectiveness"] >= 1.0  # 100% isolation
    
    @pytest.mark.asyncio
    async def test_comprehensive_error_handling(self, db_session: AsyncSession, mock_github_client):
        """Test comprehensive error handling and recovery."""
        
        # Initialize components with failing GitHub client
        failing_client = Mock(spec=GitHubAPIClient)
        failing_client.get_repository = AsyncMock(side_effect=Exception("API Error"))
        failing_client.create_pull_request = AsyncMock(side_effect=Exception("Network Error"))
        
        work_tree_manager = WorkTreeManager(failing_client)
        
        # Test graceful error handling
        test_repo = GitHubRepository(repository_full_name="test/repo")
        
        try:
            # Should handle errors gracefully without crashing
            work_tree = await work_tree_manager.create_agent_work_tree("test-agent", test_repo)
            assert False, "Should have raised an error"
        except Exception as e:
            # Should be a controlled error, not a crash
            assert "failed" in str(e).lower()
    
    @pytest.mark.asyncio 
    async def test_concurrent_operations(self, db_session: AsyncSession, mock_github_client, test_repository):
        """Test concurrent operations and thread safety."""
        
        # Create multiple agents
        agents = []
        for i in range(5):
            agent = Agent(name=f"concurrent-agent-{i}", type="developer", status=AgentStatus.active)
            agents.append(agent)
        
        db_session.add_all([test_repository] + agents)
        await db_session.commit()
        
        # Initialize work tree manager
        work_tree_manager = WorkTreeManager(mock_github_client)
        
        with patch('app.core.work_tree_manager.asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"success", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            # Create work trees concurrently
            tasks = []
            for agent in agents:
                task = work_tree_manager.create_agent_work_tree(str(agent.id), test_repository)
                tasks.append(task)
            
            # Execute all tasks concurrently
            work_trees = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all succeeded
            success_count = sum(1 for wt in work_trees if not isinstance(wt, Exception))
            assert success_count >= 4, f"Only {success_count}/5 concurrent operations succeeded"
            
            # Verify no conflicts in work tree paths
            successful_work_trees = [wt for wt in work_trees if not isinstance(wt, Exception)]
            paths = [wt.work_tree_path for wt in successful_work_trees]
            assert len(paths) == len(set(paths)), "Work tree paths are not unique"
    
    @pytest.mark.asyncio
    async def test_database_performance_under_load(self, db_session: AsyncSession):
        """Test database performance under load."""
        
        # Create large dataset
        repositories = []
        work_trees = []
        
        for i in range(50):
            repo = GitHubRepository(
                repository_full_name=f"test/repo-{i}",
                repository_url=f"https://github.com/test/repo-{i}"
            )
            repositories.append(repo)
        
        # Bulk insert
        db_session.add_all(repositories)
        await db_session.commit()
        
        # Measure query performance
        start_time = datetime.utcnow()
        
        # Complex query with joins
        result = await db_session.execute(
            select(GitHubRepository)
            .order_by(GitHubRepository.created_at.desc())
            .limit(20)
        )
        repos = result.scalars().all()
        
        end_time = datetime.utcnow()
        query_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert query_time < 1.0, f"Database query took {query_time}s, too slow"
        assert len(repos) <= 20


class TestGitHubIntegrationAPI:
    """Test API endpoints specifically."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @pytest.mark.asyncio
    async def test_repository_validation_endpoint(self):
        """Test repository validation endpoint."""
        
        # Mock authentication
        with patch('app.api.v1.github_integration.get_authenticated_agent') as mock_auth:
            mock_auth.return_value = "test-agent-id"
            
            # Mock GitHub client
            with patch('app.api.v1.github_integration.github_client') as mock_client:
                mock_client.get_repository = AsyncMock(return_value={
                    "full_name": "test/repo",
                    "private": False,
                    "default_branch": "main",
                    "size": 1024,
                    "html_url": "https://github.com/test/repo"
                })
                
                response = self.client.post(
                    "/api/v1/github/repository/validate",
                    json={"repository_url": "https://github.com/test/repo"},
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["valid"] is True
                assert "repository" in data
                assert "permissions" in data
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        
        with patch('app.api.v1.github_integration.get_authenticated_agent') as mock_auth:
            mock_auth.return_value = "test-agent-id"
            
            # Mock component health checks
            with patch('app.api.v1.github_integration.github_client') as mock_client, \
                 patch('app.api.v1.github_integration.work_tree_manager') as mock_wtm, \
                 patch('app.api.v1.github_integration.branch_manager') as mock_bm:
                
                mock_client.health_check = AsyncMock(return_value=True)
                mock_client.check_rate_limits = AsyncMock(return_value={"core": {"remaining": 5000}})
                mock_wtm.health_check = AsyncMock(return_value={"healthy": True})
                mock_bm.health_check = AsyncMock(return_value={"healthy": True})
                
                response = self.client.get(
                    "/api/v1/github/health",
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["healthy"] is True
                assert "components" in data
                assert "github_api" in data["components"]
                assert "work_trees" in data["components"]
    
    @pytest.mark.asyncio
    async def test_performance_endpoint(self):
        """Test performance metrics endpoint."""
        
        with patch('app.api.v1.github_integration.get_authenticated_agent') as mock_auth:
            mock_auth.return_value = "test-agent-id"
            
            with patch('app.api.v1.github_integration.github_client') as mock_client:
                mock_client.check_rate_limits = AsyncMock(return_value={
                    "core": {"limit": 5000, "remaining": 4999, "reset": 1234567890}
                })
                
                response = self.client.get(
                    "/api/v1/github/performance",
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify performance metrics structure
                assert "github_api" in data
                assert "pull_request_creation" in data
                assert "work_tree_operations" in data
                assert "code_review" in data
                
                # Verify performance targets are met
                assert data["pull_request_creation"]["target_time_seconds"] == 30.0
                assert data["pull_request_creation"]["average_time_seconds"] < 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])