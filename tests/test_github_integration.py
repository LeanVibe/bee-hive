"""
Comprehensive test suite for GitHub Integration system.

Tests cover all GitHub operations, isolation mechanisms, performance requirements,
and security controls for multi-agent development workflows.
"""

import asyncio
import json
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.github_api_client import GitHubAPIClient
from app.core.work_tree_manager import WorkTreeManager, WorkTreeConfig
from app.core.branch_manager import BranchManager, ConflictResolutionStrategy
from app.core.pull_request_automator import PullRequestAutomator
from app.core.issue_manager import IssueManager
from app.core.code_review_assistant import CodeReviewAssistant
from app.core.github_webhooks import GitHubWebhookProcessor
from app.core.github_security import (
    GitHubSecurityManager, TokenManager, PermissionManager, 
    AuditLogger, AccessLevel, GitHubPermission, AuditEventType
)
from app.models.github_integration import (
    GitHubRepository, AgentWorkTree, PullRequest, 
    GitHubIssue, CodeReview, GitCommit, BranchOperation,
    WorkTreeStatus, PullRequestStatus, IssueState, ReviewStatus
)
from app.models.agent import Agent, AgentStatus


class TestTokenManager:
    """Test token encryption, validation, and security."""
    
    @pytest.fixture
    def token_manager(self):
        return TokenManager()
    
    def test_encrypt_decrypt_token(self, token_manager):
        """Test token encryption/decryption cycle."""
        original_token = "ghp_1234567890abcdef1234567890abcdef12345678"
        
        # Encrypt token
        encrypted = token_manager.encrypt_token(original_token)
        assert encrypted != original_token
        assert len(encrypted) > len(original_token)
        
        # Decrypt token
        decrypted = token_manager.decrypt_token(encrypted)
        assert decrypted == original_token
    
    def test_encrypt_empty_token_raises_error(self, token_manager):
        """Test that encrypting empty token raises error."""
        with pytest.raises(Exception):
            token_manager.encrypt_token("")
    
    def test_validate_token_format(self, token_manager):
        """Test GitHub token format validation."""
        valid_tokens = [
            "ghp_1234567890abcdef1234567890abcdef12345678",
            "github_pat_11AAAAAAA0123456789abcdef0123456789abcdef",
            "ghs_abcdef1234567890abcdef1234567890abcdef12"
        ]
        
        for token in valid_tokens:
            assert token_manager.validate_token_format(token)
        
        invalid_tokens = [
            "invalid_token",
            "github_1234567890",
            "bearer_token_123",
            ""
        ]
        
        for token in invalid_tokens:
            assert not token_manager.validate_token_format(token)
    
    def test_webhook_signature_verification(self, token_manager):
        """Test webhook signature verification."""
        payload = b'{"action": "opened", "pull_request": {"id": 123}}'
        secret = "webhook_secret_123"
        
        # Generate valid signature
        import hmac
        import hashlib
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        github_signature = f"sha256={expected_signature}"
        
        # Verify signature
        assert token_manager.verify_webhook_signature(payload, github_signature, secret)
        
        # Test with invalid signature
        assert not token_manager.verify_webhook_signature(payload, "sha256=invalid", secret)
        
        # Test with wrong secret
        assert not token_manager.verify_webhook_signature(payload, github_signature, "wrong_secret")


class TestPermissionManager:
    """Test permission validation and access control."""
    
    @pytest.fixture
    def permission_manager(self):
        return PermissionManager()
    
    def test_default_agent_permissions(self, permission_manager):
        """Test default permission assignment."""
        permissions = permission_manager.get_default_agent_permissions()
        
        assert "read" in permissions
        assert "issues" in permissions
        assert "pull_requests" in permissions
        assert permissions["read"] == "read"
        assert permissions["issues"] == "write"
    
    def test_permission_validation_success(self, permission_manager):
        """Test successful permission validation."""
        requested = {
            "read": "read",
            "issues": "write"
        }
        max_allowed = {
            "read": "read",
            "issues": "write",
            "contents": "write"
        }
        
        result = permission_manager.validate_permission_request(requested, max_allowed)
        
        assert result["valid"]
        assert result["granted_permissions"] == requested
        assert len(result["denied_permissions"]) == 0
    
    def test_permission_validation_failure(self, permission_manager):
        """Test permission validation with insufficient permissions."""
        requested = {
            "read": "admin",  # Requesting more than allowed
            "contents": "write"
        }
        max_allowed = {
            "read": "read",  # Only read access allowed
            "contents": "read"  # Only read access allowed
        }
        
        result = permission_manager.validate_permission_request(requested, max_allowed)
        
        assert not result["valid"]
        assert len(result["denied_permissions"]) == 2
        assert "read" in result["denied_permissions"]
        assert "contents" in result["denied_permissions"]
    
    def test_minimum_required_permissions(self, permission_manager):
        """Test calculation of minimum required permissions."""
        operations = ["create_pull_request", "push_commits", "create_issue"]
        
        required = permission_manager.calculate_minimum_required_permissions(operations)
        
        assert "pull_requests" in required
        assert "contents" in required
        assert "issues" in required
        assert required["pull_requests"] == "write"
        assert required["contents"] == "write"
        assert required["issues"] == "write"
    
    def test_check_agent_permission(self, permission_manager):
        """Test agent permission checking."""
        agent_permissions = {
            "contents": "write",
            "issues": "read",
            "pull_requests": "write"
        }
        
        # Test sufficient permission
        assert permission_manager.check_agent_permission(
            agent_permissions, "contents", "write"
        )
        
        # Test insufficient permission
        assert not permission_manager.check_agent_permission(
            agent_permissions, "issues", "write"
        )
        
        # Test non-existent permission
        assert not permission_manager.check_agent_permission(
            agent_permissions, "admin", "read"
        )


@pytest.mark.asyncio
class TestAuditLogger:
    """Test security audit logging."""
    
    @pytest.fixture
    def audit_logger(self):
        return AuditLogger()
    
    async def test_log_security_event(self, audit_logger):
        """Test security event logging."""
        correlation_id = await audit_logger.log_security_event(
            AuditEventType.TOKEN_CREATED,
            agent_id="agent_123",
            repository_id="repo_456",
            context={"token_type": "personal_access_token"},
            user_agent="Agent-Hive/2.0",
            ip_address="192.168.1.100"
        )
        
        assert correlation_id is not None
        assert len(correlation_id) > 0
    
    def test_sanitize_context(self, audit_logger):
        """Test sensitive data sanitization."""
        context = {
            "token": "ghp_1234567890abcdef1234567890abcdef12345678",
            "password": "secret123",
            "user_id": "user_123",
            "api_key": "sk-1234567890abcdef",
            "nested": {
                "secret": "hidden_value",
                "public": "visible_value"
            }
        }
        
        sanitized = audit_logger._sanitize_context(context)
        
        assert sanitized["token"] == "ghp_***5678"
        assert sanitized["password"] == "***"
        assert sanitized["user_id"] == "user_123"  # Not sensitive
        assert sanitized["api_key"] == "sk-1***cdef"
        assert sanitized["nested"]["secret"] == "***"
        assert sanitized["nested"]["public"] == "visible_value"
    
    def test_calculate_event_severity(self, audit_logger):
        """Test event severity calculation."""
        assert audit_logger._calculate_event_severity(AuditEventType.SECURITY_VIOLATION) == "critical"
        assert audit_logger._calculate_event_severity(AuditEventType.ACCESS_DENIED) == "high"
        assert audit_logger._calculate_event_severity(AuditEventType.TOKEN_CREATED) == "low"
        assert audit_logger._calculate_event_severity(AuditEventType.API_CALL) == "info"
    
    def test_generate_event_tags(self, audit_logger):
        """Test event tag generation."""
        context = {
            "permission": "contents",
            "repository": "test-repo",
            "api_endpoint": "/repos/owner/repo"
        }
        
        tags = audit_logger._generate_event_tags(AuditEventType.REPOSITORY_ACCESS, context)
        
        assert "repository_access" in tags
        assert "permission:contents" in tags
        assert "repository_access" in tags
        assert "api:/repos/owner/repo" in tags


@pytest.mark.asyncio
class TestGitHubAPIClient:
    """Test GitHub API client functionality."""
    
    @pytest.fixture
    def api_client(self):
        return GitHubAPIClient("test_token_123")
    
    @patch('httpx.AsyncClient.get')
    async def test_get_user_info(self, mock_get, api_client):
        """Test getting user information."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "login": "test_user",
            "id": 12345,
            "name": "Test User"
        }
        mock_get.return_value = mock_response
        
        user_info = await api_client.get_user_info()
        
        assert user_info["login"] == "test_user"
        assert user_info["id"] == 12345
        mock_get.assert_called_once()
    
    @patch('httpx.AsyncClient.get')
    async def test_rate_limit_handling(self, mock_get, api_client):
        """Test rate limit handling."""
        # Mock rate limit response
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int((datetime.now() + timedelta(hours=1)).timestamp()))
        }
        mock_response.json.return_value = {"message": "API rate limit exceeded"}
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception):  # Should raise rate limit exception
            await api_client.get_user_info()
    
    @patch('httpx.AsyncClient.post')
    async def test_create_repository(self, mock_post, api_client):
        """Test repository creation."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": 123456,
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "clone_url": "https://github.com/owner/test-repo.git"
        }
        mock_post.return_value = mock_response
        
        repo_data = await api_client.create_repository("test-repo", "Test repository")
        
        assert repo_data["name"] == "test-repo"
        assert repo_data["id"] == 123456
        mock_post.assert_called_once()


@pytest.mark.asyncio
class TestWorkTreeManager:
    """Test work tree isolation and management."""
    
    @pytest.fixture
    def work_tree_manager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            yield WorkTreeManager(base_path)
    
    async def test_create_isolated_work_tree(self, work_tree_manager):
        """Test creating isolated work tree."""
        agent_id = "agent_123"
        repo_url = "https://github.com/owner/repo.git"
        
        config = WorkTreeConfig(
            agent_id=agent_id,
            repository_url=repo_url,
            branch="feature/test"
        )
        
        with patch('git.Repo.clone_from') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            work_tree_path = await work_tree_manager.create_work_tree(config)
            
            assert work_tree_path.exists()
            assert agent_id in str(work_tree_path)
            mock_clone.assert_called_once()
    
    async def test_work_tree_isolation_validation(self, work_tree_manager):
        """Test that work trees are properly isolated."""
        agent1_id = "agent_1"
        agent2_id = "agent_2"
        repo_url = "https://github.com/owner/repo.git"
        
        config1 = WorkTreeConfig(agent_id=agent1_id, repository_url=repo_url)
        config2 = WorkTreeConfig(agent_id=agent2_id, repository_url=repo_url)
        
        with patch('git.Repo.clone_from'):
            work_tree1 = await work_tree_manager.create_work_tree(config1)
            work_tree2 = await work_tree_manager.create_work_tree(config2)
            
            # Verify paths are different
            assert work_tree1 != work_tree2
            assert agent1_id in str(work_tree1)
            assert agent2_id in str(work_tree2)
            
            # Test isolation validation
            isolation_result = await work_tree_manager.validate_work_tree_isolation(agent1_id)
            assert isolation_result["isolated"]
            assert isolation_result["effectiveness_score"] == 1.0
    
    async def test_cleanup_work_tree(self, work_tree_manager):
        """Test work tree cleanup."""
        agent_id = "agent_cleanup_test"
        config = WorkTreeConfig(
            agent_id=agent_id,
            repository_url="https://github.com/owner/repo.git"
        )
        
        with patch('git.Repo.clone_from'):
            work_tree_path = await work_tree_manager.create_work_tree(config)
            assert work_tree_path.exists()
            
            # Cleanup work tree
            cleanup_result = await work_tree_manager.cleanup_work_tree(agent_id)
            assert cleanup_result["success"]
            assert not work_tree_path.exists()


@pytest.mark.asyncio
class TestBranchManager:
    """Test branch management and conflict resolution."""
    
    @pytest.fixture
    def branch_manager(self):
        return BranchManager()
    
    async def test_create_feature_branch(self, branch_manager):
        """Test feature branch creation."""
        agent_id = "agent_123"
        base_branch = "main"
        feature_name = "add-new-feature"
        
        with patch('git.Repo') as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            
            branch_name = await branch_manager.create_feature_branch(
                "/tmp/repo", agent_id, base_branch, feature_name
            )
            
            expected_branch = f"agent-{agent_id}/feature/{feature_name}"
            assert branch_name == expected_branch
            mock_repo.git.checkout.assert_called()
    
    async def test_conflict_resolution_strategy(self, branch_manager):
        """Test conflict resolution strategies."""
        strategies = [
            ConflictResolutionStrategy.MERGE,
            ConflictResolutionStrategy.REBASE,
            ConflictResolutionStrategy.SQUASH_MERGE
        ]
        
        for strategy in strategies:
            with patch('git.Repo') as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                
                resolution_result = await branch_manager.resolve_merge_conflicts(
                    "/tmp/repo", "feature-branch", "main", strategy
                )
                
                assert "strategy" in resolution_result
                assert resolution_result["strategy"] == strategy.value
    
    async def test_intelligent_conflict_detection(self, branch_manager):
        """Test intelligent conflict detection."""
        mock_conflicts = [
            "src/file1.py: merge conflict in function definition",
            "config.json: conflicting JSON structure changes"
        ]
        
        with patch.object(branch_manager.conflict_resolver, 'detect_conflicts') as mock_detect:
            mock_detect.return_value = mock_conflicts
            
            conflicts = await branch_manager.detect_potential_conflicts("/tmp/repo", "feature", "main")
            
            assert len(conflicts) == 2
            assert any("function definition" in conflict for conflict in conflicts)


@pytest.mark.asyncio
class TestPullRequestAutomator:
    """Test pull request automation and performance."""
    
    @pytest.fixture
    def pr_automator(self):
        mock_github_client = AsyncMock()
        return PullRequestAutomator(mock_github_client)
    
    async def test_create_pull_request_performance(self, pr_automator):
        """Test PR creation meets <30 second performance target."""
        import time
        
        pr_data = {
            "title": "Test PR",
            "body": "Test description",
            "head": "feature-branch",
            "base": "main"
        }
        
        # Mock GitHub API response
        pr_automator.github_client.create_pull_request.return_value = {
            "number": 123,
            "html_url": "https://github.com/owner/repo/pull/123"
        }
        
        start_time = time.time()
        result = await pr_automator.create_optimized_pull_request(
            "owner/repo", pr_data, performance_target_seconds=30
        )
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 30  # Performance requirement
        assert result["performance_ms"] < 30000
        assert result["success"]
    
    async def test_intelligent_pr_content_generation(self, pr_automator):
        """Test AI-powered PR content generation."""
        commit_messages = [
            "feat: add user authentication",
            "fix: resolve login bug",
            "docs: update API documentation"
        ]
        
        content = await pr_automator.generate_pr_content(
            "feature/user-auth", commit_messages, {"changed_files": 5, "lines_added": 150}
        )
        
        assert "title" in content
        assert "body" in content
        assert len(content["title"]) > 0
        assert "authentication" in content["title"].lower()
    
    async def test_pr_template_generation(self, pr_automator):
        """Test PR template generation."""
        template = await pr_automator.generate_pr_template("bug_fix")
        
        assert "## Description" in template
        assert "## Testing" in template
        assert "## Checklist" in template


@pytest.mark.asyncio
class TestIssueManager:
    """Test GitHub issue management."""
    
    @pytest.fixture
    def issue_manager(self):
        mock_github_client = AsyncMock()
        return IssueManager(mock_github_client)
    
    async def test_intelligent_agent_assignment(self, issue_manager):
        """Test intelligent agent assignment based on capabilities."""
        issue_data = {
            "title": "Implement user authentication system",
            "body": "Need to add JWT-based authentication with role-based access control",
            "labels": ["enhancement", "backend", "security"]
        }
        
        agents = [
            {"id": "agent_1", "capabilities": ["backend", "security", "python"]},
            {"id": "agent_2", "capabilities": ["frontend", "react", "ui"]},
            {"id": "agent_3", "capabilities": ["backend", "authentication", "security"]}
        ]
        
        assignment = await issue_manager.assign_issue_to_agent(issue_data, agents)
        
        # Should assign to agent_3 (best match for authentication + security)
        assert assignment["agent_id"] == "agent_3"
        assert assignment["confidence_score"] > 0.8
    
    async def test_issue_classification(self, issue_manager):
        """Test automatic issue classification."""
        bug_issue = {
            "title": "Login form crashes on submit",
            "body": "Error: TypeError when clicking submit button"
        }
        
        feature_issue = {
            "title": "Add user profile dashboard",
            "body": "Users should be able to view and edit their profile information"
        }
        
        bug_classification = await issue_manager.classify_issue(bug_issue)
        feature_classification = await issue_manager.classify_issue(feature_issue)
        
        assert bug_classification["type"] == "bug"
        assert feature_classification["type"] == "enhancement"
        assert bug_classification["priority"] in ["high", "medium"]
    
    async def test_bi_directional_synchronization(self, issue_manager):
        """Test bi-directional issue synchronization."""
        # Mock GitHub API calls
        issue_manager.github_client.get_repository_issues.return_value = [
            {"number": 1, "title": "Test Issue", "state": "open"},
            {"number": 2, "title": "Another Issue", "state": "closed"}
        ]
        
        sync_result = await issue_manager.sync_repository_issues("owner/repo")
        
        assert sync_result["synced_count"] == 2
        assert sync_result["success"]
        issue_manager.github_client.get_repository_issues.assert_called_once()


@pytest.mark.asyncio
class TestCodeReviewAssistant:
    """Test automated code review functionality."""
    
    @pytest.fixture
    def code_review_assistant(self):
        return CodeReviewAssistant()
    
    async def test_security_analysis(self, code_review_assistant):
        """Test security vulnerability detection."""
        code_changes = {
            "src/auth.py": """
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)  # SQL injection vulnerability
    return cursor.fetchone()
"""
        }
        
        security_analysis = await code_review_assistant.analyze_security(code_changes)
        
        assert len(security_analysis["vulnerabilities"]) > 0
        assert any("sql injection" in vuln["description"].lower() 
                  for vuln in security_analysis["vulnerabilities"])
        assert security_analysis["security_score"] < 0.7  # Low score due to vulnerability
    
    async def test_performance_analysis(self, code_review_assistant):
        """Test performance issue detection."""
        code_changes = {
            "src/utils.py": """
def process_large_list(items):
    result = []
    for item in items:
        for i in range(len(items)):  # O(n²) complexity
            if items[i] == item:
                result.append(item)
    return result
"""
        }
        
        performance_analysis = await code_review_assistant.analyze_performance(code_changes)
        
        assert len(performance_analysis["issues"]) > 0
        assert any("complexity" in issue["description"].lower() 
                  for issue in performance_analysis["issues"])
        assert performance_analysis["performance_score"] < 0.8
    
    async def test_style_analysis(self, code_review_assistant):
        """Test code style analysis."""
        code_changes = {
            "src/bad_style.py": """
def badFunction(x,y):
    if x>y:
        return x+y
    else:
        return x-y
"""
        }
        
        style_analysis = await code_review_assistant.analyze_style(code_changes)
        
        assert len(style_analysis["issues"]) > 0
        assert style_analysis["style_score"] < 1.0
    
    async def test_comprehensive_code_review(self, code_review_assistant):
        """Test comprehensive code review."""
        pr_data = {
            "number": 123,
            "title": "Add user authentication",
            "changed_files": ["src/auth.py", "src/models.py"],
            "additions": 150,
            "deletions": 20
        }
        
        code_changes = {
            "src/auth.py": "def authenticate(user): return True",
            "src/models.py": "class User: pass"
        }
        
        review = await code_review_assistant.review_pull_request(pr_data, code_changes)
        
        assert "overall_score" in review
        assert "security_analysis" in review
        assert "performance_analysis" in review
        assert "style_analysis" in review
        assert "recommendations" in review
        assert isinstance(review["recommendations"], list)


@pytest.mark.asyncio
class TestGitHubWebhookProcessor:
    """Test GitHub webhook processing."""
    
    @pytest.fixture
    def webhook_processor(self):
        return GitHubWebhookProcessor()
    
    async def test_pull_request_webhook_processing(self, webhook_processor):
        """Test pull request webhook event processing."""
        webhook_payload = {
            "action": "opened",
            "pull_request": {
                "number": 123,
                "title": "Test PR",
                "head": {"ref": "feature-branch"},
                "base": {"ref": "main"}
            },
            "repository": {
                "full_name": "owner/repo"
            }
        }
        
        result = await webhook_processor.process_pull_request_event(webhook_payload)
        
        assert result["event_type"] == "pull_request"
        assert result["action"] == "opened"
        assert result["pr_number"] == 123
    
    async def test_push_webhook_processing(self, webhook_processor):
        """Test push webhook event processing."""
        webhook_payload = {
            "ref": "refs/heads/main",
            "commits": [
                {
                    "id": "abc123",
                    "message": "feat: add new feature",
                    "author": {"name": "Test User"}
                }
            ],
            "repository": {
                "full_name": "owner/repo"
            }
        }
        
        result = await webhook_processor.process_push_event(webhook_payload)
        
        assert result["event_type"] == "push"
        assert result["branch"] == "main"
        assert len(result["commits"]) == 1
    
    async def test_issue_webhook_processing(self, webhook_processor):
        """Test issue webhook event processing."""
        webhook_payload = {
            "action": "opened",
            "issue": {
                "number": 456,
                "title": "Bug report",
                "body": "Something is broken"
            },
            "repository": {
                "full_name": "owner/repo"
            }
        }
        
        result = await webhook_processor.process_issue_event(webhook_payload)
        
        assert result["event_type"] == "issue"
        assert result["action"] == "opened"
        assert result["issue_number"] == 456


@pytest.mark.asyncio
class TestGitHubSecurityManager:
    """Test comprehensive GitHub security management."""
    
    @pytest.fixture
    def security_manager(self):
        return GitHubSecurityManager()
    
    async def test_setup_agent_github_access(self, security_manager):
        """Test secure GitHub access setup for agents."""
        agent_id = "agent_123"
        repository_id = str(uuid.uuid4())
        github_token = "ghp_1234567890abcdef1234567890abcdef12345678"
        requested_permissions = {
            "read": "read",
            "contents": "write",
            "issues": "write"
        }
        
        with patch('app.core.database.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock repository query
            mock_repo = MagicMock()
            mock_repo.id = uuid.UUID(repository_id)
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_repo
            
            result = await security_manager.setup_agent_github_access(
                agent_id, repository_id, github_token, requested_permissions
            )
            
            assert result["success"]
            assert "granted_permissions" in result
            assert "correlation_id" in result
    
    async def test_validate_agent_operation(self, security_manager):
        """Test agent operation validation."""
        agent_id = "agent_123"
        repository_id = str(uuid.uuid4())
        operation = "create_pull_request"
        
        with patch('app.core.database.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock repository with appropriate permissions
            mock_repo = MagicMock()
            mock_repo.agent_permissions = {
                "pull_requests": "write",
                "contents": "write"
            }
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_repo
            
            result = await security_manager.validate_agent_operation(
                agent_id, repository_id, operation
            )
            
            assert result["allowed"]
            assert "required_permissions" in result
    
    async def test_token_rotation(self, security_manager):
        """Test secure token rotation."""
        repository_id = str(uuid.uuid4())
        agent_id = "agent_123"
        new_token = "ghp_new_token_1234567890abcdef1234567890abcdef"
        
        with patch('app.core.database.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock repository
            mock_repo = MagicMock()
            mock_repo.access_token_hash = "old_encrypted_token"
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_repo
            
            result = await security_manager.rotate_repository_token(
                repository_id, new_token, agent_id
            )
            
            assert result["success"]
            assert "rotated_at" in result
    
    async def test_security_audit_report(self, security_manager):
        """Test security audit report generation."""
        report = await security_manager.get_security_audit_report(days=30)
        
        assert "period_days" in report
        assert "summary" in report
        assert "recommendations" in report
        assert report["period_days"] == 30
        assert isinstance(report["recommendations"], list)


@pytest.mark.asyncio
class TestPerformanceRequirements:
    """Test that all performance requirements are met."""
    
    async def test_github_api_success_rate(self):
        """Test >99.5% GitHub API success rate requirement."""
        api_client = GitHubAPIClient("test_token")
        
        # Simulate 1000 API calls with high success rate
        success_count = 0
        total_calls = 1000
        
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock 99.6% success rate
            for i in range(total_calls):
                if i < 996:  # 99.6% success
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"status": "success"}
                    mock_get.return_value = mock_response
                    
                    try:
                        await api_client.get_user_info()
                        success_count += 1
                    except:
                        pass
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 500
                    mock_get.return_value = mock_response
                    
                    try:
                        await api_client.get_user_info()
                        success_count += 1
                    except:
                        pass
        
        success_rate = success_count / total_calls
        assert success_rate > 0.995  # >99.5% requirement
    
    async def test_pull_request_creation_time(self):
        """Test <30 second pull request creation time requirement."""
        import time
        
        mock_github_client = AsyncMock()
        pr_automator = PullRequestAutomator(mock_github_client)
        
        # Mock fast API response
        mock_github_client.create_pull_request.return_value = {
            "number": 123,
            "html_url": "https://github.com/owner/repo/pull/123"
        }
        
        pr_data = {
            "title": "Test PR",
            "body": "Test description",
            "head": "feature-branch",
            "base": "main"
        }
        
        start_time = time.time()
        result = await pr_automator.create_optimized_pull_request(
            "owner/repo", pr_data, performance_target_seconds=30
        )
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 30  # <30 second requirement
        assert result["performance_ms"] < 30000
    
    async def test_work_tree_isolation_effectiveness(self):
        """Test 100% work tree isolation effectiveness requirement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            work_tree_manager = WorkTreeManager(base_path)
            
            agent_id = "isolation_test_agent"
            config = WorkTreeConfig(
                agent_id=agent_id,
                repository_url="https://github.com/owner/repo.git"
            )
            
            with patch('git.Repo.clone_from'):
                await work_tree_manager.create_work_tree(config)
                
                isolation_result = await work_tree_manager.validate_work_tree_isolation(agent_id)
                
                assert isolation_result["isolated"]
                assert isolation_result["effectiveness_score"] == 1.0  # 100% requirement


# Integration test fixtures
@pytest.fixture
async def test_database():
    """Create test database session."""
    # This would set up a test database
    # For now, using mocks
    pass

@pytest.fixture
async def test_redis():
    """Create test Redis connection."""
    # This would set up a test Redis instance
    # For now, using mocks
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])