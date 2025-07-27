"""
End-to-end integration tests for GitHub Integration system.

Tests multi-agent collaboration, conflict resolution, and complete workflows
from issue creation to pull request merge.
"""

import asyncio
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio

from app.core.github_api_client import GitHubAPIClient
from app.core.work_tree_manager import WorkTreeManager, WorkTreeConfig
from app.core.branch_manager import BranchManager, ConflictResolutionStrategy
from app.core.pull_request_automator import PullRequestAutomator
from app.core.issue_manager import IssueManager
from app.core.github_security import GitHubSecurityManager
from app.models.github_integration import (
    GitHubRepository, AgentWorkTree, PullRequest, 
    GitHubIssue, WorkTreeStatus, PullRequestStatus
)


@pytest.mark.asyncio
class TestMultiAgentCollaboration:
    """Test multi-agent collaboration scenarios."""
    
    @pytest.fixture
    async def setup_multi_agent_environment(self):
        """Set up environment for multi-agent testing."""
        # Create temporary directory for work trees
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Initialize managers
            work_tree_manager = WorkTreeManager(base_path)
            branch_manager = BranchManager()
            security_manager = GitHubSecurityManager()
            
            # Mock GitHub client
            github_client = AsyncMock()
            github_client.get_repository.return_value = {
                "id": 123456,
                "name": "test-repo",
                "full_name": "owner/test-repo",
                "default_branch": "main"
            }
            
            # Create test agents
            agents = [
                {
                    "id": "agent_backend",
                    "capabilities": ["python", "backend", "api", "database"],
                    "specialization": "backend_development"
                },
                {
                    "id": "agent_frontend", 
                    "capabilities": ["javascript", "react", "ui", "frontend"],
                    "specialization": "frontend_development"
                },
                {
                    "id": "agent_devops",
                    "capabilities": ["docker", "kubernetes", "ci/cd", "deployment"],
                    "specialization": "devops"
                }
            ]
            
            yield {
                "work_tree_manager": work_tree_manager,
                "branch_manager": branch_manager,
                "security_manager": security_manager,
                "github_client": github_client,
                "agents": agents,
                "base_path": base_path
            }
    
    async def test_concurrent_feature_development(self, setup_multi_agent_environment):
        """Test multiple agents working on different features simultaneously."""
        env = setup_multi_agent_environment
        
        repository_url = "https://github.com/owner/test-repo.git"
        
        # Simulate agents working on different features
        features = [
            {
                "agent_id": "agent_backend",
                "feature": "user-authentication",
                "files": ["src/auth.py", "src/models/user.py"]
            },
            {
                "agent_id": "agent_frontend",
                "feature": "login-ui",
                "files": ["src/components/Login.jsx", "src/styles/auth.css"]
            },
            {
                "agent_id": "agent_devops",
                "feature": "deployment-pipeline", 
                "files": ["Dockerfile", ".github/workflows/deploy.yml"]
            }
        ]
        
        work_trees = {}
        branches = {}
        
        # Each agent creates isolated work tree
        for feature in features:
            agent_id = feature["agent_id"]
            
            config = WorkTreeConfig(
                agent_id=agent_id,
                repository_url=repository_url,
                branch=f"feature/{feature['feature']}"
            )
            
            with patch('git.Repo.clone_from'):
                work_tree_path = await env["work_tree_manager"].create_work_tree(config)
                work_trees[agent_id] = work_tree_path
                
                # Create feature branch
                with patch('git.Repo'):
                    branch_name = await env["branch_manager"].create_feature_branch(
                        str(work_tree_path), agent_id, "main", feature["feature"]
                    )
                    branches[agent_id] = branch_name
        
        # Verify isolation
        assert len(work_trees) == 3
        assert len(set(work_trees.values())) == 3  # All paths are unique
        
        for agent_id in work_trees:
            isolation_result = await env["work_tree_manager"].validate_work_tree_isolation(agent_id)
            assert isolation_result["isolated"]
            assert isolation_result["effectiveness_score"] == 1.0
    
    async def test_conflicting_changes_resolution(self, setup_multi_agent_environment):
        """Test resolution of conflicting changes between agents."""
        env = setup_multi_agent_environment
        
        # Simulate two agents modifying the same file
        agent1_id = "agent_backend"
        agent2_id = "agent_frontend"
        
        # Both agents work on user management features
        config1 = WorkTreeConfig(
            agent_id=agent1_id,
            repository_url="https://github.com/owner/repo.git",
            branch="feature/user-backend"
        )
        
        config2 = WorkTreeConfig(
            agent_id=agent2_id,
            repository_url="https://github.com/owner/repo.git", 
            branch="feature/user-frontend"
        )
        
        with patch('git.Repo.clone_from'):
            work_tree1 = await env["work_tree_manager"].create_work_tree(config1)
            work_tree2 = await env["work_tree_manager"].create_work_tree(config2)
            
            # Simulate conflict detection
            mock_conflicts = [
                "src/shared/user.js: conflicting changes in User class definition",
                "src/config/api.js: different API endpoint configurations"
            ]
            
            with patch.object(env["branch_manager"].conflict_resolver, 'detect_conflicts') as mock_detect:
                mock_detect.return_value = mock_conflicts
                
                conflicts = await env["branch_manager"].detect_potential_conflicts(
                    str(work_tree1), "feature/user-backend", "main"
                )
                
                assert len(conflicts) == 2
                assert any("User class" in conflict for conflict in conflicts)
        
        # Test conflict resolution
        with patch('git.Repo'):
            resolution_result = await env["branch_manager"].resolve_merge_conflicts(
                str(work_tree1), "feature/user-backend", "main", 
                ConflictResolutionStrategy.INTELLIGENT_MERGE
            )
            
            assert resolution_result["success"]
            assert resolution_result["strategy"] == "intelligent_merge"
    
    async def test_coordinated_issue_to_pr_workflow(self, setup_multi_agent_environment):
        """Test complete workflow from issue assignment to PR creation."""
        env = setup_multi_agent_environment
        
        # Create issue manager and PR automator
        issue_manager = IssueManager(env["github_client"])
        pr_automator = PullRequestAutomator(env["github_client"])
        
        # Mock issue data
        issue_data = {
            "number": 123,
            "title": "Implement user authentication system",
            "body": """
            Need to implement a comprehensive user authentication system with:
            - JWT token generation and validation
            - Role-based access control
            - Password hashing and security
            - Login/logout endpoints
            """,
            "labels": ["enhancement", "backend", "security", "high-priority"]
        }
        
        # Test intelligent agent assignment
        agents = env["agents"]
        assignment = await issue_manager.assign_issue_to_agent(issue_data, agents)
        
        # Should assign to backend agent due to authentication + security capabilities
        assert assignment["agent_id"] == "agent_backend"
        assert assignment["confidence_score"] > 0.8
        
        assigned_agent = assignment["agent_id"]
        
        # Agent creates work tree and starts work
        config = WorkTreeConfig(
            agent_id=assigned_agent,
            repository_url="https://github.com/owner/repo.git",
            branch="feature/issue-123-user-auth"
        )
        
        with patch('git.Repo.clone_from'):
            work_tree_path = await env["work_tree_manager"].create_work_tree(config)
            
            # Create feature branch
            with patch('git.Repo'):
                branch_name = await env["branch_manager"].create_feature_branch(
                    str(work_tree_path), assigned_agent, "main", "issue-123-user-auth"
                )
                
                # Simulate development work (commits)
                commit_messages = [
                    "feat: add JWT token generation service",
                    "feat: implement password hashing utilities", 
                    "feat: create authentication middleware",
                    "feat: add login/logout API endpoints",
                    "test: add unit tests for auth system",
                    "docs: update API documentation"
                ]
                
                # Create PR
                env["github_client"].create_pull_request.return_value = {
                    "number": 456,
                    "html_url": "https://github.com/owner/repo/pull/456",
                    "title": "Implement user authentication system (fixes #123)"
                }
                
                pr_data = {
                    "title": f"Implement user authentication system (fixes #{issue_data['number']})",
                    "head": branch_name,
                    "base": "main"
                }
                
                pr_result = await pr_automator.create_optimized_pull_request(
                    "owner/repo", pr_data, performance_target_seconds=30
                )
                
                assert pr_result["success"]
                assert pr_result["pr_number"] == 456
                assert pr_result["performance_ms"] < 30000  # Performance requirement
    
    async def test_parallel_pr_creation_performance(self, setup_multi_agent_environment):
        """Test performance when multiple agents create PRs simultaneously."""
        import time
        
        env = setup_multi_agent_environment
        pr_automator = PullRequestAutomator(env["github_client"])
        
        # Mock PR creation responses
        env["github_client"].create_pull_request.side_effect = [
            {"number": 101, "html_url": "https://github.com/owner/repo/pull/101"},
            {"number": 102, "html_url": "https://github.com/owner/repo/pull/102"},
            {"number": 103, "html_url": "https://github.com/owner/repo/pull/103"}
        ]
        
        # Create PR data for each agent
        pr_requests = [
            {
                "repo": "owner/repo",
                "data": {
                    "title": "Backend: Add user authentication",
                    "head": "agent-agent_backend/feature/user-auth",
                    "base": "main"
                }
            },
            {
                "repo": "owner/repo", 
                "data": {
                    "title": "Frontend: Add login UI components",
                    "head": "agent-agent_frontend/feature/login-ui",
                    "base": "main"
                }
            },
            {
                "repo": "owner/repo",
                "data": {
                    "title": "DevOps: Add deployment pipeline",
                    "head": "agent-agent_devops/feature/deployment",
                    "base": "main"
                }
            }
        ]
        
        # Execute PRs in parallel
        start_time = time.time()
        
        tasks = []
        for pr_request in pr_requests:
            task = pr_automator.create_optimized_pull_request(
                pr_request["repo"], 
                pr_request["data"],
                performance_target_seconds=30
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_duration = end_time - start_time
        
        # Verify all PRs were created successfully
        assert len(results) == 3
        for result in results:
            assert result["success"]
            assert result["performance_ms"] < 30000
        
        # Parallel execution should be faster than sequential
        assert total_duration < 60  # Should complete much faster than 3 * 30 seconds
    
    async def test_agent_handoff_workflow(self, setup_multi_agent_environment):
        """Test workflow where one agent hands off work to another."""
        env = setup_multi_agent_environment
        
        # Agent 1 (backend) starts work on a feature
        agent1_id = "agent_backend"
        agent2_id = "agent_devops"
        
        # Backend agent creates initial implementation
        config1 = WorkTreeConfig(
            agent_id=agent1_id,
            repository_url="https://github.com/owner/repo.git",
            branch="feature/microservice-api"
        )
        
        with patch('git.Repo.clone_from'):
            work_tree1 = await env["work_tree_manager"].create_work_tree(config1)
            
            # Backend agent completes API development
            with patch('git.Repo'):
                branch1 = await env["branch_manager"].create_feature_branch(
                    str(work_tree1), agent1_id, "main", "microservice-api"
                )
                
                # Simulate handoff - DevOps agent takes over for deployment
                config2 = WorkTreeConfig(
                    agent_id=agent2_id,
                    repository_url="https://github.com/owner/repo.git",
                    branch=branch1  # Continue from backend agent's branch
                )
                
                work_tree2 = await env["work_tree_manager"].create_work_tree(config2)
                
                # Verify both agents have isolated work trees
                isolation1 = await env["work_tree_manager"].validate_work_tree_isolation(agent1_id)
                isolation2 = await env["work_tree_manager"].validate_work_tree_isolation(agent2_id)
                
                assert isolation1["isolated"]
                assert isolation2["isolated"]
                assert work_tree1 != work_tree2
                
                # DevOps agent adds deployment configuration
                deployment_branch = await env["branch_manager"].create_feature_branch(
                    str(work_tree2), agent2_id, branch1, "add-deployment-config"
                )
                
                assert agent2_id in deployment_branch
                assert "deployment" in deployment_branch


@pytest.mark.asyncio
class TestFailureRecoveryScenarios:
    """Test system recovery from various failure scenarios."""
    
    async def test_work_tree_corruption_recovery(self):
        """Test recovery from work tree corruption."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            work_tree_manager = WorkTreeManager(base_path)
            
            agent_id = "agent_test"
            config = WorkTreeConfig(
                agent_id=agent_id,
                repository_url="https://github.com/owner/repo.git"
            )
            
            with patch('git.Repo.clone_from'):
                # Create work tree
                work_tree_path = await work_tree_manager.create_work_tree(config)
                assert work_tree_path.exists()
                
                # Simulate corruption by deleting .git directory
                git_dir = work_tree_path / ".git"
                if git_dir.exists():
                    import shutil
                    shutil.rmtree(git_dir)
                
                # Attempt to validate - should detect corruption
                isolation_result = await work_tree_manager.validate_work_tree_isolation(agent_id)
                assert not isolation_result["isolated"]
                
                # Recovery should recreate work tree
                recovered_path = await work_tree_manager.create_work_tree(config)
                assert recovered_path.exists()
    
    async def test_github_api_failure_recovery(self):
        """Test recovery from GitHub API failures."""
        api_client = GitHubAPIClient("test_token")
        
        with patch('httpx.AsyncClient.get') as mock_get:
            # Simulate API failure followed by recovery
            failure_response = MagicMock()
            failure_response.status_code = 500
            failure_response.json.return_value = {"message": "Internal Server Error"}
            
            success_response = MagicMock()
            success_response.status_code = 200
            success_response.json.return_value = {"login": "test_user"}
            
            # First call fails, second succeeds (retry mechanism)
            mock_get.side_effect = [failure_response, success_response]
            
            # Should eventually succeed due to retry logic
            user_info = await api_client.get_user_info()
            assert user_info["login"] == "test_user"
            assert mock_get.call_count == 2  # Retry was attempted
    
    async def test_concurrent_access_conflict_resolution(self):
        """Test resolution when multiple agents access same resource."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            work_tree_manager = WorkTreeManager(base_path)
            
            # Two agents try to work on same repository simultaneously
            agent1_id = "agent_1"
            agent2_id = "agent_2"
            
            config1 = WorkTreeConfig(
                agent_id=agent1_id,
                repository_url="https://github.com/owner/repo.git",
                branch="feature/shared-feature"
            )
            
            config2 = WorkTreeConfig(
                agent_id=agent2_id,
                repository_url="https://github.com/owner/repo.git",
                branch="feature/shared-feature"  # Same branch
            )
            
            with patch('git.Repo.clone_from'):
                # Both agents should get isolated work trees even for same branch
                work_tree1 = await work_tree_manager.create_work_tree(config1)
                work_tree2 = await work_tree_manager.create_work_tree(config2)
                
                assert work_tree1 != work_tree2
                assert agent1_id in str(work_tree1)
                assert agent2_id in str(work_tree2)


@pytest.mark.asyncio
class TestPerformanceUnderLoad:
    """Test system performance under high load conditions."""
    
    async def test_high_volume_pr_creation(self):
        """Test PR creation performance with high volume."""
        github_client = AsyncMock()
        pr_automator = PullRequestAutomator(github_client)
        
        # Mock successful PR creation
        github_client.create_pull_request.return_value = {
            "number": 123,
            "html_url": "https://github.com/owner/repo/pull/123"
        }
        
        # Create 50 PRs concurrently
        pr_count = 50
        tasks = []
        
        for i in range(pr_count):
            pr_data = {
                "title": f"Feature PR #{i}",
                "head": f"feature-branch-{i}",
                "base": "main"
            }
            
            task = pr_automator.create_optimized_pull_request(
                "owner/repo", pr_data, performance_target_seconds=30
            )
            tasks.append(task)
        
        import time
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_duration = end_time - start_time
        
        # All PRs should succeed
        assert len(results) == pr_count
        for result in results:
            assert result["success"]
            assert result["performance_ms"] < 30000
        
        # Average time per PR should be reasonable
        avg_time_per_pr = total_duration / pr_count
        assert avg_time_per_pr < 5  # Should average less than 5 seconds per PR
    
    async def test_concurrent_work_tree_operations(self):
        """Test concurrent work tree operations performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            work_tree_manager = WorkTreeManager(base_path)
            
            # Create 20 work trees concurrently
            agent_count = 20
            tasks = []
            
            for i in range(agent_count):
                agent_id = f"agent_{i}"
                config = WorkTreeConfig(
                    agent_id=agent_id,
                    repository_url="https://github.com/owner/repo.git"
                )
                
                with patch('git.Repo.clone_from'):
                    task = work_tree_manager.create_work_tree(config)
                    tasks.append((agent_id, task))
            
            import time
            start_time = time.time()
            
            work_trees = {}
            for agent_id, task in tasks:
                work_trees[agent_id] = await task
                
            end_time = time.time()
            total_duration = end_time - start_time
            
            # All work trees should be created
            assert len(work_trees) == agent_count
            
            # All paths should be unique (proper isolation)
            work_tree_paths = set(work_trees.values())
            assert len(work_tree_paths) == agent_count
            
            # Performance should be reasonable
            avg_time_per_work_tree = total_duration / agent_count
            assert avg_time_per_work_tree < 2  # Should average less than 2 seconds per work tree


@pytest.mark.asyncio 
class TestSecurityIntegrationScenarios:
    """Test security aspects in integration scenarios."""
    
    async def test_cross_agent_access_prevention(self):
        """Test that agents cannot access each other's work trees."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            work_tree_manager = WorkTreeManager(base_path)
            security_manager = GitHubSecurityManager()
            
            # Create work trees for two agents
            agent1_id = "agent_secure_1"
            agent2_id = "agent_secure_2"
            
            config1 = WorkTreeConfig(
                agent_id=agent1_id,
                repository_url="https://github.com/owner/repo.git"
            )
            
            config2 = WorkTreeConfig(
                agent_id=agent2_id,
                repository_url="https://github.com/owner/repo.git"
            )
            
            with patch('git.Repo.clone_from'):
                work_tree1 = await work_tree_manager.create_work_tree(config1)
                work_tree2 = await work_tree_manager.create_work_tree(config2)
                
                # Verify isolation
                isolation1 = await work_tree_manager.validate_work_tree_isolation(agent1_id)
                isolation2 = await work_tree_manager.validate_work_tree_isolation(agent2_id)
                
                assert isolation1["isolated"]
                assert isolation2["isolated"]
                assert isolation1["effectiveness_score"] == 1.0
                assert isolation2["effectiveness_score"] == 1.0
                
                # Verify agents can't access each other's directories
                assert agent1_id in str(work_tree1)
                assert agent2_id in str(work_tree2)
                assert str(work_tree1) != str(work_tree2)
    
    async def test_permission_enforcement_in_workflows(self):
        """Test that permission enforcement works in complete workflows."""
        security_manager = GitHubSecurityManager()
        
        # Test agent with limited permissions
        agent_id = "limited_agent"
        repository_id = str(uuid.uuid4())
        
        operations_to_test = [
            ("read_repository", True),  # Should be allowed
            ("create_issue", True),     # Should be allowed
            ("create_pull_request", True),  # Should be allowed
            ("delete_repository", False),   # Should be blocked
            ("transfer_repository", False)  # Should be blocked
        ]
        
        with patch('app.core.database.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock repository with standard permissions
            mock_repo = MagicMock()
            mock_repo.agent_permissions = {
                "read": "read",
                "issues": "write",
                "pull_requests": "write",
                "contents": "write"
            }
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_repo
            
            for operation, should_be_allowed in operations_to_test:
                result = await security_manager.validate_agent_operation(
                    agent_id, repository_id, operation
                )
                
                assert result["allowed"] == should_be_allowed, f"Operation {operation} permission check failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])