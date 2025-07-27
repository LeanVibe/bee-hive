# External Tools Integration Guide

## Overview

The External Tools Integration Layer enables LeanVibe Agent Hive 2.0 agents to seamlessly interact with real-world development tools including Git, GitHub, Docker, and CI/CD pipelines. This revolutionary system provides secure, monitored access to external tools while maintaining comprehensive audit trails and safety guards.

## Supported Tools

### 1. Git Integration
- Repository cloning and management
- Branch creation and switching
- Commit and push operations
- Repository state monitoring
- Multi-repository coordination

### 2. GitHub API Integration
- Repository information retrieval
- Pull request creation and management
- Issue creation and tracking
- Collaborative development workflows
- Access control and permissions

### 3. Docker Integration
- Container image building
- Container lifecycle management
- Port mapping and volume mounts
- Environment variable configuration
- Resource monitoring

### 4. CI/CD Workflows
- Multi-step workflow execution
- Tool orchestration
- Automated deployment pipelines
- Quality gates and validation

## Getting Started

### Configuration

Add the following environment variables to your `.env` file:

```bash
# GitHub Integration
GITHUB_TOKEN=your_github_personal_access_token
GITHUB_API_URL=https://api.github.com

# Docker Configuration
DOCKER_HOST=unix:///var/run/docker.sock
DOCKER_REGISTRY=docker.io

# Git Configuration
GIT_DEFAULT_BRANCH=main
GIT_USER_NAME="Agent Hive Bot"
GIT_USER_EMAIL="agents@leenvibe.dev"

# Security Settings
EXTERNAL_TOOLS_SECURITY_LEVEL=moderate
ALLOW_SYSTEM_COMMANDS=false

# CI/CD Settings
CI_CD_ENABLED=true
DEPLOYMENT_TIMEOUT=1800
```

### Basic Usage Examples

#### Cloning a Repository

```python
# Via API
POST /api/v1/tools/git/clone
{
    "agent_id": "agent-123",
    "repository_url": "https://github.com/user/repo.git",
    "target_directory": "my-project",
    "branch": "main"
}

# Response
{
    "repository_id": "repo-456",
    "agent_id": "agent-123",
    "repository_url": "https://github.com/user/repo.git",
    "local_path": "/workspaces/agent-123/my-project",
    "branch": "main",
    "is_clean": true,
    "uncommitted_changes": [],
    "total_commits": 42
}
```

#### Creating a Pull Request

```python
# Via API
POST /api/v1/tools/github/pull-request
{
    "repository_id": "repo-456",
    "title": "Feature: Add new authentication system",
    "body": "This PR implements JWT-based authentication with refresh tokens.",
    "head_branch": "feature/auth-system",
    "base_branch": "main"
}

# Response
{
    "success": true,
    "pr_number": 123,
    "pr_url": "https://github.com/user/repo/pull/123",
    "message": "Pull request created successfully"
}
```

#### Building a Docker Image

```python
# Via API
POST /api/v1/tools/docker/build
{
    "agent_id": "agent-123",
    "dockerfile_path": "Dockerfile",
    "image_name": "my-app:latest",
    "context_path": ".",
    "build_args": {
        "NODE_ENV": "production",
        "API_VERSION": "v2"
    }
}

# Response
{
    "success": true,
    "output": "Successfully built image my-app:latest",
    "error": ""
}
```

## Workflow Templates

### Full Stack Deployment Workflow

```json
{
    "type": "full_stack_deployment",
    "description": "Clone repository, build Docker image, and deploy",
    "steps": [
        {
            "type": "git_clone",
            "parameters": {
                "repository_url": "https://github.com/user/my-app.git",
                "target_directory": "my-app",
                "branch": "main"
            }
        },
        {
            "type": "docker_build",
            "parameters": {
                "dockerfile_path": "Dockerfile",
                "image_name": "my-app:latest",
                "context_path": "."
            }
        },
        {
            "type": "docker_run",
            "parameters": {
                "image_name": "my-app:latest",
                "container_name": "my-app-container",
                "ports": {"8000": "8000"},
                "detached": true
            }
        }
    ]
}
```

### Feature Branch Workflow

```json
{
    "type": "feature_branch_workflow",
    "description": "Create feature branch, commit changes, and create PR",
    "steps": [
        {
            "type": "git_branch",
            "parameters": {
                "branch_name": "feature/new-dashboard",
                "checkout": true
            }
        },
        {
            "type": "git_commit",
            "parameters": {
                "message": "feat: implement responsive dashboard layout",
                "files": ["src/dashboard.js", "src/dashboard.css"]
            }
        },
        {
            "type": "git_push",
            "parameters": {
                "remote": "origin",
                "branch": "feature/new-dashboard"
            }
        },
        {
            "type": "github_pr",
            "parameters": {
                "title": "Feature: New Dashboard Layout",
                "body": "Implements a responsive dashboard with improved UX",
                "head_branch": "feature/new-dashboard",
                "base_branch": "main"
            }
        }
    ]
}
```

## Agent Integration

### Using External Tools in Agent Code

```python
from app.core.external_tools import external_tools

class DeveloperAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.tools = external_tools
    
    async def implement_feature(
        self,
        repository_url: str,
        feature_description: str,
        feature_name: str
    ):
        """Implement a new feature using external tools."""
        
        # 1. Clone repository
        success, repository, message = await self.tools.git.clone_repository(
            agent_id=self.agent_id,
            repository_url=repository_url,
            target_directory=feature_name
        )
        
        if not success:
            return f"Failed to clone repository: {message}"
        
        # 2. Create feature branch
        success, output, error = await self.tools.git.create_branch(
            repository_id=repository.id,
            branch_name=f"feature/{feature_name}",
            checkout=True
        )
        
        if not success:
            return f"Failed to create branch: {error}"
        
        # 3. Implement code changes here
        # ... code generation and file modifications ...
        
        # 4. Commit changes
        success, output, error = await self.tools.git.commit_changes(
            repository_id=repository.id,
            message=f"feat: {feature_description}"
        )
        
        if not success:
            return f"Failed to commit changes: {error}"
        
        # 5. Push to remote
        success, output, error = await self.tools.git.push_changes(
            repository_id=repository.id,
            branch=f"feature/{feature_name}"
        )
        
        if not success:
            return f"Failed to push changes: {error}"
        
        return f"Feature '{feature_name}' implemented successfully"
```

## Security Considerations

### Security Levels

1. **SAFE** - Read-only operations, basic calculations
2. **MODERATE** - File operations within workspace, Git operations
3. **RESTRICTED** - Network access, Docker operations, GitHub API
4. **DANGEROUS** - System modifications (requires human approval)

### Safety Guards

- All operations are logged and audited
- Commands executed in isolated workspaces
- Resource limits and timeouts enforced
- Sensitive data (tokens, keys) securely managed
- Human approval required for dangerous operations

### Access Control

```python
# Example: Restricting agent access to specific repositories
ALLOWED_REPOSITORIES = [
    "https://github.com/company/project-a.git",
    "https://github.com/company/project-b.git"
]

# Example: Role-based tool access
AGENT_PERMISSIONS = {
    "developer": ["git", "docker"],
    "devops": ["git", "docker", "github", "ci_cd"],
    "readonly": ["git_read"]
}
```

## Monitoring and Observability

### Operation Tracking

All external tool operations are tracked with:
- Operation ID and timestamp
- Agent ID and tool type
- Command executed and parameters
- Success/failure status and output
- Execution time and resource usage
- Security level and safety checks

### Metrics and Alerts

```python
# Example: Monitoring tool usage
GET /api/v1/tools/agents/{agent_id}/tools-summary

# Response
{
    "agent_id": "agent-123",
    "repositories": {
        "total": 3,
        "details": [...]
    },
    "containers": {
        "total": 2,
        "details": [...]
    },
    "recent_operations": {
        "total": 15,
        "details": [...]
    },
    "tools_available": {
        "git": true,
        "github": true,
        "docker": true
    }
}
```

## Error Handling and Recovery

### Common Error Scenarios

1. **Authentication Failures**
   - Invalid GitHub tokens
   - Git credential issues
   - Docker registry access

2. **Resource Constraints**
   - Disk space limitations
   - Memory constraints
   - Network timeouts

3. **Tool-Specific Errors**
   - Git merge conflicts
   - Docker build failures
   - GitHub API rate limits

### Recovery Strategies

```python
# Example: Automatic retry with exponential backoff
async def retry_operation(operation_func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await operation_func()
        except TemporaryError as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Best Practices

### For Agents

1. **Always check operation status** before proceeding
2. **Use appropriate security levels** for operations
3. **Handle errors gracefully** with fallback strategies
4. **Clean up resources** after operations
5. **Batch operations** when possible for efficiency

### For System Administrators

1. **Monitor tool usage** and resource consumption
2. **Set appropriate timeouts** and limits
3. **Regularly rotate tokens** and credentials
4. **Audit operation logs** for security
5. **Test disaster recovery** procedures

## Troubleshooting

### Common Issues

1. **Git operations fail**
   - Check Git configuration (user.name, user.email)
   - Verify repository permissions
   - Ensure network connectivity

2. **GitHub API errors**
   - Verify GitHub token validity
   - Check API rate limits
   - Confirm repository access permissions

3. **Docker operations fail**
   - Check Docker daemon status
   - Verify Dockerfile syntax
   - Ensure sufficient disk space

### Debugging Tools

```bash
# Check tool health
GET /api/v1/tools/health

# Get operation status
GET /api/v1/tools/operations/{operation_id}

# View agent tool summary
GET /api/v1/tools/agents/{agent_id}/tools-summary
```

## Contributing

To add new external tools:

1. Extend the `ToolType` enum
2. Implement the tool integration class
3. Add API endpoints
4. Update configuration and documentation
5. Add comprehensive tests

Example: Adding Terraform integration

```python
class TerraformIntegration:
    async def apply_infrastructure(self, config_path: str):
        # Implementation here
        pass
```

## Support

For issues and questions:
- Check the API documentation at `/docs`
- Review operation logs and status
- Contact the development team
- Submit issues via GitHub

---

*This guide covers the core functionality of the External Tools Integration Layer. For advanced usage and custom integrations, refer to the API documentation and source code.*