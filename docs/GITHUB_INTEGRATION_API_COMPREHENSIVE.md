# GitHub Integration System - API Reference

## Overview

LeanVibe Agent Hive 2.0 provides comprehensive GitHub integration with full REST/GraphQL API support, automated pull request management, intelligent code review, branch management with conflict resolution, and real-time webhook processing for seamless multi-agent development workflows.

## Base URL

```
Production: https://api.leanvibe.dev/api/v1/github
Development: http://localhost:8000/api/v1/github
```

## Authentication

All GitHub integration endpoints require authentication with appropriate permissions:

```bash
Authorization: Bearer <agent-token>
X-GitHub-Token: <github-personal-access-token>
```

## Core GitHub Integration Features

### Repository Management

#### List Repositories
```http
GET /api/v1/github/repositories
Authorization: Bearer <token>
X-GitHub-Token: <github-token>
```

**Query Parameters:**
- `owner` (optional): Filter by repository owner
- `type` (optional): `public`, `private`, `all` (default: `all`)
- `sort` (optional): `created`, `updated`, `pushed`, `full_name` (default: `updated`)
- `limit` (optional): Number of repositories to return (default: 50, max: 100)

**Response:**
```json
{
  "repositories": [
    {
      "id": 123456789,
      "name": "agent-hive",
      "full_name": "leanvibe-dev/agent-hive",
      "owner": {
        "login": "leanvibe-dev",
        "type": "Organization"
      },
      "private": false,
      "description": "Enterprise-grade autonomous multi-agent development platform",
      "clone_url": "https://github.com/leanvibe-dev/agent-hive.git",
      "ssh_url": "git@github.com:leanvibe-dev/agent-hive.git",
      "default_branch": "main",
      "created_at": "2025-01-15T10:00:00Z",
      "updated_at": "2025-07-29T14:30:00Z",
      "pushed_at": "2025-07-29T14:25:00Z",
      "language": "Python",
      "languages": {
        "Python": 75.2,
        "JavaScript": 15.8,
        "TypeScript": 8.1,
        "Dockerfile": 0.9
      },
      "open_issues_count": 5,
      "stargazers_count": 42,
      "watchers_count": 42,
      "forks_count": 7
    }
  ],
  "total": 12,
  "page": 1,
  "limit": 50
}
```

#### Get Repository Details
```http
GET /api/v1/github/repositories/{owner}/{repo}
Authorization: Bearer <token>
X-GitHub-Token: <github-token>
```

**Response:**
```json
{
  "repository": {
    "id": 123456789,
    "name": "agent-hive",
    "full_name": "leanvibe-dev/agent-hive",
    "owner": {
      "login": "leanvibe-dev",
      "type": "Organization"
    },
    "private": false,
    "description": "Enterprise-grade autonomous multi-agent development platform",
    "clone_url": "https://github.com/leanvibe-dev/agent-hive.git",
    "ssh_url": "git@github.com:leanvibe-dev/agent-hive.git",
    "default_branch": "main",
    "branches": [
      {
        "name": "main",
        "protected": true,
        "commit": {
          "sha": "abc123def456",
          "message": "feat: Enhanced security auth system implementation"
        }
      },
      {
        "name": "feature/documentation-update",
        "protected": false,
        "commit": {
          "sha": "def456ghi789",
          "message": "docs: Update API documentation for enterprise features"
        }
      }
    ],
    "permissions": {
      "admin": true,
      "maintain": true,
      "push": true,
      "triage": true,
      "pull": true
    },
    "security": {
      "branch_protection_enabled": true,
      "required_status_checks": ["ci/tests", "security/scan"],
      "enforce_admins": true,
      "required_pull_request_reviews": {
        "required_approving_review_count": 2,
        "dismiss_stale_reviews": true,
        "require_code_owner_reviews": true
      }
    }
  }
}
```

### Branch Management

#### Create Branch
```http
POST /api/v1/github/repositories/{owner}/{repo}/branches
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "branch_name": "feature/ai-enhanced-workflows",
  "from_branch": "main",
  "agent_context": {
    "task_id": "task_550e8400-e29b-41d4-a716-446655440000",
    "purpose": "Implement AI-enhanced multi-agent workflows",
    "estimated_duration": "4 hours"
  }
}
```

**Response:**
```json
{
  "branch": {
    "name": "feature/ai-enhanced-workflows",
    "sha": "ghi789jkl012",
    "url": "https://api.github.com/repos/leanvibe-dev/agent-hive/git/refs/heads/feature/ai-enhanced-workflows",
    "created_at": "2025-07-29T14:35:00Z",
    "created_from": "main",
    "agent_metadata": {
      "created_by_agent": "agent_550e8400-e29b-41d4-a716-446655440000",
      "task_id": "task_550e8400-e29b-41d4-a716-446655440000",
      "work_tree_id": "wt_660e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

#### Get Branch Protection
```http
GET /api/v1/github/repositories/{owner}/{repo}/branches/{branch}/protection
Authorization: Bearer <token>
X-GitHub-Token: <github-token>
```

**Response:**
```json
{
  "protection": {
    "enabled": true,
    "required_status_checks": {
      "strict": true,
      "contexts": ["ci/tests", "security/scan", "performance/benchmarks"]
    },
    "enforce_admins": true,
    "required_pull_request_reviews": {
      "required_approving_review_count": 2,
      "dismiss_stale_reviews": true,
      "require_code_owner_reviews": true,
      "dismissal_restrictions": {
        "users": ["security-admin"],
        "teams": ["security-team"]
      }
    },
    "restrictions": {
      "users": ["lead-agent"],
      "teams": ["core-team"],
      "apps": ["leanvibe-agent-hive"]
    },
    "allow_force_pushes": false,
    "allow_deletions": false,
    "block_creations": false
  }
}
```

#### Merge Branches with Conflict Resolution
```http
POST /api/v1/github/repositories/{owner}/{repo}/branches/merge
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "base": "main",
  "head": "feature/ai-enhanced-workflows",
  "commit_message": "feat: Add AI-enhanced multi-agent workflow capabilities",
  "merge_strategy": "squash",
  "conflict_resolution": {
    "strategy": "intelligent_merge",
    "prefer_head_on_conflicts": ["src/", "docs/"],
    "prefer_base_on_conflicts": ["package.json", "requirements.txt"],
    "manual_review_required": ["security/", "core/"]
  },
  "automated_testing": {
    "run_before_merge": true,
    "required_checks": ["unit_tests", "integration_tests", "security_scan"]
  }
}
```

**Response:**
```json
{
  "merge": {
    "sha": "jkl012mno345",
    "merged": true,
    "message": "feat: Add AI-enhanced multi-agent workflow capabilities",
    "merge_strategy": "squash",
    "conflicts_resolved": [
      {
        "file": "app/core/workflow_engine.py",
        "resolution_strategy": "intelligent_merge",
        "confidence": 0.95,
        "changes": {
          "lines_added": 45,
          "lines_removed": 12,
          "functions_modified": ["process_workflow", "handle_agent_coordination"]
        }
      }
    ],
    "automated_tests": {
      "status": "passed",
      "duration": "3m 42s",
      "checks": [
        {"name": "unit_tests", "status": "passed", "duration": "1m 15s"},
        {"name": "integration_tests", "status": "passed", "duration": "2m 05s"},
        {"name": "security_scan", "status": "passed", "duration": "22s"}
      ]
    },
    "merged_at": "2025-07-29T14:45:00Z"
  }
}
```

### Pull Request Management

#### Create Pull Request
```http
POST /api/v1/github/repositories/{owner}/{repo}/pulls
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "title": "feat: Implement advanced multi-agent coordination system",
  "head": "feature/ai-enhanced-workflows",
  "base": "main",
  "body": "## Summary\n\nThis PR implements advanced multi-agent coordination capabilities including:\n\n- Real-time agent graph visualization\n- Intelligent task routing and load balancing\n- Enhanced context sharing between agents\n- Performance monitoring and optimization\n\n## Changes\n\n- Added `CoordinationEngine` class for agent orchestration\n- Implemented WebSocket-based real-time updates\n- Enhanced security with RBAC integration\n- Added comprehensive test coverage (95%+)\n\n## Testing\n\n- [x] Unit tests (100% coverage)\n- [x] Integration tests\n- [x] Performance benchmarks\n- [x] Security audit\n\n## Documentation\n\n- [x] API documentation updated\n- [x] User guide created\n- [x] Architecture diagrams updated\n\nCloses #123, #124, #125",
  "draft": false,
  "maintainer_can_modify": true,
  "agent_metadata": {
    "created_by_agent": "agent_550e8400-e29b-41d4-a716-446655440000",
    "task_ids": ["task_550e8400-e29b-41d4-a716-446655440000", "task_660e8400-e29b-41d4-a716-446655440000"],
    "estimated_review_time": "30 minutes",
    "complexity_score": 8.5,
    "files_changed": 25,
    "lines_added": 1250,
    "lines_removed": 85
  }
}
```

**Response:**
```json
{
  "pull_request": {
    "id": 987654321,
    "number": 142,
    "title": "feat: Implement advanced multi-agent coordination system",
    "body": "## Summary\n\nThis PR implements advanced multi-agent coordination capabilities...",
    "state": "open",
    "draft": false,
    "head": {
      "ref": "feature/ai-enhanced-workflows",
      "sha": "jkl012mno345"
    },
    "base": {
      "ref": "main",
      "sha": "abc123def456"
    },
    "user": {
      "login": "leanvibe-agent-bot",
      "type": "Bot"
    },
    "created_at": "2025-07-29T14:50:00Z",
    "updated_at": "2025-07-29T14:50:00Z",
    "html_url": "https://github.com/leanvibe-dev/agent-hive/pull/142",
    "diff_url": "https://github.com/leanvibe-dev/agent-hive/pull/142.diff",
    "patch_url": "https://github.com/leanvibe-dev/agent-hive/pull/142.patch",
    "mergeable": true,
    "mergeable_state": "clean",
    "merged": false,
    "merge_commit_sha": null,
    "assignees": ["human-reviewer"],
    "requested_reviewers": ["senior-developer", "security-team"],
    "labels": [
      {"name": "enhancement", "color": "a2eeef"},
      {"name": "ai-feature", "color": "0052cc"},
      {"name": "priority-high", "color": "d93f0b"}
    ],
    "milestone": {
      "title": "Q3 2025 - Advanced Features",
      "number": 5
    },
    "status_checks": {
      "total_count": 5,
      "pending": 2,
      "success": 3,
      "failure": 0,
      "checks": [
        {"name": "ci/tests", "status": "success", "conclusion": "success"},
        {"name": "security/scan", "status": "success", "conclusion": "success"},
        {"name": "performance/benchmarks", "status": "success", "conclusion": "success"},
        {"name": "ci/build", "status": "in_progress", "conclusion": null},
        {"name": "docs/validate", "status": "in_progress", "conclusion": null}
      ]
    }
  }
}
```

#### Get Pull Request Reviews
```http
GET /api/v1/github/repositories/{owner}/{repo}/pulls/{pull_number}/reviews
Authorization: Bearer <token>
X-GitHub-Token: <github-token>
```

**Response:**
```json
{
  "reviews": [
    {
      "id": 123456789,
      "user": {
        "login": "senior-developer",
        "type": "User"
      },
      "body": "Excellent implementation of the coordination system! The code is well-structured and the test coverage is comprehensive. A few minor suggestions:\n\n1. Consider adding more detailed error handling in the WebSocket connection logic\n2. The performance benchmarks look great - significant improvement over the previous version\n3. Documentation is thorough and well-written\n\nApproved with minor suggestions.",
      "state": "APPROVED",
      "html_url": "https://github.com/leanvibe-dev/agent-hive/pull/142#pullrequestreview-123456789",
      "submitted_at": "2025-07-29T15:30:00Z",
      "author_association": "COLLABORATOR",
      "comments": [
        {
          "id": 987654321,
          "path": "app/core/coordination_engine.py",
          "line": 45,
          "body": "Consider adding a timeout parameter for WebSocket connections to handle network issues gracefully.",
          "created_at": "2025-07-29T15:25:00Z"
        }
      ]
    },
    {
      "id": 123456790,
      "user": {
        "login": "leanvibe-code-review-bot",
        "type": "Bot"
      },
      "body": "## Automated Code Review\n\n### ✅ Passed Checks\n- Code style compliance: 100%\n- Test coverage: 97.5%\n- Security scan: No vulnerabilities found\n- Performance impact: +15% improvement\n- Documentation coverage: 95%\n\n### 🔍 Suggestions\n- Consider extracting constants for magic numbers in `coordination_engine.py:78`\n- Add type hints to function parameters in `websocket_manager.py:123`\n\n### 📊 Metrics\n- Cyclomatic complexity: 3.2 (Good)\n- Maintainability index: 85 (Excellent)\n- Code duplication: 0.5% (Excellent)\n\nOverall assessment: **EXCELLENT** - Ready for merge after addressing minor suggestions.",
      "state": "APPROVED",
      "submitted_at": "2025-07-29T15:10:00Z",
      "author_association": "NONE"
    }
  ],
  "total": 2,
  "summary": {
    "approved": 2,
    "changes_requested": 0,
    "commented": 0,
    "dismissed": 0
  }
}
```

### Automated Code Review

#### Trigger Code Review
```http
POST /api/v1/github/repositories/{owner}/{repo}/pulls/{pull_number}/reviews/automated
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "review_type": "comprehensive",
  "focus_areas": [
    "security",
    "performance",
    "code_quality",
    "documentation",
    "test_coverage"
  ],
  "ai_analysis": {
    "enabled": true,
    "model": "claude-3.5-sonnet",
    "context_aware": true,
    "previous_reviews": true
  },
  "quality_gates": {
    "min_test_coverage": 90,
    "max_complexity": 10,
    "security_scan_required": true,
    "performance_regression_check": true
  }
}
```

**Response:**
```json
{
  "review": {
    "id": 456789123,
    "status": "completed",
    "started_at": "2025-07-29T15:00:00Z",
    "completed_at": "2025-07-29T15:05:00Z",
    "duration": "5m 12s",
    "overall_score": 8.7,
    "recommendation": "APPROVE",
    "analysis": {
      "security": {
        "score": 9.2,
        "issues": 0,
        "vulnerabilities": [],
        "recommendations": [
          "Consider adding input validation for WebSocket messages",
          "Excellent use of RBAC integration"
        ]
      },
      "performance": {
        "score": 8.9,
        "improvements": "+15% response time improvement",
        "regressions": [],
        "benchmarks": {
          "api_response_time": "42ms (↓8ms)",
          "memory_usage": "125MB (↓15MB)",
          "cpu_utilization": "12% (↓3%)"
        }
      },
      "code_quality": {
        "score": 8.5,
        "complexity": 3.2,
        "maintainability": 85,
        "duplication": 0.5,
        "style_compliance": 100
      },
      "test_coverage": {
        "score": 9.5,
        "percentage": 97.5,
        "new_lines_covered": 98.2,
        "critical_paths_covered": 100
      },
      "documentation": {
        "score": 8.8,
        "coverage": 95,
        "api_docs_updated": true,
        "readme_updated": true,
        "inline_comments": "Excellent"
      }
    },
    "detailed_feedback": [
      {
        "file": "app/core/coordination_engine.py",
        "line": 78,
        "type": "suggestion",
        "severity": "low",
        "message": "Consider extracting magic number 30 to a named constant",
        "suggestion": "Define WEBSOCKET_TIMEOUT = 30 at module level"
      },
      {
        "file": "app/core/websocket_manager.py",
        "line": 123,
        "type": "enhancement",
        "severity": "low",
        "message": "Add type hints for better code maintainability",
        "suggestion": "def handle_message(self, message: Dict[str, Any]) -> None:"
      }
    ],
    "ai_insights": [
      "The implementation follows excellent architectural patterns with clear separation of concerns",
      "Strong use of dependency injection makes the code highly testable",
      "The WebSocket implementation is robust with proper error handling",
      "Integration with existing security systems is seamless and well-designed"
    ]
  }
}
```

### Issue Management

#### Create Issue
```http
POST /api/v1/github/repositories/{owner}/{repo}/issues
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "title": "Enhancement: Add real-time performance monitoring dashboard",
  "body": "## Description\n\nWe need a real-time dashboard to monitor agent performance metrics including:\n\n- Response times for each agent\n- Memory and CPU usage\n- Task completion rates\n- Error rates and types\n- Communication latency between agents\n\n## Acceptance Criteria\n\n- [ ] Dashboard displays real-time metrics with <1s latency\n- [ ] Supports multiple agent monitoring simultaneously\n- [ ] Provides alerting for performance degradation\n- [ ] Includes historical data visualization\n- [ ] Mobile-responsive design\n\n## Technical Requirements\n\n- WebSocket integration for real-time updates\n- Integration with existing observability hooks\n- Prometheus metrics collection\n- Grafana dashboard configuration\n\n## Priority\n\nHigh - This will significantly improve operational visibility",
  "labels": ["enhancement", "dashboard", "monitoring", "priority-high"],
  "assignees": ["performance-team"],
  "milestone": 5,
  "agent_metadata": {
    "created_by_agent": "agent_550e8400-e29b-41d4-a716-446655440000",
    "related_tasks": ["task_770e8400-e29b-41d4-a716-446655440000"],
    "estimated_effort": "3 days",
    "complexity": "medium"
  }
}
```

**Response:**
```json
{
  "issue": {
    "id": 2468135790,
    "number": 156,
    "title": "Enhancement: Add real-time performance monitoring dashboard",
    "body": "## Description\n\nWe need a real-time dashboard to monitor agent performance metrics...",
    "state": "open",
    "user": {
      "login": "leanvibe-agent-bot",
      "type": "Bot"
    },
    "labels": [
      {"name": "enhancement", "color": "a2eeef"},
      {"name": "dashboard", "color": "0052cc"},
      {"name": "monitoring", "color": "fbca04"},
      {"name": "priority-high", "color": "d93f0b"}
    ],
    "assignees": [
      {"login": "performance-team", "type": "Team"}
    ],
    "milestone": {
      "title": "Q3 2025 - Advanced Features",
      "number": 5
    },
    "created_at": "2025-07-29T15:40:00Z",
    "updated_at": "2025-07-29T15:40:00Z",
    "html_url": "https://github.com/leanvibe-dev/agent-hive/issues/156",
    "comments": 0,
    "closed_at": null,
    "author_association": "NONE"
  }
}
```

#### Link Issues to Pull Requests
```http
POST /api/v1/github/repositories/{owner}/{repo}/issues/{issue_number}/link
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "pull_request": 142,
  "relationship": "closes",
  "automated_closure": true,
  "verification_required": false
}
```

**Response:**
```json
{
  "link": {
    "issue_number": 156,
    "pull_request_number": 142,
    "relationship": "closes",
    "linked_at": "2025-07-29T15:45:00Z",
    "automated_closure": true,
    "status": "active"
  }
}
```

### Webhook Management

#### Register Webhook
```http
POST /api/v1/github/repositories/{owner}/{repo}/webhooks
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "name": "web",
  "config": {
    "url": "https://api.leanvibe.dev/webhooks/github",
    "content_type": "json",
    "secret": "webhook_secret_key_here",
    "insecure_ssl": "0"
  },
  "events": [
    "push",
    "pull_request",
    "pull_request_review",
    "issues",
    "issue_comment",
    "commit_comment",
    "create",
    "delete",
    "deployment",
    "deployment_status",
    "release",
    "status",
    "check_run",
    "check_suite"
  ],
  "active": true,
  "agent_processing": {
    "enabled": true,
    "filters": {
      "branches": ["main", "develop", "feature/*"],
      "authors": ["leanvibe-agent-bot"],
      "file_patterns": ["app/**/*.py", "docs/**/*.md"]
    },
    "routing": {
      "pull_requests": "agent_pr_handler",
      "issues": "agent_issue_handler",
      "deployments": "agent_deployment_handler"
    }
  }
}
```

**Response:**
```json
{
  "webhook": {
    "id": 12345678,
    "name": "web",
    "url": "https://api.github.com/repos/leanvibe-dev/agent-hive/hooks/12345678",
    "ping_url": "https://api.github.com/repos/leanvibe-dev/agent-hive/hooks/12345678/pings",
    "test_url": "https://api.github.com/repos/leanvibe-dev/agent-hive/hooks/12345678/test",
    "active": true,
    "events": [
      "push", "pull_request", "pull_request_review", "issues",
      "issue_comment", "commit_comment", "create", "delete",
      "deployment", "deployment_status", "release", "status",
      "check_run", "check_suite"
    ],
    "config": {
      "url": "https://api.leanvibe.dev/webhooks/github",
      "content_type": "json",
      "insecure_ssl": "0"
    },
    "created_at": "2025-07-29T15:50:00Z",
    "updated_at": "2025-07-29T15:50:00Z",
    "delivery_stats": {
      "successful_deliveries": 0,
      "failed_deliveries": 0,
      "last_delivery_at": null,
      "average_response_time": null
    }
  }
}
```

#### Webhook Event Processing
```http
POST /api/v1/github/webhooks/process
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Signature-256: sha256=<signature>

{
  "action": "opened",
  "number": 142,
  "pull_request": {
    "id": 987654321,
    "number": 142,
    "title": "feat: Implement advanced multi-agent coordination system",
    "head": {
      "ref": "feature/ai-enhanced-workflows",
      "sha": "jkl012mno345"
    },
    "base": {
      "ref": "main",
      "sha": "abc123def456"
    },
    "user": {
      "login": "leanvibe-agent-bot",
      "type": "Bot"
    }
  },
  "repository": {
    "id": 123456789,
    "name": "agent-hive",
    "full_name": "leanvibe-dev/agent-hive"
  }
}
```

**Response:**
```json
{
  "processing": {
    "event_id": "evt_880e8400-e29b-41d4-a716-446655440000",
    "event_type": "pull_request.opened",
    "status": "processed",
    "processed_at": "2025-07-29T15:52:00Z",
    "processing_time": "250ms",
    "actions_triggered": [
      {
        "type": "automated_code_review",
        "status": "initiated",
        "agent_id": "agent_review_bot",
        "estimated_completion": "2025-07-29T15:57:00Z"
      },
      {
        "type": "security_scan",
        "status": "queued",
        "scanner": "security_scanner_v2",
        "priority": "high"
      },
      {
        "type": "performance_benchmark",
        "status": "scheduled",
        "benchmark_suite": "full_performance_suite",
        "scheduled_for": "2025-07-29T16:00:00Z"
      }
    ],
    "notifications_sent": [
      {
        "type": "slack",
        "channel": "#development",
        "message": "New PR opened: feat: Implement advanced multi-agent coordination system"
      },
      {
        "type": "email",
        "recipients": ["team-lead@company.com"],
        "subject": "PR Review Required: #142"
      }
    ]
  }
}
```

### Repository Analytics

#### Get Repository Insights
```http
GET /api/v1/github/repositories/{owner}/{repo}/analytics
  ?start_date=2025-07-01
  &end_date=2025-07-29
  &metrics=commits,pull_requests,issues,contributors,activity
Authorization: Bearer <token>
X-GitHub-Token: <github-token>
```

**Response:**
```json
{
  "analytics": {
    "period": {
      "start": "2025-07-01T00:00:00Z",
      "end": "2025-07-29T23:59:59Z",
      "days": 29
    },
    "commits": {
      "total": 145,
      "by_agent": 89,
      "by_human": 56,
      "average_per_day": 5.0,
      "top_contributors": [
        {"author": "leanvibe-agent-bot", "commits": 89, "type": "agent"},
        {"author": "senior-developer", "commits": 34, "type": "human"},
        {"author": "security-team", "commits": 22, "type": "human"}
      ]
    },
    "pull_requests": {
      "opened": 25,
      "merged": 22,
      "closed": 3,
      "average_merge_time": "4.2 hours",
      "by_agent": {
        "opened": 18,
        "merged": 16,
        "average_merge_time": "3.8 hours"
      },
      "by_human": {
        "opened": 7,
        "merged": 6,
        "average_merge_time": "5.2 hours"
      }
    },
    "issues": {
      "opened": 12,
      "closed": 15,
      "average_resolution_time": "2.1 days",
      "by_type": {
        "bug": 4,
        "enhancement": 6,
        "documentation": 2
      }
    },
    "code_quality": {
      "test_coverage": 94.7,
      "code_complexity": 3.2,
      "documentation_coverage": 89.3,
      "security_score": 9.1
    },
    "agent_activity": {
      "most_active_agent": "agent_550e8400-e29b-41d4-a716-446655440000",
      "agent_contributions": 61.4,
      "automated_reviews": 45,
      "automated_fixes": 12,
      "performance_improvements": 8
    }
  }
}
```

## Security Features

### Repository Security Scanning
```http
POST /api/v1/github/repositories/{owner}/{repo}/security/scan
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "scan_type": "comprehensive",
  "targets": {
    "code": true,
    "dependencies": true,
    "secrets": true,
    "container_images": true,
    "infrastructure": true
  },
  "branch": "feature/ai-enhanced-workflows",
  "priority": "high",
  "agent_context": {
    "scan_reason": "pre_merge_security_check",
    "pull_request": 142
  }
}
```

**Response:**
```json
{
  "scan": {
    "id": "scan_990e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "started_at": "2025-07-29T16:00:00Z",
    "completed_at": "2025-07-29T16:03:30Z",
    "duration": "3m 30s",
    "overall_score": 9.2,
    "risk_level": "low",
    "results": {
      "code_analysis": {
        "vulnerabilities": 0,
        "warnings": 2,
        "info": 5,
        "files_scanned": 125,
        "rules_applied": 450
      },
      "dependency_analysis": {
        "total_dependencies": 87,
        "vulnerable_dependencies": 0,
        "outdated_dependencies": 3,
        "license_issues": 0
      },
      "secret_detection": {
        "secrets_found": 0,
        "false_positives": 2,
        "files_scanned": 125
      },
      "container_security": {
        "base_image_vulnerabilities": 0,
        "configuration_issues": 1,
        "best_practices_violations": 0
      }
    },
    "recommendations": [
      "Update 3 outdated dependencies to latest versions",
      "Add security headers to container configuration",
      "Consider adding additional input validation in WebSocket handlers"
    ],
    "compliance": {
      "pci_dss": "compliant",
      "soc2": "compliant",
      "gdpr": "compliant",
      "hipaa": "not_applicable"
    }
  }
}
```

## GraphQL API Support

### Execute GraphQL Query
```http
POST /api/v1/github/graphql
Content-Type: application/json
Authorization: Bearer <token>
X-GitHub-Token: <github-token>

{
  "query": "query GetRepositoryDetails($owner: String!, $name: String!) {\n  repository(owner: $owner, name: $name) {\n    name\n    description\n    stargazerCount\n    forkCount\n    issues(states: OPEN) {\n      totalCount\n    }\n    pullRequests(states: OPEN) {\n      totalCount\n      nodes {\n        number\n        title\n        author {\n          login\n        }\n        reviewRequests {\n          totalCount\n        }\n        reviews {\n          totalCount\n        }\n      }\n    }\n    releases(first: 5, orderBy: {field: CREATED_AT, direction: DESC}) {\n      nodes {\n        name\n        tagName\n        publishedAt\n        isPrerelease\n      }\n    }\n  }\n}",
  "variables": {
    "owner": "leanvibe-dev",
    "name": "agent-hive"
  }
}
```

**Response:**
```json
{
  "data": {
    "repository": {
      "name": "agent-hive",
      "description": "Enterprise-grade autonomous multi-agent development platform",
      "stargazerCount": 42,
      "forkCount": 7,
      "issues": {
        "totalCount": 5
      },
      "pullRequests": {
        "totalCount": 3,
        "nodes": [
          {
            "number": 142,
            "title": "feat: Implement advanced multi-agent coordination system",
            "author": {
              "login": "leanvibe-agent-bot"
            },
            "reviewRequests": {
              "totalCount": 2
            },
            "reviews": {
              "totalCount": 2
            }
          }
        ]
      },
      "releases": {
        "nodes": [
          {
            "name": "v2.0.0-beta.1",
            "tagName": "v2.0.0-beta.1",
            "publishedAt": "2025-07-25T14:00:00Z",
            "isPrerelease": true
          }
        ]
      }
    }
  }
}
```

## Rate Limiting and Error Handling

### Rate Limits
- **Repository operations**: 5000 requests/hour per token
- **Pull request operations**: 2000 requests/hour per token
- **Webhook operations**: 1000 requests/hour per token
- **Security scans**: 100 scans/day per repository

### Error Response Format
```json
{
  "error": {
    "code": "GITHUB_API_ERROR",
    "message": "GitHub API rate limit exceeded",
    "details": {
      "github_error": {
        "message": "API rate limit exceeded for user ID 12345",
        "documentation_url": "https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting"
      },
      "retry_after": 3600,
      "limit": 5000,
      "remaining": 0,
      "reset": 1690636800
    }
  }
}
```

This comprehensive GitHub integration system provides enterprise-grade repository management, automated code review, intelligent branch management, and real-time webhook processing for seamless multi-agent development workflows.