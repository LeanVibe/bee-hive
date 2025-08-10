# Custom Commands System User Guide

## Phase 6.1: Multi-Agent Workflow Commands for LeanVibe Agent Hive 2.0

The Custom Commands System enables creation and execution of sophisticated multi-agent workflows through declarative command definitions. This guide provides comprehensive documentation for using the system effectively.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)  
3. [Command Definition](#command-definition)
4. [Agent Requirements](#agent-requirements)
5. [Workflow Design](#workflow-design)
6. [Security Policies](#security-policies)
7. [API Reference](#api-reference)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The Custom Commands System provides:

- **Declarative Workflow Definition**: Define complex multi-step workflows using YAML/JSON
- **Intelligent Agent Selection**: Automatic agent assignment based on capabilities and workload
- **Secure Execution Environment**: Sandboxed execution with resource limits and security policies
- **Real-time Monitoring**: Comprehensive observability and progress tracking
- **Failure Recovery**: Automatic retry and reassignment capabilities
- **Integration Ready**: Seamless integration with existing Phase 5 infrastructure

### Key Features

✅ **Multi-Agent Orchestration**: Coordinate multiple specialized agents  
✅ **Dependency Management**: Define complex step dependencies and parallel execution  
✅ **Resource Management**: CPU, memory, and execution time limits  
✅ **Security Enforcement**: Role-based permissions and operation restrictions  
✅ **Performance Optimization**: Intelligent load balancing and task distribution  
✅ **Comprehensive Monitoring**: Real-time metrics and detailed execution analytics  

## Getting Started

First complete environment setup via `docs/GETTING_STARTED.md`, then return here for custom commands specifics.

### Prerequisites

- LeanVibe Agent Hive 2.0 with Phase 5 infrastructure
- At least one active agent in the system
- Appropriate user permissions for command creation/execution

### Quick Start

1. **Define Your Command** (YAML format):

```yaml
name: "hello-world-workflow"
version: "1.0.0"
description: "Simple hello world multi-agent workflow"
category: "examples"
tags: ["tutorial", "hello-world"]

agents:
  - role: "backend-engineer"
    specialization: ["general"]
    required_capabilities: ["text_processing"]

workflow:
  - step: "generate_greeting"
    agent: "backend-engineer"
    task: "Generate a personalized greeting message"
    outputs: ["greeting.txt"]
    timeout_minutes: 5
```

2. **Register the Command**:

```bash
curl -X POST "http://localhost:8000/api/v1/custom-commands/commands" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "definition": { ... },
    "validate_agents": true,
    "dry_run": false
  }'
```

3. **Execute the Command**:

```bash
curl -X POST "http://localhost:8000/api/v1/custom-commands/execute" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "command_name": "hello-world-workflow",
    "parameters": {"name": "World"},
    "priority": "medium"
  }'
```

## Command Definition

### Basic Structure

```yaml
name: "command-name"           # Unique command identifier
version: "1.0.0"               # Semantic version
description: "Command purpose" # Human-readable description
category: "development"        # Command category
tags: ["tag1", "tag2"]        # Searchable tags

# Agent requirements
agents:
  - role: "backend-engineer"
    specialization: ["python", "fastapi"]
    min_experience_level: 3
    required_capabilities: ["coding", "testing"]

# Workflow definition
workflow:
  - step: "step_name"
    agent: "backend-engineer"
    task: "Task description"
    depends_on: ["previous_step"]
    outputs: ["file1.py", "file2.md"]
    timeout_minutes: 60

# Security configuration
security_policy:
  allowed_operations: ["file_read", "file_write"]
  network_access: false
  resource_limits:
    max_memory_mb: 1024
    max_cpu_time_seconds: 3600
```

### Command Metadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Unique command name (1-100 chars) |
| `version` | string | ✅ | Semantic version (e.g., "1.2.3") |
| `description` | string | ✅ | Command description (max 500 chars) |
| `category` | string | ❌ | Category for organization (default: "general") |
| `tags` | array | ❌ | Tags for discovery and filtering |
| `author` | string | ❌ | Command author identifier |
| `documentation_url` | string | ❌ | Link to detailed documentation |

## Agent Requirements

### Supported Agent Roles

The system supports the following standard agent roles:

- **`backend-engineer`**: Server-side development, APIs, databases
- **`frontend-builder`**: UI/UX development, client-side applications  
- **`qa-test-guardian`**: Testing, validation, quality assurance
- **`devops-specialist`**: Infrastructure, deployment, operations
- **`data-analyst`**: Data processing, analytics, reporting
- **`security-auditor`**: Security review, vulnerability assessment
- **`product-manager`**: Requirements analysis, project coordination
- **`technical-writer`**: Documentation, guides, specifications

### Agent Requirement Specification

```yaml
agents:
  - role: "backend-engineer"
    specialization: ["python", "fastapi", "postgresql"]
    min_experience_level: 3          # 1-5 scale
    required_capabilities: 
      - "api_development"
      - "database_design"
      - "test_automation"
    resource_requirements:
      min_memory_mb: 512
      preferred_cpu_cores: 2
```

### Capability Matching

Agents are matched based on:

1. **Role Match**: Primary role alignment
2. **Specialization**: Technical domain expertise
3. **Capabilities**: Specific skills and tools
4. **Experience Level**: Minimum proficiency requirement
5. **Availability**: Current workload and health status

## Workflow Design

### Step Types

#### Sequential Steps
```yaml
workflow:
  - step: "step1"
    task: "First task"
  - step: "step2" 
    task: "Second task"
    depends_on: ["step1"]  # Runs after step1 completes
```

#### Parallel Steps
```yaml
workflow:
  - step: "parallel_container"
    step_type: "parallel"
    parallel:
      - step: "task_a"
        task: "Parallel task A"
      - step: "task_b" 
        task: "Parallel task B"
```

#### Conditional Steps
```yaml
workflow:
  - step: "conditional_step"
    step_type: "conditional"
    task: "Conditional task"
    conditions:
      execute_if: "${previous_step.result.status} == 'success'"
      skip_if: "${workflow.context.skip_optional} == true"
```

### Dependency Management

Define complex dependencies between workflow steps:

```yaml
workflow:
  - step: "setup"
    task: "Initialize environment"
    
  - step: "build_backend"
    task: "Build backend services"
    depends_on: ["setup"]
    
  - step: "build_frontend"
    task: "Build frontend application"  
    depends_on: ["setup"]
    
  - step: "integration_test"
    task: "Run integration tests"
    depends_on: ["build_backend", "build_frontend"]
    
  - step: "deploy"
    task: "Deploy to production"
    depends_on: ["integration_test"]
```

### Input/Output Specification

```yaml
workflow:
  - step: "analyze_code"
    task: "Analyze codebase for issues"
    inputs: ["src/", "tests/"]           # Required inputs
    outputs: ["analysis_report.json"]    # Expected outputs
    
  - step: "generate_fixes"
    task: "Generate code fixes"
    depends_on: ["analyze_code"]
    inputs: ["analysis_report.json"]     # Uses previous output
    outputs: ["fixes.patch", "fix_summary.md"]
```

## Security Policies

### Security Configuration

```yaml
security_policy:
  # Operations allowed during execution
  allowed_operations:
    - "file_read"
    - "file_write" 
    - "code_execution"
    - "api_calls"
    
  # File system restrictions
  restricted_paths:
    - "/etc/"
    - "/root/"
    - "/sys/"
    
  # Network access control
  network_access: false
  
  # Resource limitations
  resource_limits:
    max_memory_mb: 2048
    max_cpu_time_seconds: 7200
    max_disk_space_mb: 1024
    max_processes: 5
    
  # Approval requirements
  requires_approval: false
  audit_level: "standard"
```

### Security Levels

| Level | Description | Requirements |
|-------|-------------|--------------|
| **Low** | Basic restrictions | Standard resource limits |
| **Medium** | Moderate security | No network access, limited file system |
| **High** | Strict security | Sandboxed execution, approval required |
| **Critical** | Maximum security | Manual approval, extensive auditing |

## API Reference

### Authentication

All API endpoints require authentication via Bearer token:

```bash
Authorization: Bearer YOUR_JWT_TOKEN
```

### Core Endpoints

#### Create Command
```http
POST /api/v1/custom-commands/commands
Content-Type: application/json

{
  "definition": { ... },
  "validate_agents": true,
  "dry_run": false
}
```

#### Execute Command  
```http
POST /api/v1/custom-commands/execute
Content-Type: application/json

{
  "command_name": "my-workflow",
  "command_version": "1.0.0",
  "parameters": {"key": "value"},
  "context": {"environment": "production"},
  "priority": "high"
}
```

#### Get Execution Status
```http
GET /api/v1/custom-commands/executions/{execution_id}/status
```

#### List Commands
```http
GET /api/v1/custom-commands/commands?category=development&limit=50
```

### Response Formats

#### Execution Result
```json
{
  "execution_id": "uuid",
  "command_name": "workflow-name", 
  "command_version": "1.0.0",
  "status": "completed",
  "start_time": "2025-01-26T10:00:00Z",
  "end_time": "2025-01-26T10:15:00Z", 
  "total_execution_time_seconds": 900.0,
  "step_results": [
    {
      "step_id": "step1",
      "status": "completed",
      "agent_id": "agent-uuid",
      "execution_time_seconds": 300.0,
      "outputs": {"result": "success"}
    }
  ],
  "final_outputs": {"workflow_result": "completed"},
  "total_steps": 3,
  "completed_steps": 3,
  "failed_steps": 0
}
```

## Usage Examples

### Example 1: Feature Development Workflow

```yaml
name: "feature-development-complete"
version: "2.1.0"
description: "Complete feature development workflow with testing and deployment"
category: "development"
tags: ["feature", "development", "ci-cd"]

agents:
  - role: "product-manager"
    specialization: ["requirements", "planning"]
    required_capabilities: ["analysis", "documentation"]
  - role: "backend-engineer" 
    specialization: ["python", "fastapi", "postgresql"]
    required_capabilities: ["api_development", "database_design", "testing"]
  - role: "frontend-builder"
    specialization: ["react", "typescript", "css"]
    required_capabilities: ["ui_development", "responsive_design"]
  - role: "qa-test-guardian"
    specialization: ["automation", "testing"]
    required_capabilities: ["test_automation", "quality_assurance"]
  - role: "devops-specialist"
    specialization: ["docker", "kubernetes", "ci-cd"]
    required_capabilities: ["deployment", "monitoring"]

workflow:
  # Phase 1: Analysis and Planning
  - step: "analyze_requirements"
    agent: "product-manager"
    task: "Analyze feature requirements and create detailed specifications"
    outputs: ["requirements.md", "acceptance_criteria.md", "user_stories.json"]
    timeout_minutes: 60
    
  - step: "create_technical_design"
    agent: "backend-engineer"
    task: "Create technical design and architecture documentation"
    depends_on: ["analyze_requirements"]
    inputs: ["requirements.md", "acceptance_criteria.md"]
    outputs: ["technical_design.md", "api_specification.yaml", "database_schema.sql"]
    timeout_minutes: 90

  # Phase 2: Parallel Development
  - step: "implement_backend_api"
    agent: "backend-engineer"
    task: "Implement backend API endpoints and business logic"
    depends_on: ["create_technical_design"]
    inputs: ["technical_design.md", "api_specification.yaml", "database_schema.sql"]
    outputs: ["api_code/", "unit_tests/", "migration_scripts/"]
    timeout_minutes: 240
    
  - step: "implement_frontend_ui"
    agent: "frontend-builder"
    task: "Implement frontend user interface and user experience"
    depends_on: ["create_technical_design"]  
    inputs: ["technical_design.md", "user_stories.json"]
    outputs: ["ui_components/", "frontend_tests/", "style_guide.md"]
    timeout_minutes: 180

  # Phase 3: Integration and Testing
  - step: "integration_testing"
    agent: "qa-test-guardian"
    task: "Perform comprehensive integration testing"
    depends_on: ["implement_backend_api", "implement_frontend_ui"]
    inputs: ["api_code/", "ui_components/", "acceptance_criteria.md"]
    outputs: ["test_results.json", "coverage_report.html", "bug_report.md"]
    timeout_minutes: 120
    
  - step: "performance_testing"
    agent: "qa-test-guardian"
    task: "Execute performance and load testing"
    depends_on: ["integration_testing"]
    inputs: ["api_code/", "test_results.json"]
    outputs: ["performance_report.json", "load_test_results.html"]
    timeout_minutes: 90

  # Phase 4: Deployment
  - step: "prepare_deployment"
    agent: "devops-specialist"
    task: "Prepare deployment configuration and infrastructure"
    depends_on: ["performance_testing"]
    inputs: ["api_code/", "ui_components/", "database_schema.sql"]
    outputs: ["docker_images/", "k8s_manifests/", "deployment_guide.md"]
    timeout_minutes: 60
    
  - step: "deploy_to_staging"
    agent: "devops-specialist"
    task: "Deploy feature to staging environment"
    depends_on: ["prepare_deployment"]
    inputs: ["docker_images/", "k8s_manifests/"]
    outputs: ["deployment_status.json", "staging_urls.txt"]
    timeout_minutes: 30
    
  - step: "production_deployment"
    agent: "devops-specialist"
    task: "Deploy feature to production environment"
    depends_on: ["deploy_to_staging"]
    inputs: ["docker_images/", "k8s_manifests/", "deployment_status.json"]
    outputs: ["production_status.json", "monitoring_dashboard.url"]
    timeout_minutes: 45

security_policy:
  allowed_operations: ["file_read", "file_write", "code_execution", "api_calls"]
  network_access: true
  resource_limits:
    max_memory_mb: 4096
    max_cpu_time_seconds: 14400  # 4 hours
    max_disk_space_mb: 2048
  requires_approval: true
  audit_level: "detailed"
```

### Example 2: Data Processing Pipeline

```yaml
name: "data-processing-pipeline"
version: "1.3.0"
description: "Automated data ingestion, processing, and analysis pipeline"
category: "data"
tags: ["etl", "analytics", "automation"]

agents:
  - role: "data-analyst"
    specialization: ["python", "pandas", "sql"]
    required_capabilities: ["data_extraction", "data_transformation", "statistical_analysis"]

workflow:
  - step: "data_extraction"
    agent: "data-analyst"
    task: "Extract data from multiple sources and validate schema"
    outputs: ["raw_data.csv", "extraction_log.json", "data_quality_report.md"]
    timeout_minutes: 45
    
  - step: "data_cleaning"
    agent: "data-analyst"
    task: "Clean and preprocess raw data"
    depends_on: ["data_extraction"]
    inputs: ["raw_data.csv", "data_quality_report.md"]
    outputs: ["clean_data.csv", "cleaning_report.json"]
    timeout_minutes: 60
    
  - step: "data_analysis"
    agent: "data-analyst"
    task: "Perform statistical analysis and generate insights"
    depends_on: ["data_cleaning"]
    inputs: ["clean_data.csv"]
    outputs: ["analysis_results.json", "visualizations/", "insights_report.pdf"]
    timeout_minutes: 90

security_policy:
  allowed_operations: ["file_read", "file_write", "database_access"]
  network_access: true
  resource_limits:
    max_memory_mb: 8192
    max_cpu_time_seconds: 7200
```

### Example 3: Security Audit Workflow

```yaml
name: "security-audit-comprehensive"
version: "2.0.0"
description: "Comprehensive security audit and vulnerability assessment"
category: "security"
tags: ["security", "audit", "compliance"]

agents:
  - role: "security-auditor"
    specialization: ["security_scanning", "penetration_testing", "compliance"]
    min_experience_level: 4
    required_capabilities: ["vulnerability_assessment", "compliance_checking", "reporting"]

workflow:
  - step: "static_analysis"
    agent: "security-auditor"
    task: "Perform static code analysis for security vulnerabilities"
    outputs: ["static_analysis_report.json", "vulnerability_list.csv"]
    timeout_minutes: 60
    
  - step: "dependency_scan"
    agent: "security-auditor"
    task: "Scan dependencies for known vulnerabilities"
    outputs: ["dependency_report.json", "cve_list.txt"]
    timeout_minutes: 30
    
  - step: "penetration_testing"
    agent: "security-auditor"
    task: "Conduct penetration testing and security assessment"
    depends_on: ["static_analysis", "dependency_scan"]
    inputs: ["static_analysis_report.json", "dependency_report.json"]
    outputs: ["pentest_report.pdf", "security_findings.json"]
    timeout_minutes: 180
    
  - step: "compliance_check"
    agent: "security-auditor"
    task: "Verify compliance with security standards and regulations"
    depends_on: ["penetration_testing"]
    inputs: ["pentest_report.pdf", "security_findings.json"]
    outputs: ["compliance_report.pdf", "remediation_plan.md"]
    timeout_minutes: 90

security_policy:
  allowed_operations: ["security_scanning", "network_probing", "file_read"]
  network_access: true
  resource_limits:
    max_memory_mb: 2048
    max_cpu_time_seconds: 18000  # 5 hours
  requires_approval: true
  audit_level: "maximum"
```

## Best Practices

### Command Design

1. **Modular Workflows**: Break complex processes into discrete, reusable steps
2. **Clear Dependencies**: Explicitly define step dependencies for proper sequencing
3. **Resource Planning**: Set realistic timeout and resource limits
4. **Error Handling**: Design workflows to handle and recover from failures
5. **Output Standards**: Use consistent output formats across similar commands

### Performance Optimization

1. **Parallel Execution**: Leverage parallel steps where possible
2. **Agent Specialization**: Match agents to their areas of expertise
3. **Resource Efficiency**: Optimize memory and CPU usage
4. **Batch Processing**: Group related operations for efficiency
5. **Caching**: Reuse intermediate results when appropriate

### Security Guidelines

1. **Principle of Least Privilege**: Grant minimal required permissions
2. **Input Validation**: Validate all external inputs and parameters
3. **Secure Communications**: Use encrypted channels for sensitive data
4. **Audit Trails**: Maintain comprehensive logs for security events
5. **Regular Reviews**: Periodically review and update security policies

### Monitoring and Debugging

1. **Comprehensive Logging**: Include detailed logging in workflow steps
2. **Progress Tracking**: Monitor execution progress and performance metrics
3. **Alert Configuration**: Set up alerts for failures and performance issues
4. **Debug Information**: Include debug outputs for troubleshooting
5. **Metrics Collection**: Track execution metrics for optimization

## Troubleshooting

### Common Issues

#### Command Registration Failures

**Problem**: Command validation fails during registration
```
ValidationError: Agent requirements not met
```

**Solution**: 
1. Check agent availability: `GET /api/v1/agents`
2. Verify agent capabilities match requirements
3. Use dry run to test validation: `"dry_run": true`

#### Execution Timeouts

**Problem**: Workflow steps exceed timeout limits
```
ExecutionError: Step timeout exceeded (3600s)
```

**Solution**:
1. Increase step timeout: `timeout_minutes: 120`
2. Optimize step logic for performance
3. Break large steps into smaller components
4. Check agent resource availability

#### Agent Unavailability

**Problem**: No suitable agents found for task distribution
```
DistributionError: No agents available for role 'backend-engineer'
```

**Solution**:
1. Verify agents are active and healthy
2. Check agent capability matching
3. Review resource requirements
4. Consider expanding agent requirements

#### Security Violations

**Problem**: Security policy violations during execution
```
SecurityError: Operation 'network_access' not allowed
```

**Solution**:
1. Review security policy configuration
2. Update allowed operations list
3. Request elevated permissions if needed
4. Use more restrictive execution environment

### Debugging Tools

#### Execution Status Monitoring
```bash
# Get real-time execution status
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/custom-commands/executions/$EXECUTION_ID/status"
```

#### System Metrics
```bash
# Get comprehensive system metrics
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/custom-commands/metrics"
```

#### Agent Workload Analysis
```bash
# Check agent availability and workload
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/custom-commands/agents/workload"
```

### Performance Tuning

#### Optimization Strategies

1. **Load Balancing**: Use `DistributionStrategy.HYBRID` for optimal agent selection
2. **Concurrent Limits**: Adjust `max_concurrent_executions` based on system capacity
3. **Resource Allocation**: Monitor and adjust resource limits based on actual usage
4. **Caching**: Implement caching for frequently used command definitions
5. **Database Optimization**: Ensure proper indexing for command queries

#### Monitoring Recommendations

1. **Response Time**: Monitor average execution time trends
2. **Success Rate**: Track command success rates and failure patterns
3. **Resource Usage**: Monitor CPU, memory, and disk usage patterns
4. **Agent Health**: Track agent availability and performance metrics
5. **System Load**: Monitor concurrent execution levels and queuing

## Support and Resources

### Documentation
- [API Reference](./API_REFERENCE.md)
- [Architecture Guide](./ARCHITECTURE.md)
- [Security Documentation](./SECURITY.md)

### Community
- [GitHub Issues](https://github.com/leanvibe/agent-hive/issues)
- [Discussion Forum](https://github.com/leanvibe/agent-hive/discussions)
- [Slack Community](https://leanvibe-community.slack.com)

### Professional Support
- Enterprise Support: enterprise@leanvibe.com
- Consulting Services: consulting@leanvibe.com
- Training Programs: training@leanvibe.com