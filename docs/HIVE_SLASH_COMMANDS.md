# 🤖 Hive Slash Commands - Meta-Agent Operations

## Overview

Custom slash commands with the `hive:` prefix for LeanVibe Agent Hive 2.0 meta-agent operations and advanced platform control. These commands provide Claude Code-style custom functionality specifically designed for autonomous development orchestration.

## Command Format

All hive commands follow the format: `/hive:<command> [arguments] [--options]`

## Available Commands

### 🚀 `/hive:start` - Platform Initialization
Start the multi-agent platform with all services and spawn development team.

**Usage:**
```
/hive:start [--quick] [--team-size=5]
```

**Examples:**
- `/hive:start` - Start platform with default 5-agent team
- `/hive:start --quick` - Quick start with existing configuration
- `/hive:start --team-size=7` - Start with custom team size

**Returns:**
- Platform status and agent initialization results
- Team composition with agent IDs and roles
- Ready-for-development confirmation

### 🤖 `/hive:spawn` - Agent Creation
Spawn a specific agent with the given role and capabilities.

**Usage:**
```
/hive:spawn <role> [--capabilities=cap1,cap2]
```

**Valid Roles:**
- `product_manager` - Requirements analysis, project planning, documentation
- `architect` - System design, architecture planning, technology selection  
- `backend_developer` - API development, database design, server logic
- `frontend_developer` - UI development, React, TypeScript
- `qa_engineer` - Test creation, quality assurance, validation
- `devops_engineer` - Deployment, infrastructure, monitoring
- `meta_agent` - Orchestration, meta coordination, system oversight

**Examples:**
- `/hive:spawn architect` - Spawn architect agent with default capabilities
- `/hive:spawn backend_developer --capabilities=api_development,database_design` - Custom capabilities
- `/hive:spawn meta_agent` - Spawn meta-agent for system oversight

**Returns:**
- Agent ID and role confirmation
- Assigned capabilities list
- Spawn success status

### 📊 `/hive:status` - Platform Status
Get comprehensive status of the multi-agent platform.

**Usage:**
```
/hive:status [--detailed] [--agents-only]
```

**Options:**
- `--detailed` - Include full agent information and capabilities
- `--agents-only` - Show only agent status (exclude system health)

**Examples:**
- `/hive:status` - Basic platform status
- `/hive:status --detailed` - Full status with agent details
- `/hive:status --agents-only` - Focus on agent information

**Returns:**
- Platform active status and agent count
- System health and component status
- Agent details and team composition (if detailed)

### 💻 `/hive:develop` - Autonomous Development
Start autonomous development with multi-agent coordination.

**Usage:**
```
/hive:develop <project_description> [--dashboard] [--timeout=300]
```

**Options:**
- `--dashboard` - Open oversight dashboard during development
- `--timeout=N` - Set execution timeout in seconds (default: 300)

**Examples:**
- `/hive:develop "Build authentication API with JWT"` - Start development project
- `/hive:develop "Create user management system" --dashboard` - With dashboard monitoring
- `/hive:develop "Build REST API with tests" --timeout=600` - Extended timeout

**Returns:**
- Development execution status
- Agent involvement details
- Output and execution results
- Dashboard access information (if enabled)

### 🎛️ `/hive:oversight` - Remote Dashboard
Open remote oversight dashboard for multi-agent monitoring.

**Usage:**
```
/hive:oversight [--mobile-info]
```

**Options:**
- `--mobile-info` - Include mobile access information and URLs

**Examples:**
- `/hive:oversight` - Open dashboard in browser
- `/hive:oversight --mobile-info` - Include mobile access details

**Returns:**
- Dashboard URL and open status
- Mobile access information (if requested)
- Remote oversight feature list

### 🛑 `/hive:stop` - Platform Shutdown
Stop all agents and platform services.

**Usage:**
```
/hive:stop [--force] [--agents-only]
```

**Options:**
- `--force` - Force shutdown without graceful cleanup
- `--agents-only` - Stop only agents, keep platform services running

**Examples:**
- `/hive:stop` - Graceful shutdown of entire platform
- `/hive:stop --agents-only` - Stop agents but keep infrastructure
- `/hive:stop --force` - Force immediate shutdown

**Returns:**
- Shutdown status and agent count stopped
- Platform services shutdown status
- Cleanup completion confirmation

## API Integration

### REST API Endpoints

**Execute Command:**
```http
POST /api/hive/execute
Content-Type: application/json

{
  "command": "/hive:start --team-size=5",
  "context": {
    "user_id": "optional",
    "session_id": "optional"
  }
}
```

**Quick Execute:**
```http
POST /api/hive/quick/status?args=--detailed
```

**List Commands:**
```http
GET /api/hive/list
```

**Get Command Help:**
```http
GET /api/hive/help/develop
```

### Response Format

All commands return a standardized response:

```json
{
  "success": true,
  "command": "/hive:status",
  "result": {
    "success": true,
    "platform_active": true,
    "agent_count": 5,
    "system_ready": true,
    "message": "Platform operational"
  },
  "execution_time_ms": 150.5
}
```

## Integration Examples

### Claude Code Integration

```javascript
// Execute hive command from Claude Code
const response = await fetch('http://localhost:8000/api/hive/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    command: '/hive:develop "Build authentication API"'
  })
});

const result = await response.json();
console.log('Development started:', result.success);
```

### Python Integration

```python
import requests

def execute_hive_command(command: str):
    response = requests.post(
        'http://localhost:8000/api/hive/execute',
        json={'command': command}
    )
    return response.json()

# Start platform
result = execute_hive_command('/hive:start')
print(f"Platform ready: {result['result']['system_ready']}")

# Spawn specific agent  
result = execute_hive_command('/hive:spawn architect')
print(f"Agent ID: {result['result']['agent_id']}")
```

### Bash/CLI Integration

```bash
# Execute via curl
curl -X POST http://localhost:8000/api/hive/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "/hive:status --detailed"}' | jq .

# Quick execute
curl -X POST "http://localhost:8000/api/hive/quick/start?args=--team-size=7"
```

## Command Development

### Creating Custom Commands

To add new hive commands, extend the `HiveSlashCommand` base class:

```python
class HiveCustomCommand(HiveSlashCommand):
    def __init__(self):
        super().__init__(
            name="custom",
            description="Custom meta-agent operation",
            usage="/hive:custom <args> [--options]"
        )
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None):
        # Implementation here
        return {"success": True, "message": "Custom command executed"}

# Register the command
registry = get_hive_command_registry()
registry.register_command(HiveCustomCommand())
```

### Command Validation

Commands can implement argument validation:

```python
def validate_args(self, args: List[str]) -> bool:
    """Validate command arguments."""
    if not args or len(args) < 1:
        return False
    return True
```

## Security Considerations

- Commands execute with platform privileges
- Validate all user inputs and arguments
- Implement proper error handling
- Log all command executions
- Consider rate limiting for production use

## Monitoring and Logging

All command executions are logged with:
- Command text and arguments
- Execution time and status
- User context (if provided)
- Results and errors

Monitor command usage via:
- Platform logs
- API endpoint metrics
- Dashboard monitoring
- Health check endpoints

---

**Meta-Agent Operations Enabled** 🤖  
The hive slash commands system provides powerful meta-agent capabilities for autonomous development platform control and orchestration.