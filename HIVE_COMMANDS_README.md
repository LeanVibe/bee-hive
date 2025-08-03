# 🤖 Hive Slash Commands - Claude Code Integration

## Overview

The LeanVibe Agent Hive 2.0 now supports custom slash commands directly in Claude Code with the `hive:` prefix. These commands provide powerful meta-agent operations and autonomous development platform control through Claude Code's interface.

## Installation & Setup

The hive commands are automatically available when the platform is running. The command handler is located at:
```
~/.claude/commands/hive.py
```

## Available Commands

### `/hive:start` - Start Platform
Start the multi-agent platform with all services and spawn development team.

**Examples:**
```
/hive:start
/hive:start --quick
/hive:start --team-size=7
```

### `/hive:spawn` - Spawn Agent
Spawn a specific agent with given role and capabilities.

**Examples:**
```
/hive:spawn architect
/hive:spawn backend_developer --capabilities=api_development,database_design
/hive:spawn meta_agent
```

**Available Roles:**
- `product_manager` - Requirements analysis, project planning, documentation
- `architect` - System design, architecture planning, technology selection
- `backend_developer` - API development, database design, server logic
- `frontend_developer` - UI development, React, TypeScript
- `qa_engineer` - Test creation, quality assurance, validation
- `devops_engineer` - Deployment, infrastructure, monitoring
- `meta_agent` - Orchestration, meta coordination, system oversight

### `/hive:status` - Platform Status
Get comprehensive status of the multi-agent platform.

**Examples:**
```
/hive:status
/hive:status --detailed
/hive:status --agents-only
```

### `/hive:develop` - Autonomous Development
Start autonomous development with multi-agent coordination.

**Examples:**
```
/hive:develop "Build authentication API with JWT"
/hive:develop "Create user management system" --dashboard
/hive:develop "Build REST API with tests" --timeout=600
```

### `/hive:oversight` - Remote Dashboard
Open remote oversight dashboard for multi-agent monitoring.

**Examples:**
```
/hive:oversight
/hive:oversight --mobile-info
```

### `/hive:stop` - Stop Platform
Stop all agents and platform services.

**Examples:**
```
/hive:stop
/hive:stop --agents-only
/hive:stop --force
```

## Quick Test

You can test the commands are working:

```bash
# Test platform status
/hive:status

# Spawn a new agent
/hive:spawn architect

# Get detailed status
/hive:status --detailed

# Open dashboard
/hive:oversight
```

## Integration with Claude Code

These commands integrate seamlessly with Claude Code workflows:

1. **Project Development**: Use `/hive:develop` to start autonomous development
2. **Agent Management**: Use `/hive:spawn` to add specialized agents to your team
3. **Monitoring**: Use `/hive:status` and `/hive:oversight` for real-time monitoring
4. **Platform Control**: Use `/hive:start` and `/hive:stop` for platform lifecycle

## Behind the Scenes

The commands work by:
1. Claude Code recognizes `/hive:` prefix and routes to `~/.claude/commands/hive.py`
2. The command handler makes HTTP requests to the LeanVibe Agent Hive API
3. Results are formatted and displayed in Claude Code interface
4. Platform operations happen in real-time with live feedback

## Example Workflow

```
# Start the platform
/hive:start

# Check that everything is running
/hive:status --detailed

# Open oversight dashboard
/hive:oversight --mobile-info

# Start autonomous development
/hive:develop "Build a RESTful API for task management with authentication"

# Monitor progress remotely via dashboard
# Platform coordinates multiple AI agents to build the complete solution
```

## API Integration

The commands use the same REST API as the web interface:
- **Execute API**: `POST /api/hive/execute`
- **Status API**: `GET /api/hive/status`
- **Dashboard**: `http://localhost:8000/dashboard/`

## Success Confirmation

When working correctly, you should see:
- ✅ Commands execute successfully
- 🤖 Active agent counts displayed
- 🎯 System ready confirmations
- ⏱️ Execution time feedback
- 📋 Meaningful status messages

## Troubleshooting

If commands fail:
1. Check platform is running: `make status`
2. Start platform: `/hive:start`
3. Check API health: `curl http://localhost:8000/health`
4. Review logs: `make logs`

---

**🏆 Meta-Agent Operations Now Available in Claude Code!**

You now have direct access to the complete LeanVibe Agent Hive 2.0 autonomous development platform through Claude Code slash commands. Use `/hive:` prefix commands for powerful meta-agent operations and real autonomous development workflows.