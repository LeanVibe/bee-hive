# OnAgentStart Hook - LeanVibe Agent Hive Observability

This hook captures agent initialization events for comprehensive lifecycle monitoring and startup diagnostics.

## Hook Configuration

```python
import asyncio
import json
import os
import platform
import psutil
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Hook API Configuration
HOOK_API_BASE_URL = os.getenv("LEANVIBE_HOOK_API_URL", "http://localhost:8000")
HOOK_API_TIMEOUT = int(os.getenv("LEANVIBE_HOOK_TIMEOUT", "5"))
ENABLE_HOOKS = os.getenv("LEANVIBE_ENABLE_HOOKS", "true").lower() == "true"

# Agent Configuration
SESSION_ID = os.getenv("LEANVIBE_SESSION_ID", str(uuid.uuid4()))
AGENT_ID = os.getenv("LEANVIBE_AGENT_ID", str(uuid.uuid4()))
AGENT_NAME = os.getenv("LEANVIBE_AGENT_NAME", "claude_agent")
AGENT_VERSION = os.getenv("LEANVIBE_AGENT_VERSION", "1.0.0")

async def send_hook_event(event_data: Dict[str, Any]) -> Optional[str]:
    """Send hook event to observability system."""
    if not ENABLE_HOOKS:
        return None
    
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=HOOK_API_TIMEOUT) as client:
            response = await client.post(
                f"{HOOK_API_BASE_URL}/api/v1/observability/hook-events",
                json=event_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("event_id")
            else:
                print(f"Hook API error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"Hook capture failed: {str(e)}")
        return None

def get_system_information() -> Dict[str, Any]:
    """Collect comprehensive system information for agent startup."""
    try:
        # Basic system info
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": os.sys.executable
            }
        }
        
        # Memory information
        try:
            memory = psutil.virtual_memory()
            system_info["memory"] = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent
            }
        except:
            system_info["memory"] = {"error": "unable_to_collect"}
        
        # CPU information
        try:
            system_info["cpu"] = {
                "count": psutil.cpu_count(),
                "count_physical": psutil.cpu_count(logical=False),
                "percent": psutil.cpu_percent(interval=1)
            }
        except:
            system_info["cpu"] = {"error": "unable_to_collect"}
        
        # Disk information
        try:
            disk = psutil.disk_usage('/')
            system_info["disk"] = {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 2)
            }
        except:
            system_info["disk"] = {"error": "unable_to_collect"}
        
        return system_info
        
    except Exception as e:
        return {"error": f"system_info_collection_failed: {str(e)}"}

def get_environment_information() -> Dict[str, Any]:
    """Collect non-sensitive environment information."""
    env_info = {
        "working_directory": os.getcwd(),
        "user": os.getenv("USER", "unknown"),
        "shell": os.getenv("SHELL", "unknown"),
        "path_count": len(os.getenv("PATH", "").split(":")),
        "environment_variables": {}
    }
    
    # Collect non-sensitive environment variables
    safe_env_patterns = [
        "LEANVIBE_", "CLAUDE_", "PATH", "HOME", "USER", "SHELL", 
        "LANG", "LC_", "TERM", "PWD", "EDITOR"
    ]
    
    for key, value in os.environ.items():
        if any(pattern in key for pattern in safe_env_patterns):
            # Redact sensitive values even in "safe" variables
            if any(sensitive in key.lower() for sensitive in ['password', 'token', 'secret', 'key']):
                env_info["environment_variables"][key] = "[REDACTED]"
            else:
                env_info["environment_variables"][key] = value
    
    return env_info

def get_agent_capabilities() -> Dict[str, Any]:
    """Detect and report agent capabilities and available tools."""
    capabilities = {
        "tools": [],
        "features": [],
        "integrations": []
    }
    
    # Check for common tools (this would be customized based on actual agent)
    potential_tools = [
        "bash", "python", "node", "git", "docker", "kubectl", 
        "redis-cli", "psql", "curl", "wget"
    ]
    
    for tool in potential_tools:
        try:
            import subprocess
            result = subprocess.run(['which', tool], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                capabilities["tools"].append(tool)
        except:
            pass
    
    # Check for Python packages
    try:
        import pkg_resources
        installed_packages = [pkg.project_name for pkg in pkg_resources.working_set]
        
        # Look for common packages that indicate capabilities
        capability_packages = {
            "requests": "http_client",
            "httpx": "async_http_client", 
            "redis": "redis_integration",
            "psycopg2": "postgresql_integration",
            "sqlalchemy": "orm_support",
            "fastapi": "web_api",
            "docker": "container_management",
            "kubernetes": "k8s_integration"
        }
        
        for package, feature in capability_packages.items():
            if package in installed_packages:
                capabilities["features"].append(feature)
                
    except:
        pass
    
    return capabilities

def run_startup_diagnostics() -> Dict[str, Any]:
    """Run basic startup diagnostics to detect potential issues."""
    diagnostics = {
        "status": "healthy",
        "checks": {},
        "warnings": [],
        "errors": []
    }
    
    # Check API connectivity
    try:
        import requests
        response = requests.get(f"{HOOK_API_BASE_URL}/health", timeout=5)
        diagnostics["checks"]["api_connectivity"] = response.status_code == 200
    except Exception as e:
        diagnostics["checks"]["api_connectivity"] = False
        diagnostics["errors"].append(f"API connectivity failed: {str(e)}")
    
    # Check memory availability
    try:
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            diagnostics["warnings"].append("High memory usage detected")
            diagnostics["status"] = "warning"
    except:
        pass
    
    # Check disk space
    try:
        disk = psutil.disk_usage('/')
        if (disk.free / disk.total) < 0.1:  # Less than 10% free
            diagnostics["warnings"].append("Low disk space detected")
            diagnostics["status"] = "warning"
    except:
        pass
    
    # Check required environment variables
    required_vars = ["LEANVIBE_SESSION_ID", "LEANVIBE_AGENT_ID"]
    for var in required_vars:
        if not os.getenv(var):
            diagnostics["errors"].append(f"Missing required environment variable: {var}")
            diagnostics["status"] = "error"
    
    return diagnostics

# Hook Implementation
def capture_agent_start(
    initialization_time_ms: Optional[int] = None,
    startup_context: Optional[Dict[str, Any]] = None
) -> None:
    """Capture agent startup event with comprehensive system and environment information."""
    try:
        # Collect comprehensive startup information
        system_info = get_system_information()
        environment_info = get_environment_information()
        capabilities = get_agent_capabilities()
        diagnostics = run_startup_diagnostics()
        
        # Prepare event data
        event_data = {
            "session_id": SESSION_ID,
            "agent_id": AGENT_ID,
            "event_type": "AGENT_START",
            "agent_name": AGENT_NAME,
            "agent_version": AGENT_VERSION,
            "initialization_time_ms": initialization_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": system_info,
            "environment_info": environment_info,
            "capabilities": capabilities,
            "diagnostics": diagnostics,
            "startup_context": startup_context or {},
            "metadata": {
                "startup_successful": diagnostics["status"] != "error",
                "warnings_count": len(diagnostics["warnings"]),
                "errors_count": len(diagnostics["errors"]),
                "tools_available": len(capabilities["tools"]),
                "features_enabled": len(capabilities["features"])
            }
        }
        
        # Send event asynchronously
        asyncio.create_task(send_hook_event(event_data))
        
        # Log startup status
        if diagnostics["status"] == "error":
            print(f"Agent startup completed with errors: {', '.join(diagnostics['errors'])}")
        elif diagnostics["status"] == "warning":
            print(f"Agent startup completed with warnings: {', '.join(diagnostics['warnings'])}")
        else:
            print(f"Agent {AGENT_NAME} started successfully")
        
        # Store startup timestamp for session tracking
        os.environ["LEANVIBE_AGENT_START_TIME"] = str(datetime.utcnow().timestamp())
        
    except Exception as e:
        print(f"OnAgentStart hook error: {str(e)}")

# Automatic startup detection
def auto_detect_startup():
    """Automatically detect agent startup if not already captured."""
    if not os.getenv("LEANVIBE_AGENT_STARTED"):
        capture_agent_start()
        os.environ["LEANVIBE_AGENT_STARTED"] = "true"

# Export hook function
__all__ = ["capture_agent_start", "auto_detect_startup"]
```

## Usage

This hook captures agent initialization events automatically or manually:

1. **Automatic Detection**: `auto_detect_startup()` - Called once per session
2. **Manual Capture**: `capture_agent_start(initialization_time_ms, context)`
3. **Integration**: Triggered during agent bootstrap process

## System Information Collected

- **Platform Details**: OS, version, architecture, processor
- **Python Environment**: Version, implementation, executable path
- **Resource Status**: Memory, CPU, disk usage and availability
- **Environment**: Working directory, user, shell, environment variables

## Capability Detection

- **Available Tools**: Shell commands, development tools (git, docker, etc.)
- **Python Packages**: Installed packages indicating specific capabilities
- **Integrations**: Redis, PostgreSQL, Kubernetes, etc.
- **Features**: HTTP clients, ORMs, web frameworks

## Startup Diagnostics

- **API Connectivity**: Tests connection to observability API
- **Resource Checks**: Memory and disk space availability
- **Environment Validation**: Required environment variables
- **Health Assessment**: Overall startup health status

## Performance

- **System Resource Monitoring**: Tracks CPU, memory, disk usage
- **Initialization Timing**: Measures agent startup time
- **Capability Assessment**: Evaluates available tools and features
- **Health Checks**: Validates system readiness

## Environment Variables

- `LEANVIBE_AGENT_NAME`: Agent name (default: claude_agent)
- `LEANVIBE_AGENT_VERSION`: Agent version (default: 1.0.0)
- `LEANVIBE_SESSION_ID`: Current session UUID
- `LEANVIBE_AGENT_ID`: Current agent UUID
- `LEANVIBE_HOOK_API_URL`: Hook API endpoint

## Event Schema

```json
{
  "session_id": "uuid",
  "agent_id": "uuid",
  "event_type": "AGENT_START",
  "agent_name": "string",
  "agent_version": "string",
  "initialization_time_ms": "number|null",
  "timestamp": "ISO 8601",
  "system_info": {
    "platform": "object",
    "python": "object",
    "memory": "object",
    "cpu": "object",
    "disk": "object"
  },
  "environment_info": {
    "working_directory": "string",
    "user": "string",
    "shell": "string",
    "environment_variables": "object"
  },
  "capabilities": {
    "tools": ["string"],
    "features": ["string"],
    "integrations": ["string"]
  },
  "diagnostics": {
    "status": "healthy|warning|error",
    "checks": "object",
    "warnings": ["string"],
    "errors": ["string"]
  },
  "startup_context": "object",
  "metadata": {
    "startup_successful": "boolean",
    "warnings_count": "number",
    "errors_count": "number",
    "tools_available": "number",
    "features_enabled": "number"
  }
}
```