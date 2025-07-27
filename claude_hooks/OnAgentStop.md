# OnAgentStop Hook - LeanVibe Agent Hive Observability

This hook captures agent shutdown events and session summary metrics for comprehensive lifecycle monitoring.

## Hook Configuration

```python
import asyncio
import json
import os
import psutil
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Hook API Configuration
HOOK_API_BASE_URL = os.getenv("LEANVIBE_HOOK_API_URL", "http://localhost:8000")
HOOK_API_TIMEOUT = int(os.getenv("LEANVIBE_HOOK_TIMEOUT", "5"))
ENABLE_HOOKS = os.getenv("LEANVIBE_ENABLE_HOOKS", "true").lower() == "true"

# Agent Configuration
SESSION_ID = os.getenv("LEANVIBE_SESSION_ID", str(uuid.uuid4()))
AGENT_ID = os.getenv("LEANVIBE_AGENT_ID", str(uuid.uuid4()))
AGENT_NAME = os.getenv("LEANVIBE_AGENT_NAME", "claude_agent")

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

def calculate_session_metrics() -> Dict[str, Any]:
    """Calculate comprehensive session metrics from startup to shutdown."""
    metrics = {
        "session_duration": None,
        "resource_usage": {},
        "performance": {},
        "activity": {}
    }
    
    try:
        # Calculate session duration
        start_time_str = os.getenv("LEANVIBE_AGENT_START_TIME")
        if start_time_str:
            start_time = datetime.fromtimestamp(float(start_time_str))
            end_time = datetime.utcnow()
            duration = end_time - start_time
            
            metrics["session_duration"] = {
                "total_seconds": duration.total_seconds(),
                "hours": duration.total_seconds() / 3600,
                "formatted": str(duration)
            }
        
        # Current resource usage
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            metrics["resource_usage"] = {
                "memory": {
                    "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
                    "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
                    "percent": process.memory_percent()
                },
                "cpu": {
                    "percent": process.cpu_percent(),
                    "times": process.cpu_times()._asdict()
                },
                "file_descriptors": process.num_fds() if hasattr(process, 'num_fds') else None
            }
        except Exception as e:
            metrics["resource_usage"]["error"] = str(e)
        
        # Performance metrics from environment tracking
        metrics["performance"] = {
            "tools_executed": int(os.getenv("LEANVIBE_TOOLS_EXECUTED", "0")),
            "errors_encountered": int(os.getenv("LEANVIBE_ERRORS_COUNT", "0")),
            "api_calls_made": int(os.getenv("LEANVIBE_API_CALLS", "0")),
            "average_tool_time": float(os.getenv("LEANVIBE_AVG_TOOL_TIME", "0.0"))
        }
        
        # Activity metrics
        metrics["activity"] = {
            "files_read": int(os.getenv("LEANVIBE_FILES_READ", "0")),
            "files_written": int(os.getenv("LEANVIBE_FILES_WRITTEN", "0")),
            "commands_executed": int(os.getenv("LEANVIBE_COMMANDS_EXECUTED", "0")),
            "network_requests": int(os.getenv("LEANVIBE_NETWORK_REQUESTS", "0"))
        }
        
    except Exception as e:
        metrics["calculation_error"] = str(e)
    
    return metrics

def get_final_system_state() -> Dict[str, Any]:
    """Collect final system state for comparison with startup state."""
    try:
        system_state = {
            "memory": {},
            "cpu": {},
            "disk": {},
            "processes": {}
        }
        
        # Memory state
        try:
            memory = psutil.virtual_memory()
            system_state["memory"] = {
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent,
                "cached_gb": round(memory.cached / (1024**3), 2) if hasattr(memory, 'cached') else None
            }
        except:
            system_state["memory"]["error"] = "collection_failed"
        
        # CPU state
        try:
            system_state["cpu"] = {
                "percent": psutil.cpu_percent(interval=1),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                "cpu_times": psutil.cpu_times()._asdict()
            }
        except:
            system_state["cpu"]["error"] = "collection_failed"
        
        # Disk state
        try:
            disk = psutil.disk_usage('/')
            system_state["disk"] = {
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 2)
            }
        except:
            system_state["disk"]["error"] = "collection_failed"
        
        # Process count
        try:
            system_state["processes"] = {
                "total_count": len(psutil.pids()),
                "python_processes": len([p for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()])
            }
        except:
            system_state["processes"]["error"] = "collection_failed"
        
        return system_state
        
    except Exception as e:
        return {"error": f"system_state_collection_failed: {str(e)}"}

def collect_session_artifacts() -> Dict[str, Any]:
    """Collect information about artifacts created during the session."""
    artifacts = {
        "files_created": [],
        "directories_created": [],
        "logs_generated": [],
        "temporary_files": []
    }
    
    try:
        # Check for common artifact patterns
        working_dir = os.getcwd()
        
        # Look for recently created files (within session timeframe)
        session_start = os.getenv("LEANVIBE_AGENT_START_TIME")
        if session_start:
            start_timestamp = float(session_start)
            
            for root, dirs, files in os.walk(working_dir):
                # Limit search depth to avoid performance issues
                if root.count(os.sep) - working_dir.count(os.sep) >= 3:
                    continue
                
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        stat = os.stat(file_path)
                        
                        # Check if file was created during session
                        if stat.st_ctime >= start_timestamp:
                            file_info = {
                                "path": file_path,
                                "size_bytes": stat.st_size,
                                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                "extension": os.path.splitext(file)[1]
                            }
                            
                            # Categorize files
                            if file.endswith('.log'):
                                artifacts["logs_generated"].append(file_info)
                            elif file.startswith('tmp_') or '/tmp/' in file_path:
                                artifacts["temporary_files"].append(file_info)
                            else:
                                artifacts["files_created"].append(file_info)
                    except:
                        continue  # Skip files we can't access
        
        # Limit artifact lists to prevent huge payloads
        for category in artifacts:
            if len(artifacts[category]) > 50:
                artifacts[category] = artifacts[category][:50]
                artifacts[f"{category}_truncated"] = True
        
    except Exception as e:
        artifacts["collection_error"] = str(e)
    
    return artifacts

def determine_shutdown_reason(context: Dict[str, Any]) -> Dict[str, Any]:
    """Determine and classify the reason for agent shutdown."""
    shutdown_info = {
        "reason": "unknown",
        "category": "normal",
        "planned": True,
        "clean_shutdown": True
    }
    
    # Check for explicit shutdown reasons
    if context.get("reason"):
        shutdown_info["reason"] = context["reason"]
        
        # Categorize shutdown reasons
        if context["reason"] in ["task_completed", "user_request", "session_timeout"]:
            shutdown_info["category"] = "normal"
            shutdown_info["planned"] = True
        elif context["reason"] in ["error", "exception", "crash"]:
            shutdown_info["category"] = "error"
            shutdown_info["planned"] = False
            shutdown_info["clean_shutdown"] = False
        elif context["reason"] in ["system_shutdown", "resource_limit", "timeout"]:
            shutdown_info["category"] = "system"
            shutdown_info["planned"] = False
    
    # Check environment for error indicators
    error_count = int(os.getenv("LEANVIBE_ERRORS_COUNT", "0"))
    if error_count > 0:
        shutdown_info["error_context"] = {
            "total_errors": error_count,
            "last_error_time": os.getenv("LEANVIBE_LAST_ERROR_TIME")
        }
    
    return shutdown_info

# Hook Implementation
def capture_agent_stop(
    reason: Optional[str] = None,
    shutdown_context: Optional[Dict[str, Any]] = None,
    cleanup_time_ms: Optional[int] = None
) -> None:
    """Capture agent shutdown event with comprehensive session summary."""
    try:
        # Prepare shutdown context
        context = shutdown_context or {}
        if reason:
            context["reason"] = reason
        
        # Collect comprehensive shutdown information
        session_metrics = calculate_session_metrics()
        final_system_state = get_final_system_state()
        session_artifacts = collect_session_artifacts()
        shutdown_info = determine_shutdown_reason(context)
        
        # Prepare event data
        event_data = {
            "session_id": SESSION_ID,
            "agent_id": AGENT_ID,
            "event_type": "AGENT_STOP",
            "agent_name": AGENT_NAME,
            "shutdown_reason": reason or "unspecified",
            "cleanup_time_ms": cleanup_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "session_metrics": session_metrics,
            "final_system_state": final_system_state,
            "session_artifacts": session_artifacts,
            "shutdown_info": shutdown_info,
            "shutdown_context": context,
            "metadata": {
                "clean_shutdown": shutdown_info["clean_shutdown"],
                "session_duration_hours": session_metrics.get("session_duration", {}).get("hours", 0),
                "artifacts_created": len(session_artifacts.get("files_created", [])),
                "errors_during_session": session_metrics.get("performance", {}).get("errors_encountered", 0),
                "tools_executed": session_metrics.get("performance", {}).get("tools_executed", 0)
            }
        }
        
        # Send event asynchronously
        asyncio.create_task(send_hook_event(event_data))
        
        # Log shutdown summary
        duration = session_metrics.get("session_duration", {}).get("formatted", "unknown")
        tools_executed = session_metrics.get("performance", {}).get("tools_executed", 0)
        errors = session_metrics.get("performance", {}).get("errors_encountered", 0)
        
        print(f"Agent {AGENT_NAME} shutdown complete:")
        print(f"  Duration: {duration}")
        print(f"  Tools executed: {tools_executed}")
        print(f"  Errors encountered: {errors}")
        print(f"  Shutdown reason: {reason or 'unspecified'}")
        
        # Clean up session tracking environment variables
        session_vars = [key for key in os.environ.keys() if key.startswith("LEANVIBE_")]
        for var in session_vars:
            if var not in ["LEANVIBE_SESSION_ID", "LEANVIBE_AGENT_ID"]:  # Keep core IDs
                os.environ.pop(var, None)
        
    except Exception as e:
        print(f"OnAgentStop hook error: {str(e)}")

# Automatic shutdown handler
def register_shutdown_handler():
    """Register automatic shutdown detection using signal handlers."""
    import signal
    import atexit
    
    def cleanup_handler():
        capture_agent_stop(reason="process_exit", shutdown_context={"automatic": True})
    
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        capture_agent_stop(
            reason=f"signal_{signal_name.lower()}", 
            shutdown_context={"signal": signal_name, "automatic": True}
        )
    
    # Register handlers
    atexit.register(cleanup_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# Export hook functions
__all__ = ["capture_agent_stop", "register_shutdown_handler"]
```

## Usage

This hook captures agent shutdown events automatically or manually:

1. **Manual Capture**: `capture_agent_stop(reason, context, cleanup_time_ms)`
2. **Automatic Registration**: `register_shutdown_handler()` - Captures signals and exit
3. **Integration**: Called during agent cleanup and shutdown processes

## Session Metrics

- **Duration Tracking**: Total session time from start to stop
- **Resource Usage**: Memory, CPU, file descriptor usage during session
- **Performance Metrics**: Tools executed, errors encountered, API calls made
- **Activity Summary**: Files read/written, commands executed, network requests

## Final System State

- **Memory State**: Available memory, cache usage, percent used
- **CPU State**: Usage percentage, load average, CPU times
- **Disk Usage**: Free space, utilization percentage
- **Process Count**: Total processes, Python processes running

## Session Artifacts

- **Files Created**: New files created during the session
- **Logs Generated**: Log files produced during execution
- **Temporary Files**: Temporary artifacts that may need cleanup
- **Directories**: New directories created

## Shutdown Classification

- **Reason Categories**: Normal, Error, System-initiated
- **Planned vs Unplanned**: Whether shutdown was expected
- **Clean Shutdown**: Whether cleanup completed successfully
- **Error Context**: Information about errors that occurred

## Performance

- **Resource Tracking**: Complete resource usage summary
- **Artifact Management**: Efficient collection of session artifacts
- **Signal Handling**: Graceful shutdown detection and capture
- **Cleanup**: Automatic environment variable cleanup

## Environment Variables

- `LEANVIBE_AGENT_START_TIME`: Session start timestamp
- `LEANVIBE_TOOLS_EXECUTED`: Tool execution counter
- `LEANVIBE_ERRORS_COUNT`: Error counter
- `LEANVIBE_FILES_READ/WRITTEN`: File operation counters
- Various tracking variables for session metrics

## Event Schema

```json
{
  "session_id": "uuid",
  "agent_id": "uuid",
  "event_type": "AGENT_STOP",
  "agent_name": "string",
  "shutdown_reason": "string",
  "cleanup_time_ms": "number|null",
  "timestamp": "ISO 8601",
  "session_metrics": {
    "session_duration": "object",
    "resource_usage": "object",
    "performance": "object",
    "activity": "object"
  },
  "final_system_state": {
    "memory": "object",
    "cpu": "object", 
    "disk": "object",
    "processes": "object"
  },
  "session_artifacts": {
    "files_created": ["object"],
    "logs_generated": ["object"],
    "temporary_files": ["object"]
  },
  "shutdown_info": {
    "reason": "string",
    "category": "normal|error|system",
    "planned": "boolean",
    "clean_shutdown": "boolean"
  },
  "shutdown_context": "object",
  "metadata": {
    "clean_shutdown": "boolean",
    "session_duration_hours": "number",
    "artifacts_created": "number",
    "errors_during_session": "number",
    "tools_executed": "number"
  }
}
```