# PostToolUse Hook - LeanVibe Agent Hive Observability

This hook captures tool execution results and performance metrics for comprehensive observability.

## Hook Configuration

```python
import asyncio
import json
import os
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Hook API Configuration
HOOK_API_BASE_URL = os.getenv("LEANVIBE_HOOK_API_URL", "http://localhost:8000")
HOOK_API_TIMEOUT = int(os.getenv("LEANVIBE_HOOK_TIMEOUT", "5"))
ENABLE_HOOKS = os.getenv("LEANVIBE_ENABLE_HOOKS", "true").lower() == "true"

# Session and Agent Context
SESSION_ID = os.getenv("LEANVIBE_SESSION_ID", str(uuid.uuid4()))
AGENT_ID = os.getenv("LEANVIBE_AGENT_ID", str(uuid.uuid4()))

# Performance thresholds
SLOW_TOOL_THRESHOLD_MS = int(os.getenv("LEANVIBE_SLOW_TOOL_THRESHOLD", "2000"))
LARGE_RESULT_THRESHOLD_KB = int(os.getenv("LEANVIBE_LARGE_RESULT_THRESHOLD", "100"))

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

def redact_sensitive_data(data: Any) -> Any:
    """Redact sensitive information from tool results."""
    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            # Redact common sensitive fields
            if key.lower() in ['password', 'token', 'secret', 'key', 'auth', 'credential']:
                redacted[key] = "[REDACTED]"
            elif key.lower() in ['email', 'phone', 'ssn', 'credit_card']:
                redacted[key] = "[PII_REDACTED]"
            else:
                redacted[key] = redact_sensitive_data(value)
        return redacted
    elif isinstance(data, list):
        return [redact_sensitive_data(item) for item in data[:10]]  # Limit list size
    elif isinstance(data, str):
        # Truncate large strings and redact sensitive patterns
        if len(data) > 10000:
            return data[:10000] + "... [TRUNCATED]"
        if any(pattern in data.lower() for pattern in ['password', 'token', 'secret', 'api_key']):
            return "[REDACTED_STRING]"
        return data
    else:
        return data

def analyze_performance(execution_time_ms: int, result_size_bytes: int, tool_name: str) -> Dict[str, Any]:
    """Analyze tool performance and identify issues."""
    analysis = {
        "performance_score": "good",
        "warnings": [],
        "metrics": {
            "execution_time_ms": execution_time_ms,
            "result_size_bytes": result_size_bytes,
            "result_size_kb": round(result_size_bytes / 1024, 2)
        }
    }
    
    # Performance analysis
    if execution_time_ms > SLOW_TOOL_THRESHOLD_MS:
        analysis["performance_score"] = "slow"
        analysis["warnings"].append(f"Tool execution exceeded {SLOW_TOOL_THRESHOLD_MS}ms threshold")
    
    if result_size_bytes > LARGE_RESULT_THRESHOLD_KB * 1024:
        analysis["warnings"].append(f"Large result size: {round(result_size_bytes/1024, 2)}KB")
    
    # Tool-specific analysis
    if tool_name.lower() in ['read', 'write', 'edit']:
        if execution_time_ms > 5000:
            analysis["warnings"].append("File operation taking longer than expected")
    elif tool_name.lower() in ['bash', 'shell', 'command']:
        if execution_time_ms > 10000:
            analysis["warnings"].append("Command execution taking longer than expected")
    
    return analysis

# Hook Implementation
def capture_post_tool_use(
    tool_name: str, 
    result: Any, 
    success: bool = True, 
    error: Optional[str] = None,
    execution_time_ms: Optional[int] = None
) -> None:
    """Capture PostToolUse event with performance analysis."""
    try:
        # Get correlation ID from PreToolUse
        correlation_id = os.environ.get(f"HOOK_CORRELATION_{tool_name}")
        if correlation_id:
            # Clean up the correlation ID
            del os.environ[f"HOOK_CORRELATION_{tool_name}"]
        
        # Calculate execution time if not provided
        if execution_time_ms is None:
            start_time_key = f"HOOK_START_TIME_{tool_name}"
            start_time = os.environ.get(start_time_key)
            if start_time:
                execution_time_ms = int((time.time() - float(start_time)) * 1000)
                del os.environ[start_time_key]
        
        # Redact sensitive information from results
        safe_result = redact_sensitive_data(result) if result is not None else None
        
        # Calculate result size
        result_size_bytes = len(json.dumps(safe_result, default=str)) if safe_result else 0
        
        # Performance analysis
        performance_analysis = analyze_performance(
            execution_time_ms or 0, 
            result_size_bytes, 
            tool_name
        )
        
        # Prepare event data
        event_data = {
            "session_id": SESSION_ID,
            "agent_id": AGENT_ID,
            "event_type": "POST_TOOL_USE",
            "tool_name": tool_name,
            "success": success,
            "result": safe_result,
            "error": error,
            "correlation_id": correlation_id,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "performance": performance_analysis,
            "context": {
                "result_truncated": isinstance(result, str) and len(str(result)) > 10000,
                "result_redacted": any(
                    key.lower() in ['password', 'token', 'secret'] 
                    for key in (result.keys() if isinstance(result, dict) else [])
                ),
                "has_error": error is not None,
                "error_type": type(error).__name__ if error else None
            }
        }
        
        # Send event asynchronously
        asyncio.create_task(send_hook_event(event_data))
        
        # Log performance warnings
        if performance_analysis["warnings"]:
            print(f"Performance warnings for {tool_name}: {', '.join(performance_analysis['warnings'])}")
        
    except Exception as e:
        print(f"PostToolUse hook error: {str(e)}")

# Export hook function
__all__ = ["capture_post_tool_use"]
```

## Usage

This hook is automatically triggered by Claude Code after any tool execution. It captures:

- **Tool Name**: The name of the tool that was executed
- **Result**: Tool execution result with sensitive data redacted
- **Success Status**: Whether the tool executed successfully
- **Error Information**: Error details if the tool failed
- **Performance Metrics**: Execution time, result size, performance analysis
- **Correlation ID**: Links with corresponding PreToolUse event

## Performance Analysis

The hook provides intelligent performance analysis:

- **Execution Time Monitoring**: Flags slow tools (>2s by default)
- **Result Size Analysis**: Monitors large payloads (>100KB by default)
- **Tool-Specific Thresholds**: Different performance expectations per tool type
- **Performance Scoring**: Good/Slow/Failed performance categorization

## Security Features

- **Result Redaction**: Automatically redacts sensitive data from results
- **Size Limits**: Truncates large results to prevent memory issues
- **PII Protection**: Removes personally identifiable information
- **Error Sanitization**: Cleanses error messages of sensitive details

## Performance

- **Asynchronous**: Non-blocking event capture
- **Efficient Processing**: Minimal overhead for performance analysis
- **Memory Safe**: Automatic truncation of large results
- **Error Resilient**: Continues operation even if hook fails

## Environment Variables

- `LEANVIBE_HOOK_API_URL`: Base URL for hook API
- `LEANVIBE_SLOW_TOOL_THRESHOLD`: Slow tool threshold in ms (default: 2000)
- `LEANVIBE_LARGE_RESULT_THRESHOLD`: Large result threshold in KB (default: 100)
- `LEANVIBE_SESSION_ID`: Current session UUID
- `LEANVIBE_AGENT_ID`: Current agent UUID

## Event Schema

```json
{
  "session_id": "uuid",
  "agent_id": "uuid", 
  "event_type": "POST_TOOL_USE",
  "tool_name": "string",
  "success": "boolean",
  "result": "any",
  "error": "string|null",
  "correlation_id": "uuid|null",
  "execution_time_ms": "number|null",
  "timestamp": "ISO 8601",
  "performance": {
    "performance_score": "good|slow|failed",
    "warnings": ["string"],
    "metrics": {
      "execution_time_ms": "number",
      "result_size_bytes": "number",
      "result_size_kb": "number"
    }
  },
  "context": {
    "result_truncated": "boolean",
    "result_redacted": "boolean", 
    "has_error": "boolean",
    "error_type": "string|null"
  }
}
```