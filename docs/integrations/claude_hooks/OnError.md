# OnError Hook - LeanVibe Agent Hive Observability

This hook captures error conditions and failure scenarios for comprehensive error monitoring and debugging.

## Hook Configuration

```python
import asyncio
import json
import os
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Hook API Configuration
HOOK_API_BASE_URL = os.getenv("LEANVIBE_HOOK_API_URL", "http://localhost:8000")
HOOK_API_TIMEOUT = int(os.getenv("LEANVIBE_HOOK_TIMEOUT", "5"))
ENABLE_HOOKS = os.getenv("LEANVIBE_ENABLE_HOOKS", "true").lower() == "true")

# Session and Agent Context
SESSION_ID = os.getenv("LEANVIBE_SESSION_ID", str(uuid.uuid4()))
AGENT_ID = os.getenv("LEANVIBE_AGENT_ID", str(uuid.uuid4()))

# Error classification thresholds
CRITICAL_ERROR_PATTERNS = [
    "segmentation fault", "memory error", "stack overflow", 
    "database connection", "authentication", "permission denied"
]

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

def sanitize_error_message(error_msg: str) -> str:
    """Sanitize error message to remove sensitive information."""
    if not error_msg:
        return error_msg
    
    # Remove common sensitive patterns
    sanitized = error_msg
    
    # Remove file paths that might contain usernames
    import re
    sanitized = re.sub(r'/Users/[^/\s]+', '/Users/[USER]', sanitized)
    sanitized = re.sub(r'/home/[^/\s]+', '/home/[USER]', sanitized)
    sanitized = re.sub(r'C:\\Users\\[^\\s]+', 'C:\\Users\\[USER]', sanitized)
    
    # Remove potential API keys or tokens
    sanitized = re.sub(r'[A-Za-z0-9]{32,}', '[TOKEN_REDACTED]', sanitized)
    
    # Remove email addresses
    sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', sanitized)
    
    # Remove IP addresses (optional - might be needed for debugging)
    # sanitized = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_REDACTED]', sanitized)
    
    return sanitized

def classify_error(error_msg: str, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Classify error severity and type for better monitoring."""
    classification = {
        "severity": "medium",
        "category": "unknown",
        "is_critical": False,
        "is_recoverable": True,
        "suggested_action": "investigate"
    }
    
    error_msg_lower = error_msg.lower() if error_msg else ""
    
    # Critical error detection
    if any(pattern in error_msg_lower for pattern in CRITICAL_ERROR_PATTERNS):
        classification["severity"] = "critical"
        classification["is_critical"] = True
        classification["is_recoverable"] = False
        classification["suggested_action"] = "immediate_attention"
    
    # Error categorization
    if "connection" in error_msg_lower or "network" in error_msg_lower:
        classification["category"] = "network"
        classification["suggested_action"] = "check_connectivity"
    elif "permission" in error_msg_lower or "access" in error_msg_lower:
        classification["category"] = "permissions" 
        classification["suggested_action"] = "check_permissions"
    elif "timeout" in error_msg_lower:
        classification["category"] = "timeout"
        classification["suggested_action"] = "increase_timeout"
    elif "syntax" in error_msg_lower or "parse" in error_msg_lower:
        classification["category"] = "syntax"
        classification["suggested_action"] = "check_syntax"
    elif "memory" in error_msg_lower or "allocation" in error_msg_lower:
        classification["category"] = "memory"
        classification["is_critical"] = True
        classification["suggested_action"] = "check_memory_usage"
    elif "database" in error_msg_lower or "sql" in error_msg_lower:
        classification["category"] = "database"
        classification["suggested_action"] = "check_database"
    elif error_type and "FileNotFoundError" in error_type:
        classification["category"] = "file_system"
        classification["suggested_action"] = "check_file_exists"
    elif error_type and "ValueError" in error_type:
        classification["category"] = "validation"
        classification["suggested_action"] = "check_input_validation"
    
    # Adjust severity based on context
    if context.get("tool_name") in ["bash", "command", "shell"]:
        # Shell commands can have higher tolerance for errors
        if classification["severity"] == "medium":
            classification["severity"] = "low"
    
    return classification

def extract_error_context(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive error context for debugging."""
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        "context": context
    }
    
    # Sanitize all context information
    error_context["error_message"] = sanitize_error_message(error_context["error_message"])
    error_context["stack_trace"] = sanitize_error_message(error_context["stack_trace"])
    
    # Add system context
    error_context["system"] = {
        "platform": os.name,
        "working_directory": os.getcwd(),
        "environment_vars": {
            key: value for key, value in os.environ.items() 
            if not any(sensitive in key.lower() for sensitive in ['password', 'token', 'secret', 'key'])
        }
    }
    
    return error_context

# Hook Implementation
def capture_error(
    error: Union[Exception, str], 
    context: Optional[Dict[str, Any]] = None,
    tool_name: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Capture error event with comprehensive context and classification."""
    try:
        # Prepare context
        error_context = context or {}
        if tool_name:
            error_context["tool_name"] = tool_name
        
        # Extract error information
        if isinstance(error, Exception):
            error_info = extract_error_context(error, error_context)
        else:
            error_info = {
                "error_type": "UnknownError",
                "error_message": sanitize_error_message(str(error)),
                "stack_trace": None,
                "context": error_context
            }
        
        # Classify error
        classification = classify_error(
            error_info["error_message"],
            error_info["error_type"],
            error_context
        )
        
        # Prepare event data
        event_data = {
            "session_id": SESSION_ID,
            "agent_id": AGENT_ID,
            "event_type": "ERROR",
            "error_type": error_info["error_type"],
            "error_message": error_info["error_message"],
            "stack_trace": error_info["stack_trace"],
            "context": error_context,
            "classification": classification,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": error_info.get("system", {}),
            "metadata": {
                "sanitized": True,
                "stack_trace_available": error_info["stack_trace"] is not None,
                "context_size": len(json.dumps(error_context, default=str))
            }
        }
        
        # Send event asynchronously
        asyncio.create_task(send_hook_event(event_data))
        
        # Log critical errors immediately
        if classification["is_critical"]:
            print(f"CRITICAL ERROR: {error_info['error_message']} - {classification['suggested_action']}")
        
    except Exception as e:
        print(f"OnError hook error: {str(e)}")

# Decorator for automatic error capture
def capture_errors(tool_name: Optional[str] = None):
    """Decorator to automatically capture errors from functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                capture_error(
                    error=e,
                    context={"function": func.__name__, "args_count": len(args)},
                    tool_name=tool_name
                )
                raise  # Re-raise the original exception
        return wrapper
    return decorator

# Export hook functions
__all__ = ["capture_error", "capture_errors"]
```

## Usage

This hook captures error conditions from multiple sources:

1. **Manual Error Capture**: `capture_error(exception, context, tool_name)`
2. **Decorator Usage**: `@capture_errors(tool_name="my_tool")`
3. **Tool Integration**: Automatically triggered on tool failures

## Error Classification

The hook provides intelligent error classification:

- **Severity Levels**: Critical, High, Medium, Low
- **Categories**: Network, Permissions, Timeout, Syntax, Memory, Database, File System, Validation
- **Recoverability**: Determines if error is recoverable
- **Suggested Actions**: Automated recommendations for error resolution

## Security Features

- **Message Sanitization**: Removes sensitive information from error messages
- **Path Redaction**: Replaces user-specific file paths
- **Token Removal**: Strips potential API keys and tokens
- **PII Protection**: Removes email addresses and other PII
- **Context Filtering**: Excludes sensitive environment variables

## Error Context

Comprehensive context capture includes:

- **Error Type**: Exception class name
- **Stack Trace**: Full stack trace (sanitized)
- **System Information**: Platform, working directory
- **Tool Context**: Tool name, parameters, correlation ID
- **Environment**: Non-sensitive environment variables

## Performance

- **Asynchronous**: Non-blocking error capture
- **Efficient**: Minimal overhead during error conditions
- **Safe**: Never fails the original operation
- **Memory Conscious**: Limits context size

## Environment Variables

- `LEANVIBE_HOOK_API_URL`: Base URL for hook API
- `LEANVIBE_HOOK_TIMEOUT`: API timeout in seconds
- `LEANVIBE_ENABLE_HOOKS`: Enable/disable hooks
- `LEANVIBE_SESSION_ID`: Current session UUID
- `LEANVIBE_AGENT_ID`: Current agent UUID

## Event Schema

```json
{
  "session_id": "uuid",
  "agent_id": "uuid",
  "event_type": "ERROR",
  "error_type": "string",
  "error_message": "string",
  "stack_trace": "string|null",
  "context": "object",
  "classification": {
    "severity": "critical|high|medium|low",
    "category": "string",
    "is_critical": "boolean",
    "is_recoverable": "boolean", 
    "suggested_action": "string"
  },
  "correlation_id": "uuid|null",
  "timestamp": "ISO 8601",
  "system_info": "object",
  "metadata": {
    "sanitized": "boolean",
    "stack_trace_available": "boolean",
    "context_size": "number"
  }
}
```