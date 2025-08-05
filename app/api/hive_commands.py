"""
Hive Slash Commands API for LeanVibe Agent Hive 2.0

This API provides endpoints for executing custom slash commands with the 
`hive:` prefix, enabling Claude Code-style custom commands specifically
for autonomous development orchestration.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import structlog
import time
import asyncio

from ..core.hive_slash_commands import execute_hive_command, get_hive_command_registry
from ..core.mobile_api_cache import get_mobile_cache, mobile_cache_key, cache_mobile_api_response, get_cached_mobile_response

logger = structlog.get_logger()
router = APIRouter()


class HiveCommandRequest(BaseModel):
    """Request to execute a hive slash command."""
    command: str = Field(..., description="The hive command to execute (e.g., '/hive:start')")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for command execution")
    mobile_optimized: bool = Field(default=False, description="Enable mobile-specific optimizations")
    use_cache: bool = Field(default=True, description="Use intelligent caching for faster responses")
    priority: str = Field(default="medium", description="Request priority: critical, high, medium, low")


class HiveCommandResponse(BaseModel):
    """Response from hive command execution."""
    success: bool
    command: str
    result: Dict[str, Any]
    execution_time_ms: Optional[float] = None
    mobile_optimized: bool = False
    cached: bool = False
    cache_key: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None


@router.post("/execute", response_model=HiveCommandResponse)
async def execute_command(request: HiveCommandRequest):
    """
    Execute a hive slash command with mobile optimization and intelligent caching.
    
    This endpoint provides enhanced execution with:
    - Mobile-optimized responses (<5ms target for cached content)
    - Intelligent caching based on command type and priority
    - Real-time performance metrics
    - WebSocket integration for live updates
    
    Examples:
    - `/hive:start` - Start multi-agent platform
    - `/hive:spawn backend_developer` - Spawn specific agent
    - `/hive:status --mobile --priority=high` - Mobile-optimized status
    - `/hive:focus development --mobile` - Context-aware mobile recommendations
    - `/hive:develop "Build authentication API" --dashboard` - Start autonomous development
    """
    cache = get_mobile_cache()
    start_time = time.time()
    cached = False
    cache_key = None
    
    try:
        logger.info("🎯 Executing enhanced hive command", 
                   command=request.command, 
                   mobile_optimized=request.mobile_optimized,
                   priority=request.priority)
        
        # Generate cache key for cacheable commands
        if request.use_cache and _is_cacheable_command(request.command):
            cache_key = mobile_cache_key(
                request.command, 
                request.context or {}, 
                request.mobile_optimized
            )
            
            # Try to get from cache first
            cached_result = await get_cached_mobile_response(cache_key)
            if cached_result:
                execution_time = (time.time() - start_time) * 1000
                cached = True
                
                logger.info("⚡ Cache hit - ultra-fast response", 
                           command=request.command,
                           execution_time_ms=execution_time,
                           cache_key=cache_key[:8])
                
                return HiveCommandResponse(
                    success=cached_result.get("success", False),
                    command=request.command,
                    result=cached_result,
                    execution_time_ms=execution_time,
                    mobile_optimized=request.mobile_optimized,
                    cached=True,
                    cache_key=cache_key[:8],
                    performance_metrics={
                        "cache_hit": True,
                        "response_type": "cached",
                        "mobile_optimized": request.mobile_optimized
                    }
                )
        
        # Execute the command with mobile optimization context
        enhanced_context = {
            **(request.context or {}),
            "mobile_optimized": request.mobile_optimized,
            "priority": request.priority,
            "cache_enabled": request.use_cache
        }
        
        # Add mobile flag to command if mobile_optimized
        command_to_execute = request.command
        if request.mobile_optimized and "--mobile" not in command_to_execute:
            command_to_execute += " --mobile"
        
        result = await execute_hive_command(command_to_execute, enhanced_context)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Cache the result if appropriate
        if request.use_cache and cache_key and result.get("success", False):
            ttl = _get_cache_ttl(request.command, request.priority)
            await cache_mobile_api_response(
                cache_key, 
                result, 
                ttl_seconds=ttl,
                priority=request.priority
            )
        
        # Get performance metrics
        performance_metrics = {
            "cache_hit": False,
            "response_type": "live",
            "mobile_optimized": request.mobile_optimized,
            "execution_time_ms": execution_time,
            "cache_eligible": cache_key is not None,
            "priority": request.priority
        }
        
        # Add mobile-specific performance metrics
        if request.mobile_optimized:
            performance_metrics.update({
                "mobile_response_time_target": "< 5ms (cached) / < 50ms (live)",
                "mobile_performance_score": min(100, max(0, 100 - (execution_time - 50) * 2)) if execution_time > 50 else 100,
                "mobile_optimized_alerts": result.get("performance_metrics", {}).get("mobile_optimized_alerts", 0)
            })
        
        logger.info("✅ Enhanced hive command completed", 
                   command=request.command, 
                   success=result.get("success", False),
                   execution_time_ms=execution_time,
                   mobile_optimized=request.mobile_optimized,
                   cached=cached)
        
        return HiveCommandResponse(
            success=result.get("success", False),
            command=request.command,
            result=result,
            execution_time_ms=execution_time,
            mobile_optimized=request.mobile_optimized,
            cached=cached,
            cache_key=cache_key[:8] if cache_key else None,
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error("❌ Enhanced hive command execution failed", 
                    command=request.command, 
                    error=str(e),
                    execution_time_ms=execution_time,
                    mobile_optimized=request.mobile_optimized)
        raise HTTPException(
            status_code=500,
            detail=f"Command execution failed: {str(e)}"
        )


def _is_cacheable_command(command: str) -> bool:
    """Determine if a command result can be cached."""
    cacheable_commands = [
        "status", "focus", "list", "help", "productivity", "notifications"
    ]
    non_cacheable_commands = [
        "start", "spawn", "develop", "stop"  # State-changing commands
    ]
    
    command_name = command.split()[0].replace("/hive:", "").strip()
    
    if any(non_cacheable in command_name for non_cacheable in non_cacheable_commands):
        return False
    
    return any(cacheable in command_name for cacheable in cacheable_commands)


def _get_cache_ttl(command: str, priority: str) -> int:
    """Get appropriate cache TTL based on command type and priority."""
    command_name = command.split()[0].replace("/hive:", "").strip()
    
    # Base TTL by command type
    base_ttls = {
        "status": 15,      # Status changes frequently
        "focus": 30,       # Recommendations can be cached longer
        "productivity": 60, # Productivity metrics stable
        "notifications": 10, # Notifications need to be fresh
        "list": 300,       # Command list rarely changes
        "help": 600        # Help content is static
    }
    
    # Priority multipliers
    priority_multipliers = {
        "critical": 0.5,   # Critical content needs frequent updates
        "high": 0.8,       # High priority content semi-fresh
        "medium": 1.0,     # Standard caching
        "low": 2.0         # Low priority can be cached longer
    }
    
    base_ttl = base_ttls.get(command_name, 30)  # Default 30 seconds
    multiplier = priority_multipliers.get(priority, 1.0)
    
    return max(5, int(base_ttl * multiplier))  # Minimum 5 seconds


@router.get("/list")
async def list_commands():
    """
    List all available hive slash commands.
    
    Returns a registry of all available commands with their descriptions
    and usage information.
    """
    try:
        registry = get_hive_command_registry()
        commands_info = {}
        
        for name, command in registry.commands.items():
            commands_info[name] = {
                "name": command.name,
                "description": command.description,
                "usage": command.usage,
                "full_command": f"/hive:{command.name}"
            }
        
        return {
            "success": True,
            "total_commands": len(commands_info),
            "commands": commands_info,
            "usage_examples": [
                "/hive:start --team-size=5",
                "/hive:spawn architect --capabilities=system_design,security",
                "/hive:status --detailed",
                "/hive:productivity --developer --mobile",
                "/hive:develop \"Build user authentication with JWT\"",
                "/hive:oversight --mobile-info",
                "/hive:stop --agents-only"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to list commands", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list commands: {str(e)}"
        )


@router.get("/help/{command_name}")
async def get_command_help(command_name: str, mobile: bool = False, context_aware: bool = True):
    """
    Get detailed help for a specific hive command with context-aware recommendations.
    
    Provides usage information, examples, parameter details, and intelligent
    recommendations based on current system state and development context.
    """
    try:
        registry = get_hive_command_registry()
        command = registry.get_command(command_name)
        
        if not command:
            raise HTTPException(
                status_code=404,
                detail=f"Command '{command_name}' not found"
            )
        
        # Get current system state for context-aware help
        system_context = {}
        contextual_recommendations = []
        
        if context_aware:
            try:
                # Get system status for context
                from ..core.hive_slash_commands import HiveStatusCommand
                status_command = HiveStatusCommand()
                status_result = await status_command.execute(["--mobile"] if mobile else [])
                
                system_context = {
                    "agent_count": status_result.get("agent_count", 0),
                    "system_ready": status_result.get("system_ready", False),
                    "platform_active": status_result.get("platform_active", False),
                    "mobile_optimized": mobile
                }
                
                # Generate contextual recommendations
                contextual_recommendations = _generate_contextual_help_recommendations(
                    command_name, system_context
                )
            except:
                pass  # Fallback to basic help if context gathering fails
        
        # Generate enhanced examples based on command type and context
        examples = _generate_contextual_examples(command_name, system_context, mobile)
        
        help_response = {
            "success": True,
            "command": {
                "name": command.name,
                "full_command": f"/hive:{command.name}",
                "description": command.description,
                "usage": command.usage,
                "examples": examples,
                "created_at": command.created_at.isoformat() if hasattr(command, 'created_at') else None
            },
            "system_context": system_context,
            "contextual_recommendations": contextual_recommendations,
            "mobile_optimized": mobile,
            "context_aware": context_aware
        }
        
        # Add mobile-specific help enhancements
        if mobile:
            help_response["mobile_enhancements"] = _get_mobile_help_enhancements(command_name)
        
        return help_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get command help", command=command_name, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get help for command: {str(e)}"
        )


@router.post("/quick/{command_name}")
async def quick_execute_command(command_name: str, args: Optional[str] = None):
    """
    Quick execution endpoint for hive commands without full request body.
    
    Convenient way to execute commands with simple string arguments.
    
    Examples:
    - POST /api/hive/quick/start
    - POST /api/hive/quick/spawn?args=backend_developer
    - POST /api/hive/quick/status?args=--detailed
    """
    try:
        # Construct command string
        command_text = f"/hive:{command_name}"
        if args:
            command_text += f" {args}"
        
        logger.info("🚀 Quick executing hive command", command=command_text)
        
        # Execute the command
        result = await execute_hive_command(command_text)
        
        return {
            "success": result.get("success", False),
            "command": command_text,
            "result": result
        }
        
    except Exception as e:
        logger.error("Quick command execution failed", 
                    command_name=command_name, args=args, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Quick execution failed: {str(e)}"
        )


@router.get("/status")
async def get_command_system_status():
    """
    Get status of the hive command system.
    
    Returns information about the command registry and system readiness.
    """
    try:
        registry = get_hive_command_registry()
        
        return {
            "success": True,
            "system_ready": True,
            "total_commands": len(registry.commands),
            "available_commands": list(registry.commands.keys()),
            "command_prefix": "hive:",
            "api_endpoints": {
                "execute": "/api/hive/execute",
                "list": "/api/hive/list", 
                "help": "/api/hive/help/{command_name}",
                "quick": "/api/hive/quick/{command_name}"
            }
        }
        
    except Exception as e:
        logger.error("Failed to get command system status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.post("/mobile/execute", response_model=HiveCommandResponse)
async def execute_mobile_command(request: HiveCommandRequest):
    """
    Mobile-optimized hive command execution endpoint.
    
    Automatically enables mobile optimizations with intelligent caching,
    priority-based alert filtering, and <5ms target response times.
    
    Designed specifically for mobile dashboard interfaces with:
    - Automatic mobile flag injection
    - Aggressive caching for status/focus commands
    - WebSocket integration for real-time updates
    - Performance metrics optimized for mobile networks
    """
    # Force mobile optimizations
    request.mobile_optimized = True
    request.use_cache = True
    
    # Auto-adjust priority for mobile context
    if request.priority == "medium":
        request.priority = "high"  # Mobile users expect faster responses
    
    return await execute_command(request)


@router.get("/mobile/performance")
async def get_mobile_performance_metrics():
    """
    Get mobile-specific performance metrics and cache statistics.
    
    Returns real-time metrics optimized for mobile dashboard monitoring:
    - Cache hit rates and response times
    - Mobile optimization scores
    - Alert relevance filtering effectiveness
    - WebSocket connection health
    """
    try:
        cache = get_mobile_cache()
        cache_stats = await cache.stats()
        performance_metrics = await cache.get_performance_metrics()
        
        # Calculate mobile-specific metrics
        mobile_score = cache_stats.get("mobile_optimization_percentage", 0)
        response_time_score = 100 if performance_metrics.avg_response_time_ms < 5 else max(0, 100 - (performance_metrics.avg_response_time_ms - 5) * 2)
        
        return {
            "success": True,
            "mobile_performance_score": round((mobile_score + response_time_score) / 2, 1),
            "cache_performance": {
                "hit_rate": performance_metrics.cache_hit_rate,
                "avg_response_time_ms": performance_metrics.avg_response_time_ms,
                "mobile_optimized_percentage": mobile_score,
                "total_size_mb": cache_stats["total_size_mb"],
                "utilization_percentage": cache_stats["utilization_percentage"]
            },
            "mobile_optimization": {
                "mobile_optimized_entries": cache_stats["mobile_optimized"],
                "total_entries": cache_stats["total_entries"],
                "optimization_score": performance_metrics.mobile_optimization_score,
                "alert_relevance_score": performance_metrics.alert_relevance_score
            },
            "response_time_targets": {
                "cached_target_ms": 5,
                "live_target_ms": 50,
                "current_avg_ms": performance_metrics.avg_response_time_ms,
                "performance_grade": "A" if performance_metrics.avg_response_time_ms < 10 else
                                   "B" if performance_metrics.avg_response_time_ms < 25 else
                                   "C" if performance_metrics.avg_response_time_ms < 50 else "D"
            },
            "recommendations": _generate_mobile_performance_recommendations(cache_stats, performance_metrics),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Failed to get mobile performance metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.post("/mobile/optimize")
async def optimize_mobile_performance():
    """
    Trigger mobile performance optimization.
    
    Optimizes cache contents, promotes frequently accessed mobile content,
    and clears low-priority entries to improve mobile response times.
    """
    try:
        cache = get_mobile_cache()
        optimization_results = await cache.optimize_for_mobile()
        
        return {
            "success": True,
            "optimization_results": optimization_results,
            "message": "Mobile performance optimization completed",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Failed to optimize mobile performance", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Mobile optimization failed: {str(e)}"
        )


@router.get("/mobile/cache/clear")
async def clear_mobile_cache():
    """
    Clear mobile API cache.
    
    Clears all cached entries to force fresh data on next requests.
    Use this endpoint when system state has changed significantly.
    """
    try:
        cache = get_mobile_cache()
        cleared_count = await cache.clear()
        
        return {
            "success": True,
            "cleared_entries": cleared_count,
            "message": f"Cleared {cleared_count} cache entries",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Failed to clear mobile cache", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Cache clear failed: {str(e)}"
        )


def _generate_mobile_performance_recommendations(cache_stats: Dict[str, Any], performance_metrics) -> List[Dict[str, str]]:
    """Generate mobile performance improvement recommendations."""
    recommendations = []
    
    if performance_metrics.avg_response_time_ms > 50:
        recommendations.append({
            "title": "Enable Aggressive Caching",
            "description": f"Current response time {performance_metrics.avg_response_time_ms:.1f}ms exceeds mobile target of 50ms",
            "action": "Increase cache TTL for status commands or use mobile-specific endpoints"
        })
    
    if cache_stats.get("mobile_optimization_percentage", 0) < 60:
        recommendations.append({
            "title": "Improve Mobile Optimization",
            "description": f"Only {cache_stats.get('mobile_optimization_percentage', 0):.1f}% of cache is mobile-optimized",
            "action": "Use mobile_optimized=true parameter in API requests"
        })
    
    if cache_stats.get("utilization_percentage", 0) > 80:
        recommendations.append({
            "title": "Cache Memory Optimization",
            "description": f"Cache utilization at {cache_stats.get('utilization_percentage', 0):.1f}% may impact performance",
            "action": "Consider increasing cache size or reducing TTL for low-priority content"
        })
    
    if performance_metrics.cache_hit_rate < 0.4:
        recommendations.append({
            "title": "Improve Cache Hit Rate",
            "description": f"Cache hit rate of {performance_metrics.cache_hit_rate:.1%} indicates frequent cache misses",
            "action": "Review caching strategy and increase TTL for stable content"
        })
    
    return recommendations


def _generate_contextual_help_recommendations(command_name: str, system_context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate context-aware help recommendations based on system state."""
    recommendations = []
    
    agent_count = system_context.get("agent_count", 0)
    system_ready = system_context.get("system_ready", False)
    platform_active = system_context.get("platform_active", False)
    mobile_optimized = system_context.get("mobile_optimized", False)
    
    if command_name == "start":
        if not platform_active:
            recommendations.append({
                "title": "Platform Not Active",
                "description": "Your platform appears to be offline. This command will initialize the multi-agent system.",
                "suggestion": "Start with default team size for optimal performance",
                "priority": "high"
            })
        elif agent_count < 3:
            recommendations.append({
                "title": "Scale Team Size",
                "description": f"Current team has {agent_count} agents. Consider scaling to 5 agents for optimal development capacity.",
                "suggestion": "Use --team-size=5 for full development team",
                "priority": "medium"
            })
    
    elif command_name == "status":
        if mobile_optimized:
            recommendations.append({
                "title": "Mobile Status Optimization",
                "description": "Use mobile-specific flags for faster, filtered status information on mobile devices.",
                "suggestion": "Add --mobile --priority=high for mobile-optimized status",
                "priority": "high"
            })
        if not system_ready:
            recommendations.append({
                "title": "System Health Check",
                "description": "System shows as not ready. Use detailed status to identify issues.",
                "suggestion": "Use --detailed flag to get comprehensive system diagnostics",
                "priority": "high"
            })
    
    elif command_name == "focus":
        if agent_count > 0:
            recommendations.append({
                "title": "Agent Coordination Available",
                "description": f"With {agent_count} active agents, you can use agent-specific focus commands.",
                "suggestion": "Try --agent=backend_developer --task=\"your specific task\"",
                "priority": "medium"
            })
        if mobile_optimized:
            recommendations.append({
                "title": "Mobile Focus Features",
                "description": "Mobile focus provides quick actions and simplified recommendations.",
                "suggestion": "Use --mobile for touch-optimized quick actions",
                "priority": "medium"
            })
    
    elif command_name == "develop":
        if not system_ready:
            recommendations.append({
                "title": "System Not Ready",
                "description": "Platform may need initialization before starting development tasks.",
                "suggestion": "Run /hive:start first to ensure agents are available",
                "priority": "critical"
            })
        elif agent_count >= 5:
            recommendations.append({
                "title": "Full Team Available",
                "description": f"With {agent_count} agents active, you have full development team capability.",
                "suggestion": "Use --dashboard for mobile oversight during development",
                "priority": "info"
            })
    
    return recommendations


def _generate_contextual_examples(command_name: str, system_context: Dict[str, Any], mobile: bool) -> List[str]:
    """Generate context-aware examples based on system state."""
    agent_count = system_context.get("agent_count", 0)
    system_ready = system_context.get("system_ready", False)
    platform_active = system_context.get("platform_active", False)
    
    base_examples = {
        "start": [
            "/hive:start",
            "/hive:start --quick",
            "/hive:start --team-size=7"
        ],
        "spawn": [
            "/hive:spawn product_manager",
            "/hive:spawn backend_developer --capabilities=api_development,database_design",
            "/hive:spawn architect"
        ],
        "develop": [
            "/hive:develop \"Build authentication API\"",
            "/hive:develop \"Create user management system\" --dashboard",
            "/hive:develop \"Build REST API with tests\" --timeout=600"
        ],
        "status": [
            "/hive:status",
            "/hive:status --detailed",
            "/hive:status --agents-only"
        ],
        "focus": [
            "/hive:focus",
            "/hive:focus development",
            "/hive:focus --mobile"
        ],
        "productivity": [
            "/hive:productivity",
            "/hive:productivity --developer --mobile",
            "/hive:productivity --insights --workflow=development"
        ],
        "oversight": [
            "/hive:oversight",
            "/hive:oversight --mobile-info"
        ],
        "notifications": [
            "/hive:notifications",
            "/hive:notifications --mobile --critical-only",
            "/hive:notifications --summary"
        ],
        "stop": [
            "/hive:stop",
            "/hive:stop --agents-only",
            "/hive:stop --force"
        ]
    }
    
    examples = base_examples.get(command_name, [])
    
    # Add contextual examples based on system state
    if command_name == "start" and not platform_active:
        examples.insert(0, "/hive:start  # Recommended: Platform is offline")
    
    elif command_name == "status":
        if mobile:
            examples.insert(0, "/hive:status --mobile --priority=high  # Mobile-optimized")
        if not system_ready:
            examples.append("/hive:status --detailed  # Diagnose system issues")
    
    elif command_name == "focus":
        if mobile:
            examples.insert(0, "/hive:focus --mobile  # Touch-optimized recommendations")
        if agent_count > 0:
            examples.append(f"/hive:focus development --agent=backend_developer  # Direct agent attention")
    
    elif command_name == "develop" and not system_ready:
        examples.insert(0, "# First run: /hive:start")
        examples.insert(1, "# Then run: /hive:develop \"Your project\"")
    
    # Add mobile-specific examples
    if mobile and command_name in ["status", "focus", "productivity", "notifications"]:
        mobile_example = f"/hive:{command_name} --mobile"
        if mobile_example not in examples:
            examples.append(mobile_example + "  # Mobile-optimized")
    
    return examples


def _get_mobile_help_enhancements(command_name: str) -> List[Dict[str, str]]:
    """Get mobile-specific help enhancements."""
    mobile_enhancements = {
        "status": [
            {
                "feature": "Quick Actions",
                "description": "Mobile status includes contextual quick actions for immediate execution"
            },
            {
                "feature": "Priority Filtering",
                "description": "Intelligent alert filtering shows only relevant information for mobile screens"
            },
            {
                "feature": "Response Time Optimization",
                "description": "Aggressive caching for <5ms response times on repeated status checks"
            }
        ],
        "focus": [
            {
                "feature": "Touch-Optimized Actions",
                "description": "Quick actions optimized for touch interfaces with estimated completion times"
            },
            {
                "feature": "Agent Coordination",
                "description": "Direct agent attention with mobile-friendly task assignment interface"
            }
        ],
        "notifications": [
            {
                "feature": "Push Notifications",
                "description": "Native mobile push notifications for critical system alerts"
            },
            {
                "feature": "Offline Queuing",
                "description": "Notifications queued during offline periods and delivered when reconnected"
            }
        ]
    }
    
    return mobile_enhancements.get(command_name, [])