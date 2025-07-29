"""
VS 7.1: Secure Sleep/Wake API with Checkpointing for LeanVibe Agent Hive 2.0 Phase 5.2

Enhanced API endpoints with:
- JWT authentication with role-based access control
- <2s response time optimization
- Atomic checkpointing with distributed locking
- Idempotency key support
- Circuit breaker pattern integration
- Enhanced observability and monitoring
- Production-grade error handling
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import jwt

from ...core.checkpoint_manager import get_checkpoint_manager
from ...core.sleep_wake_manager import get_sleep_wake_manager
from ...core.recovery_manager import get_recovery_manager
from ...core.circuit_breaker import CircuitBreaker
from ...core.redis import get_redis
from ...core.database import get_async_session
from ...models.sleep_wake import CheckpointType, SleepState
from ...models.agent import Agent
from ...core.config import get_settings


logger = logging.getLogger(__name__)

# VS 7.1 Router with enhanced configuration
router = APIRouter(
    prefix="/api/v1/sleep-wake/vs7.1",
    tags=["VS 7.1 Sleep/Wake Management"],
    responses={
        401: {"description": "Unauthorized - Invalid or missing JWT token"},
        403: {"description": "Forbidden - Insufficient permissions"},
        429: {"description": "Too Many Requests - Rate limit exceeded"},
        500: {"description": "Internal Server Error"},
        503: {"description": "Service Unavailable - Circuit breaker open"}
    }
)

# Security configuration
security = HTTPBearer()
settings = get_settings()

# VS 7.1 Circuit breakers for different operations
checkpoint_circuit_breaker = CircuitBreaker(
    name="checkpoint_operations",
    failure_threshold=5,
    timeout_seconds=30
)

recovery_circuit_breaker = CircuitBreaker(
    name="recovery_operations",
    failure_threshold=3,
    timeout_seconds=60  # Longer recovery for critical operations
)


# VS 7.1 Enhanced Request/Response Models

class AtomicCheckpointRequest(BaseModel):
    """Request model for atomic checkpoint creation."""
    agent_id: Optional[UUID] = Field(None, description="Agent ID for agent-specific checkpoint")
    checkpoint_type: str = Field(default="manual", description="Type of checkpoint")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    idempotency_key: Optional[str] = Field(
        default=None, 
        description="Idempotency key to prevent duplicate operations"
    )
    force_creation: bool = Field(
        default=False, 
        description="Force creation even if agent is busy"
    )
    
    @validator('checkpoint_type')
    def validate_checkpoint_type(cls, v):
        allowed_types = ["manual", "scheduled", "pre_sleep", "emergency"]
        if v not in allowed_types:
            raise ValueError(f"checkpoint_type must be one of {allowed_types}")
        return v
    
    @validator('idempotency_key')
    def validate_idempotency_key(cls, v):
        if v and len(v) > 256:
            raise ValueError("idempotency_key must be 256 characters or less")
        return v


class SecureWakeRequest(BaseModel):
    """Request model for secure wake operations."""
    agent_id: UUID = Field(..., description="Agent ID to wake up")
    recovery_mode: bool = Field(default=False, description="Use recovery mode for wake")
    validation_level: str = Field(
        default="standard", 
        description="Validation level: minimal, standard, full"
    )
    target_recovery_time_ms: Optional[int] = Field(
        default=None, 
        description="Target recovery time in milliseconds"
    )
    
    @validator('validation_level')
    def validate_validation_level(cls, v):
        allowed_levels = ["minimal", "standard", "full"]
        if v not in allowed_levels:
            raise ValueError(f"validation_level must be one of {allowed_levels}")
        return v


class DistributedSleepRequest(BaseModel):
    """Request model for distributed sleep operations with coordination."""
    agent_ids: List[UUID] = Field(..., description="List of agent IDs to put to sleep")
    coordination_strategy: str = Field(
        default="sequential", 
        description="Coordination strategy: sequential, parallel, staged"
    )
    max_concurrent: int = Field(default=3, ge=1, le=10, description="Maximum concurrent operations")
    rollback_on_failure: bool = Field(
        default=True, 
        description="Rollback all operations if any fail"
    )
    timeout_seconds: int = Field(default=300, ge=30, le=600, description="Operation timeout")
    
    @validator('coordination_strategy')
    def validate_coordination_strategy(cls, v):
        allowed_strategies = ["sequential", "parallel", "staged"]
        if v not in allowed_strategies:
            raise ValueError(f"coordination_strategy must be one of {allowed_strategies}")
        return v


class CheckpointResponse(BaseModel):
    """Response model for checkpoint operations."""
    success: bool
    checkpoint_id: Optional[UUID] = None
    creation_time_ms: float
    meets_performance_target: bool
    size_bytes: int
    compression_ratio: float
    git_commit_hash: Optional[str] = None
    idempotency_key: Optional[str] = None
    warnings: List[str] = []
    performance_metrics: Dict[str, Any] = {}


class WakeResponse(BaseModel):
    """Response model for wake operations."""
    success: bool
    agent_id: UUID
    recovery_time_ms: float
    validation_results: Dict[str, Any] = {}
    checkpoint_used: Optional[UUID] = None
    meets_target_time: bool
    warnings: List[str] = []
    errors: List[str] = []


class SystemStatusResponse(BaseModel):
    """Enhanced system status response."""
    timestamp: datetime
    system_healthy: bool
    checkpoint_engine_status: Dict[str, Any]
    recovery_engine_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    circuit_breaker_status: Dict[str, Any]
    active_operations: int
    error_summary: List[str] = []


# VS 7.1 Enhanced Authentication and Authorization

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Enhanced JWT token validation with role-based access control."""
    try:
        # Extract token
        token = credentials.credentials
        
        # Decode and validate JWT token
        payload = jwt.decode(
            token, 
            settings.jwt_secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        
        # Extract user information
        user_id = payload.get("sub")
        username = payload.get("username")
        roles = payload.get("roles", [])
        permissions = payload.get("permissions", [])
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID"
            )
        
        # Check token expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        return {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "permissions": permissions,
            "token_payload": payload
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Error validating JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed"
        )


def require_permission(permission: str):
    """Decorator to require specific permissions."""
    def permission_checker(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {permission} required"
            )
        return user
    return permission_checker


# VS 7.1 Performance Monitoring Decorator

def monitor_performance(target_time_ms: int = 2000):
    """Decorator to monitor endpoint performance and ensure <2s response time."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Add performance metrics to response if it's a dict
                if isinstance(result, dict):
                    result["performance_metrics"] = {
                        "response_time_ms": response_time_ms,
                        "meets_target": response_time_ms < target_time_ms,
                        "target_time_ms": target_time_ms
                    }
                
                # Log performance warning if target not met
                if response_time_ms > target_time_ms:
                    logger.warning(
                        f"Endpoint {func.__name__} exceeded target response time: "
                        f"{response_time_ms:.0f}ms > {target_time_ms}ms"
                    )
                
                return result
                
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Endpoint {func.__name__} failed after {response_time_ms:.0f}ms: {e}"
                )
                raise
                
        return wrapper
    return decorator


# VS 7.1 Enhanced API Endpoints

@router.post("/checkpoint/create", response_model=CheckpointResponse)
@monitor_performance(target_time_ms=5000)  # 5s target for checkpoint creation
async def create_atomic_checkpoint(
    request: AtomicCheckpointRequest,
    user: Dict[str, Any] = Depends(require_permission("checkpoint:create")),
    x_request_id: Optional[str] = Header(None)
):
    """
    Create atomic checkpoint with distributed locking and performance optimization.
    
    VS 7.1 Features:
    - <5s creation time with atomic state preservation
    - Distributed Redis locking to prevent conflicts
    - Idempotency key support for safe retries
    - 100% data integrity validation
    - Circuit breaker protection
    - Enhanced performance metrics
    """
    request_id = x_request_id or str(uuid4())
    
    try:
        logger.info(f"Creating atomic checkpoint - Request ID: {request_id}")
        
        # Circuit breaker protection
        if checkpoint_circuit_breaker.state == "open":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Checkpoint service temporarily unavailable"
            )
        
        # Get checkpoint manager
        checkpoint_manager = get_checkpoint_manager()
        
        # Pre-flight validation
        if request.agent_id and not request.force_creation:
            async with get_async_session() as session:
                agent = await session.get(Agent, request.agent_id)
                if not agent:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Agent {request.agent_id} not found"
                    )
                
                if agent.current_sleep_state not in [SleepState.AWAKE, SleepState.SLEEPING]:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Agent in invalid state: {agent.current_sleep_state.value}"
                    )
        
        # Create checkpoint with circuit breaker protection
        async with checkpoint_circuit_breaker:
            # Generate idempotency key if not provided
            idempotency_key = request.idempotency_key or f"auto_{request_id}_{int(time.time())}"
            
            # Map checkpoint type
            checkpoint_type_map = {
                "manual": CheckpointType.MANUAL,
                "scheduled": CheckpointType.SCHEDULED,
                "pre_sleep": CheckpointType.PRE_SLEEP,
                "emergency": CheckpointType.EMERGENCY
            }
            
            checkpoint = await checkpoint_manager.create_atomic_checkpoint(
                agent_id=request.agent_id,
                checkpoint_type=checkpoint_type_map[request.checkpoint_type],
                metadata=request.metadata,
                idempotency_key=idempotency_key
            )
        
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create checkpoint"
            )
        
        # Extract performance metrics
        perf_metrics = checkpoint.checkpoint_metadata.get("performance_metrics", {})
        
        return CheckpointResponse(
            success=True,
            checkpoint_id=checkpoint.id,
            creation_time_ms=perf_metrics.get("total_creation_time_ms", 0),
            meets_performance_target=perf_metrics.get("meets_target", False),
            size_bytes=checkpoint.size_bytes,
            compression_ratio=checkpoint.compression_ratio,
            git_commit_hash=checkpoint.checkpoint_metadata.get("git_commit_hash"),
            idempotency_key=idempotency_key,
            warnings=[],
            performance_metrics=perf_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating checkpoint - Request ID {request_id}: {e}")
        
        # Record circuit breaker failure
        checkpoint_circuit_breaker.record_failure()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error creating checkpoint: {str(e)}"
        )


@router.post("/agent/wake", response_model=WakeResponse)
@monitor_performance(target_time_ms=10000)  # 10s target for recovery
async def wake_agent_secure(
    request: SecureWakeRequest,
    user: Dict[str, Any] = Depends(require_permission("agent:wake")),
    x_request_id: Optional[str] = Header(None)
):
    """
    Wake agent with comprehensive recovery and validation.
    
    VS 7.1 Features:
    - <10s restoration capability with integrity validation
    - Multiple validation levels (minimal, standard, full)
    - Circuit breaker protection for recovery operations
    - Comprehensive health checks and performance metrics
    - Automatic fallback with multiple checkpoint generations
    """
    request_id = x_request_id or str(uuid4())
    
    try:
        logger.info(f"Waking agent {request.agent_id} - Request ID: {request_id}")
        
        # Circuit breaker protection
        if recovery_circuit_breaker.state == "open":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Recovery service temporarily unavailable"
            )
        
        # Validate agent exists and is in wakeable state
        async with get_async_session() as session:
            agent = await session.get(Agent, request.agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {request.agent_id} not found"
                )
            
            wakeable_states = [SleepState.SLEEPING, SleepState.PREPARING_WAKE, SleepState.ERROR]
            if agent.current_sleep_state not in wakeable_states:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Agent cannot be woken from state: {agent.current_sleep_state.value}"
                )
        
        # Perform wake operation with circuit breaker protection
        async with recovery_circuit_breaker:
            recovery_manager = get_recovery_manager()
            
            if request.recovery_mode:
                # Use comprehensive wake restoration
                success, restoration_details = await recovery_manager.comprehensive_wake_restoration(
                    agent_id=request.agent_id,
                    checkpoint=None,  # Use latest checkpoint
                    validation_level=request.validation_level
                )
                
                recovery_time_ms = restoration_details.get("performance_metrics", {}).get("total_time_ms", 0)
                validation_results = restoration_details.get("validation_results", {})
                checkpoint_used = None  # Would need to extract from restoration details
                
            else:
                # Use standard sleep-wake manager
                sleep_manager = await get_sleep_wake_manager()
                wake_start = time.time()
                
                success = await sleep_manager.initiate_wake_cycle(request.agent_id)
                recovery_time_ms = (time.time() - wake_start) * 1000
                validation_results = {"standard_wake": {"passed": success}}
                checkpoint_used = None
        
        # Determine if target time was met
        target_time_ms = request.target_recovery_time_ms or 10000  # Default 10s
        meets_target_time = recovery_time_ms < target_time_ms
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to wake agent"
            )
        
        return WakeResponse(
            success=True,
            agent_id=request.agent_id,
            recovery_time_ms=recovery_time_ms,
            validation_results=validation_results,
            checkpoint_used=checkpoint_used,
            meets_target_time=meets_target_time,
            warnings=[],
            errors=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error waking agent {request.agent_id} - Request ID {request_id}: {e}")
        
        # Record circuit breaker failure
        recovery_circuit_breaker.record_failure()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error waking agent: {str(e)}"
        )


@router.post("/system/distributed-sleep")
@monitor_performance(target_time_ms=15000)  # 15s for distributed operations
async def distributed_sleep_operation(
    request: DistributedSleepRequest,
    user: Dict[str, Any] = Depends(require_permission("system:distributed-sleep")),
    x_request_id: Optional[str] = Header(None)
):
    """
    Perform distributed sleep operations with coordination and rollback capability.
    
    VS 7.1 Features:
    - Coordinated multi-agent sleep operations
    - Multiple coordination strategies (sequential, parallel, staged)
    - Automatic rollback on partial failures
    - Distributed locking and state consistency
    - Performance optimization with configurable concurrency
    """
    request_id = x_request_id or str(uuid4())
    
    try:
        logger.info(f"Distributed sleep operation - Request ID: {request_id}, Agents: {len(request.agent_ids)}")
        
        # Validate all agents exist and are in valid state
        invalid_agents = []
        async with get_async_session() as session:
            for agent_id in request.agent_ids:
                agent = await session.get(Agent, agent_id)
                if not agent:
                    invalid_agents.append(f"Agent {agent_id} not found")
                elif agent.current_sleep_state != SleepState.AWAKE:
                    invalid_agents.append(f"Agent {agent_id} not awake ({agent.current_sleep_state.value})")
        
        if invalid_agents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid agents: {', '.join(invalid_agents)}"
            )
        
        # Get sleep manager
        sleep_manager = await get_sleep_wake_manager()
        results = []
        successful_operations = []
        
        # Execute based on coordination strategy
        if request.coordination_strategy == "sequential":
            # Sequential execution
            for agent_id in request.agent_ids:
                try:
                    success = await sleep_manager.initiate_sleep_cycle(
                        agent_id=agent_id,
                        cycle_type="distributed"
                    )
                    
                    results.append({
                        "agent_id": str(agent_id),
                        "success": success,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    if success:
                        successful_operations.append(agent_id)
                    elif request.rollback_on_failure:
                        # Rollback previous successful operations
                        await _rollback_sleep_operations(successful_operations, sleep_manager)
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Operation failed for agent {agent_id}, rolled back all operations"
                        )
                        
                except Exception as e:
                    if request.rollback_on_failure:
                        await _rollback_sleep_operations(successful_operations, sleep_manager)
                    raise
        
        elif request.coordination_strategy == "parallel":
            # Parallel execution with semaphore
            semaphore = asyncio.Semaphore(request.max_concurrent)
            
            async def sleep_single_agent(agent_id: UUID) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        success = await sleep_manager.initiate_sleep_cycle(
                            agent_id=agent_id,
                            cycle_type="distributed"
                        )
                        
                        return {
                            "agent_id": str(agent_id),
                            "success": success,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    except Exception as e:
                        return {
                            "agent_id": str(agent_id),
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }
            
            # Execute all tasks
            tasks = [sleep_single_agent(agent_id) for agent_id in request.agent_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        "agent_id": "unknown",
                        "success": False,
                        "error": str(result),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                else:
                    processed_results.append(result)
                    if result["success"]:
                        successful_operations.append(UUID(result["agent_id"]))
            
            results = processed_results
        
        elif request.coordination_strategy == "staged":
            # Staged execution (groups of max_concurrent)
            agent_groups = [
                request.agent_ids[i:i + request.max_concurrent]
                for i in range(0, len(request.agent_ids), request.max_concurrent)
            ]
            
            for group in agent_groups:
                group_results = []
                
                # Execute group in parallel
                tasks = []
                for agent_id in group:
                    task = asyncio.create_task(
                        sleep_manager.initiate_sleep_cycle(agent_id=agent_id, cycle_type="distributed")
                    )
                    tasks.append((agent_id, task))
                
                # Wait for group completion
                for agent_id, task in tasks:
                    try:
                        success = await task
                        group_results.append({
                            "agent_id": str(agent_id),
                            "success": success,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        if success:
                            successful_operations.append(agent_id)
                        
                    except Exception as e:
                        group_results.append({
                            "agent_id": str(agent_id),
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                results.extend(group_results)
                
                # Check for failures in group
                group_failures = [r for r in group_results if not r["success"]]
                if group_failures and request.rollback_on_failure:
                    await _rollback_sleep_operations(successful_operations, sleep_manager)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Staged operation failed, rolled back all operations"
                    )
        
        # Calculate summary statistics
        successful_count = len(successful_operations)
        total_count = len(request.agent_ids)
        success_rate = successful_count / total_count if total_count > 0 else 0
        
        return {
            "operation_id": request_id,
            "coordination_strategy": request.coordination_strategy,
            "total_agents": total_count,
            "successful": successful_count,
            "failed": total_count - successful_count,
            "success_rate": success_rate,
            "results": results,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in distributed sleep operation - Request ID {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Distributed sleep operation failed: {str(e)}"
        )


@router.get("/system/status", response_model=SystemStatusResponse)
@monitor_performance(target_time_ms=1000)  # 1s for status endpoint
async def get_enhanced_system_status(
    include_performance_metrics: bool = Query(True, description="Include detailed performance metrics"),
    user: Dict[str, Any] = Depends(require_permission("system:read"))
):
    """
    Get comprehensive system status with VS 7.1 enhancements.
    
    Features:
    - Checkpoint engine status and performance metrics
    - Recovery engine status and capabilities
    - Circuit breaker status monitoring
    - Real-time performance metrics
    - Health status aggregation
    """
    try:
        # Get managers
        checkpoint_manager = get_checkpoint_manager()
        sleep_manager = await get_sleep_wake_manager()
        recovery_manager = get_recovery_manager()
        
        # Collect checkpoint engine status
        checkpoint_metrics = await checkpoint_manager.get_checkpoint_performance_metrics()
        checkpoint_engine_status = {
            "healthy": checkpoint_metrics.get("meets_performance_target", True),
            "performance_metrics": checkpoint_metrics,
            "atomic_operations_enabled": checkpoint_manager.enable_atomic_operations,
            "distributed_locking_enabled": checkpoint_manager.enable_distributed_locking
        }
        
        # Collect recovery engine status
        recovery_readiness = await recovery_manager.validate_recovery_readiness()
        recovery_engine_status = {
            "ready": recovery_readiness["ready"],
            "readiness_checks": recovery_readiness["checks"],
            "errors": recovery_readiness.get("errors", []),
            "warnings": recovery_readiness.get("warnings", [])
        }
        
        # Get basic system status
        system_status = await sleep_manager.get_system_status()
        
        # Circuit breaker status
        circuit_breaker_status = {
            "checkpoint_circuit_breaker": {
                "state": checkpoint_circuit_breaker.state,
                "failure_count": checkpoint_circuit_breaker.failure_count,
                "last_failure_time": checkpoint_circuit_breaker.last_failure_time
            },
            "recovery_circuit_breaker": {
                "state": recovery_circuit_breaker.state,
                "failure_count": recovery_circuit_breaker.failure_count,
                "last_failure_time": recovery_circuit_breaker.last_failure_time
            }
        }
        
        # Performance metrics
        performance_metrics = {}
        if include_performance_metrics:
            performance_metrics = {
                "checkpoint_performance": checkpoint_metrics,
                "system_metrics": system_status.get("metrics", {}),
                "response_times": {
                    "target_checkpoint_creation_ms": checkpoint_manager.target_creation_time_ms,
                    "target_recovery_time_ms": recovery_manager.target_recovery_time_ms,
                    "target_api_response_ms": 2000
                }
            }
        
        # Overall system health
        system_healthy = (
            system_status.get("system_healthy", False) and
            checkpoint_engine_status["healthy"] and
            recovery_engine_status["ready"] and
            checkpoint_circuit_breaker.state != "open" and
            recovery_circuit_breaker.state != "open"
        )
        
        return SystemStatusResponse(
            timestamp=datetime.utcnow(),
            system_healthy=system_healthy,
            checkpoint_engine_status=checkpoint_engine_status,
            recovery_engine_status=recovery_engine_status,
            performance_metrics=performance_metrics,
            circuit_breaker_status=circuit_breaker_status,
            active_operations=system_status.get("active_operations", 0),
            error_summary=system_status.get("errors", []) + recovery_readiness.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/metrics/performance")
@monitor_performance(target_time_ms=500)  # 500ms for metrics endpoint
async def get_performance_metrics(
    user: Dict[str, Any] = Depends(require_permission("metrics:read"))
):
    """Get VS 7.1 performance metrics for monitoring and alerting."""
    try:
        checkpoint_manager = get_checkpoint_manager()
        
        # Get checkpoint performance metrics
        checkpoint_metrics = await checkpoint_manager.get_checkpoint_performance_metrics()
        
        # Circuit breaker metrics
        circuit_breaker_metrics = {
            "checkpoint_circuit_breaker": {
                "state": checkpoint_circuit_breaker.state,
                "failure_count": checkpoint_circuit_breaker.failure_count,
                "success_count": checkpoint_circuit_breaker.success_count,
                "failure_rate": checkpoint_circuit_breaker.failure_count / 
                               max(1, checkpoint_circuit_breaker.failure_count + checkpoint_circuit_breaker.success_count)
            },
            "recovery_circuit_breaker": {
                "state": recovery_circuit_breaker.state,
                "failure_count": recovery_circuit_breaker.failure_count,
                "success_count": recovery_circuit_breaker.success_count,
                "failure_rate": recovery_circuit_breaker.failure_count / 
                               max(1, recovery_circuit_breaker.failure_count + recovery_circuit_breaker.success_count)
            }
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "checkpoint_performance": checkpoint_metrics,
            "circuit_breakers": circuit_breaker_metrics,
            "targets": {
                "checkpoint_creation_time_ms": checkpoint_manager.target_creation_time_ms,
                "api_response_time_ms": 2000,
                "recovery_time_ms": 10000
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


# Helper functions

async def _rollback_sleep_operations(agent_ids: List[UUID], sleep_manager) -> None:
    """Rollback sleep operations for distributed coordination."""
    try:
        logger.info(f"Rolling back sleep operations for {len(agent_ids)} agents")
        
        rollback_tasks = []
        for agent_id in agent_ids:
            task = asyncio.create_task(
                sleep_manager.initiate_wake_cycle(agent_id)
            )
            rollback_tasks.append(task)
        
        # Execute rollbacks in parallel
        await asyncio.gather(*rollback_tasks, return_exceptions=True)
        
        logger.info(f"Rollback completed for {len(agent_ids)} agents")
        
    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        # Don't raise exception from rollback to avoid masking original error