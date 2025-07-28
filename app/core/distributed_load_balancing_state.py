"""
Distributed Load Balancing State Management for LeanVibe Agent Hive 2.0

Redis-based distributed state management for load balancing metrics, enabling
multi-instance coordination and real-time metrics sharing across the system.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis

from .config import settings
from .agent_load_balancer import AgentLoadState, LoadBalancingDecision
from .capacity_manager import ScalingDecision, ResourceAllocation, CapacityTier
from .health_monitor import HealthStatus, HealthAlert

logger = structlog.get_logger()


class StateKey(Enum):
    """Redis key patterns for distributed state."""
    AGENT_LOAD_STATE = "lb:agent_load:{agent_id}"
    LOAD_BALANCING_DECISIONS = "lb:decisions"
    SCALING_DECISIONS = "lb:scaling_decisions"
    CAPACITY_ALLOCATIONS = "lb:capacity:{agent_id}"
    HEALTH_STATUS = "lb:health:{agent_id}"
    SYSTEM_METRICS = "lb:system_metrics"
    PERFORMANCE_METRICS = "lb:performance:{timestamp}"
    CLUSTER_COORDINATION = "lb:cluster"
    LEADER_ELECTION = "lb:leader"
    DISTRIBUTED_LOCKS = "lb:locks:{resource}"


@dataclass
class ClusterNode:
    """Represents a node in the load balancing cluster."""
    node_id: str
    hostname: str
    process_id: int
    started_at: datetime
    last_heartbeat: datetime
    is_leader: bool = False
    load_balancer_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "process_id": self.process_id,
            "started_at": self.started_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "is_leader": self.is_leader,
            "load_balancer_active": self.load_balancer_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterNode':
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            hostname=data["hostname"],
            process_id=data["process_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            is_leader=data.get("is_leader", False),
            load_balancer_active=data.get("load_balancer_active", False)
        )


class DistributedLoadBalancingState:
    """
    Distributed state management for load balancing across multiple instances.
    
    Features:
    - Redis-based state synchronization
    - Leader election for coordinated actions
    - Distributed locking for critical operations
    - Real-time metrics sharing
    - Cluster coordination and heartbeats
    - Failover and recovery mechanisms
    """
    
    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        node_id: Optional[str] = None,
        cluster_name: str = "lb_cluster"
    ):
        self.redis_client = redis_client or self._create_redis_client()
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.cluster_name = cluster_name
        
        # Node information
        self.node_info = ClusterNode(
            node_id=self.node_id,
            hostname=settings.HOSTNAME if hasattr(settings, 'HOSTNAME') else "localhost",
            process_id=0,  # Would be set to actual process ID
            started_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        # State management
        self.is_leader = False
        self.leader_node_id: Optional[str] = None
        self.cluster_nodes: Dict[str, ClusterNode] = {}
        
        # Coordination tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.leader_election_task: Optional[asyncio.Task] = None
        self.state_sync_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.config = {
            "heartbeat_interval": 15,  # seconds
            "leader_election_interval": 30,  # seconds
            "state_sync_interval": 10,  # seconds
            "node_timeout": 60,  # seconds before considering node dead
            "leader_lease_duration": 45,  # seconds
            "max_state_history": 1000,
            "metrics_retention_hours": 24,
            "lock_timeout": 30  # seconds
        }
        
        logger.info("DistributedLoadBalancingState initialized",
                   node_id=self.node_id,
                   cluster_name=cluster_name)
    
    def _create_redis_client(self) -> Redis:
        """Create Redis client with connection pooling."""
        return redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
    
    async def start_distributed_coordination(self) -> None:
        """Start distributed coordination services."""
        try:
            # Start heartbeat
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start leader election
            self.leader_election_task = asyncio.create_task(self._leader_election_loop())
            
            # Start state synchronization
            self.state_sync_task = asyncio.create_task(self._state_sync_loop())
            
            logger.info("Distributed coordination started")
            
        except Exception as e:
            logger.error("Failed to start distributed coordination", error=str(e))
            raise
    
    async def stop_distributed_coordination(self) -> None:
        """Stop distributed coordination services."""
        try:
            # Cancel tasks
            tasks = [self.heartbeat_task, self.leader_election_task, self.state_sync_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
            
            # Wait for cancellation
            if tasks:
                await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
            
            # Cleanup node from cluster
            await self._cleanup_node()
            
            logger.info("Distributed coordination stopped")
            
        except Exception as e:
            logger.error("Error stopping distributed coordination", error=str(e))
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to maintain cluster membership."""
        while True:
            try:
                self.node_info.last_heartbeat = datetime.utcnow()
                
                # Update node information in Redis
                node_key = f"{self.cluster_name}:nodes:{self.node_id}"
                await self.redis_client.hset(
                    node_key,
                    mapping={k: v for k, v in self.node_info.to_dict().items()}
                )
                await self.redis_client.expire(node_key, self.config["node_timeout"])
                
                # Discover other nodes
                await self._discover_cluster_nodes()
                
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                logger.error("Error in heartbeat loop", error=str(e))
                await asyncio.sleep(self.config["heartbeat_interval"])
    
    async def _leader_election_loop(self) -> None:
        """Handle leader election process."""
        while True:
            try:
                await self._participate_in_leader_election()
                await asyncio.sleep(self.config["leader_election_interval"])
                
            except Exception as e:
                logger.error("Error in leader election loop", error=str(e))
                await asyncio.sleep(self.config["leader_election_interval"])
    
    async def _state_sync_loop(self) -> None:
        """Synchronize state across cluster nodes."""
        while True:
            try:
                if self.is_leader:
                    await self._coordinate_cluster_state()
                else:
                    await self._sync_from_leader()
                
                await asyncio.sleep(self.config["state_sync_interval"])
                
            except Exception as e:
                logger.error("Error in state sync loop", error=str(e))
                await asyncio.sleep(self.config["state_sync_interval"])
    
    async def _discover_cluster_nodes(self) -> None:
        """Discover and update cluster node information."""
        try:
            # Get all node keys
            node_pattern = f"{self.cluster_name}:nodes:*"
            node_keys = await self.redis_client.keys(node_pattern)
            
            current_nodes = {}
            
            for key in node_keys:
                node_data = await self.redis_client.hgetall(key)
                if node_data:
                    try:
                        node = ClusterNode.from_dict(node_data)
                        current_nodes[node.node_id] = node
                    except Exception as e:
                        logger.warning("Invalid node data", key=key, error=str(e))
            
            # Remove expired nodes
            now = datetime.utcnow()
            active_nodes = {}
            
            for node_id, node in current_nodes.items():
                time_since_heartbeat = (now - node.last_heartbeat).seconds
                if time_since_heartbeat < self.config["node_timeout"]:
                    active_nodes[node_id] = node
                else:
                    logger.info("Node expired", node_id=node_id,
                               time_since_heartbeat=time_since_heartbeat)
            
            self.cluster_nodes = active_nodes
            
        except Exception as e:
            logger.error("Error discovering cluster nodes", error=str(e))
    
    async def _participate_in_leader_election(self) -> None:
        """Participate in leader election process."""
        try:
            leader_key = f"{self.cluster_name}:leader"
            
            # Try to become leader if no current leader
            current_leader = await self.redis_client.get(leader_key)
            
            if not current_leader:
                # Try to acquire leadership
                success = await self.redis_client.set(
                    leader_key,
                    self.node_id,
                    ex=self.config["leader_lease_duration"],
                    nx=True  # Only set if key doesn't exist
                )
                
                if success:
                    self.is_leader = True
                    self.leader_node_id = self.node_id
                    self.node_info.is_leader = True
                    
                    logger.info("Became cluster leader", node_id=self.node_id)
                else:
                    # Someone else became leader
                    current_leader = await self.redis_client.get(leader_key)
                    self.is_leader = False
                    self.leader_node_id = current_leader
                    self.node_info.is_leader = False
            
            elif current_leader == self.node_id:
                # Extend leadership lease
                await self.redis_client.expire(leader_key, self.config["leader_lease_duration"])
                self.is_leader = True
                self.leader_node_id = self.node_id
                self.node_info.is_leader = True
            
            else:
                # Another node is leader
                self.is_leader = False
                self.leader_node_id = current_leader
                self.node_info.is_leader = False
                
                # Check if leader is still alive
                if current_leader in self.cluster_nodes:
                    leader_node = self.cluster_nodes[current_leader]
                    time_since_heartbeat = (datetime.utcnow() - leader_node.last_heartbeat).seconds
                    
                    if time_since_heartbeat > self.config["node_timeout"]:
                        # Leader appears dead, try to take over
                        logger.warning("Leader appears dead, attempting takeover",
                                     dead_leader=current_leader)
                        await self.redis_client.delete(leader_key)
        
        except Exception as e:
            logger.error("Error in leader election", error=str(e))
    
    async def _coordinate_cluster_state(self) -> None:
        """Coordinate cluster state as leader."""
        try:
            # As leader, collect and coordinate state from all nodes
            coordination_data = {
                "leader_node_id": self.node_id,
                "cluster_size": len(self.cluster_nodes),
                "active_nodes": list(self.cluster_nodes.keys()),
                "last_coordination": datetime.utcnow().isoformat()
            }
            
            # Store coordination data
            coord_key = f"{self.cluster_name}:coordination"
            await self.redis_client.hset(coord_key, mapping=coordination_data)
            await self.redis_client.expire(coord_key, 300)  # 5 minutes
            
        except Exception as e:
            logger.error("Error coordinating cluster state", error=str(e))
    
    async def _sync_from_leader(self) -> None:
        """Sync state from leader node."""
        try:
            if not self.leader_node_id:
                return
            
            # Get coordination data from leader
            coord_key = f"{self.cluster_name}:coordination"
            coord_data = await self.redis_client.hgetall(coord_key)
            
            if coord_data and coord_data.get("leader_node_id") == self.leader_node_id:
                # Update local state based on leader coordination
                pass
        
        except Exception as e:
            logger.error("Error syncing from leader", error=str(e))
    
    async def _cleanup_node(self) -> None:
        """Cleanup node information when shutting down."""
        try:
            # Remove node from cluster
            node_key = f"{self.cluster_name}:nodes:{self.node_id}"
            await self.redis_client.delete(node_key)
            
            # Release leadership if we're the leader
            if self.is_leader:
                leader_key = f"{self.cluster_name}:leader"
                await self.redis_client.delete(leader_key)
        
        except Exception as e:
            logger.error("Error cleaning up node", error=str(e))
    
    # Agent Load State Management
    
    async def store_agent_load_state(self, agent_id: str, load_state: AgentLoadState) -> None:
        """Store agent load state in distributed cache."""
        try:
            key = StateKey.AGENT_LOAD_STATE.value.format(agent_id=agent_id)
            data = asdict(load_state)
            data["last_updated"] = load_state.last_updated.isoformat()
            
            await self.redis_client.hset(key, mapping=data)
            await self.redis_client.expire(key, 300)  # 5 minutes TTL
            
        except Exception as e:
            logger.error("Error storing agent load state", agent_id=agent_id, error=str(e))
    
    async def get_agent_load_state(self, agent_id: str) -> Optional[AgentLoadState]:
        """Retrieve agent load state from distributed cache."""
        try:
            key = StateKey.AGENT_LOAD_STATE.value.format(agent_id=agent_id)
            data = await self.redis_client.hgetall(key)
            
            if not data:
                return None
            
            # Convert back to AgentLoadState
            load_state = AgentLoadState(
                agent_id=data["agent_id"],
                active_tasks=int(data.get("active_tasks", 0)),
                pending_tasks=int(data.get("pending_tasks", 0)),
                context_usage_percent=float(data.get("context_usage_percent", 0)),
                memory_usage_mb=float(data.get("memory_usage_mb", 0)),
                cpu_usage_percent=float(data.get("cpu_usage_percent", 0)),
                average_response_time_ms=float(data.get("average_response_time_ms", 0)),
                error_rate_percent=float(data.get("error_rate_percent", 0)),
                throughput_tasks_per_hour=float(data.get("throughput_tasks_per_hour", 0)),
                estimated_capacity=float(data.get("estimated_capacity", 1.0)),
                utilization_ratio=float(data.get("utilization_ratio", 0)),
                health_score=float(data.get("health_score", 1.0)),
                last_updated=datetime.fromisoformat(data["last_updated"])
            )
            
            return load_state
            
        except Exception as e:
            logger.error("Error retrieving agent load state", agent_id=agent_id, error=str(e))
            return None
    
    async def get_all_agent_load_states(self) -> Dict[str, AgentLoadState]:
        """Get all agent load states from distributed cache."""
        try:
            pattern = StateKey.AGENT_LOAD_STATE.value.format(agent_id="*")
            keys = await self.redis_client.keys(pattern)
            
            load_states = {}
            
            for key in keys:
                # Extract agent_id from key
                agent_id = key.split(":")[-1]
                load_state = await self.get_agent_load_state(agent_id)
                if load_state:
                    load_states[agent_id] = load_state
            
            return load_states
            
        except Exception as e:
            logger.error("Error retrieving all agent load states", error=str(e))
            return {}
    
    # Load Balancing Decisions
    
    async def store_load_balancing_decision(self, decision: LoadBalancingDecision) -> None:
        """Store load balancing decision for analysis."""
        try:
            decision_data = {
                "selected_agent_id": decision.selected_agent_id,
                "strategy_used": decision.strategy_used.value,
                "decision_time_ms": decision.decision_time_ms,
                "decision_confidence": decision.decision_confidence,
                "reasoning": decision.reasoning,
                "timestamp": datetime.utcnow().isoformat(),
                "node_id": self.node_id
            }
            
            # Store as list entry with timestamp-based scoring
            score = int(time.time() * 1000)  # Millisecond timestamp
            await self.redis_client.zadd(
                StateKey.LOAD_BALANCING_DECISIONS.value,
                {json.dumps(decision_data): score}
            )
            
            # Keep only recent decisions
            await self.redis_client.zremrangebyrank(
                StateKey.LOAD_BALANCING_DECISIONS.value,
                0, -self.config["max_state_history"]
            )
            
        except Exception as e:
            logger.error("Error storing load balancing decision", error=str(e))
    
    async def get_recent_load_balancing_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent load balancing decisions."""
        try:
            # Get most recent decisions
            decisions_data = await self.redis_client.zrevrange(
                StateKey.LOAD_BALANCING_DECISIONS.value,
                0, limit - 1,
                withscores=False
            )
            
            decisions = []
            for data in decisions_data:
                try:
                    decision = json.loads(data)
                    decisions.append(decision)
                except json.JSONDecodeError:
                    continue
            
            return decisions
            
        except Exception as e:
            logger.error("Error retrieving load balancing decisions", error=str(e))
            return []
    
    # System Metrics
    
    async def store_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store system-wide metrics."""
        try:
            metrics_data = {
                **metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "node_id": self.node_id
            }
            
            await self.redis_client.hset(
                StateKey.SYSTEM_METRICS.value,
                mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                        for k, v in metrics_data.items()}
            )
            await self.redis_client.expire(StateKey.SYSTEM_METRICS.value, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error("Error storing system metrics", error=str(e))
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            data = await self.redis_client.hgetall(StateKey.SYSTEM_METRICS.value)
            
            if not data:
                return {}
            
            # Parse JSON values
            metrics = {}
            for key, value in data.items():
                try:
                    metrics[key] = json.loads(value)
                except json.JSONDecodeError:
                    metrics[key] = value
            
            return metrics
            
        except Exception as e:
            logger.error("Error retrieving system metrics", error=str(e))
            return {}
    
    # Distributed Locking
    
    async def acquire_distributed_lock(
        self,
        resource: str,
        timeout: Optional[int] = None
    ) -> Optional[str]:
        """Acquire distributed lock for resource."""
        try:
            timeout = timeout or self.config["lock_timeout"]
            lock_id = f"{self.node_id}:{uuid.uuid4().hex}"
            lock_key = StateKey.DISTRIBUTED_LOCKS.value.format(resource=resource)
            
            # Try to acquire lock
            success = await self.redis_client.set(
                lock_key,
                lock_id,
                ex=timeout,
                nx=True
            )
            
            if success:
                logger.debug("Acquired distributed lock",
                           resource=resource,
                           lock_id=lock_id,
                           timeout=timeout)
                return lock_id
            
            return None
            
        except Exception as e:
            logger.error("Error acquiring distributed lock", resource=resource, error=str(e))
            return None
    
    async def release_distributed_lock(self, resource: str, lock_id: str) -> bool:
        """Release distributed lock if we own it."""
        try:
            lock_key = StateKey.DISTRIBUTED_LOCKS.value.format(resource=resource)
            
            # Lua script to atomically check and release lock
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            
            result = await self.redis_client.eval(lua_script, 1, lock_key, lock_id)
            
            success = bool(result)
            if success:
                logger.debug("Released distributed lock",
                           resource=resource,
                           lock_id=lock_id)
            
            return success
            
        except Exception as e:
            logger.error("Error releasing distributed lock", resource=resource, error=str(e))
            return False
    
    async def extend_distributed_lock(
        self,
        resource: str,
        lock_id: str,
        extend_by: int = 30
    ) -> bool:
        """Extend distributed lock TTL."""
        try:
            lock_key = StateKey.DISTRIBUTED_LOCKS.value.format(resource=resource)
            
            # Lua script to atomically check and extend lock
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("expire", KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            
            result = await self.redis_client.eval(
                lua_script, 1, lock_key, lock_id, extend_by
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error("Error extending distributed lock", resource=resource, error=str(e))
            return False
    
    # Cluster Information
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        try:
            return {
                "cluster_name": self.cluster_name,
                "node_id": self.node_id,
                "is_leader": self.is_leader,
                "leader_node_id": self.leader_node_id,
                "cluster_size": len(self.cluster_nodes),
                "nodes": {
                    node_id: node.to_dict()
                    for node_id, node in self.cluster_nodes.items()
                },
                "coordination_active": bool(self.heartbeat_task and not self.heartbeat_task.done()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting cluster status", error=str(e))
            return {"error": str(e)}
    
    # Context manager for distributed locks
    
    class DistributedLock:
        """Context manager for distributed locks."""
        
        def __init__(self, state_manager: 'DistributedLoadBalancingState', resource: str, timeout: int = 30):
            self.state_manager = state_manager
            self.resource = resource
            self.timeout = timeout
            self.lock_id: Optional[str] = None
        
        async def __aenter__(self) -> bool:
            self.lock_id = await self.state_manager.acquire_distributed_lock(
                self.resource, self.timeout
            )
            return self.lock_id is not None
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.lock_id:
                await self.state_manager.release_distributed_lock(
                    self.resource, self.lock_id
                )
    
    def distributed_lock(self, resource: str, timeout: int = 30) -> 'DistributedLoadBalancingState.DistributedLock':
        """Create distributed lock context manager."""
        return self.DistributedLock(self, resource, timeout)
    
    # Metrics aggregation across cluster
    
    async def get_cluster_wide_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all cluster nodes."""
        try:
            all_metrics = []
            
            # Collect metrics from all nodes
            for node_id in self.cluster_nodes.keys():
                node_metrics_key = f"{self.cluster_name}:node_metrics:{node_id}"
                node_metrics = await self.redis_client.hgetall(node_metrics_key)
                
                if node_metrics:
                    try:
                        parsed_metrics = {}
                        for key, value in node_metrics.items():
                            try:
                                parsed_metrics[key] = json.loads(value)
                            except json.JSONDecodeError:
                                parsed_metrics[key] = value
                        
                        all_metrics.append({
                            "node_id": node_id,
                            "metrics": parsed_metrics
                        })
                    except Exception as e:
                        logger.warning("Error parsing node metrics", node_id=node_id, error=str(e))
            
            # Aggregate metrics
            if not all_metrics:
                return {"error": "No metrics available"}
            
            # Simple aggregation - sum numeric values, average where appropriate
            aggregated = {
                "cluster_size": len(self.cluster_nodes),
                "reporting_nodes": len(all_metrics),
                "total_agents": sum(
                    m["metrics"].get("total_agents", 0) for m in all_metrics
                ),
                "avg_load_factor": statistics.mean([
                    m["metrics"].get("avg_load_factor", 0) for m in all_metrics
                    if m["metrics"].get("avg_load_factor", 0) > 0
                ]) if any(m["metrics"].get("avg_load_factor", 0) > 0 for m in all_metrics) else 0,
                "total_decisions": sum(
                    m["metrics"].get("total_decisions", 0) for m in all_metrics
                ),
                "nodes": all_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return aggregated
            
        except Exception as e:
            logger.error("Error getting cluster-wide metrics", error=str(e))
            return {"error": str(e)}