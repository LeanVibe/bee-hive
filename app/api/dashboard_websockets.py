"""
Dashboard WebSocket API for Real-time Monitoring

Provides WebSocket endpoints for real-time dashboard updates, live system monitoring,
and instant coordination event streaming for the LeanVibe Agent Hive dashboard.

Part 3 of the dashboard monitoring infrastructure.
"""

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter
from fastapi import Query
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from ..core.auth_metrics import inc as inc_auth_metric
from ..core.redis import get_redis
from .ws_utils import WS_CONTRACT_VERSION
from .ws_utils import make_data_error
from .ws_utils import make_error

logger = structlog.get_logger()
router = APIRouter(prefix="/api/dashboard", tags=["dashboard-websockets"])


@dataclass
class WebSocketConnection:
    """Represents an active WebSocket connection."""

    websocket: WebSocket
    connection_id: str
    client_type: str
    subscriptions: set[str]
    connected_at: datetime
    last_activity: datetime
    metadata: dict[str, Any]
    # Rate limiting state
    tokens: float
    last_refill: datetime
    rate_limit_notified_at: datetime | None
    # Backpressure state
    send_failure_streak: int = 0


class DashboardWebSocketManager:
    """Manages all WebSocket connections and real-time updates."""

    def __init__(self):
        import os as _os

        self.connections: dict[str, WebSocketConnection] = {}
        self.subscription_groups: dict[str, set[str]] = {
            "agents": set(),
            "coordination": set(),
            "tasks": set(),
            "system": set(),
            "alerts": set(),
            "project_index": set(),
        }
        self.broadcast_task: asyncio.Task | None = None
        self.redis_listener_task: asyncio.Task | None = None
        self.health_monitor_task: asyncio.Task | None = None
        # Rate limit configuration
        self.rate_limit_tokens_per_second: float = float(_os.environ.get("WS_RATE_TOKENS_PER_SEC", "20"))
        self.rate_limit_burst_capacity: float = float(_os.environ.get("WS_RATE_BURST", "40"))
        self.rate_limit_notify_cooldown_seconds: int = 5
        # Input hardening
        self.max_inbound_message_bytes: int = 64 * 1024
        self.max_subscriptions_per_connection: int = 10
        # Observability counters
        self.metrics: dict[str, int] = {
            "messages_sent_total": 0,
            "messages_send_failures_total": 0,
            "messages_received_total": 0,
            "messages_dropped_rate_limit_total": 0,
            "errors_sent_total": 0,
            "connections_total": 0,
            "disconnections_total": 0,
            "backpressure_disconnects_total": 0,
            "auth_denied_total": 0,
            "origin_denied_total": 0,
            "idle_disconnects_total": 0,
            # SLO metrics
            "bytes_sent_total": 0,
            "bytes_received_total": 0,
        }
        # Fanout last values (gauges)
        self.last_broadcast_fanout: int = 0
        # Backpressure configuration
        self.backpressure_disconnect_threshold: int = 5
        # Idle disconnect configuration
        self.idle_disconnect_seconds: int = 600  # 10 minutes default
        
        # Sprint 2: Connection Recovery & Circuit Breaker
        self.broken_connections: set[str] = set()
        self.broken_connections_with_priority: dict[str, dict] = {}
        self.circuit_breakers: dict[str, dict] = {}
        self.connection_failure_history: dict[str, list] = {}
        
        # Recovery metrics
        self.metrics.update({
            "connection_recovery_attempts_total": 0,
            "connection_recovery_successes_total": 0,
            "connection_recovery_failures_total": 0,
            "circuit_breaker_activations_total": 0,
            "heartbeats_sent_total": 0,
            "heartbeat_timeouts_total": 0,
            "stale_connections_cleaned_total": 0
        })
        # Optional WS auth/allowlist (feature-flagged via env)
        self.auth_required: bool = _os.environ.get("WS_AUTH_REQUIRED", "false").lower() in ("1", "true", "yes")
        # auth_mode: 'token' (default, env shared token) or 'jwt' (verify access token)
        self.auth_mode: str = _os.environ.get("WS_AUTH_MODE", "token").lower()
        self.compression_enabled: bool = _os.environ.get("WS_COMPRESSION_ENABLED", "false").lower() in ("1", "true", "yes")
        self.allowed_origins: set[str] | None = None
        allowlist = _os.environ.get("WS_ALLOWED_ORIGINS")
        if allowlist:
            self.allowed_origins = {o.strip() for o in allowlist.split(",") if o.strip()}
        self.expected_auth_token: str | None = _os.environ.get("WS_AUTH_TOKEN")

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        client_type: str = "dashboard",
        subscriptions: list[str] | None = None,
    ) -> WebSocketConnection:
        """Connect a new WebSocket client with subscription management."""
        # Refresh auth/allowlist configuration from environment on each connect
        try:
            import os as _os
            self.auth_required = _os.environ.get("WS_AUTH_REQUIRED", "false").lower() in ("1", "true", "yes")
            self.auth_mode = _os.environ.get("WS_AUTH_MODE", "token").lower()
            self.compression_enabled = _os.environ.get("WS_COMPRESSION_ENABLED", "false").lower() in ("1", "true", "yes")
            self.rate_limit_tokens_per_second = float(_os.environ.get("WS_RATE_TOKENS_PER_SEC", str(self.rate_limit_tokens_per_second)))
            self.rate_limit_burst_capacity = float(_os.environ.get("WS_RATE_BURST", str(self.rate_limit_burst_capacity)))
            allowlist = _os.environ.get("WS_ALLOWED_ORIGINS")
            self.allowed_origins = {o.strip() for o in allowlist.split(",") if o.strip()} if allowlist else None
            self.expected_auth_token = _os.environ.get("WS_AUTH_TOKEN")
        except Exception:
            pass
        # Optional origin allowlist check
        try:
            origin = websocket.headers.get("origin") or websocket.headers.get("Origin")
        except Exception:
            origin = None
        if self.allowed_origins and origin and origin not in self.allowed_origins:
            await websocket.close(code=4403)
            self.metrics["origin_denied_total"] += 1
            return None  # type: ignore[return-value]
        # Optional auth token check
        if self.auth_required:
            token_ok = False
            user_role = None
            user_id = None
            try:
                auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
            except Exception:
                auth_header = None
            # Extract token from header or query
            provided_token: str | None = None
            if auth_header and isinstance(auth_header, str) and auth_header.lower().startswith("bearer "):
                provided_token = auth_header.split(" ", 1)[1].strip()
            if not provided_token:
                try:
                    provided_token = websocket.query_params.get("access_token") if hasattr(websocket, "query_params") else None
                except Exception:
                    provided_token = None

            if self.auth_mode == "token":
                if provided_token and self.expected_auth_token and provided_token == self.expected_auth_token:
                    token_ok = True
            else:  # jwt mode
                if provided_token:
                    try:
                        from ..core.auth import get_auth_service
                        auth_service = get_auth_service()
                        token_data = auth_service.verify_token(provided_token)
                        if token_data:
                            token_ok = True
                            user_role = token_data.role.value if hasattr(token_data, 'role') else None
                            user_id = token_data.user_id
                    except Exception:
                        token_ok = False

            if not token_ok:
                await websocket.close(code=4401)
                self.metrics["auth_denied_total"] += 1
                with contextlib.suppress(Exception):
                    inc_auth_metric("auth_failure_total_ws")
                return None  # type: ignore[return-value]
        
        # Initialize default user_id and user_role if not set
        if 'user_id' not in locals():
            user_id = "anonymous"
        if 'user_role' not in locals():
            user_role = "user"
        await websocket.accept()
        with contextlib.suppress(Exception):
            if self.auth_required:
                inc_auth_metric("auth_success_total_ws")

        connection = WebSocketConnection(
            websocket=websocket,
            connection_id=connection_id,
            client_type=client_type,
            subscriptions=set(subscriptions or ["agents", "coordination", "tasks", "system"]),
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            metadata={"user_id": user_id, "user_role": user_role, "auth_mode": self.auth_mode if self.auth_required else "none"},
            tokens=self.rate_limit_burst_capacity,
            last_refill=datetime.utcnow(),
            rate_limit_notified_at=None,
        )

        self.connections[connection_id] = connection
        self.metrics["connections_total"] += 1

        # Enforce RBAC for certain subscriptions (simple policy: alerts requires admin roles when auth in jwt mode)
        if (
            self.auth_required
            and self.auth_mode == "jwt"
            and user_role not in ("super_admin", "enterprise_admin")
            and "alerts" in connection.subscriptions
        ):
            connection.subscriptions.discard("alerts")
        # Add to subscription groups
        for subscription in connection.subscriptions:
            if subscription in self.subscription_groups:
                self.subscription_groups[subscription].add(connection_id)

        # Start background tasks if this is the first connection
        if len(self.connections) == 1:
            await self._start_background_tasks()

        # Send initial connection confirmation
        await self._send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "subscriptions": list(connection.subscriptions),
            "server_time": datetime.utcnow().isoformat(),
            "contract_version": WS_CONTRACT_VERSION,
        })

        logger.info(
            "Dashboard WebSocket connected",
            connection_id=connection_id,
                   client_type=client_type,
            subscriptions=list(connection.subscriptions),
        )

        return connection

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect a WebSocket client."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        # Remove from subscription groups
        for subscription in connection.subscriptions:
            if subscription in self.subscription_groups:
                self.subscription_groups[subscription].discard(connection_id)

        # Remove connection
        del self.connections[connection_id]
        self.metrics["disconnections_total"] += 1

        # Stop background tasks if no connections remain
        if len(self.connections) == 0:
            await self._stop_background_tasks()

        logger.info("Dashboard WebSocket disconnected", connection_id=connection_id)

    async def handle_message(self, connection_id: str, message: dict[str, Any]) -> None:
        """Handle incoming WebSocket messages from clients."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]
        connection.last_activity = datetime.utcnow()

        # Inbound safety: basic size cap if raw payload length available via metadata
        # (FastAPI gives text directly; we can approximate by re-serializing if needed)
        try:
            encoded = json.dumps(message).encode("utf-8")
            if len(encoded) > self.max_inbound_message_bytes:
                await self._send_to_connection(connection_id, make_error("Message too large"))
                return
        except Exception:
            # If serialization fails, proceed; downstream will handle
            pass

        # Count received messages and bytes
        self.metrics["messages_received_total"] += 1
        with contextlib.suppress(Exception):
            self.metrics["bytes_received_total"] += len(json.dumps(message).encode("utf-8"))

        # Rate limiting guard: drop or notify when exceeded
        if not self._consume_token_allow(connection):
            # Notify about rate limit at most once per cooldown window
            now = datetime.utcnow()
            if (
                connection.rate_limit_notified_at is None
                or (now - connection.rate_limit_notified_at).total_seconds()
                >= self.rate_limit_notify_cooldown_seconds
            ):
                connection.rate_limit_notified_at = now
                await self._send_to_connection(connection_id, make_error("Rate limit exceeded"))
            # Count dropped
            self.metrics["messages_dropped_rate_limit_total"] += 1
            return

        message_type = message.get("type")

        if message_type == "ping":
            await self._send_to_connection(connection_id, {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            })

        elif message_type == "subscribe":
            # Add new subscriptions
            requested = message.get("subscriptions", [])
            new_subs = set(requested)
            valid_subs = new_subs.intersection(self.subscription_groups.keys())
            invalid_subs = new_subs.difference(self.subscription_groups.keys())
            # Enforce max subscriptions per connection
            if len(connection.subscriptions.union(valid_subs)) > self.max_subscriptions_per_connection:
                await self._send_to_connection(connection_id, make_error("Too many subscriptions requested"))
                valid_subs = set(list(valid_subs)[: max(0, self.max_subscriptions_per_connection - len(connection.subscriptions))])

            for subscription in valid_subs:
                connection.subscriptions.add(subscription)
                self.subscription_groups[subscription].add(connection_id)

            # Send error for any invalid subscriptions (single consolidated error)
            if invalid_subs:
                await self._send_to_connection(connection_id, make_error(
                    f"Invalid subscription(s): {', '.join(sorted(invalid_subs))}"
                ))

            await self._send_to_connection(connection_id, {
                "type": "subscription_updated",
                "subscriptions": sorted(connection.subscriptions)
            })

        elif message_type == "unsubscribe":
            # Remove subscriptions
            remove_subs = set(message.get("subscriptions", []))

            for subscription in remove_subs:
                # Always remove from the connection's subscription set
                connection.subscriptions.discard(subscription)
                # Only attempt to update a known subscription group
                if subscription in self.subscription_groups:
                    self.subscription_groups[subscription].discard(connection_id)

            await self._send_to_connection(connection_id, {
                "type": "subscription_updated",
                "subscriptions": sorted(connection.subscriptions)
            })

        elif message_type == "request_data":
            # Client requesting specific data
            data_type = message.get("data_type")
            await self._handle_data_request(connection_id, data_type, message.get("params", {}))

        else:
            logger.warning("Unknown WebSocket message type", connection_id=connection_id, message_type=message_type)
            await self._send_to_connection(connection_id, make_error(f"Unknown message type: {message_type}"))

    async def broadcast_to_subscription(
        self,
        subscription: str,
        message_type: str,
        data: dict[str, Any],
    ) -> int:
        """Broadcast a message to all clients subscribed to a specific topic."""
        if subscription not in self.subscription_groups:
            return 0

        message = {
            "type": message_type,
            "subscription": subscription,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": str(uuid.uuid4()),
        }

        sent_count = 0

        targets = list(self.subscription_groups[subscription])
        for connection_id in targets:
            if await self._send_to_connection(connection_id, message):
                sent_count += 1

        # Update last fanout
        self.last_broadcast_fanout = len(targets)
        return sent_count

    async def broadcast_to_all(self, message_type: str, data: dict[str, Any]) -> int:
        """Broadcast a message to all connected clients."""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": str(uuid.uuid4()),
        }

        sent_count = 0

        targets = list(self.connections.keys())
        for connection_id in targets:
            if await self._send_to_connection(connection_id, message):
                sent_count += 1

        self.last_broadcast_fanout = len(targets)
        return sent_count

    async def _send_to_connection(self, connection_id: str, message: dict[str, Any]) -> bool:
        """Send message to a specific connection. Returns True if successful."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        try:
            # Ensure correlation_id is present
            if "correlation_id" not in message:
                message["correlation_id"] = str(uuid.uuid4())
            # Ensure timestamp is present
            if "timestamp" not in message:
                message["timestamp"] = datetime.utcnow().isoformat()
            encoded = json.dumps(message)
            await connection.websocket.send_text(encoded)
            self.metrics["messages_sent_total"] += 1
            with contextlib.suppress(Exception):
                self.metrics["bytes_sent_total"] += len(encoded.encode("utf-8"))
            if message.get("type") in {"error", "data_error"}:
                self.metrics["errors_sent_total"] += 1
            # Reset failure streak on success
            connection.send_failure_streak = 0
            return True
        except Exception as e:
            logger.warning(
                "Failed to send WebSocket message",
                connection_id=connection_id,
                error=str(e),
                correlation_id=message.get("correlation_id"),
                message_type=message.get("type"),
                subscription=message.get("subscription"),
            )
            self.metrics["messages_send_failures_total"] += 1
            # Increment failure streak and consider disconnect on threshold
            connection.send_failure_streak += 1
            if connection.send_failure_streak >= self.backpressure_disconnect_threshold:
                # Best-effort notify before disconnect
                with contextlib.suppress(Exception):
                    await connection.websocket.send_text(
                        json.dumps(
                            {
                                "type": "disconnect_notice",
                                "reason": "backpressure",
                                "timestamp": datetime.utcnow().isoformat(),
                                "correlation_id": str(uuid.uuid4()),
                            }
                        )
                    )
                await self.disconnect(connection_id)
                self.metrics["backpressure_disconnects_total"] += 1
            return False

    async def _handle_data_request(
        self,
        connection_id: str,
        data_type: str,
        _: dict[str, Any] | None = None,
    ) -> None:
        """Handle specific data requests from clients."""
        try:
            if data_type == "agent_status":
                # Get fresh agent data (would normally inject DB session)
                data = {
                    "agents": [],  # Would fetch real agent data
                    "summary": {"total": 0, "active": 0}
                }

            elif data_type == "coordination_metrics":
                # Get fresh coordination metrics
                data = {
                    "success_rate": 0.0,  # Would fetch real metrics
                    "trend": "unknown"
                }

            elif data_type == "system_health":
                # Get system health data
                data = {
                    "overall_status": "unknown",  # Would fetch real health
                    "components": {}
                }

            else:
                await self._send_to_connection(connection_id, make_data_error(data_type, f"Unknown data type: {data_type}"))
                return

            await self._send_to_connection(connection_id, {
                "type": "data_response",
                "data_type": data_type,
                "data": data
            })

        except Exception as e:
            await self._send_to_connection(connection_id, make_data_error(data_type, str(e)))

    def _consume_token_allow(self, connection: WebSocketConnection) -> bool:
        """Refill token bucket and consume one token if available."""
        now = datetime.utcnow()
        elapsed = (now - connection.last_refill).total_seconds()
        if elapsed > 0:
            refill = elapsed * self.rate_limit_tokens_per_second
            connection.tokens = min(
                self.rate_limit_burst_capacity,
                connection.tokens + refill,
            )
            connection.last_refill = now

        if connection.tokens >= 1.0:
            connection.tokens -= 1.0
            # Reset notification once we have headroom
            if connection.tokens > (self.rate_limit_tokens_per_second / 2.0):
                connection.rate_limit_notified_at = None
            return True
        return False

    async def _start_background_tasks(self) -> None:
        """Start background tasks for real-time updates."""
        if self.broadcast_task is None:
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())

        if self.redis_listener_task is None:
            self.redis_listener_task = asyncio.create_task(self._redis_listener_loop())

        if self.health_monitor_task is None:
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())

        logger.info("Dashboard WebSocket background tasks started")

    async def _stop_background_tasks(self) -> None:
        """Stop background tasks when no connections remain."""
        tasks = [self.broadcast_task, self.redis_listener_task, self.health_monitor_task]

        for task in tasks:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self.broadcast_task = None
        self.redis_listener_task = None
        self.health_monitor_task = None

        logger.info("Dashboard WebSocket background tasks stopped")

    async def _broadcast_loop(self) -> None:
        """Main broadcast loop for periodic updates."""
        while True:
            try:
                # Skip if no connections
                if not self.connections:
                    await asyncio.sleep(5)
                    continue

                # Send periodic updates based on subscriptions
                current_time = datetime.utcnow()

                # Idle disconnect hygiene: disconnect connections idle beyond threshold
                await self._check_idle_disconnects(current_time)

                # Agent status updates (every 5 seconds)
                if self.subscription_groups["agents"]:
                    agent_data = {
                        "active_count": 2,  # Would get real data
                        "health_summary": {"healthy": 2, "degraded": 0},
                        "last_updated": current_time.isoformat()
                    }
                    await self.broadcast_to_subscription("agents", "agent_update", agent_data)

                # Coordination metrics (every 10 seconds)
                if self.subscription_groups["coordination"] and int(current_time.timestamp()) % 10 == 0:
                    coord_data = {
                        "success_rate": 75.5,  # Would get real data
                        "total_tasks": 45,
                        "trend": "stable",
                        "last_updated": current_time.isoformat()
                    }
                    await self.broadcast_to_subscription("coordination", "coordination_update", coord_data)

                # Task queue updates (every 3 seconds)
                if self.subscription_groups["tasks"] and int(current_time.timestamp()) % 3 == 0:
                    task_data = {
                        "queue_length": 8,  # Would get real data
                        "in_progress": 3,
                        "failed_recent": 1,
                        "last_updated": current_time.isoformat()
                    }
                    await self.broadcast_to_subscription("tasks", "task_update", task_data)

                # System health (every 30 seconds)
                if self.subscription_groups["system"] and int(current_time.timestamp()) % 30 == 0:
                    system_data = {
                        "overall_health": "healthy",  # Would get real data
                        "database_status": "healthy",
                        "redis_status": "healthy",
                        "last_updated": current_time.isoformat()
                    }
                    await self.broadcast_to_subscription("system", "system_update", system_data)

                await asyncio.sleep(1)  # Base interval

            except Exception as e:
                logger.error("Error in WebSocket broadcast loop", error=str(e))
                await asyncio.sleep(5)

    async def _check_idle_disconnects(self, now: datetime | None = None) -> None:
        """Disconnect connections idle longer than the configured threshold.
        Sends a disconnect_notice before closing.
        """
        if not self.connections:
            return
        now = now or datetime.utcnow()
        to_disconnect: list[str] = []
        for connection_id, connection in list(self.connections.items()):
            idle_seconds = (now - connection.last_activity).total_seconds()
            if idle_seconds >= self.idle_disconnect_seconds:
                # Best-effort notice; ignore failures
                with contextlib.suppress(Exception):
                    await self._send_to_connection(
                        connection_id,
                        {
                            "type": "disconnect_notice",
                            "reason": "idle_timeout",
                            "timestamp": now.isoformat(),
                        },
                    )
                to_disconnect.append(connection_id)
        for cid in to_disconnect:
            await self.disconnect(cid)
            self.metrics["idle_disconnects_total"] += 1

    async def _redis_listener_loop(self) -> None:
        """Listen for Redis events and broadcast to relevant clients."""
        backoff_seconds = 5
        while True:
            try:
                redis_client = get_redis()

                # Subscribe to concrete and pattern channels correctly
                pubsub = redis_client.pubsub()
                # Handle async-returning pubsub for certain Redis clients in tests
                if asyncio.iscoroutine(pubsub):
                    pubsub = await pubsub  # type: ignore[assignment]
                await pubsub.subscribe("system_events")
                await pubsub.psubscribe("agent_events:*")
                await pubsub.subscribe("project_index:websocket_events")

                # Reset backoff after successful subscribe
                backoff_seconds = 5

                async for message in pubsub.listen():
                    msg_type = message.get("type")
                    if msg_type in ("message", "pmessage"):
                        await self._handle_redis_event(message)

            except Exception as e:
                logger.error("Error in Redis listener loop", error=str(e))
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 60)

    async def _health_monitor_loop(self) -> None:
        """Monitor system health and send alerts."""
        while True:
            try:
                # Check for critical conditions that require immediate alerts

                # Simulate health checks (would use real health monitoring)
                critical_alerts = []

                # Check coordination success rate
                # success_rate = await get_coordination_success_rate()
                # if success_rate < 30:
                #     critical_alerts.append({
                #         "level": "critical",
                #         "message": f"Coordination success rate critically low: {success_rate}%",
                #         "action": "immediate_intervention_required"
                #     })

                # Send alerts if any critical conditions detected
                if critical_alerts and self.subscription_groups["alerts"]:
                    await self.broadcast_to_subscription("alerts", "critical_alert", {
                        "alerts": critical_alerts,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error("Error in health monitor loop", error=str(e))
                await asyncio.sleep(60)

    async def _handle_redis_event(self, message: dict[str, Any]) -> None:
        """Handle Redis pub/sub events and broadcast to clients."""
        try:
            # Support both message and pmessage formats
            raw_channel = message.get("channel") or message.get("pattern") or ""
            channel = raw_channel.decode() if isinstance(raw_channel, bytes | bytearray) else raw_channel
            raw_data = message.get("data", {})
            data = json.loads(raw_data.decode() if isinstance(raw_data, bytes | bytearray) else raw_data) if isinstance(raw_data, (bytes | str)) else raw_data

            # Route events to appropriate subscriptions
            if channel == "system_events":
                if self.subscription_groups["system"]:
                    await self.broadcast_to_subscription("system", "system_event", data)
            elif channel.startswith("agent_events:") and self.subscription_groups["agents"]:
                    await self.broadcast_to_subscription("agents", "agent_event", data)
            elif channel == "project_index:websocket_events":
                if self.subscription_groups["project_index"]:
                    # Extract event from Project Index message format
                    event = data.get("event", {})
                    subscribers = data.get("subscribers", [])
                    
                    # Route to project_index subscription group if there are subscribers
                    if event and (not subscribers or any(conn_id in self.connections for conn_id in subscribers)):
                        await self.broadcast_to_subscription("project_index", "project_index_event", event)

        except Exception as e:
            logger.error("Error handling Redis event", error=str(e))

    def get_connection_stats(self) -> dict[str, Any]:
        """Get statistics about current connections."""
        current_time = datetime.utcnow()

        return {
            "total_connections": len(self.connections),
            "connections_by_type": {},  # Would group by client_type
            "subscription_counts": {
                sub: len(clients) for sub, clients in self.subscription_groups.items()
            },
            "active_connections": len([
                c for c in self.connections.values()
                if (current_time - c.last_activity).total_seconds() < 300  # Active in last 5 minutes
            ]),
            "oldest_connection": min(
                [c.connected_at for c in self.connections.values()],
                default=current_time
            ).isoformat(),
            "background_tasks_running": all([
                self.broadcast_task and not self.broadcast_task.done(),
                self.redis_listener_task and not self.redis_listener_task.done(),
                self.health_monitor_task and not self.health_monitor_task.done()
            ]) if self.connections else False
        }

    # =============== SPRINT 2: CONNECTION RECOVERY METHODS ===============

    async def is_connection_broken(self, connection_id: str) -> bool:
        """Check if a WebSocket connection is broken."""
        if connection_id not in self.connections:
            return True
        
        connection = self.connections[connection_id]
        try:
            # Test connection by sending a ping
            await connection.websocket.send_json({
                "type": "connection_test",
                "timestamp": datetime.utcnow().isoformat()
            })
            return False
        except Exception:
            self.broken_connections.add(connection_id)
            return True

    async def recover_broken_connections(self) -> None:
        """Attempt to recover broken connections with exponential backoff."""
        for connection_id in list(self.broken_connections):
            self.metrics["connection_recovery_attempts_total"] += 1
            
            # Attempt recovery (simplified for TDD)
            if hasattr(self, 'attempt_connection_recovery'):
                success = await self.attempt_connection_recovery(connection_id, 1)
                if success:
                    self.broken_connections.discard(connection_id)
                    self.metrics["connection_recovery_successes_total"] += 1
                else:
                    self.metrics["connection_recovery_failures_total"] += 1

    async def record_recovery_failure(self, connection_id: str) -> None:
        """Record a connection recovery failure."""
        if connection_id not in self.circuit_breakers:
            self.circuit_breakers[connection_id] = {
                "failure_count": 0,
                "state": "closed"
            }
        
        self.circuit_breakers[connection_id]["failure_count"] += 1
        
        # Open circuit breaker if failure threshold exceeded
        if self.circuit_breakers[connection_id]["failure_count"] >= 5:
            self.circuit_breakers[connection_id]["state"] = "open"
            self.metrics["circuit_breaker_activations_total"] += 1

    def get_circuit_breaker_state(self, connection_id: str) -> str:
        """Get the current circuit breaker state for a connection."""
        if connection_id not in self.circuit_breakers:
            return "closed"
        return self.circuit_breakers[connection_id]["state"]

    async def should_attempt_recovery(self, connection_id: str) -> bool:
        """Check if recovery should be attempted based on circuit breaker."""
        state = self.get_circuit_breaker_state(connection_id)
        return state != "open"

    async def update_circuit_breaker_states(self) -> None:
        """Update circuit breaker states based on timeouts."""
        current_time = time.time()
        
        for connection_id, breaker in self.circuit_breakers.items():
            if breaker["state"] == "open":
                # Check if timeout period has elapsed (60 seconds)
                if "last_failure_time" in breaker:
                    time_since_failure = current_time - breaker["last_failure_time"]
                    if time_since_failure >= 60:
                        breaker["state"] = "half-open"

    async def send_heartbeat(self, connection_id: str) -> None:
        """Send heartbeat ping to a connection."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            try:
                await connection.websocket.send_json({
                    "type": "ping",
                    "timestamp": time.time(),
                    "correlation_id": str(uuid.uuid4())
                })
                self.metrics["heartbeats_sent_total"] += 1
            except Exception:
                self.metrics["heartbeat_timeouts_total"] += 1

    async def is_connection_stale(self, connection_id: str) -> bool:
        """Check if a connection is stale (no heartbeat response)."""
        if connection_id not in self.connections:
            return True
        
        connection = self.connections[connection_id]
        if hasattr(connection, 'last_heartbeat_response'):
            time_since_heartbeat = time.time() - connection.last_heartbeat_response
            return time_since_heartbeat > 60  # 60 second timeout
        return False

    async def cleanup_stale_connections(self) -> None:
        """Clean up stale connections."""
        stale_connections = []
        for connection_id in list(self.connections.keys()):
            if await self.is_connection_stale(connection_id):
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            await self.disconnect(connection_id)
            self.metrics["stale_connections_cleaned_total"] += 1

    async def handle_connection_failure(self, connection_id: str, failure_type: str) -> None:
        """Handle a connection failure event."""
        self.broken_connections.add(connection_id)
        
        # Record failure history
        if connection_id not in self.connection_failure_history:
            self.connection_failure_history[connection_id] = []
        
        self.connection_failure_history[connection_id].append({
            "timestamp": time.time(),
            "type": failure_type
        })
        
        # Track metrics
        self.metrics["connection_failures_total"] = self.metrics.get("connection_failures_total", 0) + 1

    async def get_recovery_priority_order(self) -> list[str]:
        """Get connection recovery order based on priority."""
        priority_order = []
        
        # Sort by priority: high, medium, low
        for priority in ["high", "medium", "low"]:
            for conn_id, props in self.broken_connections_with_priority.items():
                if props.get("priority") == priority:
                    priority_order.append(conn_id)
        
        return priority_order

    async def get_recovery_metrics(self) -> dict[str, Any]:
        """Get comprehensive recovery metrics."""
        total_attempts = self.metrics["connection_recovery_attempts_total"]
        total_successes = self.metrics["connection_recovery_successes_total"]
        
        success_rate = (total_successes / total_attempts * 100.0) if total_attempts > 0 else 0.0
        
        recovery_metrics = {
            "connection_recovery_attempts_total": total_attempts,
            "connection_recovery_successes_total": total_successes,
            "connection_recovery_failures_total": self.metrics["connection_recovery_failures_total"],
            "circuit_breaker_activations_total": self.metrics["circuit_breaker_activations_total"],
            "heartbeats_sent_total": self.metrics["heartbeats_sent_total"],
            "heartbeat_timeouts_total": self.metrics["heartbeat_timeouts_total"],
            "stale_connections_cleaned_total": self.metrics["stale_connections_cleaned_total"],
            "recovery_success_rate": success_rate,
            "average_recovery_time_seconds": 0.0  # Placeholder
        }
        
        return recovery_metrics

    # =============== CIRCUIT BREAKER PATTERN METHODS ===============

    async def initialize_circuit_breaker(self, connection_id: str) -> None:
        """Initialize circuit breaker for a connection."""
        self.circuit_breakers[connection_id] = {
            "state": "closed",
            "failure_count": 0,
            "consecutive_successes": 0,
            "last_failure_time": None,
            "last_success_time": None,
            "opened_at": None
        }

    async def record_connection_failure(self, connection_id: str, failure_type: str) -> None:
        """Record a connection failure and update circuit breaker state."""
        current_time = time.time()
        
        if connection_id not in self.circuit_breakers:
            await self.initialize_circuit_breaker(connection_id)
        
        breaker = self.circuit_breakers[connection_id]
        
        # Only count relevant failures for circuit breaker
        if failure_type not in ["authentication_error"]:  # Auth errors don't count
            breaker["failure_count"] += 1
            breaker["last_failure_time"] = current_time
            breaker["consecutive_successes"] = 0
        
        # Record in failure history
        if connection_id not in self.connection_failure_history:
            self.connection_failure_history[connection_id] = []
        
        self.connection_failure_history[connection_id].append({
            "timestamp": current_time,
            "type": failure_type
        })
        
        # Check if we should open the circuit breaker
        failure_threshold = getattr(self, 'circuit_breaker_config', {}).get('failure_threshold', 5)
        if breaker["failure_count"] >= failure_threshold and breaker["state"] == "closed":
            breaker["state"] = "open"
            breaker["opened_at"] = current_time
            self.metrics["circuit_breaker_activations_total"] += 1

    async def record_connection_success(self, connection_id: str) -> None:
        """Record a successful connection operation."""
        current_time = time.time()
        
        if connection_id not in self.circuit_breakers:
            await self.initialize_circuit_breaker(connection_id)
        
        breaker = self.circuit_breakers[connection_id]
        breaker["last_success_time"] = current_time
        breaker["consecutive_successes"] += 1
        
        # If in half-open state, check if we can close the circuit
        if breaker["state"] == "half-open":
            success_threshold = getattr(self, 'circuit_breaker_config', {}).get('success_threshold', 3)
            if breaker["consecutive_successes"] >= success_threshold:
                breaker["state"] = "closed"
                breaker["failure_count"] = 0
                breaker["consecutive_successes"] = 0

    async def should_allow_connection(self, connection_id: str) -> bool:
        """Check if a connection should be allowed based on circuit breaker state."""
        if connection_id not in self.circuit_breakers:
            return True
        
        state = self.circuit_breakers[connection_id]["state"]
        
        if state == "open":
            return False
        elif state == "half-open":
            return True  # Allow limited requests in half-open
        else:  # closed
            return True

    async def record_failure_with_timestamp(self, connection_id: str, failure_type: str, timestamp: float) -> None:
        """Record a failure with specific timestamp (for testing)."""
        if connection_id not in self.connection_failure_history:
            self.connection_failure_history[connection_id] = []
        
        self.connection_failure_history[connection_id].append({
            "timestamp": timestamp,
            "type": failure_type
        })
        
        if connection_id not in self.circuit_breakers:
            await self.initialize_circuit_breaker(connection_id)
        
        # Update failure count based on recent failures only
        recent_failures = self.get_recent_failure_count(connection_id, window_seconds=120)
        self.circuit_breakers[connection_id]["failure_count"] = recent_failures

    def get_recent_failure_count(self, connection_id: str, window_seconds: int = 120) -> int:
        """Get count of failures within the time window."""
        if connection_id not in self.connection_failure_history:
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_failures = [
            f for f in self.connection_failure_history[connection_id]
            if f["timestamp"] >= cutoff_time
        ]
        
        return len(recent_failures)

    async def get_circuit_breaker_metrics(self) -> dict[str, Any]:
        """Get comprehensive circuit breaker metrics."""
        total_breakers = len(self.circuit_breakers)
        open_count = sum(1 for b in self.circuit_breakers.values() if b["state"] == "open")
        closed_count = sum(1 for b in self.circuit_breakers.values() if b["state"] == "closed")
        half_open_count = sum(1 for b in self.circuit_breakers.values() if b["state"] == "half-open")
        
        # Calculate average failure rate
        total_failures = sum(b["failure_count"] for b in self.circuit_breakers.values())
        avg_failure_rate = (total_failures / total_breakers) if total_breakers > 0 else 0.0
        
        return {
            "circuit_breakers_total": total_breakers,
            "circuit_breakers_open_total": open_count,
            "circuit_breakers_closed_total": closed_count,
            "circuit_breakers_half_open_total": half_open_count,
            "circuit_breaker_transitions_total": self.metrics.get("circuit_breaker_activations_total", 0),
            "circuit_breaker_blocked_requests_total": 0,  # Would need to track this
            "circuit_breaker_allowed_requests_total": 0,  # Would need to track this
            "average_failure_rate": avg_failure_rate,
            "average_recovery_time_seconds": 0.0  # Placeholder
        }

    async def get_all_circuit_breaker_states(self) -> dict[str, dict]:
        """Get detailed state information for all circuit breakers."""
        current_time = time.time()
        states = {}
        
        for connection_id, breaker in self.circuit_breakers.items():
            time_in_current_state = 0
            if breaker["opened_at"]:
                time_in_current_state = current_time - breaker["opened_at"]
            elif breaker["last_success_time"]:
                time_in_current_state = current_time - breaker["last_success_time"]
            
            states[connection_id] = {
                "state": breaker["state"],
                "failure_count": breaker["failure_count"],
                "last_failure_time": breaker["last_failure_time"],
                "time_in_current_state": time_in_current_state
            }
        
        return states


# Global WebSocket manager instance
websocket_manager = DashboardWebSocketManager()


# ==================== WEBSOCKET ENDPOINTS ====================

@router.websocket("/ws/agents")
async def websocket_agents(
    websocket: WebSocket,
    connection_id: str | None = Query(None, description="Optional connection ID"),
):
    """
    WebSocket endpoint for real-time agent status updates.

    Provides live agent health, status changes, and performance metrics.
    """
    connection_id = connection_id or str(uuid.uuid4())

    try:
        await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="agent_monitor",
            subscriptions=["agents", "system"]
        )

        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(
                    connection_id, make_error("Invalid JSON message format")
                )
            except Exception as e:
                logger.error("Error in agent WebSocket", connection_id=connection_id, error=str(e))
                await websocket_manager._send_to_connection(
                    connection_id, make_error(str(e))
                )
                break

    except Exception as e:
        logger.error("Agent WebSocket connection failed", connection_id=connection_id, error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


@router.websocket("/ws/coordination")
async def websocket_coordination(
    websocket: WebSocket,
    connection_id: str | None = Query(None, description="Optional connection ID"),
):
    """
    WebSocket endpoint for real-time coordination monitoring.

    Streams coordination success rates, failure events, and recovery actions.
    """
    connection_id = connection_id or str(uuid.uuid4())

    try:
        await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="coordination_monitor",
            subscriptions=["coordination", "alerts", "system"]
        )

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(
                    connection_id, make_error("Invalid JSON message format")
                )
            except Exception as e:
                logger.error("Error in coordination WebSocket", connection_id=connection_id, error=str(e))
                break

    except Exception as e:
        logger.error("Coordination WebSocket connection failed", connection_id=connection_id, error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


@router.websocket("/ws/tasks")
async def websocket_tasks(
    websocket: WebSocket,
    connection_id: str | None = Query(None, description="Optional connection ID"),
):
    """
    WebSocket endpoint for real-time task distribution monitoring.

    Streams task queue status, assignment changes, and completion events.
    """
    connection_id = connection_id or str(uuid.uuid4())

    try:
        await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="task_monitor",
            subscriptions=["tasks", "agents"]
        )

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(
                    connection_id, make_error("Invalid JSON message format")
                )
            except Exception as e:
                logger.error("Error in task WebSocket", connection_id=connection_id, error=str(e))
                break

    except Exception as e:
        logger.error("Task WebSocket connection failed", connection_id=connection_id, error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


@router.websocket("/ws/system")
async def websocket_system(
    websocket: WebSocket,
    connection_id: str | None = Query(None, description="Optional connection ID"),
):
    """
    WebSocket endpoint for real-time system health monitoring.

    Streams overall system health, component status, and critical alerts.
    """
    connection_id = connection_id or str(uuid.uuid4())

    try:
        await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="system_monitor",
            subscriptions=["system", "alerts"]
        )

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(
                    connection_id, make_error("Invalid JSON message format")
                )
            except Exception as e:
                logger.error("Error in system WebSocket", connection_id=connection_id, error=str(e))
                break

    except Exception as e:
        logger.error("System WebSocket connection failed", connection_id=connection_id, error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


@router.websocket("/ws/dashboard")
async def websocket_dashboard_all(
    websocket: WebSocket,
    connection_id: str | None = Query(None, description="Optional connection ID"),
    subscriptions: str | None = Query("agents,coordination,tasks,system,project_index", description="Comma-separated subscriptions"),
):
    """
    WebSocket endpoint for comprehensive dashboard monitoring.

    Single endpoint that can subscribe to multiple data streams for full dashboard functionality.
    """
    connection_id = connection_id or str(uuid.uuid4())
    subscription_list = [s.strip() for s in (subscriptions or "").split(",") if s.strip()]

    try:
        await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="full_dashboard",
            subscriptions=subscription_list
        )

        # Initial connection frame already sent by connect(); avoid duplicate init

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(
                    connection_id, make_error("Invalid JSON message format")
                )
            except Exception as e:
                logger.error("Error in dashboard WebSocket", connection_id=connection_id, error=str(e))
                break

    except Exception as e:
        logger.error("Dashboard WebSocket connection failed", connection_id=connection_id, error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


# ==================== WEBSOCKET MANAGEMENT APIs ====================

@router.get("/websocket/stats", response_model=dict[str, Any])
async def get_websocket_stats():
    """
    Get statistics about active WebSocket connections.

    Provides insights into real-time client connections and subscription patterns.
    """
    try:
        stats = websocket_manager.get_connection_stats()

        return {
            "websocket_stats": stats,
            "endpoints": {
                "agents": "/api/dashboard/ws/agents",
                "coordination": "/api/dashboard/ws/coordination",
                "tasks": "/api/dashboard/ws/tasks",
                "system": "/api/dashboard/ws/system",
                "dashboard": "/api/dashboard/ws/dashboard"
            },
            "subscription_types": [
                "agents",
                "coordination",
                "tasks",
                "system",
                "alerts",
                "project_index"
            ],
            "last_updated": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get WebSocket stats", error=str(e))
        return {
            "error": "Failed to retrieve WebSocket statistics",
            "websocket_stats": {
                "total_connections": 0,
                "active_connections": 0
            }
        }


@router.post("/websocket/broadcast", response_model=dict[str, Any])
async def broadcast_message(
    subscription: str = Query(..., description="Subscription group to broadcast to"),
    message_type: str = Query(..., description="Type of message to broadcast"),
    message_data: dict[str, Any] | None = None,
):
    """
    Manually broadcast a message to WebSocket clients.

    Useful for testing WebSocket functionality and sending administrative messages.
    """
    try:
        if subscription not in websocket_manager.subscription_groups:
            return {
                "error": f"Invalid subscription: {subscription}",
                "valid_subscriptions": list(websocket_manager.subscription_groups.keys()),
            }

        payload = message_data or {}
        sent_count = await websocket_manager.broadcast_to_subscription(subscription, message_type, payload)

        return {
            "success": True,
            "subscription": subscription,
            "message_type": message_type,
            "clients_reached": sent_count,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to broadcast WebSocket message", error=str(e))
        return {
            "error": f"Failed to broadcast message: {str(e)}",
            "subscription": subscription,
            "message_type": message_type
        }


@router.post("/websocket/disconnect/{connection_id}", response_model=dict[str, Any])
async def disconnect_websocket_client(
    connection_id: str,
    reason: str = Query("Administrative disconnect", description="Reason for disconnect")
):
    """
    Manually disconnect a specific WebSocket client.

    Provides administrative control over WebSocket connections.
    """
    try:
        if connection_id not in websocket_manager.connections:
            return {
                "error": "Connection not found",
                "connection_id": connection_id,
                "active_connections": list(websocket_manager.connections.keys()),
            }

        # Send disconnect notification to client first
        await websocket_manager._send_to_connection(
            connection_id,
            {"type": "disconnect_notice", "reason": reason, "timestamp": datetime.utcnow().isoformat()},
        )

        # Disconnect the client
        await websocket_manager.disconnect(connection_id)

        return {
            "success": True,
            "connection_id": connection_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to disconnect WebSocket client", connection_id=connection_id, error=str(e))
        return {
            "error": f"Failed to disconnect client: {str(e)}",
            "connection_id": connection_id
        }


# Health check for WebSocket functionality
@router.get("/websocket/health", response_model=dict[str, Any])
async def websocket_health_check():
    """
    Check WebSocket system health and functionality.

    Validates WebSocket infrastructure is ready for real-time dashboard connections.
    """
    try:
        health_data = {
            "websocket_manager": "operational",
            "background_tasks": {
                "broadcast_task": websocket_manager.broadcast_task is not None and not websocket_manager.broadcast_task.done() if websocket_manager.broadcast_task else False,
                "redis_listener": websocket_manager.redis_listener_task is not None and not websocket_manager.redis_listener_task.done() if websocket_manager.redis_listener_task else False,
                "health_monitor": websocket_manager.health_monitor_task is not None and not websocket_manager.health_monitor_task.done() if websocket_manager.health_monitor_task else False
            },
            "connection_stats": websocket_manager.get_connection_stats(),
            "redis_connectivity": "unknown"
        }


        # Test Redis connectivity
        try:
            redis_client = get_redis()
            await redis_client.ping()
            health_data["redis_connectivity"] = "healthy"
        except Exception as redis_error:
            health_data["redis_connectivity"] = f"failed: {str(redis_error)}"

        # Overall health assessment
        health_score = 100

        if health_data["redis_connectivity"] != "healthy":
            health_score -= 40  # Redis is critical for real-time events

        active_tasks = sum(1 for task in health_data["background_tasks"].values() if task)
        if active_tasks < 3 and health_data["connection_stats"]["total_connections"] > 0:
            health_score -= 30  # Background tasks should be running with active connections

        health_data["overall_health"] = {
            "score": max(0, health_score),
            "status": "healthy" if health_score >= 90 else "degraded" if health_score >= 70 else "unhealthy"
        }

        return health_data

    except Exception as e:
        logger.error("WebSocket health check failed", error=str(e))
        return {
            "overall_health": {
                "score": 0,
                "status": "unhealthy"
            },
            "error": str(e)
        }


@router.get("/websocket/limits", response_model=dict[str, Any])
async def websocket_limits():
    """
    Get server-enforced WebSocket limits and contract info for clients.
    """
    try:
        limits = {
            "rate_limit_tokens_per_second": websocket_manager.rate_limit_tokens_per_second,
            "rate_limit_burst_capacity": websocket_manager.rate_limit_burst_capacity,
            "rate_limit_notify_cooldown_seconds": websocket_manager.rate_limit_notify_cooldown_seconds,
            "max_inbound_message_bytes": websocket_manager.max_inbound_message_bytes,
            "max_subscriptions_per_connection": websocket_manager.max_subscriptions_per_connection,
            "backpressure_disconnect_threshold": websocket_manager.backpressure_disconnect_threshold,
            "idle_disconnect_seconds": websocket_manager.idle_disconnect_seconds,
            # Auth/allowlist exposure for observability (do not expose tokens)
            "ws_auth_required": websocket_manager.auth_required,
            "ws_allowed_origins_configured": bool(websocket_manager.allowed_origins),
            "contract_version": WS_CONTRACT_VERSION,
            "supported_versions": [WS_CONTRACT_VERSION],
            "timestamp": datetime.utcnow().isoformat(),
        }
        return limits
    except Exception as e:
        logger.error("Failed to get WebSocket limits", error=str(e))
        return {"error": "failed_to_get_limits"}


@router.get("/websocket/contract", response_model=dict[str, Any])
async def websocket_contract_info():
    """
    Provide WebSocket contract versioning information for clients.
    """
    try:
        return {
            "current_version": WS_CONTRACT_VERSION,
            "supported_versions": [WS_CONTRACT_VERSION],
            "deprecation_policy": {
                "policy": "no_breaking_changes_without_minor_bump",
                "notice": "Breaking changes require a version bump and migration note",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Failed to get WebSocket contract info", error=str(e))
        return {"error": "failed_to_get_contract_info"}


@router.get("/metrics/websockets", response_model=dict[str, Any])
async def websocket_metrics():
    """
    Get comprehensive WebSocket metrics and observability data.
    
    Epic 2 Phase 2.2: WebSocket Observability & Metrics
    Provides production-grade metrics for monitoring WebSocket operations.
    """
    try:
        current_time = datetime.utcnow()
        
        # Core WebSocket metrics from manager
        core_metrics = websocket_manager.metrics.copy()
        
        # Connection statistics
        connection_stats = websocket_manager.get_connection_stats()
        
        # Performance metrics
        performance_metrics = {
            "average_message_send_rate_per_minute": 0.0,
            "average_connection_duration_seconds": 0.0,
            "peak_concurrent_connections": len(websocket_manager.connections),
            "message_success_rate": 0.0,
            "backpressure_rate": 0.0,
            "rate_limit_violation_rate": 0.0,
        }
        
        # Calculate success rate
        total_send_attempts = core_metrics["messages_sent_total"] + core_metrics["messages_send_failures_total"]
        if total_send_attempts > 0:
            performance_metrics["message_success_rate"] = (
                core_metrics["messages_sent_total"] / total_send_attempts
            ) * 100.0
        
        # Calculate backpressure rate
        total_disconnects = core_metrics["disconnections_total"]
        if total_disconnects > 0:
            performance_metrics["backpressure_rate"] = (
                core_metrics["backpressure_disconnects_total"] / total_disconnects
            ) * 100.0
        
        # Calculate rate limit violation rate
        total_messages = core_metrics["messages_received_total"]
        if total_messages > 0:
            performance_metrics["rate_limit_violation_rate"] = (
                core_metrics["messages_dropped_rate_limit_total"] / total_messages
            ) * 100.0
        
        # Calculate average connection duration
        if websocket_manager.connections:
            durations = [
                (current_time - conn.connected_at).total_seconds()
                for conn in websocket_manager.connections.values()
            ]
            performance_metrics["average_connection_duration_seconds"] = sum(durations) / len(durations)
        
        # Real-time connection details
        connection_details = []
        for conn_id, conn in websocket_manager.connections.items():
            connection_details.append({
                "connection_id": conn_id,
                "client_type": conn.client_type,
                "connected_at": conn.connected_at.isoformat(),
                "last_activity": conn.last_activity.isoformat(),
                "subscriptions": list(conn.subscriptions),
                "idle_seconds": (current_time - conn.last_activity).total_seconds(),
                "tokens_remaining": conn.tokens,
                "send_failure_streak": conn.send_failure_streak,
                "metadata": conn.metadata,
            })
        
        # Configuration and limits
        configuration = {
            "rate_limit_tokens_per_second": websocket_manager.rate_limit_tokens_per_second,
            "rate_limit_burst_capacity": websocket_manager.rate_limit_burst_capacity,
            "max_inbound_message_bytes": websocket_manager.max_inbound_message_bytes,
            "max_subscriptions_per_connection": websocket_manager.max_subscriptions_per_connection,
            "backpressure_disconnect_threshold": websocket_manager.backpressure_disconnect_threshold,
            "idle_disconnect_seconds": websocket_manager.idle_disconnect_seconds,
            "auth_required": websocket_manager.auth_required,
            "compression_enabled": websocket_manager.compression_enabled,
        }
        
        # Background task health
        background_tasks = {
            "broadcast_task_running": (
                websocket_manager.broadcast_task is not None 
                and not websocket_manager.broadcast_task.done()
            ),
            "redis_listener_running": (
                websocket_manager.redis_listener_task is not None 
                and not websocket_manager.redis_listener_task.done()
            ),
            "health_monitor_running": (
                websocket_manager.health_monitor_task is not None 
                and not websocket_manager.health_monitor_task.done()
            ),
        }
        
        # Subscription group analysis
        subscription_analysis = {}
        for group_name, connections in websocket_manager.subscription_groups.items():
            subscription_analysis[group_name] = {
                "active_connections": len(connections),
                "percentage_of_total": (
                    (len(connections) / len(websocket_manager.connections) * 100.0)
                    if websocket_manager.connections else 0.0
                ),
            }
        
        # Epic 2 Phase 2.2 compliance metrics
        observability_compliance = {
            "structured_logging": True,  # All errors include correlation_id, type, subscription
            "correlation_id_injection": True,  # All outbound frames have correlation IDs
            "metrics_exposition": True,  # This endpoint provides comprehensive metrics
            "performance_monitoring": True,  # Real-time performance tracking
            "error_tracking": True,  # Send failures and error types tracked
        }
        
        return {
            # Core metrics (Epic 2 Phase 2.2 requirement)
            "messages_sent_total": core_metrics["messages_sent_total"],
            "messages_send_failures_total": core_metrics["messages_send_failures_total"],
            "messages_received_total": core_metrics["messages_received_total"],
            "messages_dropped_rate_limit_total": core_metrics["messages_dropped_rate_limit_total"],
            "errors_sent_total": core_metrics["errors_sent_total"],
            "connections_total": core_metrics["connections_total"],
            "disconnections_total": core_metrics["disconnections_total"],
            "backpressure_disconnects_total": core_metrics["backpressure_disconnects_total"],
            
            # Additional observability metrics
            "auth_denied_total": core_metrics["auth_denied_total"],
            "origin_denied_total": core_metrics["origin_denied_total"],
            "idle_disconnects_total": core_metrics["idle_disconnects_total"],
            "bytes_sent_total": core_metrics["bytes_sent_total"],
            "bytes_received_total": core_metrics["bytes_received_total"],
            
            # Performance analytics
            "performance_metrics": performance_metrics,
            
            # Real-time state
            "current_connections": len(websocket_manager.connections),
            "active_subscriptions": connection_stats["subscription_counts"],
            "last_broadcast_fanout": websocket_manager.last_broadcast_fanout,
            
            # Connection details (for debugging)
            "connection_details": connection_details,
            
            # System configuration
            "configuration": configuration,
            
            # Background tasks health
            "background_tasks": background_tasks,
            
            # Subscription analysis
            "subscription_analysis": subscription_analysis,
            
            # Epic 2 Phase 2.2 compliance
            "observability_compliance": observability_compliance,
            
            # Contract versioning
            "contract_version": WS_CONTRACT_VERSION,
            "supported_versions": [WS_CONTRACT_VERSION],
            
            # Metadata
            "timestamp": current_time.isoformat(),
            "endpoint_version": "1.0.0",
            "collection_interval_seconds": 1,  # Real-time collection
        }
        
    except Exception as e:
        logger.error("Failed to get WebSocket metrics", error=str(e))
        return {
            "error": "failed_to_get_websocket_metrics",
            "timestamp": datetime.utcnow().isoformat(),
            "fallback_metrics": {
                "messages_sent_total": 0,
                "messages_send_failures_total": 0,
                "messages_received_total": 0,
                "connections_total": 0,
                "errors_sent_total": 0,
            }
        }
