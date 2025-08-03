"""
Enterprise Tmux Session Manager for LeanVibe Agent Hive 2.0

Provides enterprise-grade tmux session management with automatic recovery,
process monitoring, and fault tolerance patterns for 99.9% uptime.

Based on strategic analysis from Gemini CLI and Tmux-Orchestrator patterns.
"""

import asyncio
import json
import os
import time
import signal
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil

import structlog
import libtmux
from redis.asyncio import Redis

logger = structlog.get_logger()


class ServiceStatus(Enum):
    """Service health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    RESTARTING = "restarting"


class RecoveryStrategy(Enum):
    """Service recovery strategies."""
    RESTART_SERVICE = "restart_service"
    RESTART_WINDOW = "restart_window"
    RESTART_SESSION = "restart_session"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class ServiceConfig:
    """Configuration for a tmux service."""
    name: str
    command: str
    health_check_url: Optional[str] = None
    health_check_command: Optional[str] = None
    restart_policy: str = "always"
    max_restarts: int = 5
    restart_interval: int = 30  # seconds
    timeout: int = 60  # seconds
    dependencies: List[str] = None
    environment: Dict[str, str] = None
    working_directory: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.environment is None:
            self.environment = {}


@dataclass
class ServiceHealth:
    """Service health status and metrics."""
    name: str
    status: ServiceStatus
    last_check: datetime
    response_time_ms: float
    error_count: int
    restart_count: int
    uptime_seconds: float
    last_error: Optional[str] = None
    recovery_strategy: Optional[RecoveryStrategy] = None


class CircuitBreaker:
    """Circuit breaker for service fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def record_failure(self):
        """Record a service failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "closed"
        
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout_seconds:
                self.state = "half-open"
                return True
            return False
            
        # half-open state
        return True


class ProcessMonitor:
    """Monitor and manage tmux service processes."""
    
    def __init__(self, service_config: ServiceConfig):
        self.config = service_config
        self.process = None
        self.start_time = None
        self.restart_count = 0
        self.last_restart = None
        self.circuit_breaker = CircuitBreaker()
        
    async def start(self) -> bool:
        """Start the service process."""
        try:
            if self.is_running():
                logger.info(f"Service {self.config.name} already running")
                return True
                
            logger.info(f"Starting service {self.config.name}")
            
            # Set working directory
            cwd = self.config.working_directory or Path.cwd()
            
            # Set environment
            env = dict(os.environ)
            env.update(self.config.environment)
            
            # Start process in background
            self.process = await asyncio.create_subprocess_shell(
                self.config.command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.start_time = datetime.utcnow()
            self.circuit_breaker.record_success()
            
            logger.info(f"Service {self.config.name} started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start service {self.config.name}: {e}")
            self.circuit_breaker.record_failure()
            return False
    
    async def stop(self) -> bool:
        """Stop the service process gracefully."""
        try:
            if not self.is_running():
                return True
                
            logger.info(f"Stopping service {self.config.name}")
            
            # Send SIGTERM first
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                # Force kill if graceful shutdown fails
                self.process.kill()
                await self.process.wait()
            
            self.process = None
            self.start_time = None
            
            logger.info(f"Service {self.config.name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop service {self.config.name}: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if the service process is running."""
        if self.process is None:
            return False
        return self.process.returncode is None
    
    async def health_check(self) -> ServiceHealth:
        """Perform health check on the service."""
        start_time = time.time()
        
        try:
            if not self.is_running():
                return ServiceHealth(
                    name=self.config.name,
                    status=ServiceStatus.STOPPED,
                    last_check=datetime.utcnow(),
                    response_time_ms=0,
                    error_count=self.circuit_breaker.failure_count,
                    restart_count=self.restart_count,
                    uptime_seconds=0
                )
            
            # Check via health check URL if provided
            if self.config.health_check_url:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.config.health_check_url, timeout=5) as response:
                        healthy = response.status == 200
            
            # Check via health check command if provided
            elif self.config.health_check_command:
                proc = await asyncio.create_subprocess_shell(
                    self.config.health_check_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.wait()
                healthy = proc.returncode == 0
            
            # Default: assume healthy if process is running
            else:
                healthy = True
            
            response_time = (time.time() - start_time) * 1000
            uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
            
            status = ServiceStatus.HEALTHY if healthy else ServiceStatus.UNHEALTHY
            
            if healthy:
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()
            
            return ServiceHealth(
                name=self.config.name,
                status=status,
                last_check=datetime.utcnow(),
                response_time_ms=response_time,
                error_count=self.circuit_breaker.failure_count,
                restart_count=self.restart_count,
                uptime_seconds=uptime
            )
            
        except Exception as e:
            logger.error(f"Health check failed for {self.config.name}: {e}")
            self.circuit_breaker.record_failure()
            
            return ServiceHealth(
                name=self.config.name,
                status=ServiceStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_count=self.circuit_breaker.failure_count,
                restart_count=self.restart_count,
                uptime_seconds=0,
                last_error=str(e)
            )
    
    async def restart(self) -> bool:
        """Restart the service with circuit breaker protection."""
        if not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {self.config.name}, skipping restart")
            return False
        
        # Check restart limits
        if self.restart_count >= self.config.max_restarts:
            logger.error(f"Service {self.config.name} exceeded max restarts ({self.config.max_restarts})")
            return False
        
        # Check restart interval
        if self.last_restart and (datetime.utcnow() - self.last_restart).total_seconds() < self.config.restart_interval:
            logger.warning(f"Service {self.config.name} restart too soon, waiting")
            return False
        
        logger.info(f"Restarting service {self.config.name}")
        
        await self.stop()
        await asyncio.sleep(2)  # Brief pause before restart
        
        success = await self.start()
        if success:
            self.restart_count += 1
            self.last_restart = datetime.utcnow()
        
        return success


class EnterpriseTmuxManager:
    """
    Enterprise-grade tmux session manager with fault tolerance.
    
    Features:
    - Automatic service recovery
    - Circuit breaker protection
    - Health monitoring
    - Inter-service communication via Redis
    - Graceful degradation
    """
    
    def __init__(self, session_name: str = "leanvibe-enterprise", redis_url: str = "redis://localhost:6379"):
        self.session_name = session_name
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        self.tmux_server = libtmux.Server()
        self.session: Optional[libtmux.Session] = None
        self.services: Dict[str, ProcessMonitor] = {}
        self.health_checks_running = False
        self.is_running = False
        
    async def initialize(self):
        """Initialize the enterprise tmux manager."""
        try:
            # Connect to Redis for inter-service communication
            self.redis = Redis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis for tmux coordination")
            
            # Get or create tmux session
            try:
                self.session = self.tmux_server.sessions.get(session_name=self.session_name)
                logger.info(f"Attached to existing tmux session: {self.session_name}")
            except:
                self.session = self.tmux_server.new_session(session_name=self.session_name)
                logger.info(f"Created new tmux session: {self.session_name}")
            
            self.is_running = True
            logger.info("Enterprise tmux manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize tmux manager: {e}")
            raise
    
    async def add_service(self, config: ServiceConfig) -> bool:
        """Add a service to the tmux session."""
        try:
            if config.name in self.services:
                logger.warning(f"Service {config.name} already exists")
                return False
            
            # Create tmux window for the service
            window = self.session.new_window(window_name=config.name)
            
            # Create process monitor
            monitor = ProcessMonitor(config)
            self.services[config.name] = monitor
            
            logger.info(f"Added service {config.name} to tmux session")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add service {config.name}: {e}")
            return False
    
    async def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        monitor = self.services[service_name]
        success = await monitor.start()
        
        if success:
            # Send command to tmux window
            window = self.session.windows.get(window_name=service_name)
            window.panes[0].send_keys(monitor.config.command, enter=True)
        
        return success
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        monitor = self.services[service_name]
        return await monitor.stop()
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        monitor = self.services[service_name]
        return await monitor.restart()
    
    async def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status of a specific service."""
        if service_name not in self.services:
            return None
        
        monitor = self.services[service_name]
        return await monitor.health_check()
    
    async def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all services."""
        health_status = {}
        
        for service_name, monitor in self.services.items():
            health_status[service_name] = await monitor.health_check()
        
        return health_status
    
    async def start_health_monitoring(self, check_interval: int = 30):
        """Start continuous health monitoring with automatic recovery."""
        if self.health_checks_running:
            logger.warning("Health monitoring already running")
            return
        
        self.health_checks_running = True
        logger.info(f"Starting health monitoring (interval: {check_interval}s)")
        
        while self.health_checks_running and self.is_running:
            try:
                health_status = await self.get_all_service_health()
                
                for service_name, health in health_status.items():
                    # Publish health status to Redis
                    await self.publish_health_status(service_name, health)
                    
                    # Auto-recovery logic
                    if health.status == ServiceStatus.UNHEALTHY:
                        logger.warning(f"Service {service_name} is unhealthy, attempting restart")
                        await self.restart_service(service_name)
                    
                    elif health.status == ServiceStatus.STOPPED:
                        logger.warning(f"Service {service_name} is stopped, attempting start")
                        await self.start_service(service_name)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(check_interval)
    
    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        self.health_checks_running = False
        logger.info("Health monitoring stopped")
    
    async def publish_health_status(self, service_name: str, health: ServiceHealth):
        """Publish service health status to Redis."""
        try:
            if self.redis:
                await self.redis.publish(
                    f"tmux:health:{service_name}",
                    json.dumps(asdict(health), default=str)
                )
        except Exception as e:
            logger.error(f"Failed to publish health status for {service_name}: {e}")
    
    async def send_inter_service_message(self, from_service: str, to_service: str, message: dict):
        """Send message between services via Redis."""
        try:
            if self.redis:
                await self.redis.publish(
                    f"tmux:message:{to_service}",
                    json.dumps({
                        "from": from_service,
                        "to": to_service,
                        "message": message,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                )
        except Exception as e:
            logger.error(f"Failed to send message from {from_service} to {to_service}: {e}")
    
    async def graceful_shutdown(self):
        """Gracefully shutdown all services and tmux session."""
        logger.info("Starting graceful shutdown of tmux session")
        
        # Stop health monitoring
        await self.stop_health_monitoring()
        
        # Stop all services
        for service_name in self.services:
            await self.stop_service(service_name)
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        self.is_running = False
        logger.info("Graceful shutdown completed")


# Factory function for creating pre-configured enterprise services
def create_enterprise_services() -> List[ServiceConfig]:
    """Create standard enterprise service configurations."""
    return [
        ServiceConfig(
            name="api-server",
            command="uvicorn app.main:app --reload --host 0.0.0.0 --port 8000",
            health_check_url="http://localhost:8000/health",
            max_restarts=3,
            restart_interval=30,
            dependencies=["infrastructure"]
        ),
        ServiceConfig(
            name="observability",
            command="python -m app.core.enterprise_observability",
            health_check_url="http://localhost:8001/metrics",
            max_restarts=5,
            restart_interval=15
        ),
        ServiceConfig(
            name="infrastructure",
            command="docker compose up postgres redis",
            health_check_command="docker compose ps | grep -E '(postgres|redis)' | grep -v Exit",
            max_restarts=2,
            restart_interval=60
        ),
        ServiceConfig(
            name="agent-pool",
            command="python -m app.agents.pool_manager",
            health_check_command="curl -s http://localhost:8000/api/v1/agents | jq '.status'",
            max_restarts=3,
            restart_interval=30,
            dependencies=["api-server"]
        ),
        ServiceConfig(
            name="monitoring",
            command="python -m app.monitoring.health_dashboard",
            health_check_url="http://localhost:8002/health",
            max_restarts=3,
            restart_interval=30
        )
    ]


# Example usage
async def main():
    """Example usage of the enterprise tmux manager."""
    manager = EnterpriseTmuxManager()
    
    try:
        await manager.initialize()
        
        # Add enterprise services
        services = create_enterprise_services()
        for service_config in services:
            await manager.add_service(service_config)
        
        # Start services with dependency order
        dependency_order = ["infrastructure", "api-server", "observability", "agent-pool", "monitoring"]
        for service_name in dependency_order:
            success = await manager.start_service(service_name)
            if success:
                logger.info(f"Started {service_name}")
                await asyncio.sleep(5)  # Allow service to stabilize
            else:
                logger.error(f"Failed to start {service_name}")
        
        # Start health monitoring
        await manager.start_health_monitoring()
        
        # Keep running
        logger.info("Enterprise tmux session operational")
        await asyncio.Event().wait()  # Run forever
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await manager.graceful_shutdown()


if __name__ == "__main__":
    asyncio.run(main())