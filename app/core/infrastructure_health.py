"""
Infrastructure Health Check System for Epic 6 Phase 2

Provides comprehensive health monitoring for database, Redis, and other infrastructure components.
Designed for production readiness with detailed diagnostics and monitoring capabilities.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from sqlalchemy import text

from app.core.database import get_session, DatabaseHealthCheck
from app.core.config import settings
from app.core.logging_service import get_component_logger

logger = get_component_logger("infrastructure_health")


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"


@dataclass
class HealthCheckResult:
    """Structured health check result."""
    service: str
    status: HealthStatus
    response_time_ms: float
    message: str
    details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        return result


class InfrastructureHealthChecker:
    """Comprehensive infrastructure health monitoring system."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._health_history: List[Dict[str, HealthCheckResult]] = []
        self._max_history = 100
        
    async def initialize(self) -> None:
        """Initialize health checker connections."""
        try:
            # Initialize Redis connection for health checks
            self.redis_client = redis.from_url(settings.REDIS_URL)
            logger.info("Infrastructure health checker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize health checker: {e}")
            
    async def check_database_health(self) -> HealthCheckResult:
        """Comprehensive database health check."""
        start_time = time.time()
        service = "postgresql"
        
        try:
            # Basic connectivity check
            is_connected = await DatabaseHealthCheck.check_connection()
            if not is_connected:
                return HealthCheckResult(
                    service=service,
                    status=HealthStatus.DOWN,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message="Database connection failed",
                    details={"error": "Connection refused or timeout"},
                    timestamp=time.time()
                )
            
            # Connection pool statistics
            pool_stats = await DatabaseHealthCheck.get_connection_stats()
            
            # Extension availability check
            extensions = await DatabaseHealthCheck.check_extensions()
            
            # Performance test - simple query
            perf_start = time.time()
            async with get_session() as session:
                await session.execute(text("SELECT COUNT(*) FROM pg_stat_activity"))
            perf_time = (time.time() - perf_start) * 1000
            
            # Determine health status based on metrics
            status = HealthStatus.HEALTHY
            warnings = []
            
            # Check pool utilization
            if isinstance(pool_stats.get('checked_out'), int) and isinstance(pool_stats.get('pool_size'), int):
                pool_utilization = pool_stats['checked_out'] / pool_stats['pool_size']
                if pool_utilization > 0.8:
                    status = HealthStatus.WARNING
                    warnings.append("High connection pool utilization")
                elif pool_utilization > 0.95:
                    status = HealthStatus.CRITICAL
                    warnings.append("Critical connection pool utilization")
            
            # Check query performance
            if perf_time > 1000:  # 1 second
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.WARNING
                warnings.append("Slow query performance")
            
            # Check extensions
            if not extensions.get('pgvector', False):
                status = HealthStatus.WARNING
                warnings.append("pgvector extension not available")
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service=service,
                status=status,
                response_time_ms=response_time,
                message=f"Database healthy{' with warnings' if warnings else ''}",
                details={
                    "pool_stats": pool_stats,
                    "extensions": extensions,
                    "query_performance_ms": perf_time,
                    "warnings": warnings
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Database health check failed: {e}")
            
            return HealthCheckResult(
                service=service,
                status=HealthStatus.DOWN,
                response_time_ms=response_time,
                message=f"Database health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                timestamp=time.time()
            )
    
    async def check_redis_health(self) -> HealthCheckResult:
        """Comprehensive Redis health check."""
        start_time = time.time()
        service = "redis"
        
        try:
            if not self.redis_client:
                await self.initialize()
            
            if not self.redis_client:
                raise Exception("Redis client not initialized")
            
            # Basic connectivity and latency check
            ping_start = time.time()
            pong = await self.redis_client.ping()
            ping_time = (time.time() - ping_start) * 1000
            
            if not pong:
                raise Exception("Redis ping failed")
            
            # Memory and performance stats
            info = await self.redis_client.info()
            
            # Test basic operations
            ops_start = time.time()
            test_key = "health_check_test"
            await self.redis_client.set(test_key, "test_value", ex=60)
            value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            ops_time = (time.time() - ops_start) * 1000
            
            if value.decode() != "test_value":
                raise Exception("Redis basic operations test failed")
            
            # Test pub/sub functionality
            pubsub_start = time.time()
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("health_check_channel")
            await self.redis_client.publish("health_check_channel", "health_test")
            
            # Try to get the subscription confirmation and test message
            try:
                message = await asyncio.wait_for(pubsub.get_message(), timeout=1.0)
                if message and message['type'] == 'subscribe':
                    message = await asyncio.wait_for(pubsub.get_message(), timeout=1.0)
            except asyncio.TimeoutError:
                message = None
                
            await pubsub.unsubscribe("health_check_channel")
            await pubsub.aclose()
            pubsub_time = (time.time() - pubsub_start) * 1000
            
            # Determine health status
            status = HealthStatus.HEALTHY
            warnings = []
            
            # Check memory usage
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            if max_memory > 0:
                memory_usage = used_memory / max_memory
                if memory_usage > 0.8:
                    status = HealthStatus.WARNING
                    warnings.append("High memory usage")
                elif memory_usage > 0.95:
                    status = HealthStatus.CRITICAL
                    warnings.append("Critical memory usage")
            
            # Check latency
            if ping_time > 100:  # 100ms
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.WARNING
                warnings.append("High Redis latency")
            
            # Check pub/sub functionality
            pubsub_working = message and message.get('type') == 'message'
            if not pubsub_working:
                warnings.append("Pub/sub functionality may be impaired")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service=service,
                status=status,
                response_time_ms=response_time,
                message=f"Redis healthy{' with warnings' if warnings else ''}",
                details={
                    "ping_time_ms": ping_time,
                    "operations_time_ms": ops_time,
                    "pubsub_time_ms": pubsub_time,
                    "pubsub_working": pubsub_working,
                    "memory_info": {
                        "used_memory": used_memory,
                        "max_memory": max_memory,
                        "memory_usage_pct": (used_memory / max_memory * 100) if max_memory > 0 else 0
                    },
                    "connected_clients": info.get('connected_clients', 0),
                    "total_commands_processed": info.get('total_commands_processed', 0),
                    "warnings": warnings
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Redis health check failed: {e}")
            
            return HealthCheckResult(
                service=service,
                status=HealthStatus.DOWN,
                response_time_ms=response_time,
                message=f"Redis health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                timestamp=time.time()
            )
    
    async def check_docker_services_health(self) -> HealthCheckResult:
        """Check Docker services health (if running in Docker)."""
        start_time = time.time()
        service = "docker_services"
        
        try:
            import subprocess
            import json
            
            # Check if we're running in Docker
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return HealthCheckResult(
                    service=service,
                    status=HealthStatus.WARNING,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message="Docker Compose not available or not running",
                    details={"error": result.stderr.strip() if result.stderr else "Docker compose command failed"},
                    timestamp=time.time()
                )
            
            # Parse Docker Compose status
            services = []
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    try:
                        service_info = json.loads(line)
                        services.append(service_info)
                    except json.JSONDecodeError:
                        continue
            
            # Analyze service health
            healthy_services = []
            unhealthy_services = []
            
            for svc in services:
                service_name = svc.get('Service', 'unknown')
                state = svc.get('State', 'unknown')
                health = svc.get('Health', 'unknown')
                
                if state == 'running' and health in ['healthy', '']:
                    healthy_services.append(service_name)
                else:
                    unhealthy_services.append({
                        'name': service_name,
                        'state': state,
                        'health': health
                    })
            
            # Determine overall status
            if unhealthy_services:
                status = HealthStatus.CRITICAL if len(unhealthy_services) > len(healthy_services) else HealthStatus.WARNING
                message = f"Some services unhealthy: {len(unhealthy_services)} of {len(services)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {len(services)} Docker services healthy"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service=service,
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    "total_services": len(services),
                    "healthy_services": healthy_services,
                    "unhealthy_services": unhealthy_services,
                    "services_detail": services
                },
                timestamp=time.time()
            )
            
        except subprocess.TimeoutExpired:
            return HealthCheckResult(
                service=service,
                status=HealthStatus.WARNING,
                response_time_ms=(time.time() - start_time) * 1000,
                message="Docker health check timed out",
                details={"error": "Command timeout"},
                timestamp=time.time()
            )
        except Exception as e:
            return HealthCheckResult(
                service=service,
                status=HealthStatus.WARNING,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Docker health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                timestamp=time.time()
            )
    
    async def perform_comprehensive_health_check(self) -> Dict[str, HealthCheckResult]:
        """Perform comprehensive health check of all infrastructure components."""
        logger.info("Starting comprehensive infrastructure health check")
        
        # Run all health checks concurrently
        tasks = [
            self.check_database_health(),
            self.check_redis_health(),
            self.check_docker_services_health()
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Health check gather failed: {e}")
            results = [e] * len(tasks)
        
        # Process results
        health_report = {}
        service_names = ["database", "redis", "docker_services"]
        
        for i, result in enumerate(results):
            service_name = service_names[i]
            if isinstance(result, Exception):
                health_report[service_name] = HealthCheckResult(
                    service=service_name,
                    status=HealthStatus.DOWN,
                    response_time_ms=0,
                    message=f"Health check failed: {str(result)}",
                    details={"error": str(result), "error_type": type(result).__name__},
                    timestamp=time.time()
                )
            else:
                health_report[service_name] = result
        
        # Store in history
        self._health_history.append(health_report)
        if len(self._health_history) > self._max_history:
            self._health_history.pop(0)
        
        # Log summary
        status_summary = {status.value: 0 for status in HealthStatus}
        for service_result in health_report.values():
            status_summary[service_result.status.value] += 1
        
        logger.info(f"Health check completed: {status_summary}")
        
        return health_report
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary with recommendations."""
        health_report = await self.perform_comprehensive_health_check()
        
        # Calculate overall system status
        critical_count = sum(1 for result in health_report.values() if result.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for result in health_report.values() if result.status == HealthStatus.WARNING)
        down_count = sum(1 for result in health_report.values() if result.status == HealthStatus.DOWN)
        
        if down_count > 0 or critical_count > 1:
            overall_status = HealthStatus.CRITICAL
        elif critical_count > 0 or warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Generate recommendations
        recommendations = []
        for service_name, result in health_report.items():
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.DOWN]:
                recommendations.append(f"{service_name}: {result.message}")
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "services": {name: result.to_dict() for name, result in health_report.items()},
            "summary": {
                "total_services": len(health_report),
                "healthy": sum(1 for r in health_report.values() if r.status == HealthStatus.HEALTHY),
                "warning": warning_count,
                "critical": critical_count,
                "down": down_count
            },
            "recommendations": recommendations,
            "history_available": len(self._health_history)
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.aclose()
            self.redis_client = None


# Global health checker instance
health_checker = InfrastructureHealthChecker()


async def get_infrastructure_health() -> Dict[str, Any]:
    """Get current infrastructure health status."""
    return await health_checker.get_health_summary()


async def initialize_health_monitoring() -> None:
    """Initialize health monitoring system."""
    await health_checker.initialize()