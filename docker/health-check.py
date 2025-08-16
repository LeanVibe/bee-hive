#!/usr/bin/env python3
"""
Comprehensive Health Check Script for Project Index Universal Installer
Validates service health, connectivity, and operational status
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse

import aiohttp
import asyncpg
import redis.asyncio as redis
import psutil


class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    component: str
    status: HealthStatus
    response_time_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None
    timestamp: float = 0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class HealthChecker:
    """Comprehensive health checker for Project Index components"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.results: List[HealthCheckResult] = []

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    async def check_api_service(self) -> HealthCheckResult:
        """Check Project Index API service health"""
        start_time = time.time()
        component = "project-index-api"
        
        try:
            api_url = self.config.get("api_url", "http://localhost:8100")
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Check health endpoint
                async with session.get(f"{api_url}/health") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Analyze health response
                        status = HealthStatus.HEALTHY
                        if health_data.get("status") == "degraded":
                            status = HealthStatus.DEGRADED
                        elif health_data.get("status") == "unhealthy":
                            status = HealthStatus.UNHEALTHY
                            
                        return HealthCheckResult(
                            component=component,
                            status=status,
                            response_time_ms=response_time,
                            details={
                                "version": health_data.get("version", "unknown"),
                                "components": health_data.get("components", {}),
                                "summary": health_data.get("summary", {}),
                                "url": api_url
                            }
                        )
                    else:
                        return HealthCheckResult(
                            component=component,
                            status=HealthStatus.UNHEALTHY,
                            response_time_ms=response_time,
                            details={"status_code": response.status},
                            error=f"HTTP {response.status}"
                        )
                        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"API health check failed: {str(e)}")
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                details={"error_type": type(e).__name__},
                error=str(e)
            )

    async def check_database(self) -> HealthCheckResult:
        """Check PostgreSQL database connectivity and health"""
        start_time = time.time()
        component = "postgresql"
        
        try:
            # Build connection string
            db_config = {
                "host": self.config.get("db_host", "localhost"),
                "port": self.config.get("db_port", 5433),
                "user": self.config.get("db_user", "project_user"),
                "password": self.config.get("db_password", ""),
                "database": self.config.get("db_name", "project_index")
            }
            
            # Test connection
            conn = await asyncpg.connect(**db_config)
            
            # Run health queries
            version = await conn.fetchval("SELECT version()")
            table_count = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            
            # Check extensions
            extensions = await conn.fetch("""
                SELECT extname FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp')
            """)
            
            await conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=component,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                details={
                    "version": version.split()[1] if version else "unknown",
                    "table_count": table_count,
                    "extensions": [ext["extname"] for ext in extensions],
                    "host": f"{db_config['host']}:{db_config['port']}"
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"Database health check failed: {str(e)}")
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                details={"error_type": type(e).__name__},
                error=str(e)
            )

    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity and health"""
        start_time = time.time()
        component = "redis"
        
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6380")
            
            # Parse Redis URL
            redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test basic operations
            await redis_client.ping()
            info = await redis_client.info()
            
            # Test key operations
            test_key = "health_check_test"
            await redis_client.set(test_key, "test_value", ex=60)
            test_value = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            
            await redis_client.close()
            
            response_time = (time.time() - start_time) * 1000
            
            # Check memory usage
            used_memory = info.get("used_memory_human", "unknown")
            max_memory = info.get("maxmemory_human", "unlimited")
            
            return HealthCheckResult(
                component=component,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                details={
                    "version": info.get("redis_version", "unknown"),
                    "memory_used": used_memory,
                    "memory_max": max_memory,
                    "connected_clients": info.get("connected_clients", 0),
                    "test_operations": test_value == "test_value"
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"Redis health check failed: {str(e)}")
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                details={"error_type": type(e).__name__},
                error=str(e)
            )

    async def check_file_monitor(self) -> HealthCheckResult:
        """Check file monitor service health"""
        start_time = time.time()
        component = "file-monitor"
        
        try:
            # Check if workspace directory exists and is accessible
            workspace_path = Path(self.config.get("workspace_path", "/workspace"))
            
            if not workspace_path.exists():
                return HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    details={"workspace_path": str(workspace_path)},
                    error="Workspace directory does not exist"
                )
            
            # Check directory accessibility
            if not os.access(workspace_path, os.R_OK):
                return HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    details={"workspace_path": str(workspace_path)},
                    error="Workspace directory not readable"
                )
            
            # Count files in workspace
            file_count = sum(1 for _ in workspace_path.rglob("*") if _.is_file())
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=component,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                details={
                    "workspace_path": str(workspace_path),
                    "file_count": file_count,
                    "accessible": True
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"File monitor health check failed: {str(e)}")
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                details={"error_type": type(e).__name__},
                error=str(e)
            )

    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        start_time = time.time()
        component = "system-resources"
        
        try:
            # Get system metrics
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            warnings = []
            
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                warnings.append("High memory usage")
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                warnings.append("Elevated memory usage")
            
            if disk.percent > 95:
                status = HealthStatus.UNHEALTHY
                warnings.append("Disk space critical")
            elif disk.percent > 85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                warnings.append("Disk space low")
            
            if cpu_percent > 95:
                status = HealthStatus.UNHEALTHY
                warnings.append("High CPU usage")
            elif cpu_percent > 80:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                warnings.append("Elevated CPU usage")
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=component,
                status=status,
                response_time_ms=response_time,
                details={
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "cpu_percent": cpu_percent,
                    "warnings": warnings
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"System resources health check failed: {str(e)}")
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                details={"error_type": type(e).__name__},
                error=str(e)
            )

    async def run_all_checks(self) -> Tuple[Dict[str, Any], int]:
        """Run all health checks and return summary"""
        self.logger.info("Starting comprehensive health check...")
        
        # Run all health checks concurrently
        checks = [
            self.check_api_service(),
            self.check_database(),
            self.check_redis(),
            self.check_file_monitor(),
        ]
        
        # Add system resource check (synchronous)
        system_check = self.check_system_resources()
        
        # Execute async checks
        async_results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Collect all results
        self.results = []
        for result in async_results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check exception: {result}")
                self.results.append(HealthCheckResult(
                    component="unknown",
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0,
                    details={},
                    error=str(result)
                ))
            else:
                self.results.append(result)
        
        # Add system check result
        self.results.append(system_check)
        
        # Generate summary
        summary = self._generate_summary()
        exit_code = self._determine_exit_code(summary)
        
        return summary, exit_code

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate health check summary"""
        healthy_count = sum(1 for r in self.results if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in self.results if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in self.results if r.status == HealthStatus.UNHEALTHY)
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        
        return {
            "timestamp": time.time(),
            "overall_status": overall_status.value,
            "summary": {
                "total_checks": len(self.results),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "components": {
                result.component: {
                    "status": result.status.value,
                    "response_time_ms": result.response_time_ms,
                    "details": result.details,
                    "error": result.error,
                    "timestamp": result.timestamp
                }
                for result in self.results
            }
        }

    def _determine_exit_code(self, summary: Dict[str, Any]) -> int:
        """Determine exit code based on health status"""
        status = summary["overall_status"]
        if status == "healthy":
            return 0
        elif status == "degraded":
            return 1
        else:  # unhealthy
            return 2


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    return {
        "api_url": os.getenv("PROJECT_INDEX_URL", "http://localhost:8100"),
        "db_host": os.getenv("DATABASE_HOST", "localhost"),
        "db_port": int(os.getenv("POSTGRES_PORT", "5433")),
        "db_user": os.getenv("DATABASE_USER", "project_user"),
        "db_password": os.getenv("PROJECT_INDEX_PASSWORD", ""),
        "db_name": os.getenv("DATABASE_NAME", "project_index"),
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6380"),
        "workspace_path": os.getenv("HOST_PROJECT_PATH", "/workspace")
    }


async def main():
    """Main health check execution"""
    parser = argparse.ArgumentParser(description="Project Index Health Check")
    parser.add_argument("--output", choices=["json", "text"], default="text",
                       help="Output format")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout in seconds")
    args = parser.parse_args()
    
    config = load_config()
    checker = HealthChecker(config)
    
    try:
        summary, exit_code = await asyncio.wait_for(
            checker.run_all_checks(),
            timeout=args.timeout
        )
        
        if args.output == "json":
            print(json.dumps(summary, indent=2))
        else:
            # Text output
            print(f"Project Index Health Check - {summary['overall_status'].upper()}")
            print(f"Timestamp: {time.ctime(summary['timestamp'])}")
            print(f"Total Components: {summary['summary']['total_checks']}")
            print(f"Healthy: {summary['summary']['healthy']}")
            print(f"Degraded: {summary['summary']['degraded']}")
            print(f"Unhealthy: {summary['summary']['unhealthy']}")
            print()
            
            for component, data in summary['components'].items():
                status = data['status'].upper()
                response_time = data['response_time_ms']
                print(f"{component}: {status} ({response_time:.1f}ms)")
                if data['error']:
                    print(f"  Error: {data['error']}")
                print()
        
        sys.exit(exit_code)
        
    except asyncio.TimeoutError:
        print("Health check timed out", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Health check failed: {str(e)}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    asyncio.run(main())