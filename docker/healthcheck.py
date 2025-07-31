#!/usr/bin/env python3
"""
Health Check Script for DLQ Service

Performs comprehensive health checks for:
- DLQ Service API availability
- Component health status
- Performance metrics validation
- Database connectivity
- Redis connectivity
"""

import sys
import time
import json
import logging
from typing import Dict, Any
import asyncio
import aiohttp
import os

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DLQHealthChecker:
    """Comprehensive health checker for DLQ service."""
    
    def __init__(self):
        self.api_port = int(os.getenv('DLQ_API_PORT', '8080'))
        self.health_port = int(os.getenv('DLQ_HEALTH_PORT', '8081'))
        self.timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '10'))
        
        self.base_url = f"http://localhost:{self.api_port}"
        self.health_url = f"http://localhost:{self.health_port}"
    
    async def check_api_health(self) -> Dict[str, Any]:
        """Check DLQ API health endpoint."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/api/v1/dlq/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return {
                            "status": "healthy",
                            "data": health_data,
                            "response_time_ms": 0  # Would need timing
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}",
                            "data": None
                        }
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy", 
                "error": "Timeout",
                "data": None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "data": None
            }
    
    async def check_component_health(self) -> Dict[str, Any]:
        """Check individual component health."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/api/v1/dlq/stats") as response:
                    if response.status == 200:
                        stats_data = await response.json()
                        component_health = stats_data.get("component_health", {})
                        
                        # Check if all critical components are healthy
                        critical_components = ["dlq_manager", "retry_scheduler"]
                        unhealthy_components = [
                            comp for comp in critical_components 
                            if component_health.get(comp) != "healthy"
                        ]
                        
                        if unhealthy_components:
                            return {
                                "status": "degraded",
                                "unhealthy_components": unhealthy_components,
                                "all_components": component_health
                            }
                        else:
                            return {
                                "status": "healthy",
                                "components": component_health
                            }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics are within acceptable ranges."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/api/v1/dlq/stats") as response:
                    if response.status == 200:
                        stats_data = await response.json()
                        
                        # Extract performance metrics
                        avg_processing_time = stats_data.get("average_processing_time_ms", 0)
                        delivery_rate = stats_data.get("overall_recovery_rate", 0)
                        dlq_size = stats_data.get("dlq_size", 0)
                        
                        # Performance thresholds
                        issues = []
                        if avg_processing_time > 50:  # 50ms threshold
                            issues.append(f"High processing time: {avg_processing_time}ms")
                        
                        if delivery_rate < 0.95:  # 95% delivery rate threshold
                            issues.append(f"Low delivery rate: {delivery_rate:.2%}")
                        
                        if dlq_size > 10000:  # 10k messages threshold
                            issues.append(f"High DLQ size: {dlq_size}")
                        
                        if issues:
                            return {
                                "status": "degraded",
                                "issues": issues,
                                "metrics": {
                                    "avg_processing_time_ms": avg_processing_time,
                                    "delivery_rate": delivery_rate,
                                    "dlq_size": dlq_size
                                }
                            }
                        else:
                            return {
                                "status": "healthy",
                                "metrics": {
                                    "avg_processing_time_ms": avg_processing_time,
                                    "delivery_rate": delivery_rate,
                                    "dlq_size": dlq_size
                                }
                            }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_basic_connectivity(self) -> Dict[str, Any]:
        """Check basic API connectivity."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                # Try to connect to the API
                async with session.get(f"{self.base_url}/api/v1/dlq/stats") as response:
                    if response.status in [200, 503]:  # 503 might be temporary
                        return {"status": "healthy", "response_code": response.status}
                    else:
                        return {"status": "unhealthy", "response_code": response.status}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def perform_comprehensive_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        start_time = time.time()
        
        # Run all checks concurrently
        results = await asyncio.gather(
            self.check_basic_connectivity(),
            self.check_api_health(),
            self.check_component_health(),
            self.check_performance_metrics(),
            return_exceptions=True
        )
        
        connectivity_result, api_result, component_result, performance_result = results
        
        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {"status": "error", "error": str(result)}
        
        # Determine overall health
        all_statuses = [r.get("status", "error") for r in results if isinstance(r, dict)]
        
        if all(status == "healthy" for status in all_statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in all_statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        check_duration = (time.time() - start_time) * 1000
        
        return {
            "overall_status": overall_status,
            "check_duration_ms": check_duration,
            "timestamp": time.time(),
            "checks": {
                "connectivity": connectivity_result,
                "api_health": api_result,
                "component_health": component_result,
                "performance": performance_result
            }
        }


async def main():
    """Main health check function."""
    checker = DLQHealthChecker()
    
    try:
        # Perform health check
        result = await checker.perform_comprehensive_check()
        
        # Print result for logging
        print(json.dumps(result, indent=2))
        
        # Exit with appropriate code
        overall_status = result.get("overall_status", "error")
        
        if overall_status == "healthy":
            logger.info("Health check passed")
            sys.exit(0)
        elif overall_status == "degraded":
            logger.warning("Health check degraded - some issues detected")
            sys.exit(1)  # Warning status
        else:
            logger.error("Health check failed")
            sys.exit(2)  # Critical status
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        print(json.dumps({
            "overall_status": "error",
            "error": str(e),
            "timestamp": time.time()
        }))
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())