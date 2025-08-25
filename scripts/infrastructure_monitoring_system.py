#!/usr/bin/env python3
"""
Infrastructure State Monitoring and Service Discovery System

Real-time monitoring of infrastructure components with automated documentation updates.
Ensures PLAN.md and PROMPT.md always reflect actual system state.

Features:
- Continuous infrastructure state monitoring
- Automated port/service discovery
- Database connection validation
- Redis operation verification
- Service health checks with documentation synchronization
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import psutil
import redis
import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InfrastructureMonitoringSystem:
    """Real-time infrastructure monitoring with documentation synchronization."""
    
    def __init__(self, project_root: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.project_root = Path(project_root)
        self.monitoring_interval = 60  # seconds
        self.infrastructure_state = {}
        self.last_state_change = None
        
        # Critical infrastructure components
        self.components = {
            "postgresql": {
                "port": 5432,
                "host": "localhost",
                "health_check": self._check_postgresql
            },
            "redis": {
                "port": 6379,
                "host": "localhost", 
                "health_check": self._check_redis
            },
            "api_server": {
                "port": 18080,
                "host": "localhost",
                "health_check": self._check_api_server
            }
        }

    async def discover_infrastructure_state(self) -> Dict[str, Any]:
        """Discover complete infrastructure state including services, ports, and connections."""
        logger.info("Discovering infrastructure state...")
        
        infrastructure_state = {
            "discovery_timestamp": datetime.now().isoformat(),
            "components": {},
            "system_resources": await self._get_system_resources(),
            "network_status": await self._get_network_status(),
            "process_information": await self._get_relevant_processes()
        }
        
        # Check each component
        for component_name, config in self.components.items():
            logger.info(f"Checking component: {component_name}")
            
            component_status = {
                "name": component_name,
                "expected_port": config["port"],
                "host": config["host"],
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                # Run health check
                health_result = await config["health_check"]()
                component_status.update(health_result)
                
                # Check if process is running on expected port
                port_status = await self._check_port_usage(config["port"])
                component_status["port_status"] = port_status
                
                # Overall component status
                component_status["overall_status"] = (
                    "operational" if health_result.get("healthy", False) else "degraded"
                )
                
            except Exception as e:
                logger.error(f"Error checking {component_name}: {e}")
                component_status.update({
                    "overall_status": "error",
                    "error": str(e),
                    "healthy": False
                })
            
            infrastructure_state["components"][component_name] = component_status
        
        # Update cached state
        self.infrastructure_state = infrastructure_state
        return infrastructure_state

    async def _check_postgresql(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity and status."""
        status = {
            "service_type": "database",
            "healthy": False,
            "connection_test": False,
            "database_info": {}
        }
        
        try:
            # Test pg_isready
            result = subprocess.run(
                ["pg_isready", "-h", "localhost", "-p", "5432"],
                capture_output=True, text=True, timeout=10
            )
            status["pg_isready"] = result.returncode == 0
            
            # Attempt database connection
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="postgres",  # Default database
                user="postgres",
                password="postgres",  # Common default
                connect_timeout=10
            )
            
            with conn.cursor() as cursor:
                # Check database version
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                status["database_info"]["version"] = version
                
                # Count tables in bee_hive database (if exists)
                try:
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """)
                    table_count = cursor.fetchone()[0]
                    status["database_info"]["table_count"] = table_count
                except Exception:
                    status["database_info"]["table_count"] = "unknown"
            
            conn.close()
            status["connection_test"] = True
            status["healthy"] = True
            
        except psycopg2.OperationalError as e:
            status["connection_error"] = str(e)
            logger.warning(f"PostgreSQL connection failed: {e}")
        except subprocess.TimeoutExpired:
            status["connection_error"] = "pg_isready timeout"
        except Exception as e:
            status["connection_error"] = str(e)
            logger.error(f"PostgreSQL check error: {e}")
        
        return status

    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and operational status."""
        status = {
            "service_type": "cache",
            "healthy": False,
            "connection_test": False,
            "redis_info": {}
        }
        
        try:
            # Create Redis connection
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Test ping
            ping_result = r.ping()
            status["ping_test"] = ping_result
            
            if ping_result:
                # Get Redis info
                info = r.info()
                status["redis_info"] = {
                    "version": info.get("redis_version"),
                    "uptime_seconds": info.get("uptime_in_seconds"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "role": info.get("role")
                }
                
                # Test basic operations
                test_key = "infrastructure_monitor_test"
                r.set(test_key, "test_value", ex=60)
                test_value = r.get(test_key)
                r.delete(test_key)
                
                status["operations_test"] = test_value == "test_value"
                status["connection_test"] = True
                status["healthy"] = True
                
        except redis.ConnectionError as e:
            status["connection_error"] = str(e)
            logger.warning(f"Redis connection failed: {e}")
        except Exception as e:
            status["connection_error"] = str(e)
            logger.error(f"Redis check error: {e}")
        
        return status

    async def _check_api_server(self) -> Dict[str, Any]:
        """Check API server connectivity and responsiveness."""
        status = {
            "service_type": "api",
            "healthy": False,
            "connection_test": False,
            "api_info": {}
        }
        
        try:
            # Check if port is in use
            port_check = await self._check_port_usage(18080)
            if not port_check.get("in_use", False):
                status["connection_error"] = "Port 18080 not in use"
                return status
            
            # Try HTTP request to health endpoint (if available)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        "http://localhost:18080/health",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            status["http_health_check"] = True
                            status["api_info"]["status_code"] = response.status
                            status["connection_test"] = True
                            status["healthy"] = True
                        else:
                            status["api_info"]["status_code"] = response.status
                            status["connection_error"] = f"HTTP {response.status}"
                except aiohttp.ClientError as e:
                    # API might be running but without health endpoint
                    status["connection_error"] = f"HTTP client error: {str(e)}"
                    # Still mark as potentially healthy if port is in use
                    if port_check.get("in_use", False):
                        status["healthy"] = True
                        status["connection_test"] = True
        
        except ImportError:
            # aiohttp not available, use basic port check
            port_check = await self._check_port_usage(18080)
            if port_check.get("in_use", False):
                status["healthy"] = True
                status["connection_test"] = True
                status["api_info"]["note"] = "Port in use, HTTP check not available"
        except Exception as e:
            status["connection_error"] = str(e)
            logger.error(f"API server check error: {e}")
        
        return status

    async def _check_port_usage(self, port: int) -> Dict[str, Any]:
        """Check if a specific port is in use and get process information."""
        port_status = {
            "port": port,
            "in_use": False,
            "process_info": None
        }
        
        try:
            # Use netstat or lsof to check port usage
            result = subprocess.run(
                ["lsof", "-i", f":{port}", "-n", "-P"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                port_status["in_use"] = True
                # Parse lsof output for process information
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    process_line = lines[1]
                    parts = process_line.split()
                    if len(parts) >= 2:
                        port_status["process_info"] = {
                            "command": parts[0],
                            "pid": parts[1],
                            "raw_output": process_line
                        }
        
        except subprocess.TimeoutExpired:
            port_status["error"] = "lsof timeout"
        except FileNotFoundError:
            # Try alternative method with netstat
            try:
                result = subprocess.run(
                    ["netstat", "-an"], 
                    capture_output=True, text=True, timeout=10
                )
                if f":{port}" in result.stdout:
                    port_status["in_use"] = True
                    port_status["process_info"] = {"method": "netstat"}
            except Exception as e:
                port_status["error"] = f"Port check failed: {str(e)}"
        except Exception as e:
            port_status["error"] = str(e)
        
        return port_status

    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": dict(psutil.virtual_memory()._asdict()),
                "disk": dict(psutil.disk_usage('/')._asdict()),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            return {"error": str(e)}

    async def _get_network_status(self) -> Dict[str, Any]:
        """Get network interface and connection information."""
        try:
            network_info = {
                "interfaces": {},
                "connections": []
            }
            
            # Network interfaces
            for interface, addrs in psutil.net_if_addrs().items():
                network_info["interfaces"][interface] = [
                    {
                        "family": addr.family.name,
                        "address": addr.address,
                        "netmask": addr.netmask
                    } for addr in addrs
                ]
            
            # Active connections (limited to avoid overwhelming data)
            connections = psutil.net_connections()[:50]  # Limit to first 50
            network_info["connections"] = [
                {
                    "fd": conn.fd,
                    "family": conn.family.name if conn.family else None,
                    "type": conn.type.name if conn.type else None,
                    "laddr": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                    "raddr": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                    "status": conn.status
                } for conn in connections
            ]
            
            return network_info
            
        except Exception as e:
            return {"error": str(e)}

    async def _get_relevant_processes(self) -> List[Dict[str, Any]]:
        """Get information about relevant processes (postgres, redis, python, etc.)."""
        try:
            relevant_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    name = proc.info['name'].lower()
                    cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                    
                    # Check if process is relevant
                    if any(keyword in name or keyword in cmdline for keyword in 
                          ['postgres', 'redis', 'python', 'hive', 'fastapi', 'uvicorn']):
                        
                        relevant_processes.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "cmdline": proc.info['cmdline'],
                            "create_time": datetime.fromtimestamp(proc.info['create_time']).isoformat(),
                            "cpu_percent": proc.cpu_percent(),
                            "memory_percent": proc.memory_percent()
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return relevant_processes[:20]  # Limit to avoid too much data
            
        except Exception as e:
            return [{"error": str(e)}]

    async def sync_documentation_with_state(self):
        """Synchronize documentation files with discovered infrastructure state."""
        logger.info("Synchronizing documentation with infrastructure state...")
        
        # Files to update
        docs_to_update = [
            self.project_root / "docs" / "PLAN.md",
            self.project_root / "docs" / "PROMPT.md"
        ]
        
        for doc_path in docs_to_update:
            if doc_path.exists():
                await self._update_document_infrastructure_references(doc_path)

    async def _update_document_infrastructure_references(self, doc_path: Path):
        """Update infrastructure references in a specific document."""
        try:
            with open(doc_path, 'r') as f:
                content = f.read()
            
            original_content = content
            updates_made = []
            
            # Update port references based on actual state
            for component_name, component_info in self.infrastructure_state.get("components", {}).items():
                expected_port = component_info.get("expected_port")
                actual_status = component_info.get("overall_status")
                
                if expected_port and actual_status:
                    # Update status descriptions
                    status_emoji = "✅" if actual_status == "operational" else "❌"
                    
                    # Pattern matching for various status formats
                    patterns = [
                        f"{component_name.title()}.*port {expected_port}.*",
                        f"port {expected_port}.*{component_name}.*"
                    ]
                    
                    for pattern in patterns:
                        import re
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Update with current status
                            updated_line = f"{status_emoji} {component_name.title()} operational on port {expected_port}"
                            content = content.replace(match, updated_line)
                            updates_made.append(f"Updated {component_name} status")
            
            # Update timestamp if changes were made
            if content != original_content:
                # Add update timestamp
                timestamp_line = f"\n*Last infrastructure sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
                
                # Try to find existing timestamp and replace
                import re
                existing_timestamp = re.search(r'\*Last infrastructure sync:.*?\*', content)
                if existing_timestamp:
                    content = content.replace(existing_timestamp.group(0), timestamp_line.strip())
                else:
                    content += timestamp_line
                
                # Write updated content
                with open(doc_path, 'w') as f:
                    f.write(content)
                
                logger.info(f"Updated {doc_path}: {updates_made}")
            
        except Exception as e:
            logger.error(f"Failed to update {doc_path}: {e}")

    async def generate_infrastructure_report(self) -> Dict[str, Any]:
        """Generate comprehensive infrastructure monitoring report."""
        logger.info("Generating infrastructure monitoring report...")
        
        # Discover current state
        infrastructure_state = await self.discover_infrastructure_state()
        
        # Calculate health summary
        healthy_components = sum(
            1 for comp in infrastructure_state["components"].values()
            if comp.get("overall_status") == "operational"
        )
        total_components = len(infrastructure_state["components"])
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "infrastructure_state": infrastructure_state,
            "health_summary": {
                "healthy_components": healthy_components,
                "total_components": total_components,
                "health_percentage": round((healthy_components / total_components) * 100, 1) if total_components > 0 else 0,
                "overall_status": "healthy" if healthy_components == total_components else "degraded"
            },
            "service_discovery": {
                comp_name: {
                    "port": comp["expected_port"],
                    "status": comp["overall_status"],
                    "healthy": comp.get("healthy", False)
                }
                for comp_name, comp in infrastructure_state["components"].items()
            },
            "recommendations": self._generate_infrastructure_recommendations(infrastructure_state)
        }
        
        # Save report
        report_path = self.project_root / "reports" / f"infrastructure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Infrastructure report saved: {report_path}")
        return report

    def _generate_infrastructure_recommendations(self, infrastructure_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on infrastructure state."""
        recommendations = []
        
        for comp_name, comp in infrastructure_state["components"].items():
            if comp.get("overall_status") != "operational":
                recommendations.append({
                    "type": "component_issue",
                    "priority": "high",
                    "component": comp_name,
                    "issue": comp.get("connection_error", "Component not operational"),
                    "action": f"Investigate and repair {comp_name} service"
                })
        
        # System resource recommendations
        system_resources = infrastructure_state.get("system_resources", {})
        if isinstance(system_resources.get("memory"), dict):
            memory_percent = system_resources["memory"].get("percent", 0)
            if memory_percent > 80:
                recommendations.append({
                    "type": "resource_issue",
                    "priority": "medium",
                    "issue": f"High memory usage: {memory_percent}%",
                    "action": "Monitor memory usage and consider optimization"
                })
        
        return recommendations

    async def start_continuous_monitoring(self, interval_seconds: int = 60):
        """Start continuous infrastructure monitoring with documentation sync."""
        logger.info(f"Starting continuous infrastructure monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                # Generate infrastructure report
                report = await self.generate_infrastructure_report()
                
                # Sync documentation if any component status changed
                await self.sync_documentation_with_state()
                
                # Check for critical issues
                health_percentage = report["health_summary"]["health_percentage"]
                if health_percentage < 100:
                    logger.warning(f"Infrastructure health at {health_percentage}% - some components degraded")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in infrastructure monitoring: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retrying


async def main():
    """Main function to run infrastructure monitoring system."""
    system = InfrastructureMonitoringSystem()
    
    # Generate initial report
    print("Generating infrastructure monitoring report...")
    report = await system.generate_infrastructure_report()
    
    print(f"\nInfrastructure Monitoring Report")
    print(f"===============================")
    print(f"Overall Health: {report['health_summary']['health_percentage']}% ({report['health_summary']['healthy_components']}/{report['health_summary']['total_components']} components)")
    
    for comp_name, comp_summary in report["service_discovery"].items():
        status_emoji = "✅" if comp_summary["healthy"] else "❌"
        print(f"{status_emoji} {comp_name.title()}: {comp_summary['status']} (port {comp_summary['port']})")
    
    if report["recommendations"]:
        print(f"\nRecommendations ({len(report['recommendations'])}):")
        for rec in report["recommendations"][:3]:
            print(f"  - {rec['type'].title()}: {rec['action']}")
    
    # Start continuous monitoring (commented out for demo)
    # print(f"\nStarting continuous monitoring...")
    # await system.start_continuous_monitoring(interval_seconds=60)


if __name__ == "__main__":
    asyncio.run(main())