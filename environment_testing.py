"""
Environment Testing Framework for Project Index

This module provides comprehensive environment testing capabilities including:
- Docker container health and resource usage validation
- Network connectivity and port availability testing
- Database migration and schema validation
- Redis persistence and cluster functionality testing
- File system permissions and access validation
- Operating system compatibility testing
"""

import asyncio
import docker
import json
import logging
import os
import platform
import psutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import aiofiles
import asyncpg
import redis.asyncio as redis

from validation_framework import BaseValidator, TestResult, TestStatus, ValidationConfig


logger = logging.getLogger(__name__)


class DockerEnvironmentValidator(BaseValidator):
    """Validates Docker container environment."""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.docker_client = None
        self.containers_to_check = [
            'postgres',
            'redis', 
            'project-index-api',
            'project-index-worker'
        ]
    
    async def setup(self):
        """Setup Docker client."""
        try:
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
            logger.info("Docker client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            self.docker_client = None
    
    async def test_docker_availability(self) -> Dict[str, Any]:
        """Test if Docker is available and running."""
        try:
            await self.setup()
            
            if self.docker_client is None:
                return {
                    'success': False,
                    'error': 'Docker client not available'
                }
            
            # Get Docker info
            docker_info = self.docker_client.info()
            
            return {
                'success': True,
                'docker_info': {
                    'version': docker_info.get('ServerVersion'),
                    'containers_running': docker_info.get('ContainersRunning', 0),
                    'containers_total': docker_info.get('Containers', 0),
                    'images': docker_info.get('Images', 0),
                    'driver': docker_info.get('Driver'),
                    'memory_limit': docker_info.get('MemoryLimit', False),
                    'swap_limit': docker_info.get('SwapLimit', False)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Docker availability test failed: {e}'
            }
    
    async def test_container_health(self) -> Dict[str, Any]:
        """Test health of project-related containers."""
        if self.docker_client is None:
            await self.setup()
            
        if self.docker_client is None:
            return {
                'success': False,
                'error': 'Docker not available'
            }
        
        container_results = {}
        healthy_containers = 0
        
        try:
            containers = self.docker_client.containers.list(all=True)
            
            for container_name in self.containers_to_check:
                # Find container by name
                container = None
                for c in containers:
                    if container_name in c.name or any(container_name in tag for tag in c.image.tags):
                        container = c
                        break
                
                if container is None:
                    container_results[container_name] = {
                        'found': False,
                        'status': 'not_found',
                        'healthy': False
                    }
                    continue
                
                # Check container status
                container.reload()
                is_running = container.status == 'running'
                
                # Check health if available
                health_status = None
                if hasattr(container, 'attrs') and 'State' in container.attrs:
                    health = container.attrs['State'].get('Health', {})
                    health_status = health.get('Status')
                
                is_healthy = is_running and (health_status in [None, 'healthy'])
                
                container_results[container_name] = {
                    'found': True,
                    'status': container.status,
                    'health_status': health_status,
                    'healthy': is_healthy,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'created': container.attrs['Created'],
                    'ports': container.ports
                }
                
                if is_healthy:
                    healthy_containers += 1
            
            success_rate = healthy_containers / len(self.containers_to_check)
            
            return {
                'success': success_rate >= 0.75,  # At least 75% containers should be healthy
                'healthy_containers': healthy_containers,
                'total_containers': len(self.containers_to_check),
                'health_rate': success_rate,
                'containers': container_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Container health check failed: {e}'
            }
    
    async def test_container_resources(self) -> Dict[str, Any]:
        """Test container resource usage."""
        if self.docker_client is None:
            await self.setup()
            
        if self.docker_client is None:
            return {
                'success': False,
                'error': 'Docker not available'
            }
        
        resource_results = {}
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                if any(name in container.name for name in self.containers_to_check):
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
                    
                    # Calculate memory usage
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
                    
                    resource_results[container.name] = {
                        'cpu_percent': cpu_percent,
                        'memory_usage_mb': memory_usage / (1024 * 1024),
                        'memory_limit_mb': memory_limit / (1024 * 1024),
                        'memory_percent': memory_percent,
                        'network_rx_bytes': stats['networks']['eth0']['rx_bytes'] if 'networks' in stats else 0,
                        'network_tx_bytes': stats['networks']['eth0']['tx_bytes'] if 'networks' in stats else 0
                    }
            
            # Check if any containers are using too many resources
            high_cpu_containers = [
                name for name, stats in resource_results.items() 
                if stats['cpu_percent'] > self.config.max_cpu_usage_percent
            ]
            
            high_memory_containers = [
                name for name, stats in resource_results.items()
                if stats['memory_usage_mb'] > self.config.max_memory_usage_mb
            ]
            
            return {
                'success': len(high_cpu_containers) == 0 and len(high_memory_containers) == 0,
                'resource_stats': resource_results,
                'high_cpu_containers': high_cpu_containers,
                'high_memory_containers': high_memory_containers,
                'thresholds': {
                    'max_cpu_percent': self.config.max_cpu_usage_percent,
                    'max_memory_mb': self.config.max_memory_usage_mb
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Container resource check failed: {e}'
            }


class NetworkConnectivityValidator(BaseValidator):
    """Validates network connectivity and port availability."""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.required_ports = {
            5432: 'PostgreSQL',
            6379: 'Redis',
            8000: 'API Server',
            8080: 'WebSocket Server'
        }
    
    async def test_port_availability(self) -> Dict[str, Any]:
        """Test if required ports are available and responding."""
        port_results = {}
        
        for port, service in self.required_ports.items():
            try:
                # Test if port is open
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                is_open = result == 0
                
                port_results[port] = {
                    'service': service,
                    'is_open': is_open,
                    'status': 'open' if is_open else 'closed'
                }
                
            except Exception as e:
                port_results[port] = {
                    'service': service,
                    'is_open': False,
                    'status': 'error',
                    'error': str(e)
                }
        
        open_ports = sum(1 for result in port_results.values() if result['is_open'])
        total_ports = len(self.required_ports)
        
        return {
            'success': open_ports >= total_ports * 0.8,  # At least 80% ports should be open
            'open_ports': open_ports,
            'total_ports': total_ports,
            'availability_rate': open_ports / total_ports,
            'port_status': port_results
        }
    
    async def test_network_latency(self) -> Dict[str, Any]:
        """Test network latency to key services."""
        hosts_to_test = [
            ('localhost', 'Local services'),
            ('127.0.0.1', 'Loopback'),
        ]
        
        latency_results = {}
        
        for host, description in hosts_to_test:
            try:
                # Simple ping test using subprocess
                result = subprocess.run(
                    ['ping', '-c', '3', host],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    # Parse ping output for average latency
                    output_lines = result.stdout.split('\n')
                    for line in output_lines:
                        if 'avg' in line or 'min/avg/max' in line:
                            # Extract average latency
                            parts = line.split('/')
                            if len(parts) >= 4:
                                avg_latency = float(parts[-3])
                                latency_results[host] = {
                                    'description': description,
                                    'latency_ms': avg_latency,
                                    'status': 'success'
                                }
                                break
                    else:
                        latency_results[host] = {
                            'description': description,
                            'status': 'success_no_parse',
                            'raw_output': result.stdout
                        }
                else:
                    latency_results[host] = {
                        'description': description,
                        'status': 'failed',
                        'error': result.stderr
                    }
                    
            except Exception as e:
                latency_results[host] = {
                    'description': description,
                    'status': 'error',
                    'error': str(e)
                }
        
        successful_tests = sum(
            1 for result in latency_results.values() 
            if result['status'] == 'success'
        )
        
        return {
            'success': successful_tests > 0,
            'latency_results': latency_results,
            'successful_hosts': successful_tests,
            'total_hosts': len(hosts_to_test)
        }


class DatabaseEnvironmentValidator(BaseValidator):
    """Validates database environment and migrations."""
    
    async def test_database_migrations(self) -> Dict[str, Any]:
        """Test database migration status."""
        try:
            conn = await asyncpg.connect(self.config.database_url)
            
            try:
                # Check if migration tracking table exists
                migration_table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'alembic_version'
                    )
                """)
                
                if not migration_table_exists:
                    return {
                        'success': False,
                        'error': 'Migration tracking table not found',
                        'details': 'Alembic version table missing'
                    }
                
                # Get current migration version
                current_version = await conn.fetchval(
                    "SELECT version_num FROM alembic_version ORDER BY version_num DESC LIMIT 1"
                )
                
                # Check if all required tables exist
                required_tables = [
                    'project_indexes',
                    'file_entries',
                    'dependency_relationships',
                    'analysis_sessions',
                    'index_snapshots'
                ]
                
                missing_tables = []
                for table in required_tables:
                    exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = $1
                        )
                    """, table)
                    
                    if not exists:
                        missing_tables.append(table)
                
                return {
                    'success': len(missing_tables) == 0,
                    'current_migration': current_version,
                    'required_tables': required_tables,
                    'missing_tables': missing_tables,
                    'migration_status': 'up_to_date' if len(missing_tables) == 0 else 'needs_migration'
                }
                
            finally:
                await conn.close()
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Database migration test failed: {e}'
            }
    
    async def test_database_performance(self) -> Dict[str, Any]:
        """Test database performance characteristics."""
        try:
            conn = await asyncpg.connect(self.config.database_url)
            
            try:
                performance_metrics = {}
                
                # Test simple query performance
                start_time = time.time()
                await conn.fetch("SELECT 1")
                simple_query_time = (time.time() - start_time) * 1000
                performance_metrics['simple_query_ms'] = simple_query_time
                
                # Test table scan performance (if tables exist)
                start_time = time.time()
                try:
                    result = await conn.fetch("SELECT COUNT(*) FROM project_indexes")
                    scan_time = (time.time() - start_time) * 1000
                    performance_metrics['table_scan_ms'] = scan_time
                    performance_metrics['project_count'] = result[0][0] if result else 0
                except:
                    performance_metrics['table_scan_ms'] = None
                    performance_metrics['project_count'] = None
                
                # Test connection pool
                start_time = time.time()
                await conn.execute("SELECT pg_sleep(0.001)")  # 1ms sleep
                sleep_time = (time.time() - start_time) * 1000
                performance_metrics['sleep_accuracy_ms'] = sleep_time
                
                # Check database size
                db_size = await conn.fetchval("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                performance_metrics['database_size'] = db_size
                
                # Performance thresholds
                issues = []
                if simple_query_time > 100:  # 100ms for simple query is too slow
                    issues.append(f"Simple query too slow: {simple_query_time:.1f}ms")
                
                if performance_metrics['table_scan_ms'] and performance_metrics['table_scan_ms'] > 1000:
                    issues.append(f"Table scan too slow: {performance_metrics['table_scan_ms']:.1f}ms")
                
                return {
                    'success': len(issues) == 0,
                    'performance_metrics': performance_metrics,
                    'performance_issues': issues,
                    'thresholds': {
                        'max_simple_query_ms': 100,
                        'max_table_scan_ms': 1000
                    }
                }
                
            finally:
                await conn.close()
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Database performance test failed: {e}'
            }


class FileSystemValidator(BaseValidator):
    """Validates file system permissions and access."""
    
    async def test_file_system_permissions(self) -> Dict[str, Any]:
        """Test file system read/write permissions."""
        test_paths = [
            Path.cwd(),  # Current working directory
            Path('/tmp'),  # Temporary directory
            Path.home() if Path.home().exists() else Path('/tmp')  # Home directory
        ]
        
        permission_results = {}
        
        for test_path in test_paths:
            try:
                path_result = {
                    'path': str(test_path),
                    'exists': test_path.exists(),
                    'is_dir': test_path.is_dir() if test_path.exists() else False,
                    'readable': False,
                    'writable': False,
                    'executable': False
                }
                
                if test_path.exists():
                    # Test read permission
                    try:
                        list(test_path.iterdir() if test_path.is_dir() else [test_path])
                        path_result['readable'] = True
                    except PermissionError:
                        path_result['readable'] = False
                    
                    # Test write permission
                    try:
                        test_file = test_path / f'test_write_{uuid.uuid4().hex[:8]}.tmp'
                        test_file.write_text('test')
                        test_file.unlink()
                        path_result['writable'] = True
                    except (PermissionError, OSError):
                        path_result['writable'] = False
                    
                    # Test execute permission (for directories)
                    if test_path.is_dir():
                        try:
                            os.access(test_path, os.X_OK)
                            path_result['executable'] = True
                        except:
                            path_result['executable'] = False
                
                permission_results[str(test_path)] = path_result
                
            except Exception as e:
                permission_results[str(test_path)] = {
                    'path': str(test_path),
                    'error': str(e)
                }
        
        # Check if we have at least one writable location
        writable_paths = [
            result for result in permission_results.values() 
            if result.get('writable', False)
        ]
        
        return {
            'success': len(writable_paths) > 0,
            'permission_results': permission_results,
            'writable_paths': len(writable_paths),
            'total_paths_tested': len(test_paths)
        }
    
    async def test_disk_space(self) -> Dict[str, Any]:
        """Test available disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            
            total_gb = disk_usage.total / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            
            # Calculate usage percentage
            usage_percent = (used_gb / total_gb) * 100
            
            # Check if we have enough free space (at least 1GB)
            min_free_space_gb = 1.0
            has_enough_space = free_gb >= min_free_space_gb
            
            return {
                'success': has_enough_space and usage_percent < 95,  # Less than 95% full
                'disk_stats': {
                    'total_gb': round(total_gb, 2),
                    'used_gb': round(used_gb, 2),
                    'free_gb': round(free_gb, 2),
                    'usage_percent': round(usage_percent, 1)
                },
                'requirements': {
                    'min_free_space_gb': min_free_space_gb,
                    'max_usage_percent': 95
                },
                'issues': [
                    'Insufficient free space' if free_gb < min_free_space_gb else '',
                    'Disk usage too high' if usage_percent >= 95 else ''
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Disk space test failed: {e}'
            }


class OperatingSystemValidator(BaseValidator):
    """Validates operating system compatibility and resources."""
    
    async def test_os_compatibility(self) -> Dict[str, Any]:
        """Test operating system compatibility."""
        try:
            system_info = {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'python_implementation': platform.python_implementation()
            }
            
            # Check compatibility
            compatible_systems = ['Linux', 'Darwin', 'Windows']  # macOS is Darwin
            is_compatible = platform.system() in compatible_systems
            
            # Check Python version (require 3.8+)
            python_version = tuple(map(int, platform.python_version().split('.')))
            python_compatible = python_version >= (3, 8)
            
            issues = []
            if not is_compatible:
                issues.append(f"Unsupported operating system: {platform.system()}")
            
            if not python_compatible:
                issues.append(f"Python version too old: {platform.python_version()}")
            
            return {
                'success': is_compatible and python_compatible,
                'system_info': system_info,
                'compatibility': {
                    'os_compatible': is_compatible,
                    'python_compatible': python_compatible,
                    'supported_systems': compatible_systems,
                    'min_python_version': '3.8.0'
                },
                'issues': [issue for issue in issues if issue]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'OS compatibility test failed: {e}'
            }
    
    async def test_system_resources(self) -> Dict[str, Any]:
        """Test system resource availability."""
        try:
            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_usage_percent = memory.percent
            
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Load average (Unix systems)
            load_avg = None
            try:
                load_avg = os.getloadavg()
            except (AttributeError, OSError):
                # Windows doesn't have getloadavg
                pass
            
            # Resource requirements check
            min_memory_gb = 2.0  # Minimum 2GB RAM
            max_cpu_usage = 80.0  # Maximum 80% CPU usage
            
            resource_issues = []
            
            if memory_gb < min_memory_gb:
                resource_issues.append(f"Insufficient RAM: {memory_gb:.1f}GB (minimum {min_memory_gb}GB)")
            
            if cpu_percent > max_cpu_usage:
                resource_issues.append(f"High CPU usage: {cpu_percent}% (should be below {max_cpu_usage}%)")
            
            if load_avg and load_avg[0] > cpu_count * 2:
                resource_issues.append(f"High system load: {load_avg[0]:.2f} (should be below {cpu_count * 2})")
            
            return {
                'success': len(resource_issues) == 0,
                'resource_stats': {
                    'memory_total_gb': round(memory_gb, 2),
                    'memory_available_gb': round(memory_available_gb, 2),
                    'memory_usage_percent': memory_usage_percent,
                    'cpu_count': cpu_count,
                    'cpu_usage_percent': cpu_percent,
                    'load_average': load_avg
                },
                'requirements': {
                    'min_memory_gb': min_memory_gb,
                    'max_cpu_usage_percent': max_cpu_usage
                },
                'resource_issues': resource_issues
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'System resource test failed: {e}'
            }


class EnvironmentTestSuite(BaseValidator):
    """Main environment testing suite orchestrator."""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.docker_validator = DockerEnvironmentValidator(config)
        self.network_validator = NetworkConnectivityValidator(config)
        self.database_validator = DatabaseEnvironmentValidator(config)
        self.filesystem_validator = FileSystemValidator(config)
        self.os_validator = OperatingSystemValidator(config)
    
    async def run_comprehensive_environment_tests(self) -> Dict[str, Any]:
        """Run complete environment test suite."""
        logger.info("Starting comprehensive environment tests")
        
        test_categories = [
            ('operating_system', self.os_validator.test_os_compatibility()),
            ('system_resources', self.os_validator.test_system_resources()),
            ('file_system_permissions', self.filesystem_validator.test_file_system_permissions()),
            ('disk_space', self.filesystem_validator.test_disk_space()),
            ('network_connectivity', self.network_validator.test_port_availability()),
            ('network_latency', self.network_validator.test_network_latency()),
            ('database_migrations', self.database_validator.test_database_migrations()),
            ('database_performance', self.database_validator.test_database_performance()),
            ('docker_availability', self.docker_validator.test_docker_availability()),
            ('container_health', self.docker_validator.test_container_health()),
            ('container_resources', self.docker_validator.test_container_resources())
        ]
        
        results = {}
        overall_success = True
        critical_failures = []
        
        for category, test_coro in test_categories:
            try:
                logger.info(f"Running {category} tests...")
                result = await test_coro
                results[category] = result
                
                if not result.get('success', False):
                    logger.warning(f"{category} tests failed: {result.get('error', 'Unknown error')}")
                    
                    # Mark critical failures
                    if category in ['operating_system', 'system_resources', 'file_system_permissions']:
                        critical_failures.append(category)
                        overall_success = False
                else:
                    logger.info(f"{category} tests passed")
                    
            except Exception as e:
                logger.error(f"{category} tests encountered error: {e}")
                results[category] = {
                    'success': False,
                    'error': str(e)
                }
                
                if category in ['operating_system', 'system_resources', 'file_system_permissions']:
                    critical_failures.append(category)
                    overall_success = False
        
        # Calculate summary metrics
        successful_categories = sum(1 for r in results.values() if r.get('success', False))
        total_categories = len(test_categories)
        
        return {
            'success': overall_success,
            'summary': {
                'total_categories': total_categories,
                'successful_categories': successful_categories,
                'success_rate': successful_categories / total_categories,
                'critical_failures': critical_failures,
                'environment_ready': len(critical_failures) == 0
            },
            'results': results,
            'recommendations': self._generate_environment_recommendations(results, critical_failures)
        }
    
    def _generate_environment_recommendations(self, results: Dict[str, Any], critical_failures: List[str]) -> List[str]:
        """Generate environment-specific recommendations."""
        recommendations = []
        
        if critical_failures:
            recommendations.append("Critical environment issues detected - address before proceeding")
        
        # OS compatibility recommendations
        os_result = results.get('operating_system', {})
        if not os_result.get('success', False):
            recommendations.append("Update operating system or Python version for compatibility")
        
        # Resource recommendations
        resource_result = results.get('system_resources', {})
        if not resource_result.get('success', False):
            recommendations.append("Increase system resources (RAM/CPU) or reduce system load")
        
        # Disk space recommendations
        disk_result = results.get('disk_space', {})
        if not disk_result.get('success', False):
            recommendations.append("Free up disk space before installation")
        
        # Network recommendations
        network_result = results.get('network_connectivity', {})
        if not network_result.get('success', False):
            recommendations.append("Check network configuration and firewall settings")
        
        # Database recommendations
        db_migration_result = results.get('database_migrations', {})
        if not db_migration_result.get('success', False):
            recommendations.append("Run database migrations or check database configuration")
        
        # Docker recommendations
        docker_result = results.get('docker_availability', {})
        if not docker_result.get('success', False):
            recommendations.append("Install and configure Docker for containerized deployment")
        
        container_result = results.get('container_health', {})
        if container_result.get('success', False) == False:
            recommendations.append("Start required containers or check container configuration")
        
        return recommendations