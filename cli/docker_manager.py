"""
Docker Infrastructure Manager

Intelligent Docker container orchestration and infrastructure automation
for the Project Index system. This module handles container lifecycle,
resource management, and service coordination.
"""

import os
import json
import yaml
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
from datetime import datetime
import docker
from docker.models.containers import Container
from docker.models.networks import Network
from docker.models.volumes import Volume

class ServiceStatus(Enum):
    """Service status enumeration"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UNHEALTHY = "unhealthy"

class DeploymentProfile(Enum):
    """Deployment profile enumeration"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

@dataclass
class ServiceConfig:
    """Configuration for a Docker service"""
    name: str
    image: str
    ports: Dict[str, int]
    environment: Dict[str, str]
    volumes: List[str]
    depends_on: List[str]
    memory_limit: str
    cpu_limit: float
    health_check: Optional[Dict[str, Any]] = None
    restart_policy: str = "unless-stopped"
    networks: List[str] = None

@dataclass
class InfrastructureConfig:
    """Complete infrastructure configuration"""
    profile: DeploymentProfile
    project_name: str
    project_path: str
    services: Dict[str, ServiceConfig]
    networks: Dict[str, Dict[str, Any]]
    volumes: Dict[str, Dict[str, Any]]
    secrets: Dict[str, str]
    monitoring_enabled: bool
    backup_enabled: bool
    ssl_enabled: bool

@dataclass
class ServiceHealth:
    """Service health information"""
    name: str
    status: ServiceStatus
    uptime: Optional[int]
    memory_usage: Optional[float]
    cpu_usage: Optional[float]
    restart_count: int
    last_error: Optional[str]
    health_checks_passed: int
    health_checks_failed: int

class DockerManager:
    """Advanced Docker infrastructure management system"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.docker_client = None
        self.compose_file_path = None
        self.env_file_path = None
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            raise
        
        # Service configurations by profile
        self.profile_configs = {
            DeploymentProfile.SMALL: {
                'memory_multiplier': 0.5,
                'cpu_multiplier': 0.5,
                'replicas': 1,
                'monitoring': False,
                'ssl': False
            },
            DeploymentProfile.MEDIUM: {
                'memory_multiplier': 1.0,
                'cpu_multiplier': 1.0,
                'replicas': 1,
                'monitoring': True,
                'ssl': False
            },
            DeploymentProfile.LARGE: {
                'memory_multiplier': 2.0,
                'cpu_multiplier': 2.0,
                'replicas': 2,
                'monitoring': True,
                'ssl': True
            },
            DeploymentProfile.ENTERPRISE: {
                'memory_multiplier': 4.0,
                'cpu_multiplier': 4.0,
                'replicas': 3,
                'monitoring': True,
                'ssl': True
            }
        }
    
    def generate_infrastructure_config(self, 
                                     profile: DeploymentProfile,
                                     project_name: str,
                                     project_path: str,
                                     detected_frameworks: List[str],
                                     ports: Dict[str, int],
                                     passwords: Dict[str, str]) -> InfrastructureConfig:
        """Generate complete infrastructure configuration"""
        self.logger.info(f"Generating infrastructure config for profile: {profile.value}")
        
        profile_config = self.profile_configs[profile]
        
        # Base services that are always included
        services = {}
        
        # PostgreSQL Database
        services['postgres'] = ServiceConfig(
            name='postgres',
            image='pgvector/pgvector:pg15',
            ports={'5432': 5432},
            environment={
                'POSTGRES_DB': f'{project_name}_db',
                'POSTGRES_USER': 'project_index',
                'POSTGRES_PASSWORD': passwords['database'],
                'POSTGRES_INITDB_ARGS': '--encoding=UTF-8 --lc-collate=C --lc-ctype=C',
            },
            volumes=[
                f'{project_name}_postgres_data:/var/lib/postgresql/data',
                './init-scripts:/docker-entrypoint-initdb.d'
            ],
            depends_on=[],
            memory_limit=f"{int(512 * profile_config['memory_multiplier'])}m",
            cpu_limit=0.5 * profile_config['cpu_multiplier'],
            health_check={
                'test': ['CMD-SHELL', 'pg_isready -U project_index'],
                'interval': '10s',
                'timeout': '5s',
                'retries': 5,
                'start_period': '30s'
            }
        )
        
        # Redis Cache
        services['redis'] = ServiceConfig(
            name='redis',
            image='redis:7-alpine',
            ports={'6379': 6379},
            environment={
                'REDIS_PASSWORD': passwords['redis']
            },
            volumes=[
                f'{project_name}_redis_data:/data',
                './redis.conf:/usr/local/etc/redis/redis.conf'
            ],
            depends_on=[],
            memory_limit=f"{int(256 * profile_config['memory_multiplier'])}m",
            cpu_limit=0.25 * profile_config['cpu_multiplier'],
            health_check={
                'test': ['CMD', 'redis-cli', '--raw', 'incr', 'ping'],
                'interval': '10s',
                'timeout': '3s',
                'retries': 5
            }
        )
        
        # Project Index API
        services['project-index-api'] = ServiceConfig(
            name='project-index-api',
            image='project-index:latest',
            ports={'8100': ports.get('api', 8100)},
            environment={
                'DATABASE_URL': f'postgresql://project_index:{passwords["database"]}@postgres:5432/{project_name}_db',
                'REDIS_URL': f'redis://:{passwords["redis"]}@redis:6379/0',
                'PROJECT_PATH': '/project',
                'ENVIRONMENT': 'production',
                'LOG_LEVEL': 'INFO'
            },
            volumes=[
                f'{project_path}:/project:ro',
                f'{project_name}_api_logs:/app/logs'
            ],
            depends_on=['postgres', 'redis'],
            memory_limit=f"{int(1024 * profile_config['memory_multiplier'])}m",
            cpu_limit=1.0 * profile_config['cpu_multiplier'],
            health_check={
                'test': ['CMD', 'curl', '-f', 'http://localhost:8100/health'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3,
                'start_period': '60s'
            }
        )
        
        # Dashboard (if enabled)
        if 'dashboard' in ports:
            services['project-index-dashboard'] = ServiceConfig(
                name='project-index-dashboard',
                image='project-index-dashboard:latest',
                ports={'8080': ports['dashboard']},
                environment={
                    'API_URL': f'http://project-index-api:8100',
                    'ENVIRONMENT': 'production'
                },
                volumes=[],
                depends_on=['project-index-api'],
                memory_limit=f"{int(512 * profile_config['memory_multiplier'])}m",
                cpu_limit=0.5 * profile_config['cpu_multiplier'],
                health_check={
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8080'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3
                }
            )
        
        # Monitoring services (if enabled)
        if profile_config['monitoring']:
            # Prometheus
            services['prometheus'] = ServiceConfig(
                name='prometheus',
                image='prom/prometheus:latest',
                ports={'9090': ports.get('metrics', 9090)},
                environment={},
                volumes=[
                    './prometheus.yml:/etc/prometheus/prometheus.yml',
                    f'{project_name}_prometheus_data:/prometheus'
                ],
                depends_on=[],
                memory_limit=f"{int(256 * profile_config['memory_multiplier'])}m",
                cpu_limit=0.25 * profile_config['cpu_multiplier']
            )
            
            # Grafana
            services['grafana'] = ServiceConfig(
                name='grafana',
                image='grafana/grafana:latest',
                ports={'3000': 3001},
                environment={
                    'GF_SECURITY_ADMIN_PASSWORD': passwords.get('grafana', 'admin'),
                    'GF_INSTALL_PLUGINS': 'grafana-piechart-panel'
                },
                volumes=[
                    f'{project_name}_grafana_data:/var/lib/grafana',
                    './grafana/dashboards:/var/lib/grafana/dashboards',
                    './grafana/provisioning:/etc/grafana/provisioning'
                ],
                depends_on=['prometheus'],
                memory_limit=f"{int(256 * profile_config['memory_multiplier'])}m",
                cpu_limit=0.25 * profile_config['cpu_multiplier']
            )
        
        # Framework-specific services
        if 'Docker' in detected_frameworks:
            # Add Docker-in-Docker for containerized builds
            services['docker-dind'] = ServiceConfig(
                name='docker-dind',
                image='docker:dind',
                ports={},
                environment={'DOCKER_TLS_CERTDIR': ''},
                volumes=[
                    f'{project_name}_docker_data:/var/lib/docker'
                ],
                depends_on=[],
                memory_limit=f"{int(512 * profile_config['memory_multiplier'])}m",
                cpu_limit=0.5 * profile_config['cpu_multiplier']
            )
        
        # Networks
        networks = {
            'project-index-network': {
                'driver': 'bridge',
                'attachable': True,
                'labels': {
                    'project': project_name,
                    'component': 'project-index'
                }
            }
        }
        
        # Volumes
        volumes = {
            f'{project_name}_postgres_data': {'driver': 'local'},
            f'{project_name}_redis_data': {'driver': 'local'},
            f'{project_name}_api_logs': {'driver': 'local'}
        }
        
        if profile_config['monitoring']:
            volumes[f'{project_name}_prometheus_data'] = {'driver': 'local'}
            volumes[f'{project_name}_grafana_data'] = {'driver': 'local'}
        
        return InfrastructureConfig(
            profile=profile,
            project_name=project_name,
            project_path=project_path,
            services=services,
            networks=networks,
            volumes=volumes,
            secrets=passwords,
            monitoring_enabled=profile_config['monitoring'],
            backup_enabled=profile in [DeploymentProfile.LARGE, DeploymentProfile.ENTERPRISE],
            ssl_enabled=profile_config['ssl']
        )
    
    def generate_docker_compose_file(self, config: InfrastructureConfig, output_path: Path) -> Path:
        """Generate Docker Compose file from configuration"""
        self.logger.info("Generating Docker Compose file")
        
        compose_data = {
            'version': '3.8',
            'services': {},
            'networks': config.networks,
            'volumes': config.volumes
        }
        
        # Convert services to Docker Compose format
        for service_name, service_config in config.services.items():
            compose_service = {
                'image': service_config.image,
                'container_name': f"{config.project_name}_{service_name}",
                'restart': service_config.restart_policy,
                'environment': service_config.environment,
                'volumes': service_config.volumes,
                'networks': ['project-index-network'],
                'mem_limit': service_config.memory_limit,
                'cpus': service_config.cpu_limit
            }
            
            # Add ports if specified
            if service_config.ports:
                compose_service['ports'] = [
                    f"{host_port}:{container_port}" 
                    for container_port, host_port in service_config.ports.items()
                ]
            
            # Add dependencies
            if service_config.depends_on:
                compose_service['depends_on'] = service_config.depends_on
            
            # Add health check
            if service_config.health_check:
                compose_service['healthcheck'] = service_config.health_check
            
            compose_data['services'][service_name] = compose_service
        
        # Write Docker Compose file
        compose_file_path = output_path / 'docker-compose.yml'
        with open(compose_file_path, 'w') as f:
            yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
        
        self.compose_file_path = compose_file_path
        self.logger.info(f"Docker Compose file generated: {compose_file_path}")
        return compose_file_path
    
    def generate_environment_file(self, config: InfrastructureConfig, output_path: Path) -> Path:
        """Generate environment file with all configuration"""
        self.logger.info("Generating environment file")
        
        env_content = []
        env_content.append("# Project Index Environment Configuration")
        env_content.append(f"# Generated on {datetime.now().isoformat()}")
        env_content.append("")
        
        # Project information
        env_content.append("# Project Information")
        env_content.append(f"PROJECT_NAME={config.project_name}")
        env_content.append(f"PROJECT_PATH={config.project_path}")
        env_content.append(f"DEPLOYMENT_PROFILE={config.profile.value}")
        env_content.append("")
        
        # Database configuration
        env_content.append("# Database Configuration")
        env_content.append(f"POSTGRES_DB={config.project_name}_db")
        env_content.append("POSTGRES_USER=project_index")
        env_content.append(f"POSTGRES_PASSWORD={config.secrets['database']}")
        env_content.append("")
        
        # Redis configuration
        env_content.append("# Redis Configuration")
        env_content.append(f"REDIS_PASSWORD={config.secrets['redis']}")
        env_content.append("")
        
        # API configuration
        env_content.append("# API Configuration")
        api_service = config.services['project-index-api']
        api_port = list(api_service.ports.values())[0]
        env_content.append(f"API_PORT={api_port}")
        env_content.append(f"API_URL=http://localhost:{api_port}")
        env_content.append("")
        
        # Feature flags
        env_content.append("# Feature Flags")
        env_content.append(f"MONITORING_ENABLED={str(config.monitoring_enabled).lower()}")
        env_content.append(f"BACKUP_ENABLED={str(config.backup_enabled).lower()}")
        env_content.append(f"SSL_ENABLED={str(config.ssl_enabled).lower()}")
        env_content.append("")
        
        # Monitoring configuration (if enabled)
        if config.monitoring_enabled:
            env_content.append("# Monitoring Configuration")
            if 'prometheus' in config.services:
                prometheus_port = list(config.services['prometheus'].ports.values())[0]
                env_content.append(f"PROMETHEUS_PORT={prometheus_port}")
            if 'grafana' in config.services:
                grafana_port = list(config.services['grafana'].ports.values())[0]
                env_content.append(f"GRAFANA_PORT={grafana_port}")
                env_content.append(f"GRAFANA_PASSWORD={config.secrets.get('grafana', 'admin')}")
            env_content.append("")
        
        # Write environment file
        env_file_path = output_path / '.env'
        with open(env_file_path, 'w') as f:
            f.write('\n'.join(env_content))
        
        self.env_file_path = env_file_path
        self.logger.info(f"Environment file generated: {env_file_path}")
        return env_file_path
    
    def generate_supporting_files(self, config: InfrastructureConfig, output_path: Path):
        """Generate supporting configuration files"""
        self.logger.info("Generating supporting configuration files")
        
        # Create directories
        config_dir = output_path / 'config'
        config_dir.mkdir(exist_ok=True)
        
        scripts_dir = output_path / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # PostgreSQL initialization scripts
        init_scripts_dir = output_path / 'init-scripts'
        init_scripts_dir.mkdir(exist_ok=True)
        
        init_sql = """
-- Project Index Database Initialization
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS project_index;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set default permissions
GRANT ALL PRIVILEGES ON SCHEMA project_index TO project_index;
GRANT ALL PRIVILEGES ON SCHEMA monitoring TO project_index;
        """
        
        with open(init_scripts_dir / '01-init.sql', 'w') as f:
            f.write(init_sql)
        
        # Redis configuration
        redis_conf = """
# Redis configuration for Project Index
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
        """
        
        with open(output_path / 'redis.conf', 'w') as f:
            f.write(redis_conf)
        
        # Prometheus configuration (if monitoring enabled)
        if config.monitoring_enabled:
            prometheus_config = {
                'global': {
                    'scrape_interval': '15s',
                    'evaluation_interval': '15s'
                },
                'scrape_configs': [
                    {
                        'job_name': 'project-index-api',
                        'static_configs': [
                            {'targets': ['project-index-api:8100']}
                        ],
                        'scrape_interval': '30s',
                        'metrics_path': '/metrics'
                    }
                ]
            }
            
            with open(output_path / 'prometheus.yml', 'w') as f:
                yaml.dump(prometheus_config, f)
        
        # Health check script
        health_check_script = '''#!/bin/bash
# Project Index Health Check Script

set -e

PROJECT_NAME="${PROJECT_NAME:-project-index}"
API_URL="${API_URL:-http://localhost:8100}"

echo "🔍 Checking Project Index health..."

# Check API health
echo "📡 Checking API health..."
if curl -sf "$API_URL/health" > /dev/null; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
    exit 1
fi

# Check database connection
echo "🗄️  Checking database connection..."
if docker exec "${PROJECT_NAME}_postgres" pg_isready -U project_index > /dev/null; then
    echo "✅ Database is healthy"
else
    echo "❌ Database is not responding"
    exit 1
fi

# Check Redis connection
echo "📦 Checking Redis connection..."
if docker exec "${PROJECT_NAME}_redis" redis-cli ping > /dev/null; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
    exit 1
fi

echo "🎉 All services are healthy!"
        '''
        
        health_script_path = scripts_dir / 'health-check.sh'
        with open(health_script_path, 'w') as f:
            f.write(health_check_script)
        health_script_path.chmod(0o755)
        
        # Start script
        start_script = f'''#!/bin/bash
# Project Index Start Script

set -e

PROJECT_NAME="{config.project_name}"
COMPOSE_FILE="docker-compose.yml"

echo "🚀 Starting Project Index infrastructure..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose -f "$COMPOSE_FILE" pull

# Build custom images if needed
echo "🔨 Building custom images..."
docker-compose -f "$COMPOSE_FILE" build

# Start services
echo "🌟 Starting services..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run health check
if [ -f "scripts/health-check.sh" ]; then
    echo "🔍 Running health check..."
    ./scripts/health-check.sh
fi

echo "✅ Project Index is ready!"
echo "📊 API: http://localhost:{list(config.services['project-index-api'].ports.values())[0]}"
if 'project-index-dashboard' in config.services:
    echo "📈 Dashboard: http://localhost:{list(config.services['project-index-dashboard'].ports.values())[0]}"
if config.monitoring_enabled and 'prometheus' in config.services:
    echo "📊 Metrics: http://localhost:{list(config.services['prometheus'].ports.values())[0]}"
        '''
        
        start_script_path = scripts_dir / 'start.sh'
        with open(start_script_path, 'w') as f:
            f.write(start_script)
        start_script_path.chmod(0o755)
        
        # Stop script
        stop_script = f'''#!/bin/bash
# Project Index Stop Script

set -e

COMPOSE_FILE="docker-compose.yml"

echo "🛑 Stopping Project Index infrastructure..."

if [ -f "$COMPOSE_FILE" ]; then
    docker-compose -f "$COMPOSE_FILE" down
    echo "✅ All services stopped"
else
    echo "❌ Docker Compose file not found"
    exit 1
fi
        '''
        
        stop_script_path = scripts_dir / 'stop.sh'
        with open(stop_script_path, 'w') as f:
            f.write(stop_script)
        stop_script_path.chmod(0o755)
        
        self.logger.info("Supporting files generated successfully")
    
    def deploy_infrastructure(self, config: InfrastructureConfig, output_path: Path) -> bool:
        """Deploy the infrastructure using Docker Compose"""
        self.logger.info("Deploying infrastructure...")
        
        try:
            # Change to output directory
            original_cwd = os.getcwd()
            os.chdir(output_path)
            
            # Pull images
            self.logger.info("Pulling Docker images...")
            result = subprocess.run([
                'docker-compose', 'pull'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.warning(f"Image pull had issues: {result.stderr}")
            
            # Build custom images
            self.logger.info("Building custom images...")
            result = subprocess.run([
                'docker-compose', 'build'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                self.logger.error(f"Image build failed: {result.stderr}")
                return False
            
            # Start services
            self.logger.info("Starting services...")
            result = subprocess.run([
                'docker-compose', 'up', '-d'
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                self.logger.error(f"Service startup failed: {result.stderr}")
                return False
            
            self.logger.info("Infrastructure deployed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Infrastructure deployment timed out")
            return False
        except Exception as e:
            self.logger.error(f"Infrastructure deployment failed: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def get_service_health(self, project_name: str) -> List[ServiceHealth]:
        """Get health status of all services"""
        services = []
        
        try:
            containers = self.docker_client.containers.list(all=True)
            project_containers = [c for c in containers if c.name.startswith(f"{project_name}_")]
            
            for container in project_containers:
                service_name = container.name.replace(f"{project_name}_", "")
                
                # Get container status
                status_map = {
                    'created': ServiceStatus.STOPPED,
                    'running': ServiceStatus.RUNNING,
                    'paused': ServiceStatus.STOPPED,
                    'restarting': ServiceStatus.STARTING,
                    'removing': ServiceStatus.STOPPING,
                    'exited': ServiceStatus.STOPPED,
                    'dead': ServiceStatus.ERROR
                }
                
                status = status_map.get(container.status, ServiceStatus.UNKNOWN)
                
                # Get container stats
                memory_usage = None
                cpu_usage = None
                uptime = None
                
                if status == ServiceStatus.RUNNING:
                    try:
                        stats = container.stats(stream=False)
                        # Calculate memory usage percentage
                        memory_usage = (stats['memory_stats']['usage'] / 
                                      stats['memory_stats']['limit']) * 100
                        
                        # Calculate CPU usage (simplified)
                        cpu_delta = (stats['cpu_stats']['cpu_usage']['total_usage'] - 
                                   stats['precpu_stats']['cpu_usage']['total_usage'])
                        system_delta = (stats['cpu_stats']['system_cpu_usage'] - 
                                      stats['precpu_stats']['system_cpu_usage'])
                        cpu_usage = (cpu_delta / system_delta) * 100
                        
                        # Get uptime from container creation time
                        created_time = container.attrs['Created']
                        uptime = int((datetime.now() - datetime.fromisoformat(
                            created_time.replace('Z', '+00:00'))).total_seconds())
                    except:
                        pass
                
                # Get restart count
                restart_count = container.attrs['RestartCount']
                
                # Check health status
                health_status = container.attrs.get('State', {}).get('Health', {})
                health_checks_passed = 0
                health_checks_failed = 0
                
                if health_status:
                    health_log = health_status.get('Log', [])
                    for entry in health_log:
                        if entry.get('ExitCode') == 0:
                            health_checks_passed += 1
                        else:
                            health_checks_failed += 1
                    
                    # Update status based on health
                    if health_status.get('Status') == 'unhealthy':
                        status = ServiceStatus.UNHEALTHY
                
                services.append(ServiceHealth(
                    name=service_name,
                    status=status,
                    uptime=uptime,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    restart_count=restart_count,
                    last_error=None,  # TODO: Extract from logs
                    health_checks_passed=health_checks_passed,
                    health_checks_failed=health_checks_failed
                ))
        
        except Exception as e:
            self.logger.error(f"Failed to get service health: {e}")
        
        return services
    
    def stop_infrastructure(self, project_name: str, remove_volumes: bool = False) -> bool:
        """Stop and optionally remove infrastructure"""
        self.logger.info(f"Stopping infrastructure for project: {project_name}")
        
        try:
            # Stop containers
            containers = self.docker_client.containers.list(all=True)
            project_containers = [c for c in containers if c.name.startswith(f"{project_name}_")]
            
            for container in project_containers:
                if container.status == 'running':
                    container.stop(timeout=30)
                container.remove()
            
            # Remove network
            try:
                network = self.docker_client.networks.get('project-index-network')
                network.remove()
            except:
                pass
            
            # Remove volumes if requested
            if remove_volumes:
                volumes = self.docker_client.volumes.list()
                project_volumes = [v for v in volumes if v.name.startswith(f"{project_name}_")]
                
                for volume in project_volumes:
                    volume.remove()
            
            self.logger.info("Infrastructure stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop infrastructure: {e}")
            return False
    
    def validate_docker_environment(self) -> Tuple[bool, List[str]]:
        """Validate Docker environment and capabilities"""
        errors = []
        warnings = []
        
        try:
            # Check Docker daemon
            self.docker_client.ping()
        except Exception as e:
            errors.append(f"Docker daemon not accessible: {e}")
            return False, errors
        
        # Check Docker version
        try:
            version_info = self.docker_client.version()
            docker_version = version_info['Version']
            api_version = version_info['ApiVersion']
            
            self.logger.info(f"Docker version: {docker_version}, API version: {api_version}")
        except Exception as e:
            warnings.append(f"Could not get Docker version: {e}")
        
        # Check Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                errors.append("Docker Compose not available")
        except FileNotFoundError:
            errors.append("Docker Compose not installed")
        
        # Check available resources
        try:
            info = self.docker_client.info()
            total_memory = info.get('MemTotal', 0) / (1024**3)  # Convert to GB
            
            if total_memory < 2:
                warnings.append(f"Low memory available: {total_memory:.1f}GB (recommended: 2GB+)")
            
        except Exception as e:
            warnings.append(f"Could not check system resources: {e}")
        
        # Check for required images
        required_base_images = ['postgres:15', 'redis:7-alpine', 'nginx:alpine']
        for image in required_base_images:
            try:
                self.docker_client.images.get(image)
            except docker.errors.ImageNotFound:
                # Image will be pulled during deployment
                pass
        
        # Show warnings
        for warning in warnings:
            self.logger.warning(warning)
        
        return len(errors) == 0, errors


# Example usage and testing
if __name__ == "__main__":
    import sys
    from cli.project_detector import ProjectAnalysis, ProjectType, Language, DeploymentProfile
    
    # Mock project analysis for testing
    mock_analysis = ProjectAnalysis(
        project_path="/tmp/test-project",
        project_name="test-project",
        project_type=ProjectType.WEB_APPLICATION,
        primary_language=Language.PYTHON,
        languages={Language.PYTHON: 0.8, Language.JAVASCRIPT: 0.2},
        frameworks=[],
        build_systems=[],
        development_tools=[],
        file_count=1500,
        line_count=15000,
        estimated_complexity="medium"
    )
    
    # Initialize Docker manager
    docker_manager = DockerManager()
    
    # Validate environment
    valid, errors = docker_manager.validate_docker_environment()
    if not valid:
        print("❌ Docker environment validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print("✅ Docker environment validated")
    
    # Generate infrastructure config
    config = docker_manager.generate_infrastructure_config(
        profile=DeploymentProfile.MEDIUM,
        project_name="test-project",
        project_path="/tmp/test-project",
        detected_frameworks=['Flask', 'Redis'],
        ports={'api': 8100, 'dashboard': 8101, 'metrics': 9090},
        passwords={'database': 'test-db-pass', 'redis': 'test-redis-pass'}
    )
    
    print(f"✅ Infrastructure config generated for {len(config.services)} services")
    
    # Generate files
    output_path = Path("/tmp/project-index-test")
    output_path.mkdir(exist_ok=True)
    
    compose_file = docker_manager.generate_docker_compose_file(config, output_path)
    env_file = docker_manager.generate_environment_file(config, output_path)
    docker_manager.generate_supporting_files(config, output_path)
    
    print(f"✅ Configuration files generated in {output_path}")
    print(f"  - Docker Compose: {compose_file}")
    print(f"  - Environment: {env_file}")
    print(f"  - Supporting files: config/, scripts/")