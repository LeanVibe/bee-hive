#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Port Configuration Resolver
====================================================
DevOps-Engineer: Script to resolve port configuration conflicts
and standardize production deployment ports.
"""

import os
import sys
import subprocess
import json
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

class PortConflictResolver:
    """Resolves port configuration conflicts for production deployment."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_files = [
            ".env",
            ".env.production",
            ".env.production.unified",
            "docker-compose.yml",
            "docker-compose.production.yml"
        ]
        
        # Standard production port mapping
        self.production_ports = {
            "api_internal": 8000,        # API inside container
            "api_external": 80,          # External via Nginx
            "api_ssl": 443,              # SSL external
            "postgres_internal": 5432,   # PostgreSQL inside container
            "redis_internal": 6379,      # Redis inside container
            "prometheus": 9090,          # Prometheus
            "grafana": 3000,             # Grafana
            "nginx_http": 80,            # Nginx HTTP
            "nginx_https": 443,          # Nginx HTTPS
        }
        
        # Development non-standard ports (to avoid conflicts)
        self.development_ports = {
            "api_dev": 18080,           # API development
            "postgres_dev": 15432,      # PostgreSQL development
            "redis_dev": 16379,         # Redis development
            "pwa_dev": 18443,           # PWA development
            "pwa_preview": 18444,       # PWA preview
            "prometheus_dev": 19090,    # Prometheus development
        }
    
    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False
    
    def get_docker_container_ports(self) -> Dict[str, List[str]]:
        """Get currently running Docker container ports."""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', 'json'], 
                capture_output=True, text=True
            )
            
            containers = {}
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        container = json.loads(line)
                        containers[container.get('Names', 'unknown')] = container.get('Ports', '')
            
            return containers
        except Exception as e:
            print(f"Warning: Could not get Docker container info: {e}")
            return {}
    
    def analyze_current_configuration(self) -> Dict:
        """Analyze current port configuration across all files."""
        analysis = {
            "env_configs": {},
            "docker_configs": {},
            "running_containers": {},
            "conflicts": [],
            "recommendations": []
        }
        
        # Analyze .env files
        for env_file in self.env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                analysis["env_configs"][env_file] = self._parse_env_file(env_path)
        
        # Get Docker container status
        analysis["running_containers"] = self.get_docker_container_ports()
        
        # Check for conflicts
        analysis["conflicts"] = self._detect_conflicts(analysis["env_configs"])
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _parse_env_file(self, file_path: Path) -> Dict:
        """Parse environment file for port configurations."""
        config = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if 'PORT' in key.upper() or 'URL' in key.upper():
                            config[key] = value
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
        
        return config
    
    def _detect_conflicts(self, env_configs: Dict) -> List[str]:
        """Detect port configuration conflicts."""
        conflicts = []
        
        # Check for inconsistent port definitions across files
        all_ports = {}
        for file_name, config in env_configs.items():
            for key, value in config.items():
                if key not in all_ports:
                    all_ports[key] = {}
                all_ports[key][file_name] = value
        
        for key, files in all_ports.items():
            if len(set(files.values())) > 1:
                conflicts.append(f"Port conflict for {key}: {files}")
        
        return conflicts
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations to resolve conflicts."""
        recommendations = []
        
        # Check if production services are running on development ports
        for container_name, ports in analysis["running_containers"].items():
            if "leanvibe" in container_name and ports:
                if "5432->5432" in ports:
                    recommendations.append(
                        f"Container {container_name} using standard PostgreSQL port 5432 - "
                        "consider using development port 15432 to avoid conflicts"
                    )
                if "6379->6379" in ports:
                    recommendations.append(
                        f"Container {container_name} using standard Redis port 6379 - "
                        "consider using development port 16379 to avoid conflicts"
                    )
        
        # Check for missing production configuration
        if ".env.production" not in analysis["env_configs"]:
            recommendations.append("Create .env.production with standardized production ports")
        
        if analysis["conflicts"]:
            recommendations.append("Resolve port conflicts between configuration files")
        
        return recommendations
    
    def create_standardized_production_config(self) -> None:
        """Create standardized production configuration."""
        production_config = """# LeanVibe Agent Hive 2.0 - Standardized Production Configuration
# ================================================================
# Resolves all port conflicts and provides production-ready setup

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database Configuration (Standard Production Ports)
DATABASE_URL=postgresql+asyncpg://leanvibe_user:${POSTGRES_PASSWORD}@postgres:5432/leanvibe_agent_hive
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0

# API Configuration (Standard Production Ports)
API_HOST=0.0.0.0
API_PORT=8000  # Internal container port
API_WORKERS=4

# External Access (via Nginx)
NGINX_HTTP_PORT=80
NGINX_HTTPS_PORT=443

# Security Configuration
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# CORS Configuration (Production Domains)
CORS_ORIGINS=https://${DOMAIN_NAME},https://www.${DOMAIN_NAME},https://app.${DOMAIN_NAME}
ALLOWED_HOSTS=${DOMAIN_NAME},www.${DOMAIN_NAME},app.${DOMAIN_NAME}

# Monitoring (Standard Ports)
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Performance Configuration
MAX_REQUESTS_PER_MINUTE=2000
CACHE_TTL=3600
WORKER_CONNECTIONS=1000

# Contract Testing Framework
CONTRACT_TESTING_ENABLED=true
CONTRACT_VALIDATION_ON_DEPLOY=true
CONTRACT_PERFORMANCE_THRESHOLD_MS=5

# Health Check Configuration
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10
HEALTH_CHECK_RETRIES=3
"""
        
        config_path = self.project_root / ".env.production.standardized"
        with open(config_path, 'w') as f:
            f.write(production_config)
        
        print(f"✅ Created standardized production configuration: {config_path}")
    
    def create_development_config(self) -> None:
        """Create development configuration with non-standard ports."""
        development_config = """# LeanVibe Agent Hive 2.0 - Development Configuration (Non-Standard Ports)
# ================================================================
# Uses non-standard ports to avoid conflicts with other services

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# Database Configuration (Non-Standard Development Ports)
POSTGRES_PORT=15432
REDIS_PORT=16379
DATABASE_URL=postgresql+asyncpg://leanvibe_user:leanvibe_secure_pass@localhost:15432/leanvibe_agent_hive
REDIS_URL=redis://localhost:16379/0

# API Configuration (Non-Standard Development Ports)
API_PORT=18080
PWA_DEV_PORT=18443
PWA_PREVIEW_PORT=18444

# CORS Configuration (Development)
CORS_ORIGINS=http://localhost:18080,http://localhost:18443,http://localhost:18444,http://127.0.0.1:18080
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Monitoring (Non-Standard Development Ports)
PROMETHEUS_PORT=19090
GRAFANA_PORT=13000

# Security Configuration (Development)
SECRET_KEY=development-secret-key-change-in-production-minimum-32-chars
JWT_SECRET_KEY=development-jwt-secret-key-change-in-production-minimum-64-chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Performance Configuration (Development)
MAX_REQUESTS_PER_MINUTE=100
CACHE_TTL=300
WORKER_CONNECTIONS=100

# Contract Testing Framework
CONTRACT_TESTING_ENABLED=true
CONTRACT_VALIDATION_ON_DEPLOY=false
CONTRACT_PERFORMANCE_THRESHOLD_MS=10

# Development Features
ENABLE_HOT_RELOAD=true
ENABLE_DEBUG_TOOLBAR=true
SKIP_STARTUP_INIT=false
"""
        
        config_path = self.project_root / ".env.development.standardized"
        with open(config_path, 'w') as f:
            f.write(development_config)
        
        print(f"✅ Created standardized development configuration: {config_path}")
    
    def stop_conflicting_services(self) -> None:
        """Stop Docker services that might conflict with production deployment."""
        print("🛑 Stopping potentially conflicting Docker services...")
        
        # Stop fast/development services
        try:
            subprocess.run(['docker-compose', '-f', 'docker-compose.fast.yml', 'down'], 
                         cwd=self.project_root, capture_output=True)
            print("✅ Stopped fast development services")
        except Exception as e:
            print(f"Warning: Could not stop fast services: {e}")
        
        # Stop individual conflicting containers
        conflicting_containers = [
            'leanvibe_postgres_fast',
            'leanvibe_redis_fast',
            'leanvibe_api_fast'
        ]
        
        for container in conflicting_containers:
            try:
                subprocess.run(['docker', 'stop', container], capture_output=True)
                print(f"✅ Stopped container: {container}")
            except Exception:
                pass  # Container might not exist
    
    def validate_production_readiness(self) -> bool:
        """Validate that the system is ready for production deployment."""
        print("🔍 Validating production readiness...")
        
        issues = []
        
        # Check for required configuration files
        required_files = [
            ".env.production.standardized",
            "docker-compose.production.yml"
        ]
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                issues.append(f"Missing required file: {file_path}")
        
        # Check for port availability
        critical_ports = [80, 443, 5432, 6379]
        for port in critical_ports:
            if not self.check_port_availability(port):
                issues.append(f"Port {port} is not available for production deployment")
        
        # Check Docker is running
        try:
            subprocess.run(['docker', 'info'], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            issues.append("Docker is not running")
        
        if issues:
            print("❌ Production readiness validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ Production readiness validation passed")
            return True

def main():
    parser = argparse.ArgumentParser(description="Resolve LeanVibe Agent Hive port conflicts")
    parser.add_argument("--analyze", action="store_true", help="Analyze current configuration")
    parser.add_argument("--resolve", action="store_true", help="Resolve conflicts and create standardized configs")
    parser.add_argument("--validate", action="store_true", help="Validate production readiness")
    parser.add_argument("--stop-conflicts", action="store_true", help="Stop conflicting services")
    parser.add_argument("--all", action="store_true", help="Run all operations")
    
    args = parser.parse_args()
    
    resolver = PortConflictResolver()
    
    if args.all or args.analyze:
        print("🔍 Analyzing current port configuration...")
        analysis = resolver.analyze_current_configuration()
        
        print("\n📊 Configuration Analysis:")
        print(f"Environment files found: {len(analysis['env_configs'])}")
        print(f"Running containers: {len(analysis['running_containers'])}")
        print(f"Conflicts detected: {len(analysis['conflicts'])}")
        
        if analysis['conflicts']:
            print("\n❌ Conflicts detected:")
            for conflict in analysis['conflicts']:
                print(f"   • {conflict}")
        
        if analysis['recommendations']:
            print("\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   • {rec}")
        
        print("\n📝 Running containers:")
        for name, ports in analysis['running_containers'].items():
            print(f"   • {name}: {ports}")
    
    if args.all or args.stop_conflicts:
        resolver.stop_conflicting_services()
    
    if args.all or args.resolve:
        print("\n🔧 Creating standardized configurations...")
        resolver.create_standardized_production_config()
        resolver.create_development_config()
    
    if args.all or args.validate:
        print("\n✅ Validating production readiness...")
        is_ready = resolver.validate_production_readiness()
        
        if is_ready:
            print("\n🚀 System is ready for production deployment!")
            print("Run: ./production-deployment-pipeline.sh")
        else:
            print("\n⚠️ System is not ready for production deployment.")
            print("Please resolve the issues above before deploying.")
            sys.exit(1)

if __name__ == "__main__":
    main()