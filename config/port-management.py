#!/usr/bin/env python3
"""
LeanVibe Agent Hive - Port Management Utility
Centralized port configuration and conflict detection
"""

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import json
from dataclasses import dataclass
from enum import Enum

class PortStatus(Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"
    RESERVED = "reserved"
    CONFLICT = "conflict"

@dataclass
class PortConfig:
    name: str
    port: int
    description: str
    category: str
    required: bool = True
    status: PortStatus = PortStatus.AVAILABLE

class PortManager:
    """Centralized port management for LeanVibe Agent Hive"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        # Use the actual .env.ports file in the project root
        self.config_file = config_file or self.project_root / ".env.ports"
        self.ports: Dict[str, PortConfig] = {}
        try:
            self.load_configuration()
        except FileNotFoundError:
            print(f"Warning: Port configuration file not found at {self.config_file}")
            print("Creating default configuration...")
            self.create_default_configuration()
    
    def load_configuration(self) -> None:
        """Load port configuration from .env.ports file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Port configuration file not found: {self.config_file}")
        
        # Parse .env.ports file
        env_vars = {}
        with open(self.config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Handle comments at end of line
                        value = value.split('#')[0].strip()
                        try:
                            env_vars[key.strip()] = int(value.strip())
                        except ValueError:
                            continue
        
        # Define port categories and descriptions
        port_definitions = {
            # Database Services
            'POSTGRES_PORT': ('PostgreSQL Main Database', 'database', True),
            'POSTGRES_TEST_PORT': ('PostgreSQL Test Database', 'database', False),
            'REDIS_PORT': ('Redis Main Cache', 'database', True),
            'REDIS_TEST_PORT': ('Redis Test Cache', 'database', False),
            
            # API Services
            'MAIN_API_PORT': ('Main FastAPI Server', 'api', True),
            'PROJECT_INDEX_PORT': ('Project Index API', 'api', True),
            'MOBILE_BACKEND_PORT': ('Mobile PWA Backend', 'api', False),
            'TEST_API_PORT': ('Test API Server', 'api', False),
            
            # Frontend Services
            'FRONTEND_DEV_PORT': ('Frontend Development Server', 'frontend', False),
            'DASHBOARD_PORT': ('Main Dashboard', 'frontend', True),
            'PWA_DEV_PORT': ('PWA Development Server', 'frontend', False),
            
            # Monitoring Services
            'PROMETHEUS_PORT': ('Prometheus Metrics', 'monitoring', True),
            'GRAFANA_PORT': ('Grafana Dashboard', 'monitoring', True),
            'ALERTMANAGER_PORT': ('AlertManager', 'monitoring', False),
            
            # Exporters
            'POSTGRES_EXPORTER_PORT': ('PostgreSQL Exporter', 'exporters', False),
            'REDIS_EXPORTER_PORT': ('Redis Exporter', 'exporters', False),
            'NODE_EXPORTER_PORT': ('Node Exporter', 'exporters', False),
            'CADVISOR_PORT': ('cAdvisor Container Metrics', 'exporters', False),
            
            # Management Tools
            'PGADMIN_PORT': ('pgAdmin Database Management', 'management', False),
            'REDIS_INSIGHT_PORT': ('Redis Insight', 'management', False),
            'JUPYTER_PORT': ('Jupyter Notebooks', 'management', False),
            
            # Logging Stack
            'ELASTICSEARCH_PORT': ('Elasticsearch', 'logging', False),
            'LOGSTASH_BEATS_PORT': ('Logstash Beats Input', 'logging', False),
            'LOGSTASH_TCP_PORT': ('Logstash TCP Input', 'logging', False),
            'LOGSTASH_API_PORT': ('Logstash API', 'logging', False),
            'KIBANA_PORT': ('Kibana Log Visualization', 'logging', False),
            
            # Web Server
            'HTTP_PORT': ('HTTP Server', 'web', False),
            'HTTPS_PORT': ('HTTPS Server', 'web', False),
            'NGINX_HTTP_PORT': ('Nginx HTTP Proxy', 'web', False),
            'NGINX_HTTPS_PORT': ('Nginx HTTPS Proxy', 'web', False),
        }
        
        # Create PortConfig objects
        for env_key, (description, category, required) in port_definitions.items():
            if env_key in env_vars:
                port_num = env_vars[env_key]
                self.ports[env_key] = PortConfig(
                    name=env_key,
                    port=port_num,
                    description=description,
                    category=category,
                    required=required
                )
    
    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available (not in use)"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result != 0  # Port is available if connection failed
        except Exception:
            return True  # Assume available if check fails
    
    def scan_all_ports(self) -> Dict[str, PortStatus]:
        """Scan all configured ports and return their status"""
        results = {}
        for name, config in self.ports.items():
            if self.check_port_availability(config.port):
                config.status = PortStatus.AVAILABLE
            else:
                config.status = PortStatus.IN_USE
            results[name] = config.status
        return results
    
    def find_conflicts(self) -> List[Tuple[str, str]]:
        """Find port conflicts between different services"""
        conflicts = []
        port_to_service = {}
        
        for name, config in self.ports.items():
            port = config.port
            if port in port_to_service:
                conflicts.append((port_to_service[port], name))
            else:
                port_to_service[port] = name
        
        return conflicts
    
    def get_available_port(self, start_port: int = 8000, max_attempts: int = 100) -> int:
        """Find next available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if self.check_port_availability(port):
                # Make sure it's not already configured for another service
                if not any(config.port == port for config in self.ports.values()):
                    return port
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")
    
    def suggest_port_fixes(self) -> Dict[str, int]:
        """Suggest port changes to resolve conflicts"""
        conflicts = self.find_conflicts()
        suggestions = {}
        
        for service1, service2 in conflicts:
            # Suggest new port for the second service
            config = self.ports[service2]
            try:
                new_port = self.get_available_port(config.port + 1)
                suggestions[service2] = new_port
            except RuntimeError:
                # Try a different range
                new_port = self.get_available_port(config.port + 1000)
                suggestions[service2] = new_port
        
        return suggestions
    
    def generate_env_file(self, output_file: Optional[str] = None) -> str:
        """Generate .env file with current port configuration"""
        output_file = output_file or str(self.project_root / ".env.ports.generated")
        
        content = [
            "# Generated LeanVibe Agent Hive Port Configuration",
            "# Auto-generated by port-management.py",
            "",
        ]
        
        # Group by category
        categories = {}
        for config in self.ports.values():
            if config.category not in categories:
                categories[config.category] = []
            categories[config.category].append(config)
        
        for category, configs in categories.items():
            content.append(f"# {category.title()} Services")
            for config in sorted(configs, key=lambda x: x.port):
                status_indicator = "✓" if config.status == PortStatus.AVAILABLE else "✗"
                content.append(f"{config.name}={config.port}  # {config.description} {status_indicator}")
            content.append("")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))
        
        return output_file
    
    def export_docker_compose_vars(self) -> Dict[str, str]:
        """Export port configuration for docker-compose environment"""
        return {name: str(config.port) for name, config in self.ports.items()}
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current port configuration"""
        issues = []
        
        # Check for conflicts
        conflicts = self.find_conflicts()
        if conflicts:
            for service1, service2 in conflicts:
                port = self.ports[service1].port
                issues.append(f"Port conflict: {service1} and {service2} both use port {port}")
        
        # Check for well-known port usage
        well_known_ports = {
            80: "HTTP",
            443: "HTTPS", 
            22: "SSH",
            25: "SMTP",
            53: "DNS",
            110: "POP3",
            143: "IMAP",
            993: "IMAPS",
            995: "POP3S"
        }
        
        for name, config in self.ports.items():
            if config.port in well_known_ports:
                service = well_known_ports[config.port]
                if not (name.endswith('_HTTP_PORT') or name.endswith('_HTTPS_PORT')):
                    issues.append(f"Warning: {name} uses well-known port {config.port} ({service})")
        
        # Check required services
        for name, config in self.ports.items():
            if config.required and config.status == PortStatus.IN_USE:
                issues.append(f"Required service {name} port {config.port} is in use")
        
        return len(issues) == 0, issues
    
    def generate_status_report(self) -> str:
        """Generate comprehensive status report"""
        self.scan_all_ports()
        is_valid, issues = self.validate_configuration()
        
        report = [
            "🔧 LeanVibe Agent Hive - Port Configuration Status",
            "=" * 60,
            "",
            f"📊 Total configured ports: {len(self.ports)}",
            f"✅ Configuration valid: {'Yes' if is_valid else 'No'}",
            "",
        ]
        
        if issues:
            report.extend([
                "⚠️  Issues Found:",
                *[f"   • {issue}" for issue in issues],
                "",
            ])
        
        # Group by status
        by_status = {}
        for config in self.ports.values():
            status = config.status
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(config)
        
        for status, configs in by_status.items():
            if not configs:
                continue
                
            icon = {"available": "✅", "in_use": "🔴", "reserved": "🟡", "conflict": "⚠️"}[status.value]
            report.append(f"{icon} {status.value.title()} Ports ({len(configs)}):")
            
            for config in sorted(configs, key=lambda x: x.port):
                required_tag = " [REQUIRED]" if config.required else ""
                report.append(f"   {config.port:5d} - {config.description}{required_tag}")
            report.append("")
        
        # Category breakdown
        categories = {}
        for config in self.ports.values():
            if config.category not in categories:
                categories[config.category] = []
            categories[config.category].append(config)
        
        report.extend([
            "📋 By Category:",
            "",
        ])
        
        for category, configs in sorted(categories.items()):
            report.append(f"🔹 {category.title()}: {len(configs)} ports")
            for config in sorted(configs, key=lambda x: x.port):
                status_icon = {"available": "✅", "in_use": "🔴"}[config.status.value]
                report.append(f"   {config.port:5d} {status_icon} {config.description}")
            report.append("")
        
        return '\n'.join(report)
    
    def create_default_configuration(self) -> None:
        """Create a default port configuration if none exists"""
        # Load the default port definitions directly
        port_definitions = {
            # Database Services - Non-standard ports to avoid conflicts
            'POSTGRES_PORT': (5433, 'PostgreSQL Main Database', 'database', True),
            'POSTGRES_TEST_PORT': (5434, 'PostgreSQL Test Database', 'database', False),
            'REDIS_PORT': (6380, 'Redis Main Cache', 'database', True),
            'REDIS_TEST_PORT': (6381, 'Redis Test Cache', 'database', False),
            
            # API Services - Non-standard ports to avoid conflicts
            'MAIN_API_PORT': (8100, 'Main FastAPI Server', 'api', True),
            'PROJECT_INDEX_PORT': (8101, 'Project Index API', 'api', True),
            'MOBILE_BACKEND_PORT': (8102, 'Mobile PWA Backend', 'api', False),
            'TEST_API_PORT': (8103, 'Test API Server', 'api', False),
            
            # Frontend Services
            'FRONTEND_DEV_PORT': (5173, 'Frontend Development Server', 'frontend', False),
            'DASHBOARD_PORT': (3001, 'Main Dashboard', 'frontend', True),
            'PWA_DEV_PORT': (3002, 'PWA Development Server', 'frontend', False),
            
            # Monitoring Services
            'PROMETHEUS_PORT': (9090, 'Prometheus Metrics', 'monitoring', True),
            'GRAFANA_PORT': (3101, 'Grafana Dashboard', 'monitoring', True),
            'ALERTMANAGER_PORT': (9093, 'AlertManager', 'monitoring', False),
            
            # Exporters
            'POSTGRES_EXPORTER_PORT': (9187, 'PostgreSQL Exporter', 'exporters', False),
            'REDIS_EXPORTER_PORT': (9121, 'Redis Exporter', 'exporters', False),
            'NODE_EXPORTER_PORT': (9100, 'Node Exporter', 'exporters', False),
            'CADVISOR_PORT': (8180, 'cAdvisor Container Metrics', 'exporters', False),  # Changed from 8080 to avoid conflicts
            
            # Management Tools
            'PGADMIN_PORT': (5150, 'pgAdmin Database Management', 'management', False),
            'REDIS_INSIGHT_PORT': (8201, 'Redis Insight', 'management', False),
            'JUPYTER_PORT': (8888, 'Jupyter Notebooks', 'management', False),
            
            # Logging Stack
            'ELASTICSEARCH_PORT': (9200, 'Elasticsearch', 'logging', False),
            'LOGSTASH_BEATS_PORT': (5044, 'Logstash Beats Input', 'logging', False),
            'LOGSTASH_TCP_PORT': (5000, 'Logstash TCP Input', 'logging', False),
            'LOGSTASH_API_PORT': (9600, 'Logstash API', 'logging', False),
            'KIBANA_PORT': (5601, 'Kibana Log Visualization', 'logging', False),
            
            # Web Server - Non-standard HTTPS to avoid conflicts
            'HTTP_PORT': (80, 'HTTP Server', 'web', False),
            'HTTPS_PORT': (443, 'HTTPS Server', 'web', False),
            'NGINX_HTTP_PORT': (8000, 'Nginx HTTP Proxy', 'web', False),
            'NGINX_HTTPS_PORT': (8443, 'Nginx HTTPS Proxy', 'web', False),
        }
        
        # Create PortConfig objects
        for env_key, (port_num, description, category, required) in port_definitions.items():
            self.ports[env_key] = PortConfig(
                name=env_key,
                port=port_num,
                description=description,
                category=category,
                required=required
            )

def main():
    """CLI interface for port management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LeanVibe Agent Hive Port Management")
    parser.add_argument("--scan", action="store_true", help="Scan all configured ports")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--report", action="store_true", help="Generate status report")
    parser.add_argument("--fix", action="store_true", help="Suggest fixes for conflicts")
    parser.add_argument("--generate-env", help="Generate .env file")
    parser.add_argument("--config", help="Port configuration file")
    
    args = parser.parse_args()
    
    try:
        manager = PortManager(args.config)
        
        if args.scan:
            print("🔍 Scanning ports...")
            results = manager.scan_all_ports()
            for name, status in results.items():
                config = manager.ports[name]
                status_icon = {"available": "✅", "in_use": "🔴"}[status.value]
                print(f"{config.port:5d} {status_icon} {config.description}")
        
        elif args.validate:
            is_valid, issues = manager.validate_configuration()
            if is_valid:
                print("✅ Port configuration is valid!")
            else:
                print("❌ Port configuration has issues:")
                for issue in issues:
                    print(f"   • {issue}")
                sys.exit(1)
        
        elif args.report:
            print(manager.generate_status_report())
        
        elif args.fix:
            conflicts = manager.find_conflicts()
            if not conflicts:
                print("✅ No port conflicts found!")
            else:
                print("⚠️  Port conflicts detected. Suggested fixes:")
                suggestions = manager.suggest_port_fixes()
                for service, new_port in suggestions.items():
                    old_port = manager.ports[service].port
                    print(f"   {service}: {old_port} → {new_port}")
        
        elif args.generate_env:
            output_file = manager.generate_env_file(args.generate_env)
            print(f"✅ Generated port configuration: {output_file}")
        
        else:
            print(manager.generate_status_report())
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()