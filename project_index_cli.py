#!/usr/bin/env python3
"""
Project Index Universal CLI Installer

A comprehensive, user-friendly installer that provides seamless one-command 
installation of the Project Index system for any codebase.

Usage:
    project-index install          # Interactive installation wizard
    project-index install --quick  # One-command setup with defaults
    project-index configure       # Modify existing configuration
    project-index validate        # Validate installation
    project-index troubleshoot    # Diagnose and fix issues
    project-index status          # Show system status
    project-index update          # Update to latest version
    project-index uninstall       # Remove installation
"""

import sys
import os
import argparse
import json
import time
import subprocess
import platform
import shutil
import tempfile
import getpass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
from datetime import datetime

# Color and formatting utilities
class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    @staticmethod
    def colored(text: str, color: str) -> str:
        """Return colored text"""
        return f"{color}{text}{Colors.RESET}"
    
    @staticmethod
    def success(text: str) -> str:
        return Colors.colored(f"✅ {text}", Colors.GREEN)
    
    @staticmethod
    def error(text: str) -> str:
        return Colors.colored(f"❌ {text}", Colors.RED)
    
    @staticmethod
    def warning(text: str) -> str:
        return Colors.colored(f"⚠️  {text}", Colors.YELLOW)
    
    @staticmethod
    def info(text: str) -> str:
        return Colors.colored(f"ℹ️  {text}", Colors.BLUE)
    
    @staticmethod
    def step(text: str) -> str:
        return Colors.colored(f"🔄 {text}", Colors.CYAN)
    
    @staticmethod
    def header(text: str) -> str:
        return Colors.colored(f"\n{'='*len(text)}\n{text}\n{'='*len(text)}\n", Colors.PURPLE + Colors.BOLD)

class InstallationProfile(Enum):
    """Installation profiles for different project sizes"""
    SMALL = "small"
    MEDIUM = "medium" 
    LARGE = "large"
    ENTERPRISE = "enterprise"

@dataclass
class ProfileConfig:
    """Configuration for installation profiles"""
    name: str
    description: str
    max_files: int
    developers: str
    memory_gb: float
    cpu_cores: float
    services: List[str]
    features: List[str]

@dataclass
class SystemRequirements:
    """System requirements and capabilities"""
    os_type: str
    os_version: str
    arch: str
    memory_gb: float
    cpu_cores: int
    disk_gb: float
    docker_available: bool
    docker_compose_available: bool
    python_version: str
    node_available: bool

@dataclass
class InstallationConfig:
    """Installation configuration"""
    profile: InstallationProfile
    project_path: str
    project_name: str
    detected_frameworks: List[str]
    host_port_api: int
    host_port_dashboard: int
    host_port_metrics: int
    database_password: str
    redis_password: str
    api_key: Optional[str]
    enable_monitoring: bool
    enable_enterprise_features: bool
    auto_start: bool
    installation_id: str

class ProgressBar:
    """Simple progress bar implementation"""
    
    def __init__(self, total: int, description: str = "Progress", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
    
    def update(self, amount: int = 1, description: Optional[str] = None):
        """Update progress bar"""
        self.current += amount
        if description:
            self.description = description
        self._render()
    
    def set_progress(self, current: int, description: Optional[str] = None):
        """Set absolute progress"""
        self.current = current
        if description:
            self.description = description
        self._render()
    
    def _render(self):
        """Render the progress bar"""
        percentage = min(100, (self.current / self.total) * 100)
        filled = int(self.width * self.current / self.total)
        bar = '█' * filled + '░' * (self.width - filled)
        elapsed = time.time() - self.start_time
        
        print(f"\r{Colors.CYAN}[{bar}]{Colors.RESET} {percentage:.1f}% | {self.description} | {elapsed:.1f}s", end="", flush=True)
    
    def complete(self, description: str = "Complete"):
        """Mark progress as complete"""
        self.current = self.total
        self.description = description
        self._render()
        print()  # New line

class ProjectIndexCLI:
    """Main CLI application class"""
    
    PROFILES = {
        InstallationProfile.SMALL: ProfileConfig(
            name="Small",
            description="< 1k files, 1 developer",
            max_files=1000,
            developers="1 developer",
            memory_gb=1.0,
            cpu_cores=0.5,
            services=["api", "database", "cache"],
            features=["basic indexing", "simple validation"]
        ),
        InstallationProfile.MEDIUM: ProfileConfig(
            name="Medium", 
            description="1k-10k files, 2-5 developers",
            max_files=10000,
            developers="2-5 developers",
            memory_gb=2.0,
            cpu_cores=1.0,
            services=["api", "database", "cache", "monitoring"],
            features=["advanced indexing", "team coordination", "performance optimization"]
        ),
        InstallationProfile.LARGE: ProfileConfig(
            name="Large",
            description="> 10k files, team development", 
            max_files=100000,
            developers="5+ developers",
            memory_gb=4.0,
            cpu_cores=2.0,
            services=["api", "database", "cache", "monitoring", "analytics"],
            features=["enterprise features", "advanced analytics", "high availability"]
        ),
        InstallationProfile.ENTERPRISE: ProfileConfig(
            name="Enterprise",
            description="Enterprise-grade deployment",
            max_files=1000000,
            developers="Large teams",
            memory_gb=8.0,
            cpu_cores=4.0,
            services=["api", "database", "cache", "monitoring", "analytics", "security", "compliance"],
            features=["full enterprise features", "compliance", "advanced security", "clustering"]
        )
    }
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.config_dir = Path.home() / ".project-index"
        self.config_file = self.config_dir / "config.json"
        self.ensure_config_dir()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("project-index")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "installer.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def ensure_config_dir(self):
        """Ensure configuration directory exists"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def show_banner(self):
        """Display the application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🚀 Project Index Universal CLI Installer                          ║
║                                                                      ║
║   Intelligent project analysis and context optimization             ║
║   for any codebase in under 5 minutes                              ║
║                                                                      ║
║   🎯 One command to rule them all!                                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """
        print(Colors.colored(banner, Colors.PURPLE + Colors.BOLD))
    
    def get_system_info(self) -> SystemRequirements:
        """Gather system information and requirements"""
        print(Colors.step("Gathering system information..."))
        
        # Operating system
        os_type = platform.system().lower()
        os_version = platform.release()
        arch = platform.machine()
        
        # Memory (approximate)
        try:
            if os_type == "linux":
                with open("/proc/meminfo", "r") as f:
                    mem_total = int([line for line in f if "MemTotal" in line][0].split()[1])
                    memory_gb = mem_total / 1024 / 1024
            elif os_type == "darwin":
                result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
                memory_gb = int(result.stdout.strip()) / 1024 / 1024 / 1024
            else:
                memory_gb = 4.0  # Default assumption
        except:
            memory_gb = 4.0
        
        # CPU cores
        cpu_cores = os.cpu_count() or 2
        
        # Disk space (current directory)
        disk_usage = shutil.disk_usage(".")
        disk_gb = disk_usage.free / 1024 / 1024 / 1024
        
        # Docker availability
        docker_available = shutil.which("docker") is not None
        if docker_available:
            try:
                subprocess.run(["docker", "info"], capture_output=True, check=True)
            except subprocess.CalledProcessError:
                docker_available = False
        
        # Docker Compose availability
        docker_compose_available = False
        if docker_available:
            try:
                result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
                docker_compose_available = result.returncode == 0
            except:
                pass
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Node.js availability
        node_available = shutil.which("node") is not None
        
        return SystemRequirements(
            os_type=os_type,
            os_version=os_version,
            arch=arch,
            memory_gb=memory_gb,
            cpu_cores=cpu_cores,
            disk_gb=disk_gb,
            docker_available=docker_available,
            docker_compose_available=docker_compose_available,
            python_version=python_version,
            node_available=node_available
        )
    
    def validate_system_requirements(self, requirements: SystemRequirements) -> Tuple[bool, List[str]]:
        """Validate system meets minimum requirements"""
        print(Colors.step("Validating system requirements..."))
        
        errors = []
        warnings = []
        
        # Operating system
        if requirements.os_type not in ["linux", "darwin", "windows"]:
            errors.append(f"Unsupported operating system: {requirements.os_type}")
        
        # Memory
        if requirements.memory_gb < 1.0:
            errors.append(f"Insufficient memory: {requirements.memory_gb:.1f}GB (minimum 1GB required)")
        elif requirements.memory_gb < 2.0:
            warnings.append(f"Low memory detected: {requirements.memory_gb:.1f}GB (recommended 2GB+)")
        
        # Disk space
        if requirements.disk_gb < 1.0:
            errors.append(f"Insufficient disk space: {requirements.disk_gb:.1f}GB (minimum 1GB required)")
        
        # Docker
        if not requirements.docker_available:
            errors.append("Docker is not installed or not accessible")
        
        if not requirements.docker_compose_available:
            errors.append("Docker Compose V2 is not available")
        
        # Python version
        major, minor = map(int, requirements.python_version.split('.')[:2])
        if major < 3 or (major == 3 and minor < 8):
            errors.append(f"Python 3.8+ required, found {requirements.python_version}")
        
        # Show warnings
        for warning in warnings:
            print(Colors.warning(warning))
        
        # Show system info
        print(Colors.info(f"System: {requirements.os_type} {requirements.os_version} ({requirements.arch})"))
        print(Colors.info(f"Resources: {requirements.memory_gb:.1f}GB RAM, {requirements.cpu_cores} CPU cores"))
        print(Colors.info(f"Available disk space: {requirements.disk_gb:.1f}GB"))
        print(Colors.info(f"Python: {requirements.python_version}"))
        
        if errors:
            print(Colors.error("System validation failed:"))
            for error in errors:
                print(f"  - {error}")
            return False, errors
        else:
            print(Colors.success("System requirements satisfied"))
            return True, []
    
    def interactive_installation_wizard(self, project_path: Optional[str] = None) -> InstallationConfig:
        """Interactive installation wizard"""
        print(Colors.header("🧙‍♂️ Interactive Installation Wizard"))
        print(Colors.info("Let's configure your Project Index installation step by step."))
        print()
        
        # Project path selection
        if not project_path:
            while True:
                project_path = input(Colors.colored("📁 Enter the path to your project: ", Colors.CYAN)).strip()
                project_path = os.path.expanduser(project_path)
                
                if not project_path:
                    print(Colors.error("Project path cannot be empty"))
                    continue
                
                if not os.path.exists(project_path):
                    print(Colors.error(f"Path does not exist: {project_path}"))
                    continue
                
                if not os.path.isdir(project_path):
                    print(Colors.error(f"Path is not a directory: {project_path}"))
                    continue
                
                break
        
        project_path = os.path.abspath(project_path)
        project_name = os.path.basename(project_path)
        
        print(Colors.success(f"Project: {project_name}"))
        print(Colors.info(f"Path: {project_path}"))
        print()
        
        # Detect project characteristics
        print(Colors.step("Analyzing project..."))
        detected_frameworks = self.detect_project_frameworks(project_path)
        file_count = self.count_project_files(project_path)
        
        print(Colors.info(f"Files detected: ~{file_count:,}"))
        if detected_frameworks:
            print(Colors.info(f"Frameworks: {', '.join(detected_frameworks)}"))
        print()
        
        # Profile selection
        recommended_profile = self.recommend_profile(file_count, detected_frameworks)
        print(Colors.info(f"Recommended profile: {Colors.colored(recommended_profile.value, Colors.GREEN + Colors.BOLD)}"))
        print()
        
        print("Available profiles:")
        for profile, config in self.PROFILES.items():
            marker = "👉 " if profile == recommended_profile else "   "
            color = Colors.GREEN if profile == recommended_profile else Colors.WHITE
            print(f"{marker}{Colors.colored(f'{config.name:10}', color)} | {config.description:25} | {config.memory_gb}GB RAM, {config.cpu_cores} CPU")
        print()
        
        while True:
            choice = input(Colors.colored(f"Select profile [{recommended_profile.value}]: ", Colors.CYAN)).strip().lower()
            if not choice:
                selected_profile = recommended_profile
                break
            
            try:
                selected_profile = InstallationProfile(choice)
                break
            except ValueError:
                print(Colors.error(f"Invalid profile: {choice}. Choose from: {', '.join(p.value for p in InstallationProfile)}"))
        
        print(Colors.success(f"Selected profile: {selected_profile.value}"))
        print()
        
        # Port configuration
        print(Colors.step("Configuring network ports..."))
        
        host_port_api = self.get_available_port(8100, "API")
        host_port_dashboard = self.get_available_port(8101, "Dashboard") 
        host_port_metrics = self.get_available_port(9090, "Metrics")
        
        print(Colors.info(f"API port: {host_port_api}"))
        print(Colors.info(f"Dashboard port: {host_port_dashboard}"))
        print(Colors.info(f"Metrics port: {host_port_metrics}"))
        print()
        
        # Features configuration
        enable_monitoring = self.ask_yes_no("Enable monitoring and metrics?", default=True)
        enable_enterprise_features = self.ask_yes_no("Enable enterprise features?", default=False)
        auto_start = self.ask_yes_no("Start services after installation?", default=True)
        
        # Generate secure passwords
        database_password = self.generate_secure_password()
        redis_password = self.generate_secure_password()
        
        # Installation ID for tracking
        installation_id = f"pi-{int(time.time())}-{os.getlogin()}"
        
        config = InstallationConfig(
            profile=selected_profile,
            project_path=project_path,
            project_name=project_name,
            detected_frameworks=detected_frameworks,
            host_port_api=host_port_api,
            host_port_dashboard=host_port_dashboard,
            host_port_metrics=host_port_metrics,
            database_password=database_password,
            redis_password=redis_password,
            api_key=None,
            enable_monitoring=enable_monitoring,
            enable_enterprise_features=enable_enterprise_features,
            auto_start=auto_start,
            installation_id=installation_id
        )
        
        # Show configuration summary
        self.show_configuration_summary(config)
        
        if not self.ask_yes_no("Proceed with installation?", default=True):
            print(Colors.info("Installation cancelled by user"))
            sys.exit(0)
        
        return config
    
    def detect_project_frameworks(self, project_path: str) -> List[str]:
        """Detect frameworks and technologies in the project"""
        frameworks = []
        
        # Check for common files and patterns
        files_to_check = {
            "package.json": ["Node.js", "JavaScript"],
            "requirements.txt": ["Python"],
            "Pipfile": ["Python", "Pipenv"],
            "pyproject.toml": ["Python"],
            "Cargo.toml": ["Rust"],
            "go.mod": ["Go"],
            "pom.xml": ["Java", "Maven"],
            "build.gradle": ["Java", "Gradle"],
            "composer.json": ["PHP"],
            "Gemfile": ["Ruby"],
            "pubspec.yaml": ["Flutter", "Dart"],
            "yarn.lock": ["Node.js", "Yarn"],
            "package-lock.json": ["Node.js", "npm"],
            "docker-compose.yml": ["Docker"],
            "Dockerfile": ["Docker"],
            "kubernetes": ["Kubernetes"],
            "terraform": ["Terraform"],
            ".env": ["Environment Variables"],
            "tsconfig.json": ["TypeScript"],
            "angular.json": ["Angular"],
            "vue.config.js": ["Vue.js"],
            "nuxt.config.js": ["Nuxt.js"],
            "next.config.js": ["Next.js"],
            "svelte.config.js": ["Svelte"],
            "webpack.config.js": ["Webpack"],
            "vite.config.js": ["Vite"],
            "rollup.config.js": ["Rollup"],
            "tailwind.config.js": ["Tailwind CSS"],
            "jest.config.js": ["Jest"],
            "vitest.config.js": ["Vitest"],
            "cypress.json": ["Cypress"],
            "playwright.config.js": ["Playwright"],
            ".github/workflows": ["GitHub Actions"],
            ".gitlab-ci.yml": ["GitLab CI"],
            "Jenkinsfile": ["Jenkins"],
            "serverless.yml": ["Serverless Framework"],
            "amplify.yml": ["AWS Amplify"],
        }
        
        for file_pattern, techs in files_to_check.items():
            file_path = Path(project_path) / file_pattern
            if file_path.exists():
                frameworks.extend(techs)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(frameworks))
    
    def count_project_files(self, project_path: str) -> int:
        """Count files in the project (excluding common ignore patterns)"""
        ignore_patterns = {
            ".git", "node_modules", "__pycache__", ".pytest_cache", 
            "venv", ".venv", "env", ".env", "dist", "build", 
            "target", ".idea", ".vscode", "*.pyc", "*.pyo", 
            "*.pyd", ".DS_Store", "Thumbs.db", "coverage", 
            ".coverage", "*.egg-info", ".tox", ".mypy_cache"
        }
        
        file_count = 0
        try:
            for root, dirs, files in os.walk(project_path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if d not in ignore_patterns]
                
                for file in files:
                    if not any(pattern in file for pattern in ignore_patterns if "*" in pattern):
                        file_count += 1
                
                # Limit scan for performance
                if file_count > 100000:
                    break
        except:
            file_count = 1000  # Default estimate
        
        return file_count
    
    def recommend_profile(self, file_count: int, frameworks: List[str]) -> InstallationProfile:
        """Recommend installation profile based on project characteristics"""
        # Base recommendation on file count
        if file_count < 1000:
            base_profile = InstallationProfile.SMALL
        elif file_count < 10000:
            base_profile = InstallationProfile.MEDIUM
        else:
            base_profile = InstallationProfile.LARGE
        
        # Adjust based on frameworks
        complex_frameworks = {
            "Kubernetes", "Docker", "Terraform", "AWS Amplify", 
            "Serverless Framework", "Angular", "React", "Vue.js"
        }
        
        if any(fw in complex_frameworks for fw in frameworks):
            if base_profile == InstallationProfile.SMALL:
                base_profile = InstallationProfile.MEDIUM
            elif base_profile == InstallationProfile.MEDIUM:
                base_profile = InstallationProfile.LARGE
        
        return base_profile
    
    def get_available_port(self, preferred_port: int, service_name: str) -> int:
        """Find an available port, starting with the preferred port"""
        import socket
        
        for port in range(preferred_port, preferred_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return port
                except OSError:
                    continue
        
        # Fallback to any available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            return s.getsockname()[1]
    
    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question with default"""
        default_str = "Y/n" if default else "y/N"
        response = input(Colors.colored(f"{question} [{default_str}]: ", Colors.CYAN)).strip().lower()
        
        if not response:
            return default
        
        return response in ["y", "yes", "true", "1"]
    
    def generate_secure_password(self, length: int = 32) -> str:
        """Generate a secure random password"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def show_configuration_summary(self, config: InstallationConfig):
        """Display configuration summary"""
        print(Colors.header("📋 Installation Configuration Summary"))
        
        profile_config = self.PROFILES[config.profile]
        
        print(f"Project: {Colors.colored(config.project_name, Colors.BOLD)}")
        print(f"Path: {config.project_path}")
        print(f"Profile: {Colors.colored(profile_config.name, Colors.GREEN)} ({profile_config.description})")
        print(f"Resources: {profile_config.memory_gb}GB RAM, {profile_config.cpu_cores} CPU cores")
        
        if config.detected_frameworks:
            print(f"Frameworks: {', '.join(config.detected_frameworks)}")
        
        print(f"\nNetwork Configuration:")
        print(f"  API: http://localhost:{config.host_port_api}")
        print(f"  Dashboard: http://localhost:{config.host_port_dashboard}")
        if config.enable_monitoring:
            print(f"  Metrics: http://localhost:{config.host_port_metrics}")
        
        print(f"\nFeatures:")
        print(f"  Monitoring: {'✅' if config.enable_monitoring else '❌'}")
        print(f"  Enterprise: {'✅' if config.enable_enterprise_features else '❌'}")
        print(f"  Auto-start: {'✅' if config.auto_start else '❌'}")
        print()

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Project Index Universal CLI Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  project-index install                    # Interactive installation wizard
  project-index install --quick           # Quick install with defaults
  project-index install --path /my/repo   # Install for specific project
  project-index configure                 # Modify configuration
  project-index validate                  # Validate installation
  project-index status                    # Show system status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install Project Index")
    install_parser.add_argument("--quick", action="store_true", help="Quick install with defaults")
    install_parser.add_argument("--path", help="Project path")
    install_parser.add_argument("--profile", choices=[p.value for p in InstallationProfile], help="Installation profile")
    install_parser.add_argument("--no-start", action="store_true", help="Don't start services after installation")
    install_parser.add_argument("--yes", action="store_true", help="Accept all defaults (non-interactive)")
    
    # Other commands (stubs for now)
    subparsers.add_parser("configure", help="Modify configuration")
    subparsers.add_parser("validate", help="Validate installation") 
    subparsers.add_parser("troubleshoot", help="Diagnose and fix issues")
    subparsers.add_parser("status", help="Show system status")
    subparsers.add_parser("update", help="Update to latest version")
    subparsers.add_parser("uninstall", help="Remove installation")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ProjectIndexCLI()
    
    if args.command == "install":
        cli.show_banner()
        
        # Get system information
        system_info = cli.get_system_info()
        
        # Validate system requirements
        valid, errors = cli.validate_system_requirements(system_info)
        if not valid:
            print(Colors.error("Cannot proceed with installation due to system requirement failures."))
            print(Colors.info("Please address the issues above and try again."))
            sys.exit(1)
        
        # Run installation wizard
        config = cli.interactive_installation_wizard(args.path)
        
        # TODO: Continue with actual installation process
        print(Colors.success("Installation configuration complete!"))
        print(Colors.info("Next: Implementing Docker infrastructure automation..."))
        
    else:
        print(Colors.warning(f"Command '{args.command}' is not yet implemented."))
        print(Colors.info("Available commands: install"))

if __name__ == "__main__":
    main()