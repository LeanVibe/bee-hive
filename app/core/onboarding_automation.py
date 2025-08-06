"""
Zero-Setup Development Environment with Guided Onboarding

Provides automated project setup, intelligent templates, and guided learning experiences
to reduce developer onboarding time from hours to minutes.
"""

import os
import sys
import json
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.tree import Tree

console = Console()


@dataclass
class ProjectTemplate:
    """Represents a project template with setup instructions."""
    name: str
    description: str
    category: str
    technologies: List[str]
    setup_commands: List[str]
    template_files: Dict[str, str]
    learning_resources: List[str]
    estimated_setup_time: int  # in minutes


@dataclass
class DevelopmentEnvironment:
    """Represents a detected or configured development environment."""
    os_type: str
    python_version: str
    node_version: Optional[str]
    docker_available: bool
    git_configured: bool
    editor_detected: Optional[str]
    shell_type: str


class EnvironmentDetector:
    """Detects current development environment for smart configuration."""
    
    def detect_environment(self) -> DevelopmentEnvironment:
        """Detect current development environment capabilities."""
        env = DevelopmentEnvironment(
            os_type=self._detect_os(),
            python_version=self._detect_python(),
            node_version=self._detect_node(),
            docker_available=self._check_docker(),
            git_configured=self._check_git(),
            editor_detected=self._detect_editor(),
            shell_type=self._detect_shell()
        )
        return env
    
    def _detect_os(self) -> str:
        """Detect operating system."""
        if sys.platform.startswith('win'):
            return "windows"
        elif sys.platform.startswith('darwin'):
            return "macos"
        elif sys.platform.startswith('linux'):
            return "linux"
        else:
            return "unknown"
    
    def _detect_python(self) -> str:
        """Detect Python version."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _detect_node(self) -> Optional[str]:
        """Detect Node.js version if available."""
        try:
            result = subprocess.run(
                ["node", "--version"], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_git(self) -> bool:
        """Check if Git is configured."""
        try:
            result = subprocess.run(
                ["git", "config", "--get", "user.name"], 
                capture_output=True, timeout=5
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except:
            return False
    
    def _detect_editor(self) -> Optional[str]:
        """Detect preferred code editor."""
        editors = {
            "code": "VS Code",
            "cursor": "Cursor",
            "subl": "Sublime Text",
            "atom": "Atom",
            "vim": "Vim",
            "nvim": "Neovim"
        }
        
        for command, name in editors.items():
            try:
                result = subprocess.run(
                    ["which", command] if sys.platform != "win32" else ["where", command],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return name
            except:
                continue
        
        return None
    
    def _detect_shell(self) -> str:
        """Detect shell type."""
        shell = os.environ.get('SHELL', '')
        if 'zsh' in shell:
            return "zsh"
        elif 'bash' in shell:
            return "bash"
        elif 'fish' in shell:
            return "fish"
        else:
            return "unknown"


class ProjectTemplateManager:
    """Manages project templates and smart initialization."""
    
    def __init__(self):
        self.templates = self._load_builtin_templates()
    
    def _load_builtin_templates(self) -> Dict[str, ProjectTemplate]:
        """Load built-in project templates."""
        templates = {}
        
        # AI-Enhanced Web API Template
        templates["ai_web_api"] = ProjectTemplate(
            name="AI-Enhanced Web API",
            description="FastAPI + PostgreSQL + Redis with LeanVibe agent integration",
            category="backend",
            technologies=["Python", "FastAPI", "PostgreSQL", "Redis", "LeanVibe"],
            setup_commands=[
                "python -m venv venv",
                "source venv/bin/activate",
                "pip install -e .",
                "alembic upgrade head",
                "uvicorn app.main:app --reload"
            ],
            template_files={
                "app/main.py": self._get_fastapi_main_template(),
                "app/models/__init__.py": "",
                "app/api/__init__.py": "",
                "requirements.txt": self._get_requirements_template(),
                "alembic.ini": self._get_alembic_template(),
                ".env.example": self._get_env_template(),
                "README.md": self._get_readme_template("AI-Enhanced Web API"),
            },
            learning_resources=[
                "FastAPI Documentation: https://fastapi.tiangolo.com/",
                "LeanVibe Integration Guide: /docs/integration/",
                "PostgreSQL Best Practices: /docs/database/"
            ],
            estimated_setup_time=8
        )
        
        # Autonomous Task Processor Template
        templates["task_processor"] = ProjectTemplate(
            name="Autonomous Task Processor",
            description="Background task processing with AI agent coordination",
            category="automation",
            technologies=["Python", "Celery", "Redis", "LeanVibe"],
            setup_commands=[
                "python -m venv venv",
                "source venv/bin/activate", 
                "pip install -e .",
                "celery -A app.worker worker --loglevel=info"
            ],
            template_files={
                "app/worker.py": self._get_celery_worker_template(),
                "app/tasks.py": self._get_tasks_template(),
                "requirements.txt": self._get_celery_requirements_template(),
                "README.md": self._get_readme_template("Autonomous Task Processor"),
            },
            learning_resources=[
                "Celery Documentation: https://docs.celeryproject.org/",
                "LeanVibe Task Integration: /docs/tasks/",
                "Redis Configuration: /docs/redis/"
            ],
            estimated_setup_time=5
        )
        
        # Full-Stack AI Application Template  
        templates["fullstack_ai"] = ProjectTemplate(
            name="Full-Stack AI Application",
            description="React + FastAPI + LeanVibe agent dashboard",
            category="fullstack", 
            technologies=["TypeScript", "React", "Python", "FastAPI", "LeanVibe"],
            setup_commands=[
                "npm create react-app frontend --template typescript",
                "python -m venv backend/venv",
                "cd backend && source venv/bin/activate && pip install -e .",
                "cd frontend && npm start"
            ],
            template_files={
                "backend/app/main.py": self._get_fastapi_main_template(),
                "frontend/src/components/Dashboard.tsx": self._get_react_dashboard_template(),
                "frontend/package.json": self._get_package_json_template(),
                "README.md": self._get_readme_template("Full-Stack AI Application"),
            },
            learning_resources=[
                "React Documentation: https://react.dev/",
                "TypeScript Guide: https://www.typescriptlang.org/",
                "LeanVibe Dashboard: /docs/dashboard/"
            ],
            estimated_setup_time=12
        )
        
        return templates
    
    def list_templates(self, category: Optional[str] = None) -> List[ProjectTemplate]:
        """List available project templates, optionally filtered by category."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return sorted(templates, key=lambda t: t.estimated_setup_time)
    
    def get_template(self, name: str) -> Optional[ProjectTemplate]:
        """Get a specific project template."""
        return self.templates.get(name)
    
    def get_recommended_templates(self, environment: DevelopmentEnvironment) -> List[ProjectTemplate]:
        """Get templates recommended for the current environment."""
        recommendations = []
        
        # Recommend based on available tools
        if environment.docker_available:
            recommendations.extend([
                self.templates["ai_web_api"],
                self.templates["fullstack_ai"]
            ])
        
        if environment.node_version:
            recommendations.append(self.templates["fullstack_ai"])
        
        # Always recommend simple templates
        recommendations.append(self.templates["task_processor"])
        
        return list(dict.fromkeys(recommendations))  # Remove duplicates
    
    def _get_fastapi_main_template(self) -> str:
        return '''"""
FastAPI Application with LeanVibe Integration
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI-Enhanced API",
    description="FastAPI application with LeanVibe agent integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AI-Enhanced API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2025-01-01T00:00:00Z"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    def _get_requirements_template(self) -> str:
        return '''fastapi[all]>=0.104.1
uvicorn[standard]>=0.24.0
sqlalchemy[asyncio]>=2.0.23
asyncpg>=0.29.0
alembic>=1.12.1
redis[hiredis]>=5.0.1
python-dotenv>=1.0.0
pydantic[email]>=2.5.0
'''
    
    def _get_celery_requirements_template(self) -> str:
        return '''celery[redis]>=5.3.4
redis[hiredis]>=5.0.1
python-dotenv>=1.0.0
pydantic>=2.5.0
'''
    
    def _get_alembic_template(self) -> str:
        return '''[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = postgresql://user:password@localhost:5432/dbname

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
'''
    
    def _get_env_template(self) -> str:
        return '''# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# LeanVibe Integration
LEANVIBE_API_URL=http://localhost:8000
ANTHROPIC_API_KEY=your_api_key_here
'''
    
    def _get_readme_template(self, project_name: str) -> str:
        return f'''# {project_name}

AI-powered application built with LeanVibe Agent Hive integration.

## Quick Start

```bash
# Setup
lv init {project_name.lower().replace(' ', '-')}
cd {project_name.lower().replace(' ', '-')}

# Run
lv start
lv develop "Implement user authentication"
```

## Features

- 🤖 AI agent integration
- 🚀 Fast development with LeanVibe
- 📊 Real-time monitoring
- 🔄 Automated workflows

## Development

```bash
# Install dependencies
pip install -e .

# Run tests
pytest

# Start development server
uvicorn app.main:app --reload
```

## LeanVibe Integration

This project includes LeanVibe Agent Hive integration for autonomous development:

- Smart code generation
- Automated testing
- Intelligent debugging
- Real-time collaboration

See the [LeanVibe documentation](/docs/) for more information.
'''
    
    def _get_celery_worker_template(self) -> str:
        return '''"""
Celery worker with LeanVibe integration
"""

from celery import Celery

app = Celery(
    'task_processor',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=['app.tasks']
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

if __name__ == '__main__':
    app.start()
'''
    
    def _get_tasks_template(self) -> str:
        return '''"""
Task definitions with LeanVibe integration
"""

from celery import shared_task
import time

@shared_task
def process_data(data):
    """Process data with AI assistance."""
    # Simulate processing
    time.sleep(2)
    return {"status": "completed", "processed_data": data}

@shared_task
def generate_report(report_type):
    """Generate reports using AI agents."""
    # This would integrate with LeanVibe agents
    return {"report_type": report_type, "generated_at": time.time()}
'''
    
    def _get_react_dashboard_template(self) -> str:
        return '''import React, { useState, useEffect } from 'react';

interface DashboardProps {}

const Dashboard: React.FC<DashboardProps> = () => {
  const [agentStatus, setAgentStatus] = useState('loading');
  const [metrics, setMetrics] = useState({});

  useEffect(() => {
    // Fetch agent status from LeanVibe API
    fetch('/api/agents/status')
      .then(res => res.json())
      .then(data => {
        setAgentStatus(data.status);
        setMetrics(data.metrics);
      });
  }, []);

  return (
    <div className="dashboard">
      <h1>LeanVibe Agent Dashboard</h1>
      
      <div className="status-card">
        <h2>Agent Status</h2>
        <p>Status: {agentStatus}</p>
      </div>
      
      <div className="metrics-card">
        <h2>Performance Metrics</h2>
        <pre>{JSON.stringify(metrics, null, 2)}</pre>
      </div>
    </div>
  );
};

export default Dashboard;
'''
    
    def _get_package_json_template(self) -> str:
        return '''{
  "name": "leanvibe-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "typescript": "^5.0.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test"
  }
}'''


class GuidedOnboardingSystem:
    """Provides guided onboarding with interactive tutorials and validation."""
    
    def __init__(self):
        self.environment = EnvironmentDetector().detect_environment()
        self.template_manager = ProjectTemplateManager()
        self.progress_tracking = {}
    
    async def start_guided_setup(self, project_name: Optional[str] = None) -> bool:
        """Start the guided onboarding experience."""
        console.print(Panel.fit(
            "🚀 [bold blue]LeanVibe Zero-Setup Development Environment[/bold blue]\n"
            "Guided onboarding for instant productivity",
            border_style="blue"
        ))
        
        # Show environment detection
        self._show_environment_analysis()
        
        # Template selection
        template = await self._guided_template_selection()
        if not template:
            return False
        
        # Project setup
        project_dir = await self._create_project_structure(project_name or "my-leanvibe-project", template)
        if not project_dir:
            return False
        
        # Guided configuration
        await self._guided_configuration(project_dir, template)
        
        # First success validation
        success = await self._validate_setup(project_dir, template)
        
        if success:
            self._show_success_summary(project_dir, template)
        else:
            self._show_troubleshooting_help()
        
        return success
    
    def _show_environment_analysis(self):
        """Display detected environment information."""
        console.print("\n🔍 [bold]Environment Analysis:[/bold]")
        
        env_table = Table()
        env_table.add_column("Component", style="cyan")
        env_table.add_column("Status", style="green")
        env_table.add_column("Details", style="white")
        
        env_table.add_row("Operating System", "✅ Detected", self.environment.os_type)
        env_table.add_row("Python", "✅ Available", self.environment.python_version)
        
        if self.environment.node_version:
            env_table.add_row("Node.js", "✅ Available", self.environment.node_version)
        else:
            env_table.add_row("Node.js", "❌ Not Found", "Install for full-stack templates")
        
        if self.environment.docker_available:
            env_table.add_row("Docker", "✅ Available", "Ready for containerized development")
        else:
            env_table.add_row("Docker", "⚠️ Not Found", "Recommended for production setup")
        
        if self.environment.git_configured:
            env_table.add_row("Git", "✅ Configured", "Ready for version control")
        else:
            env_table.add_row("Git", "⚠️ Not Configured", "Run: git config --global user.name 'Name'")
        
        if self.environment.editor_detected:
            env_table.add_row("Code Editor", "✅ Detected", self.environment.editor_detected)
        else:
            env_table.add_row("Code Editor", "❓ Unknown", "VS Code recommended")
        
        console.print(env_table)
    
    async def _guided_template_selection(self) -> Optional[ProjectTemplate]:
        """Guide user through template selection."""
        console.print("\n📋 [bold]Project Template Selection:[/bold]")
        
        # Show recommendations based on environment
        recommended = self.template_manager.get_recommended_templates(self.environment)
        
        if recommended:
            console.print("💡 [yellow]Recommended for your environment:[/yellow]")
            
            choices = {}
            for i, template in enumerate(recommended, 1):
                console.print(f"   {i}. [green]{template.name}[/green] - {template.description}")
                console.print(f"      Technologies: {', '.join(template.technologies)}")
                console.print(f"      Setup time: ~{template.estimated_setup_time} minutes")
                choices[str(i)] = template
            
            console.print("\n📚 [bold]All Templates:[/bold]")
            all_templates = self.template_manager.list_templates()
            for i, template in enumerate(all_templates, len(recommended) + 1):
                if template not in recommended:
                    console.print(f"   {i}. [blue]{template.name}[/blue] - {template.description}")
                    choices[str(i)] = template
            
            choice = Prompt.ask(
                "Select a template",
                choices=list(choices.keys()),
                default="1"
            )
            
            return choices.get(choice)
        
        return None
    
    async def _create_project_structure(self, project_name: str, template: ProjectTemplate) -> Optional[Path]:
        """Create project structure from template."""
        console.print(f"\n🏗️ [bold]Creating project: {project_name}[/bold]")
        
        project_dir = Path.cwd() / project_name
        
        if project_dir.exists():
            if not Confirm.ask(f"Directory {project_name} exists. Continue anyway?"):
                return None
        
        project_dir.mkdir(exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Creating project structure...", total=len(template.template_files))
            
            for file_path, content in template.template_files.items():
                file_full_path = project_dir / file_path
                file_full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_full_path, 'w') as f:
                    f.write(content)
                
                progress.advance(task)
            
            progress.update(task, description="✅ Project structure created")
        
        return project_dir
    
    async def _guided_configuration(self, project_dir: Path, template: ProjectTemplate):
        """Guide user through project configuration."""
        console.print("\n⚙️ [bold]Project Configuration:[/bold]")
        
        # Environment file configuration
        env_file = project_dir / ".env"
        if (project_dir / ".env.example").exists():
            console.print("📝 Setting up environment variables...")
            
            # Copy example to .env
            shutil.copy2(project_dir / ".env.example", env_file)
            
            # Interactive configuration for common variables
            if "ANTHROPIC_API_KEY" in env_file.read_text():
                api_key = Prompt.ask(
                    "Enter your Anthropic API key (optional for local development)",
                    default="",
                    show_default=False
                )
                if api_key:
                    content = env_file.read_text()
                    content = content.replace("your_api_key_here", api_key)
                    env_file.write_text(content)
        
        # Database setup (if needed)
        if "postgresql" in str(template.technologies).lower():
            if Confirm.ask("Set up local PostgreSQL database?", default=True):
                console.print("💡 Database will be configured with Docker Compose")
        
        console.print("✅ Configuration completed")
    
    async def _validate_setup(self, project_dir: Path, template: ProjectTemplate) -> bool:
        """Validate that the project setup is working correctly."""
        console.print("\n🧪 [bold]Validating Setup:[/bold]")
        
        os.chdir(project_dir)
        
        validation_success = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            for i, command in enumerate(template.setup_commands):
                task = progress.add_task(f"Running: {command}", total=None)
                
                try:
                    # Skip interactive commands for validation
                    if any(interactive in command for interactive in ["--reload", "start", "worker"]):
                        progress.update(task, description=f"⏭️ Skipping interactive: {command}")
                        continue
                    
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        progress.update(task, description=f"✅ Completed: {command}")
                    else:
                        progress.update(task, description=f"❌ Failed: {command}")
                        validation_success = False
                        
                except subprocess.TimeoutExpired:
                    progress.update(task, description=f"⏰ Timeout: {command}")
                    validation_success = False
                except Exception as e:
                    progress.update(task, description=f"❌ Error: {command}")
                    validation_success = False
        
        return validation_success
    
    def _show_success_summary(self, project_dir: Path, template: ProjectTemplate):
        """Show success summary with next steps."""
        console.print("\n🎉 [bold green]Setup Completed Successfully![/bold green]")
        
        summary_panel = Panel.fit(
            f"✅ [bold]{template.name}[/bold] is ready!\n"
            f"📁 Location: {project_dir}\n"
            f"⏱️ Setup time: ~{template.estimated_setup_time} minutes\n"
            f"🚀 You're ready to start developing!",
            title="Success Summary",
            border_style="green"
        )
        console.print(summary_panel)
        
        # Next steps
        console.print("\n📋 [bold yellow]Next Steps:[/bold yellow]")
        console.print(f"   1. cd {project_dir.name}")
        console.print("   2. lv start                    # Start LeanVibe services")
        console.print('   3. lv develop "your idea"      # Start autonomous development')
        console.print("   4. lv dashboard                # Monitor progress")
        
        # Learning resources
        if template.learning_resources:
            console.print("\n📚 [bold blue]Learning Resources:[/bold blue]")
            for resource in template.learning_resources:
                console.print(f"   • {resource}")
    
    def _show_troubleshooting_help(self):
        """Show troubleshooting help if setup failed."""
        console.print("\n⚠️ [bold yellow]Setup encountered issues[/bold yellow]")
        console.print("\n🔧 [bold]Troubleshooting Steps:[/bold]")
        console.print("   1. Check that all required tools are installed")
        console.print("   2. Run: lv health              # Check system status")
        console.print("   3. Run: lv debug               # Advanced diagnostics")
        console.print("   4. Visit: /docs/troubleshooting/")
        console.print("\n💬 Need help? Create an issue at: https://github.com/leanvibe/agent-hive/issues")


# Export main functions for CLI integration
async def run_zero_setup_onboarding(project_name: Optional[str] = None) -> bool:
    """Main entry point for zero-setup onboarding."""
    onboarding = GuidedOnboardingSystem()
    return await onboarding.start_guided_setup(project_name)