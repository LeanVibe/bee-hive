"""
CLI Integration Commands for Project Index Framework Setup

Provides command-line tools for easy framework integration setup.
Supports interactive setup, code generation, and configuration management.
"""

import click
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from . import IntegrationManager, detect_framework, quick_integrate
from .python import (
    add_project_index_fastapi, add_project_index_django, 
    add_project_index_flask, add_project_index_celery
)
from .javascript import (
    generate_express_integration, generate_nextjs_integration,
    generate_react_integration, generate_vue_integration, generate_angular_integration
)
from .other_languages import (
    generate_go_integration, generate_rust_integration, generate_java_integration
)


class ProjectIndexCLI:
    """Command-line interface for Project Index framework integration."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_file = self.project_root / '.project-index.json'
    
    def load_config(self) -> Dict[str, Any]:
        """Load Project Index configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                click.echo(f"Warning: Could not load config file: {e}", err=True)
        
        return {}
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save Project Index configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            click.echo(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            click.echo(f"❌ Failed to save configuration: {e}", err=True)
    
    def detect_project_info(self) -> Dict[str, Any]:
        """Detect project information and framework."""
        info = {
            'root': str(self.project_root),
            'framework': detect_framework(),
            'files': {
                'package.json': (self.project_root / 'package.json').exists(),
                'requirements.txt': (self.project_root / 'requirements.txt').exists(),
                'pyproject.toml': (self.project_root / 'pyproject.toml').exists(),
                'Cargo.toml': (self.project_root / 'Cargo.toml').exists(),
                'go.mod': (self.project_root / 'go.mod').exists(),
                'pom.xml': (self.project_root / 'pom.xml').exists(),
                'build.gradle': (self.project_root / 'build.gradle').exists(),
            }
        }
        
        # Try to extract project name
        if info['files']['package.json']:
            try:
                with open(self.project_root / 'package.json', 'r') as f:
                    package_data = json.load(f)
                    info['name'] = package_data.get('name', self.project_root.name)
            except Exception:
                info['name'] = self.project_root.name
        else:
            info['name'] = self.project_root.name
        
        return info
    
    def check_dependencies(self, framework: str) -> Dict[str, bool]:
        """Check if required dependencies are installed."""
        checks = {}
        
        if framework in ['fastapi', 'django', 'flask']:
            checks['python'] = self._check_command('python', '--version')
            checks['pip'] = self._check_command('pip', '--version')
        elif framework in ['express', 'nextjs', 'react', 'vue', 'angular']:
            checks['node'] = self._check_command('node', '--version')
            checks['npm'] = self._check_command('npm', '--version')
        elif framework == 'go':
            checks['go'] = self._check_command('go', 'version')
        elif framework == 'rust':
            checks['cargo'] = self._check_command('cargo', '--version')
        elif framework == 'java':
            checks['java'] = self._check_command('java', '--version')
            checks['mvn'] = self._check_command('mvn', '--version')
        
        return checks
    
    def _check_command(self, command: str, *args) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([command] + list(args), 
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def generate_integration_code(self, framework: str, **options) -> bool:
        """Generate integration code for the specified framework."""
        try:
            api_url = options.get('api_url', 'http://localhost:8000/project-index')
            
            if framework == 'fastapi':
                # For FastAPI, we provide instructions since it's runtime integration
                self._generate_fastapi_instructions(api_url)
            elif framework == 'django':
                self._generate_django_instructions(api_url)
            elif framework == 'flask':
                self._generate_flask_instructions(api_url)
            elif framework == 'express':
                generate_express_integration(api_url, **options)
            elif framework == 'nextjs':
                generate_nextjs_integration(api_url, **options)
            elif framework == 'react':
                generate_react_integration(api_url, **options)
            elif framework == 'vue':
                generate_vue_integration(api_url, **options)
            elif framework == 'angular':
                generate_angular_integration(api_url, **options)
            elif framework == 'go':
                go_framework = options.get('go_framework', 'gin')
                generate_go_integration(go_framework, api_url, **options)
            elif framework == 'rust':
                rust_framework = options.get('rust_framework', 'axum')
                generate_rust_integration(rust_framework, api_url, **options)
            elif framework == 'java':
                package_name = options.get('package_name', 'com.example.projectindex')
                generate_java_integration(api_url, package_name, **options)
            else:
                click.echo(f"❌ Framework '{framework}' not supported yet")
                return False
            
            return True
        except Exception as e:
            click.echo(f"❌ Failed to generate integration code: {e}", err=True)
            return False
    
    def _generate_fastapi_instructions(self, api_url: str) -> None:
        """Generate FastAPI integration instructions."""
        instructions = f"""
# FastAPI Project Index Integration

## 1. Install Dependencies
```bash
pip install project-index-client  # When available
```

## 2. Add Integration Code

Add this to your FastAPI application:

```python
from fastapi import FastAPI
from app.integrations.python import add_project_index_fastapi

app = FastAPI()

# One-line integration!
adapter = add_project_index_fastapi(app)

# Optional: Configure API URL
adapter.set_api_url("{api_url}")
```

## 3. Access Project Index

Your FastAPI app now has these endpoints:
- GET /project-index/status
- POST /project-index/analyze  
- GET /project-index/projects
- WebSocket /project-index/ws

## 4. Use in Routes

```python
@app.get("/analyze-current")
async def analyze_current(request: Request):
    project_index = request.app.state.project_index
    # Use project_index.indexer for analysis
    return {{"status": "analyzing"}}
```
"""
        
        self._write_file('PROJECT_INDEX_INTEGRATION.md', instructions)
    
    def _generate_django_instructions(self, api_url: str) -> None:
        """Generate Django integration instructions."""
        instructions = f"""
# Django Project Index Integration

## 1. Add to INSTALLED_APPS

```python
INSTALLED_APPS = [
    # ... your apps
    'app.integrations.django_app',  # Add Project Index app
]
```

## 2. Add Configuration

```python
# settings.py
PROJECT_INDEX_CONFIG = {{
    'API_URL': '{api_url}',
    'CACHE_ENABLED': True,
    'MONITORING_ENABLED': True,
}}
```

## 3. Add URLs

```python
# urls.py
from django.urls import path, include

urlpatterns = [
    # ... your URLs
    path('api/project-index/', include('app.integrations.django_urls')),
]
```

## 4. Use in Views

```python
from app.integrations.django import get_project_index_client

def my_view(request):
    client = get_project_index_client()
    # Use client for analysis
    return JsonResponse({{"status": "active"}})
```
"""
        
        self._write_file('DJANGO_INTEGRATION.md', instructions)
    
    def _generate_flask_instructions(self, api_url: str) -> None:
        """Generate Flask integration instructions."""
        instructions = f"""
# Flask Project Index Integration

## 1. Install and Import

```python
from flask import Flask
from app.integrations.python import add_project_index_flask

app = Flask(__name__)

# One-line integration!
adapter = add_project_index_flask(app)
```

## 2. Configure

```python
# Optional configuration
app.config['PROJECT_INDEX_API_URL'] = '{api_url}'
app.config['PROJECT_INDEX_CACHE_ENABLED'] = True
```

## 3. Use in Routes

```python
@app.route('/analyze')
def analyze():
    project_index = current_app.extensions['project_index']
    # Use project_index for analysis
    return {{"status": "analyzing"}}
```

## 4. Available Endpoints

- GET /project-index/status
- POST /project-index/analyze
- GET /project-index/projects
"""
        
        self._write_file('FLASK_INTEGRATION.md', instructions)
    
    def _write_file(self, file_path: str, content: str) -> None:
        """Write content to file."""
        full_path = self.project_root / file_path
        try:
            full_path.write_text(content)
            click.echo(f"✅ Generated {file_path}")
        except Exception as e:
            click.echo(f"❌ Failed to generate {file_path}: {e}", err=True)


# Create CLI instance
cli_manager = ProjectIndexCLI()


@click.group()
def pi():
    """Project Index CLI - Framework integration made easy."""
    pass


@pi.command()
def status():
    """Show Project Index integration status."""
    project_info = cli_manager.detect_project_info()
    config = cli_manager.load_config()
    
    click.echo("🔍 Project Index Status")
    click.echo(f"Project: {project_info['name']}")
    click.echo(f"Root: {project_info['root']}")
    click.echo(f"Detected Framework: {project_info['framework'] or 'Unknown'}")
    
    if config:
        click.echo(f"Configuration: ✅ Found (.project-index.json)")
        click.echo(f"  Framework: {config.get('framework', 'Not set')}")
        click.echo(f"  API URL: {config.get('api_url', 'Not set')}")
    else:
        click.echo("Configuration: ❌ Not found")
    
    # Check dependencies
    if project_info['framework']:
        deps = cli_manager.check_dependencies(project_info['framework'])
        click.echo("Dependencies:")
        for dep, available in deps.items():
            status_icon = "✅" if available else "❌"
            click.echo(f"  {dep}: {status_icon}")


@pi.command()
@click.option('--framework', '-f', help='Specify framework (auto-detected if not provided)')
@click.option('--api-url', '-u', default='http://localhost:8000/project-index', help='Project Index API URL')
@click.option('--interactive/--no-interactive', '-i/-I', default=True, help='Interactive setup')
def setup(framework, api_url, interactive):
    """Set up Project Index integration for your project."""
    project_info = cli_manager.detect_project_info()
    
    # Auto-detect framework if not provided
    if not framework:
        framework = project_info['framework']
        if not framework and interactive:
            supported_frameworks = IntegrationManager.list_supported_frameworks()
            click.echo("Could not auto-detect framework. Please choose:")
            for i, fw in enumerate(supported_frameworks, 1):
                click.echo(f"  {i}. {fw}")
            
            choice = click.prompt("Enter choice", type=int, default=1)
            if 1 <= choice <= len(supported_frameworks):
                framework = supported_frameworks[choice - 1]
    
    if not framework:
        click.echo("❌ No framework specified or detected", err=True)
        return
    
    if framework not in IntegrationManager.list_supported_frameworks():
        click.echo(f"❌ Framework '{framework}' not supported", err=True)
        click.echo(f"Supported frameworks: {', '.join(IntegrationManager.list_supported_frameworks())}")
        return
    
    click.echo(f"🚀 Setting up Project Index integration for {framework}")
    
    # Check dependencies
    deps = cli_manager.check_dependencies(framework)
    missing_deps = [dep for dep, available in deps.items() if not available]
    if missing_deps:
        click.echo(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        if interactive and click.confirm("Continue anyway?"):
            pass
        elif not interactive:
            return
    
    # Interactive configuration
    options = {}
    if interactive:
        if framework in ['go']:
            go_frameworks = ['gin', 'echo', 'fiber', 'stdlib']
            go_fw = click.prompt("Go framework", default='gin', type=click.Choice(go_frameworks))
            options['go_framework'] = go_fw
        elif framework in ['rust']:
            rust_frameworks = ['axum', 'rocket']
            rust_fw = click.prompt("Rust framework", default='axum', type=click.Choice(rust_frameworks))
            options['rust_framework'] = rust_fw
        elif framework in ['java']:
            package_name = click.prompt("Java package name", default='com.example.projectindex')
            options['package_name'] = package_name
        
        # Common options
        api_url = click.prompt("API URL", default=api_url)
    
    options['api_url'] = api_url
    
    # Generate integration code
    success = cli_manager.generate_integration_code(framework, **options)
    
    if success:
        # Save configuration
        config = {
            'framework': framework,
            'api_url': api_url,
            'options': options,
            'setup_date': str(click.DateTime().today())
        }
        cli_manager.save_config(config)
        
        click.echo("✅ Project Index integration setup complete!")
        click.echo("Next steps:")
        click.echo("1. Review generated files")
        click.echo("2. Install any missing dependencies")
        click.echo("3. Start your application")
        click.echo("4. Test integration endpoints")
    else:
        click.echo("❌ Setup failed")


@pi.command()
def detect():
    """Detect current project framework and dependencies."""
    project_info = cli_manager.detect_project_info()
    
    click.echo("🔍 Project Detection Results")
    click.echo(f"Project: {project_info['name']}")
    click.echo(f"Framework: {project_info['framework'] or 'Unknown'}")
    
    click.echo("Project Files:")
    for file_name, exists in project_info['files'].items():
        status_icon = "✅" if exists else "❌"
        click.echo(f"  {file_name}: {status_icon}")
    
    if project_info['framework']:
        deps = cli_manager.check_dependencies(project_info['framework'])
        click.echo(f"Dependencies for {project_info['framework']}:")
        for dep, available in deps.items():
            status_icon = "✅" if available else "❌"
            click.echo(f"  {dep}: {status_icon}")


@pi.command()
def list():
    """List supported frameworks."""
    frameworks = IntegrationManager.list_supported_frameworks()
    
    click.echo("🔧 Supported Frameworks:")
    
    # Group by language
    python_frameworks = [f for f in frameworks if f in ['fastapi', 'django', 'flask', 'celery']]
    js_frameworks = [f for f in frameworks if f in ['express', 'nextjs', 'react', 'vue', 'angular']]
    other_frameworks = [f for f in frameworks if f not in python_frameworks + js_frameworks]
    
    if python_frameworks:
        click.echo("  Python:")
        for fw in python_frameworks:
            click.echo(f"    • {fw}")
    
    if js_frameworks:
        click.echo("  JavaScript/TypeScript:")
        for fw in js_frameworks:
            click.echo(f"    • {fw}")
    
    if other_frameworks:
        click.echo("  Other Languages:")
        for fw in other_frameworks:
            click.echo(f"    • {fw}")


@pi.command()
@click.argument('framework')
@click.option('--output', '-o', help='Output directory for generated code')
def generate(framework, output):
    """Generate integration code for specified framework."""
    if framework not in IntegrationManager.list_supported_frameworks():
        click.echo(f"❌ Framework '{framework}' not supported", err=True)
        return
    
    if output:
        original_cwd = os.getcwd()
        try:
            os.chdir(output)
            success = cli_manager.generate_integration_code(framework)
        finally:
            os.chdir(original_cwd)
    else:
        success = cli_manager.generate_integration_code(framework)
    
    if success:
        click.echo(f"✅ Generated {framework} integration code")
    else:
        click.echo(f"❌ Failed to generate {framework} integration code")


@pi.command()
@click.option('--api-url', '-u', help='Project Index API URL to test')
def test(api_url):
    """Test Project Index API connection."""
    import requests
    
    if not api_url:
        config = cli_manager.load_config()
        api_url = config.get('api_url', 'http://localhost:8000/project-index')
    
    click.echo(f"🧪 Testing connection to {api_url}")
    
    try:
        response = requests.get(f"{api_url}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            click.echo("✅ Connection successful!")
            click.echo(f"  Status: {data.get('status', 'unknown')}")
            click.echo(f"  Initialized: {data.get('initialized', False)}")
        else:
            click.echo(f"❌ API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        click.echo("❌ Could not connect to Project Index API")
        click.echo("  Make sure the Project Index service is running")
    except requests.exceptions.Timeout:
        click.echo("❌ Connection timeout")
    except Exception as e:
        click.echo(f"❌ Connection error: {e}")


# Make CLI available as a module
if __name__ == '__main__':
    pi()


# Export for use in other modules
__all__ = ['pi', 'ProjectIndexCLI', 'cli_manager']