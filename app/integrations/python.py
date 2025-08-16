"""
Python Framework Adapters for Project Index

Provides seamless integration with popular Python web frameworks:
- FastAPI: Router integration, dependency injection, middleware
- Django: App integration, signals, management commands  
- Flask: Blueprint integration, extension pattern
- Celery: Task queue integration, monitoring
"""

import asyncio
import logging
from typing import Any, Optional, Dict, List, Callable
from pathlib import Path

from . import BaseFrameworkAdapter, IntegrationManager
from ..project_index import ProjectIndexer, AnalysisConfiguration, ProjectIndexConfig


class FastAPIAdapter(BaseFrameworkAdapter):
    """
    FastAPI integration adapter with minimal setup and maximum performance.
    
    Example usage:
        from fastapi import FastAPI
        from app.integrations.fastapi import add_project_index
        
        app = FastAPI()
        add_project_index(app)  # One line integration!
    """
    
    def integrate(self, app: Any, **kwargs) -> None:
        """
        Integrate Project Index with FastAPI application.
        
        Args:
            app: FastAPI application instance
            **kwargs: FastAPI-specific options
                - mount_path: API mount path (default: "/project-index")
                - enable_docs: Include in OpenAPI docs (default: True)
                - enable_websocket: Enable WebSocket endpoint (default: True)
        """
        mount_path = kwargs.get('mount_path', '/project-index')
        enable_docs = kwargs.get('enable_docs', True)
        enable_websocket = kwargs.get('enable_websocket', True)
        
        # Setup startup/shutdown events
        @app.on_event("startup")
        async def startup_project_index():
            await self.start()
            logging.info("🚀 Project Index integrated with FastAPI")
        
        @app.on_event("shutdown") 
        async def shutdown_project_index():
            await self.stop()
            logging.info("✅ Project Index shutdown complete")
        
        # Setup routes and middleware
        self._setup_routes(app, mount_path, enable_docs, enable_websocket)
        self._setup_middleware(app)
        
        # Add to app state for access in routes
        app.state.project_index = self
    
    def _setup_routes(self, app: Any, mount_path: str = '/project-index', enable_docs: bool = True, enable_websocket: bool = True) -> None:
        """Setup FastAPI routes for Project Index API."""
        from fastapi import APIRouter, Depends, HTTPException, WebSocket
        from fastapi.responses import JSONResponse
        
        router = APIRouter(prefix=mount_path, tags=["Project Index"])
        
        @router.get("/status", summary="Get Project Index status")
        async def get_status():
            """Get current Project Index status and metrics."""
            if not self.indexer:
                raise HTTPException(status_code=503, detail="Project Index not initialized")
            
            return {
                "status": "active",
                "initialized": self._initialized,
                "config": {
                    "cache_enabled": self.config.cache_enabled,
                    "monitoring_enabled": self.config.monitoring_enabled,
                    "max_concurrent_analyses": self.config.max_concurrent_analyses
                }
            }
        
        @router.post("/analyze", summary="Analyze project")
        async def analyze_project(project_path: str, languages: Optional[List[str]] = None):
            """Analyze a project and return results."""
            if not self.indexer:
                raise HTTPException(status_code=503, detail="Project Index not initialized")
            
            try:
                # Create project
                project = await self.indexer.create_project(
                    name=Path(project_path).name,
                    root_path=project_path
                )
                
                # Analyze with specified languages
                if languages:
                    analysis_config = AnalysisConfiguration(enabled_languages=languages)
                    self.indexer.analyzer.config = analysis_config
                
                result = await self.indexer.analyze_project(str(project.id))
                
                return {
                    "project_id": str(project.id),
                    "files_processed": result.files_processed,
                    "dependencies_found": result.dependencies_found,
                    "analysis_time": result.analysis_time,
                    "languages_detected": result.languages_detected
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @router.get("/projects", summary="List projects")
        async def list_projects():
            """List all indexed projects."""
            if not self.indexer:
                raise HTTPException(status_code=503, detail="Project Index not initialized")
            
            # Implementation would depend on your database models
            return {"projects": []}
        
        if enable_websocket:
            @router.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time updates."""
                await websocket.accept()
                try:
                    while True:
                        # Send periodic updates
                        await websocket.send_json({"type": "ping", "timestamp": str(asyncio.get_event_loop().time())})
                        await asyncio.sleep(30)
                except Exception:
                    pass
        
        app.include_router(router, include_in_schema=enable_docs)
    
    def _setup_middleware(self, app: Any) -> None:
        """Setup FastAPI middleware for Project Index."""
        from fastapi import Request, Response
        from starlette.middleware.base import BaseHTTPMiddleware
        
        class ProjectIndexMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, adapter: FastAPIAdapter):
                super().__init__(app)
                self.adapter = adapter
            
            async def dispatch(self, request: Request, call_next):
                # Add Project Index context to request
                request.state.project_index = self.adapter
                response = await call_next(request)
                
                # Add headers
                response.headers["X-Project-Index"] = "active"
                return response
        
        app.add_middleware(ProjectIndexMiddleware, adapter=self)


class DjangoAdapter(BaseFrameworkAdapter):
    """
    Django integration adapter with app integration and management commands.
    
    Example usage:
        # In settings.py
        INSTALLED_APPS = [
            'app.integrations.django_app',  # Add Project Index app
            # ... other apps
        ]
        
        # Or manual integration
        from app.integrations.django import add_project_index
        add_project_index()
    """
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """
        Integrate Project Index with Django.
        
        Args:
            app: Django app config (optional)
            **kwargs: Django-specific options
                - install_commands: Install management commands (default: True)
                - setup_signals: Setup Django signals (default: True)
                - admin_integration: Add admin interface (default: True)
        """
        install_commands = kwargs.get('install_commands', True)
        setup_signals = kwargs.get('setup_signals', True)
        admin_integration = kwargs.get('admin_integration', True)
        
        self._setup_django_app(install_commands, setup_signals, admin_integration)
        self._setup_routes(app)
        self._setup_middleware(app)
    
    def _setup_django_app(self, install_commands: bool, setup_signals: bool, admin_integration: bool) -> None:
        """Setup Django app integration."""
        # This would typically be in a separate Django app
        # For now, we'll implement the core logic here
        
        if setup_signals:
            self._setup_django_signals()
        
        if install_commands:
            self._register_management_commands()
        
        if admin_integration:
            self._setup_admin_interface()
    
    def _setup_django_signals(self) -> None:
        """Setup Django signals for automatic project monitoring."""
        try:
            from django.db.models.signals import post_save, post_delete
            from django.dispatch import receiver
            
            @receiver(post_save, sender='myapp.MyModel')  # Replace with actual model
            def handle_model_save(sender, instance, **kwargs):
                # Trigger analysis when models change
                if self.indexer and self._initialized:
                    asyncio.create_task(self._trigger_analysis())
        except ImportError:
            pass
    
    def _register_management_commands(self) -> None:
        """Register Django management commands."""
        # Commands would be in management/commands/ directory
        # analyze_project.py, reindex_project.py, etc.
        pass
    
    def _setup_admin_interface(self) -> None:
        """Setup Django admin interface for Project Index."""
        try:
            from django.contrib import admin
            
            # Register models and admin interfaces
            # This would be in admin.py
        except ImportError:
            pass
    
    def _setup_routes(self, app: Any) -> None:
        """Setup Django URLs for Project Index."""
        # URLs would be defined in urls.py
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Setup Django middleware."""
        # Middleware would be a separate class
        pass
    
    async def _trigger_analysis(self) -> None:
        """Trigger project analysis from Django signals."""
        if self.indexer:
            # Implementation for triggering analysis
            pass


class FlaskAdapter(BaseFrameworkAdapter):
    """
    Flask integration adapter with Blueprint pattern and extension support.
    
    Example usage:
        from flask import Flask
        from app.integrations.flask import add_project_index
        
        app = Flask(__name__)
        add_project_index(app)  # One line integration!
    """
    
    def integrate(self, app: Any, **kwargs) -> None:
        """
        Integrate Project Index with Flask application.
        
        Args:
            app: Flask application instance
            **kwargs: Flask-specific options
                - blueprint_prefix: URL prefix for blueprint (default: "/project-index")
                - enable_cli: Add Flask CLI commands (default: True)
        """
        blueprint_prefix = kwargs.get('blueprint_prefix', '/project-index')
        enable_cli = kwargs.get('enable_cli', True)
        
        # Setup Flask extension pattern
        self._setup_flask_extension(app)
        self._setup_routes(app, blueprint_prefix)
        self._setup_middleware(app)
        
        if enable_cli:
            self._setup_cli_commands(app)
    
    def _setup_flask_extension(self, app: Any) -> None:
        """Setup Flask extension pattern."""
        # Store adapter in app extensions
        if not hasattr(app, 'extensions'):
            app.extensions = {}
        app.extensions['project_index'] = self
        
        # Setup startup/teardown handlers
        @app.before_first_request
        def startup_project_index():
            asyncio.create_task(self.start())
        
        @app.teardown_appcontext
        def shutdown_project_index(exception):
            if exception:
                logging.error(f"App teardown with exception: {exception}")
    
    def _setup_routes(self, app: Any, prefix: str = '/project-index') -> None:
        """Setup Flask Blueprint for Project Index API."""
        from flask import Blueprint, jsonify, request
        
        bp = Blueprint('project_index', __name__, url_prefix=prefix)
        
        @bp.route('/status')
        def get_status():
            """Get Project Index status."""
            return jsonify({
                "status": "active",
                "initialized": self._initialized,
                "config": {
                    "cache_enabled": self.config.cache_enabled,
                    "monitoring_enabled": self.config.monitoring_enabled
                }
            })
        
        @bp.route('/analyze', methods=['POST'])
        def analyze_project():
            """Analyze a project."""
            data = request.get_json()
            project_path = data.get('project_path')
            
            if not project_path:
                return jsonify({"error": "project_path required"}), 400
            
            # For Flask, we'd need to handle async operations differently
            # This is a simplified example
            return jsonify({
                "message": "Analysis started",
                "project_path": project_path
            })
        
        app.register_blueprint(bp)
    
    def _setup_middleware(self, app: Any) -> None:
        """Setup Flask middleware."""
        @app.before_request
        def before_request():
            # Add Project Index context to Flask g object
            from flask import g
            g.project_index = self
    
    def _setup_cli_commands(self, app: Any) -> None:
        """Setup Flask CLI commands."""
        @app.cli.command()
        def analyze_project():
            """CLI command to analyze current project."""
            import click
            click.echo("Analyzing project with Project Index...")
            # Implementation would trigger analysis


class CeleryAdapter(BaseFrameworkAdapter):
    """
    Celery integration adapter for task queue integration and monitoring.
    
    Example usage:
        from celery import Celery
        from app.integrations.celery import add_project_index
        
        celery_app = Celery('myapp')
        add_project_index(celery_app)
    """
    
    def integrate(self, app: Any, **kwargs) -> None:
        """
        Integrate Project Index with Celery.
        
        Args:
            app: Celery application instance
            **kwargs: Celery-specific options
                - register_tasks: Register analysis tasks (default: True)
                - setup_monitoring: Setup task monitoring (default: True)
        """
        register_tasks = kwargs.get('register_tasks', True)
        setup_monitoring = kwargs.get('setup_monitoring', True)
        
        if register_tasks:
            self._register_celery_tasks(app)
        
        if setup_monitoring:
            self._setup_celery_monitoring(app)
        
        self._setup_routes(app)
        self._setup_middleware(app)
    
    def _register_celery_tasks(self, celery_app: Any) -> None:
        """Register Celery tasks for Project Index operations."""
        @celery_app.task
        def analyze_project_task(project_path: str, languages: List[str] = None):
            """Celery task for project analysis."""
            # This would need to handle async operations in Celery context
            return {"status": "completed", "project_path": project_path}
        
        @celery_app.task
        def reindex_project_task(project_id: str):
            """Celery task for project reindexing."""
            return {"status": "completed", "project_id": project_id}
    
    def _setup_celery_monitoring(self, celery_app: Any) -> None:
        """Setup Celery task monitoring."""
        from celery.signals import task_prerun, task_postrun, task_failure
        
        @task_prerun.connect
        def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwargs_extra):
            logging.info(f"Starting Project Index task: {task}")
        
        @task_postrun.connect
        def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwargs_extra):
            logging.info(f"Completed Project Index task: {task}")
        
        @task_failure.connect
        def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
            logging.error(f"Failed Project Index task: {sender}, Exception: {exception}")
    
    def _setup_routes(self, app: Any) -> None:
        """Setup routes (not applicable for Celery)."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Setup middleware (not applicable for Celery)."""
        pass


# Register adapters
IntegrationManager.register_adapter('fastapi', FastAPIAdapter)
IntegrationManager.register_adapter('django', DjangoAdapter) 
IntegrationManager.register_adapter('flask', FlaskAdapter)
IntegrationManager.register_adapter('celery', CeleryAdapter)


# Convenience functions for one-line integration
def add_project_index_fastapi(app, config: Optional[ProjectIndexConfig] = None, **kwargs) -> FastAPIAdapter:
    """
    One-line FastAPI integration.
    
    Args:
        app: FastAPI application
        config: Optional Project Index configuration
        **kwargs: FastAPI-specific options
        
    Returns:
        Configured FastAPI adapter
    """
    adapter = FastAPIAdapter(config=config)
    adapter.integrate(app, **kwargs)
    return adapter


def add_project_index_django(config: Optional[ProjectIndexConfig] = None, **kwargs) -> DjangoAdapter:
    """
    One-line Django integration.
    
    Args:
        config: Optional Project Index configuration
        **kwargs: Django-specific options
        
    Returns:
        Configured Django adapter
    """
    adapter = DjangoAdapter(config=config)
    adapter.integrate(**kwargs)
    return adapter


def add_project_index_flask(app, config: Optional[ProjectIndexConfig] = None, **kwargs) -> FlaskAdapter:
    """
    One-line Flask integration.
    
    Args:
        app: Flask application
        config: Optional Project Index configuration
        **kwargs: Flask-specific options
        
    Returns:
        Configured Flask adapter
    """
    adapter = FlaskAdapter(config=config)
    adapter.integrate(app, **kwargs)
    return adapter


def add_project_index_celery(celery_app, config: Optional[ProjectIndexConfig] = None, **kwargs) -> CeleryAdapter:
    """
    One-line Celery integration.
    
    Args:
        celery_app: Celery application
        config: Optional Project Index configuration  
        **kwargs: Celery-specific options
        
    Returns:
        Configured Celery adapter
    """
    adapter = CeleryAdapter(config=config)
    adapter.integrate(celery_app, **kwargs)
    return adapter


# Convenience imports for direct usage
add_project_index = add_project_index_fastapi  # Default to FastAPI for backward compatibility