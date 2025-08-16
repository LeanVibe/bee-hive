"""
Mock Services for Project Index Testing

This module provides lightweight mock services for isolated testing scenarios:
- Mock API Server for endpoint testing
- Mock Database for data layer testing  
- Mock Redis for caching/messaging testing
- Mock WebSocket Server for real-time testing
- Mock File System for file operations testing
"""

import asyncio
import json
import logging
import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from unittest.mock import Mock, AsyncMock
import aiohttp
from aiohttp import web, WSMsgType
import aiofiles
import fakeredis.aioredis


logger = logging.getLogger(__name__)


class MockDatabase:
    """Mock database for testing Project Index functionality."""
    
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or ":memory:"
        self.connection = None
        self.tables_created = False
    
    def connect(self):
        """Connect to the mock database."""
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row  # Return rows as dictionaries
        self._create_tables()
    
    def _create_tables(self):
        """Create mock database tables."""
        if self.tables_created:
            return
        
        cursor = self.connection.cursor()
        
        # Project indexes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_indexes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                root_path TEXT NOT NULL,
                git_repository_url TEXT,
                git_branch TEXT,
                git_commit_hash TEXT,
                status TEXT DEFAULT 'inactive',
                file_count INTEGER DEFAULT 0,
                dependency_count INTEGER DEFAULT 0,
                configuration TEXT DEFAULT '{}',
                analysis_settings TEXT DEFAULT '{}',
                file_patterns TEXT DEFAULT '{}',
                ignore_patterns TEXT DEFAULT '{}',
                meta_data TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_indexed_at TIMESTAMP,
                last_analysis_at TIMESTAMP
            )
        """)
        
        # File entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_entries (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_extension TEXT,
                file_type TEXT,
                language TEXT,
                file_size INTEGER,
                line_count INTEGER,
                sha256_hash TEXT,
                is_binary BOOLEAN DEFAULT FALSE,
                is_generated BOOLEAN DEFAULT FALSE,
                analysis_data TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project_indexes (id)
            )
        """)
        
        # Dependency relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dependency_relationships (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_file_id TEXT,
                target_file_id TEXT,
                target_path TEXT,
                target_name TEXT NOT NULL,
                dependency_type TEXT,
                line_number INTEGER,
                column_number INTEGER,
                source_text TEXT,
                is_external BOOLEAN DEFAULT FALSE,
                is_dynamic BOOLEAN DEFAULT FALSE,
                confidence_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project_indexes (id),
                FOREIGN KEY (source_file_id) REFERENCES file_entries (id),
                FOREIGN KEY (target_file_id) REFERENCES file_entries (id)
            )
        """)
        
        # Analysis sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                session_name TEXT NOT NULL,
                session_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                progress_percentage INTEGER DEFAULT 0,
                current_phase TEXT,
                files_total INTEGER DEFAULT 0,
                files_processed INTEGER DEFAULT 0,
                dependencies_found INTEGER DEFAULT 0,
                errors_count INTEGER DEFAULT 0,
                error_log TEXT DEFAULT '[]',
                configuration TEXT DEFAULT '{}',
                result_data TEXT DEFAULT '{}',
                performance_metrics TEXT DEFAULT '{}',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project_indexes (id)
            )
        """)
        
        self.connection.commit()
        self.tables_created = True
        logger.info("Mock database tables created")
    
    def insert_project(self, project_data: Dict[str, Any]) -> str:
        """Insert a mock project."""
        project_id = str(uuid.uuid4())
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO project_indexes (
                id, name, description, root_path, git_repository_url, 
                git_branch, status, configuration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id,
            project_data.get('name'),
            project_data.get('description'),
            project_data.get('root_path'),
            project_data.get('git_repository_url'),
            project_data.get('git_branch'),
            project_data.get('status', 'inactive'),
            json.dumps(project_data.get('configuration', {}))
        ))
        
        self.connection.commit()
        return project_id
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a project by ID."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM project_indexes WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        
        if row:
            project = dict(row)
            project['configuration'] = json.loads(project['configuration'])
            return project
        return None
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM project_indexes ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        projects = []
        for row in rows:
            project = dict(row)
            project['configuration'] = json.loads(project['configuration'])
            projects.append(project)
        
        return projects
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


class MockRedis:
    """Mock Redis implementation for testing."""
    
    def __init__(self):
        self.redis = fakeredis.aioredis.FakeRedis()
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value by key."""
        return await self.redis.get(key)
    
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set key-value pair."""
        return await self.redis.set(key, value, ex=ex)
    
    async def delete(self, key: str) -> int:
        """Delete key."""
        return await self.redis.delete(key)
    
    async def ping(self) -> bool:
        """Ping Redis."""
        return await self.redis.ping()
    
    async def xadd(self, stream: str, fields: Dict[str, Any]) -> str:
        """Add to Redis stream."""
        return await self.redis.xadd(stream, fields)
    
    async def xread(self, streams: Dict[str, str]) -> List:
        """Read from Redis streams."""
        return await self.redis.xread(streams)
    
    async def close(self):
        """Close Redis connection."""
        await self.redis.close()


class MockAPIServer:
    """Mock API server for testing Project Index endpoints."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.app = web.Application()
        self.database = MockDatabase()
        self.runner = None
        self.site = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes."""
        # Health endpoint
        self.app.router.add_get('/health', self.health_handler)
        
        # Project Index endpoints
        self.app.router.add_get('/api/project-index/projects', self.list_projects_handler)
        self.app.router.add_post('/api/project-index/projects', self.create_project_handler)
        self.app.router.add_get('/api/project-index/projects/{project_id}', self.get_project_handler)
        self.app.router.add_put('/api/project-index/projects/{project_id}', self.update_project_handler)
        self.app.router.add_delete('/api/project-index/projects/{project_id}', self.delete_project_handler)
        
        # Analysis endpoints
        self.app.router.add_post('/api/project-index/projects/{project_id}/analyze', self.analyze_project_handler)
        self.app.router.add_get('/api/project-index/projects/{project_id}/status', self.project_status_handler)
        
        # Dashboard endpoints
        self.app.router.add_get('/api/dashboard/status', self.dashboard_status_handler)
    
    async def health_handler(self, request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'mock-project-index-api',
            'version': '1.0.0'
        })
    
    async def list_projects_handler(self, request):
        """List all projects."""
        projects = self.database.list_projects()
        return web.json_response({
            'projects': projects,
            'total': len(projects)
        })
    
    async def create_project_handler(self, request):
        """Create a new project."""
        try:
            data = await request.json()
            
            # Validate required fields
            required_fields = ['name', 'root_path']
            for field in required_fields:
                if field not in data:
                    return web.json_response({
                        'error': f'Missing required field: {field}'
                    }, status=400)
            
            # Insert project
            project_id = self.database.insert_project(data)
            project = self.database.get_project(project_id)
            
            return web.json_response(project, status=201)
            
        except json.JSONDecodeError:
            return web.json_response({
                'error': 'Invalid JSON in request body'
            }, status=400)
        except Exception as e:
            return web.json_response({
                'error': f'Internal server error: {str(e)}'
            }, status=500)
    
    async def get_project_handler(self, request):
        """Get project by ID."""
        project_id = request.match_info['project_id']
        project = self.database.get_project(project_id)
        
        if project:
            return web.json_response(project)
        else:
            return web.json_response({
                'error': 'Project not found'
            }, status=404)
    
    async def update_project_handler(self, request):
        """Update project."""
        project_id = request.match_info['project_id']
        
        # For mock, just return the project (no actual update)
        project = self.database.get_project(project_id)
        
        if project:
            return web.json_response(project)
        else:
            return web.json_response({
                'error': 'Project not found'
            }, status=404)
    
    async def delete_project_handler(self, request):
        """Delete project."""
        project_id = request.match_info['project_id']
        project = self.database.get_project(project_id)
        
        if project:
            return web.json_response({
                'message': 'Project deleted successfully'
            })
        else:
            return web.json_response({
                'error': 'Project not found'
            }, status=404)
    
    async def analyze_project_handler(self, request):
        """Analyze project."""
        project_id = request.match_info['project_id']
        project = self.database.get_project(project_id)
        
        if not project:
            return web.json_response({
                'error': 'Project not found'
            }, status=404)
        
        # Mock analysis result
        analysis_result = {
            'project_id': project_id,
            'status': 'completed',
            'files_analyzed': 25,
            'dependencies_found': 15,
            'analysis_duration_ms': 1500,
            'started_at': datetime.utcnow().isoformat(),
            'completed_at': datetime.utcnow().isoformat()
        }
        
        return web.json_response(analysis_result)
    
    async def project_status_handler(self, request):
        """Get project analysis status."""
        project_id = request.match_info['project_id']
        project = self.database.get_project(project_id)
        
        if not project:
            return web.json_response({
                'error': 'Project not found'
            }, status=404)
        
        return web.json_response({
            'project_id': project_id,
            'status': project['status'],
            'last_analysis_at': project.get('last_analysis_at'),
            'file_count': project['file_count'],
            'dependency_count': project['dependency_count']
        })
    
    async def dashboard_status_handler(self, request):
        """Dashboard status endpoint."""
        return web.json_response({
            'status': 'active',
            'projects_count': len(self.database.list_projects()),
            'services': {
                'api': 'healthy',
                'database': 'healthy',
                'redis': 'healthy'
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def start(self):
        """Start the mock API server."""
        self.database.connect()
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
        
        logger.info(f"Mock API server started on http://localhost:{self.port}")
    
    async def stop(self):
        """Stop the mock API server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        if self.database:
            self.database.close()
        
        logger.info("Mock API server stopped")


class MockWebSocketServer:
    """Mock WebSocket server for testing real-time functionality."""
    
    def __init__(self, port: int = 8002):
        self.port = port
        self.app = web.Application()
        self.connected_clients: Set[web.WebSocketResponse] = set()
        self.runner = None
        self.site = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup WebSocket routes."""
        self.app.router.add_get('/ws/dashboard', self.websocket_handler)
        self.app.router.add_get('/health', self.health_handler)
    
    async def health_handler(self, request):
        """Health check for WebSocket server."""
        return web.json_response({
            'status': 'healthy',
            'connected_clients': len(self.connected_clients),
            'service': 'mock-websocket-server'
        })
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.connected_clients.add(ws)
        logger.info(f"WebSocket client connected. Total clients: {len(self.connected_clients)}")
        
        try:
            # Send welcome message
            await ws.send_str(json.dumps({
                'type': 'welcome',
                'message': 'Connected to mock WebSocket server',
                'timestamp': datetime.utcnow().isoformat()
            }))
            
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON',
                            'timestamp': datetime.utcnow().isoformat()
                        }))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
        
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        
        finally:
            self.connected_clients.discard(ws)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.connected_clients)}")
        
        return ws
    
    async def handle_message(self, ws: web.WebSocketResponse, data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        message_type = data.get('type')
        correlation_id = data.get('correlation_id')
        
        if message_type == 'subscribe':
            # Handle subscription
            topic = data.get('topic', 'default')
            await ws.send_str(json.dumps({
                'type': 'subscription_confirmed',
                'topic': topic,
                'correlation_id': correlation_id,
                'timestamp': datetime.utcnow().isoformat()
            }))
        
        elif message_type == 'ping':
            # Handle ping
            await ws.send_str(json.dumps({
                'type': 'pong',
                'correlation_id': correlation_id,
                'timestamp': datetime.utcnow().isoformat()
            }))
        
        else:
            # Echo unknown messages
            await ws.send_str(json.dumps({
                'type': 'echo',
                'original_message': data,
                'timestamp': datetime.utcnow().isoformat()
            }))
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.connected_clients:
            return
        
        message_str = json.dumps(message)
        disconnected_clients = []
        
        for client in self.connected_clients:
            try:
                await client.send_str(message_str)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.discard(client)
    
    async def start(self):
        """Start the mock WebSocket server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
        
        logger.info(f"Mock WebSocket server started on ws://localhost:{self.port}")
    
    async def stop(self):
        """Stop the mock WebSocket server."""
        # Close all client connections
        for client in list(self.connected_clients):
            await client.close()
        
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Mock WebSocket server stopped")


class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(tempfile.mkdtemp(prefix="mock_fs_"))
        self.files: Dict[str, str] = {}  # path -> content
        self.directories: Set[str] = set()
        self.access_log: List[Dict[str, Any]] = []
    
    def create_file(self, path: str, content: str = "") -> Path:
        """Create a mock file."""
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.files[path] = content
        self.directories.add(str(full_path.parent.relative_to(self.base_path)))
        
        # Write actual file for compatibility
        full_path.write_text(content)
        
        self.access_log.append({
            'operation': 'create_file',
            'path': path,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return full_path
    
    def create_directory(self, path: str) -> Path:
        """Create a mock directory."""
        full_path = self.base_path / path
        full_path.mkdir(parents=True, exist_ok=True)
        
        self.directories.add(path)
        
        self.access_log.append({
            'operation': 'create_directory',
            'path': path,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return full_path
    
    def read_file(self, path: str) -> str:
        """Read mock file content."""
        self.access_log.append({
            'operation': 'read_file',
            'path': path,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if path in self.files:
            return self.files[path]
        
        # Try reading actual file
        full_path = self.base_path / path
        if full_path.exists():
            return full_path.read_text()
        
        raise FileNotFoundError(f"File not found: {path}")
    
    def write_file(self, path: str, content: str):
        """Write to mock file."""
        self.files[path] = content
        
        # Write actual file
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        
        self.access_log.append({
            'operation': 'write_file',
            'path': path,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def delete_file(self, path: str):
        """Delete mock file."""
        if path in self.files:
            del self.files[path]
        
        # Delete actual file
        full_path = self.base_path / path
        if full_path.exists():
            full_path.unlink()
        
        self.access_log.append({
            'operation': 'delete_file',
            'path': path,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def list_files(self) -> List[str]:
        """List all mock files."""
        self.access_log.append({
            'operation': 'list_files',
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return list(self.files.keys())
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get file system access log."""
        return self.access_log.copy()
    
    def cleanup(self):
        """Clean up mock file system."""
        import shutil
        if self.base_path.exists():
            shutil.rmtree(self.base_path)


class MockServiceManager:
    """Manager for all mock services."""
    
    def __init__(self):
        self.api_server = MockAPIServer()
        self.websocket_server = MockWebSocketServer()
        self.redis = MockRedis()
        self.file_system = MockFileSystem()
        self.running = False
    
    async def start_all(self):
        """Start all mock services."""
        logger.info("Starting all mock services...")
        
        await self.api_server.start()
        await self.websocket_server.start()
        
        # Redis doesn't need explicit start
        await self.redis.ping()  # Test connection
        
        self.running = True
        logger.info("All mock services started successfully")
    
    async def stop_all(self):
        """Stop all mock services."""
        logger.info("Stopping all mock services...")
        
        await self.api_server.stop()
        await self.websocket_server.stop()
        await self.redis.close()
        self.file_system.cleanup()
        
        self.running = False
        logger.info("All mock services stopped")
    
    def get_service_urls(self) -> Dict[str, str]:
        """Get URLs for all mock services."""
        return {
            'api': f'http://localhost:{self.api_server.port}',
            'websocket': f'ws://localhost:{self.websocket_server.port}/ws/dashboard',
            'redis': 'redis://localhost:6379',  # Mock Redis uses standard URL format
            'file_system': str(self.file_system.base_path)
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all mock services."""
        health = {}
        
        try:
            # Check API server
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{self.api_server.port}/health') as response:
                    health['api'] = response.status == 200
        except:
            health['api'] = False
        
        try:
            # Check WebSocket server
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{self.websocket_server.port}/health') as response:
                    health['websocket'] = response.status == 200
        except:
            health['websocket'] = False
        
        try:
            # Check Redis
            health['redis'] = await self.redis.ping()
        except:
            health['redis'] = False
        
        # File system is always available
        health['file_system'] = self.file_system.base_path.exists()
        
        return health