"""
Universal Project Index Client Library - Python
Provides easy integration with Project Index API for any Python project
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlencode

try:
    import httpx
    import websockets
except ImportError as e:
    raise ImportError(
        f"Required dependencies not installed: {e}\n"
        "Install with: pip install httpx websockets"
    )


class ProjectIndexClient:
    """
    Universal Project Index client for Python applications.
    
    Provides comprehensive access to Project Index API including:
    - Project management
    - Code analysis
    - File operations  
    - Dependency tracking
    - Real-time monitoring
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8100",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        retries: int = 3,
        debug: bool = False
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.debug = debug
        
        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                'User-Agent': 'ProjectIndexClient-Python/1.0.0'
            }
        )
        
        # WebSocket connection
        self.websocket = None
        self.subscribers = {}
        self.ws_task = None
        
        # Logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Initialized Project Index client for {base_url}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the client."""
        logger = logging.getLogger('ProjectIndexClient')
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            logger.setLevel(logging.WARNING)
            logger.addHandler(logging.NullHandler())
            
        return logger

    async def _request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.
        """
        url = f"{self.base_url}/api{endpoint}"
        headers = kwargs.get('headers', {})
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        for attempt in range(1, self.retries + 1):
            try:
                self.logger.debug(f"Request attempt {attempt}: {method} {url}")
                
                response = await self.client.request(
                    method,
                    url,
                    params=params,
                    json=json_data,
                    headers=headers,
                    **kwargs
                )
                
                response.raise_for_status()
                
                if response.headers.get('content-type', '').startswith('application/json'):
                    data = response.json()
                else:
                    data = {'content': response.text}
                
                self.logger.debug(f"Request successful: {response.status_code}")
                return data
                
            except Exception as error:
                self.logger.debug(f"Request attempt {attempt} failed: {error}")
                
                if attempt == self.retries:
                    raise Exception(f"Request failed after {self.retries} attempts: {error}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

    # =============================================================================
    # PROJECT MANAGEMENT
    # =============================================================================

    async def get_project(self) -> Dict[str, Any]:
        """Get current project information."""
        return await self._request('/projects')

    async def create_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update project configuration."""
        return await self._request('/projects', method='POST', json_data=project_data)

    async def update_project(self, project_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update project configuration."""
        return await self._request(f'/projects/{project_id}', method='PATCH', json_data=update_data)

    async def get_project_stats(self) -> Dict[str, Any]:
        """Get project statistics."""
        return await self._request('/projects/stats')

    # =============================================================================
    # ANALYSIS OPERATIONS
    # =============================================================================

    async def analyze_project(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trigger project analysis.
        
        Args:
            options: Analysis options including:
                - type: 'smart', 'full', 'fast', 'custom'
                - force: Force re-analysis
                - include_dependencies: Include dependency analysis
                - include_complexity: Include complexity metrics
                - max_depth: Maximum analysis depth
        """
        analysis_options = {
            'analysis_type': 'smart',
            'force': False,
            'include_dependencies': True,
            'include_complexity': True,
            'max_depth': 3,
            **(options or {})
        }
        
        return await self._request('/projects/analyze', method='POST', json_data=analysis_options)

    async def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis status."""
        return await self._request(f'/analysis/{analysis_id}/status')

    async def get_analysis_results(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis results."""
        return await self._request(f'/analysis/{analysis_id}/results')

    async def get_latest_analysis(self) -> Dict[str, Any]:
        """Get latest analysis results."""
        return await self._request('/analysis/latest')

    async def cancel_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """Cancel running analysis."""
        return await self._request(f'/analysis/{analysis_id}/cancel', method='POST')

    # =============================================================================
    # FILE OPERATIONS
    # =============================================================================

    async def get_file_analysis(self, file_path: str) -> Dict[str, Any]:
        """Get file analysis."""
        params = {'path': file_path}
        return await self._request('/files/analyze', params=params)

    async def get_file_dependencies(self, file_path: str) -> Dict[str, Any]:
        """Get file dependencies."""
        params = {'path': file_path}
        return await self._request('/files/dependencies', params=params)

    async def get_file_history(self, file_path: str, limit: int = 10) -> Dict[str, Any]:
        """Get file history."""
        params = {'path': file_path, 'limit': limit}
        return await self._request('/files/history', params=params)

    async def search_files(
        self,
        query: str,
        language: Optional[str] = None,
        file_type: Optional[str] = None,
        include_content: bool = False,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Search files by content."""
        search_options = {
            'query': query,
            'include_content': include_content,
            'limit': limit
        }
        
        if language:
            search_options['language'] = language
        if file_type:
            search_options['file_type'] = file_type
            
        return await self._request('/files/search', method='POST', json_data=search_options)

    # =============================================================================
    # DEPENDENCY ANALYSIS
    # =============================================================================

    async def get_dependency_graph(
        self,
        format: str = 'json',
        include_external: bool = False,
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """Get dependency graph."""
        params = {
            'format': format,
            'include_external': include_external,
            'max_depth': max_depth
        }
        return await self._request('/dependencies/graph', params=params)

    async def get_dependency_cycles(self) -> Dict[str, Any]:
        """Get dependency cycles."""
        return await self._request('/dependencies/cycles')

    async def get_external_dependencies(self) -> Dict[str, Any]:
        """Get external dependencies."""
        return await self._request('/dependencies/external')

    async def get_dependency_impact(self, file_path: str) -> Dict[str, Any]:
        """Get dependency impact analysis."""
        params = {'path': file_path}
        return await self._request('/dependencies/impact', params=params)

    # =============================================================================
    # REAL-TIME MONITORING
    # =============================================================================

    async def connect_websocket(
        self,
        events: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        auto_reconnect: bool = True
    ) -> None:
        """
        Connect to WebSocket for real-time updates.
        
        Args:
            events: List of event types to subscribe to
            filters: Event filters
            auto_reconnect: Automatically reconnect on disconnect
        """
        ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/api/ws'
        
        if self.api_key:
            ws_url += f'?token={self.api_key}'
        
        self.logger.info(f"Connecting to WebSocket: {ws_url}")
        
        try:
            self.websocket = await websockets.connect(ws_url)
            self.logger.info("WebSocket connected")
            
            # Subscribe to events
            if events:
                await self.subscribe_to_events(events, filters)
            
            # Start message handling task
            self.ws_task = asyncio.create_task(self._handle_websocket_messages(auto_reconnect))
            
        except Exception as error:
            self.logger.error(f"WebSocket connection failed: {error}")
            raise

    async def subscribe_to_events(
        self,
        event_types: Union[str, List[str]],
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Subscribe to specific event types."""
        if not self.websocket:
            raise Exception("WebSocket not connected")
        
        if isinstance(event_types, str):
            event_types = [event_types]
        
        subscription = {
            'action': 'subscribe',
            'event_types': event_types,
            'filters': filters or {}
        }
        
        await self.websocket.send(json.dumps(subscription))
        self.logger.info(f"Subscribed to events: {event_types}")

    def on(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add event listener for specific event types."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = set()
        self.subscribers[event_type].add(callback)

    def off(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove event listener."""
        if event_type in self.subscribers:
            self.subscribers[event_type].discard(callback)

    async def _handle_websocket_messages(self, auto_reconnect: bool) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._emit_event(data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse WebSocket message: {message}")
                except Exception as error:
                    self.logger.error(f"Error handling WebSocket message: {error}")
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
            
            if auto_reconnect:
                self.logger.info("Attempting to reconnect...")
                await asyncio.sleep(5)
                try:
                    await self.connect_websocket(auto_reconnect=auto_reconnect)
                except Exception as error:
                    self.logger.error(f"Reconnection failed: {error}")
        
        except Exception as error:
            self.logger.error(f"WebSocket error: {error}")

    async def _emit_event(self, data: Dict[str, Any]) -> None:
        """Emit event to subscribers."""
        event_type = data.get('type', 'unknown')
        self.logger.debug(f"WebSocket event received: {event_type}")
        
        # Emit to specific event type subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as error:
                    self.logger.error(f"Error in event callback: {error}")
        
        # Emit to wildcard subscribers
        if '*' in self.subscribers:
            for callback in self.subscribers['*']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as error:
                    self.logger.error(f"Error in wildcard callback: {error}")

    async def disconnect(self) -> None:
        """Disconnect WebSocket and cleanup."""
        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
            self.ws_task = None
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.subscribers.clear()
        self.logger.info("Disconnected and cleaned up")

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        try:
            response = await self._request('/health')
            return {'healthy': True, **response}
        except Exception as error:
            return {'healthy': False, 'error': str(error)}

    async def get_api_info(self) -> Dict[str, Any]:
        """Get API information."""
        return await self._request('/info')

    async def export_project(self, format: str = 'json') -> Dict[str, Any]:
        """Export project data."""
        params = {'format': format}
        return await self._request('/projects/export', params=params)

    async def get_configuration(self) -> Dict[str, Any]:
        """Get configuration."""
        return await self._request('/config')

    async def update_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration."""
        return await self._request('/config', method='PATCH', json_data=config)

    # =============================================================================
    # CONTEXT MANAGERS
    # =============================================================================

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close client and cleanup resources."""
        await self.disconnect()
        await self.client.aclose()
        self.logger.info("Client closed")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def analyze_project(
    project_path: Optional[str] = None,
    base_url: str = "http://localhost:8100",
    **options
) -> Dict[str, Any]:
    """
    Quick project analysis with sensible defaults.
    
    Args:
        project_path: Path to project (default: current directory)
        base_url: Project Index API URL
        **options: Additional client and analysis options
    """
    if project_path is None:
        project_path = os.getcwd()
    
    project_path = str(Path(project_path).resolve())
    
    async with ProjectIndexClient(base_url=base_url, **options) as client:
        # Create project if it doesn't exist
        project_name = Path(project_path).name
        await client.create_project({
            'name': project_name,
            'root_path': project_path,
            'description': f'Auto-analyzed project at {project_path}'
        })
        
        # Trigger analysis
        analysis = await client.analyze_project({
            'type': 'smart',
            'include_dependencies': True,
            'include_complexity': True
        })
        
        # Wait for completion and return results
        analysis_id = analysis['analysis_id']
        
        # Poll for completion
        while True:
            status = await client.get_analysis_status(analysis_id)
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
            await asyncio.sleep(1)
        
        if status['status'] == 'completed':
            return await client.get_analysis_results(analysis_id)
        else:
            raise Exception(f"Analysis {status['status']}: {status.get('error', 'Unknown error')}")


class ProjectMonitor:
    """
    Context manager for monitoring project changes in real-time.
    """
    
    def __init__(
        self,
        callback: Callable[[Dict[str, Any]], None],
        events: Optional[List[str]] = None,
        base_url: str = "http://localhost:8100",
        **options
    ):
        self.callback = callback
        self.events = events or ['file_change', 'analysis_progress', 'dependency_changed']
        self.client = ProjectIndexClient(base_url=base_url, **options)
        self.monitor_task = None

    async def __aenter__(self):
        """Start monitoring."""
        await self.client.connect_websocket(
            events=self.events,
            auto_reconnect=True
        )
        
        # Subscribe to all specified events
        for event in self.events:
            self.client.on(event, self.callback)
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring."""
        await self.client.close()


async def monitor_project(
    callback: Callable[[Dict[str, Any]], None],
    events: Optional[List[str]] = None,
    base_url: str = "http://localhost:8100",
    **options
) -> ProjectMonitor:
    """
    Monitor project for real-time changes.
    
    Args:
        callback: Function to call when events occur
        events: List of event types to monitor
        base_url: Project Index API URL
        **options: Additional client options
    
    Returns:
        ProjectMonitor context manager
    """
    return ProjectMonitor(callback, events, base_url, **options)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example usage of the Project Index client."""
        
        # Basic client usage
        async with ProjectIndexClient(debug=True) as client:
            # Health check
            health = await client.health_check()
            print(f"Service healthy: {health['healthy']}")
            
            # Get project info
            try:
                project = await client.get_project()
                print(f"Project: {project['name']}")
            except Exception:
                print("No project configured yet")
            
            # Trigger analysis
            analysis = await client.analyze_project({
                'type': 'smart',
                'include_dependencies': True
            })
            print(f"Analysis started: {analysis['analysis_id']}")
            
            # Get dependency graph
            dependencies = await client.get_dependency_graph()
            print(f"Dependencies: {len(dependencies.get('nodes', []))} nodes")
        
        # Quick project analysis
        try:
            results = await analyze_project('/path/to/project')
            print(f"Quick analysis: {results['files_analyzed']} files analyzed")
        except Exception as e:
            print(f"Quick analysis failed: {e}")
        
        # Real-time monitoring example
        def handle_event(event):
            print(f"Project event: {event['type']} - {event.get('message', '')}")
        
        async with monitor_project(handle_event) as monitor:
            print("Monitoring project for 10 seconds...")
            await asyncio.sleep(10)
        
        print("Example complete!")
    
    # Run example
    asyncio.run(example_usage())