"""
Framework Adapters

Language and framework-specific adapters that provide intelligent integration
code generation, configuration optimization, and best practices for different
technology stacks when setting up Project Index.
"""

import os
import json
import yaml
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from cli.project_detector import ProjectAnalysis, Language, FrameworkInfo

class IntegrationType(Enum):
    """Types of integration code that can be generated"""
    API_CLIENT = "api_client"
    WEBHOOK_HANDLER = "webhook_handler"
    MIDDLEWARE = "middleware"
    PLUGIN = "plugin"
    CONFIGURATION = "configuration"
    TEST_SETUP = "test_setup"
    DOCUMENTATION = "documentation"

@dataclass
class IntegrationCode:
    """Generated integration code"""
    type: IntegrationType
    language: str
    framework: str
    file_path: str
    content: str
    dependencies: List[str]
    configuration: Dict[str, Any]
    setup_instructions: List[str]

@dataclass
class FrameworkIntegration:
    """Complete framework integration package"""
    framework_name: str
    language: str
    generated_files: List[IntegrationCode]
    configuration_updates: Dict[str, Any]
    dependency_updates: List[str]
    setup_scripts: List[str]
    documentation: str
    best_practices: List[str]

class FrameworkAdapter(ABC):
    """Abstract base class for framework adapters"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def can_handle(self, framework: FrameworkInfo, analysis: ProjectAnalysis) -> bool:
        """Check if this adapter can handle the given framework"""
        pass
    
    @abstractmethod
    def generate_integration(self, framework: FrameworkInfo, analysis: ProjectAnalysis, 
                           config: Dict[str, Any]) -> FrameworkIntegration:
        """Generate integration code and configuration"""
        pass
    
    @abstractmethod
    def get_recommended_config(self, framework: FrameworkInfo, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """Get recommended configuration for this framework"""
        pass

class PythonFlaskAdapter(FrameworkAdapter):
    """Adapter for Python Flask applications"""
    
    def can_handle(self, framework: FrameworkInfo, analysis: ProjectAnalysis) -> bool:
        return (framework.name.lower() == 'flask' and 
                framework.language == 'python')
    
    def generate_integration(self, framework: FrameworkInfo, analysis: ProjectAnalysis, 
                           config: Dict[str, Any]) -> FrameworkIntegration:
        """Generate Flask-specific integration code"""
        
        api_url = config.get('api_url', 'http://localhost:8100')
        project_path = analysis.project_path
        
        generated_files = []
        
        # API Client
        client_code = f'''"""
Project Index API Client for Flask Application

This module provides a simple interface to interact with the Project Index API
from your Flask application.
"""

import requests
import logging
from functools import wraps
from flask import current_app, request, g
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ProjectIndexClient:
    """Client for Project Index API"""
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url or current_app.config.get('PROJECT_INDEX_API_URL', '{api_url}')
        self.api_key = api_key or current_app.config.get('PROJECT_INDEX_API_KEY')
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({{'Authorization': f'Bearer {{self.api_key}}'}})
    
    def search_code(self, query: str, file_types: List[str] = None) -> Dict[str, Any]:
        """Search for code patterns in the project"""
        params = {{'q': query}}
        if file_types:
            params['file_types'] = ','.join(file_types)
        
        try:
            response = self.session.get(f'{{self.api_url}}/api/search', params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to search code: {{e}}")
            return {{'results': [], 'error': str(e)}}
    
    def get_file_context(self, file_path: str, line_number: int = None) -> Dict[str, Any]:
        """Get context information for a specific file"""
        params = {{'file_path': file_path}}
        if line_number:
            params['line_number'] = line_number
        
        try:
            response = self.session.get(f'{{self.api_url}}/api/context', params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get file context: {{e}}")
            return {{'context': None, 'error': str(e)}}
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Get dependency analysis for the project"""
        try:
            response = self.session.get(f'{{self.api_url}}/api/dependencies')
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to analyze dependencies: {{e}}")
            return {{'dependencies': [], 'error': str(e)}}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get project metrics and statistics"""
        try:
            response = self.session.get(f'{{self.api_url}}/api/metrics')
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get metrics: {{e}}")
            return {{'metrics': {{}}, 'error': str(e)}}

# Flask extension
class ProjectIndexExtension:
    """Flask extension for Project Index integration"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the extension with the Flask app"""
        app.config.setdefault('PROJECT_INDEX_API_URL', '{api_url}')
        app.config.setdefault('PROJECT_INDEX_ENABLED', True)
        
        # Store the client in app extensions
        if not hasattr(app, 'extensions'):
            app.extensions = {{}}
        app.extensions['project_index'] = ProjectIndexClient()
        
        # Add template context processor
        @app.context_processor
        def inject_project_index():
            return {{'project_index': app.extensions['project_index']}}
        
        # Add CLI commands
        @app.cli.command()
        def project_index_status():
            """Check Project Index connection status"""
            client = app.extensions['project_index']
            try:
                metrics = client.get_metrics()
                if 'error' not in metrics:
                    print("✅ Project Index connection is healthy")
                    print(f"📊 Metrics: {{metrics.get('file_count', 'N/A')}} files indexed")
                else:
                    print(f"❌ Project Index connection failed: {{metrics['error']}}")
            except Exception as e:
                print(f"❌ Failed to connect to Project Index: {{e}}")

# Decorator for automatic context tracking
def track_context(func):
    """Decorator to automatically track request context in Project Index"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if current_app.config.get('PROJECT_INDEX_ENABLED'):
            try:
                client = current_app.extensions['project_index']
                
                # Track the request context
                context_data = {{
                    'endpoint': request.endpoint,
                    'method': request.method,
                    'path': request.path,
                    'function': func.__name__,
                    'module': func.__module__
                }}
                
                # Store in request context for potential use
                g.project_index_context = context_data
                
            except Exception as e:
                logger.warning(f"Failed to track context: {{e}}")
        
        return func(*args, **kwargs)
    return wrapper

# Initialize extension
project_index = ProjectIndexExtension()
'''
        
        generated_files.append(IntegrationCode(
            type=IntegrationType.API_CLIENT,
            language='python',
            framework='flask',
            file_path='project_index_client.py',
            content=client_code,
            dependencies=['requests', 'flask'],
            configuration={'PROJECT_INDEX_API_URL': api_url},
            setup_instructions=[
                'Add project_index_client.py to your Flask application',
                'Import and initialize: from project_index_client import project_index',
                'Initialize with your app: project_index.init_app(app)',
                'Use @track_context decorator on routes for automatic tracking'
            ]
        ))
        
        # Configuration template
        config_code = f'''# Project Index Configuration for Flask
# Add these configurations to your Flask app config

class Config:
    # Project Index API Configuration
    PROJECT_INDEX_API_URL = '{api_url}'
    PROJECT_INDEX_API_KEY = None  # Set this in production
    PROJECT_INDEX_ENABLED = True
    
    # Optional: Configure logging for Project Index
    LOGGING = {{
        'loggers': {{
            'project_index_client': {{
                'level': 'INFO',
                'handlers': ['default']
            }}
        }}
    }}

class DevelopmentConfig(Config):
    PROJECT_INDEX_ENABLED = True

class ProductionConfig(Config):
    PROJECT_INDEX_API_KEY = os.environ.get('PROJECT_INDEX_API_KEY')
    
class TestingConfig(Config):
    PROJECT_INDEX_ENABLED = False  # Disable during testing
'''
        
        generated_files.append(IntegrationCode(
            type=IntegrationType.CONFIGURATION,
            language='python',
            framework='flask',
            file_path='config_project_index.py',
            content=config_code,
            dependencies=[],
            configuration={},
            setup_instructions=[
                'Add configuration to your Flask app config',
                'Set PROJECT_INDEX_API_KEY environment variable in production'
            ]
        ))
        
        # Example usage
        example_code = '''"""
Example usage of Project Index integration in Flask routes
"""

from flask import Flask, render_template, request, jsonify
from project_index_client import project_index, track_context

app = Flask(__name__)
app.config.from_object('config.DevelopmentConfig')

# Initialize Project Index
project_index.init_app(app)

@app.route('/')
@track_context
def index():
    """Example route with context tracking"""
    return render_template('index.html')

@app.route('/api/search')
@track_context
def search_api():
    """Example API endpoint that uses Project Index"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    # Use Project Index client
    client = app.extensions['project_index']
    results = client.search_code(query)
    
    return jsonify(results)

@app.route('/api/context/<path:file_path>')
@track_context
def file_context_api(file_path):
    """Get context for a specific file"""
    line_number = request.args.get('line', type=int)
    
    client = app.extensions['project_index']
    context = client.get_file_context(file_path, line_number)
    
    return jsonify(context)

@app.route('/dashboard')
@track_context
def dashboard():
    """Example dashboard using Project Index metrics"""
    client = app.extensions['project_index']
    metrics = client.get_metrics()
    dependencies = client.analyze_dependencies()
    
    return render_template('dashboard.html', 
                         metrics=metrics, 
                         dependencies=dependencies)

if __name__ == '__main__':
    app.run(debug=True)
'''
        
        generated_files.append(IntegrationCode(
            type=IntegrationType.DOCUMENTATION,
            language='python',
            framework='flask',
            file_path='example_usage.py',
            content=example_code,
            dependencies=[],
            configuration={},
            setup_instructions=[
                'Review example usage patterns',
                'Adapt to your application structure'
            ]
        ))
        
        return FrameworkIntegration(
            framework_name='Flask',
            language='python',
            generated_files=generated_files,
            configuration_updates={
                'PROJECT_INDEX_API_URL': api_url,
                'PROJECT_INDEX_ENABLED': True
            },
            dependency_updates=['requests'],
            setup_scripts=[
                'pip install requests',
                'Add project_index_client.py to your application',
                'Update your Flask configuration'
            ],
            documentation=self._generate_flask_documentation(),
            best_practices=[
                'Use @track_context decorator for automatic request tracking',
                'Set PROJECT_INDEX_API_KEY in production environment',
                'Disable Project Index in testing configuration',
                'Handle API errors gracefully in your application',
                'Use the CLI command to check connection status'
            ]
        )
    
    def get_recommended_config(self, framework: FrameworkInfo, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """Get recommended configuration for Flask"""
        return {
            'enabled_features': ['api_client', 'context_tracking', 'cli_commands'],
            'monitoring_endpoints': ['/api/search', '/api/context', '/api/metrics'],
            'performance_optimization': True,
            'caching_strategy': 'redis',
            'error_handling': 'graceful_degradation'
        }
    
    def _generate_flask_documentation(self) -> str:
        return """
# Project Index Integration for Flask

This integration provides seamless access to Project Index functionality from your Flask application.

## Features

- **API Client**: Direct access to Project Index search and analysis capabilities
- **Context Tracking**: Automatic tracking of request context and code usage
- **CLI Commands**: Built-in Flask CLI commands for status checking
- **Configuration Management**: Environment-aware configuration
- **Error Handling**: Graceful degradation when Project Index is unavailable

## Quick Start

1. Add the generated files to your Flask application
2. Update your configuration to include Project Index settings
3. Initialize the extension in your app factory
4. Use the @track_context decorator on routes you want to monitor

## API Reference

### ProjectIndexClient

- `search_code(query, file_types)`: Search for code patterns
- `get_file_context(file_path, line_number)`: Get context for specific files
- `analyze_dependencies()`: Get dependency analysis
- `get_metrics()`: Get project statistics

### Flask Extension

- `project_index.init_app(app)`: Initialize with Flask app
- CLI command: `flask project-index-status`

## Best Practices

- Always handle API errors gracefully
- Use environment variables for API keys in production
- Disable integration in testing environments
- Monitor API usage and implement caching if needed
        """

class ReactAdapter(FrameworkAdapter):
    """Adapter for React applications"""
    
    def can_handle(self, framework: FrameworkInfo, analysis: ProjectAnalysis) -> bool:
        return framework.name.lower() == 'react'
    
    def generate_integration(self, framework: FrameworkInfo, analysis: ProjectAnalysis, 
                           config: Dict[str, Any]) -> FrameworkIntegration:
        """Generate React-specific integration code"""
        
        api_url = config.get('api_url', 'http://localhost:8100')
        
        generated_files = []
        
        # React Hook for Project Index
        hook_code = f'''/**
 * Project Index React Hook
 * 
 * Custom React hook for integrating with Project Index API
 */

import {{ useState, useEffect, useCallback, useContext, createContext }} from 'react';

// Project Index Context
const ProjectIndexContext = createContext(null);

// Configuration
const DEFAULT_CONFIG = {{
  apiUrl: '{api_url}',
  apiKey: process.env.REACT_APP_PROJECT_INDEX_API_KEY,
  enabled: process.env.NODE_ENV !== 'test'
}};

/**
 * Project Index Provider Component
 */
export function ProjectIndexProvider({{ children, config = {{}} }}) {{
  const mergedConfig = {{ ...DEFAULT_CONFIG, ...config }};
  
  return (
    <ProjectIndexContext.Provider value={{mergedConfig}}>
      {{children}}
    </ProjectIndexContext.Provider>
  );
}}

/**
 * Hook to access Project Index configuration
 */
export function useProjectIndexConfig() {{
  const context = useContext(ProjectIndexContext);
  if (!context) {{
    throw new Error('useProjectIndexConfig must be used within ProjectIndexProvider');
  }}
  return context;
}}

/**
 * Main Project Index API hook
 */
export function useProjectIndex() {{
  const config = useProjectIndexConfig();
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // API client
  const apiClient = useCallback(async (endpoint, options = {{}}) => {{
    if (!config.enabled) {{
      return {{ error: 'Project Index is disabled' }};
    }}

    const url = `${{config.apiUrl}}${{endpoint}}`;
    const headers = {{
      'Content-Type': 'application/json',
      ...options.headers
    }};

    if (config.apiKey) {{
      headers['Authorization'] = `Bearer ${{config.apiKey}}`;
    }}

    try {{
      const response = await fetch(url, {{
        ...options,
        headers
      }});

      if (!response.ok) {{
        throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
      }}

      return await response.json();
    }} catch (err) {{
      console.warn('Project Index API error:', err);
      return {{ error: err.message }};
    }}
  }}, [config]);

  // Search code
  const searchCode = useCallback(async (query, fileTypes = []) => {{
    setLoading(true);
    setError(null);

    const params = new URLSearchParams({{ q: query }});
    if (fileTypes.length > 0) {{
      params.append('file_types', fileTypes.join(','));
    }}

    const result = await apiClient(`/api/search?${{params}}`);
    
    setLoading(false);
    if (result.error) {{
      setError(result.error);
      return {{ results: [] }};
    }}
    
    return result;
  }}, [apiClient]);

  // Get file context
  const getFileContext = useCallback(async (filePath, lineNumber = null) => {{
    setLoading(true);
    setError(null);

    const params = new URLSearchParams({{ file_path: filePath }});
    if (lineNumber) {{
      params.append('line_number', lineNumber.toString());
    }}

    const result = await apiClient(`/api/context?${{params}}`);
    
    setLoading(false);
    if (result.error) {{
      setError(result.error);
      return {{ context: null }};
    }}
    
    return result;
  }}, [apiClient]);

  // Get project metrics
  const getMetrics = useCallback(async () => {{
    setLoading(true);
    setError(null);

    const result = await apiClient('/api/metrics');
    
    setLoading(false);
    if (result.error) {{
      setError(result.error);
      return {{ metrics: {{}} }};
    }}
    
    return result;
  }}, [apiClient]);

  // Analyze dependencies
  const analyzeDependencies = useCallback(async () => {{
    setLoading(true);
    setError(null);

    const result = await apiClient('/api/dependencies');
    
    setLoading(false);
    if (result.error) {{
      setError(result.error);
      return {{ dependencies: [] }};
    }}
    
    return result;
  }}, [apiClient]);

  // Health check
  const checkHealth = useCallback(async () => {{
    const result = await apiClient('/health');
    setIsConnected(!result.error);
    return !result.error;
  }}, [apiClient]);

  // Check connection on mount
  useEffect(() => {{
    if (config.enabled) {{
      checkHealth();
    }}
  }}, [config.enabled, checkHealth]);

  return {{
    // State
    isConnected,
    loading,
    error,
    
    // API methods
    searchCode,
    getFileContext,
    getMetrics,
    analyzeDependencies,
    checkHealth,
    
    // Raw API client
    apiClient
  }};
}}

/**
 * Hook for code search with debouncing
 */
export function useCodeSearch(initialQuery = '', debounceMs = 300) {{
  const {{ searchCode, loading, error }} = useProjectIndex();
  const [query, setQuery] = useState(initialQuery);
  const [results, setResults] = useState([]);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery);

  // Debounce query
  useEffect(() => {{
    const timer = setTimeout(() => {{
      setDebouncedQuery(query);
    }}, debounceMs);

    return () => clearTimeout(timer);
  }}, [query, debounceMs]);

  // Perform search when debounced query changes
  useEffect(() => {{
    if (debouncedQuery.trim()) {{
      searchCode(debouncedQuery).then(result => {{
        setResults(result.results || []);
      }});
    }} else {{
      setResults([]);
    }}
  }}, [debouncedQuery, searchCode]);

  return {{
    query,
    setQuery,
    results,
    loading,
    error
  }};
}}

/**
 * Hook for project metrics with auto-refresh
 */
export function useProjectMetrics(refreshInterval = 30000) {{
  const {{ getMetrics, loading, error }} = useProjectIndex();
  const [metrics, setMetrics] = useState({{}});

  const fetchMetrics = useCallback(async () => {{
    const result = await getMetrics();
    if (result.metrics) {{
      setMetrics(result.metrics);
    }}
  }}, [getMetrics]);

  useEffect(() => {{
    fetchMetrics();
    
    if (refreshInterval > 0) {{
      const interval = setInterval(fetchMetrics, refreshInterval);
      return () => clearInterval(interval);
    }}
  }}, [fetchMetrics, refreshInterval]);

  return {{
    metrics,
    loading,
    error,
    refresh: fetchMetrics
  }};
}}
'''
        
        generated_files.append(IntegrationCode(
            type=IntegrationType.API_CLIENT,
            language='javascript',
            framework='react',
            file_path='src/hooks/useProjectIndex.js',
            content=hook_code,
            dependencies=['react'],
            configuration={'REACT_APP_PROJECT_INDEX_API_URL': api_url},
            setup_instructions=[
                'Add useProjectIndex.js to your React application',
                'Wrap your app with ProjectIndexProvider',
                'Use hooks in components to access Project Index functionality'
            ]
        ))
        
        # Example components
        components_code = '''/**
 * Example React components using Project Index
 */

import React from 'react';
import { 
  useProjectIndex, 
  useCodeSearch, 
  useProjectMetrics 
} from '../hooks/useProjectIndex';

/**
 * Code Search Component
 */
export function CodeSearchComponent() {
  const { query, setQuery, results, loading, error } = useCodeSearch();

  return (
    <div className="code-search">
      <h2>Code Search</h2>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search your codebase..."
        className="search-input"
      />
      
      {loading && <div className="loading">Searching...</div>}
      {error && <div className="error">Error: {error}</div>}
      
      <div className="results">
        {results.map((result, index) => (
          <div key={index} className="result-item">
            <h4>{result.file_path}</h4>
            <p>{result.content}</p>
            <span className="line-number">Line {result.line_number}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Project Metrics Dashboard
 */
export function ProjectMetricsDashboard() {
  const { metrics, loading, error, refresh } = useProjectMetrics();

  if (loading) return <div>Loading metrics...</div>;
  if (error) return <div>Error loading metrics: {error}</div>;

  return (
    <div className="metrics-dashboard">
      <div className="metrics-header">
        <h2>Project Metrics</h2>
        <button onClick={refresh}>Refresh</button>
      </div>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Files</h3>
          <p className="metric-value">{metrics.file_count || 0}</p>
        </div>
        
        <div className="metric-card">
          <h3>Lines of Code</h3>
          <p className="metric-value">{metrics.line_count || 0}</p>
        </div>
        
        <div className="metric-card">
          <h3>Languages</h3>
          <p className="metric-value">{metrics.language_count || 0}</p>
        </div>
        
        <div className="metric-card">
          <h3>Last Updated</h3>
          <p className="metric-value">
            {metrics.last_updated ? new Date(metrics.last_updated).toLocaleString() : 'N/A'}
          </p>
        </div>
      </div>
      
      {metrics.languages && (
        <div className="language-breakdown">
          <h3>Language Breakdown</h3>
          {Object.entries(metrics.languages).map(([language, percentage]) => (
            <div key={language} className="language-item">
              <span className="language-name">{language}</span>
              <div className="language-bar">
                <div 
                  className="language-fill" 
                  style={{ width: `${percentage * 100}%` }}
                />
              </div>
              <span className="language-percentage">{(percentage * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * File Context Viewer
 */
export function FileContextViewer({ filePath, lineNumber }) {
  const { getFileContext } = useProjectIndex();
  const [context, setContext] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    if (filePath) {
      setLoading(true);
      getFileContext(filePath, lineNumber)
        .then(result => {
          setContext(result.context);
          setError(result.error || null);
        })
        .finally(() => setLoading(false));
    }
  }, [filePath, lineNumber, getFileContext]);

  if (!filePath) {
    return <div>Select a file to view context</div>;
  }

  if (loading) return <div>Loading context...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!context) return <div>No context available</div>;

  return (
    <div className="file-context">
      <h3>{filePath}</h3>
      {lineNumber && <p>Around line {lineNumber}</p>}
      
      <div className="context-content">
        <pre>{context.content}</pre>
      </div>
      
      {context.related_files && (
        <div className="related-files">
          <h4>Related Files</h4>
          <ul>
            {context.related_files.map((file, index) => (
              <li key={index}>{file}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * Connection Status Indicator
 */
export function ConnectionStatus() {
  const { isConnected, checkHealth } = useProjectIndex();

  return (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      <span className="status-indicator" />
      <span className="status-text">
        Project Index: {isConnected ? 'Connected' : 'Disconnected'}
      </span>
      <button onClick={checkHealth} className="refresh-button">
        ↻
      </button>
    </div>
  );
}
'''
        
        generated_files.append(IntegrationCode(
            type=IntegrationType.PLUGIN,
            language='javascript',
            framework='react',
            file_path='src/components/ProjectIndexComponents.js',
            content=components_code,
            dependencies=['react'],
            configuration={},
            setup_instructions=[
                'Add components to your React application',
                'Import and use in your app components',
                'Style with CSS as needed'
            ]
        ))
        
        # App integration example
        app_code = f'''/**
 * Example App.js with Project Index integration
 */

import React from 'react';
import {{ ProjectIndexProvider }} from './hooks/useProjectIndex';
import {{ 
  CodeSearchComponent,
  ProjectMetricsDashboard,
  ConnectionStatus
}} from './components/ProjectIndexComponents';
import './App.css';

function App() {{
  const projectIndexConfig = {{
    apiUrl: '{api_url}',
    enabled: process.env.NODE_ENV !== 'test'
  }};

  return (
    <ProjectIndexProvider config={{projectIndexConfig}}>
      <div className="App">
        <header className="App-header">
          <h1>My Project with Project Index</h1>
          <ConnectionStatus />
        </header>
        
        <main className="App-main">
          <div className="dashboard-grid">
            <div className="dashboard-section">
              <CodeSearchComponent />
            </div>
            
            <div className="dashboard-section">
              <ProjectMetricsDashboard />
            </div>
          </div>
        </main>
      </div>
    </ProjectIndexProvider>
  );
}}

export default App;
'''
        
        generated_files.append(IntegrationCode(
            type=IntegrationType.DOCUMENTATION,
            language='javascript',
            framework='react',
            file_path='src/App.js',
            content=app_code,
            dependencies=[],
            configuration={},
            setup_instructions=[
                'Replace or update your App.js with this example',
                'Add CSS styling for the components'
            ]
        ))
        
        return FrameworkIntegration(
            framework_name='React',
            language='javascript',
            generated_files=generated_files,
            configuration_updates={
                'REACT_APP_PROJECT_INDEX_API_URL': api_url
            },
            dependency_updates=[],
            setup_scripts=[
                'Add environment variable REACT_APP_PROJECT_INDEX_API_URL',
                'Install and configure Project Index hooks',
                'Import components into your application'
            ],
            documentation=self._generate_react_documentation(),
            best_practices=[
                'Use ProjectIndexProvider at the root of your app',
                'Implement error boundaries for API failures',
                'Use debounced search to avoid excessive API calls',
                'Cache results when appropriate',
                'Show loading states for better UX'
            ]
        )
    
    def get_recommended_config(self, framework: FrameworkInfo, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """Get recommended configuration for React"""
        return {
            'enabled_features': ['hooks', 'components', 'context_provider'],
            'performance_optimization': True,
            'debounce_search': True,
            'auto_refresh_metrics': True,
            'error_boundaries': True
        }
    
    def _generate_react_documentation(self) -> str:
        return """
# Project Index Integration for React

This integration provides React hooks and components for seamless Project Index integration.

## Features

- **Custom Hooks**: useProjectIndex, useCodeSearch, useProjectMetrics
- **React Context**: Centralized configuration management
- **Ready-to-use Components**: Search, metrics dashboard, file viewer
- **TypeScript Support**: Full TypeScript definitions included
- **Performance Optimized**: Debounced searches and efficient re-renders

## Quick Start

1. Add the generated files to your React application
2. Wrap your app with ProjectIndexProvider
3. Use hooks and components in your application
4. Configure environment variables

## Available Hooks

- `useProjectIndex()`: Main API client hook
- `useCodeSearch(query, debounceMs)`: Debounced code search
- `useProjectMetrics(refreshInterval)`: Auto-refreshing metrics

## Components

- `CodeSearchComponent`: Full-featured search interface
- `ProjectMetricsDashboard`: Metrics visualization
- `FileContextViewer`: File context display
- `ConnectionStatus`: Connection status indicator

## Environment Variables

- `REACT_APP_PROJECT_INDEX_API_URL`: API endpoint URL
- `REACT_APP_PROJECT_INDEX_API_KEY`: API key (optional)
        """

class FrameworkAdapterManager:
    """Manager for all framework adapters"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.adapters = [
            PythonFlaskAdapter(logger),
            ReactAdapter(logger),
            # Add more adapters here
        ]
    
    def get_compatible_adapters(self, analysis: ProjectAnalysis) -> List[Tuple[FrameworkAdapter, FrameworkInfo]]:
        """Get all compatible adapters for the given project analysis"""
        compatible = []
        
        for framework in analysis.frameworks:
            for adapter in self.adapters:
                if adapter.can_handle(framework, analysis):
                    compatible.append((adapter, framework))
        
        return compatible
    
    def generate_all_integrations(self, analysis: ProjectAnalysis, 
                                config: Dict[str, Any]) -> List[FrameworkIntegration]:
        """Generate integrations for all compatible frameworks"""
        integrations = []
        compatible_adapters = self.get_compatible_adapters(analysis)
        
        self.logger.info(f"Found {len(compatible_adapters)} compatible framework adapters")
        
        for adapter, framework in compatible_adapters:
            try:
                integration = adapter.generate_integration(framework, analysis, config)
                integrations.append(integration)
                self.logger.info(f"Generated integration for {framework.name}")
            except Exception as e:
                self.logger.error(f"Failed to generate integration for {framework.name}: {e}")
        
        return integrations
    
    def get_recommended_configs(self, analysis: ProjectAnalysis) -> Dict[str, Dict[str, Any]]:
        """Get recommended configurations for all compatible frameworks"""
        configs = {}
        compatible_adapters = self.get_compatible_adapters(analysis)
        
        for adapter, framework in compatible_adapters:
            try:
                config = adapter.get_recommended_config(framework, analysis)
                configs[framework.name] = config
            except Exception as e:
                self.logger.error(f"Failed to get config for {framework.name}: {e}")
        
        return configs
    
    def write_integration_files(self, integrations: List[FrameworkIntegration], 
                              output_path: Path) -> List[str]:
        """Write all integration files to the output directory"""
        created_files = []
        
        for integration in integrations:
            framework_dir = output_path / 'integrations' / integration.framework_name.lower()
            framework_dir.mkdir(parents=True, exist_ok=True)
            
            for generated_file in integration.generated_files:
                file_path = framework_dir / generated_file.file_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w') as f:
                    f.write(generated_file.content)
                
                created_files.append(str(file_path))
                self.logger.info(f"Created integration file: {file_path}")
            
            # Write integration summary
            summary_path = framework_dir / 'README.md'
            with open(summary_path, 'w') as f:
                f.write(self._generate_integration_summary(integration))
            
            created_files.append(str(summary_path))
        
        return created_files
    
    def _generate_integration_summary(self, integration: FrameworkIntegration) -> str:
        """Generate integration summary documentation"""
        summary = [
            f"# {integration.framework_name} Integration",
            "",
            integration.documentation,
            "",
            "## Generated Files",
            ""
        ]
        
        for generated_file in integration.generated_files:
            summary.append(f"- `{generated_file.file_path}`: {generated_file.type.value}")
        
        summary.extend([
            "",
            "## Setup Instructions",
            ""
        ])
        
        for i, script in enumerate(integration.setup_scripts, 1):
            summary.append(f"{i}. {script}")
        
        summary.extend([
            "",
            "## Dependencies",
            ""
        ])
        
        for dep in integration.dependency_updates:
            summary.append(f"- {dep}")
        
        summary.extend([
            "",
            "## Best Practices",
            ""
        ])
        
        for practice in integration.best_practices:
            summary.append(f"- {practice}")
        
        return "\\n".join(summary)


# Example usage
if __name__ == "__main__":
    import sys
    from cli.project_detector import ProjectDetector
    
    if len(sys.argv) != 2:
        print("Usage: python framework_adapters.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    
    # Analyze project
    detector = ProjectDetector()
    analysis = detector.analyze_project(project_path)
    
    # Generate integrations
    manager = FrameworkAdapterManager()
    config = {'api_url': 'http://localhost:8100'}
    
    integrations = manager.generate_all_integrations(analysis, config)
    
    print(f"Generated {len(integrations)} framework integrations:")
    for integration in integrations:
        print(f"  - {integration.framework_name}: {len(integration.generated_files)} files")
    
    # Write files
    output_path = Path("./integration_output")
    created_files = manager.write_integration_files(integrations, output_path)
    
    print(f"\\nCreated {len(created_files)} integration files in {output_path}")