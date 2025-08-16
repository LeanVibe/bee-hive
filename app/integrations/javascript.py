"""
JavaScript/TypeScript Framework Integration Templates for Project Index

Provides code generation and integration patterns for popular JS/TS frameworks:
- Express.js: Middleware integration, route mounting
- Next.js: API routes, middleware, build integration  
- React: Component integration, development tools
- Vue.js: Plugin system, development integration
- Angular: Service integration, CLI integration

Note: These generate JavaScript/TypeScript code that integrates with the Project Index API
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Dict, List

from . import BaseFrameworkAdapter, IntegrationManager
from ..project_index import ProjectIndexConfig


class JSFrameworkAdapter(BaseFrameworkAdapter):
    """Base adapter for JavaScript/TypeScript frameworks."""
    
    def __init__(self, config: Optional[ProjectIndexConfig] = None):
        super().__init__(config)
        self.project_root = Path.cwd()
        self.api_base_url = "http://localhost:8000/project-index"  # Default API URL
    
    def set_api_url(self, url: str) -> None:
        """Set the Project Index API base URL."""
        self.api_base_url = url
    
    def _create_package_json_entry(self) -> Dict[str, Any]:
        """Create package.json entry for Project Index dependency."""
        return {
            "project-index-client": "^1.0.0",
            "axios": "^1.0.0"  # For HTTP requests
        }
    
    def _generate_client_code(self) -> str:
        """Generate JavaScript client code for Project Index API."""
        return f"""
// Project Index API Client
import axios from 'axios';

class ProjectIndexClient {{
    constructor(baseURL = '{self.api_base_url}') {{
        this.api = axios.create({{
            baseURL,
            timeout: 30000,
            headers: {{
                'Content-Type': 'application/json'
            }}
        }});
    }}

    async getStatus() {{
        const response = await this.api.get('/status');
        return response.data;
    }}

    async analyzeProject(projectPath, languages = null) {{
        const response = await this.api.post('/analyze', {{
            project_path: projectPath,
            languages
        }});
        return response.data;
    }}

    async listProjects() {{
        const response = await this.api.get('/projects');
        return response.data;
    }}

    // WebSocket connection for real-time updates
    connectWebSocket(onMessage) {{
        const wsUrl = this.api.defaults.baseURL.replace('http', 'ws') + '/ws';
        const ws = new WebSocket(wsUrl);
        
        ws.onmessage = (event) => {{
            const data = JSON.parse(event.data);
            onMessage(data);
        }};
        
        return ws;
    }}
}}

export default ProjectIndexClient;
"""


class ExpressAdapter(JSFrameworkAdapter):
    """
    Express.js integration adapter.
    
    Generates Express middleware and routes for Project Index integration.
    """
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """
        Generate Express.js integration code.
        
        Args:
            app: Not used for code generation
            **kwargs: Express-specific options
                - middleware_path: Path for middleware file (default: "middleware/projectIndex.js")
                - routes_path: Path for routes file (default: "routes/projectIndex.js")
        """
        middleware_path = kwargs.get('middleware_path', 'middleware/projectIndex.js')
        routes_path = kwargs.get('routes_path', 'routes/projectIndex.js')
        
        self._generate_express_middleware(middleware_path)
        self._generate_express_routes(routes_path)
        self._generate_express_integration_example()
    
    def _generate_express_middleware(self, file_path: str) -> None:
        """Generate Express middleware for Project Index."""
        middleware_code = f"""
// Project Index Express Middleware
import ProjectIndexClient from '../utils/projectIndexClient.js';

const projectIndexClient = new ProjectIndexClient('{self.api_base_url}');

export const projectIndexMiddleware = async (req, res, next) => {{
    try {{
        // Add Project Index client to request object
        req.projectIndex = projectIndexClient;
        
        // Add helper methods
        req.analyzeCurrentProject = async (languages) => {{
            return await projectIndexClient.analyzeProject(process.cwd(), languages);
        }};
        
        next();
    }} catch (error) {{
        console.error('Project Index middleware error:', error);
        next(); // Continue even if Project Index is unavailable
    }}
}};

export const projectIndexErrorHandler = (error, req, res, next) => {{
    if (error.name === 'ProjectIndexError') {{
        res.status(503).json({{
            error: 'Project Index service unavailable',
            message: error.message
        }});
    }} else {{
        next(error);
    }}
}};
"""
        
        self._write_file(file_path, middleware_code)
    
    def _generate_express_routes(self, file_path: str) -> None:
        """Generate Express routes for Project Index."""
        routes_code = f"""
// Project Index Express Routes
import express from 'express';
import ProjectIndexClient from '../utils/projectIndexClient.js';

const router = express.Router();
const projectIndexClient = new ProjectIndexClient('{self.api_base_url}');

// Proxy routes to Project Index API
router.get('/status', async (req, res) => {{
    try {{
        const status = await projectIndexClient.getStatus();
        res.json(status);
    }} catch (error) {{
        res.status(503).json({{ error: 'Project Index unavailable' }});
    }}
}});

router.post('/analyze', async (req, res) => {{
    try {{
        const {{ projectPath, languages }} = req.body;
        const result = await projectIndexClient.analyzeProject(projectPath || process.cwd(), languages);
        res.json(result);
    }} catch (error) {{
        res.status(400).json({{ error: error.message }});
    }}
}});

router.get('/projects', async (req, res) => {{
    try {{
        const projects = await projectIndexClient.listProjects();
        res.json(projects);
    }} catch (error) {{
        res.status(503).json({{ error: 'Project Index unavailable' }});
    }}
}});

// WebSocket proxy endpoint
router.get('/ws-info', (req, res) => {{
    res.json({{
        websocket_url: '{self.api_base_url}/ws'.replace('http', 'ws')
    }});
}});

export default router;
"""
        
        self._write_file(file_path, routes_code)
    
    def _generate_express_integration_example(self) -> None:
        """Generate integration example for Express."""
        example_code = f"""
// Example: Express.js Project Index Integration
import express from 'express';
import {{ projectIndexMiddleware, projectIndexErrorHandler }} from './middleware/projectIndex.js';
import projectIndexRoutes from './routes/projectIndex.js';

const app = express();

// Add Project Index middleware
app.use(projectIndexMiddleware);

// Mount Project Index routes
app.use('/api/project-index', projectIndexRoutes);

// Add error handler
app.use(projectIndexErrorHandler);

// Example route using Project Index
app.get('/api/analyze-current', async (req, res) => {{
    try {{
        const result = await req.analyzeCurrentProject(['javascript', 'typescript']);
        res.json(result);
    }} catch (error) {{
        res.status(500).json({{ error: error.message }});
    }}
}});

app.listen(3000, () => {{
    console.log('🚀 Express server with Project Index integration running on port 3000');
}});
"""
        
        self._write_file('examples/express-integration.js', example_code)
    
    def _setup_routes(self, app: Any) -> None:
        """Not applicable for code generation."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Not applicable for code generation."""
        pass


class NextJSAdapter(JSFrameworkAdapter):
    """
    Next.js integration adapter.
    
    Generates Next.js API routes and middleware for Project Index integration.
    """
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """
        Generate Next.js integration code.
        
        Args:
            app: Not used for code generation  
            **kwargs: Next.js-specific options
                - api_dir: API directory path (default: "pages/api" or "app/api")
                - middleware_enabled: Generate middleware (default: True)
        """
        api_dir = kwargs.get('api_dir', self._detect_nextjs_api_dir())
        middleware_enabled = kwargs.get('middleware_enabled', True)
        
        self._generate_nextjs_api_routes(api_dir)
        if middleware_enabled:
            self._generate_nextjs_middleware()
        self._generate_nextjs_hook()
        self._generate_nextjs_integration_example()
    
    def _detect_nextjs_api_dir(self) -> str:
        """Detect Next.js API directory structure."""
        if (self.project_root / "app").exists():
            return "app/api"  # App Router
        else:
            return "pages/api"  # Pages Router
    
    def _generate_nextjs_api_routes(self, api_dir: str) -> None:
        """Generate Next.js API routes for Project Index."""
        
        # Status endpoint
        status_code = f"""
// Next.js API Route: Project Index Status
import {{ NextRequest, NextResponse }} from 'next/server';

export async function GET() {{
    try {{
        const response = await fetch('{self.api_base_url}/status');
        const data = await response.json();
        return NextResponse.json(data);
    }} catch (error) {{
        return NextResponse.json(
            {{ error: 'Project Index unavailable' }},
            {{ status: 503 }}
        );
    }}
}}
"""
        
        # Analyze endpoint
        analyze_code = f"""
// Next.js API Route: Project Index Analysis
import {{ NextRequest, NextResponse }} from 'next/server';

export async function POST(request: NextRequest) {{
    try {{
        const body = await request.json();
        const {{ projectPath, languages }} = body;
        
        const response = await fetch('{self.api_base_url}/analyze', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json'
            }},
            body: JSON.stringify({{
                project_path: projectPath || process.cwd(),
                languages
            }})
        }});
        
        const data = await response.json();
        return NextResponse.json(data);
    }} catch (error) {{
        return NextResponse.json(
            {{ error: error.message }},
            {{ status: 400 }}
        );
    }}
}}
"""
        
        self._write_file(f"{api_dir}/project-index/status/route.ts", status_code)
        self._write_file(f"{api_dir}/project-index/analyze/route.ts", analyze_code)
    
    def _generate_nextjs_middleware(self) -> None:
        """Generate Next.js middleware for Project Index."""
        middleware_code = f"""
// Next.js Middleware: Project Index Integration
import {{ NextResponse, NextRequest }} from 'next/server';

export function middleware(request: NextRequest) {{
    // Add Project Index headers
    const response = NextResponse.next();
    response.headers.set('X-Project-Index', 'enabled');
    response.headers.set('X-Project-Index-API', '{self.api_base_url}');
    
    return response;
}}

export const config = {{
    matcher: [
        '/api/project-index/:path*',
        '/((?!api|_next/static|_next/image|favicon.ico).*)'
    ]
}};
"""
        
        self._write_file('middleware.ts', middleware_code)
    
    def _generate_nextjs_hook(self) -> None:
        """Generate React hook for Project Index."""
        hook_code = f"""
// React Hook: useProjectIndex
import {{ useState, useEffect }} from 'react';

interface ProjectIndexStatus {{
    status: string;
    initialized: boolean;
    config: {{
        cache_enabled: boolean;
        monitoring_enabled: boolean;
        max_concurrent_analyses: number;
    }};
}}

interface AnalysisResult {{
    project_id: string;
    files_processed: number;
    dependencies_found: number;
    analysis_time: number;
    languages_detected: string[];
}}

export function useProjectIndex() {{
    const [status, setStatus] = useState<ProjectIndexStatus | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchStatus = async () => {{
        try {{
            setLoading(true);
            const response = await fetch('/api/project-index/status');
            const data = await response.json();
            setStatus(data);
            setError(null);
        }} catch (err) {{
            setError(err instanceof Error ? err.message : 'Unknown error');
        }} finally {{
            setLoading(false);
        }}
    }};

    const analyzeProject = async (projectPath?: string, languages?: string[]): Promise<AnalysisResult> => {{
        setLoading(true);
        try {{
            const response = await fetch('/api/project-index/analyze', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify({{ projectPath, languages }})
            }});
            
            if (!response.ok) {{
                throw new Error('Analysis failed');
            }}
            
            const result = await response.json();
            setError(null);
            return result;
        }} catch (err) {{
            const error = err instanceof Error ? err.message : 'Analysis failed';
            setError(error);
            throw new Error(error);
        }} finally {{
            setLoading(false);
        }}
    }};

    useEffect(() => {{
        fetchStatus();
    }}, []);

    return {{
        status,
        loading,
        error,
        fetchStatus,
        analyzeProject
    }};
}}
"""
        
        self._write_file('hooks/useProjectIndex.ts', hook_code)
    
    def _generate_nextjs_integration_example(self) -> None:
        """Generate Next.js integration example."""
        example_code = """
// Example: Next.js Project Index Component
'use client';

import { useProjectIndex } from '../hooks/useProjectIndex';
import { useState } from 'react';

export default function ProjectIndexDashboard() {
    const { status, loading, error, analyzeProject } = useProjectIndex();
    const [analyzing, setAnalyzing] = useState(false);
    const [analysisResult, setAnalysisResult] = useState(null);

    const handleAnalyze = async () => {
        setAnalyzing(true);
        try {
            const result = await analyzeProject();
            setAnalysisResult(result);
        } catch (err) {
            console.error('Analysis failed:', err);
        } finally {
            setAnalyzing(false);
        }
    };

    if (loading) return <div>Loading Project Index status...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div className="p-6 max-w-4xl mx-auto">
            <h1 className="text-2xl font-bold mb-6">Project Index Dashboard</h1>
            
            {status && (
                <div className="bg-green-100 p-4 rounded-lg mb-6">
                    <h2 className="font-semibold">Status: {status.status}</h2>
                    <p>Initialized: {status.initialized ? 'Yes' : 'No'}</p>
                    <p>Cache Enabled: {status.config.cache_enabled ? 'Yes' : 'No'}</p>
                </div>
            )}

            <button
                onClick={handleAnalyze}
                disabled={analyzing}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
            >
                {analyzing ? 'Analyzing...' : 'Analyze Current Project'}
            </button>

            {analysisResult && (
                <div className="mt-6 bg-gray-100 p-4 rounded-lg">
                    <h3 className="font-semibold mb-2">Analysis Results</h3>
                    <p>Files Processed: {analysisResult.files_processed}</p>
                    <p>Dependencies Found: {analysisResult.dependencies_found}</p>
                    <p>Analysis Time: {analysisResult.analysis_time}s</p>
                    <p>Languages: {analysisResult.languages_detected.join(', ')}</p>
                </div>
            )}
        </div>
    );
}
"""
        
        self._write_file('components/ProjectIndexDashboard.tsx', example_code)
    
    def _setup_routes(self, app: Any) -> None:
        """Not applicable for code generation."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Not applicable for code generation."""
        pass


class ReactAdapter(JSFrameworkAdapter):
    """React development tools integration adapter."""
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """Generate React integration components and hooks."""
        self._generate_react_components()
        self._generate_react_dev_tools()
    
    def _generate_react_components(self) -> None:
        """Generate React components for Project Index."""
        # Already generated in NextJS adapter
        pass
    
    def _generate_react_dev_tools(self) -> None:
        """Generate React development tools."""
        dev_tools_code = """
// React DevTools for Project Index
import React from 'react';

export function ProjectIndexDevTools() {
    return (
        <div style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            background: '#333',
            color: 'white',
            padding: '10px',
            borderRadius: '5px',
            fontSize: '12px',
            zIndex: 9999
        }}>
            <div>🔍 Project Index Active</div>
            <div>API: Connected</div>
        </div>
    );
}
"""
        
        self._write_file('devtools/ProjectIndexDevTools.tsx', dev_tools_code)
    
    def _setup_routes(self, app: Any) -> None:
        """Not applicable for React."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Not applicable for React."""
        pass


class VueAdapter(JSFrameworkAdapter):
    """Vue.js plugin integration adapter."""
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """Generate Vue.js plugin for Project Index."""
        self._generate_vue_plugin()
        self._generate_vue_composable()
    
    def _generate_vue_plugin(self) -> None:
        """Generate Vue.js plugin."""
        plugin_code = f"""
// Vue.js Project Index Plugin
import {{ App }} from 'vue';

interface ProjectIndexOptions {{
    apiUrl?: string;
}}

export default {{
    install(app: App, options: ProjectIndexOptions = {{}}) {{
        const apiUrl = options.apiUrl || '{self.api_base_url}';
        
        app.config.globalProperties.$projectIndex = {{
            apiUrl,
            async getStatus() {{
                const response = await fetch(`${{apiUrl}}/status`);
                return response.json();
            }},
            async analyzeProject(projectPath?: string, languages?: string[]) {{
                const response = await fetch(`${{apiUrl}}/analyze`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ project_path: projectPath, languages }})
                }});
                return response.json();
            }}
        }};
        
        app.provide('projectIndex', app.config.globalProperties.$projectIndex);
    }}
}};
"""
        
        self._write_file('plugins/projectIndex.ts', plugin_code)
    
    def _generate_vue_composable(self) -> None:
        """Generate Vue composable for Project Index."""
        composable_code = f"""
// Vue Composable: useProjectIndex
import {{ ref, onMounted, inject }} from 'vue';

export function useProjectIndex() {{
    const projectIndex = inject('projectIndex');
    const status = ref(null);
    const loading = ref(false);
    const error = ref(null);

    const fetchStatus = async () => {{
        try {{
            loading.value = true;
            status.value = await projectIndex.getStatus();
            error.value = null;
        }} catch (err) {{
            error.value = err.message;
        }} finally {{
            loading.value = false;
        }}
    }};

    const analyzeProject = async (projectPath?: string, languages?: string[]) => {{
        loading.value = true;
        try {{
            const result = await projectIndex.analyzeProject(projectPath, languages);
            error.value = null;
            return result;
        }} catch (err) {{
            error.value = err.message;
            throw err;
        }} finally {{
            loading.value = false;
        }}
    }};

    onMounted(() => {{
        fetchStatus();
    }});

    return {{
        status,
        loading,
        error,
        fetchStatus,
        analyzeProject
    }};
}}
"""
        
        self._write_file('composables/useProjectIndex.ts', composable_code)
    
    def _setup_routes(self, app: Any) -> None:
        """Not applicable for Vue."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Not applicable for Vue."""
        pass


class AngularAdapter(JSFrameworkAdapter):
    """Angular service integration adapter."""
    
    def integrate(self, app: Any = None, **kwargs) -> None:
        """Generate Angular service and module for Project Index."""
        self._generate_angular_service()
        self._generate_angular_module()
    
    def _generate_angular_service(self) -> None:
        """Generate Angular service for Project Index."""
        service_code = f"""
// Angular Service: Project Index
import {{ Injectable }} from '@angular/core';
import {{ HttpClient }} from '@angular/common/http';
import {{ Observable }} from 'rxjs';

export interface ProjectIndexStatus {{
    status: string;
    initialized: boolean;
    config: {{
        cache_enabled: boolean;
        monitoring_enabled: boolean;
        max_concurrent_analyses: number;
    }};
}}

export interface AnalysisResult {{
    project_id: string;
    files_processed: number;
    dependencies_found: number;
    analysis_time: number;
    languages_detected: string[];
}}

@Injectable({{
    providedIn: 'root'
}})
export class ProjectIndexService {{
    private readonly apiUrl = '{self.api_base_url}';

    constructor(private http: HttpClient) {{}}

    getStatus(): Observable<ProjectIndexStatus> {{
        return this.http.get<ProjectIndexStatus>(`${{this.apiUrl}}/status`);
    }}

    analyzeProject(projectPath?: string, languages?: string[]): Observable<AnalysisResult> {{
        return this.http.post<AnalysisResult>(`${{this.apiUrl}}/analyze`, {{
            project_path: projectPath,
            languages
        }});
    }}

    listProjects(): Observable<any> {{
        return this.http.get(`${{this.apiUrl}}/projects`);
    }}
}}
"""
        
        self._write_file('services/project-index.service.ts', service_code)
    
    def _generate_angular_module(self) -> None:
        """Generate Angular module for Project Index."""
        module_code = """
// Angular Module: Project Index
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { ProjectIndexService } from './services/project-index.service';

@NgModule({
    imports: [
        CommonModule,
        HttpClientModule
    ],
    providers: [
        ProjectIndexService
    ]
})
export class ProjectIndexModule {}
"""
        
        self._write_file('project-index.module.ts', module_code)
    
    def _setup_routes(self, app: Any) -> None:
        """Not applicable for Angular."""
        pass
    
    def _setup_middleware(self, app: Any) -> None:
        """Not applicable for Angular."""
        pass
    
    def _write_file(self, file_path: str, content: str) -> None:
        """Write content to file, creating directories as needed."""
        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            full_path.write_text(content)
            print(f"✅ Generated {file_path}")
        except Exception as e:
            print(f"❌ Failed to generate {file_path}: {e}")


# Add _write_file method to base JSFrameworkAdapter
JSFrameworkAdapter._write_file = AngularAdapter._write_file

# Register JavaScript/TypeScript adapters
IntegrationManager.register_adapter('express', ExpressAdapter)
IntegrationManager.register_adapter('nextjs', NextJSAdapter)
IntegrationManager.register_adapter('react', ReactAdapter)
IntegrationManager.register_adapter('vue', VueAdapter)
IntegrationManager.register_adapter('angular', AngularAdapter)


# Convenience functions for code generation
def generate_express_integration(api_url: str = "http://localhost:8000/project-index", **kwargs) -> ExpressAdapter:
    """Generate Express.js integration code."""
    adapter = ExpressAdapter()
    adapter.set_api_url(api_url)
    adapter.integrate(**kwargs)
    return adapter


def generate_nextjs_integration(api_url: str = "http://localhost:8000/project-index", **kwargs) -> NextJSAdapter:
    """Generate Next.js integration code."""
    adapter = NextJSAdapter()
    adapter.set_api_url(api_url)
    adapter.integrate(**kwargs)
    return adapter


def generate_react_integration(api_url: str = "http://localhost:8000/project-index", **kwargs) -> ReactAdapter:
    """Generate React integration code."""
    adapter = ReactAdapter()
    adapter.set_api_url(api_url)
    adapter.integrate(**kwargs)
    return adapter


def generate_vue_integration(api_url: str = "http://localhost:8000/project-index", **kwargs) -> VueAdapter:
    """Generate Vue.js integration code."""
    adapter = VueAdapter()
    adapter.set_api_url(api_url)
    adapter.integrate(**kwargs)
    return adapter


def generate_angular_integration(api_url: str = "http://localhost:8000/project-index", **kwargs) -> AngularAdapter:
    """Generate Angular integration code."""
    adapter = AngularAdapter()
    adapter.set_api_url(api_url)
    adapter.integrate(**kwargs)
    return adapter