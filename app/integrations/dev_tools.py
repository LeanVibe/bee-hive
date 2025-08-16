"""
Development Tools and IDE Extensions for Project Index

Provides development tools, IDE integrations, and editor extensions
to make working with Project Index seamless for developers.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class IDEExtensionGenerator:
    """Generator for IDE extensions and development tools."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.api_base_url = "http://localhost:8000/project-index"
    
    def generate_vscode_extension(self) -> None:
        """Generate VS Code extension for Project Index."""
        self._create_vscode_manifest()
        self._create_vscode_extension_code()
        self._create_vscode_commands()
        self._create_vscode_settings()
    
    def _create_vscode_manifest(self) -> None:
        """Create VS Code extension manifest."""
        package_json = {
            "name": "project-index-integration",
            "displayName": "Project Index Integration",
            "description": "Seamless Project Index integration for VS Code",
            "version": "1.0.0",
            "publisher": "leanvibe",
            "engines": {
                "vscode": "^1.60.0"
            },
            "categories": [
                "Other",
                "Debuggers",
                "Extension Packs"
            ],
            "keywords": [
                "project-index",
                "code-analysis",
                "dependency-tracking",
                "ai-assistance"
            ],
            "activationEvents": [
                "onStartupFinished",
                "workspaceContains:**/.project-index.json"
            ],
            "main": "./out/extension.js",
            "contributes": {
                "commands": [
                    {
                        "command": "projectIndex.analyze",
                        "title": "Analyze Project",
                        "category": "Project Index"
                    },
                    {
                        "command": "projectIndex.status",
                        "title": "Show Status",
                        "category": "Project Index"
                    },
                    {
                        "command": "projectIndex.setup",
                        "title": "Setup Integration",
                        "category": "Project Index"
                    },
                    {
                        "command": "projectIndex.showDashboard",
                        "title": "Show Dashboard",
                        "category": "Project Index"
                    }
                ],
                "menus": {
                    "explorer/context": [
                        {
                            "command": "projectIndex.analyze",
                            "group": "navigation",
                            "when": "explorerResourceIsFolder"
                        }
                    ],
                    "editor/context": [
                        {
                            "command": "projectIndex.analyze",
                            "group": "navigation"
                        }
                    ]
                },
                "views": {
                    "explorer": [
                        {
                            "id": "projectIndexView",
                            "name": "Project Index",
                            "when": "projectIndexActive"
                        }
                    ]
                },
                "viewsContainers": {
                    "activitybar": [
                        {
                            "id": "projectIndex",
                            "title": "Project Index",
                            "icon": "$(search-view-icon)"
                        }
                    ]
                },
                "configuration": {
                    "title": "Project Index",
                    "properties": {
                        "projectIndex.apiUrl": {
                            "type": "string",
                            "default": "http://localhost:8000/project-index",
                            "description": "Project Index API URL"
                        },
                        "projectIndex.autoAnalyze": {
                            "type": "boolean",
                            "default": true,
                            "description": "Automatically analyze project on startup"
                        },
                        "projectIndex.showStatusBar": {
                            "type": "boolean",
                            "default": true,
                            "description": "Show Project Index status in status bar"
                        }
                    }
                }
            },
            "scripts": {
                "vscode:prepublish": "npm run compile",
                "compile": "tsc -p ./",
                "watch": "tsc -watch -p ./"
            },
            "devDependencies": {
                "@types/vscode": "^1.60.0",
                "@types/node": "^16.0.0",
                "typescript": "^4.4.0"
            },
            "dependencies": {
                "axios": "^1.0.0"
            }
        }
        
        self._write_file('.vscode-extension/package.json', json.dumps(package_json, indent=2))
    
    def _create_vscode_extension_code(self) -> None:
        """Create VS Code extension TypeScript code."""
        extension_code = f"""
// VS Code Extension for Project Index
import * as vscode from 'vscode';
import axios from 'axios';

interface ProjectIndexConfig {{
    apiUrl: string;
    autoAnalyze: boolean;
    showStatusBar: boolean;
}}

interface ProjectStatus {{
    status: string;
    initialized: boolean;
    config: {{
        cache_enabled: boolean;
        monitoring_enabled: boolean;
        max_concurrent_analyses: number;
    }};
}}

class ProjectIndexExtension {{
    private statusBarItem: vscode.StatusBarItem;
    private outputChannel: vscode.OutputChannel;
    private webviewPanel: vscode.WebviewPanel | undefined;
    
    constructor(private context: vscode.ExtensionContext) {{
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
        this.outputChannel = vscode.window.createOutputChannel('Project Index');
        
        this.setupCommands();
        this.setupStatusBar();
        
        if (this.getConfig().autoAnalyze) {{
            this.analyzeProject();
        }}
    }}
    
    private getConfig(): ProjectIndexConfig {{
        const config = vscode.workspace.getConfiguration('projectIndex');
        return {{
            apiUrl: config.get('apiUrl', '{self.api_base_url}'),
            autoAnalyze: config.get('autoAnalyze', true),
            showStatusBar: config.get('showStatusBar', true)
        }};
    }}
    
    private setupCommands(): void {{
        const commands = [
            vscode.commands.registerCommand('projectIndex.analyze', () => this.analyzeProject()),
            vscode.commands.registerCommand('projectIndex.status', () => this.showStatus()),
            vscode.commands.registerCommand('projectIndex.setup', () => this.setupIntegration()),
            vscode.commands.registerCommand('projectIndex.showDashboard', () => this.showDashboard())
        ];
        
        commands.forEach(cmd => this.context.subscriptions.push(cmd));
    }}
    
    private setupStatusBar(): void {{
        if (this.getConfig().showStatusBar) {{
            this.statusBarItem.text = "$(search) Project Index";
            this.statusBarItem.command = 'projectIndex.status';
            this.statusBarItem.show();
            this.context.subscriptions.push(this.statusBarItem);
        }}
    }}
    
    private async analyzeProject(): Promise<void> {{
        const config = this.getConfig();
        const workspaceFolders = vscode.workspace.workspaceFolders;
        
        if (!workspaceFolders) {{
            vscode.window.showErrorMessage('No workspace folder open');
            return;
        }}
        
        const projectPath = workspaceFolders[0].uri.fsPath;
        
        try {{
            this.statusBarItem.text = "$(sync~spin) Analyzing...";
            this.outputChannel.appendLine(`Analyzing project: ${{projectPath}}`);
            
            const response = await axios.post(`${{config.apiUrl}}/analyze`, {{
                project_path: projectPath,
                languages: ['javascript', 'typescript', 'python', 'go', 'rust', 'java']
            }});
            
            const result = response.data;
            this.statusBarItem.text = `$(check) ${{result.files_processed}} files`;
            
            this.outputChannel.appendLine(`Analysis complete:`);
            this.outputChannel.appendLine(`  Files processed: ${{result.files_processed}}`);
            this.outputChannel.appendLine(`  Dependencies found: ${{result.dependencies_found}}`);
            this.outputChannel.appendLine(`  Analysis time: ${{result.analysis_time}}s`);
            this.outputChannel.appendLine(`  Languages detected: ${{result.languages_detected.join(', ')}}`);
            
            vscode.window.showInformationMessage(
                `Project analyzed: ${{result.files_processed}} files, ${{result.dependencies_found}} dependencies`
            );
            
        }} catch (error) {{
            this.statusBarItem.text = "$(error) Analysis failed";
            this.outputChannel.appendLine(`Analysis failed: ${{error}}`);
            vscode.window.showErrorMessage('Project analysis failed. Check Project Index service.');
        }}
    }}
    
    private async showStatus(): Promise<void> {{
        const config = this.getConfig();
        
        try {{
            const response = await axios.get(`${{config.apiUrl}}/status`);
            const status: ProjectStatus = response.data;
            
            const message = `Status: ${{status.status}} | Initialized: ${{status.initialized ? 'Yes' : 'No'}} | Cache: ${{status.config.cache_enabled ? 'On' : 'Off'}}`;
            
            vscode.window.showInformationMessage(message);
            this.outputChannel.appendLine(`Project Index Status: ${{JSON.stringify(status, null, 2)}}`);
            
        }} catch (error) {{
            vscode.window.showErrorMessage('Cannot connect to Project Index service');
            this.outputChannel.appendLine(`Status check failed: ${{error}}`);
        }}
    }}
    
    private async setupIntegration(): Promise<void> {{
        const frameworks = [
            'FastAPI', 'Django', 'Flask', 'Express.js', 'Next.js', 
            'React', 'Vue.js', 'Angular', 'Go (Gin)', 'Rust (Axum)', 'Java (Spring Boot)'
        ];
        
        const selectedFramework = await vscode.window.showQuickPick(frameworks, {{
            placeHolder: 'Select your framework for integration'
        }});
        
        if (selectedFramework) {{
            const terminal = vscode.window.createTerminal('Project Index Setup');
            const frameworkMap: {{ [key: string]: string }} = {{
                'FastAPI': 'fastapi',
                'Django': 'django',
                'Flask': 'flask',
                'Express.js': 'express',
                'Next.js': 'nextjs',
                'React': 'react',
                'Vue.js': 'vue',
                'Angular': 'angular',
                'Go (Gin)': 'go',
                'Rust (Axum)': 'rust',
                'Java (Spring Boot)': 'java'
            }};
            
            const framework = frameworkMap[selectedFramework];
            terminal.sendText(`python -m app.integrations.cli setup --framework ${{framework}}`);
            terminal.show();
        }}
    }}
    
    private showDashboard(): void {{
        if (this.webviewPanel) {{
            this.webviewPanel.reveal();
            return;
        }}
        
        this.webviewPanel = vscode.window.createWebviewPanel(
            'projectIndexDashboard',
            'Project Index Dashboard',
            vscode.ViewColumn.One,
            {{
                enableScripts: true,
                retainContextWhenHidden: true
            }}
        );
        
        this.webviewPanel.webview.html = this.getDashboardHtml();
        
        this.webviewPanel.onDidDispose(() => {{
            this.webviewPanel = undefined;
        }});
    }}
    
    private getDashboardHtml(): string {{
        const config = this.getConfig();
        
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Project Index Dashboard</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }}
                .card {{ background: #f8f9fa; border-radius: 8px; padding: 16px; margin: 16px 0; }}
                .status {{ display: flex; align-items: center; gap: 8px; }}
                .status.active {{ color: #28a745; }}
                .status.inactive {{ color: #dc3545; }}
                button {{ background: #007acc; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }}
                button:hover {{ background: #005a9e; }}
            </style>
        </head>
        <body>
            <h1>🔍 Project Index Dashboard</h1>
            
            <div class="card">
                <h3>Service Status</h3>
                <div id="status" class="status">
                    <span>Checking...</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Quick Actions</h3>
                <button onclick="analyzeProject()">Analyze Project</button>
                <button onclick="refreshStatus()">Refresh Status</button>
                <button onclick="openAPI()">Open API Docs</button>
            </div>
            
            <div class="card">
                <h3>Recent Analysis</h3>
                <div id="analysis-results">
                    <p>No recent analysis data</p>
                </div>
            </div>
            
            <script>
                const API_URL = '${{config.apiUrl}}';
                
                async function checkStatus() {{
                    try {{
                        const response = await fetch(`${{API_URL}}/status`);
                        const data = await response.json();
                        
                        const statusEl = document.getElementById('status');
                        statusEl.className = 'status active';
                        statusEl.innerHTML = `
                            <span>✅ Active</span>
                            <span>| Initialized: ${{data.initialized ? 'Yes' : 'No'}}</span>
                            <span>| Cache: ${{data.config.cache_enabled ? 'On' : 'Off'}}</span>
                        `;
                    }} catch (error) {{
                        const statusEl = document.getElementById('status');
                        statusEl.className = 'status inactive';
                        statusEl.innerHTML = '<span>❌ Disconnected</span>';
                    }}
                }}
                
                async function analyzeProject() {{
                    // This would trigger the VS Code command
                    console.log('Analysis triggered');
                }}
                
                function refreshStatus() {{
                    checkStatus();
                }}
                
                function openAPI() {{
                    // This would open the API documentation
                    console.log('Opening API docs');
                }}
                
                // Initialize
                checkStatus();
                setInterval(checkStatus, 30000); // Refresh every 30 seconds
            </script>
        </body>
        </html>
        `;
    }}
}}

export function activate(context: vscode.ExtensionContext) {{
    const extension = new ProjectIndexExtension(context);
    
    // Set context for when extension is active
    vscode.commands.executeCommand('setContext', 'projectIndexActive', true);
}}

export function deactivate() {{
    vscode.commands.executeCommand('setContext', 'projectIndexActive', false);
}}
"""
        
        self._write_file('.vscode-extension/src/extension.ts', extension_code)
    
    def _create_vscode_commands(self) -> None:
        """Create VS Code commands configuration."""
        tsconfig = {
            "compilerOptions": {
                "module": "commonjs",
                "target": "es6",
                "outDir": "out",
                "lib": ["es6"],
                "sourceMap": True,
                "rootDir": "src",
                "strict": True
            },
            "exclude": ["node_modules", ".vscode-test"]
        }
        
        self._write_file('.vscode-extension/tsconfig.json', json.dumps(tsconfig, indent=2))
    
    def _create_vscode_settings(self) -> None:
        """Create VS Code workspace settings for Project Index."""
        settings = {
            "projectIndex.apiUrl": self.api_base_url,
            "projectIndex.autoAnalyze": True,
            "projectIndex.showStatusBar": True,
            "files.associations": {
                ".project-index.json": "json"
            },
            "json.schemas": [
                {
                    "fileMatch": [".project-index.json"],
                    "url": f"{self.api_base_url}/schema"
                }
            ]
        }
        
        self._write_file('.vscode/settings.json', json.dumps(settings, indent=2))
    
    def generate_intellij_plugin(self) -> None:
        """Generate IntelliJ IDEA plugin for Project Index."""
        self._create_intellij_manifest()
        self._create_intellij_plugin_code()
    
    def _create_intellij_manifest(self) -> None:
        """Create IntelliJ plugin manifest."""
        plugin_xml = f"""
<?xml version="1.0" encoding="UTF-8"?>
<idea-plugin>
    <id>com.leanvibe.projectindex</id>
    <name>Project Index Integration</name>
    <vendor email="team@leanvibe.com" url="https://leanvibe.com">LeanVibe</vendor>
    
    <description><![CDATA[
        Seamless Project Index integration for IntelliJ IDEA and other JetBrains IDEs.
        Provides code analysis, dependency tracking, and AI-powered insights.
    ]]></description>
    
    <change-notes><![CDATA[
        <ul>
            <li>Initial release with basic Project Index integration</li>
            <li>Real-time code analysis</li>
            <li>Dependency tracking</li>
            <li>AI-powered insights</li>
        </ul>
    ]]></change-notes>
    
    <idea-version since-build="203"/>
    
    <depends>com.intellij.modules.platform</depends>
    <depends>com.intellij.modules.java</depends>
    
    <extensions defaultExtensionNs="com.intellij">
        <!-- Tool Window -->
        <toolWindow id="ProjectIndex" 
                   secondary="true" 
                   anchor="right" 
                   factoryClass="com.leanvibe.projectindex.toolwindow.ProjectIndexToolWindowFactory"/>
        
        <!-- Actions -->
        <action id="ProjectIndex.Analyze" 
                class="com.leanvibe.projectindex.actions.AnalyzeProjectAction" 
                text="Analyze with Project Index"
                description="Analyze current project with Project Index">
            <add-to-group group-id="ProjectViewPopupMenu" anchor="last"/>
            <add-to-group group-id="EditorPopupMenu" anchor="last"/>
        </action>
        
        <!-- Status Bar -->
        <statusBarWidgetFactory implementation="com.leanvibe.projectindex.statusbar.ProjectIndexStatusBarWidgetFactory" 
                               id="ProjectIndexStatus"/>
        
        <!-- Settings -->
        <applicationConfigurable id="ProjectIndexSettings" 
                               displayName="Project Index" 
                               instance="com.leanvibe.projectindex.settings.ProjectIndexConfigurable"/>
        
        <!-- Startup Activity -->
        <postStartupActivity implementation="com.leanvibe.projectindex.startup.ProjectIndexStartupActivity"/>
    </extensions>
    
    <applicationListeners>
        <listener class="com.leanvibe.projectindex.listeners.ProjectIndexFileListener" 
                 topic="com.intellij.openapi.vfs.VirtualFileListener"/>
    </applicationListeners>
</idea-plugin>
"""
        
        self._write_file('.intellij-plugin/src/main/resources/META-INF/plugin.xml', plugin_xml)
    
    def _create_intellij_plugin_code(self) -> None:
        """Create IntelliJ plugin Java code."""
        
        # Main plugin class
        plugin_class = f"""
package com.leanvibe.projectindex;

import com.intellij.openapi.project.Project;
import com.intellij.openapi.startup.StartupActivity;
import org.jetbrains.annotations.NotNull;

public class ProjectIndexPlugin implements StartupActivity {{
    
    @Override
    public void runActivity(@NotNull Project project) {{
        // Initialize Project Index integration
        ProjectIndexService service = ProjectIndexService.getInstance(project);
        if (service.isAutoAnalyzeEnabled()) {{
            service.analyzeProjectAsync();
        }}
    }}
}}
"""
        
        # Service class
        service_class = f"""
package com.leanvibe.projectindex;

import com.intellij.openapi.components.Service;
import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.notification.NotificationDisplayType;
import com.intellij.notification.NotificationGroup;
import com.intellij.notification.NotificationType;
import com.intellij.notification.Notifications;

import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.util.concurrent.CompletableFuture;

@Service
public final class ProjectIndexService {{
    
    private static final String DEFAULT_API_URL = "{self.api_base_url}";
    private static final NotificationGroup NOTIFICATION_GROUP = 
        NotificationGroup.balloonGroup("Project Index");
    
    private final Project project;
    private final HttpClient httpClient;
    
    public ProjectIndexService(Project project) {{
        this.project = project;
        this.httpClient = HttpClient.newHttpClient();
    }}
    
    public static ProjectIndexService getInstance(Project project) {{
        return ServiceManager.getService(project, ProjectIndexService.class);
    }}
    
    public CompletableFuture<Void> analyzeProjectAsync() {{
        return CompletableFuture.runAsync(() -> {{
            try {{
                String projectPath = project.getBasePath();
                if (projectPath == null) return;
                
                String requestBody = String.format(
                    "{{\"project_path\": \"%s\", \"languages\": [\"java\", \"kotlin\", \"javascript\", \"python\"]}}",
                    projectPath.replace("\\\\", "\\\\\\\\")
                );
                
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(getApiUrl() + "/analyze"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .build();
                
                HttpResponse<String> response = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofString());
                
                if (response.statusCode() == 200) {{
                    ApplicationManager.getApplication().invokeLater(() -> {{
                        NOTIFICATION_GROUP.createNotification(
                            "Project Index", 
                            "Project analysis completed successfully",
                            NotificationType.INFORMATION,
                            null
                        ).notify(project);
                    }});
                }} else {{
                    showError("Analysis failed with status: " + response.statusCode());
                }}
                
            }} catch (Exception e) {{
                showError("Analysis failed: " + e.getMessage());
            }}
        }});
    }}
    
    public CompletableFuture<String> getStatusAsync() {{
        return CompletableFuture.supplyAsync(() -> {{
            try {{
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(getApiUrl() + "/status"))
                    .GET()
                    .build();
                
                HttpResponse<String> response = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofString());
                
                return response.body();
            }} catch (Exception e) {{
                return "{{\"error\": \"" + e.getMessage() + "\"}}";
            }}
        }});
    }}
    
    private void showError(String message) {{
        ApplicationManager.getApplication().invokeLater(() -> {{
            NOTIFICATION_GROUP.createNotification(
                "Project Index Error", 
                message,
                NotificationType.ERROR,
                null
            ).notify(project);
        }});
    }}
    
    private String getApiUrl() {{
        // Get from settings, fallback to default
        return DEFAULT_API_URL;
    }}
    
    public boolean isAutoAnalyzeEnabled() {{
        // Get from settings
        return true;
    }}
}}
"""
        
        self._write_file('.intellij-plugin/src/main/java/com/leanvibe/projectindex/ProjectIndexPlugin.java', plugin_class)
        self._write_file('.intellij-plugin/src/main/java/com/leanvibe/projectindex/ProjectIndexService.java', service_class)
    
    def generate_dev_server_integration(self) -> None:
        """Generate development server integration scripts."""
        
        # Webpack plugin for JavaScript projects
        webpack_plugin = f"""
// Webpack Plugin for Project Index Integration
class ProjectIndexWebpackPlugin {{
    constructor(options = {{}}) {{
        this.options = {{
            apiUrl: '{self.api_base_url}',
            autoAnalyze: true,
            ...options
        }};
    }}
    
    apply(compiler) {{
        const pluginName = 'ProjectIndexWebpackPlugin';
        
        compiler.hooks.afterCompile.tapAsync(pluginName, (compilation, callback) => {{
            if (this.options.autoAnalyze) {{
                this.analyzeProject().then(() => callback()).catch(() => callback());
            }} else {{
                callback();
            }}
        }});
        
        compiler.hooks.done.tap(pluginName, (stats) => {{
            console.log('🔍 Project Index: Build complete, analysis available');
        }});
    }}
    
    async analyzeProject() {{
        try {{
            const response = await fetch(`${{this.options.apiUrl}}/analyze`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    project_path: process.cwd(),
                    languages: ['javascript', 'typescript']
                }})
            }});
            
            if (response.ok) {{
                const result = await response.json();
                console.log(`🔍 Project Index: Analyzed ${{result.files_processed}} files`);
            }}
        }} catch (error) {{
            console.warn('🔍 Project Index: Analysis failed', error.message);
        }}
    }}
}}

module.exports = ProjectIndexWebpackPlugin;

// Usage example:
// const ProjectIndexWebpackPlugin = require('./project-index-webpack-plugin');
//
// module.exports = {{
//   plugins: [
//     new ProjectIndexWebpackPlugin({{ apiUrl: 'http://localhost:8000/project-index' }})
//   ]
// }};
"""
        
        # Vite plugin for modern JavaScript projects
        vite_plugin = f"""
// Vite Plugin for Project Index Integration
import {{ Plugin }} from 'vite';

interface ProjectIndexOptions {{
    apiUrl?: string;
    autoAnalyze?: boolean;
}}

export function projectIndexPlugin(options: ProjectIndexOptions = {{}}): Plugin {{
    const config = {{
        apiUrl: '{self.api_base_url}',
        autoAnalyze: true,
        ...options
    }};
    
    return {{
        name: 'project-index',
        buildStart() {{
            console.log('🔍 Project Index: Starting analysis...');
            if (config.autoAnalyze) {{
                this.analyzeProject();
            }}
        }},
        buildEnd() {{
            console.log('🔍 Project Index: Build complete');
        }},
        async analyzeProject() {{
            try {{
                const response = await fetch(`${{config.apiUrl}}/analyze`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        project_path: process.cwd(),
                        languages: ['javascript', 'typescript', 'vue', 'react']
                    }})
                }});
                
                if (response.ok) {{
                    const result = await response.json();
                    console.log(`🔍 Project Index: Analyzed ${{result.files_processed}} files`);
                }}
            }} catch (error) {{
                console.warn('🔍 Project Index: Analysis failed', error.message);
            }}
        }}
    }};
}}

// Usage example:
// import {{ defineConfig }} from 'vite';
// import {{ projectIndexPlugin }} from './project-index-vite-plugin';
//
// export default defineConfig({{
//   plugins: [
//     projectIndexPlugin({{ apiUrl: 'http://localhost:8000/project-index' }})
//   ]
// }});
"""
        
        # Rollup plugin
        rollup_plugin = f"""
// Rollup Plugin for Project Index Integration
export function projectIndexPlugin(options = {{}}) {{
    const config = {{
        apiUrl: '{self.api_base_url}',
        autoAnalyze: true,
        ...options
    }};
    
    return {{
        name: 'project-index',
        buildStart() {{
            console.log('🔍 Project Index: Starting analysis...');
            if (config.autoAnalyze) {{
                this.analyzeProject();
            }}
        }},
        generateBundle() {{
            console.log('🔍 Project Index: Bundle generated');
        }},
        async analyzeProject() {{
            try {{
                const fetch = (await import('node-fetch')).default;
                const response = await fetch(`${{config.apiUrl}}/analyze`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        project_path: process.cwd(),
                        languages: ['javascript', 'typescript']
                    }})
                }});
                
                if (response.ok) {{
                    const result = await response.json();
                    console.log(`🔍 Project Index: Analyzed ${{result.files_processed}} files`);
                }}
            }} catch (error) {{
                console.warn('🔍 Project Index: Analysis failed', error.message);
            }}
        }}
    }};
}}

// Usage example:
// import {{ projectIndexPlugin }} from './project-index-rollup-plugin.js';
//
// export default {{
//   plugins: [
//     projectIndexPlugin({{ apiUrl: 'http://localhost:8000/project-index' }})
//   ]
// }};
"""
        
        self._write_file('dev-tools/project-index-webpack-plugin.js', webpack_plugin)
        self._write_file('dev-tools/project-index-vite-plugin.js', vite_plugin)
        self._write_file('dev-tools/project-index-rollup-plugin.js', rollup_plugin)
    
    def generate_browser_devtools(self) -> None:
        """Generate browser developer tools integration."""
        
        # Chrome extension manifest
        chrome_manifest = {
            "manifest_version": 3,
            "name": "Project Index DevTools",
            "version": "1.0.0",
            "description": "Browser developer tools for Project Index integration",
            "permissions": [
                "activeTab",
                "storage"
            ],
            "host_permissions": [
                "http://localhost:8000/*",
                "https://*.leanvibe.com/*"
            ],
            "devtools_page": "devtools.html",
            "background": {
                "service_worker": "background.js"
            },
            "action": {
                "default_popup": "popup.html",
                "default_title": "Project Index"
            },
            "icons": {
                "16": "icons/icon16.png",
                "48": "icons/icon48.png",
                "128": "icons/icon128.png"
            }
        }
        
        # DevTools panel
        devtools_panel = f"""
// Browser DevTools Panel for Project Index
class ProjectIndexDevToolsPanel {{
    constructor() {{
        this.apiUrl = '{self.api_base_url}';
        this.init();
    }}
    
    init() {{
        this.createPanel();
        this.setupEventListeners();
        this.loadStatus();
    }}
    
    createPanel() {{
        chrome.devtools.panels.create(
            'Project Index',
            'icons/icon16.png',
            'panel.html',
            (panel) => {{
                this.panel = panel;
                panel.onShown.addListener(this.onPanelShown.bind(this));
                panel.onHidden.addListener(this.onPanelHidden.bind(this));
            }}
        );
    }}
    
    onPanelShown(window) {{
        this.panelWindow = window;
        this.setupPanelInterface();
    }}
    
    onPanelHidden() {{
        // Cleanup when panel is hidden
    }}
    
    setupPanelInterface() {{
        if (!this.panelWindow) return;
        
        const doc = this.panelWindow.document;
        
        // Create interface
        doc.body.innerHTML = `
            <div style="padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                <h2>🔍 Project Index</h2>
                
                <div id="status-section">
                    <h3>Service Status</h3>
                    <div id="status">Loading...</div>
                    <button id="refresh-status">Refresh</button>
                </div>
                
                <div id="analysis-section">
                    <h3>Project Analysis</h3>
                    <button id="analyze-project">Analyze Current Project</button>
                    <div id="analysis-results"></div>
                </div>
                
                <div id="network-section">
                    <h3>Network Requests</h3>
                    <div id="network-log"></div>
                </div>
            </div>
        `;
        
        // Add event listeners
        doc.getElementById('refresh-status').onclick = () => this.loadStatus();
        doc.getElementById('analyze-project').onclick = () => this.analyzeProject();
    }}
    
    async loadStatus() {{
        try {{
            const response = await fetch(`${{this.apiUrl}}/status`);
            const status = await response.json();
            
            const statusEl = this.panelWindow.document.getElementById('status');
            statusEl.innerHTML = `
                <div style="color: green;">✅ Connected</div>
                <div>Status: ${{status.status}}</div>
                <div>Initialized: ${{status.initialized}}</div>
                <div>Cache: ${{status.config.cache_enabled ? 'Enabled' : 'Disabled'}}</div>
            `;
        }} catch (error) {{
            const statusEl = this.panelWindow.document.getElementById('status');
            statusEl.innerHTML = '<div style="color: red;">❌ Disconnected</div>';
        }}
    }}
    
    async analyzeProject() {{
        try {{
            const response = await fetch(`${{this.apiUrl}}/analyze`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    project_path: '.',
                    languages: ['javascript', 'typescript', 'html', 'css']
                }})
            }});
            
            const result = await response.json();
            const resultsEl = this.panelWindow.document.getElementById('analysis-results');
            resultsEl.innerHTML = `
                <div>✅ Analysis Complete</div>
                <div>Files: ${{result.files_processed}}</div>
                <div>Dependencies: ${{result.dependencies_found}}</div>
                <div>Time: ${{result.analysis_time}}s</div>
                <div>Languages: ${{result.languages_detected.join(', ')}}</div>
            `;
        }} catch (error) {{
            const resultsEl = this.panelWindow.document.getElementById('analysis-results');
            resultsEl.innerHTML = '<div style="color: red;">❌ Analysis failed</div>';
        }}
    }}
    
    setupEventListeners() {{
        // Listen for network requests
        chrome.devtools.network.onRequestFinished.addListener((request) => {{
            if (request.request.url.includes('project-index')) {{
                this.logNetworkRequest(request);
            }}
        }});
    }}
    
    logNetworkRequest(request) {{
        if (!this.panelWindow) return;
        
        const networkEl = this.panelWindow.document.getElementById('network-log');
        const logEntry = this.panelWindow.document.createElement('div');
        logEntry.innerHTML = `
            <div style="border-bottom: 1px solid #eee; padding: 8px;">
                <strong>${{request.request.method}}</strong> ${{request.request.url}}
                <span style="color: ${{request.response.status < 400 ? 'green' : 'red'}};">
                    ${{request.response.status}}
                </span>
            </div>
        `;
        
        networkEl.appendChild(logEntry);
    }}
}}

// Initialize DevTools panel
new ProjectIndexDevToolsPanel();
"""
        
        self._write_file('browser-extension/devtools.js', devtools_panel)
        self._write_file('browser-extension/manifest.json', json.dumps(chrome_manifest, indent=2))
    
    def _write_file(self, file_path: str, content: str) -> None:
        """Write content to file, creating directories as needed."""
        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            full_path.write_text(content)
            print(f"✅ Generated {file_path}")
        except Exception as e:
            print(f"❌ Failed to generate {file_path}: {e}")


def generate_all_dev_tools(project_root: Optional[Path] = None, api_url: str = "http://localhost:8000/project-index") -> None:
    """Generate all development tools and IDE extensions."""
    generator = IDEExtensionGenerator(project_root)
    generator.api_base_url = api_url
    
    print("🛠️  Generating development tools and IDE extensions...")
    
    generator.generate_vscode_extension()
    generator.generate_intellij_plugin()
    generator.generate_dev_server_integration()
    generator.generate_browser_devtools()
    
    print("✅ All development tools generated successfully!")


# Export main components
__all__ = ['IDEExtensionGenerator', 'generate_all_dev_tools']