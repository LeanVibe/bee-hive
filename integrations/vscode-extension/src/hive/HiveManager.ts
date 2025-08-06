import * as vscode from 'vscode';
import * as WebSocket from 'ws';
import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { EventEmitter } from 'events';
import { ConfigurationManager, HiveConfig } from '../utils/ConfigurationManager';
import { Logger } from '../utils/Logger';

export interface HiveStatus {
    connected: boolean;
    agents: Agent[];
    tasks: Task[];
    logs: LogEntry[];
    version?: string;
    uptime?: number;
}

export interface Agent {
    id: string;
    name: string;
    role: string;
    status: 'active' | 'idle' | 'busy' | 'error' | 'stopped';
    currentTask?: string;
    performance?: {
        tasksCompleted: number;
        averageTime: number;
        successRate: number;
    };
    capabilities?: string[];
    created: Date;
    lastActive: Date;
}

export interface Task {
    id: string;
    title: string;
    description: string;
    status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
    priority: 'low' | 'medium' | 'high' | 'critical';
    assignedAgentId?: string;
    progress?: number;
    created: Date;
    updated: Date;
    estimated?: number;
    actual?: number;
}

export interface LogEntry {
    id: string;
    timestamp: Date;
    level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
    source: string;
    message: string;
    metadata?: any;
}

export interface DeploymentRequest {
    projectPath: string;
    environment: 'development' | 'staging' | 'production';
    options?: {
        skipTests?: boolean;
        buildOnly?: boolean;
        rollback?: boolean;
    };
}

export interface CodeOptimizationRequest {
    filePath: string;
    content: string;
    language: string;
    context?: string;
    optimizationType: 'performance' | 'readability' | 'security' | 'all';
}

export interface CodeOptimizationResult {
    originalCode: string;
    optimizedCode: string;
    changes: {
        type: string;
        description: string;
        impact: 'low' | 'medium' | 'high';
    }[];
    metrics: {
        linesReduced?: number;
        complexityImprovement?: number;
        performanceGain?: number;
    };
}

export class HiveManager extends EventEmitter implements vscode.Disposable {
    private api: AxiosInstance;
    private ws: WebSocket | null = null;
    private config: HiveConfig;
    private status: HiveStatus = {
        connected: false,
        agents: [],
        tasks: [],
        logs: []
    };
    private reconnectTimer?: NodeJS.Timeout;
    private heartbeatTimer?: NodeJS.Timeout;

    constructor(private configManager: ConfigurationManager) {
        super();
        this.config = configManager.getConfig();
        this.api = this.createApiClient();
        
        // Listen for config changes
        configManager.on('configChanged', (newConfig: HiveConfig) => {
            this.config = newConfig;
            this.api = this.createApiClient();
            if (this.status.connected) {
                this.reconnect();
            }
        });
    }

    private createApiClient(): AxiosInstance {
        return axios.create({
            baseURL: this.config.apiUrl,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json',
                ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` })
            }
        });
    }

    async start(): Promise<boolean> {
        try {
            Logger.info('Starting Agent Hive connection...');
            
            // First check if the API is available
            const healthCheck = await this.checkHealth();
            if (!healthCheck) {
                vscode.window.showErrorMessage('Cannot connect to Agent Hive API. Please check the URL in settings.');
                return false;
            }

            // Connect WebSocket for real-time updates
            await this.connectWebSocket();
            
            // Initial data fetch
            await this.refreshStatus();
            
            // Start heartbeat
            this.startHeartbeat();
            
            Logger.info('Agent Hive connected successfully');
            vscode.window.showInformationMessage('Agent Hive connected successfully!');
            return true;
        } catch (error) {
            Logger.error('Failed to start Agent Hive', error);
            vscode.window.showErrorMessage(`Failed to connect to Agent Hive: ${error}`);
            return false;
        }
    }

    async stop(): Promise<void> {
        Logger.info('Stopping Agent Hive connection...');
        
        this.cleanup();
        this.status = {
            connected: false,
            agents: [],
            tasks: [],
            logs: []
        };
        this.emit('statusChanged', this.status);
        
        Logger.info('Agent Hive disconnected');
        vscode.window.showInformationMessage('Agent Hive disconnected');
    }

    private cleanup(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = undefined;
        }
        
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = undefined;
        }
    }

    private async checkHealth(): Promise<boolean> {
        try {
            const response = await this.api.get('/health');
            return response.status === 200;
        } catch {
            return false;
        }
    }

    private async connectWebSocket(): Promise<void> {
        return new Promise((resolve, reject) => {
            const wsUrl = this.config.apiUrl.replace(/^http/, 'ws') + '/ws/realtime';
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.on('open', () => {
                Logger.info('WebSocket connected');
                this.status.connected = true;
                resolve();
            });
            
            this.ws.on('message', (data: WebSocket.Data) => {
                try {
                    const message = JSON.parse(data.toString());
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    Logger.error('Failed to parse WebSocket message', error);
                }
            });
            
            this.ws.on('close', (code, reason) => {
                Logger.warn(`WebSocket closed: ${code} ${reason}`);
                this.status.connected = false;
                this.emit('statusChanged', this.status);
                this.scheduleReconnect();
            });
            
            this.ws.on('error', (error) => {
                Logger.error('WebSocket error', error);
                reject(error);
            });
            
            // Connection timeout
            setTimeout(() => {
                if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 5000);
        });
    }

    private handleWebSocketMessage(message: any): void {
        switch (message.type) {
            case 'agent_status_update':
                this.updateAgent(message.data);
                break;
            case 'task_update':
                this.updateTask(message.data);
                break;
            case 'log_entry':
                this.addLogEntry(message.data);
                break;
            case 'system_status':
                this.updateSystemStatus(message.data);
                break;
            default:
                Logger.debug('Unknown WebSocket message type', message.type);
        }
    }

    private updateAgent(agentData: any): void {
        const agentIndex = this.status.agents.findIndex(a => a.id === agentData.id);
        if (agentIndex >= 0) {
            this.status.agents[agentIndex] = { ...this.status.agents[agentIndex], ...agentData };
        } else {
            this.status.agents.push(agentData);
        }
        this.emit('statusChanged', this.status);
    }

    private updateTask(taskData: any): void {
        const taskIndex = this.status.tasks.findIndex(t => t.id === taskData.id);
        if (taskIndex >= 0) {
            this.status.tasks[taskIndex] = { ...this.status.tasks[taskIndex], ...taskData };
        } else {
            this.status.tasks.push(taskData);
        }
        this.emit('statusChanged', this.status);
    }

    private addLogEntry(logData: any): void {
        this.status.logs.unshift(logData);
        
        // Keep only last 1000 log entries
        if (this.status.logs.length > 1000) {
            this.status.logs = this.status.logs.slice(0, 1000);
        }
        
        this.emit('statusChanged', this.status);
    }

    private updateSystemStatus(systemData: any): void {
        this.status = { ...this.status, ...systemData };
        this.emit('statusChanged', this.status);
    }

    private scheduleReconnect(): void {
        if (this.reconnectTimer) return;
        
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = undefined;
            this.reconnect();
        }, 5000);
    }

    private async reconnect(): Promise<void> {
        try {
            await this.connectWebSocket();
            await this.refreshStatus();
            this.startHeartbeat();
            Logger.info('Reconnected to Agent Hive');
        } catch (error) {
            Logger.error('Reconnection failed', error);
            this.scheduleReconnect();
        }
    }

    private startHeartbeat(): void {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
        }
        
        this.heartbeatTimer = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.ping();
            }
        }, 30000);
    }

    private async refreshStatus(): Promise<void> {
        try {
            const [agentsResponse, tasksResponse] = await Promise.all([
                this.api.get('/api/v1/agents'),
                this.api.get('/api/v1/tasks')
            ]);

            this.status.agents = agentsResponse.data.agents || [];
            this.status.tasks = tasksResponse.data.tasks || [];
            
            this.emit('statusChanged', this.status);
        } catch (error) {
            Logger.error('Failed to refresh status', error);
        }
    }

    // Public API methods
    getStatus(): HiveStatus {
        return this.status;
    }

    async createAgent(role: string, name: string): Promise<Agent | null> {
        try {
            const response = await this.api.post('/api/v1/agents', { role, name });
            const agent = response.data.agent;
            this.updateAgent(agent);
            return agent;
        } catch (error) {
            Logger.error('Failed to create agent', error);
            vscode.window.showErrorMessage(`Failed to create agent: ${error}`);
            return null;
        }
    }

    async startAgent(agentId: string): Promise<boolean> {
        try {
            await this.api.post(`/api/v1/agents/${agentId}/start`);
            return true;
        } catch (error) {
            Logger.error(`Failed to start agent ${agentId}`, error);
            vscode.window.showErrorMessage(`Failed to start agent: ${error}`);
            return false;
        }
    }

    async stopAgent(agentId: string): Promise<boolean> {
        try {
            await this.api.post(`/api/v1/agents/${agentId}/stop`);
            return true;
        } catch (error) {
            Logger.error(`Failed to stop agent ${agentId}`, error);
            vscode.window.showErrorMessage(`Failed to stop agent: ${error}`);
            return false;
        }
    }

    async deployProject(request: DeploymentRequest): Promise<boolean> {
        try {
            const response = await this.api.post('/api/v1/deploy', request);
            const taskId = response.data.taskId;
            
            vscode.window.showInformationMessage(`Deployment started (Task: ${taskId})`);
            return true;
        } catch (error) {
            Logger.error('Deployment failed', error);
            vscode.window.showErrorMessage(`Deployment failed: ${error}`);
            return false;
        }
    }

    async optimizeCode(request: CodeOptimizationRequest): Promise<CodeOptimizationResult | null> {
        try {
            const response = await this.api.post('/api/v1/code/optimize', request);
            return response.data;
        } catch (error) {
            Logger.error('Code optimization failed', error);
            vscode.window.showErrorMessage(`Code optimization failed: ${error}`);
            return null;
        }
    }

    async generateDocumentation(projectPath: string): Promise<boolean> {
        try {
            const response = await this.api.post('/api/v1/code/documentation', { projectPath });
            const taskId = response.data.taskId;
            
            vscode.window.showInformationMessage(`Documentation generation started (Task: ${taskId})`);
            return true;
        } catch (error) {
            Logger.error('Documentation generation failed', error);
            vscode.window.showErrorMessage(`Documentation generation failed: ${error}`);
            return false;
        }
    }

    async runTests(): Promise<boolean> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                vscode.window.showErrorMessage('No workspace folder found');
                return false;
            }

            const response = await this.api.post('/api/v1/test', {
                projectPath: workspaceFolder.uri.fsPath
            });
            const taskId = response.data.taskId;
            
            vscode.window.showInformationMessage(`AI-enhanced testing started (Task: ${taskId})`);
            return true;
        } catch (error) {
            Logger.error('Test execution failed', error);
            vscode.window.showErrorMessage(`Test execution failed: ${error}`);
            return false;
        }
    }

    async cancelTask(taskId: string): Promise<boolean> {
        try {
            await this.api.post(`/api/v1/tasks/${taskId}/cancel`);
            return true;
        } catch (error) {
            Logger.error(`Failed to cancel task ${taskId}`, error);
            vscode.window.showErrorMessage(`Failed to cancel task: ${error}`);
            return false;
        }
    }

    async retryTask(taskId: string): Promise<boolean> {
        try {
            await this.api.post(`/api/v1/tasks/${taskId}/retry`);
            return true;
        } catch (error) {
            Logger.error(`Failed to retry task ${taskId}`, error);
            vscode.window.showErrorMessage(`Failed to retry task: ${error}`);
            return false;
        }
    }

    dispose(): void {
        this.cleanup();
        this.removeAllListeners();
    }
}