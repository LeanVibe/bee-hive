/**
 * API v2 Demo View - Test and demonstrate API v2 integration
 * 
 * This view provides a comprehensive test interface for the new API v2 
 * integration, allowing developers and stakeholders to validate the
 * PWA-backend connection with real-time updates.
 */

import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { apiV2Service, Agent, Task, AgentCreateRequest } from '../services/api-v2';
import { webSocketV2Client } from '../services/websocket-v2';

@customElement('api-v2-demo')
export class ApiV2Demo extends LitElement {
  @state() agents: Agent[] = [];
  @state() tasks: Task[] = [];
  @state() systemStatus: any = null;
  @state() connectionStatus: string = 'disconnected';
  @state() logs: string[] = [];
  @state() isLoading: boolean = false;

  private cleanupFunctions: (() => void)[] = [];

  static styles = css`
    :host {
      display: block;
      padding: 20px;
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 1200px;
      margin: 0 auto;
    }

    .header {
      text-align: center;
      margin-bottom: 30px;
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-radius: 12px;
    }

    .demo-section {
      margin-bottom: 30px;
      padding: 20px;
      border: 1px solid #e5e5e5;
      border-radius: 8px;
      background: white;
    }

    .demo-section h3 {
      margin-top: 0;
      color: #333;
      border-bottom: 2px solid #f0f0f0;
      padding-bottom: 10px;
    }

    .status-indicator {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: bold;
      margin-left: 10px;
    }

    .status-connected { background: #d4edda; color: #155724; }
    .status-disconnected { background: #f8d7da; color: #721c24; }
    .status-connecting { background: #fff3cd; color: #856404; }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 20px;
    }

    button {
      padding: 10px 20px;
      border: none;
      border-radius: 6px;
      background: #007bff;
      color: white;
      cursor: pointer;
      font-size: 14px;
      transition: background 0.2s;
    }

    button:hover:not(:disabled) {
      background: #0056b3;
    }

    button:disabled {
      background: #6c757d;
      cursor: not-allowed;
    }

    button.secondary {
      background: #6c757d;
    }

    button.danger {
      background: #dc3545;
    }

    button.success {
      background: #28a745;
    }

    .data-grid {
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }

    .data-card {
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 15px;
      background: #f8f9fa;
    }

    .data-card h4 {
      margin-top: 0;
      margin-bottom: 10px;
      color: #495057;
    }

    .agent-card {
      border-left: 4px solid #007bff;
    }

    .task-card {
      border-left: 4px solid #28a745;
    }

    .logs {
      background: #f8f9fa;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 15px;
      height: 200px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 12px;
    }

    .log-entry {
      margin-bottom: 5px;
      padding: 2px 0;
    }

    .log-info { color: #007bff; }
    .log-success { color: #28a745; }
    .log-warning { color: #ffc107; }
    .log-error { color: #dc3545; }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 15px;
      margin-bottom: 20px;
    }

    .metric-card {
      text-align: center;
      padding: 15px;
      border-radius: 8px;
      background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    }

    .metric-value {
      font-size: 24px;
      font-weight: bold;
      color: #495057;
    }

    .metric-label {
      font-size: 12px;
      color: #6c757d;
      text-transform: uppercase;
      margin-top: 5px;
    }

    .form-group {
      margin-bottom: 15px;
    }

    label {
      display: block;
      margin-bottom: 5px;
      font-weight: bold;
      color: #495057;
    }

    select, input {
      width: 100%;
      padding: 8px 12px;
      border: 1px solid #ced4da;
      border-radius: 4px;
      font-size: 14px;
    }

    .loading {
      opacity: 0.6;
      pointer-events: none;
    }

    @media (max-width: 768px) {
      :host {
        padding: 10px;
      }
      
      .data-grid {
        grid-template-columns: 1fr;
      }
      
      .metrics {
        grid-template-columns: repeat(2, 1fr);
      }
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    this.log('🚀 API v2 Demo initialized', 'info');
    this.initializeWebSocket();
    this.loadInitialData();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.cleanupFunctions.forEach(cleanup => cleanup());
    webSocketV2Client.disconnect();
  }

  private async initializeWebSocket() {
    try {
      this.connectionStatus = 'connecting';
      this.log('🔌 Connecting to WebSocket v2...', 'info');
      
      // Set up event listeners
      webSocketV2Client.on('connected', () => {
        this.connectionStatus = 'connected';
        this.log('✅ WebSocket v2 connected successfully', 'success');
        this.requestUpdate();
      });

      webSocketV2Client.on('disconnected', (data) => {
        this.connectionStatus = 'disconnected';
        this.log(`🔌 WebSocket v2 disconnected: ${data.reason}`, 'warning');
        this.requestUpdate();
      });

      webSocketV2Client.on('agentUpdate', (data) => {
        this.log(`🤖 Agent update: ${data.agentId}`, 'info');
        this.loadAgents(); // Refresh agents list
      });

      webSocketV2Client.on('taskUpdate', (data) => {
        this.log(`📋 Task update: ${data.taskId}`, 'info');
        this.loadTasks(); // Refresh tasks list
      });

      webSocketV2Client.on('systemStatusUpdate', (data) => {
        this.systemStatus = data;
        this.log('📊 System status updated', 'info');
        this.requestUpdate();
      });

      // Connect to WebSocket
      await webSocketV2Client.connect();
      
    } catch (error) {
      this.connectionStatus = 'disconnected';
      this.log(`❌ WebSocket connection failed: ${error}`, 'error');
    }
  }

  private async loadInitialData() {
    this.isLoading = true;
    try {
      await Promise.all([
        this.loadAgents(),
        this.loadTasks(),
        this.loadSystemStatus()
      ]);
      this.log('✅ Initial data loaded successfully', 'success');
    } catch (error) {
      this.log(`❌ Failed to load initial data: ${error}`, 'error');
    } finally {
      this.isLoading = false;
    }
  }

  private async loadAgents() {
    try {
      const response = await apiV2Service.listAgents();
      this.agents = response.agents;
      this.log(`📡 Loaded ${response.agents.length} agents`, 'info');
      this.requestUpdate();
    } catch (error) {
      this.log(`❌ Failed to load agents: ${error}`, 'error');
    }
  }

  private async loadTasks() {
    try {
      const response = await apiV2Service.listTasks();
      this.tasks = response.tasks;
      this.log(`📋 Loaded ${response.tasks.length} tasks`, 'info');
      this.requestUpdate();
    } catch (error) {
      this.log(`❌ Failed to load tasks: ${error}`, 'error');
    }
  }

  private async loadSystemStatus() {
    try {
      const status = await apiV2Service.getSystemStatus();
      this.systemStatus = status;
      this.log('📊 System status loaded', 'info');
      this.requestUpdate();
    } catch (error) {
      this.log(`⚠️ Could not load system status: ${error}`, 'warning');
    }
  }

  private async createTestAgent() {
    const roles = ['backend_developer', 'frontend_developer', 'qa_engineer', 'devops_engineer'];
    const randomRole = roles[Math.floor(Math.random() * roles.length)];
    
    const request: AgentCreateRequest = {
      role: randomRole,
      agent_type: 'claude_code'
    };

    try {
      this.isLoading = true;
      this.log(`🚀 Creating ${randomRole} agent...`, 'info');
      
      const agent = await apiV2Service.createAgent(request);
      this.log(`✅ Agent created: ${agent.id} (${agent.role})`, 'success');
      
      await this.loadAgents(); // Refresh agents list
    } catch (error) {
      this.log(`❌ Failed to create agent: ${error}`, 'error');
    } finally {
      this.isLoading = false;
    }
  }

  private async createTestTask() {
    const tasks = [
      { title: 'Fix authentication bug', priority: 'high' as const },
      { title: 'Update documentation', priority: 'medium' as const },
      { title: 'Optimize database queries', priority: 'high' as const },
      { title: 'Add unit tests', priority: 'medium' as const },
      { title: 'Deploy to staging', priority: 'low' as const }
    ];
    
    const randomTask = tasks[Math.floor(Math.random() * tasks.length)];

    try {
      this.isLoading = true;
      this.log(`📝 Creating task: ${randomTask.title}...`, 'info');
      
      const task = await apiV2Service.createTask({
        ...randomTask,
        description: `Auto-generated test task: ${randomTask.title}`
      });
      
      this.log(`✅ Task created: ${task.id} (${task.title})`, 'success');
      await this.loadTasks(); // Refresh tasks list
    } catch (error) {
      this.log(`❌ Failed to create task: ${error}`, 'error');
    } finally {
      this.isLoading = false;
    }
  }

  private async deleteAgent(agentId: string) {
    if (!confirm('Are you sure you want to delete this agent?')) return;

    try {
      this.isLoading = true;
      this.log(`🗑️ Deleting agent: ${agentId}...`, 'info');
      
      await apiV2Service.deleteAgent(agentId);
      this.log(`✅ Agent deleted: ${agentId}`, 'success');
      
      await this.loadAgents(); // Refresh agents list
    } catch (error) {
      this.log(`❌ Failed to delete agent: ${error}`, 'error');
    } finally {
      this.isLoading = false;
    }
  }

  private log(message: string, level: 'info' | 'success' | 'warning' | 'error' = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    this.logs = [...this.logs.slice(-49), `[${timestamp}] ${message}`]; // Keep last 50 logs
    this.requestUpdate();
  }

  render() {
    return html`
      <div class="${this.isLoading ? 'loading' : ''}">
        <div class="header">
          <h1>🚀 API v2 Integration Demo</h1>
          <p>Real-time PWA ↔ Backend Connection Test</p>
          <div>
            WebSocket Status: 
            <span class="status-indicator status-${this.connectionStatus}">
              ${this.connectionStatus.toUpperCase()}
            </span>
          </div>
        </div>

        <div class="demo-section">
          <h3>📊 System Metrics</h3>
          <div class="metrics">
            <div class="metric-card">
              <div class="metric-value">${this.agents.length}</div>
              <div class="metric-label">Total Agents</div>
            </div>
            <div class="metric-card">
              <div class="metric-value">${this.agents.filter(a => a.status === 'active').length}</div>
              <div class="metric-label">Active Agents</div>
            </div>
            <div class="metric-card">
              <div class="metric-value">${this.tasks.length}</div>
              <div class="metric-label">Total Tasks</div>
            </div>
            <div class="metric-card">
              <div class="metric-value">${this.tasks.filter(t => t.status === 'in_progress').length}</div>
              <div class="metric-label">Active Tasks</div>
            </div>
          </div>
        </div>

        <div class="demo-section">
          <h3>🎮 Controls</h3>
          <div class="controls">
            <button @click="${this.createTestAgent}" ?disabled="${this.isLoading}" class="success">
              🤖 Create Test Agent
            </button>
            <button @click="${this.createTestTask}" ?disabled="${this.isLoading}" class="success">
              📝 Create Test Task
            </button>
            <button @click="${this.loadInitialData}" ?disabled="${this.isLoading}">
              🔄 Refresh Data
            </button>
            <button @click="${() => webSocketV2Client.requestSystemStatus()}" ?disabled="${this.connectionStatus !== 'connected'}">
              📊 Request Status
            </button>
            <button @click="${() => this.logs = []}" class="secondary">
              🗑️ Clear Logs
            </button>
          </div>
        </div>

        <div class="data-grid">
          <div class="demo-section">
            <h3>🤖 Agents (${this.agents.length})</h3>
            <div style="max-height: 400px; overflow-y: auto;">
              ${this.agents.map(agent => html`
                <div class="data-card agent-card">
                  <h4>${agent.role.replace('_', ' ')}</h4>
                  <p><strong>ID:</strong> ${agent.id}</p>
                  <p><strong>Status:</strong> 
                    <span class="status-indicator status-${agent.status === 'active' ? 'connected' : 'disconnected'}">
                      ${agent.status.toUpperCase()}
                    </span>
                  </p>
                  <p><strong>Created:</strong> ${new Date(agent.created_at).toLocaleString()}</p>
                  <p><strong>Last Activity:</strong> ${new Date(agent.last_activity).toLocaleString()}</p>
                  ${agent.current_task_id ? html`<p><strong>Current Task:</strong> ${agent.current_task_id}</p>` : ''}
                  <button @click="${() => this.deleteAgent(agent.id)}" class="danger" style="margin-top: 10px;">
                    Delete
                  </button>
                </div>
              `)}
              ${this.agents.length === 0 ? html`<p>No agents found. Create one to get started!</p>` : ''}
            </div>
          </div>

          <div class="demo-section">
            <h3>📋 Tasks (${this.tasks.length})</h3>
            <div style="max-height: 400px; overflow-y: auto;">
              ${this.tasks.map(task => html`
                <div class="data-card task-card">
                  <h4>${task.title}</h4>
                  <p><strong>ID:</strong> ${task.id}</p>
                  <p><strong>Status:</strong> 
                    <span class="status-indicator status-${task.status === 'completed' ? 'connected' : task.status === 'error' ? 'disconnected' : 'connecting'}">
                      ${task.status.toUpperCase()}
                    </span>
                  </p>
                  <p><strong>Priority:</strong> ${task.priority.toUpperCase()}</p>
                  ${task.description ? html`<p><strong>Description:</strong> ${task.description}</p>` : ''}
                  ${task.assigned_to ? html`<p><strong>Assigned to:</strong> ${task.assigned_to}</p>` : ''}
                  <p><strong>Created:</strong> ${new Date(task.created_at).toLocaleString()}</p>
                </div>
              `)}
              ${this.tasks.length === 0 ? html`<p>No tasks found. Create one to get started!</p>` : ''}
            </div>
          </div>
        </div>

        <div class="demo-section">
          <h3>📝 Activity Logs</h3>
          <div class="logs">
            ${this.logs.map(log => html`
              <div class="log-entry log-${log.includes('✅') ? 'success' : log.includes('❌') ? 'error' : log.includes('⚠️') ? 'warning' : 'info'}">
                ${log}
              </div>
            `)}
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'api-v2-demo': ApiV2Demo;
  }
}