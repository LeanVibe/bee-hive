import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../../services/websocket'

export interface AgentStatus {
  id: string
  name: string
  type: 'architect' | 'developer' | 'qa' | 'reviewer' | 'meta'
  status: 'active' | 'idle' | 'working' | 'error' | 'offline'
  activity: string
  progress: number
  lastActivity: string
  performance: {
    cpu: number
    memory: number
    tasksCompleted: number
    successRate: number
  }
  health: {
    responseTime: number
    errorCount: number
    uptime: number
    lastHealthCheck: string
  }
  currentTask?: {
    id: string
    title: string
    progress: number
    estimatedTime: string
  }
}

export interface SystemMetrics {
  totalAgents: number
  activeAgents: number
  totalTasks: number
  completedTasks: number
  errorRate: number
  averageResponseTime: number
  systemLoad: number
  lastUpdated: string
}

@customElement('live-agent-monitor')
export class LiveAgentMonitor extends LitElement {
  @property({ type: Boolean }) declare compact: boolean
  @property({ type: Number }) declare maxAgents: number

  @state() private declare agents: AgentStatus[]
  @state() private declare systemMetrics: SystemMetrics
  @state() private declare connectionStatus: boolean
  @state() private declare lastUpdate: Date | null
  @state() private declare selectedAgent: AgentStatus | null
  @state() private declare connectionQuality: 'excellent' | 'good' | 'poor' | 'offline'
  @state() private declare autoRefresh: boolean

  private websocketService: WebSocketService
  private updateSubscriptions: (() => void)[] = []
  private performanceHistory: Map<string, number[]> = new Map()

  static styles = css`
    :host {
      display: block;
      background: white;
      border-radius: 12px;
      padding: 1rem;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .monitor-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .header-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: #111827;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .connection-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }

    .connection-indicator.poor {
      background: #f59e0b;
    }

    .connection-indicator.offline {
      background: #ef4444;
      animation: blink 1s infinite;
    }

    .system-overview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1.5rem;
      padding: 0.75rem;
      background: #f8fafc;
      border-radius: 8px;
    }

    .metric-item {
      text-align: center;
    }

    .metric-value {
      font-size: 1.25rem;
      font-weight: 600;
      color: #111827;
      margin-bottom: 0.25rem;
    }

    .metric-label {
      font-size: 0.75rem;
      color: #6b7280;
      text-transform: uppercase;
      font-weight: 500;
    }

    .agents-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.75rem;
    }

    .agents-grid.compact {
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }

    .agent-card {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 0.75rem;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }

    .agent-card:hover {
      border-color: #d1d5db;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .agent-card.selected {
      border-color: #3b82f6;
      box-shadow: 0 0 0 1px #3b82f6;
    }

    .agent-card.active::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 3px;
      height: 100%;
      background: linear-gradient(180deg, #10b981 0%, #059669 100%);
      animation: activityPulse 2s infinite;
    }

    .agent-card.working::before {
      background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%);
      animation: workingPulse 1.5s infinite;
    }

    .agent-card.error::before {
      background: linear-gradient(180deg, #ef4444 0%, #dc2626 100%);
      animation: errorBlink 1s infinite;
    }

    .agent-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .agent-info {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .agent-avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.875rem;
      color: white;
      font-weight: 600;
    }

    .agent-avatar.architect {
      background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    }

    .agent-avatar.developer {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }

    .agent-avatar.qa {
      background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }

    .agent-avatar.reviewer {
      background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    .agent-avatar.meta {
      background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    }

    .agent-name {
      font-weight: 600;
      color: #111827;
      font-size: 0.875rem;
    }

    .agent-type {
      font-size: 0.75rem;
      color: #6b7280;
      text-transform: capitalize;
    }

    .status-badge {
      padding: 0.125rem 0.375rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
      text-transform: uppercase;
    }

    .status-badge.active {
      background: #d1fae5;
      color: #065f46;
    }

    .status-badge.working {
      background: #dbeafe;
      color: #1e40af;
    }

    .status-badge.idle {
      background: #f3f4f6;
      color: #374151;
    }

    .status-badge.error {
      background: #fee2e2;
      color: #991b1b;
    }

    .status-badge.offline {
      background: #f3f4f6;
      color: #6b7280;
    }

    .agent-activity {
      font-size: 0.75rem;
      color: #4b5563;
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }

    .activity-icon {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #10b981;
      animation: activityPing 2s infinite;
    }

    .agent-metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }

    .metric-small {
      font-size: 0.75rem;
      color: #6b7280;
    }

    .metric-small .value {
      font-weight: 600;
      color: #111827;
    }

    .progress-bar {
      width: 100%;
      height: 4px;
      background: #e5e7eb;
      border-radius: 2px;
      overflow: hidden;
      margin-top: 0.5rem;
    }

    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981 0%, #059669 100%);
      transition: width 0.3s ease;
      position: relative;
    }

    .progress-fill::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
      animation: progressShimmer 2s infinite;
    }

    .current-task {
      background: #f8fafc;
      border-radius: 4px;
      padding: 0.5rem;
      margin-top: 0.5rem;
      border-left: 2px solid #3b82f6;
    }

    .task-title {
      font-size: 0.75rem;
      font-weight: 600;
      color: #111827;
      margin-bottom: 0.25rem;
    }

    .task-progress {
      font-size: 0.75rem;
      color: #6b7280;
    }

    .performance-chart {
      height: 20px;
      margin-top: 0.5rem;
      position: relative;
      background: #f3f4f6;
      border-radius: 2px;
      overflow: hidden;
    }

    .performance-line {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 100%;
      background: linear-gradient(180deg, transparent 0%, #3b82f6 100%);
      opacity: 0.7;
    }

    .empty-state {
      text-align: center;
      padding: 2rem 1rem;
      color: #6b7280;
    }

    .empty-icon {
      font-size: 2.5rem;
      margin-bottom: 1rem;
      opacity: 0.5;
    }

    .controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .control-button {
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 4px;
      transition: all 0.2s;
    }

    .control-button:hover {
      background: #f3f4f6;
      color: #374151;
    }

    .control-button.active {
      color: #3b82f6;
      background: #eff6ff;
    }

    /* Animations */
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }

    @keyframes blink {
      0%, 50% { opacity: 1; }
      25%, 75% { opacity: 0.3; }
    }

    @keyframes activityPulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }

    @keyframes workingPulse {
      0%, 100% { opacity: 1; transform: scaleY(1); }
      50% { opacity: 0.8; transform: scaleY(0.9); }
    }

    @keyframes errorBlink {
      0%, 50% { opacity: 1; }
      25%, 75% { opacity: 0.4; }
    }

    @keyframes activityPing {
      75%, 100% {
        transform: scale(1.2);
        opacity: 0;
      }
    }

    @keyframes progressShimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }

    /* Compact mode adjustments */
    .compact .agent-card {
      padding: 0.5rem;
    }

    .compact .agent-avatar {
      width: 24px;
      height: 24px;
      font-size: 0.75rem;
    }

    .compact .agent-name {
      font-size: 0.75rem;
    }

    .compact .agent-metrics {
      display: none;
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
      :host {
        background: #1f2937;
        color: #f9fafb;
      }

      .agent-card {
        background: #374151;
        border-color: #4b5563;
      }

      .system-overview {
        background: #374151;
      }

      .current-task {
        background: #374151;
      }
    }
  `

  constructor() {
    super()
    this.compact = false
    this.maxAgents = 10
    this.agents = []
    this.systemMetrics = {
      totalAgents: 0,
      activeAgents: 0,
      totalTasks: 0,
      completedTasks: 0,
      errorRate: 0,
      averageResponseTime: 0,
      systemLoad: 0,
      lastUpdated: new Date().toISOString()
    }
    this.connectionStatus = false
    this.lastUpdate = null
    this.selectedAgent = null
    this.connectionQuality = 'offline'
    this.autoRefresh = true

    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.initializeRealTimeMonitoring()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanup()
  }

  private async initializeRealTimeMonitoring() {
    try {
      // Subscribe to real-time WebSocket events
      this.setupWebSocketSubscriptions()
      
      // Enable mobile dashboard mode for optimized streaming
      this.websocketService.enableMobileDashboardMode()
      
      // Request initial data
      await this.loadInitialData()
      
      console.log('🔴 Live agent monitor initialized')
    } catch (error) {
      console.error('Failed to initialize live agent monitor:', error)
    }
  }

  private setupWebSocketSubscriptions() {
    // Agent status updates
    const agentSub = this.websocketService.subscribeToAgentMetrics((data) => {
      this.handleAgentMetricsUpdate(data)
    })
    this.updateSubscriptions.push(agentSub)

    // System metrics updates  
    const systemSub = this.websocketService.subscribeToSystemMetrics((data) => {
      this.handleSystemMetricsUpdate(data)
    })
    this.updateSubscriptions.push(systemSub)

    // Connection quality monitoring
    const connectionSub = this.websocketService.subscribeToConnectionQuality((data) => {
      this.connectionQuality = data.quality as any
      this.requestUpdate()
    })
    this.updateSubscriptions.push(connectionSub)

    // Critical events
    const criticalSub = this.websocketService.subscribeToCriticalEvents((data) => {
      this.handleCriticalEvent(data)
    })
    this.updateSubscriptions.push(criticalSub)

    // Connection status
    this.websocketService.on('connected', () => {
      this.connectionStatus = true
      this.loadInitialData()
    })

    this.websocketService.on('disconnected', () => {
      this.connectionStatus = false
    })
  }

  private async loadInitialData() {
    try {
      // Request current agent status
      this.websocketService.requestAgentStatus()
      
      // Request system metrics
      this.websocketService.requestSystemMetrics()
      
      this.lastUpdate = new Date()
    } catch (error) {
      console.error('Failed to load initial data:', error)
      this.generateMockData()
    }
  }

  private handleAgentMetricsUpdate(data: any) {
    if (data.agents) {
      this.agents = data.agents.map((agent: any) => this.transformAgentData(agent))
    }
    this.lastUpdate = new Date()
    this.requestUpdate()
  }

  private handleSystemMetricsUpdate(data: any) {
    if (data.system_overview) {
      this.systemMetrics = {
        ...this.systemMetrics,
        ...data.system_overview,
        lastUpdated: new Date().toISOString()
      }
    }
    this.requestUpdate()
  }

  private handleCriticalEvent(data: any) {
    // Handle critical events that affect agents
    if (data.agent_id) {
      const agent = this.agents.find(a => a.id === data.agent_id)
      if (agent) {
        // Update agent status based on critical event
        agent.status = data.status || 'error'
        agent.activity = data.message || 'Critical event occurred'
        this.requestUpdate()
      }
    }
  }

  private transformAgentData(rawAgent: any): AgentStatus {
    return {
      id: rawAgent.id || rawAgent.agent_id,
      name: rawAgent.name || `Agent ${rawAgent.id}`,
      type: rawAgent.type || rawAgent.agent_type || 'developer',
      status: rawAgent.status || 'idle',
      activity: rawAgent.current_activity || rawAgent.activity || 'Waiting for tasks',
      progress: rawAgent.progress || 0,
      lastActivity: rawAgent.last_activity || new Date().toISOString(),
      performance: {
        cpu: rawAgent.cpu_usage || 0,
        memory: rawAgent.memory_usage || 0,
        tasksCompleted: rawAgent.tasks_completed || 0,
        successRate: rawAgent.success_rate || 0
      },
      health: {
        responseTime: rawAgent.response_time || 0,
        errorCount: rawAgent.error_count || 0,
        uptime: rawAgent.uptime || 0,
        lastHealthCheck: rawAgent.last_health_check || new Date().toISOString()
      },
      currentTask: rawAgent.current_task ? {
        id: rawAgent.current_task.id,
        title: rawAgent.current_task.title || rawAgent.current_task.description,
        progress: rawAgent.current_task.progress || 0,
        estimatedTime: rawAgent.current_task.estimated_time || 'Unknown'
      } : undefined
    }
  }

  private generateMockData() {
    this.agents = [
      {
        id: 'architect-1',
        name: 'System Architect',
        type: 'architect',
        status: 'active',
        activity: 'Designing system architecture',
        progress: 75,
        lastActivity: new Date().toISOString(),
        performance: {
          cpu: 45,
          memory: 62,
          tasksCompleted: 8,
          successRate: 95
        },
        health: {
          responseTime: 150,
          errorCount: 0,
          uptime: 98.5,
          lastHealthCheck: new Date().toISOString()
        },
        currentTask: {
          id: 'task-1',
          title: 'API Architecture Review',
          progress: 75,
          estimatedTime: '15 min'
        }
      },
      {
        id: 'developer-1',
        name: 'Backend Developer',
        type: 'developer',
        status: 'working',
        activity: 'Implementing user authentication',
        progress: 40,
        lastActivity: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
        performance: {
          cpu: 78,
          memory: 84,
          tasksCompleted: 12,
          successRate: 88
        },
        health: {
          responseTime: 200,
          errorCount: 1,
          uptime: 97.2,
          lastHealthCheck: new Date().toISOString()
        },
        currentTask: {
          id: 'task-2',
          title: 'User Authentication Service',
          progress: 40,
          estimatedTime: '25 min'
        }
      },
      {
        id: 'qa-1',
        name: 'QA Engineer',
        type: 'qa',
        status: 'idle',
        activity: 'Waiting for code to test',
        progress: 0,
        lastActivity: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
        performance: {
          cpu: 15,
          memory: 35,
          tasksCompleted: 5,
          successRate: 92
        },
        health: {
          responseTime: 100,
          errorCount: 0,
          uptime: 99.1,
          lastHealthCheck: new Date().toISOString()
        }
      }
    ]

    this.systemMetrics = {
      totalAgents: 5,
      activeAgents: 2,
      totalTasks: 15,
      completedTasks: 12,
      errorRate: 2.1,
      averageResponseTime: 150,
      systemLoad: 65,
      lastUpdated: new Date().toISOString()
    }
  }

  private handleAgentClick(agent: AgentStatus) {
    this.selectedAgent = this.selectedAgent?.id === agent.id ? null : agent
    
    this.dispatchEvent(new CustomEvent('agent-selected', {
      detail: { agent: this.selectedAgent },
      bubbles: true,
      composed: true
    }))
  }

  private toggleAutoRefresh() {
    this.autoRefresh = !this.autoRefresh
    
    if (this.autoRefresh && this.connectionStatus) {
      this.websocketService.enableHighFrequencyMode()
    } else {
      this.websocketService.enableLowFrequencyMode()
    }
  }

  private cleanup() {
    // Unsubscribe from all WebSocket subscriptions
    this.updateSubscriptions.forEach(unsubscribe => unsubscribe())
    this.updateSubscriptions = []
  }

  private getAgentIcon(type: string): string {
    const icons = {
      architect: '🏗️',
      developer: '💻',
      qa: '🧪',
      reviewer: '👀',
      meta: '🤖'
    }
    return icons[type as keyof typeof icons] || '⚡'
  }

  private getTypeAbbreviation(type: string): string {
    const abbrevs = {
      architect: 'AR',
      developer: 'DE',
      qa: 'QA',
      reviewer: 'RE',
      meta: 'ME'
    }
    return abbrevs[type as keyof typeof abbrevs] || 'AG'
  }

  private renderSystemOverview() {
    return html`
      <div class="system-overview">
        <div class="metric-item">
          <div class="metric-value">${this.systemMetrics.activeAgents}/${this.systemMetrics.totalAgents}</div>
          <div class="metric-label">Active</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${this.systemMetrics.completedTasks}</div>
          <div class="metric-label">Completed</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${this.systemMetrics.averageResponseTime}ms</div>
          <div class="metric-label">Avg Response</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${this.systemMetrics.systemLoad}%</div>
          <div class="metric-label">System Load</div>
        </div>
      </div>
    `
  }

  private renderAgent(agent: AgentStatus) {
    return html`
      <div 
        class="agent-card ${agent.status} ${this.selectedAgent?.id === agent.id ? 'selected' : ''}"
        @click="${() => this.handleAgentClick(agent)}"
      >
        <div class="agent-header">
          <div class="agent-info">
            <div class="agent-avatar ${agent.type}">
              ${this.getTypeAbbreviation(agent.type)}
            </div>
            <div>
              <div class="agent-name">${agent.name}</div>
              <div class="agent-type">${agent.type}</div>
            </div>
          </div>
          <div class="status-badge ${agent.status}">
            ${agent.status}
          </div>
        </div>

        <div class="agent-activity">
          <div class="activity-icon"></div>
          ${agent.activity}
        </div>

        ${!this.compact ? html`
          <div class="agent-metrics">
            <div class="metric-small">
              CPU: <span class="value">${agent.performance.cpu}%</span>
            </div>
            <div class="metric-small">
              Memory: <span class="value">${agent.performance.memory}%</span>
            </div>
            <div class="metric-small">
              Tasks: <span class="value">${agent.performance.tasksCompleted}</span>
            </div>
            <div class="metric-small">
              Success: <span class="value">${agent.performance.successRate}%</span>
            </div>
          </div>
        ` : ''}

        ${agent.currentTask ? html`
          <div class="current-task">
            <div class="task-title">${agent.currentTask.title}</div>
            <div class="task-progress">${agent.currentTask.progress}% • ${agent.currentTask.estimatedTime}</div>
            <div class="progress-bar">
              <div class="progress-fill" style="width: ${agent.currentTask.progress}%"></div>
            </div>
          </div>
        ` : ''}

        <div class="performance-chart">
          <div class="performance-line" style="height: ${agent.performance.cpu}%"></div>
        </div>
      </div>
    `
  }

  render() {
    if (this.agents.length === 0) {
      return html`
        <div class="monitor-header">
          <h3 class="header-title">
            <span class="connection-indicator ${this.connectionQuality}"></span>
            Live Agent Monitor
          </h3>
        </div>
        <div class="empty-state">
          <div class="empty-icon">🤖</div>
          <p>No agents currently active</p>
          <p>Agents will appear here when they come online</p>
        </div>
      `
    }

    const displayAgents = this.maxAgents ? this.agents.slice(0, this.maxAgents) : this.agents

    return html`
      <div class="monitor-header">
        <h3 class="header-title">
          <span class="connection-indicator ${this.connectionQuality}"></span>
          Live Agent Monitor
        </h3>
        <div class="controls">
          <button
            class="control-button ${this.autoRefresh ? 'active' : ''}"
            @click="${this.toggleAutoRefresh}"
            title="Auto-refresh"
          >
            🔄
          </button>
          <button
            class="control-button"
            @click="${this.loadInitialData}"
            title="Refresh now"
          >
            ↻
          </button>
        </div>
      </div>

      ${this.renderSystemOverview()}

      <div class="agents-grid ${this.compact ? 'compact' : ''}">
        ${displayAgents.map(agent => this.renderAgent(agent))}
      </div>

      ${this.lastUpdate ? html`
        <div style="text-align: center; margin-top: 1rem; font-size: 0.75rem; color: #6b7280;">
          Last updated: ${this.lastUpdate.toLocaleTimeString()}
        </div>
      ` : ''}
    `
  }
}