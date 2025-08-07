import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../../services/websocket'
import { Agent, AgentStatus, AgentRole } from '../../types/api'

interface AgentOversightMetrics {
  id: string
  name: string
  role: AgentRole
  status: AgentStatus
  health: 'excellent' | 'good' | 'fair' | 'critical'
  performance: {
    efficiency: number
    accuracy: number
    responsiveness: number
    taskCompletionRate: number
  }
  currentTask?: {
    id: string
    title: string
    progress: number
    estimatedCompletion?: string
  }
  recentActivity: ActivityEvent[]
  connectionLatency: number
  lastHeartbeat: Date
}

interface ActivityEvent {
  timestamp: string
  type: 'task-started' | 'task-completed' | 'error' | 'communication' | 'decision'
  description: string
  severity: 'info' | 'warning' | 'error'
}

interface TeamCoordinationMetrics {
  coordinationEfficiency: number
  communicationVolume: number
  decisionMakingSpeed: number
  conflictResolution: number
  overallTeamHealth: 'excellent' | 'good' | 'fair' | 'critical'
}

@customElement('multi-agent-oversight-dashboard')
export class MultiAgentOversightDashboard extends LitElement {
  @property({ type: Boolean }) declare fullscreen: boolean
  @property({ type: String }) declare viewMode: 'grid' | 'list' | 'performance' | 'coordination'
  
  @state() private declare agents: AgentOversightMetrics[]
  @state() private declare teamMetrics: TeamCoordinationMetrics
  @state() private declare selectedAgents: Set<string>
  @state() private declare alertsCount: number
  @state() private declare systemLoad: number
  @state() private declare connectionQuality: 'excellent' | 'good' | 'poor' | 'offline'
  @state() private declare lastUpdate: Date
  @state() private declare emergencyMode: boolean
  @state() private declare filterStatus: 'all' | AgentStatus
  @state() private declare showPerformanceDetails: boolean
  @state() private declare realtimeStreaming: boolean
  
  private websocketService: WebSocketService
  private metricsUpdateInterval: number | null = null
  private connectionQualitySubscription?: () => void
  private agentMetricsSubscription?: () => void
  private criticalEventsSubscription?: () => void

  static styles = css`
    :host {
      display: block;
      height: 100vh;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      overflow: hidden;
    }

    .dashboard-container {
      height: 100%;
      display: flex;
      flex-direction: column;
      position: relative;
    }

    .emergency-banner {
      background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
      color: white;
      padding: 1rem;
      text-align: center;
      font-weight: bold;
      animation: pulse 1s infinite;
      z-index: 1000;
    }

    .header {
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(148, 163, 184, 0.2);
      padding: 1rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: between;
      z-index: 100;
    }

    .header-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
    }

    .header-title {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .status-indicator {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #10b981;
      position: relative;
    }

    .status-indicator.good { background: #f59e0b; }
    .status-indicator.poor { background: #ef4444; }
    .status-indicator.offline { background: #64748b; }

    .status-indicator::after {
      content: '';
      position: absolute;
      top: -4px;
      left: -4px;
      right: -4px;
      bottom: -4px;
      border-radius: 50%;
      background: currentColor;
      opacity: 0.3;
      animation: ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;
    }

    .header-stats {
      display: flex;
      align-items: center;
      gap: 2rem;
      font-size: 0.875rem;
    }

    .stat-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.25rem;
    }

    .stat-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #10b981;
    }

    .stat-value.warning { color: #f59e0b; }
    .stat-value.error { color: #ef4444; }

    .stat-label {
      color: #94a3b8;
      text-transform: uppercase;
      font-size: 0.75rem;
      font-weight: 600;
    }

    .controls {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .control-button {
      background: rgba(59, 130, 246, 0.1);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #60a5fa;
      padding: 0.5rem 1rem;
      border-radius: 8px;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .control-button:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: rgba(59, 130, 246, 0.5);
    }

    .control-button.active {
      background: #3b82f6;
      color: white;
      border-color: #3b82f6;
    }

    .control-button.emergency {
      background: rgba(239, 68, 68, 0.1);
      border-color: rgba(239, 68, 68, 0.3);
      color: #f87171;
    }

    .control-button.emergency:hover {
      background: #ef4444;
      color: white;
    }

    .filters {
      background: rgba(30, 41, 59, 0.8);
      padding: 1rem 1.5rem;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
      display: flex;
      align-items: center;
      gap: 1rem;
      overflow-x: auto;
    }

    .filter-chip {
      padding: 0.5rem 1rem;
      border-radius: 20px;
      border: 1px solid rgba(148, 163, 184, 0.3);
      background: transparent;
      color: #94a3b8;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }

    .filter-chip:hover {
      border-color: #3b82f6;
      color: #60a5fa;
    }

    .filter-chip.active {
      background: #3b82f6;
      border-color: #3b82f6;
      color: white;
    }

    .main-content {
      flex: 1;
      overflow: hidden;
      position: relative;
    }

    .agents-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
      gap: 1.5rem;
      padding: 1.5rem;
      height: 100%;
      overflow-y: auto;
    }

    .agent-card {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 16px;
      padding: 1.5rem;
      transition: all 0.3s;
      position: relative;
      overflow: hidden;
    }

    .agent-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: #10b981;
      transition: all 0.3s;
    }

    .agent-card.good::before { background: #f59e0b; }
    .agent-card.fair::before { background: #ef4444; }
    .agent-card.critical::before { 
      background: #dc2626; 
      animation: pulse 1s infinite;
    }

    .agent-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
      border-color: rgba(59, 130, 246, 0.5);
    }

    .agent-card.selected {
      border-color: #3b82f6;
      background: rgba(59, 130, 246, 0.1);
    }

    .agent-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .agent-info {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .agent-avatar {
      width: 56px;
      height: 56px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.75rem;
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      position: relative;
    }

    .agent-avatar.product-manager { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .agent-avatar.architect { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
    .agent-avatar.backend-developer { background: linear-gradient(135deg, #10b981, #059669); }
    .agent-avatar.frontend-developer { background: linear-gradient(135deg, #ec4899, #db2777); }
    .agent-avatar.qa-engineer { background: linear-gradient(135deg, #06b6d4, #0891b2); }

    .agent-meta {
      flex: 1;
    }

    .agent-name {
      font-size: 1.125rem;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 0.25rem;
    }

    .agent-role {
      color: #94a3b8;
      font-size: 0.875rem;
      text-transform: capitalize;
    }

    .agent-status-badge {
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .agent-status-badge.active {
      background: rgba(16, 185, 129, 0.2);
      color: #10b981;
    }

    .agent-status-badge.idle {
      background: rgba(245, 158, 11, 0.2);
      color: #f59e0b;
    }

    .agent-status-badge.busy {
      background: rgba(59, 130, 246, 0.2);
      color: #3b82f6;
    }

    .agent-status-badge.error {
      background: rgba(239, 68, 68, 0.2);
      color: #ef4444;
    }

    .performance-metrics {
      margin: 1.5rem 0;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
    }

    .metric-item {
      background: rgba(15, 23, 42, 0.6);
      border-radius: 8px;
      padding: 1rem;
      text-align: center;
    }

    .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #10b981;
      margin-bottom: 0.25rem;
    }

    .metric-label {
      font-size: 0.75rem;
      color: #94a3b8;
      text-transform: uppercase;
    }

    .current-task {
      background: rgba(15, 23, 42, 0.4);
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
      border-left: 4px solid #3b82f6;
    }

    .task-title {
      font-weight: 600;
      margin-bottom: 0.5rem;
    }

    .task-progress {
      width: 100%;
      height: 6px;
      background: rgba(148, 163, 184, 0.2);
      border-radius: 3px;
      overflow: hidden;
    }

    .task-progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #3b82f6, #1d4ed8);
      transition: width 0.5s ease;
    }

    .recent-activity {
      max-height: 200px;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }

    .activity-event {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      padding: 0.75rem;
      border-radius: 6px;
      margin-bottom: 0.5rem;
      background: rgba(15, 23, 42, 0.3);
    }

    .activity-event.warning {
      border-left: 3px solid #f59e0b;
    }

    .activity-event.error {
      border-left: 3px solid #ef4444;
    }

    .activity-time {
      font-size: 0.75rem;
      color: #64748b;
      white-space: nowrap;
    }

    .activity-description {
      font-size: 0.875rem;
      line-height: 1.4;
    }

    .agent-actions {
      display: flex;
      gap: 0.5rem;
      margin-top: 1rem;
    }

    .agent-action-btn {
      flex: 1;
      background: rgba(59, 130, 246, 0.1);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #60a5fa;
      padding: 0.5rem;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s;
    }

    .agent-action-btn:hover {
      background: rgba(59, 130, 246, 0.2);
    }

    .agent-action-btn.danger {
      border-color: rgba(239, 68, 68, 0.3);
      color: #f87171;
    }

    .agent-action-btn.danger:hover {
      background: rgba(239, 68, 68, 0.1);
    }

    .team-coordination-panel {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 20px;
      padding: 2rem;
      width: 90%;
      max-width: 800px;
      max-height: 80vh;
      overflow-y: auto;
      z-index: 1000;
    }

    .floating-controls {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      z-index: 100;
    }

    .floating-btn {
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      border: none;
      color: white;
      font-size: 1.5rem;
      cursor: pointer;
      box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
      transition: all 0.3s;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .floating-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 48px rgba(59, 130, 246, 0.4);
    }

    .floating-btn.emergency {
      background: linear-gradient(135deg, #ef4444, #dc2626);
      box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
    }

    /* Responsive design */
    @media (max-width: 768px) {
      .agents-grid {
        grid-template-columns: 1fr;
        padding: 1rem;
        gap: 1rem;
      }

      .header-stats {
        display: none;
      }

      .performance-metrics {
        grid-template-columns: 1fr;
      }
    }

    /* Animations */
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }

    @keyframes ping {
      75%, 100% {
        transform: scale(2);
        opacity: 0;
      }
    }

    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .agent-card {
      animation: slideIn 0.3s ease-out;
    }
  `

  constructor() {
    super()
    this.fullscreen = false
    this.viewMode = 'grid'
    this.agents = []
    this.teamMetrics = {
      coordinationEfficiency: 0,
      communicationVolume: 0,
      decisionMakingSpeed: 0,
      conflictResolution: 0,
      overallTeamHealth: 'good'
    }
    this.selectedAgents = new Set()
    this.alertsCount = 0
    this.systemLoad = 0
    this.connectionQuality = 'offline'
    this.lastUpdate = new Date()
    this.emergencyMode = false
    this.filterStatus = 'all'
    this.showPerformanceDetails = false
    this.realtimeStreaming = true
    
    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupWebSocketSubscriptions()
    this.startRealTimeUpdates()
    this.loadInitialData()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanup()
  }

  private setupWebSocketSubscriptions() {
    // Subscribe to enhanced agent metrics
    this.agentMetricsSubscription = this.websocketService.subscribeToAgentMetrics(
      (data) => this.handleAgentMetricsUpdate(data)
    )

    // Subscribe to connection quality
    this.connectionQualitySubscription = this.websocketService.subscribeToConnectionQuality(
      (data) => this.handleConnectionQualityUpdate(data)
    )

    // Subscribe to critical events
    this.criticalEventsSubscription = this.websocketService.subscribeToCriticalEvents(
      (data) => this.handleCriticalEvent(data)
    )

    // Enable high-frequency streaming for real-time oversight
    if (this.realtimeStreaming) {
      this.websocketService.enableHighFrequencyMode()
    }
  }

  private async loadInitialData() {
    // Load initial agent data (mock data for demonstration)
    this.agents = await this.generateMockAgentData()
    this.updateSystemMetrics()
  }

  private async generateMockAgentData(): Promise<AgentOversightMetrics[]> {
    const roles: AgentRole[] = [
      AgentRole.PRODUCT_MANAGER,
      AgentRole.ARCHITECT,
      AgentRole.BACKEND_DEVELOPER,
      AgentRole.FRONTEND_DEVELOPER,
      AgentRole.QA_ENGINEER
    ]

    const statuses: AgentStatus[] = [
      AgentStatus.ACTIVE,
      AgentStatus.BUSY,
      AgentStatus.IDLE,
      AgentStatus.ACTIVE,
      AgentStatus.BUSY
    ]

    return roles.map((role, index) => ({
      id: `agent-${index + 1}`,
      name: `${role.replace('_', ' ')} Agent ${index + 1}`,
      role,
      status: statuses[index],
      health: index === 2 ? 'fair' : index === 4 ? 'excellent' : 'good',
      performance: {
        efficiency: 75 + Math.random() * 25,
        accuracy: 80 + Math.random() * 20,
        responsiveness: 70 + Math.random() * 30,
        taskCompletionRate: 85 + Math.random() * 15
      },
      currentTask: statuses[index] !== AgentStatus.IDLE ? {
        id: `task-${index + 1}`,
        title: `Implementing ${role.toLowerCase().replace('_', ' ')} feature`,
        progress: Math.floor(Math.random() * 100),
        estimatedCompletion: new Date(Date.now() + Math.random() * 3600000).toISOString()
      } : undefined,
      recentActivity: this.generateMockActivity(),
      connectionLatency: Math.floor(Math.random() * 200),
      lastHeartbeat: new Date()
    }))
  }

  private generateMockActivity(): ActivityEvent[] {
    const activities = [
      { type: 'task-started' as const, description: 'Started implementing user authentication', severity: 'info' as const },
      { type: 'communication' as const, description: 'Coordinated with backend team on API design', severity: 'info' as const },
      { type: 'decision' as const, description: 'Chose React over Vue for frontend framework', severity: 'info' as const },
      { type: 'task-completed' as const, description: 'Completed database schema design', severity: 'info' as const },
      { type: 'error' as const, description: 'Connection timeout to external API', severity: 'warning' as const }
    ]

    return activities.slice(0, 3).map(activity => ({
      ...activity,
      timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString()
    }))
  }

  private handleAgentMetricsUpdate(data: any) {
    // Update agent metrics from real-time data
    const agentId = data.agentId
    const agent = this.agents.find(a => a.id === agentId)
    
    if (agent && data.metrics) {
      agent.performance = {
        ...agent.performance,
        ...data.metrics
      }
      agent.lastHeartbeat = new Date()
      agent.connectionLatency = data.latency || agent.connectionLatency
      
      // Update health based on performance
      const avgPerformance = (
        agent.performance.efficiency +
        agent.performance.accuracy +
        agent.performance.responsiveness +
        agent.performance.taskCompletionRate
      ) / 4
      
      if (avgPerformance >= 90) agent.health = 'excellent'
      else if (avgPerformance >= 75) agent.health = 'good'
      else if (avgPerformance >= 60) agent.health = 'fair'
      else agent.health = 'critical'
      
      this.requestUpdate()
    }
  }

  private handleConnectionQualityUpdate(data: { quality: string, timestamp: string }) {
    this.connectionQuality = data.quality as any
    this.lastUpdate = new Date(data.timestamp)
  }

  private handleCriticalEvent(data: any) {
    // Handle critical events that may require immediate attention
    if (data.severity === 'critical') {
      this.emergencyMode = true
      this.alertsCount++
      
      // Auto-clear emergency mode after 30 seconds if no new critical events
      setTimeout(() => {
        this.emergencyMode = false
      }, 30000)
    }
  }

  private updateSystemMetrics() {
    // Update system-wide metrics
    const activeAgents = this.agents.filter(a => a.status === AgentStatus.ACTIVE).length
    const totalAgents = this.agents.length
    const avgPerformance = this.agents.reduce((sum, agent) => {
      const agentAvg = (
        agent.performance.efficiency +
        agent.performance.accuracy +
        agent.performance.responsiveness +
        agent.performance.taskCompletionRate
      ) / 4
      return sum + agentAvg
    }, 0) / totalAgents

    this.systemLoad = (activeAgents / totalAgents) * 100
    
    // Update team coordination metrics
    this.teamMetrics = {
      coordinationEfficiency: Math.min(avgPerformance + 5, 100),
      communicationVolume: 60 + Math.random() * 40,
      decisionMakingSpeed: 70 + Math.random() * 30,
      conflictResolution: 85 + Math.random() * 15,
      overallTeamHealth: avgPerformance >= 85 ? 'excellent' : avgPerformance >= 70 ? 'good' : 'fair'
    }
  }

  private startRealTimeUpdates() {
    // Update metrics every 5 seconds
    this.metricsUpdateInterval = window.setInterval(() => {
      this.updateSystemMetrics()
      this.requestUpdate()
    }, 5000)
  }

  private cleanup() {
    if (this.metricsUpdateInterval) {
      clearInterval(this.metricsUpdateInterval)
    }
    
    // Cleanup subscriptions
    if (this.connectionQualitySubscription) {
      this.connectionQualitySubscription()
    }
    if (this.agentMetricsSubscription) {
      this.agentMetricsSubscription()
    }
    if (this.criticalEventsSubscription) {
      this.criticalEventsSubscription()
    }
  }

  private toggleAgentSelection(agentId: string) {
    if (this.selectedAgents.has(agentId)) {
      this.selectedAgents.delete(agentId)
    } else {
      this.selectedAgents.add(agentId)
    }
    this.requestUpdate()
  }

  private async executeEmergencyStop() {
    if (confirm('Are you sure you want to execute an emergency stop for all agents?')) {
      this.websocketService.sendEmergencyStop('Manual emergency stop from oversight dashboard')
      this.emergencyMode = true
    }
  }

  private async restartSelectedAgents() {
    const selectedIds = Array.from(this.selectedAgents)
    if (selectedIds.length > 0) {
      this.websocketService.sendBulkAgentCommand(selectedIds, 'restart')
      this.selectedAgents.clear()
      this.requestUpdate()
    }
  }

  private getAgentIcon(role: AgentRole): string {
    const icons = {
      [AgentRole.PRODUCT_MANAGER]: '📋',
      [AgentRole.ARCHITECT]: '🏗️',
      [AgentRole.BACKEND_DEVELOPER]: '⚙️',
      [AgentRole.FRONTEND_DEVELOPER]: '🎨',
      [AgentRole.QA_ENGINEER]: '🔍'
    }
    return icons[role] || '🤖'
  }

  private formatTime(date: Date): string {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    
    if (diff < 60000) return 'now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m`
    return `${Math.floor(diff / 3600000)}h`
  }

  private getFilteredAgents(): AgentOversightMetrics[] {
    if (this.filterStatus === 'all') {
      return this.agents
    }
    return this.agents.filter(agent => agent.status === this.filterStatus)
  }

  render() {
    const filteredAgents = this.getFilteredAgents()
    const activeAgents = this.agents.filter(a => a.status === AgentStatus.ACTIVE).length
    const errorAgents = this.agents.filter(a => a.health === 'critical').length

    return html`
      <div class="dashboard-container">
        ${this.emergencyMode ? html`
          <div class="emergency-banner">
            🚨 EMERGENCY MODE ACTIVE - System requires immediate attention
          </div>
        ` : ''}

        <div class="header">
          <div class="header-content">
            <div class="header-title">
              <div class="status-indicator ${this.connectionQuality}"></div>
              <h1>🤖 Multi-Agent Oversight</h1>
            </div>

            <div class="header-stats">
              <div class="stat-item">
                <div class="stat-value ${activeAgents < 3 ? 'warning' : ''}">${activeAgents}</div>
                <div class="stat-label">Active</div>
              </div>
              <div class="stat-item">
                <div class="stat-value ${errorAgents > 0 ? 'error' : ''}">${errorAgents}</div>
                <div class="stat-label">Issues</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">${Math.round(this.systemLoad)}%</div>
                <div class="stat-label">Load</div>
              </div>
            </div>

            <div class="controls">
              <button class="control-button ${this.realtimeStreaming ? 'active' : ''}"
                      @click=${() => { 
                        this.realtimeStreaming = !this.realtimeStreaming
                        if (this.realtimeStreaming) {
                          this.websocketService.enableHighFrequencyMode()
                        } else {
                          this.websocketService.enableLowFrequencyMode()
                        }
                      }}>
                📡 Live
              </button>
              
              <button class="control-button emergency" @click=${this.executeEmergencyStop}>
                🛑 Emergency Stop
              </button>
            </div>
          </div>
        </div>

        <div class="filters">
          ${(['all', AgentStatus.ACTIVE, AgentStatus.BUSY, AgentStatus.IDLE, AgentStatus.ERROR] as const).map(status => html`
            <button class="filter-chip ${this.filterStatus === status ? 'active' : ''}"
                    @click=${() => { this.filterStatus = status; this.requestUpdate() }}>
              ${status === 'all' ? 'All Agents' : status}
              ${status !== 'all' ? html`<span>(${this.agents.filter(a => a.status === status).length})</span>` : ''}
            </button>
          `)}
        </div>

        <div class="main-content">
          <div class="agents-grid">
            ${filteredAgents.map(agent => html`
              <div class="agent-card ${agent.health} ${this.selectedAgents.has(agent.id) ? 'selected' : ''}"
                   @click=${() => this.toggleAgentSelection(agent.id)}>
                
                <div class="agent-header">
                  <div class="agent-info">
                    <div class="agent-avatar ${agent.role}">
                      ${this.getAgentIcon(agent.role)}
                    </div>
                    <div class="agent-meta">
                      <div class="agent-name">${agent.name}</div>
                      <div class="agent-role">${agent.role.replace('_', ' ')}</div>
                    </div>
                  </div>
                  <div class="agent-status-badge ${agent.status}">
                    <div class="status-indicator ${agent.status}"></div>
                    ${agent.status}
                  </div>
                </div>

                <div class="performance-metrics">
                  <div class="metric-item">
                    <div class="metric-value">${Math.round(agent.performance.efficiency)}%</div>
                    <div class="metric-label">Efficiency</div>
                  </div>
                  <div class="metric-item">
                    <div class="metric-value">${Math.round(agent.performance.accuracy)}%</div>
                    <div class="metric-label">Accuracy</div>
                  </div>
                  <div class="metric-item">
                    <div class="metric-value">${agent.connectionLatency}ms</div>
                    <div class="metric-label">Latency</div>
                  </div>
                  <div class="metric-item">
                    <div class="metric-value">${Math.round(agent.performance.taskCompletionRate)}%</div>
                    <div class="metric-label">Success Rate</div>
                  </div>
                </div>

                ${agent.currentTask ? html`
                  <div class="current-task">
                    <div class="task-title">${agent.currentTask.title}</div>
                    <div class="task-progress">
                      <div class="task-progress-fill" style="width: ${agent.currentTask.progress}%"></div>
                    </div>
                  </div>
                ` : ''}

                <div class="recent-activity">
                  ${agent.recentActivity.map(activity => html`
                    <div class="activity-event ${activity.severity}">
                      <div class="activity-time">${this.formatTime(new Date(activity.timestamp))}</div>
                      <div class="activity-description">${activity.description}</div>
                    </div>
                  `)}
                </div>

                <div class="agent-actions">
                  <button class="agent-action-btn" 
                          @click=${(e: Event) => {
                            e.stopPropagation()
                            this.websocketService.sendAgentCommand(agent.id, 'pause')
                          }}>
                    Pause
                  </button>
                  <button class="agent-action-btn" 
                          @click=${(e: Event) => {
                            e.stopPropagation()
                            this.websocketService.sendAgentCommand(agent.id, 'restart')
                          }}>
                    Restart
                  </button>
                  <button class="agent-action-btn danger" 
                          @click=${(e: Event) => {
                            e.stopPropagation()
                            this.websocketService.sendAgentCommand(agent.id, 'terminate')
                          }}>
                    Stop
                  </button>
                </div>
              </div>
            `)}
          </div>
        </div>

        ${this.selectedAgents.size > 0 ? html`
          <div class="floating-controls">
            <button class="floating-btn" @click=${this.restartSelectedAgents} title="Restart Selected">
              🔄
            </button>
            <button class="floating-btn emergency" @click=${this.executeEmergencyStop} title="Emergency Stop">
              🛑
            </button>
          </div>
        ` : ''}
      </div>
    `
  }
}