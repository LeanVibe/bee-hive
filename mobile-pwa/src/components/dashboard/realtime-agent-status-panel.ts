import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

interface AgentHealthIndicator {
  agent_id: string
  name: string
  status: string // 'online', 'offline', 'error'
  health_score: number // 0-100
  last_heartbeat?: string
  specialization_badges: string[]
  performance_metrics: {
    context_usage?: number
    response_time?: number
    task_completion_rate?: number
  }
  current_task?: string
  error_count: number
}

interface CommunicationHealth {
  redis_status: string
  average_latency: number
  message_throughput: number
  error_rate: number
  connected_agents: number
  recent_latencies: number[]
}

@customElement('realtime-agent-status-panel')
export class RealtimeAgentStatusPanel extends LitElement {
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: Boolean }) declare compact: boolean
  @property({ type: String }) declare viewMode: string // 'grid', 'list', 'compact'

  @state() private declare agents: AgentHealthIndicator[]
  @state() private declare communicationHealth: CommunicationHealth | null
  @state() private declare selectedAgent: AgentHealthIndicator | null
  @state() private declare isLoading: boolean
  @state() private declare error: string
  @state() private declare lastUpdate: Date | null
  @state() private declare sortBy: string
  @state() private declare filterStatus: string

  private updateInterval: number | null = null

  static styles = css`
    :host {
      display: block;
      background: white;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }

    .panel-header {
      background: linear-gradient(135deg, #059669 0%, #10b981 100%);
      color: white;
      padding: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .panel-title {
      font-size: 1.125rem;
      font-weight: 600;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .communication-status {
      display: flex;
      align-items: center;
      gap: 1rem;
      background: rgba(255, 255, 255, 0.1);
      padding: 0.5rem 1rem;
      border-radius: 8px;
    }

    .redis-status {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.875rem;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .status-dot.healthy {
      background: #10b981;
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
    }

    .status-dot.unhealthy {
      background: #ef4444;
      box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
    }

    .latency-display {
      font-size: 0.875rem;
    }

    .panel-content {
      padding: 1rem;
    }

    .controls-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .view-controls {
      display: flex;
      gap: 0.5rem;
    }

    .control-btn {
      padding: 0.5rem 0.75rem;
      border: 1px solid #d1d5db;
      background: white;
      border-radius: 6px;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s;
    }

    .control-btn:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }

    .control-btn.active {
      background: #3b82f6;
      color: white;
      border-color: #3b82f6;
    }

    .filter-controls {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .filter-select {
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      font-size: 0.875rem;
      background: white;
    }

    .agents-container {
      min-height: 300px;
    }

    .agents-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1rem;
    }

    .agents-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .agents-compact {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 0.75rem;
    }

    .agent-card {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1rem;
      cursor: pointer;
      transition: all 0.2s;
      position: relative;
    }

    .agent-card:hover {
      border-color: #3b82f6;
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .agent-card.selected {
      border-color: #3b82f6;
      background: #eff6ff;
    }

    .agent-card.error {
      border-color: #ef4444;
      background: #fef2f2;
    }

    .agent-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }

    .agent-name {
      font-weight: 600;
      color: #111827;
      margin: 0;
    }

    .agent-status-badge {
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .status-online {
      background: #dcfce7;
      color: #166534;
    }

    .status-offline {
      background: #fef3c7;
      color: #92400e;
    }

    .status-error {
      background: #fee2e2;
      color: #991b1b;
    }

    .health-score-display {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }

    .health-score-circle {
      position: relative;
      width: 60px;
      height: 60px;
    }

    .health-score-svg {
      transform: rotate(-90deg);
    }

    .health-score-bg {
      fill: none;
      stroke: #e5e7eb;
      stroke-width: 4;
    }

    .health-score-fill {
      fill: none;
      stroke-width: 4;
      stroke-linecap: round;
      transition: stroke-dasharray 0.6s ease;
    }

    .health-score-text {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 0.875rem;
      font-weight: 600;
    }

    .performance-metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
      margin-bottom: 0.75rem;
    }

    .metric-item {
      text-align: center;
    }

    .metric-value {
      font-weight: 600;
      color: #111827;
    }

    .metric-label {
      font-size: 0.75rem;
      color: #6b7280;
    }

    .specialization-badges {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 0.75rem;
    }

    .badge {
      padding: 0.25rem 0.5rem;
      background: #3b82f6;
      color: white;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
    }

    .current-task {
      background: #f3f4f6;
      border-radius: 6px;
      padding: 0.5rem;
      font-size: 0.875rem;
      color: #374151;
      border-left: 3px solid #3b82f6;
    }

    .last-heartbeat {
      font-size: 0.75rem;
      color: #6b7280;
      margin-top: 0.5rem;
      text-align: right;
    }

    .agent-actions {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      display: flex;
      gap: 0.25rem;
      opacity: 0;
      transition: opacity 0.2s;
    }

    .agent-card:hover .agent-actions {
      opacity: 1;
    }

    .action-btn {
      width: 28px;
      height: 28px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      transition: all 0.2s;
    }

    .restart-btn {
      background: #f59e0b;
      color: white;
    }

    .restart-btn:hover {
      background: #d97706;
    }

    .error-count-badge {
      position: absolute;
      top: -8px;
      right: -8px;
      background: #ef4444;
      color: white;
      border-radius: 50%;
      width: 24px;
      height: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      font-weight: 600;
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 200px;
      color: #6b7280;
      gap: 1rem;
    }

    .loading-state {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 200px;
      color: #6b7280;
    }

    .error-state {
      background: #fef2f2;
      color: #dc2626;
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
      margin: 1rem 0;
    }

    @media (max-width: 768px) {
      .controls-bar {
        flex-direction: column;
        align-items: flex-start;
      }

      .agents-grid {
        grid-template-columns: 1fr;
      }

      .agents-compact {
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      }

      .agent-card {
        padding: 0.75rem;
      }

      .performance-metrics {
        grid-template-columns: 1fr;
      }
    }
  `

  constructor() {
    super()
    this.realtime = true
    this.compact = false
    this.viewMode = 'grid'
    this.agents = []
    this.communicationHealth = null
    this.selectedAgent = null
    this.isLoading = true
    this.error = ''
    this.lastUpdate = null
    this.sortBy = 'name'
    this.filterStatus = 'all'
  }

  connectedCallback() {
    super.connectedCallback()
    this.loadData()
    
    if (this.realtime) {
      this.startRealtimeUpdates()
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.stopRealtimeUpdates()
  }

  private async loadData() {
    this.isLoading = true
    this.error = ''

    try {
      // Load coordination dashboard data for agent health
      const response = await fetch('/api/v1/coordination-monitoring/dashboard')
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const dashboardData = await response.json()
      
      this.agents = dashboardData.agent_health || []
      this.communicationHealth = dashboardData.communication_health
      this.lastUpdate = new Date()

    } catch (error) {
      console.error('Failed to load agent status data:', error)
      this.error = error instanceof Error ? error.message : 'Unknown error occurred'
    } finally {
      this.isLoading = false
    }
  }

  private startRealtimeUpdates() {
    // Update every 3 seconds for real-time agent monitoring
    this.updateInterval = window.setInterval(() => {
      this.loadData()
    }, 3000)
  }

  private stopRealtimeUpdates() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
      this.updateInterval = null
    }
  }

  private async restartAgent(agentId: string) {
    try {
      const response = await fetch(`/api/v1/coordination-monitoring/recovery-actions/restart-agent/${agentId}`, {
        method: 'POST'
      })

      if (!response.ok) {
        throw new Error(`Restart failed: ${response.statusText}`)
      }

      const result = await response.json()
      console.log('Agent restart completed:', result)

      // Refresh data after restart
      setTimeout(() => this.loadData(), 2000)

      // Dispatch event
      this.dispatchEvent(new CustomEvent('agent-restarted', {
        detail: { agentId, result },
        bubbles: true,
        composed: true
      }))

    } catch (error) {
      console.error('Agent restart failed:', error)
      this.error = `Restart failed: ${error}`
    }
  }

  private selectAgent(agent: AgentHealthIndicator) {
    this.selectedAgent = this.selectedAgent?.agent_id === agent.agent_id ? null : agent

    this.dispatchEvent(new CustomEvent('agent-selected', {
      detail: { agent: this.selectedAgent },
      bubbles: true,
      composed: true
    }))
  }

  private getFilteredAndSortedAgents() {
    let filtered = this.agents

    // Apply status filter
    if (this.filterStatus !== 'all') {
      filtered = filtered.filter(agent => agent.status === this.filterStatus)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (this.sortBy) {
        case 'name':
          return a.name.localeCompare(b.name)
        case 'health':
          return b.health_score - a.health_score
        case 'status':
          return a.status.localeCompare(b.status)
        case 'errors':
          return b.error_count - a.error_count
        default:
          return 0
      }
    })

    return filtered
  }

  private getHealthScoreColor(score: number): string {
    if (score >= 80) return '#10b981'
    if (score >= 60) return '#f59e0b'
    return '#ef4444'
  }

  private formatLastHeartbeat(timestamp?: string): string {
    if (!timestamp) return 'Never'
    
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    
    if (diff < 60000) return 'Just now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
    return date.toLocaleDateString()
  }

  private renderHealthScoreCircle(score: number) {
    const radius = 24
    const circumference = 2 * Math.PI * radius
    const strokeDasharray = `${(score / 100) * circumference} ${circumference}`
    const color = this.getHealthScoreColor(score)

    return html`
      <div class="health-score-circle">
        <svg width="60" height="60" class="health-score-svg">
          <circle cx="30" cy="30" r="24" class="health-score-bg"></circle>
          <circle 
            cx="30" 
            cy="30" 
            r="24" 
            class="health-score-fill"
            stroke="${color}"
            stroke-dasharray="${strokeDasharray}"
          ></circle>
        </svg>
        <div class="health-score-text" style="color: ${color}">
          ${score.toFixed(0)}
        </div>
      </div>
    `
  }

  private renderAgent(agent: AgentHealthIndicator) {
    return html`
      <div 
        class="agent-card ${agent.status} ${this.selectedAgent?.agent_id === agent.agent_id ? 'selected' : ''}"
        @click=${() => this.selectAgent(agent)}
        role="button"
        tabindex="0"
        aria-label="Agent ${agent.name}, status ${agent.status}, health score ${agent.health_score}"
      >
        ${agent.error_count > 0 ? html`
          <div class="error-count-badge" title="${agent.error_count} recent errors">
            ${agent.error_count}
          </div>
        ` : ''}

        <div class="agent-actions">
          <button 
            class="action-btn restart-btn"
            @click=${(e: Event) => {
              e.stopPropagation()
              this.restartAgent(agent.agent_id)
            }}
            title="Restart agent"
            aria-label="Restart agent ${agent.name}"
          >
            🔄
          </button>
        </div>

        <div class="agent-header">
          <h3 class="agent-name">${agent.name}</h3>
          <div class="agent-status-badge status-${agent.status}">
            ${agent.status}
          </div>
        </div>

        <div class="health-score-display">
          ${this.renderHealthScoreCircle(agent.health_score)}
          <div>
            <div class="metric-value">${agent.health_score.toFixed(0)}/100</div>
            <div class="metric-label">Health Score</div>
          </div>
        </div>

        ${!this.compact ? html`
          <div class="performance-metrics">
            <div class="metric-item">
              <div class="metric-value">
                ${(agent.performance_metrics.context_usage || 0).toFixed(1)}%
              </div>
              <div class="metric-label">Context Usage</div>
            </div>

            <div class="metric-item">
              <div class="metric-value">
                ${(agent.performance_metrics.response_time || 0).toFixed(0)}ms
              </div>
              <div class="metric-label">Response Time</div>
            </div>

            <div class="metric-item">
              <div class="metric-value">
                ${(agent.performance_metrics.task_completion_rate || 0).toFixed(1)}%
              </div>
              <div class="metric-label">Task Success</div>
            </div>
          </div>

          ${agent.specialization_badges.length > 0 ? html`
            <div class="specialization-badges">
              ${agent.specialization_badges.map(badge => html`
                <span class="badge">${badge}</span>
              `)}
            </div>
          ` : ''}

          ${agent.current_task ? html`
            <div class="current-task">
              <strong>Current Task:</strong> ${agent.current_task}
            </div>
          ` : ''}
        ` : ''}

        <div class="last-heartbeat">
          Last seen: ${this.formatLastHeartbeat(agent.last_heartbeat)}
        </div>
      </div>
    `
  }

  render() {
    if (this.isLoading && this.agents.length === 0) {
      return html`
        <div class="loading-state">
          <div>Loading agent status data...</div>
        </div>
      `
    }

    const filteredAgents = this.getFilteredAndSortedAgents()

    return html`
      <div class="panel-header">
        <div class="panel-title">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Real-time Agent Status (${this.agents.length})
        </div>

        ${this.communicationHealth ? html`
          <div class="communication-status">
            <div class="redis-status">
              <div class="status-dot ${this.communicationHealth.redis_status === 'healthy' ? 'healthy' : 'unhealthy'}"></div>
              Redis ${this.communicationHealth.redis_status}
            </div>
            <div class="latency-display">
              ${this.communicationHealth.average_latency.toFixed(1)}ms avg
            </div>
          </div>
        ` : ''}
      </div>

      <div class="panel-content">
        ${this.error ? html`
          <div class="error-state">
            <p><strong>Error:</strong> ${this.error}</p>
            <button class="control-btn" @click=${() => this.loadData()}>
              Retry
            </button>
          </div>
        ` : ''}

        <div class="controls-bar">
          <div class="view-controls">
            <button 
              class="control-btn ${this.viewMode === 'grid' ? 'active' : ''}"
              @click=${() => this.viewMode = 'grid'}
            >
              Grid
            </button>
            <button 
              class="control-btn ${this.viewMode === 'list' ? 'active' : ''}"
              @click=${() => this.viewMode = 'list'}
            >
              List
            </button>
            <button 
              class="control-btn ${this.viewMode === 'compact' ? 'active' : ''}"
              @click=${() => this.viewMode = 'compact'}
            >
              Compact
            </button>
          </div>

          <div class="filter-controls">
            <select 
              class="filter-select"
              .value=${this.filterStatus}
              @change=${(e: Event) => this.filterStatus = (e.target as HTMLSelectElement).value}
            >
              <option value="all">All Status</option>
              <option value="online">Online</option>
              <option value="offline">Offline</option>
              <option value="error">Error</option>
            </select>

            <select 
              class="filter-select"
              .value=${this.sortBy}
              @change=${(e: Event) => this.sortBy = (e.target as HTMLSelectElement).value}
            >
              <option value="name">Sort by Name</option>
              <option value="health">Sort by Health</option>
              <option value="status">Sort by Status</option>
              <option value="errors">Sort by Errors</option>
            </select>
          </div>
        </div>

        <div class="agents-container">
          ${filteredAgents.length === 0 ? html`
            <div class="empty-state">
              <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M12 4.354a4 4 0 110 6.292 4 4 0 010-6.292zM15 21v-7a3 3 0 00-3-3H6a3 3 0 00-3 3v7" />
              </svg>
              <p>No agents found</p>
              <button class="control-btn" @click=${() => this.loadData()}>
                Refresh
              </button>
            </div>
          ` : html`
            <div class="agents-${this.viewMode}">
              ${filteredAgents.map(agent => this.renderAgent(agent))}
            </div>
          `}
        </div>

        ${this.lastUpdate ? html`
          <div class="last-update" style="font-size: 0.75rem; color: #6b7280; text-align: right; margin-top: 1rem;">
            Last updated: ${this.lastUpdate.toLocaleTimeString()}
          </div>
        ` : ''}
      </div>
    `
  }
}