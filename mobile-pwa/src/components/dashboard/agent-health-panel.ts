import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { repeat } from 'lit/directives/repeat.js'
import '../charts/sparkline-chart'

export interface AgentMetrics {
  cpuUsage: number[]
  memoryUsage: number[]
  tokenUsage: number[]
  tasksCompleted: number[]
  errorRate: number[]
  responseTime: number[]
  timestamps: string[]
}

export interface AgentStatus {
  id: string
  name: string
  status: 'active' | 'idle' | 'error' | 'offline'
  uptime: number
  lastSeen: string
  currentTask?: string
  metrics: AgentMetrics
  performance: {
    score: number
    trend: 'up' | 'down' | 'stable'
  }
}

@customElement('agent-health-panel')
export class AgentHealthPanel extends LitElement {
  @property({ type: Array }) agents: AgentStatus[] = []
  @property({ type: Boolean }) compact: boolean = false
  @property({ type: String }) sortBy: 'name' | 'status' | 'performance' | 'uptime' = 'name'
  @property({ type: String }) filterStatus: string = 'all'
  
  @state() private selectedAgent: string | null = null
  @state() private isRefreshing: boolean = false
  
  static styles = css`
    :host {
      display: block;
      background: white;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }
    
    .health-panel-header {
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
      background: #f9fafb;
    }
    
    .header-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }
    
    .panel-title {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .refresh-button {
      background: none;
      border: 1px solid #d1d5db;
      color: #374151;
      padding: 0.375rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .refresh-button:hover {
      background: #f3f4f6;
      border-color: #9ca3af;
    }
    
    .refresh-button.spinning svg {
      animation: spin 1s linear infinite;
    }
    
    .panel-filters {
      display: flex;
      gap: 0.5rem;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .filter-select {
      padding: 0.375rem 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.25rem;
      font-size: 0.875rem;
      background: white;
    }
    
    .status-summary {
      display: flex;
      gap: 1rem;
      font-size: 0.875rem;
      color: #6b7280;
    }
    
    .status-count {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    
    .status-dot.active { background: #10b981; }
    .status-dot.idle { background: #f59e0b; }
    .status-dot.error { background: #ef4444; }
    .status-dot.offline { background: #6b7280; }
    
    .agents-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 1rem;
      padding: 1rem;
    }
    
    .agents-grid.compact {
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 0.75rem;
      padding: 0.75rem;
    }
    
    .agent-card {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      transition: all 0.2s;
      cursor: pointer;
      position: relative;
      overflow: hidden;
    }
    
    .agent-card:hover {
      border-color: #d1d5db;
      box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
      transform: translateY(-1px);
    }
    
    .agent-card.selected {
      border-color: #3b82f6;
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
    }
    
    .agent-card.compact {
      padding: 0.75rem;
    }
    
    .agent-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }
    
    .agent-info {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex: 1;
      min-width: 0;
    }
    
    .agent-status-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 2px solid white;
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.1);
      flex-shrink: 0;
    }
    
    .agent-status-indicator.active { background: #10b981; }
    .agent-status-indicator.idle { background: #f59e0b; }
    .agent-status-indicator.error { background: #ef4444; animation: pulse 2s infinite; }
    .agent-status-indicator.offline { background: #6b7280; }
    
    .agent-name {
      font-weight: 600;
      font-size: 0.875rem;
      color: #111827;
      truncate: ellipsis;
      overflow: hidden;
      white-space: nowrap;
    }
    
    .performance-score {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.75rem;
      font-weight: 500;
    }
    
    .performance-score.up { color: #059669; }
    .performance-score.down { color: #dc2626; }
    .performance-score.stable { color: #6b7280; }
    
    .agent-task {
      font-size: 0.75rem;
      color: #6b7280;
      margin-bottom: 0.75rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    
    .agent-metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    
    .agent-metrics.compact {
      gap: 0.5rem;
    }
    
    .metric-item {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    
    .metric-label {
      font-size: 0.6875rem;
      font-weight: 500;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .agent-footer {
      margin-top: 0.75rem;
      padding-top: 0.75rem;
      border-top: 1px solid #f3f4f6;
      display: flex;
      justify-content: between;
      align-items: center;
      font-size: 0.75rem;
      color: #9ca3af;
    }
    
    .uptime-badge {
      background: #f3f4f6;
      color: #374151;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-weight: 500;
    }
    
    .empty-state {
      padding: 3rem 1rem;
      text-align: center;
      color: #9ca3af;
    }
    
    .empty-icon {
      width: 48px;
      height: 48px;
      margin: 0 auto 1rem;
      opacity: 0.5;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    @media (max-width: 768px) {
      .agents-grid {
        grid-template-columns: 1fr;
        padding: 0.75rem;
        gap: 0.75rem;
      }
      
      .health-panel-header {
        padding: 0.75rem;
      }
      
      .panel-filters {
        gap: 0.25rem;
      }
      
      .status-summary {
        gap: 0.75rem;
        font-size: 0.8125rem;
      }
    }
  `
  
  private get filteredAgents() {
    let filtered = this.agents
    
    // Filter by status
    if (this.filterStatus !== 'all') {
      filtered = filtered.filter(agent => agent.status === this.filterStatus)
    }
    
    // Sort agents
    filtered.sort((a, b) => {
      switch (this.sortBy) {
        case 'name':
          return a.name.localeCompare(b.name)
        case 'status':
          return a.status.localeCompare(b.status)
        case 'performance':
          return b.performance.score - a.performance.score
        case 'uptime':
          return b.uptime - a.uptime
        default:
          return 0
      }
    })
    
    return filtered
  }
  
  private get statusCounts() {
    return this.agents.reduce((counts, agent) => {
      counts[agent.status] = (counts[agent.status] || 0) + 1
      return counts
    }, {} as Record<string, number>)
  }
  
  private formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    
    if (hours > 24) {
      const days = Math.floor(hours / 24)
      return `${days}d ${hours % 24}h`
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`
    } else {
      return `${minutes}m`
    }
  }
  
  private formatPerformanceScore(score: number): string {
    return `${Math.round(score)}%`
  }
  
  private getTrendIcon(trend: 'up' | 'down' | 'stable') {
    switch (trend) {
      case 'up':
        return html`<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 17l9.2-9.2M17 17V7H7"/>
        </svg>`
      case 'down':
        return html`<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 7l-9.2 9.2M7 7v10h10"/>
        </svg>`
      case 'stable':
      default:
        return html`<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
        </svg>`
    }
  }
  
  private handleAgentClick(agent: AgentStatus) {
    this.selectedAgent = this.selectedAgent === agent.id ? null : agent.id
    
    const clickEvent = new CustomEvent('agent-selected', {
      detail: { agent, selected: this.selectedAgent === agent.id },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(clickEvent)
  }
  
  private async handleRefresh() {
    this.isRefreshing = true
    
    const refreshEvent = new CustomEvent('refresh-agents', {
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(refreshEvent)
    
    // Auto-stop spinning after 2 seconds
    setTimeout(() => {
      this.isRefreshing = false
    }, 2000)
  }
  
  private handleSortChange(e: Event) {
    this.sortBy = (e.target as HTMLSelectElement).value as any
  }
  
  private handleFilterChange(e: Event) {
    this.filterStatus = (e.target as HTMLSelectElement).value
  }
  
  render() {
    const counts = this.statusCounts
    const filtered = this.filteredAgents
    
    return html`
      <div class="health-panel-header">
        <div class="header-content">
          <h3 class="panel-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Agent Health
          </h3>
          
          <button
            class="refresh-button ${this.isRefreshing ? 'spinning' : ''}"
            @click=${this.handleRefresh}
            ?disabled=${this.isRefreshing}
            title="Refresh agent data"
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
        
        <div class="panel-filters">
          <select class="filter-select" .value=${this.sortBy} @change=${this.handleSortChange}>
            <option value="name">Sort by Name</option>
            <option value="status">Sort by Status</option>
            <option value="performance">Sort by Performance</option>
            <option value="uptime">Sort by Uptime</option>
          </select>
          
          <select class="filter-select" .value=${this.filterStatus} @change=${this.handleFilterChange}>
            <option value="all">All Agents</option>
            <option value="active">Active Only</option>
            <option value="idle">Idle Only</option>
            <option value="error">Errors Only</option>
            <option value="offline">Offline Only</option>
          </select>
        </div>
        
        <div class="status-summary">
          <div class="status-count">
            <div class="status-dot active"></div>
            Active: ${counts.active || 0}
          </div>
          <div class="status-count">
            <div class="status-dot idle"></div>
            Idle: ${counts.idle || 0}
          </div>
          <div class="status-count">
            <div class="status-dot error"></div>
            Errors: ${counts.error || 0}
          </div>
          <div class="status-count">
            <div class="status-dot offline"></div>
            Offline: ${counts.offline || 0}
          </div>
        </div>
      </div>
      
      ${filtered.length === 0 ? html`
        <div class="empty-state">
          <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p>No agents found matching the current filters.</p>
        </div>
      ` : html`
        <div class="agents-grid ${this.compact ? 'compact' : ''}">
          ${repeat(
            filtered,
            agent => agent.id,
            agent => html`
              <div 
                class="agent-card ${this.compact ? 'compact' : ''} ${this.selectedAgent === agent.id ? 'selected' : ''}"
                @click=${() => this.handleAgentClick(agent)}
              >
                <div class="agent-header">
                  <div class="agent-info">
                    <div class="agent-status-indicator ${agent.status}"></div>
                    <div class="agent-name" title="${agent.name}">${agent.name}</div>
                  </div>
                  <div class="performance-score ${agent.performance.trend}">
                    ${this.getTrendIcon(agent.performance.trend)}
                    ${this.formatPerformanceScore(agent.performance.score)}
                  </div>
                </div>
                
                ${agent.currentTask ? html`
                  <div class="agent-task" title="${agent.currentTask}">
                    ${agent.currentTask}
                  </div>
                ` : ''}
                
                <div class="agent-metrics ${this.compact ? 'compact' : ''}">
                  <div class="metric-item">
                    <div class="metric-label">CPU Usage</div>
                    <sparkline-chart
                      .data=${agent.metrics.cpuUsage.map((value, index) => ({ 
                        value, 
                        timestamp: agent.metrics.timestamps[index] 
                      }))}
                      width="80"
                      height="24"
                      color="#f59e0b"
                      fillColor="rgba(245, 158, 11, 0.1)"
                    ></sparkline-chart>
                  </div>
                  
                  <div class="metric-item">
                    <div class="metric-label">Memory</div>
                    <sparkline-chart
                      .data=${agent.metrics.memoryUsage.map((value, index) => ({ 
                        value, 
                        timestamp: agent.metrics.timestamps[index] 
                      }))}
                      width="80"
                      height="24"
                      color="#3b82f6"
                      fillColor="rgba(59, 130, 246, 0.1)"
                    ></sparkline-chart>
                  </div>
                  
                  <div class="metric-item">
                    <div class="metric-label">Tokens/min</div>
                    <sparkline-chart
                      .data=${agent.metrics.tokenUsage.map((value, index) => ({ 
                        value, 
                        timestamp: agent.metrics.timestamps[index] 
                      }))}
                      width="80"
                      height="24"
                      color="#10b981"
                      fillColor="rgba(16, 185, 129, 0.1)"
                    ></sparkline-chart>
                  </div>
                  
                  <div class="metric-item">
                    <div class="metric-label">Response Time</div>
                    <sparkline-chart
                      .data=${agent.metrics.responseTime.map((value, index) => ({ 
                        value, 
                        timestamp: agent.metrics.timestamps[index] 
                      }))}
                      width="80"
                      height="24"
                      color="#8b5cf6"
                      fillColor="rgba(139, 92, 246, 0.1)"
                    ></sparkline-chart>
                  </div>
                </div>
                
                <div class="agent-footer">
                  <span class="uptime-badge">
                    Uptime: ${this.formatUptime(agent.uptime)}
                  </span>
                  <span>
                    Last seen: ${new Date(agent.lastSeen).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            `
          )}
        </div>
      `}
    `
  }
}