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
  @property({ type: Array }) declare agents: AgentStatus[]
  @property({ type: Boolean }) declare compact: boolean
  @property({ type: String }) declare sortBy: 'name' | 'status' | 'performance' | 'uptime'
  @property({ type: String }) declare filterStatus: string
  
  @state() private declare selectedAgent: string | null
  @state() private declare isRefreshing: boolean
  @state() private declare teamActivationInProgress: boolean
  @state() private declare showTeamControls: boolean
  @state() private declare bulkSelectedAgents: Set<string>
  @state() private declare showBulkActions: boolean
  @state() private declare agentActionsInProgress: Map<string, string>
  @state() private declare showAdvancedControls: boolean
  
  constructor() {
    super()
    
    // Initialize reactive properties
    this.agents = []
    this.compact = false
    this.sortBy = 'name'
    this.filterStatus = 'all'
    this.selectedAgent = null
    this.isRefreshing = false
    this.teamActivationInProgress = false
    this.showTeamControls = true
    this.bulkSelectedAgents = new Set<string>()
    this.showBulkActions = false
    this.agentActionsInProgress = new Map<string, string>()
    this.showAdvancedControls = true
  }
  
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
    
    /* Enhanced Agent Management Styles */
    .team-controls {
      background: #f8fafc;
      border-bottom: 1px solid #e5e7eb;
      padding: 1rem;
      display: flex;
      gap: 0.75rem;
      align-items: center;
      flex-wrap: wrap;
    }

    .team-activation-button {
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 0.5rem;
      font-weight: 600;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }

    .team-activation-button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }

    .team-activation-button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .team-status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
      color: #374151;
      font-weight: 500;
    }

    .system-ready-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
    }

    .system-ready-indicator.not-ready {
      background: #f59e0b;
      animation: pulse 2s infinite;
    }

    .bulk-actions {
      background: #fef3c7;
      border: 1px solid #f59e0b;
      border-radius: 0.5rem;
      padding: 0.75rem;
      margin: 0.5rem 1rem;
      display: flex;
      gap: 0.5rem;
      align-items: center;
      flex-wrap: wrap;
    }

    .bulk-action-button {
      background: #f59e0b;
      color: white;
      border: none;
      padding: 0.375rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s;
    }

    .bulk-action-button:hover {
      background: #d97706;
    }

    .bulk-action-button.danger {
      background: #dc2626;
    }

    .bulk-action-button.danger:hover {
      background: #b91c1c;
    }

    .agent-controls {
      display: flex;
      gap: 0.375rem;
      margin-top: 0.75rem;
      flex-wrap: wrap;
    }

    .agent-control-button {
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      border: 1px solid #cbd5e1;
      color: #475569;
      padding: 0.5rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.375rem;
      min-height: 32px;
      position: relative;
      overflow: hidden;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .agent-control-button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
      border-color: #94a3b8;
    }

    .agent-control-button:active:not(:disabled) {
      transform: translateY(0);
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }

    .agent-control-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }

    .agent-control-button.primary {
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      border-color: #2563eb;
      color: white;
      box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }

    .agent-control-button.primary:hover:not(:disabled) {
      background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
      box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }

    .agent-control-button.success {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      border-color: #059669;
      color: white;
      box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }

    .agent-control-button.success:hover:not(:disabled) {
      background: linear-gradient(135deg, #059669 0%, #047857 100%);
      box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
    }

    .agent-control-button.warning {
      background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
      border-color: #d97706;
      color: white;
      box-shadow: 0 2px 4px rgba(245, 158, 11, 0.2);
    }

    .agent-control-button.warning:hover:not(:disabled) {
      background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
      box-shadow: 0 4px 8px rgba(245, 158, 11, 0.3);
    }

    .agent-control-button.danger {
      background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
      border-color: #b91c1c;
      color: white;
      box-shadow: 0 2px 4px rgba(220, 38, 38, 0.2);
    }

    .agent-control-button.danger:hover:not(:disabled) {
      background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
      box-shadow: 0 4px 8px rgba(220, 38, 38, 0.3);
    }

    .control-button-loading {
      position: relative;
    }

    .control-button-loading::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 12px;
      height: 12px;
      margin: -6px 0 0 -6px;
      border: 2px solid transparent;
      border-top: 2px solid currentColor;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    .control-button-loading .button-content {
      opacity: 0;
    }

    .agent-control-icon {
      width: 14px;
      height: 14px;
      flex-shrink: 0;
    }

    .agent-checkbox {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      width: 16px;
      height: 16px;
      cursor: pointer;
    }

    .performance-trends {
      margin-top: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid #f3f4f6;
    }

    .trend-indicator {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.6875rem;
      color: #6b7280;
    }

    .trend-arrow {
      width: 12px;
      height: 12px;
    }

    .trend-arrow.up { color: #059669; }
    .trend-arrow.down { color: #dc2626; }
    .trend-arrow.stable { color: #6b7280; }

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
  
  private renderTeamControls() {
    if (!this.showTeamControls) return ''

    const systemReady = this.agents.length >= 3 && this.agents.filter(a => a.status === 'active').length >= 2
    
    return html`
      <div class="team-controls">
        <button
          class="team-activation-button"
          @click=${this.handleTeamActivation}
          ?disabled=${this.teamActivationInProgress}
        >
          <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          ${this.teamActivationInProgress ? 'Activating Team...' : 'Activate 5-Agent Team'}
        </button>

        <div class="team-status">
          <div class="system-ready-indicator ${systemReady ? '' : 'not-ready'}"></div>
          <span>
            System ${systemReady ? 'Ready' : 'Not Ready'} 
            (${this.agents.filter(a => a.status === 'active').length}/${this.agents.length} agents active)
          </span>
        </div>
      </div>
    `
  }

  private renderBulkActions() {
    if (!this.showBulkActions || this.bulkSelectedAgents.size === 0) return ''
    
    return html`
      <div class="bulk-actions">
        <span>${this.bulkSelectedAgents.size} agents selected:</span>
        <button class="bulk-action-button" @click=${() => this.handleBulkAction('restart')}>
          Restart
        </button>
        <button class="bulk-action-button" @click=${() => this.handleBulkAction('pause')}>
          Pause
        </button>
        <button class="bulk-action-button" @click=${() => this.handleBulkAction('resume')}>
          Resume
        </button>
        <button class="bulk-action-button" @click=${() => this.handleBulkAction('configure')}>
          Configure
        </button>
        <button class="bulk-action-button danger" @click=${() => { this.bulkSelectedAgents = new Set(); this.showBulkActions = false; }}>
          Cancel
        </button>
      </div>
    `
  }

  private renderAgentControls(agent: AgentStatus) {
    const isActionInProgress = this.agentActionsInProgress.has(agent.id)
    const currentAction = this.agentActionsInProgress.get(agent.id)
    
    const getButtonClass = (baseClass: string, action: string) => {
      return `agent-control-button ${baseClass} ${
        isActionInProgress && currentAction === action ? 'control-button-loading' : ''
      }`
    }
    
    const isButtonDisabled = (action: string) => {
      return isActionInProgress && currentAction !== action
    }

    return html`
      <div class="agent-controls">
        <!-- Primary Action Button -->
        <button 
          class=${getButtonClass(agent.status === 'active' ? 'warning' : 'success', agent.status === 'active' ? 'pause' : 'resume')}
          @click=${() => this.handleAgentAction(agent.id, agent.status === 'active' ? 'pause' : 'resume')}
          ?disabled=${isButtonDisabled(agent.status === 'active' ? 'pause' : 'resume')}
          title="${agent.status === 'active' ? 'Pause Agent' : 'Resume Agent'}"
        >
          <span class="button-content">
            <svg class="agent-control-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              ${agent.status === 'active' ? html`
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              ` : html`
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M15 14h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              `}
            </svg>
            ${agent.status === 'active' ? 'Pause' : 'Resume'}
          </span>
        </button>

        <!-- Configuration Button -->
        <button 
          class=${getButtonClass('primary', 'configure')}
          @click=${() => this.handleAgentAction(agent.id, 'configure')}
          ?disabled=${isButtonDisabled('configure')}
          title="Configure Agent Settings"
        >
          <span class="button-content">
            <svg class="agent-control-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Config
          </span>
        </button>

        <!-- Restart Button -->
        <button 
          class=${getButtonClass('', 'restart')}
          @click=${() => this.handleAgentAction(agent.id, 'restart')}
          ?disabled=${isButtonDisabled('restart')}
          title="Restart Agent"
        >
          <span class="button-content">
            <svg class="agent-control-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Restart
          </span>
        </button>

        ${this.showAdvancedControls ? html`
          <!-- Logs Button -->
          <button 
            class=${getButtonClass('', 'logs')}
            @click=${() => this.handleAgentAction(agent.id, 'logs')}
            ?disabled=${isButtonDisabled('logs')}
            title="View Agent Logs"
          >
            <span class="button-content">
              <svg class="agent-control-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Logs
            </span>
          </button>

          <!-- Stop Button (Danger Action) -->
          ${agent.status !== 'offline' ? html`
            <button 
              class=${getButtonClass('danger', 'stop')}
              @click=${() => this.handleAgentAction(agent.id, 'stop')}
              ?disabled=${isButtonDisabled('stop')}
              title="Stop Agent"
            >
              <span class="button-content">
                <svg class="agent-control-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10h6v4H9z" />
                </svg>
                Stop
              </span>
            </button>
          ` : ''}
        ` : ''}
      </div>
    `
  }

  private renderPerformanceTrends(agent: AgentStatus) {
    // Calculate simple trends based on recent data
    const cpuTrend = agent.metrics.cpuUsage.length >= 2 ? 
      (agent.metrics.cpuUsage[agent.metrics.cpuUsage.length - 1] > agent.metrics.cpuUsage[agent.metrics.cpuUsage.length - 2] ? 'up' : 
       agent.metrics.cpuUsage[agent.metrics.cpuUsage.length - 1] < agent.metrics.cpuUsage[agent.metrics.cpuUsage.length - 2] ? 'down' : 'stable') : 'stable'
    
    const memoryTrend = agent.metrics.memoryUsage.length >= 2 ? 
      (agent.metrics.memoryUsage[agent.metrics.memoryUsage.length - 1] > agent.metrics.memoryUsage[agent.metrics.memoryUsage.length - 2] ? 'up' : 
       agent.metrics.memoryUsage[agent.metrics.memoryUsage.length - 1] < agent.metrics.memoryUsage[agent.metrics.memoryUsage.length - 2] ? 'down' : 'stable') : 'stable'

    return html`
      <div class="performance-trends">
        <div class="trend-indicator">
          <svg class="trend-arrow ${cpuTrend}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            ${cpuTrend === 'up' ? html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 17l9.2-9.2M17 17V7H7"/>` :
              cpuTrend === 'down' ? html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 7l-9.2 9.2M7 7v10h10"/>` :
              html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14"/>`}
          </svg>
          CPU ${cpuTrend}
        </div>
        <div class="trend-indicator">
          <svg class="trend-arrow ${memoryTrend}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            ${memoryTrend === 'up' ? html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 17l9.2-9.2M17 17V7H7"/>` :
              memoryTrend === 'down' ? html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 7l-9.2 9.2M7 7v10h10"/>` :
              html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14"/>`}
          </svg>
          Memory ${memoryTrend}
        </div>
      </div>
    `
  }

  private async handleTeamActivation() {
    if (this.teamActivationInProgress) return

    this.teamActivationInProgress = true
    
    try {
      // Dispatch custom event for team activation
      const event = new CustomEvent('activate-agent-team', {
        detail: {
          teamSize: 5,
          roles: ['product_manager', 'architect', 'backend_developer', 'frontend_developer', 'qa_engineer'],
          autoStartTasks: true
        },
        bubbles: true,
        composed: true
      })
      
      this.dispatchEvent(event)
      
      // Simulate team activation process
      await new Promise(resolve => setTimeout(resolve, 2000))
      
    } catch (error) {
      console.error('Team activation failed:', error)
    } finally {
      this.teamActivationInProgress = false
    }
  }

  private handleAgentSelect(agentId: string, event: Event) {
    const checkbox = event.target as HTMLInputElement
    const newSelected = new Set(this.bulkSelectedAgents)
    
    if (checkbox.checked) {
      newSelected.add(agentId)
    } else {
      newSelected.delete(agentId)
    }
    
    this.bulkSelectedAgents = newSelected
    this.showBulkActions = newSelected.size > 0
  }

  private handleBulkAction(action: 'restart' | 'pause' | 'resume' | 'configure') {
    const selectedIds = Array.from(this.bulkSelectedAgents)
    
    const event = new CustomEvent('bulk-agent-action', {
      detail: {
        action,
        agentIds: selectedIds
      },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(event)
    
    // Clear selection after action
    this.bulkSelectedAgents = new Set()
    this.showBulkActions = false
  }

  private async handleAgentAction(agentId: string, action: 'configure' | 'restart' | 'pause' | 'resume' | 'stop' | 'logs') {
    // Set loading state
    this.agentActionsInProgress.set(agentId, action)
    this.requestUpdate()
    
    try {
      // Dispatch action event with enhanced details
      const event = new CustomEvent('agent-action', {
        detail: {
          agentId,
          action,
          agent: this.agents.find(a => a.id === agentId),
          timestamp: new Date().toISOString()
        },
        bubbles: true,
        composed: true
      })
      
      this.dispatchEvent(event)
      
      // Simulate action processing time for better UX
      const processingTime = this.getActionProcessingTime(action)
      await new Promise(resolve => setTimeout(resolve, processingTime))
      
      // Show success feedback
      this.showActionFeedback(agentId, action, 'success')
      
    } catch (error) {
      console.error(`Failed to ${action} agent ${agentId}:`, error)
      
      // Show error feedback
      this.showActionFeedback(agentId, action, 'error')
      
      // Dispatch error event
      const errorEvent = new CustomEvent('agent-action-error', {
        detail: {
          agentId,
          action,
          error: error instanceof Error ? error.message : 'Unknown error'
        },
        bubbles: true,
        composed: true
      })
      
      this.dispatchEvent(errorEvent)
      
    } finally {
      // Clear loading state
      this.agentActionsInProgress.delete(agentId)
      this.requestUpdate()
    }
  }
  
  private getActionProcessingTime(action: string): number {
    const times = {
      'configure': 800,
      'restart': 2000,
      'pause': 500,
      'resume': 600,
      'stop': 1000,
      'logs': 300
    }
    return times[action as keyof typeof times] || 500
  }
  
  private showActionFeedback(agentId: string, action: string, type: 'success' | 'error') {
    const agent = this.agents.find(a => a.id === agentId)
    const agentName = agent?.name || `Agent ${agentId}`
    
    const messages = {
      success: {
        'configure': `${agentName} configuration updated`,
        'restart': `${agentName} restarted successfully`,
        'pause': `${agentName} paused`,
        'resume': `${agentName} resumed`,
        'stop': `${agentName} stopped`,
        'logs': `Opened logs for ${agentName}`
      },
      error: {
        'configure': `Failed to configure ${agentName}`,
        'restart': `Failed to restart ${agentName}`,
        'pause': `Failed to pause ${agentName}`,
        'resume': `Failed to resume ${agentName}`,
        'stop': `Failed to stop ${agentName}`,
        'logs': `Failed to open logs for ${agentName}`
      }
    }
    
    const message = messages[type][action as keyof typeof messages.success]
    
    // Dispatch feedback event for toast/notification system
    const feedbackEvent = new CustomEvent('agent-action-feedback', {
      detail: {
        type,
        message,
        agentId,
        action
      },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(feedbackEvent)
  }

  render() {
    const counts = this.statusCounts
    const filtered = this.filteredAgents
    
    return html`
      ${this.renderTeamControls()}
      ${this.renderBulkActions()}
      
      <div class="health-panel-header">
        <div class="header-content">
          <h3 class="panel-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Agent Health & Management
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
                <input 
                  type="checkbox" 
                  class="agent-checkbox"
                  .checked=${this.bulkSelectedAgents.has(agent.id)}
                  @click=${(e: Event) => e.stopPropagation()}
                  @change=${(e: Event) => this.handleAgentSelect(agent.id, e)}
                />
                
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
                
                ${this.renderPerformanceTrends(agent)}
                ${this.renderAgentControls(agent)}
                
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