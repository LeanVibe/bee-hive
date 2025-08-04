import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../services/websocket'
import { backendAdapter } from '../services/backend-adapter'

interface PriorityAlert {
  id: string
  priority: 'critical' | 'high' | 'medium' | 'info'
  type: string
  title: string
  message: string
  action?: string
  command?: string
  timestamp: string
  estimatedTime?: string
}

interface QuickAction {
  id: string
  title: string
  command: string
  priority: 'critical' | 'high' | 'medium'
  icon: string
  estimatedTime: string
  description: string
}

interface SystemStatus {
  overall: 'operational' | 'degraded' | 'critical'
  agentCount: number
  activeAgents: number
  criticalIssues: number
  requiresAttention: boolean
}

@customElement('mobile-enhanced-dashboard-view')
export class MobileEnhancedDashboardView extends LitElement {
  @property({ type: Boolean }) declare mobile: boolean
  @property({ type: Boolean }) declare decisionMode: boolean

  @state() private declare alerts: PriorityAlert[]
  @state() private declare quickActions: QuickAction[]
  @state() private declare systemStatus: SystemStatus
  @state() private declare isLoading: boolean
  @state() private declare lastUpdate: Date | null
  @state() private declare selectedPriority: 'all' | 'critical' | 'high' | 'medium'
  @state() private declare isRefreshing: boolean

  private websocketService: WebSocketService
  private updateInterval: number | null = null

  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Mobile-First Design */
    .mobile-dashboard {
      padding: 1rem;
      height: 100%;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }

    .dashboard-header {
      margin-bottom: 1.5rem;
    }

    .status-indicator {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: white;
      border-radius: 12px;
      padding: 1rem;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      margin-bottom: 1rem;
    }

    .status-main {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .status-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }

    .status-dot.degraded {
      background: #f59e0b;
    }

    .status-dot.critical {
      background: #ef4444;
    }

    .status-text {
      font-weight: 600;
      color: #111827;
      font-size: 1rem;
    }

    .status-subtitle {
      font-size: 0.875rem;
      color: #6b7280;
      margin-top: 0.25rem;
    }

    .refresh-button {
      background: none;
      border: none;
      padding: 0.5rem;
      border-radius: 8px;
      cursor: pointer;
      color: #6b7280;
      transition: all 0.2s;
    }

    .refresh-button:hover {
      background: #f3f4f6;
      color: #374151;
    }

    .refresh-button.spinning {
      animation: spin 1s linear infinite;
    }

    /* Priority Filter Tabs */
    .priority-tabs {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      padding-bottom: 0.5rem;
    }

    .priority-tab {
      background: white;
      border: 1px solid #e5e7eb;
      color: #6b7280;
      padding: 0.5rem 1rem;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
      font-size: 0.875rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.375rem;
    }

    .priority-tab:hover {
      background: #f9fafb;
      border-color: #d1d5db;
    }

    .priority-tab.active {
      background: #3b82f6;
      border-color: #3b82f6;
      color: white;
    }

    .priority-badge {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #dc2626;
    }

    .priority-badge.high {
      background: #f59e0b;
    }

    .priority-badge.medium {
      background: #10b981;
    }

    .priority-badge.info {
      background: #6b7280;
    }

    /* Critical Alerts Section */
    .alerts-section {
      margin-bottom: 2rem;
    }

    .section-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: #111827;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .alert-card {
      background: white;
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: 0.75rem;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      border-left: 4px solid #ef4444;
      transition: all 0.2s;
    }

    .alert-card.high {
      border-left-color: #f59e0b;
    }

    .alert-card.medium {
      border-left-color: #10b981;
    }

    .alert-card.info {
      border-left-color: #6b7280;
    }

    .alert-card:active {
      transform: translateY(1px);
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
    }

    .alert-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .alert-title {
      font-weight: 600;
      color: #111827;
      font-size: 0.875rem;
      line-height: 1.4;
    }

    .alert-priority {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      background: #fee2e2;
      color: #dc2626;
    }

    .alert-priority.high {
      background: #fef3c7;
      color: #d97706;
    }

    .alert-priority.medium {
      background: #d1fae5;
      color: #059669;
    }

    .alert-priority.info {
      background: #f3f4f6;
      color: #6b7280;
    }

    .alert-message {
      font-size: 0.875rem;
      color: #4b5563;
      line-height: 1.4;
      margin-bottom: 0.75rem;
    }

    .alert-action {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-top: 0.75rem;
      border-top: 1px solid #f3f4f6;
    }

    .action-text {
      font-size: 0.8125rem;
      color: #3b82f6;
      font-weight: 500;
    }

    .estimated-time {
      font-size: 0.75rem;
      color: #6b7280;
      background: #f9fafb;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
    }

    /* Quick Actions Section */
    .quick-actions-section {
      margin-bottom: 2rem;
    }

    .quick-actions-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }

    .quick-action-card {
      background: white;
      border-radius: 12px;
      padding: 1rem;
      text-align: center;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      cursor: pointer;
      transition: all 0.2s;
      border: 2px solid transparent;
    }

    .quick-action-card:active {
      transform: scale(0.98);
      border-color: #3b82f6;
    }

    .quick-action-card.critical {
      border-color: #fee2e2;
      background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
    }

    .quick-action-card.high {
      border-color: #fef3c7;
      background: linear-gradient(135deg, #ffffff 0%, #fffbeb 100%);
    }

    .quick-action-icon {
      font-size: 1.5rem;
      margin-bottom: 0.5rem;
    }

    .quick-action-title {
      font-weight: 600;
      color: #111827;
      font-size: 0.875rem;
      margin-bottom: 0.25rem;
    }

    .quick-action-time {
      font-size: 0.75rem;
      color: #6b7280;
    }

    /* Empty States */
    .empty-state {
      text-align: center;
      padding: 2rem 1rem;
      color: #6b7280;
    }

    .empty-state-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
      opacity: 0.5;
    }

    .empty-state-text {
      font-size: 1rem;
      font-weight: 500;
      margin-bottom: 0.5rem;
    }

    .empty-state-subtitle {
      font-size: 0.875rem;
      opacity: 0.8;
    }

    /* Loading States */
    .loading-skeleton {
      background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%);
      background-size: 200% 100%;
      animation: loading 1.5s infinite;
      border-radius: 8px;
      height: 1rem;
      margin: 0.5rem 0;
    }

    /* Animations */
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes loading {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }

    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
      * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :host {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      }
      
      .status-indicator,
      .alert-card,
      .quick-action-card {
        background: #1e293b;
        color: #f1f5f9;
        border-color: #334155;
      }
      
      .section-title,
      .alert-title,
      .quick-action-title {
        color: #f1f5f9;
      }
      
      .alert-message,
      .status-subtitle {
        color: #94a3b8;
      }
    }

    /* High contrast mode support */
    @media (prefers-contrast: high) {
      .alert-card,
      .quick-action-card {
        border: 2px solid currentColor;
      }
    }
  `

  constructor() {
    super()
    this.mobile = true
    this.decisionMode = true
    this.alerts = []
    this.quickActions = []
    this.systemStatus = {
      overall: 'operational',
      agentCount: 0,
      activeAgents: 0,
      criticalIssues: 0,
      requiresAttention: false
    }
    this.isLoading = true
    this.lastUpdate = null
    this.selectedPriority = 'all'
    this.isRefreshing = false

    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.initializeMobileDashboard()
    this.startRealTimeUpdates()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
    }
  }

  private async initializeMobileDashboard() {
    this.isLoading = true
    try {
      await this.loadDashboardData()
      this.isLoading = false
    } catch (error) {
      console.error('Failed to initialize mobile dashboard:', error)
      this.isLoading = false
    }
  }

  private async loadDashboardData() {
    try {
      // Simulate API call to enhanced hive commands
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:status --mobile --alerts-only --priority=high'
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.success && data.result.mobile_optimized) {
          this.updateFromHiveData(data.result)
        } else {
          // Fallback to mock data
          this.generateMockMobileData()
        }
      } else {
        this.generateMockMobileData()
      }

      this.lastUpdate = new Date()
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
      this.generateMockMobileData()
    }
  }

  private updateFromHiveData(hiveData: any) {
    // Update system status
    this.systemStatus = {
      overall: hiveData.system_state || 'operational',
      agentCount: hiveData.agent_count || 0,
      activeAgents: hiveData.agent_count || 0,
      criticalIssues: hiveData.critical_alerts?.length || 0,
      requiresAttention: hiveData.requires_attention || false
    }

    // Transform alerts
    this.alerts = (hiveData.critical_alerts || []).map((alert: any) => ({
      id: alert.id || Math.random().toString(36),
      priority: alert.priority,
      type: alert.type || 'system',
      title: alert.title || alert.message,
      message: alert.message,
      action: alert.action,
      command: alert.command,
      timestamp: alert.timestamp || new Date().toISOString(),
      estimatedTime: alert.estimated_time || '< 5 min'
    }))

    // Transform quick actions
    this.quickActions = (hiveData.quick_actions || []).map((action: any) => ({
      id: action.action || Math.random().toString(36),
      title: action.description || action.title,
      command: action.command,
      priority: action.priority || 'medium',
      icon: this.getActionIcon(action.action),
      estimatedTime: action.time || '< 5 min',
      description: action.description || action.title
    }))
  }

  private generateMockMobileData() {
    // Mock system status
    this.systemStatus = {
      overall: 'operational',
      agentCount: 5,
      activeAgents: 4,
      criticalIssues: 1,
      requiresAttention: true
    }

    // Mock priority alerts
    this.alerts = [
      {
        id: '1',
        priority: 'critical',
        type: 'system_health',
        title: 'Agent Communication Failure',
        message: 'Backend developer agent lost connection - may impact active development tasks',
        action: 'Restart agent or check network connectivity',
        command: '/hive:spawn backend_developer',
        timestamp: new Date().toISOString(),
        estimatedTime: '2-3 min'
      },
      {
        id: '2',
        priority: 'high',
        type: 'performance',
        title: 'High Memory Usage Detected',
        message: 'System memory usage at 87% - consider scaling or optimization',
        action: 'Review running processes and scale resources',
        timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
        estimatedTime: '5-10 min'
      }
    ]

    // Mock quick actions
    this.quickActions = [
      {
        id: 'start_dev',
        title: 'Start Development',
        command: '/hive:develop',
        priority: 'high',
        icon: '🚀',
        estimatedTime: '5-30 min',
        description: 'Begin autonomous development session'
      },
      {
        id: 'check_agents',
        title: 'Check Agent Health',
        command: '/hive:status --detailed',
        priority: 'medium',
        icon: '🔍',
        estimatedTime: '< 1 min',
        description: 'Review all agent status and performance'
      },
      {
        id: 'spawn_agent',
        title: 'Add Agent',
        command: '/hive:spawn',
        priority: 'medium',
        icon: '🤖',
        estimatedTime: '1-2 min',
        description: 'Spawn additional team member'
      },
      {
        id: 'oversight',
        title: 'Full Dashboard',
        command: '/hive:oversight',
        priority: 'medium',
        icon: '📊',
        estimatedTime: '< 30 sec',
        description: 'Open complete oversight interface'
      }
    ]
  }

  private getActionIcon(action: string): string {
    const icons: Record<string, string> = {
      'start_platform': '🚀',
      'spawn_agents': '🤖',
      'start_development': '💻',
      'check_health': '🔍',
      'oversight': '📊',
      'restart_agent': '🔄'
    }
    return icons[action] || '⚡'
  }

  private startRealTimeUpdates() {
    // Update every 30 seconds for mobile optimization
    this.updateInterval = window.setInterval(() => {
      if (!this.isRefreshing) {
        this.loadDashboardData()
      }
    }, 30000)
  }

  private async handleRefresh() {
    this.isRefreshing = true
    try {
      await this.loadDashboardData()
    } finally {
      this.isRefreshing = false
    }
  }

  private handlePriorityFilter(priority: 'all' | 'critical' | 'high' | 'medium') {
    this.selectedPriority = priority
  }

  private handleQuickAction(action: QuickAction) {
    // Dispatch event for parent to handle command execution
    const event = new CustomEvent('execute-command', {
      detail: { 
        command: action.command,
        title: action.title,
        estimatedTime: action.estimatedTime
      },
      bubbles: true,
      composed: true
    })
    this.dispatchEvent(event)
  }

  private handleAlertAction(alert: PriorityAlert) {
    if (alert.command) {
      const event = new CustomEvent('execute-command', {
        detail: { 
          command: alert.command,
          title: alert.title,
          estimatedTime: alert.estimatedTime
        },
        bubbles: true,
        composed: true
      })
      this.dispatchEvent(event)
    }
  }

  private get filteredAlerts() {
    if (this.selectedPriority === 'all') {
      return this.alerts
    }
    return this.alerts.filter(alert => alert.priority === this.selectedPriority)
  }

  private get priorityCountBadges() {
    const counts = {
      critical: this.alerts.filter(a => a.priority === 'critical').length,
      high: this.alerts.filter(a => a.priority === 'high').length,
      medium: this.alerts.filter(a => a.priority === 'medium').length,
      info: this.alerts.filter(a => a.priority === 'info').length
    }
    return counts
  }

  private renderPriorityTabs() {
    const badges = this.priorityCountBadges
    
    return html`
      <div class="priority-tabs">
        <button 
          class="priority-tab ${this.selectedPriority === 'all' ? 'active' : ''}"
          @click=${() => this.handlePriorityFilter('all')}
        >
          All Alerts
          ${this.alerts.length > 0 ? html`<span class="priority-badge">${this.alerts.length}</span>` : ''}
        </button>
        
        ${badges.critical > 0 ? html`
          <button 
            class="priority-tab ${this.selectedPriority === 'critical' ? 'active' : ''}"
            @click=${() => this.handlePriorityFilter('critical')}
          >
            <span class="priority-badge"></span>
            Critical (${badges.critical})
          </button>
        ` : ''}
        
        ${badges.high > 0 ? html`
          <button 
            class="priority-tab ${this.selectedPriority === 'high' ? 'active' : ''}"
            @click=${() => this.handlePriorityFilter('high')}
          >
            <span class="priority-badge high"></span>
            High (${badges.high})
          </button>
        ` : ''}
        
        ${badges.medium > 0 ? html`
          <button 
            class="priority-tab ${this.selectedPriority === 'medium' ? 'active' : ''}"
            @click=${() => this.handlePriorityFilter('medium')}
          >
            <span class="priority-badge medium"></span>
            Medium (${badges.medium})
          </button>
        ` : ''}
      </div>
    `
  }

  private renderAlerts() {
    const alerts = this.filteredAlerts

    if (alerts.length === 0) {
      return html`
        <div class="empty-state">
          <div class="empty-state-icon">✅</div>
          <div class="empty-state-text">All Clear</div>
          <div class="empty-state-subtitle">No alerts requiring immediate attention</div>
        </div>
      `
    }

    return html`
      ${alerts.map(alert => html`
        <div 
          class="alert-card ${alert.priority}"
          @click=${() => this.handleAlertAction(alert)}
        >
          <div class="alert-header">
            <div class="alert-title">${alert.title}</div>
            <div class="alert-priority ${alert.priority}">${alert.priority}</div>
          </div>
          
          <div class="alert-message">${alert.message}</div>
          
          ${alert.action ? html`
            <div class="alert-action">
              <div class="action-text">${alert.action}</div>
              ${alert.estimatedTime ? html`
                <div class="estimated-time">${alert.estimatedTime}</div>
              ` : ''}
            </div>
          ` : ''}
        </div>
      `)}
    `
  }

  private renderQuickActions() {
    return html`
      <div class="quick-actions-grid">
        ${this.quickActions.map(action => html`
          <div 
            class="quick-action-card ${action.priority}"
            @click=${() => this.handleQuickAction(action)}
          >
            <div class="quick-action-icon">${action.icon}</div>
            <div class="quick-action-title">${action.title}</div>
            <div class="quick-action-time">${action.estimatedTime}</div>
          </div>
        `)}
      </div>
    `
  }

  private renderStatusIndicator() {
    return html`
      <div class="status-indicator">
        <div class="status-main">
          <div class="status-dot ${this.systemStatus.overall}"></div>
          <div>
            <div class="status-text">
              ${this.systemStatus.overall === 'operational' ? 'System Operational' :
                this.systemStatus.overall === 'degraded' ? 'System Degraded' : 'Critical Issues'}
            </div>
            <div class="status-subtitle">
              ${this.systemStatus.activeAgents}/${this.systemStatus.agentCount} agents active
              ${this.systemStatus.criticalIssues > 0 ? ` • ${this.systemStatus.criticalIssues} critical issues` : ''}
            </div>
          </div>
        </div>
        
        <button 
          class="refresh-button ${this.isRefreshing ? 'spinning' : ''}"
          @click=${this.handleRefresh}
          ?disabled=${this.isRefreshing}
        >
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>
    `
  }

  render() {
    if (this.isLoading) {
      return html`
        <div class="mobile-dashboard">
          <div class="loading-skeleton" style="height: 80px; margin-bottom: 1rem;"></div>
          <div class="loading-skeleton" style="height: 60px; margin-bottom: 1rem;"></div>
          <div class="loading-skeleton" style="height: 120px; margin-bottom: 1rem;"></div>
          <div class="loading-skeleton" style="height: 100px;"></div>
        </div>
      `
    }

    return html`
      <div class="mobile-dashboard">
        <div class="dashboard-header">
          ${this.renderStatusIndicator()}
          ${this.renderPriorityTabs()}
        </div>

        <div class="alerts-section">
          <h2 class="section-title">
            🚨 Priority Alerts
            ${this.systemStatus.requiresAttention ? '(Attention Required)' : ''}
          </h2>
          ${this.renderAlerts()}
        </div>

        <div class="quick-actions-section">
          <h2 class="section-title">⚡ Quick Actions</h2>
          ${this.renderQuickActions()}
        </div>
      </div>
    `
  }
}