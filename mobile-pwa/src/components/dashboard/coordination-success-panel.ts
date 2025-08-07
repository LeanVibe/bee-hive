import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { backendAdapter } from '../../services/backend-adapter'

interface CoordinationSuccessData {
  current_rate: number
  trend: string
  last_hour_rate: number
  last_24h_rate: number
  total_attempts: number
  alert_status: string
  timestamp: string
}

interface FailureAnalysis {
  serialization_errors: number
  workflow_state_errors: number
  agent_timeout_errors: number
  communication_errors: number
  task_assignment_errors: number
  unknown_errors: number
  total_failures: number
}

@customElement('coordination-success-panel')
export class CoordinationSuccessPanel extends LitElement {
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: Boolean }) declare compact: boolean
  @property({ type: String }) declare timeRange: string

  @state() private declare successData: CoordinationSuccessData | null
  @state() private declare failureData: FailureAnalysis | null
  @state() private declare historicalData: { timestamp: string; rate: number }[]
  @state() private declare isLoading: boolean
  @state() private declare error: string
  @state() private declare lastUpdate: Date | null

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
      background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
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

    .success-rate-display {
      display: flex;
      align-items: center;
      gap: 1rem;
      background: rgba(255, 255, 255, 0.1);
      padding: 0.75rem;
      border-radius: 8px;
    }

    .rate-value {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1;
    }

    .rate-label {
      font-size: 0.875rem;
      opacity: 0.9;
    }

    .trend-indicator {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.875rem;
      font-weight: 500;
    }

    .trend-improving {
      color: #10b981;
    }

    .trend-declining {
      color: #ef4444;
    }

    .trend-stable {
      color: #6b7280;
    }

    .alert-badge {
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .alert-healthy {
      background: #dcfce7;
      color: #166534;
    }

    .alert-warning {
      background: #fef3c7;
      color: #92400e;
    }

    .alert-critical {
      background: #fee2e2;
      color: #991b1b;
      animation: pulse-critical 2s infinite;
    }

    @keyframes pulse-critical {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }

    .panel-content {
      padding: 1rem;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .metric-card {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1rem;
      text-align: center;
    }

    .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.25rem;
    }

    .metric-label {
      font-size: 0.875rem;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .failure-analysis {
      margin-top: 1.5rem;
    }

    .section-title {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid #e5e7eb;
    }

    .failure-types {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.75rem;
    }

    .failure-type {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.75rem;
      background: #f9fafb;
      border-radius: 6px;
      border-left: 4px solid #ef4444;
    }

    .failure-type.zero-failures {
      border-left-color: #10b981;
      background: #f0fdf4;
    }

    .failure-name {
      font-size: 0.875rem;
      font-weight: 500;
      color: #374151;
    }

    .failure-count {
      font-size: 1rem;
      font-weight: 700;
      color: #ef4444;
    }

    .failure-count.zero {
      color: #10b981;
    }

    .trend-chart {
      height: 120px;
      margin: 1rem 0;
      background: #f9fafb;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #6b7280;
      font-size: 0.875rem;
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
    }

    .last-update {
      font-size: 0.75rem;
      color: #6b7280;
      text-align: right;
      margin-top: 1rem;
    }

    .realtime-indicator {
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.75rem;
      opacity: 0.9;
    }

    .live-dot {
      width: 6px;
      height: 6px;
      background: #10b981;
      border-radius: 50%;
      animation: pulse-live 2s infinite;
    }

    @keyframes pulse-live {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    .recovery-actions {
      margin-top: 1.5rem;
      padding-top: 1rem;
      border-top: 1px solid #e5e7eb;
    }

    .action-buttons {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .action-btn {
      padding: 0.5rem 1rem;
      border: 1px solid #d1d5db;
      background: white;
      border-radius: 6px;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }

    .action-btn:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }

    .action-btn.emergency {
      background: #ef4444;
      color: white;
      border-color: #dc2626;
    }

    .action-btn.emergency:hover {
      background: #dc2626;
    }

    @media (max-width: 768px) {
      .panel-header {
        flex-direction: column;
        gap: 1rem;
        align-items: flex-start;
      }

      .success-rate-display {
        flex-direction: column;
        gap: 0.5rem;
        align-items: flex-start;
      }

      .rate-value {
        font-size: 1.5rem;
      }

      .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
      }

      .failure-types {
        grid-template-columns: 1fr;
      }

      .action-buttons {
        flex-direction: column;
      }

      .action-btn {
        width: 100%;
      }
    }
  `

  constructor() {
    super()
    this.realtime = true
    this.compact = false
    this.timeRange = '1h'
    this.successData = null
    this.failureData = null
    this.historicalData = []
    this.isLoading = true
    this.error = ''
    this.lastUpdate = null
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
      // Load coordination dashboard data from new API endpoint
      const response = await fetch('/api/v1/coordination-monitoring/dashboard')
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const dashboardData = await response.json()
      
      this.successData = dashboardData.success_rate
      this.failureData = dashboardData.failure_analysis
      this.lastUpdate = new Date()

      // Load historical data for trend visualization
      await this.loadHistoricalData()

    } catch (error) {
      console.error('Failed to load coordination success data:', error)
      this.error = error instanceof Error ? error.message : 'Unknown error occurred'
    } finally {
      this.isLoading = false
    }
  }

  private async loadHistoricalData() {
    try {
      // Generate mock historical data for trend visualization
      // In production, this would come from the backend
      const now = Date.now()
      const points = []
      
      for (let i = 23; i >= 0; i--) {
        const timestamp = new Date(now - i * 60 * 60 * 1000).toISOString()
        const rate = Math.max(0, Math.min(100, 
          (this.successData?.current_rate || 20) + 
          (Math.random() - 0.5) * 20
        ))
        points.push({ timestamp, rate })
      }
      
      this.historicalData = points

    } catch (error) {
      console.warn('Failed to load historical data:', error)
    }
  }

  private startRealtimeUpdates() {
    // Update every 5 seconds for real-time monitoring
    this.updateInterval = window.setInterval(() => {
      this.loadData()
    }, 5000)
  }

  private stopRealtimeUpdates() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
      this.updateInterval = null
    }
  }

  private async executeRecoveryAction(action: string) {
    try {
      let endpoint = ''
      let method = 'POST'

      switch (action) {
        case 'reset-coordination':
          endpoint = '/api/v1/coordination-monitoring/recovery-actions/reset-coordination'
          break
        case 'generate-test-data':
          endpoint = '/api/v1/coordination-monitoring/test/generate-coordination-data'
          break
        default:
          throw new Error(`Unknown action: ${action}`)
      }

      const response = await fetch(endpoint, { method })
      
      if (!response.ok) {
        throw new Error(`Recovery action failed: ${response.statusText}`)
      }

      const result = await response.json()
      console.log('Recovery action completed:', result)

      // Refresh data after action
      setTimeout(() => this.loadData(), 1000)

      // Dispatch event for parent components
      this.dispatchEvent(new CustomEvent('recovery-action', {
        detail: { action, result },
        bubbles: true,
        composed: true
      }))

    } catch (error) {
      console.error('Recovery action failed:', error)
      this.error = `Recovery action failed: ${error}`
    }
  }

  private getTrendIcon(trend: string) {
    switch (trend) {
      case 'improving':
        return '↗️'
      case 'declining':
        return '↘️'
      case 'stable':
        return '→'
      default:
        return '❓'
    }
  }

  private getAlertStatusColor(status: string) {
    switch (status) {
      case 'healthy':
        return '#10b981'
      case 'warning':
        return '#f59e0b'
      case 'critical':
        return '#ef4444'
      default:
        return '#6b7280'
    }
  }

  render() {
    if (this.isLoading && !this.successData) {
      return html`
        <div class="loading-state">
          <div>Loading coordination success data...</div>
        </div>
      `
    }

    if (this.error && !this.successData) {
      return html`
        <div class="error-state">
          <p><strong>Error:</strong> ${this.error}</p>
          <button class="action-btn" @click=${() => this.loadData()}>
            Retry
          </button>
        </div>
      `
    }

    const successData = this.successData!
    const failureData = this.failureData

    return html`
      <div class="panel-header">
        <div class="panel-title">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Coordination Success Rate
          ${this.realtime ? html`
            <span class="realtime-indicator">
              <div class="live-dot"></div>
              LIVE
            </span>
          ` : ''}
        </div>

        <div class="success-rate-display">
          <div>
            <div class="rate-value" style="color: ${this.getAlertStatusColor(successData.alert_status)}">
              ${successData.current_rate.toFixed(1)}%
            </div>
            <div class="rate-label">Current Success Rate</div>
          </div>

          <div class="trend-indicator trend-${successData.trend}">
            <span>${this.getTrendIcon(successData.trend)}</span>
            ${successData.trend}
          </div>

          <div class="alert-badge alert-${successData.alert_status}">
            ${successData.alert_status}
          </div>
        </div>
      </div>

      <div class="panel-content">
        ${this.error ? html`
          <div class="error-state" style="margin-bottom: 1rem;">
            <p>${this.error}</p>
          </div>
        ` : ''}

        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-value">${successData.last_hour_rate.toFixed(1)}%</div>
            <div class="metric-label">Last Hour</div>
          </div>

          <div class="metric-card">
            <div class="metric-value">${successData.last_24h_rate.toFixed(1)}%</div>
            <div class="metric-label">Last 24 Hours</div>
          </div>

          <div class="metric-card">
            <div class="metric-value">${successData.total_attempts}</div>
            <div class="metric-label">Total Attempts</div>
          </div>
        </div>

        <div class="trend-chart">
          ${this.historicalData.length > 0 ? 
            html`Success rate trend: ${this.historicalData[this.historicalData.length - 1].rate.toFixed(1)}% (24h)` :
            html`Trend visualization loading...`
          }
        </div>

        ${failureData ? html`
          <div class="failure-analysis">
            <div class="section-title">
              Failure Analysis (${failureData.total_failures} total)
            </div>

            <div class="failure-types">
              <div class="failure-type ${failureData.serialization_errors === 0 ? 'zero-failures' : ''}">
                <span class="failure-name">Serialization Errors</span>
                <span class="failure-count ${failureData.serialization_errors === 0 ? 'zero' : ''}">${failureData.serialization_errors}</span>
              </div>

              <div class="failure-type ${failureData.workflow_state_errors === 0 ? 'zero-failures' : ''}">
                <span class="failure-name">Workflow State Errors</span>
                <span class="failure-count ${failureData.workflow_state_errors === 0 ? 'zero' : ''}">${failureData.workflow_state_errors}</span>
              </div>

              <div class="failure-type ${failureData.agent_timeout_errors === 0 ? 'zero-failures' : ''}">
                <span class="failure-name">Agent Timeouts</span>
                <span class="failure-count ${failureData.agent_timeout_errors === 0 ? 'zero' : ''}">${failureData.agent_timeout_errors}</span>
              </div>

              <div class="failure-type ${failureData.communication_errors === 0 ? 'zero-failures' : ''}">
                <span class="failure-name">Communication Errors</span>
                <span class="failure-count ${failureData.communication_errors === 0 ? 'zero' : ''}">${failureData.communication_errors}</span>
              </div>

              <div class="failure-type ${failureData.task_assignment_errors === 0 ? 'zero-failures' : ''}">
                <span class="failure-name">Task Assignment Errors</span>
                <span class="failure-count ${failureData.task_assignment_errors === 0 ? 'zero' : ''}">${failureData.task_assignment_errors}</span>
              </div>

              <div class="failure-type ${failureData.unknown_errors === 0 ? 'zero-failures' : ''}">
                <span class="failure-name">Unknown Errors</span>
                <span class="failure-count ${failureData.unknown_errors === 0 ? 'zero' : ''}">${failureData.unknown_errors}</span>
              </div>
            </div>
          </div>
        ` : ''}

        ${successData.alert_status === 'critical' ? html`
          <div class="recovery-actions">
            <div class="section-title">Emergency Recovery Actions</div>
            <div class="action-buttons">
              <button class="action-btn emergency" @click=${() => this.executeRecoveryAction('reset-coordination')}>
                🚨 Reset Coordination System
              </button>
              <button class="action-btn" @click=${() => this.executeRecoveryAction('generate-test-data')}>
                🧪 Generate Test Data
              </button>
            </div>
          </div>
        ` : ''}

        ${this.lastUpdate ? html`
          <div class="last-update">
            Last updated: ${this.lastUpdate.toLocaleTimeString()}
          </div>
        ` : ''}
      </div>
    `
  }
}