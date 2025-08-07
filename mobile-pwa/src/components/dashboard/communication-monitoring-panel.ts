import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

interface CommunicationHealth {
  redis_status: string
  average_latency: number
  message_throughput: number
  error_rate: number
  connected_agents: number
  recent_latencies: number[]
}

interface MessageLatencyPoint {
  timestamp: number
  latency: number
  agent_id?: string
  message_type?: string
}

interface RedisMetrics {
  connected_clients: number
  memory_usage: number
  operations_per_second: number
  keyspace_hits: number
  keyspace_misses: number
  uptime: number
}

interface CommunicationError {
  id: string
  timestamp: string
  agent_id: string
  error_type: string
  message: string
  severity: 'low' | 'medium' | 'high' | 'critical'
}

@customElement('communication-monitoring-panel')
export class CommunicationMonitoringPanel extends LitElement {
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: Boolean }) declare compact: boolean
  @property({ type: String }) declare timeRange: string

  @state() private declare communicationHealth: CommunicationHealth | null
  @state() private declare redisMetrics: RedisMetrics | null
  @state() private declare latencyHistory: MessageLatencyPoint[]
  @state() private declare recentErrors: CommunicationError[]
  @state() private declare isLoading: boolean
  @state() private declare error: string
  @state() private declare lastUpdate: Date | null
  @state() private declare selectedMetric: string

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
      background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
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

    .connection-indicator {
      display: flex;
      align-items: center;
      gap: 1rem;
      background: rgba(255, 255, 255, 0.1);
      padding: 0.5rem 1rem;
      border-radius: 8px;
    }

    .status-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      position: relative;
    }

    .status-dot.healthy {
      background: #10b981;
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
    }

    .status-dot.warning {
      background: #f59e0b;
      box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
    }

    .status-dot.error {
      background: #ef4444;
      box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
      animation: pulse-error 2s infinite;
    }

    @keyframes pulse-error {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }

    .status-dot.healthy::after,
    .status-dot.warning::after,
    .status-dot.error::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 100%;
      height: 100%;
      border-radius: 50%;
      background: currentColor;
      transform: translate(-50%, -50%);
      animation: ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;
      opacity: 0.4;
    }

    @keyframes ping {
      75%, 100% {
        transform: translate(-50%, -50%) scale(2);
        opacity: 0;
      }
    }

    .connection-text {
      font-size: 0.875rem;
    }

    .panel-content {
      padding: 1rem;
    }

    .metrics-overview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .metric-card {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
    }

    .metric-card:hover {
      border-color: #0891b2;
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .metric-card.selected {
      border-color: #0891b2;
      background: #ecfeff;
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

    .metric-trend {
      font-size: 0.75rem;
      margin-top: 0.25rem;
    }

    .trend-up {
      color: #10b981;
    }

    .trend-down {
      color: #ef4444;
    }

    .trend-stable {
      color: #6b7280;
    }

    .detailed-section {
      margin-bottom: 2rem;
    }

    .section-title {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .latency-chart {
      height: 200px;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      margin-bottom: 1rem;
    }

    .chart-placeholder {
      color: #6b7280;
      font-size: 0.875rem;
      text-align: center;
    }

    .latency-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 1rem;
    }

    .latency-stat {
      text-align: center;
      padding: 0.75rem;
      background: white;
      border-radius: 6px;
      border: 1px solid #e5e7eb;
    }

    .latency-stat-value {
      font-size: 1.25rem;
      font-weight: 600;
      color: #111827;
    }

    .latency-stat-label {
      font-size: 0.75rem;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .redis-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }

    .redis-metric {
      background: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 8px;
      padding: 1rem;
    }

    .redis-metric-name {
      font-weight: 600;
      color: #111827;
      margin-bottom: 0.5rem;
    }

    .redis-metric-value {
      font-size: 1.125rem;
      font-weight: 700;
      color: #0891b2;
    }

    .redis-metric-unit {
      font-size: 0.875rem;
      color: #6b7280;
      margin-left: 0.25rem;
    }

    .error-log {
      max-height: 300px;
      overflow-y: auto;
    }

    .error-item {
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 0.75rem;
      transition: all 0.2s;
    }

    .error-item:hover {
      border-color: #f87171;
    }

    .error-item.medium {
      background: #fffbeb;
      border-color: #fed7aa;
    }

    .error-item.low {
      background: #f0f9ff;
      border-color: #bae6fd;
    }

    .error-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .error-type {
      font-weight: 600;
      color: #111827;
    }

    .error-severity {
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }

    .severity-critical {
      background: #fee2e2;
      color: #991b1b;
    }

    .severity-high {
      background: #fed7aa;
      color: #ea580c;
    }

    .severity-medium {
      background: #fef3c7;
      color: #92400e;
    }

    .severity-low {
      background: #dcfce7;
      color: #166534;
    }

    .error-message {
      color: #6b7280;
      font-size: 0.875rem;
      margin-bottom: 0.5rem;
    }

    .error-meta {
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 0.75rem;
      color: #9ca3af;
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 150px;
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

    .connection-actions {
      display: flex;
      gap: 0.5rem;
      margin-top: 1rem;
    }

    .action-btn {
      padding: 0.5rem 1rem;
      border: 1px solid #d1d5db;
      background: white;
      border-radius: 6px;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s;
    }

    .action-btn:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }

    .action-btn.primary {
      background: #0891b2;
      color: white;
      border-color: #0891b2;
    }

    .action-btn.primary:hover {
      background: #0e7490;
    }

    @media (max-width: 768px) {
      .metrics-overview {
        grid-template-columns: repeat(2, 1fr);
      }

      .latency-stats {
        grid-template-columns: repeat(2, 1fr);
      }

      .redis-metrics {
        grid-template-columns: 1fr;
      }

      .connection-indicator {
        flex-direction: column;
        gap: 0.5rem;
        text-align: center;
      }
    }
  `

  constructor() {
    super()
    this.realtime = true
    this.compact = false
    this.timeRange = '1h'
    this.communicationHealth = null
    this.redisMetrics = null
    this.latencyHistory = []
    this.recentErrors = []
    this.isLoading = true
    this.error = ''
    this.lastUpdate = null
    this.selectedMetric = 'latency'
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
      // Load communication health data from coordination monitoring API
      const response = await fetch('/api/v1/coordination-monitoring/dashboard')
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const dashboardData = await response.json()
      
      this.communicationHealth = dashboardData.communication_health
      
      // Generate mock Redis metrics and latency history
      await Promise.all([
        this.loadRedisMetrics(),
        this.loadLatencyHistory(),
        this.loadRecentErrors()
      ])
      
      this.lastUpdate = new Date()

    } catch (error) {
      console.error('Failed to load communication monitoring data:', error)
      this.error = error instanceof Error ? error.message : 'Unknown error occurred'
    } finally {
      this.isLoading = false
    }
  }

  private async loadRedisMetrics() {
    try {
      // Mock Redis metrics - in production, this would come from Redis monitoring
      this.redisMetrics = {
        connected_clients: this.communicationHealth?.connected_agents || 0,
        memory_usage: Math.random() * 100, // MB
        operations_per_second: Math.random() * 1000,
        keyspace_hits: Math.floor(Math.random() * 10000),
        keyspace_misses: Math.floor(Math.random() * 1000),
        uptime: Date.now() - (Math.random() * 86400000) // Up to 1 day
      }

    } catch (error) {
      console.warn('Failed to load Redis metrics:', error)
    }
  }

  private async loadLatencyHistory() {
    try {
      // Generate mock latency history
      const now = Date.now()
      const points = []
      
      for (let i = 59; i >= 0; i--) {
        const timestamp = now - i * 60 * 1000 // Last 60 minutes
        const baseLatency = this.communicationHealth?.average_latency || 50
        const latency = Math.max(0, baseLatency + (Math.random() - 0.5) * 20)
        
        points.push({
          timestamp,
          latency,
          agent_id: `agent-${Math.floor(Math.random() * 5)}`,
          message_type: ['task_assignment', 'heartbeat', 'coordination', 'status_update'][Math.floor(Math.random() * 4)]
        })
      }
      
      this.latencyHistory = points

    } catch (error) {
      console.warn('Failed to load latency history:', error)
    }
  }

  private async loadRecentErrors() {
    try {
      // Generate mock communication errors
      const errorTypes = [
        'connection_timeout', 'message_serialization_failed', 
        'redis_connection_lost', 'agent_unreachable', 'queue_overflow'
      ]
      
      const errors = []
      const errorCount = Math.floor(Math.random() * 5)
      
      for (let i = 0; i < errorCount; i++) {
        const timestamp = new Date(Date.now() - Math.random() * 3600000) // Last hour
        errors.push({
          id: `error-${i}`,
          timestamp: timestamp.toISOString(),
          agent_id: `agent-${Math.floor(Math.random() * 5)}`,
          error_type: errorTypes[Math.floor(Math.random() * errorTypes.length)],
          message: 'Mock communication error for demonstration',
          severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any
        })
      }
      
      this.recentErrors = errors.sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )

    } catch (error) {
      console.warn('Failed to load recent errors:', error)
    }
  }

  private startRealtimeUpdates() {
    // Update every 3 seconds for communication monitoring
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

  private async testConnection() {
    try {
      // Test Redis connection
      const response = await fetch('/api/v1/coordination-monitoring/test/redis-connection', {
        method: 'POST'
      })

      if (response.ok) {
        const result = await response.json()
        console.log('Connection test result:', result)
        
        // Refresh data after test
        await this.loadData()
      }

    } catch (error) {
      console.error('Connection test failed:', error)
      this.error = `Connection test failed: ${error}`
    }
  }

  private async resetConnection() {
    try {
      // Reset Redis connection
      const response = await fetch('/api/v1/coordination-monitoring/recovery-actions/reconnect-redis', {
        method: 'POST'
      })

      if (response.ok) {
        const result = await response.json()
        console.log('Connection reset result:', result)
        
        // Refresh data after reset
        setTimeout(() => this.loadData(), 2000)
      }

    } catch (error) {
      console.error('Connection reset failed:', error)
      this.error = `Connection reset failed: ${error}`
    }
  }

  private formatLatency(latency: number): string {
    if (latency < 1) return `${(latency * 1000).toFixed(1)}μs`
    if (latency < 1000) return `${latency.toFixed(1)}ms`
    return `${(latency / 1000).toFixed(2)}s`
  }

  private formatUptime(uptime: number): string {
    const hours = Math.floor(uptime / 3600000)
    const minutes = Math.floor((uptime % 3600000) / 60000)
    
    if (hours > 0) return `${hours}h ${minutes}m`
    return `${minutes}m`
  }

  private getConnectionStatus(): { status: string; color: string; message: string } {
    if (!this.communicationHealth) {
      return { status: 'unknown', color: '#6b7280', message: 'Status unknown' }
    }

    const { redis_status, error_rate, average_latency } = this.communicationHealth

    if (redis_status !== 'healthy') {
      return { status: 'error', color: '#ef4444', message: 'Redis connection issues' }
    }

    if (error_rate > 10) {
      return { status: 'warning', color: '#f59e0b', message: `${error_rate.toFixed(1)}% error rate` }
    }

    if (average_latency > 100) {
      return { status: 'warning', color: '#f59e0b', message: `High latency: ${this.formatLatency(average_latency)}` }
    }

    return { status: 'healthy', color: '#10b981', message: 'All systems operational' }
  }

  render() {
    if (this.isLoading && !this.communicationHealth) {
      return html`
        <div class="loading-state">
          <div>Loading communication monitoring data...</div>
        </div>
      `
    }

    if (this.error && !this.communicationHealth) {
      return html`
        <div class="error-state">
          <p><strong>Error:</strong> ${this.error}</p>
          <button class="action-btn" @click=${() => this.loadData()}>
            Retry
          </button>
        </div>
      `
    }

    const health = this.communicationHealth!
    const connectionStatus = this.getConnectionStatus()
    
    // Calculate latency statistics
    const latencies = this.latencyHistory.map(p => p.latency)
    const avgLatency = latencies.length > 0 ? latencies.reduce((sum, l) => sum + l, 0) / latencies.length : 0
    const maxLatency = latencies.length > 0 ? Math.max(...latencies) : 0
    const minLatency = latencies.length > 0 ? Math.min(...latencies) : 0

    return html`
      <div class="panel-header">
        <div class="panel-title">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
          </svg>
          Agent Communication Monitoring
        </div>

        <div class="connection-indicator">
          <div class="status-dot ${connectionStatus.status}"></div>
          <div class="connection-text">
            <div style="font-weight: 600;">${health.redis_status.toUpperCase()}</div>
            <div style="font-size: 0.75rem; opacity: 0.9;">${connectionStatus.message}</div>
          </div>
        </div>
      </div>

      <div class="panel-content">
        ${this.error ? html`
          <div class="error-state" style="margin-bottom: 1rem;">
            <p>${this.error}</p>
          </div>
        ` : ''}

        <!-- Communication Metrics Overview -->
        <div class="metrics-overview">
          <div class="metric-card ${this.selectedMetric === 'latency' ? 'selected' : ''}"
               @click=${() => this.selectedMetric = 'latency'}>
            <div class="metric-value">${this.formatLatency(health.average_latency)}</div>
            <div class="metric-label">Average Latency</div>
            <div class="metric-trend trend-${health.average_latency < 50 ? 'stable' : 'up'}">
              ${health.average_latency < 50 ? '→' : '↗'} Real-time
            </div>
          </div>

          <div class="metric-card ${this.selectedMetric === 'throughput' ? 'selected' : ''}"
               @click=${() => this.selectedMetric = 'throughput'}>
            <div class="metric-value">${health.message_throughput.toFixed(1)}</div>
            <div class="metric-label">Messages/sec</div>
            <div class="metric-trend trend-stable">→ Stable</div>
          </div>

          <div class="metric-card ${this.selectedMetric === 'errors' ? 'selected' : ''}"
               @click=${() => this.selectedMetric = 'errors'}>
            <div class="metric-value">${health.error_rate.toFixed(1)}%</div>
            <div class="metric-label">Error Rate</div>
            <div class="metric-trend trend-${health.error_rate < 5 ? 'stable' : 'up'}">
              ${health.error_rate < 5 ? '→' : '↗'} ${health.error_rate < 5 ? 'Good' : 'High'}
            </div>
          </div>

          <div class="metric-card ${this.selectedMetric === 'agents' ? 'selected' : ''}"
               @click=${() => this.selectedMetric = 'agents'}>
            <div class="metric-value">${health.connected_agents}</div>
            <div class="metric-label">Connected Agents</div>
            <div class="metric-trend trend-stable">→ Online</div>
          </div>
        </div>

        <!-- Message Latency Analysis -->
        <div class="detailed-section">
          <div class="section-title">
            📊 Message Latency Analysis
          </div>

          <div class="latency-chart">
            <div class="chart-placeholder">
              Real-time latency chart<br>
              <small>Last 60 minutes of message latency data</small><br>
              <small>Average: ${this.formatLatency(avgLatency)} | Range: ${this.formatLatency(minLatency)} - ${this.formatLatency(maxLatency)}</small>
            </div>
          </div>

          <div class="latency-stats">
            <div class="latency-stat">
              <div class="latency-stat-value">${this.formatLatency(avgLatency)}</div>
              <div class="latency-stat-label">Average</div>
            </div>
            <div class="latency-stat">
              <div class="latency-stat-value">${this.formatLatency(maxLatency)}</div>
              <div class="latency-stat-label">Peak</div>
            </div>
            <div class="latency-stat">
              <div class="latency-stat-value">${this.formatLatency(minLatency)}</div>
              <div class="latency-stat-label">Minimum</div>
            </div>
            <div class="latency-stat">
              <div class="latency-stat-value">${health.recent_latencies.length}</div>
              <div class="latency-stat-label">Recent Samples</div>
            </div>
          </div>
        </div>

        <!-- Redis Health Metrics -->
        ${this.redisMetrics ? html`
          <div class="detailed-section">
            <div class="section-title">
              🔧 Redis Message Bus Health
            </div>

            <div class="redis-metrics">
              <div class="redis-metric">
                <div class="redis-metric-name">Connected Clients</div>
                <div class="redis-metric-value">
                  ${this.redisMetrics.connected_clients}
                  <span class="redis-metric-unit">clients</span>
                </div>
              </div>

              <div class="redis-metric">
                <div class="redis-metric-name">Memory Usage</div>
                <div class="redis-metric-value">
                  ${this.redisMetrics.memory_usage.toFixed(1)}
                  <span class="redis-metric-unit">MB</span>
                </div>
              </div>

              <div class="redis-metric">
                <div class="redis-metric-name">Operations/sec</div>
                <div class="redis-metric-value">
                  ${this.redisMetrics.operations_per_second.toFixed(0)}
                  <span class="redis-metric-unit">ops</span>
                </div>
              </div>

              <div class="redis-metric">
                <div class="redis-metric-name">Hit Rate</div>
                <div class="redis-metric-value">
                  ${((this.redisMetrics.keyspace_hits / (this.redisMetrics.keyspace_hits + this.redisMetrics.keyspace_misses)) * 100).toFixed(1)}
                  <span class="redis-metric-unit">%</span>
                </div>
              </div>

              <div class="redis-metric">
                <div class="redis-metric-name">Uptime</div>
                <div class="redis-metric-value">
                  ${this.formatUptime(this.redisMetrics.uptime)}
                </div>
              </div>
            </div>
          </div>
        ` : ''}

        <!-- Recent Communication Errors -->
        <div class="detailed-section">
          <div class="section-title">
            ⚠️ Communication Errors (${this.recentErrors.length})
          </div>

          ${this.recentErrors.length === 0 ? html`
            <div class="empty-state">
              <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p>No communication errors detected</p>
            </div>
          ` : html`
            <div class="error-log">
              ${this.recentErrors.map(error => html`
                <div class="error-item ${error.severity}">
                  <div class="error-header">
                    <div class="error-type">${error.error_type.replace(/_/g, ' ')}</div>
                    <div class="error-severity severity-${error.severity}">
                      ${error.severity}
                    </div>
                  </div>
                  <div class="error-message">${error.message}</div>
                  <div class="error-meta">
                    <span>Agent: ${error.agent_id}</span>
                    <span>Time: ${new Date(error.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              `)}
            </div>
          `}
        </div>

        <!-- Connection Actions -->
        <div class="connection-actions">
          <button class="action-btn" @click=${() => this.loadData()}>
            🔄 Refresh Data
          </button>
          <button class="action-btn" @click=${this.testConnection}>
            🔍 Test Connection
          </button>
          <button class="action-btn primary" @click=${this.resetConnection}>
            🔌 Reset Connection
          </button>
        </div>

        ${this.lastUpdate ? html`
          <div style="font-size: 0.75rem; color: #6b7280; text-align: right; margin-top: 1rem;">
            Last updated: ${this.lastUpdate.toLocaleTimeString()}
            • ${this.latencyHistory.length} latency samples
          </div>
        ` : ''}
      </div>
    `
  }
}