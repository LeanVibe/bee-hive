/**
 * Performance Metrics Panel
 * 
 * Real-time performance monitoring dashboard with charts, metrics, and alerts
 * Priority: Critical - Essential for enterprise-grade monitoring
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export interface PerformanceMetrics {
  system_metrics: {
    cpu_usage: number
    memory_usage: number
    network_usage: number
    disk_usage: number
  }
  agent_metrics: Record<string, {
    performance_score: number
    task_completion_rate: number
    error_rate: number
    uptime: number
  }>
  response_times: {
    api_response_time: number
    websocket_latency: number
    database_query_time: number
  }
  throughput: {
    requests_per_second: number
    tasks_completed_per_hour: number
    agent_operations_per_minute: number
  }
  alerts: Array<{
    id: string
    type: 'performance' | 'threshold' | 'anomaly'
    severity: 'critical' | 'warning' | 'info'
    message: string
    timestamp: string
    metric: string
    current_value: number
    threshold_value: number
  }>
  timestamp: string
}

@customElement('performance-metrics-panel')
export class PerformanceMetricsPanel extends LitElement {
  @property({ type: Object }) declare metrics: PerformanceMetrics | null
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: Boolean }) declare compact: boolean
  @property({ type: String }) declare timeRange: '1h' | '6h' | '24h' | '7d'
  
  @state() private selectedMetric: string = 'overview'
  @state() private selectedAgent: string = 'all'
  @state() private showAlerts: boolean = true
  @state() private autoRefresh: boolean = true
  @state() private lastUpdate: Date | null = null
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: white;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }
    
    .performance-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
    }
    
    .header-title {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 1.125rem;
      font-weight: 600;
    }
    
    .performance-icon {
      width: 20px;
      height: 20px;
    }
    
    .header-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .control-button {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .control-button:hover {
      background: rgba(255, 255, 255, 0.2);
    }
    
    .control-button.active {
      background: rgba(255, 255, 255, 0.3);
      border-color: rgba(255, 255, 255, 0.4);
    }
    
    .realtime-indicator {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.75rem;
      opacity: 0.9;
    }
    
    .realtime-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }
    
    .realtime-dot.paused {
      background: #f59e0b;
      animation: none;
    }
    
    .performance-content {
      height: calc(100% - 70px);
      overflow-y: auto;
    }
    
    .metrics-tabs {
      display: flex;
      border-bottom: 1px solid #e5e7eb;
      background: #f9fafb;
      overflow-x: auto;
    }
    
    .tab-button {
      background: none;
      border: none;
      padding: 0.75rem 1rem;
      cursor: pointer;
      transition: all 0.2s;
      font-size: 0.875rem;
      color: #6b7280;
      white-space: nowrap;
      border-bottom: 2px solid transparent;
    }
    
    .tab-button:hover {
      color: #374151;
      background: #f3f4f6;
    }
    
    .tab-button.active {
      color: #3b82f6;
      border-bottom-color: #3b82f6;
      background: white;
    }
    
    .metrics-panel {
      padding: 1rem;
    }
    
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    
    .metric-card {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      position: relative;
      overflow: hidden;
    }
    
    .metric-card.critical {
      border-color: #fecaca;
      background: #fef2f2;
    }
    
    .metric-card.warning {
      border-color: #fed7aa;
      background: #fffbeb;
    }
    
    .metric-card.healthy {
      border-color: #bbf7d0;
      background: #f0fdf4;
    }
    
    .metric-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.5rem;
    }
    
    .metric-label {
      font-size: 0.875rem;
      color: #6b7280;
      font-weight: 500;
    }
    
    .metric-trend {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.75rem;
    }
    
    .trend-up {
      color: #ef4444;
    }
    
    .trend-down {
      color: #10b981;
    }
    
    .trend-stable {
      color: #6b7280;
    }
    
    .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
    }
    
    .metric-value.critical {
      color: #dc2626;
    }
    
    .metric-value.warning {
      color: #d97706;
    }
    
    .metric-value.healthy {
      color: #059669;
    }
    
    .metric-bar {
      height: 4px;
      background: #e5e7eb;
      border-radius: 2px;
      overflow: hidden;
      margin-top: 0.5rem;
    }
    
    .metric-fill {
      height: 100%;
      border-radius: 2px;
      transition: all 0.3s ease;
    }
    
    .metric-fill.critical {
      background: linear-gradient(90deg, #dc2626, #ef4444);
    }
    
    .metric-fill.warning {
      background: linear-gradient(90deg, #d97706, #f59e0b);
    }
    
    .metric-fill.healthy {
      background: linear-gradient(90deg, #059669, #10b981);
    }
    
    .chart-container {
      height: 200px;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 2rem;
      position: relative;
    }
    
    .chart-placeholder {
      color: #6b7280;
      text-align: center;
    }
    
    .alerts-section {
      margin-top: 2rem;
    }
    
    .alerts-header {
      display: flex;
      justify-content: between;
      align-items: center;
      margin-bottom: 1rem;
    }
    
    .alerts-title {
      font-size: 1rem;
      font-weight: 600;
      color: #374151;
    }
    
    .alerts-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    
    .alert-item {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      padding: 0.75rem;
      border-radius: 0.5rem;
      border: 1px solid transparent;
    }
    
    .alert-item.critical {
      background: #fef2f2;
      border-color: #fecaca;
    }
    
    .alert-item.warning {
      background: #fffbeb;
      border-color: #fed7aa;
    }
    
    .alert-item.info {
      background: #eff6ff;
      border-color: #bfdbfe;
    }
    
    .alert-icon {
      width: 16px;
      height: 16px;
      flex-shrink: 0;
      margin-top: 0.125rem;
    }
    
    .alert-content {
      flex: 1;
    }
    
    .alert-message {
      font-size: 0.875rem;
      color: #374151;
      margin-bottom: 0.25rem;
    }
    
    .alert-details {
      font-size: 0.75rem;
      color: #6b7280;
    }
    
    .alert-time {
      font-size: 0.75rem;
      color: #9ca3af;
      white-space: nowrap;
    }
    
    .empty-state {
      text-align: center;
      padding: 3rem 1rem;
      color: #6b7280;
    }
    
    .loading-state {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 200px;
      gap: 1rem;
    }
    
    .spinner {
      width: 20px;
      height: 20px;
      border: 2px solid #e5e7eb;
      border-top: 2px solid #3b82f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    @media (max-width: 768px) {
      .metrics-grid {
        grid-template-columns: 1fr;
      }
      
      .performance-header {
        flex-direction: column;
        gap: 0.5rem;
        align-items: stretch;
      }
      
      .header-controls {
        justify-content: center;
      }
    }
  `
  
  constructor() {
    super()
    this.metrics = null
    this.realtime = true
    this.compact = false
    this.timeRange = '1h'
  }
  
  private getMetricStatus(value: number, thresholds: { warning: number; critical: number }): 'healthy' | 'warning' | 'critical' {
    if (value >= thresholds.critical) return 'critical'
    if (value >= thresholds.warning) return 'warning'
    return 'healthy'
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
      default:
        return html`<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14"/>
        </svg>`
    }
  }
  
  private renderSystemMetrics() {
    if (!this.metrics) return this.renderLoadingState()
    
    const { system_metrics } = this.metrics
    const cpuStatus = this.getMetricStatus(system_metrics.cpu_usage, { warning: 70, critical: 90 })
    const memoryStatus = this.getMetricStatus(system_metrics.memory_usage, { warning: 80, critical: 95 })
    const networkStatus = this.getMetricStatus(system_metrics.network_usage, { warning: 75, critical: 90 })
    const diskStatus = this.getMetricStatus(system_metrics.disk_usage, { warning: 85, critical: 95 })
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card ${cpuStatus}">
          <div class="metric-header">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-trend trend-${system_metrics.cpu_usage > 70 ? 'up' : 'stable'}">
              ${this.getTrendIcon(system_metrics.cpu_usage > 70 ? 'up' : 'stable')}
              ${system_metrics.cpu_usage > 70 ? '+5%' : '±0%'}
            </div>
          </div>
          <div class="metric-value ${cpuStatus}">
            ${Math.round(system_metrics.cpu_usage)}%
          </div>
          <div class="metric-bar">
            <div class="metric-fill ${cpuStatus}" style="width: ${system_metrics.cpu_usage}%"></div>
          </div>
        </div>
        
        <div class="metric-card ${memoryStatus}">
          <div class="metric-header">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-trend trend-${system_metrics.memory_usage > 80 ? 'up' : 'stable'}">
              ${this.getTrendIcon(system_metrics.memory_usage > 80 ? 'up' : 'stable')}
              ${system_metrics.memory_usage > 80 ? '+3%' : '±0%'}
            </div>
          </div>
          <div class="metric-value ${memoryStatus}">
            ${Math.round(system_metrics.memory_usage)}%
          </div>
          <div class="metric-bar">
            <div class="metric-fill ${memoryStatus}" style="width: ${system_metrics.memory_usage}%"></div>
          </div>
        </div>
        
        <div class="metric-card ${networkStatus}">
          <div class="metric-header">
            <div class="metric-label">Network Usage</div>
            <div class="metric-trend trend-stable">
              ${this.getTrendIcon('stable')}
              ±1%
            </div>
          </div>
          <div class="metric-value ${networkStatus}">
            ${Math.round(system_metrics.network_usage)}%
          </div>
          <div class="metric-bar">
            <div class="metric-fill ${networkStatus}" style="width: ${system_metrics.network_usage}%"></div>
          </div>
        </div>
        
        <div class="metric-card ${diskStatus}">
          <div class="metric-header">
            <div class="metric-label">Disk Usage</div>
            <div class="metric-trend trend-${system_metrics.disk_usage > 85 ? 'up' : 'stable'}">
              ${this.getTrendIcon(system_metrics.disk_usage > 85 ? 'up' : 'stable')}
              ${system_metrics.disk_usage > 85 ? '+2%' : '±0%'}
            </div>
          </div>
          <div class="metric-value ${diskStatus}">
            ${Math.round(system_metrics.disk_usage)}%
          </div>
          <div class="metric-bar">
            <div class="metric-fill ${diskStatus}" style="width: ${system_metrics.disk_usage}%"></div>
          </div>
        </div>
      </div>
      
      <div class="chart-container">
        <div class="chart-placeholder">
          📊 Real-time Performance Chart<br/>
          <small>System metrics visualization will be displayed here</small>
        </div>
      </div>
    `
  }
  
  private renderResponseTimes() {
    if (!this.metrics) return this.renderLoadingState()
    
    const { response_times } = this.metrics
    const apiStatus = this.getMetricStatus(response_times.api_response_time, { warning: 500, critical: 1000 })
    const wsStatus = this.getMetricStatus(response_times.websocket_latency, { warning: 100, critical: 200 })
    const dbStatus = this.getMetricStatus(response_times.database_query_time, { warning: 100, critical: 300 })
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card ${apiStatus}">
          <div class="metric-header">
            <div class="metric-label">API Response Time</div>
            <div class="metric-trend trend-${apiStatus === 'critical' ? 'up' : 'stable'}">
              ${this.getTrendIcon(apiStatus === 'critical' ? 'up' : 'stable')}
              ${apiStatus === 'critical' ? '+50ms' : '±5ms'}
            </div>
          </div>
          <div class="metric-value ${apiStatus}">
            ${Math.round(response_times.api_response_time)}ms
          </div>
          <div class="metric-bar">
            <div class="metric-fill ${apiStatus}" style="width: ${Math.min(response_times.api_response_time / 10, 100)}%"></div>
          </div>
        </div>
        
        <div class="metric-card ${wsStatus}">
          <div class="metric-header">
            <div class="metric-label">WebSocket Latency</div>
            <div class="metric-trend trend-stable">
              ${this.getTrendIcon('stable')}
              ±2ms
            </div>
          </div>
          <div class="metric-value ${wsStatus}">
            ${Math.round(response_times.websocket_latency)}ms
          </div>
          <div class="metric-bar">
            <div class="metric-fill ${wsStatus}" style="width: ${Math.min(response_times.websocket_latency / 2, 100)}%"></div>
          </div>
        </div>
        
        <div class="metric-card ${dbStatus}">
          <div class="metric-header">
            <div class="metric-label">Database Query Time</div>
            <div class="metric-trend trend-${dbStatus === 'critical' ? 'up' : 'stable'}">
              ${this.getTrendIcon(dbStatus === 'critical' ? 'up' : 'stable')}
              ${dbStatus === 'critical' ? '+20ms' : '±3ms'}
            </div>
          </div>
          <div class="metric-value ${dbStatus}">
            ${Math.round(response_times.database_query_time)}ms
          </div>
          <div class="metric-bar">
            <div class="metric-fill ${dbStatus}" style="width: ${Math.min(response_times.database_query_time / 3, 100)}%"></div>
          </div>
        </div>
      </div>
    `
  }
  
  private renderThroughput() {
    if (!this.metrics) return this.renderLoadingState()
    
    const { throughput } = this.metrics
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card healthy">
          <div class="metric-header">
            <div class="metric-label">Requests per Second</div>
            <div class="metric-trend trend-up">
              ${this.getTrendIcon('up')}
              +15%
            </div>
          </div>
          <div class="metric-value healthy">
            ${Math.round(throughput.requests_per_second)}
          </div>
        </div>
        
        <div class="metric-card healthy">
          <div class="metric-header">
            <div class="metric-label">Tasks Completed/Hour</div>
            <div class="metric-trend trend-stable">
              ${this.getTrendIcon('stable')}
              ±2%
            </div>
          </div>
          <div class="metric-value healthy">
            ${Math.round(throughput.tasks_completed_per_hour)}
          </div>
        </div>
        
        <div class="metric-card healthy">
          <div class="metric-header">
            <div class="metric-label">Agent Operations/Min</div>
            <div class="metric-trend trend-up">
              ${this.getTrendIcon('up')}
              +8%
            </div>
          </div>
          <div class="metric-value healthy">
            ${Math.round(throughput.agent_operations_per_minute)}
          </div>
        </div>
      </div>
    `
  }
  
  private renderAlerts() {
    if (!this.metrics || !this.metrics.alerts.length) {
      return html`
        <div class="empty-state">
          <p>No performance alerts</p>
          <small>System is running smoothly</small>
        </div>
      `
    }
    
    return html`
      <div class="alerts-list">
        ${this.metrics.alerts.map(alert => html`
          <div class="alert-item ${alert.severity}">
            <svg class="alert-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              ${alert.severity === 'critical' ? html`
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
              ` : alert.severity === 'warning' ? html`
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
              ` : html`
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
              `}
            </svg>
            <div class="alert-content">
              <div class="alert-message">${alert.message}</div>
              <div class="alert-details">
                ${alert.metric}: ${alert.current_value} (threshold: ${alert.threshold_value})
              </div>
            </div>
            <div class="alert-time">
              ${new Date(alert.timestamp).toLocaleTimeString()}
            </div>
          </div>
        `)}
      </div>
    `
  }
  
  private renderLoadingState() {
    return html`
      <div class="loading-state">
        <div class="spinner"></div>
        <span>Loading performance metrics...</span>
      </div>
    `
  }
  
  private renderCurrentTab() {
    switch (this.selectedMetric) {
      case 'system':
        return this.renderSystemMetrics()
      case 'response-times':
        return this.renderResponseTimes()
      case 'throughput':
        return this.renderThroughput()
      case 'alerts':
        return this.renderAlerts()
      default:
        return html`
          ${this.renderSystemMetrics()}
          ${this.showAlerts && this.metrics?.alerts?.length ? html`
            <div class="alerts-section">
              <div class="alerts-header">
                <h3 class="alerts-title">Performance Alerts</h3>
              </div>
              ${this.renderAlerts()}
            </div>
          ` : ''}
        `
    }
  }
  
  private handleTabChange(tab: string) {
    this.selectedMetric = tab
    this.dispatchEvent(new CustomEvent('tab-changed', {
      detail: { tab },
      bubbles: true,
      composed: true
    }))
  }
  
  private toggleAutoRefresh() {
    this.autoRefresh = !this.autoRefresh
    this.dispatchEvent(new CustomEvent('auto-refresh-toggled', {
      detail: { enabled: this.autoRefresh },
      bubbles: true,
      composed: true
    }))
  }
  
  updated(changedProperties: Map<string, any>) {
    if (changedProperties.has('metrics') && this.metrics) {
      this.lastUpdate = new Date()
    }
  }
  
  render() {
    return html`
      <div class="performance-header">
        <div class="header-title">
          <svg class="performance-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          Performance Monitor
        </div>
        <div class="header-controls">
          <div class="realtime-indicator">
            <div class="realtime-dot ${this.realtime && this.autoRefresh ? '' : 'paused'}"></div>
            ${this.realtime && this.autoRefresh ? 'Live' : 'Paused'}
          </div>
          <button 
            class="control-button ${this.autoRefresh ? 'active' : ''}"
            @click=${this.toggleAutoRefresh}
            title="${this.autoRefresh ? 'Disable' : 'Enable'} auto-refresh"
          >
            ${this.autoRefresh ? '⏸️' : '▶️'}
          </button>
          ${this.lastUpdate ? html`
            <span class="realtime-indicator">
              Updated: ${this.lastUpdate.toLocaleTimeString()}
            </span>
          ` : ''}
        </div>
      </div>
      
      <div class="performance-content">
        <div class="metrics-tabs">
          <button 
            class="tab-button ${this.selectedMetric === 'overview' ? 'active' : ''}"
            @click=${() => this.handleTabChange('overview')}
          >
            Overview
          </button>
          <button 
            class="tab-button ${this.selectedMetric === 'system' ? 'active' : ''}"
            @click=${() => this.handleTabChange('system')}
          >
            System
          </button>
          <button 
            class="tab-button ${this.selectedMetric === 'response-times' ? 'active' : ''}"
            @click=${() => this.handleTabChange('response-times')}
          >
            Response Times
          </button>
          <button 
            class="tab-button ${this.selectedMetric === 'throughput' ? 'active' : ''}"
            @click=${() => this.handleTabChange('throughput')}
          >
            Throughput
          </button>
          <button 
            class="tab-button ${this.selectedMetric === 'alerts' ? 'active' : ''}"
            @click=${() => this.handleTabChange('alerts')}
          >
            Alerts ${this.metrics?.alerts?.length ? `(${this.metrics.alerts.length})` : ''}
          </button>
        </div>
        
        <div class="metrics-panel">
          ${this.renderCurrentTab()}
        </div>
      </div>
    `
  }
}