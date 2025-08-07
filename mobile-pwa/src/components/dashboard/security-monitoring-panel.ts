/**
 * Security Monitoring Panel
 * 
 * Real-time security dashboard with threat detection, monitoring, and alerts
 * Priority: Critical - Essential for enterprise security oversight
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export interface SecurityAlert {
  id: string
  type: 'intrusion' | 'authentication' | 'permission' | 'data_breach' | 'rate_limit' | 'suspicious_activity'
  severity: 'critical' | 'high' | 'medium' | 'low'
  title: string
  message: string
  source: string
  timestamp: string
  status: 'active' | 'investigating' | 'resolved' | 'false_positive'
  affected_agents?: string[]
  metadata: Record<string, any>
}

export interface SecurityMetrics {
  threat_detection: {
    active_threats: number
    resolved_today: number
    false_positives: number
    threat_level: 'minimal' | 'elevated' | 'high' | 'critical'
  }
  authentication: {
    successful_logins: number
    failed_attempts: number
    suspicious_logins: number
    active_sessions: number
    mfa_compliance_rate: number
  }
  access_control: {
    permission_violations: number
    unauthorized_access_attempts: number
    privilege_escalations: number
    data_access_anomalies: number
  }
  network_security: {
    blocked_connections: number
    malicious_requests: number
    rate_limit_violations: number
    ddos_attempts: number
  }
  data_protection: {
    encryption_status: 'healthy' | 'degraded' | 'critical'
    backup_status: 'current' | 'delayed' | 'failed'
    data_integrity_score: number
    compliance_violations: number
  }
  system_security: {
    vulnerability_score: number
    patch_compliance: number
    security_updates_pending: number
    configuration_drift: number
  }
  timestamp: string
}

@customElement('security-monitoring-panel')
export class SecurityMonitoringPanel extends LitElement {
  @property({ type: Object }) declare metrics: SecurityMetrics | null
  @property({ type: Array }) declare alerts: SecurityAlert[]
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: Boolean }) declare compact: boolean
  
  @state() private selectedView: string = 'overview'
  @state() private selectedSeverity: 'all' | 'critical' | 'high' | 'medium' | 'low' = 'all'
  @state() private showResolved: boolean = false
  @state() private autoRefresh: boolean = true
  @state() private lastUpdate: Date | null = null
  @state() private emergencyMode: boolean = false
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: white;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }
    
    .security-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
      background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
      color: white;
    }
    
    .security-header.emergency {
      background: linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%);
      animation: emergencyPulse 2s ease-in-out infinite;
    }
    
    @keyframes emergencyPulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.8; }
    }
    
    .header-title {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 1.125rem;
      font-weight: 600;
    }
    
    .security-icon {
      width: 20px;
      height: 20px;
    }
    
    .threat-level-indicator {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
    }
    
    .threat-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }
    
    .threat-dot.minimal {
      background: #10b981;
    }
    
    .threat-dot.elevated {
      background: #f59e0b;
    }
    
    .threat-dot.high {
      background: #ef4444;
    }
    
    .threat-dot.critical {
      background: #dc2626;
      animation: criticalPulse 1s infinite;
    }
    
    @keyframes criticalPulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.2); opacity: 0.7; }
    }
    
    .header-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .emergency-button {
      background: rgba(255, 255, 255, 0.9);
      border: 2px solid white;
      color: #dc2626;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      text-transform: uppercase;
    }
    
    .emergency-button:hover {
      background: white;
      transform: scale(1.05);
    }
    
    .emergency-button.active {
      background: #fef2f2;
      color: #b91c1c;
      animation: emergencyButtonPulse 1.5s infinite;
    }
    
    @keyframes emergencyButtonPulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
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
    }
    
    .security-content {
      height: calc(100% - 70px);
      overflow-y: auto;
    }
    
    .security-tabs {
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
      position: relative;
    }
    
    .tab-button:hover {
      color: #374151;
      background: #f3f4f6;
    }
    
    .tab-button.active {
      color: #dc2626;
      border-bottom-color: #dc2626;
      background: white;
    }
    
    .tab-button .alert-count {
      background: #dc2626;
      color: white;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.625rem;
      font-weight: 600;
      margin-left: 0.375rem;
    }
    
    .security-panel {
      padding: 1rem;
    }
    
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
      background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    .metric-card.high {
      border-color: #fed7aa;
      background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .metric-card.healthy {
      border-color: #bbf7d0;
      background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    .metric-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.75rem;
    }
    
    .metric-label {
      font-size: 0.875rem;
      color: #6b7280;
      font-weight: 500;
    }
    
    .metric-icon {
      width: 16px;
      height: 16px;
      opacity: 0.7;
    }
    
    .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
    }
    
    .metric-value.critical {
      color: #dc2626;
    }
    
    .metric-value.high {
      color: #d97706;
    }
    
    .metric-value.healthy {
      color: #059669;
    }
    
    .metric-description {
      font-size: 0.75rem;
      color: #6b7280;
      line-height: 1.4;
    }
    
    .alerts-section {
      margin-top: 2rem;
    }
    
    .alerts-header {
      display: flex;
      justify-content: between;
      align-items: center;
      margin-bottom: 1rem;
      gap: 1rem;
    }
    
    .alerts-title {
      font-size: 1rem;
      font-weight: 600;
      color: #374151;
    }
    
    .alerts-filters {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    
    .filter-select {
      background: white;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      padding: 0.25rem 0.5rem;
      font-size: 0.75rem;
      cursor: pointer;
    }
    
    .alerts-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      max-height: 400px;
      overflow-y: auto;
    }
    
    .alert-item {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      padding: 1rem;
      border-radius: 0.5rem;
      border: 1px solid transparent;
      transition: all 0.2s;
      cursor: pointer;
    }
    
    .alert-item:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .alert-item.critical {
      background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
      border-color: #fecaca;
    }
    
    .alert-item.high {
      background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
      border-color: #fed7aa;
    }
    
    .alert-item.medium {
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      border-color: #bfdbfe;
    }
    
    .alert-item.low {
      background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
      border-color: #e5e7eb;
    }
    
    .alert-icon {
      width: 20px;
      height: 20px;
      flex-shrink: 0;
      margin-top: 0.125rem;
    }
    
    .alert-content {
      flex: 1;
    }
    
    .alert-title {
      font-size: 0.875rem;
      font-weight: 600;
      color: #374151;
      margin-bottom: 0.25rem;
    }
    
    .alert-message {
      font-size: 0.75rem;
      color: #6b7280;
      margin-bottom: 0.5rem;
      line-height: 1.4;
    }
    
    .alert-metadata {
      display: flex;
      gap: 1rem;
      font-size: 0.625rem;
      color: #9ca3af;
    }
    
    .alert-actions {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      align-items: flex-end;
    }
    
    .alert-time {
      font-size: 0.625rem;
      color: #9ca3af;
      white-space: nowrap;
    }
    
    .alert-status {
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 500;
      text-transform: uppercase;
    }
    
    .alert-status.active {
      background: #fef2f2;
      color: #dc2626;
    }
    
    .alert-status.investigating {
      background: #fffbeb;
      color: #d97706;
    }
    
    .alert-status.resolved {
      background: #f0fdf4;
      color: #059669;
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
      border-top: 2px solid #dc2626;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    @media (max-width: 768px) {
      .metrics-grid {
        grid-template-columns: 1fr;
      }
      
      .security-header {
        flex-direction: column;
        gap: 0.5rem;
        align-items: stretch;
      }
      
      .header-controls {
        justify-content: center;
      }
      
      .alerts-header {
        flex-direction: column;
        align-items: stretch;
      }
    }
  `
  
  constructor() {
    super()
    this.metrics = null
    this.alerts = []
    this.realtime = true
    this.compact = false
  }
  
  private get filteredAlerts() {
    let filtered = this.alerts
    
    if (this.selectedSeverity !== 'all') {
      filtered = filtered.filter(alert => alert.severity === this.selectedSeverity)
    }
    
    if (!this.showResolved) {
      filtered = filtered.filter(alert => alert.status !== 'resolved' && alert.status !== 'false_positive')
    }
    
    return filtered.sort((a, b) => {
      // Sort by severity first, then by timestamp
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 }
      const severityDiff = severityOrder[b.severity] - severityOrder[a.severity]
      if (severityDiff !== 0) return severityDiff
      
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    })
  }
  
  private get criticalAlertCount() {
    return this.alerts.filter(a => a.severity === 'critical' && a.status === 'active').length
  }
  
  private get highAlertCount() {
    return this.alerts.filter(a => a.severity === 'high' && a.status === 'active').length
  }
  
  private getThreatLevelColor(level: string) {
    switch (level) {
      case 'minimal': return '#10b981'
      case 'elevated': return '#f59e0b'
      case 'high': return '#ef4444'
      case 'critical': return '#dc2626'
      default: return '#6b7280'
    }
  }
  
  private getSecurityIcon(type: string) {
    switch (type) {
      case 'intrusion':
        return html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>`
      case 'authentication':
        return html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>`
      case 'permission':
        return html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>`
      case 'data_breach':
        return html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>`
      case 'rate_limit':
        return html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>`
      default:
        return html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.618 5.984A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>`
    }
  }
  
  private renderThreatDetection() {
    if (!this.metrics) return this.renderLoadingState()
    
    const { threat_detection } = this.metrics
    const threatColor = this.getThreatLevelColor(threat_detection.threat_level)
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card ${threat_detection.threat_level === 'minimal' ? 'healthy' : threat_detection.threat_level === 'critical' || threat_detection.threat_level === 'high' ? 'critical' : 'high'}">
          <div class="metric-header">
            <div class="metric-label">Active Threats</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
            </svg>
          </div>
          <div class="metric-value ${threat_detection.threat_level === 'minimal' ? 'healthy' : threat_detection.threat_level === 'critical' || threat_detection.threat_level === 'high' ? 'critical' : 'high'}">
            ${threat_detection.active_threats}
          </div>
          <div class="metric-description">
            Threat Level: <strong style="color: ${threatColor}">${threat_detection.threat_level.toUpperCase()}</strong>
          </div>
        </div>
        
        <div class="metric-card healthy">
          <div class="metric-header">
            <div class="metric-label">Resolved Today</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <div class="metric-value healthy">
            ${threat_detection.resolved_today}
          </div>
          <div class="metric-description">
            Security incidents successfully resolved
          </div>
        </div>
        
        <div class="metric-card ${threat_detection.false_positives > 5 ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">False Positives</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <div class="metric-value ${threat_detection.false_positives > 5 ? 'high' : 'healthy'}">
            ${threat_detection.false_positives}
          </div>
          <div class="metric-description">
            Incorrectly flagged security events
          </div>
        </div>
      </div>
    `
  }
  
  private renderAuthentication() {
    if (!this.metrics) return this.renderLoadingState()
    
    const { authentication } = this.metrics
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card ${authentication.failed_attempts > 10 ? 'critical' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Failed Login Attempts</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
            </svg>
          </div>
          <div class="metric-value ${authentication.failed_attempts > 10 ? 'critical' : 'healthy'}">
            ${authentication.failed_attempts}
          </div>
          <div class="metric-description">
            Unsuccessful authentication attempts today
          </div>
        </div>
        
        <div class="metric-card healthy">
          <div class="metric-header">
            <div class="metric-label">Active Sessions</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"/>
            </svg>
          </div>
          <div class="metric-value healthy">
            ${authentication.active_sessions}
          </div>
          <div class="metric-description">
            Currently authenticated user sessions
          </div>
        </div>
        
        <div class="metric-card ${authentication.mfa_compliance_rate < 90 ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">MFA Compliance</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
            </svg>
          </div>
          <div class="metric-value ${authentication.mfa_compliance_rate < 90 ? 'high' : 'healthy'}">
            ${Math.round(authentication.mfa_compliance_rate)}%
          </div>
          <div class="metric-description">
            Multi-factor authentication adoption rate
          </div>
        </div>
      </div>
    `
  }
  
  private renderAlerts() {
    const filteredAlerts = this.filteredAlerts
    
    if (!filteredAlerts.length) {
      return html`
        <div class="empty-state">
          <p>No security alerts</p>
          <small>System security is operating normally</small>
        </div>
      `
    }
    
    return html`
      <div class="alerts-list">
        ${filteredAlerts.map(alert => html`
          <div class="alert-item ${alert.severity}" @click=${() => this.handleAlertClick(alert)}>
            <svg class="alert-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              ${this.getSecurityIcon(alert.type)}
            </svg>
            <div class="alert-content">
              <div class="alert-title">${alert.title}</div>
              <div class="alert-message">${alert.message}</div>
              <div class="alert-metadata">
                <span>Source: ${alert.source}</span>
                ${alert.affected_agents ? html`<span>Agents: ${alert.affected_agents.join(', ')}</span>` : ''}
              </div>
            </div>
            <div class="alert-actions">
              <div class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</div>
              <div class="alert-status ${alert.status}">${alert.status.replace('_', ' ')}</div>
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
        <span>Loading security metrics...</span>
      </div>
    `
  }
  
  private renderCurrentTab() {
    switch (this.selectedView) {
      case 'threats':
        return this.renderThreatDetection()
      case 'auth':
        return this.renderAuthentication()
      case 'alerts':
        return this.renderAlerts()
      default:
        return html`
          ${this.renderThreatDetection()}
          ${this.alerts.length > 0 ? html`
            <div class="alerts-section">
              <div class="alerts-header">
                <h3 class="alerts-title">Recent Security Alerts</h3>
                <div class="alerts-filters">
                  <select class="filter-select" .value=${this.selectedSeverity} @change=${this.handleSeverityFilter}>
                    <option value="all">All Severities</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                  <label>
                    <input type="checkbox" .checked=${this.showResolved} @change=${this.handleShowResolved}> 
                    Show Resolved
                  </label>
                </div>
              </div>
              ${this.renderAlerts()}
            </div>
          ` : ''}
        `
    }
  }
  
  private handleTabChange(tab: string) {
    this.selectedView = tab
    this.dispatchEvent(new CustomEvent('tab-changed', {
      detail: { tab },
      bubbles: true,
      composed: true
    }))
  }
  
  private handleSeverityFilter(event: Event) {
    const target = event.target as HTMLSelectElement
    this.selectedSeverity = target.value as any
  }
  
  private handleShowResolved(event: Event) {
    const target = event.target as HTMLInputElement
    this.showResolved = target.checked
  }
  
  private handleAlertClick(alert: SecurityAlert) {
    console.log('Alert clicked:', alert)
    this.dispatchEvent(new CustomEvent('alert-selected', {
      detail: { alert },
      bubbles: true,
      composed: true
    }))
  }
  
  private toggleEmergencyMode() {
    this.emergencyMode = !this.emergencyMode
    this.dispatchEvent(new CustomEvent('emergency-mode-toggled', {
      detail: { enabled: this.emergencyMode },
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
    if ((changedProperties.has('metrics') || changedProperties.has('alerts')) && (this.metrics || this.alerts.length)) {
      this.lastUpdate = new Date()
      
      // Check if we should enter emergency mode
      const criticalThreats = this.criticalAlertCount
      if (criticalThreats > 0 && !this.emergencyMode) {
        this.emergencyMode = true
      }
    }
  }
  
  render() {
    const threatLevel = this.metrics?.threat_detection?.threat_level || 'minimal'
    
    return html`
      <div class="security-header ${this.emergencyMode ? 'emergency' : ''}">
        <div class="header-title">
          <svg class="security-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.618 5.984A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
          </svg>
          Security Monitor
          ${this.emergencyMode ? html`<span>⚠️ EMERGENCY</span>` : ''}
        </div>
        <div class="threat-level-indicator">
          <div class="threat-dot ${threatLevel}"></div>
          Threat Level: ${threatLevel.toUpperCase()}
        </div>
        <div class="header-controls">
          <button 
            class="emergency-button ${this.emergencyMode ? 'active' : ''}"
            @click=${this.toggleEmergencyMode}
            title="${this.emergencyMode ? 'Exit' : 'Enter'} emergency mode"
          >
            ${this.emergencyMode ? 'EXIT EMERGENCY' : 'EMERGENCY'}
          </button>
          <button 
            class="control-button ${this.autoRefresh ? 'active' : ''}"
            @click=${this.toggleAutoRefresh}
            title="${this.autoRefresh ? 'Disable' : 'Enable'} auto-refresh"
          >
            ${this.autoRefresh ? '⏸️' : '▶️'}
          </button>
          ${this.lastUpdate ? html`
            <span class="control-button">
              Updated: ${this.lastUpdate.toLocaleTimeString()}
            </span>
          ` : ''}
        </div>
      </div>
      
      <div class="security-content">
        <div class="security-tabs">
          <button 
            class="tab-button ${this.selectedView === 'overview' ? 'active' : ''}"
            @click=${() => this.handleTabChange('overview')}
          >
            Overview
          </button>
          <button 
            class="tab-button ${this.selectedView === 'threats' ? 'active' : ''}"
            @click=${() => this.handleTabChange('threats')}
          >
            Threats
          </button>
          <button 
            class="tab-button ${this.selectedView === 'auth' ? 'active' : ''}"
            @click=${() => this.handleTabChange('auth')}
          >
            Authentication
          </button>
          <button 
            class="tab-button ${this.selectedView === 'alerts' ? 'active' : ''}"
            @click=${() => this.handleTabChange('alerts')}
          >
            Alerts
            ${this.criticalAlertCount + this.highAlertCount > 0 ? html`
              <span class="alert-count">${this.criticalAlertCount + this.highAlertCount}</span>
            ` : ''}
          </button>
        </div>
        
        <div class="security-panel">
          ${this.renderCurrentTab()}
        </div>
      </div>
    `
  }
}