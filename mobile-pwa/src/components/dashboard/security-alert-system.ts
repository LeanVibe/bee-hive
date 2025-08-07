/**
 * Security Alert System
 * 
 * Manages real-time security alerts, risk heat maps, emergency controls,
 * and comprehensive notification system for the security dashboard.
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { securityMonitoringService } from '../../services/security-monitoring'
import type { SecurityNotification } from '../common/security-alert-notification'
import '../common/security-alert-notification'

export interface RiskHeatMapData {
  component: string
  risk_score: number
  risk_level: 'minimal' | 'low' | 'medium' | 'high' | 'critical'
  issues: string[]
  last_updated: string
}

export interface EmergencyControl {
  id: string
  label: string
  description: string
  action: string
  severity: 'warning' | 'danger' | 'critical'
  confirmation_required: boolean
  estimated_impact: string
}

@customElement('security-alert-system')
export class SecurityAlertSystem extends LitElement {
  @property({ type: Boolean }) declare mobile: boolean
  @property({ type: Boolean }) declare compact: boolean

  @state() private notifications: SecurityNotification[] = []
  @state() private riskHeatMap: RiskHeatMapData[] = []
  @state() private emergencyMode = false
  @state() private emergencyControls: EmergencyControl[] = []
  @state() private alertCounts = {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0
  }
  @state() private showEmergencyControls = false
  @state() private confirmationDialog: { control: EmergencyControl } | null = null

  static styles = css`
    :host {
      display: block;
      width: 100%;
    }

    .alert-system {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .alert-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1rem;
      background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
      color: white;
      border-radius: 0.5rem;
      position: relative;
      overflow: hidden;
    }

    .alert-header.emergency {
      background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%);
      animation: emergencyPulse 2s ease-in-out infinite;
    }

    .alert-header::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
      animation: shine 3s infinite;
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .alert-icon {
      width: 24px;
      height: 24px;
    }

    .alert-title {
      font-size: 1.125rem;
      font-weight: 600;
    }

    .alert-status {
      font-size: 0.875rem;
      opacity: 0.9;
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .alert-counts {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .alert-count {
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 600;
    }

    .alert-count.critical {
      background: rgba(220, 38, 38, 0.2);
      color: #fca5a5;
      border: 1px solid rgba(220, 38, 38, 0.3);
    }

    .alert-count.high {
      background: rgba(245, 158, 11, 0.2);
      color: #fcd34d;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .alert-count.medium {
      background: rgba(37, 99, 235, 0.2);
      color: #93c5fd;
      border: 1px solid rgba(37, 99, 235, 0.3);
    }

    .alert-count.low {
      background: rgba(16, 185, 129, 0.2);
      color: #6ee7b7;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .emergency-button {
      background: rgba(220, 38, 38, 0.9);
      border: 2px solid rgba(255, 255, 255, 0.3);
      color: white;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
      text-transform: uppercase;
    }

    .emergency-button:hover {
      background: rgba(185, 28, 28, 1);
      transform: scale(1.05);
    }

    .emergency-button.active {
      animation: emergencyButtonPulse 1.5s infinite;
    }

    .notifications-container {
      max-height: 400px;
      overflow-y: auto;
      padding: 0.5rem;
      background: #f9fafb;
      border-radius: 0.5rem;
    }

    .notifications-empty {
      text-align: center;
      padding: 2rem;
      color: #6b7280;
      font-style: italic;
    }

    .risk-heat-map {
      background: white;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }

    .heat-map-header {
      padding: 1rem;
      background: #f9fafb;
      border-bottom: 1px solid #e5e7eb;
      font-weight: 600;
      color: #374151;
    }

    .heat-map-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1px;
      background: #e5e7eb;
    }

    .heat-map-cell {
      background: white;
      padding: 1rem;
      position: relative;
      cursor: pointer;
      transition: all 0.2s;
    }

    .heat-map-cell:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .heat-map-cell.minimal {
      background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
      border-left: 4px solid #10b981;
    }

    .heat-map-cell.low {
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      border-left: 4px solid #3b82f6;
    }

    .heat-map-cell.medium {
      background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
      border-left: 4px solid #f59e0b;
    }

    .heat-map-cell.high {
      background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
      border-left: 4px solid #ef4444;
    }

    .heat-map-cell.critical {
      background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
      border-left: 4px solid #dc2626;
      color: white;
      animation: criticalPulse 2s ease-in-out infinite;
    }

    .cell-component {
      font-weight: 600;
      font-size: 0.875rem;
      margin-bottom: 0.5rem;
    }

    .cell-risk-score {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 0.25rem;
    }

    .cell-risk-level {
      font-size: 0.75rem;
      text-transform: uppercase;
      font-weight: 500;
      opacity: 0.8;
    }

    .cell-issues {
      margin-top: 0.5rem;
      font-size: 0.625rem;
      opacity: 0.7;
    }

    .emergency-controls {
      background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%);
      color: white;
      border-radius: 0.5rem;
      overflow: hidden;
      border: 2px solid #dc2626;
    }

    .emergency-controls-header {
      padding: 1rem;
      background: rgba(0, 0, 0, 0.2);
      font-weight: 700;
      text-transform: uppercase;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .emergency-controls-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1px;
      background: rgba(0, 0, 0, 0.1);
    }

    .emergency-control {
      background: rgba(255, 255, 255, 0.1);
      padding: 1rem;
      cursor: pointer;
      transition: all 0.2s;
      border: none;
      color: white;
      text-align: left;
    }

    .emergency-control:hover {
      background: rgba(255, 255, 255, 0.2);
    }

    .emergency-control.warning {
      border-left: 4px solid #f59e0b;
    }

    .emergency-control.danger {
      border-left: 4px solid #ef4444;
    }

    .emergency-control.critical {
      border-left: 4px solid #dc2626;
      animation: emergencyControlPulse 2s ease-in-out infinite;
    }

    .control-label {
      font-weight: 600;
      font-size: 0.875rem;
      margin-bottom: 0.25rem;
    }

    .control-description {
      font-size: 0.75rem;
      opacity: 0.9;
      margin-bottom: 0.5rem;
    }

    .control-impact {
      font-size: 0.625rem;
      opacity: 0.7;
      font-style: italic;
    }

    .confirmation-dialog {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
    }

    .dialog-content {
      background: white;
      padding: 2rem;
      border-radius: 0.5rem;
      max-width: 400px;
      width: 90%;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }

    .dialog-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #dc2626;
      margin-bottom: 1rem;
    }

    .dialog-message {
      color: #374151;
      margin-bottom: 1.5rem;
      line-height: 1.5;
    }

    .dialog-actions {
      display: flex;
      gap: 1rem;
      justify-content: flex-end;
    }

    .dialog-button {
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-weight: 500;
      cursor: pointer;
      border: none;
      transition: all 0.2s;
    }

    .dialog-button.cancel {
      background: #f3f4f6;
      color: #374151;
    }

    .dialog-button.cancel:hover {
      background: #e5e7eb;
    }

    .dialog-button.confirm {
      background: #dc2626;
      color: white;
    }

    .dialog-button.confirm:hover {
      background: #b91c1c;
    }

    @keyframes emergencyPulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.8; }
    }

    @keyframes emergencyButtonPulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
    }

    @keyframes emergencyControlPulse {
      0%, 100% { background: rgba(255, 255, 255, 0.1); }
      50% { background: rgba(255, 255, 255, 0.2); }
    }

    @keyframes criticalPulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.02); }
    }

    @keyframes shine {
      0% { left: -100%; }
      100% { left: 100%; }
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
      .alert-header {
        flex-direction: column;
        gap: 1rem;
        align-items: stretch;
      }

      .header-right {
        justify-content: center;
      }

      .heat-map-grid {
        grid-template-columns: 1fr;
      }

      .emergency-controls-grid {
        grid-template-columns: 1fr;
      }

      .emergency-controls-header {
        flex-direction: column;
        gap: 0.5rem;
        text-align: center;
      }
    }

    /* Compact mode */
    :host([compact]) .risk-heat-map,
    :host([compact]) .emergency-controls {
      display: none;
    }

    :host([compact]) .notifications-container {
      max-height: 200px;
    }
  `

  constructor() {
    super()
    this.mobile = false
    this.compact = false
    this.initializeAlertSystem()
  }

  private async initializeAlertSystem() {
    // Set up emergency controls
    this.emergencyControls = [
      {
        id: 'lockdown_system',
        label: 'System Lockdown',
        description: 'Immediately lock down all agent operations and API access',
        action: 'emergency_lockdown',
        severity: 'critical',
        confirmation_required: true,
        estimated_impact: 'All operations suspended for 15-30 minutes'
      },
      {
        id: 'disable_md5',
        label: 'Disable MD5 Usage',
        description: 'Force immediate shutdown of all MD5-using components',
        action: 'disable_md5_components',
        severity: 'danger',
        confirmation_required: true,
        estimated_impact: 'Authentication services may be temporarily disrupted'
      },
      {
        id: 'rotate_keys',
        label: 'Emergency Key Rotation',
        description: 'Rotate all API keys and cryptographic keys immediately',
        action: 'emergency_key_rotation',
        severity: 'warning',
        confirmation_required: true,
        estimated_impact: 'Services will restart automatically, ~5 minutes downtime'
      },
      {
        id: 'isolate_agents',
        label: 'Isolate High-Risk Agents',
        description: 'Quarantine agents with critical security violations',
        action: 'isolate_high_risk_agents',
        severity: 'danger',
        confirmation_required: false,
        estimated_impact: 'Affected agents will be suspended immediately'
      }
    ]

    // Initialize risk heat map with mock data
    this.riskHeatMap = [
      {
        component: 'Authentication Service',
        risk_score: 0.95,
        risk_level: 'critical',
        issues: ['MD5 hash usage', 'Weak password policy'],
        last_updated: new Date().toISOString()
      },
      {
        component: 'API Gateway',
        risk_score: 0.72,
        risk_level: 'high',
        issues: ['Rate limiting gaps', 'Insufficient logging'],
        last_updated: new Date().toISOString()
      },
      {
        component: 'Database Layer',
        risk_score: 0.45,
        risk_level: 'medium',
        issues: ['Outdated encryption', 'Missing backup verification'],
        last_updated: new Date().toISOString()
      },
      {
        component: 'Agent Orchestrator',
        risk_score: 0.23,
        risk_level: 'low',
        issues: ['Minor configuration drift'],
        last_updated: new Date().toISOString()
      },
      {
        component: 'Message Bus',
        risk_score: 0.12,
        risk_level: 'minimal',
        issues: [],
        last_updated: new Date().toISOString()
      }
    ]

    // Generate initial security notifications
    this.generateSecurityNotifications()

    // Set up real-time event listeners
    securityMonitoringService.addEventListener('security_event', this.handleSecurityEvent.bind(this))
    securityMonitoringService.addEventListener('vulnerability_update', this.handleVulnerabilityUpdate.bind(this))
    securityMonitoringService.addEventListener('emergency_mode_activated', this.handleEmergencyMode.bind(this))
  }

  private generateSecurityNotifications() {
    // Critical MD5 usage notification
    const md5Notification: SecurityNotification = {
      id: 'md5-critical-001',
      type: 'critical',
      title: 'Critical: MD5 Hash Algorithm Detected',
      message: 'Deprecated MD5 hash algorithm found in authentication service. This blocks enterprise deployment and poses significant security risks.',
      timestamp: new Date().toISOString(),
      component: 'authentication-service',
      persistent: true,
      actions: [
        {
          label: 'Disable MD5 Components',
          action: 'disable_md5_components',
          severity: 'danger'
        },
        {
          label: 'View Remediation Plan',
          action: 'view_md5_remediation',
          severity: 'primary'
        }
      ],
      metadata: {
        cve_severity: 'CRITICAL',
        enterprise_impact: 'BLOCKS_DEPLOYMENT',
        remediation_time: '2-4_hours'
      }
    }

    // High severity vulnerability notification
    const vulnNotification: SecurityNotification = {
      id: 'vuln-high-001',
      type: 'high',
      title: 'High Risk Vulnerability Detected',
      message: 'Multiple authentication bypass attempts detected from suspicious IP ranges.',
      timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
      component: 'auth-monitor',
      autoClose: 30000, // 30 seconds
      actions: [
        {
          label: 'Block IP Range',
          action: 'block_suspicious_ips',
          severity: 'danger'
        },
        {
          label: 'Investigate',
          action: 'investigate_auth_bypass',
          severity: 'primary'
        }
      ],
      metadata: {
        affected_ips: '5',
        attempt_count: '47',
        success_rate: '0%'
      }
    }

    this.notifications = [md5Notification, vulnNotification]
    this.updateAlertCounts()
  }

  private handleSecurityEvent(data: any) {
    if (data.notification) {
      this.addNotification(data.notification)
    }
  }

  private handleVulnerabilityUpdate(data: any) {
    if (data.critical_count > 0) {
      this.emergencyMode = true
    }
  }

  private handleEmergencyMode(data: any) {
    this.emergencyMode = true
    
    // Add emergency notification
    const emergencyNotification: SecurityNotification = {
      id: `emergency-${Date.now()}`,
      type: 'emergency',
      title: 'SECURITY EMERGENCY ACTIVATED',
      message: 'Critical security issues detected requiring immediate attention. Emergency controls are now available.',
      timestamp: new Date().toISOString(),
      persistent: true,
      actions: [
        {
          label: 'View Emergency Controls',
          action: 'show_emergency_controls',
          severity: 'primary'
        }
      ],
      metadata: data.reasons
    }

    this.addNotification(emergencyNotification)
  }

  private addNotification(notification: SecurityNotification) {
    this.notifications = [notification, ...this.notifications.slice(0, 9)] // Keep max 10
    this.updateAlertCounts()
    this.requestUpdate()
  }

  private updateAlertCounts() {
    this.alertCounts = {
      critical: this.notifications.filter(n => n.type === 'critical' || n.type === 'emergency').length,
      high: this.notifications.filter(n => n.type === 'high').length,
      medium: this.notifications.filter(n => n.type === 'medium').length,
      low: this.notifications.filter(n => n.type === 'low').length
    }
  }

  private handleNotificationAction(event: CustomEvent) {
    const { notification, action } = event.detail
    
    switch (action) {
      case 'show_emergency_controls':
        this.showEmergencyControls = true
        break
      case 'disable_md5_components':
        this.handleEmergencyControl('disable_md5')
        break
      case 'view_md5_remediation':
        this.dispatchEvent(new CustomEvent('show-remediation-plan', {
          detail: { type: 'md5_upgrade' },
          bubbles: true,
          composed: true
        }))
        break
      case 'block_suspicious_ips':
        this.dispatchEvent(new CustomEvent('security-action', {
          detail: { action: 'block_ips', notification },
          bubbles: true,
          composed: true
        }))
        break
      case 'investigate_auth_bypass':
        this.dispatchEvent(new CustomEvent('start-investigation', {
          detail: { type: 'auth_bypass', notification },
          bubbles: true,
          composed: true
        }))
        break
    }
  }

  private handleNotificationClosed(event: CustomEvent) {
    const { notification } = event.detail
    this.notifications = this.notifications.filter(n => n.id !== notification.id)
    this.updateAlertCounts()
  }

  private handleEmergencyControl(controlId: string) {
    const control = this.emergencyControls.find(c => c.id === controlId)
    if (!control) return

    if (control.confirmation_required) {
      this.confirmationDialog = { control }
    } else {
      this.executeEmergencyControl(control)
    }
  }

  private executeEmergencyControl(control: EmergencyControl) {
    console.log(`Executing emergency control: ${control.action}`)
    
    // Dispatch event for parent components to handle
    this.dispatchEvent(new CustomEvent('emergency-control-executed', {
      detail: { control },
      bubbles: true,
      composed: true
    }))

    // Add notification about the action
    const actionNotification: SecurityNotification = {
      id: `action-${Date.now()}`,
      type: 'high',
      title: `Emergency Action: ${control.label}`,
      message: `Emergency control executed: ${control.description}`,
      timestamp: new Date().toISOString(),
      autoClose: 10000,
      metadata: {
        estimated_impact: control.estimated_impact
      }
    }

    this.addNotification(actionNotification)
    this.confirmationDialog = null
  }

  private handleRiskHeatMapClick(component: RiskHeatMapData) {
    this.dispatchEvent(new CustomEvent('risk-component-selected', {
      detail: { component },
      bubbles: true,
      composed: true
    }))
  }

  render() {
    return html`
      <div class="alert-system">
        <div class="alert-header ${this.emergencyMode ? 'emergency' : ''}">
          <div class="header-left">
            <svg class="alert-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M20.618 5.984A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
            </svg>
            <div>
              <div class="alert-title">Security Alert System</div>
              <div class="alert-status">
                ${this.emergencyMode ? 'EMERGENCY MODE ACTIVE' : 'Monitoring Active'}
              </div>
            </div>
          </div>
          
          <div class="header-right">
            <div class="alert-counts">
              ${this.alertCounts.critical > 0 ? html`
                <span class="alert-count critical">Critical: ${this.alertCounts.critical}</span>
              ` : ''}
              ${this.alertCounts.high > 0 ? html`
                <span class="alert-count high">High: ${this.alertCounts.high}</span>
              ` : ''}
              ${this.alertCounts.medium > 0 ? html`
                <span class="alert-count medium">Medium: ${this.alertCounts.medium}</span>
              ` : ''}
              ${this.alertCounts.low > 0 ? html`
                <span class="alert-count low">Low: ${this.alertCounts.low}</span>
              ` : ''}
            </div>
            
            ${this.emergencyMode ? html`
              <button 
                class="emergency-button ${this.showEmergencyControls ? 'active' : ''}"
                @click=${() => this.showEmergencyControls = !this.showEmergencyControls}
              >
                ${this.showEmergencyControls ? 'Hide' : 'Show'} Emergency Controls
              </button>
            ` : ''}
          </div>
        </div>

        <div class="notifications-container">
          ${this.notifications.length > 0 ? html`
            ${this.notifications.map(notification => html`
              <security-alert-notification
                .notification=${notification}
                .compact=${this.compact}
                .mobile=${this.mobile}
                @notification-action=${this.handleNotificationAction}
                @notification-closed=${this.handleNotificationClosed}
              ></security-alert-notification>
            `)}
          ` : html`
            <div class="notifications-empty">
              No active security alerts. System monitoring is active.
            </div>
          `}
        </div>

        ${!this.compact ? html`
          <div class="risk-heat-map">
            <div class="heat-map-header">System Risk Heat Map</div>
            <div class="heat-map-grid">
              ${this.riskHeatMap.map(component => html`
                <div 
                  class="heat-map-cell ${component.risk_level}"
                  @click=${() => this.handleRiskHeatMapClick(component)}
                >
                  <div class="cell-component">${component.component}</div>
                  <div class="cell-risk-score">${Math.round(component.risk_score * 100)}%</div>
                  <div class="cell-risk-level">${component.risk_level} risk</div>
                  ${component.issues.length > 0 ? html`
                    <div class="cell-issues">
                      Issues: ${component.issues.join(', ')}
                    </div>
                  ` : ''}
                </div>
              `)}
            </div>
          </div>
        ` : ''}

        ${this.showEmergencyControls && this.emergencyMode ? html`
          <div class="emergency-controls">
            <div class="emergency-controls-header">
              <div>🚨 Emergency Security Controls</div>
              <button 
                style="background: none; border: none; color: white; cursor: pointer;"
                @click=${() => this.showEmergencyControls = false}
              >
                ✕ Close
              </button>
            </div>
            <div class="emergency-controls-grid">
              ${this.emergencyControls.map(control => html`
                <button 
                  class="emergency-control ${control.severity}"
                  @click=${() => this.handleEmergencyControl(control.id)}
                >
                  <div class="control-label">${control.label}</div>
                  <div class="control-description">${control.description}</div>
                  <div class="control-impact">Impact: ${control.estimated_impact}</div>
                </button>
              `)}
            </div>
          </div>
        ` : ''}
      </div>

      ${this.confirmationDialog ? html`
        <div class="confirmation-dialog">
          <div class="dialog-content">
            <div class="dialog-title">⚠️ Confirm Emergency Action</div>
            <div class="dialog-message">
              <strong>${this.confirmationDialog.control.label}</strong><br/>
              ${this.confirmationDialog.control.description}<br/><br/>
              <em>Estimated Impact: ${this.confirmationDialog.control.estimated_impact}</em><br/><br/>
              Are you sure you want to proceed with this emergency action?
            </div>
            <div class="dialog-actions">
              <button 
                class="dialog-button cancel"
                @click=${() => this.confirmationDialog = null}
              >
                Cancel
              </button>
              <button 
                class="dialog-button confirm"
                @click=${() => this.executeEmergencyControl(this.confirmationDialog!.control)}
              >
                Execute Now
              </button>
            </div>
          </div>
        </div>
      ` : ''}
    `
  }
}