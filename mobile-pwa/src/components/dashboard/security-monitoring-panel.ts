/**
 * Security Monitoring Panel
 * 
 * Real-time security dashboard with threat detection, monitoring, and alerts
 * Priority: Critical - Essential for enterprise security oversight
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { securityMonitoringService } from '../../services/security-monitoring'

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

export interface VulnerabilityScan {
  scan_id: string
  scan_type: 'infrastructure' | 'application' | 'dependency' | 'configuration'
  started_at: string
  completed_at: string
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  vulnerabilities_found: number
  critical_count: number
  high_count: number
  medium_count: number
  low_count: number
  remediation_recommendations: string[]
  baseline_comparison?: {
    previous_scan_date: string
    vulnerabilities_new: number
    vulnerabilities_fixed: number
    trend: 'improving' | 'degrading' | 'stable'
  }
}

export interface VulnerabilityItem {
  vulnerability_id: string
  cve_id?: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  title: string
  description: string
  affected_component: string
  cvss_score: number
  age_days: number
  sla_compliant: boolean
  remediation_status: 'open' | 'in_progress' | 'resolved' | 'accepted_risk'
  remediation_timeline?: string
  patch_available: boolean
  exploit_available: boolean
}

export interface CryptographicHealth {
  algorithms_in_use: {
    algorithm: string
    strength: 'strong' | 'weak' | 'deprecated'
    usage_count: number
    replacement_recommendation?: string
  }[]
  certificate_status: {
    total_certificates: number
    expiring_30_days: number
    expiring_90_days: number
    expired: number
  }
  key_rotation_status: {
    total_keys: number
    rotation_compliant: number
    overdue_rotation: number
    last_rotation_check: string
  }
  encryption_compliance: {
    overall_score: number
    tls_version_compliance: boolean
    cipher_strength_compliance: boolean
    key_length_compliance: boolean
  }
  detected_issues: {
    md5_usage_count: number
    weak_ciphers: string[]
    short_key_lengths: number
    deprecated_protocols: string[]
  }
}

export interface ComplianceStatus {
  framework: string
  overall_score: number
  status: 'compliant' | 'non_compliant' | 'partial'
  last_assessment: string
  next_assessment: string
  controls: {
    total: number
    compliant: number
    non_compliant: number
    not_assessed: number
  }
  active_violations: {
    violation_id: string
    control_id: string
    severity: 'critical' | 'high' | 'medium' | 'low'
    description: string
    deadline: string
    status: 'open' | 'remediation_in_progress' | 'pending_review'
  }[]
  audit_readiness: {
    score: number
    missing_evidence: string[]
    documentation_gaps: string[]
    recommendations: string[]
  }
}

export interface SecurityMetrics {
  threat_detection: {
    active_threats: number
    resolved_today: number
    false_positives: number
    threat_level: 'minimal' | 'elevated' | 'high' | 'critical'
  }
  vulnerability_tracking: {
    total_vulnerabilities: number
    critical_count: number
    high_count: number
    medium_count: number
    low_count: number
    trend_7d: 'improving' | 'degrading' | 'stable'
    sla_compliant_percentage: number
    avg_remediation_time_days: number
    overdue_count: number
  }
  authentication: {
    successful_logins: number
    failed_attempts: number
    suspicious_logins: number
    active_sessions: number
    mfa_compliance_rate: number
    anomaly_detection_alerts: number
    geographic_anomalies: number
  }
  access_control: {
    permission_violations: number
    unauthorized_access_attempts: number
    privilege_escalations: number
    data_access_anomalies: number
    api_key_violations: number
    session_violations: number
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
  cryptographic_health: CryptographicHealth
  compliance_summary: {
    frameworks_tracked: number
    overall_compliance_score: number
    frameworks_compliant: number
    active_violations_total: number
    audit_readiness_score: number
  }
  security_scans: {
    last_scan_date: string
    scans_completed_today: number
    failed_scans: number
    scan_coverage_percentage: number
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
  @state() private vulnerabilityData: VulnerabilityItem[] = []
  @state() private securityScans: VulnerabilityScan[] = []
  @state() private complianceData: ComplianceStatus[] = []
  @state() private scanHistoryVisible: boolean = false
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
      flex-wrap: wrap;
    }
    
    .md5-warning {
      background: #dc2626;
      color: white;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 700;
      animation: criticalPulse 1.5s infinite;
    }
    
    .critical-vuln-warning {
      background: #b91c1c;
      color: white;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 700;
      animation: criticalPulse 1.5s infinite;
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
    
    /* New styles for enhanced security dashboard */
    .vulnerability-list {
      margin-top: 2rem;
    }
    
    .vulnerability-item {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      margin-bottom: 0.5rem;
      transition: all 0.2s;
    }
    
    .vulnerability-item:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .vulnerability-item.critical {
      border-left: 4px solid #dc2626;
      background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    .vulnerability-item.high {
      border-left: 4px solid #f59e0b;
      background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .vulnerability-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    
    .vulnerability-title {
      font-weight: 600;
      color: #374151;
    }
    
    .vulnerability-severity {
      padding: 0.125rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    
    .vulnerability-severity.critical {
      background: #fecaca;
      color: #dc2626;
    }
    
    .vulnerability-severity.high {
      background: #fed7aa;
      color: #d97706;
    }
    
    .vulnerability-cvss {
      font-size: 0.75rem;
      color: #6b7280;
      font-weight: 500;
    }
    
    .vulnerability-details {
      display: flex;
      gap: 1rem;
      font-size: 0.75rem;
      color: #6b7280;
    }
    
    .sla-status.compliant {
      color: #059669;
    }
    
    .sla-status.overdue {
      color: #dc2626;
      font-weight: 600;
    }
    
    .scan-history {
      margin-top: 2rem;
      border-top: 1px solid #e5e7eb;
      padding-top: 1rem;
    }
    
    .scan-history-toggle {
      background: none;
      border: none;
      cursor: pointer;
      padding: 0;
    }
    
    .scan-item {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      margin-bottom: 0.5rem;
    }
    
    .scan-item.completed {
      border-left: 4px solid #10b981;
    }
    
    .scan-item.failed {
      border-left: 4px solid #ef4444;
    }
    
    .scan-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    
    .scan-type {
      font-weight: 600;
      color: #374151;
    }
    
    .scan-status {
      padding: 0.125rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    
    .scan-status.completed {
      background: #d1fae5;
      color: #065f46;
    }
    
    .scan-status.failed {
      background: #fecaca;
      color: #dc2626;
    }
    
    .scan-results {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    
    .vulnerability-counts {
      display: flex;
      gap: 1rem;
      font-size: 0.75rem;
    }
    
    .vulnerability-counts .count {
      font-weight: 500;
    }
    
    .vulnerability-counts .count.critical {
      color: #dc2626;
    }
    
    .vulnerability-counts .count.high {
      color: #d97706;
    }
    
    .vulnerability-counts .count.medium {
      color: #2563eb;
    }
    
    .vulnerability-counts .count.low {
      color: #6b7280;
    }
    
    .baseline-comparison .trend {
      font-size: 0.75rem;
      font-weight: 500;
    }
    
    .baseline-comparison .trend.improving {
      color: #059669;
    }
    
    .baseline-comparison .trend.degrading {
      color: #dc2626;
    }
    
    .crypto-issues {
      margin-top: 2rem;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 0.5rem;
      padding: 1rem;
    }
    
    .issue-category {
      margin-bottom: 0.5rem;
      font-size: 0.875rem;
      color: #991b1b;
    }
    
    .compliance-frameworks {
      margin-top: 2rem;
    }
    
    .framework-item {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      margin-bottom: 0.5rem;
    }
    
    .framework-item.compliant {
      border-left: 4px solid #10b981;
    }
    
    .framework-item.non_compliant {
      border-left: 4px solid #ef4444;
    }
    
    .framework-item.partial {
      border-left: 4px solid #f59e0b;
    }
    
    .framework-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    
    .framework-name {
      font-weight: 600;
      color: #374151;
    }
    
    .framework-score {
      font-weight: 600;
      color: #059669;
    }
    
    .framework-status {
      padding: 0.125rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    
    .framework-status.compliant {
      background: #d1fae5;
      color: #065f46;
    }
    
    .framework-status.non_compliant {
      background: #fecaca;
      color: #dc2626;
    }
    
    .framework-status.partial {
      background: #fed7aa;
      color: #d97706;
    }
    
    .framework-details {
      display: flex;
      gap: 1rem;
      font-size: 0.75rem;
      color: #6b7280;
    }
    
    .vulnerability-action {
      margin-top: 0.5rem;
      text-align: right;
    }
    
    .remediation-button {
      background: #dc2626;
      color: white;
      border: none;
      padding: 0.25rem 0.75rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .remediation-button:hover {
      background: #b91c1c;
      transform: scale(1.05);
    }
    
    .framework-violations {
      margin-top: 0.75rem;
      padding-top: 0.75rem;
      border-top: 1px solid #e5e7eb;
      font-size: 0.75rem;
    }
    
    .violation-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 0.25rem 0;
      padding: 0.25rem;
      background: rgba(239, 68, 68, 0.1);
      border-radius: 0.25rem;
    }
    
    .violation-severity {
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-weight: 600;
      font-size: 0.625rem;
    }
    
    .violation-severity.critical {
      background: #fecaca;
      color: #dc2626;
    }
    
    .violation-severity.high {
      background: #fed7aa;
      color: #d97706;
    }
    
    .violation-severity.medium {
      background: #bfdbfe;
      color: #2563eb;
    }
    
    .violation-description {
      flex: 1;
      margin: 0 0.5rem;
      color: #374151;
    }
    
    .violation-deadline {
      color: #6b7280;
      font-size: 0.625rem;
    }
    
    .more-violations {
      text-align: center;
      color: #6b7280;
      font-style: italic;
      margin-top: 0.25rem;
    }
    
    .vulnerability-item {
      cursor: pointer;
    }
    
    .framework-item {
      cursor: pointer;
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
      
      .security-tabs {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
      }
      
      .vulnerability-header,
      .scan-header,
      .framework-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
      }
      
      .vulnerability-details,
      .vulnerability-counts,
      .framework-details {
        flex-direction: column;
        gap: 0.25rem;
      }
    }
  `
  
  constructor() {
    super()
    this.metrics = null
    this.alerts = []
    this.realtime = true
    this.compact = false
    this.vulnerabilityData = []
    this.securityScans = []
    this.complianceData = []
    this.scanHistoryVisible = false
    
    // Initialize service connections
    this.initializeSecurityMonitoring()
  }

  private async initializeSecurityMonitoring() {
    try {
      // Set up event listeners for real-time updates
      securityMonitoringService.addEventListener('security_event', this.handleSecurityEvent.bind(this))
      securityMonitoringService.addEventListener('vulnerability_update', this.handleVulnerabilityUpdate.bind(this))
      securityMonitoringService.addEventListener('compliance_alert', this.handleComplianceAlert.bind(this))
      securityMonitoringService.addEventListener('connection_status', this.handleConnectionStatus.bind(this))
      
      // Connect WebSocket for real-time updates
      if (this.realtime) {
        await securityMonitoringService.connectWebSocket()
      }
      
      // Load initial data
      await this.loadSecurityData()
    } catch (error) {
      console.error('Failed to initialize security monitoring:', error)
      // Continue with offline/mock data
      this.loadMockData()
    }
  }

  private async loadSecurityData() {
    try {
      const [metrics, vulnerabilities, scans, compliance] = await Promise.all([
        securityMonitoringService.getSecurityOverview(),
        securityMonitoringService.getVulnerabilities(),
        securityMonitoringService.getSecurityScans(),
        securityMonitoringService.getComplianceStatus()
      ])
      
      this.metrics = metrics
      this.vulnerabilityData = vulnerabilities
      this.securityScans = scans
      this.complianceData = compliance
      
      this.requestUpdate()
    } catch (error) {
      console.error('Failed to load security data:', error)
      this.loadMockData()
    }
  }

  private loadMockData() {
    // Load comprehensive mock data for development
    this.metrics = {
      threat_detection: {
        active_threats: 5,
        resolved_today: 12,
        false_positives: 3,
        threat_level: 'elevated'
      },
      vulnerability_tracking: {
        total_vulnerabilities: 23,
        critical_count: 2,
        high_count: 6,
        medium_count: 8,
        low_count: 7,
        trend_7d: 'improving',
        sla_compliant_percentage: 87,
        avg_remediation_time_days: 5.2,
        overdue_count: 3
      },
      authentication: {
        successful_logins: 245,
        failed_attempts: 12,
        suspicious_logins: 2,
        active_sessions: 34,
        mfa_compliance_rate: 94.5,
        anomaly_detection_alerts: 3,
        geographic_anomalies: 1
      },
      access_control: {
        permission_violations: 7,
        unauthorized_access_attempts: 4,
        privilege_escalations: 1,
        data_access_anomalies: 2,
        api_key_violations: 3,
        session_violations: 1
      },
      network_security: {
        blocked_connections: 156,
        malicious_requests: 23,
        rate_limit_violations: 45,
        ddos_attempts: 0
      },
      data_protection: {
        encryption_status: 'healthy',
        backup_status: 'current',
        data_integrity_score: 0.98,
        compliance_violations: 2
      },
      system_security: {
        vulnerability_score: 0.23,
        patch_compliance: 0.95,
        security_updates_pending: 3,
        configuration_drift: 2
      },
      cryptographic_health: {
        algorithms_in_use: [
          {
            algorithm: 'SHA-256',
            strength: 'strong',
            usage_count: 45,
          },
          {
            algorithm: 'MD5',
            strength: 'deprecated',
            usage_count: 2,
            replacement_recommendation: 'Upgrade to SHA-256'
          }
        ],
        certificate_status: {
          total_certificates: 12,
          expiring_30_days: 1,
          expiring_90_days: 3,
          expired: 0
        },
        key_rotation_status: {
          total_keys: 24,
          rotation_compliant: 22,
          overdue_rotation: 2,
          last_rotation_check: new Date().toISOString()
        },
        encryption_compliance: {
          overall_score: 0.92,
          tls_version_compliance: true,
          cipher_strength_compliance: true,
          key_length_compliance: false
        },
        detected_issues: {
          md5_usage_count: 2,
          weak_ciphers: ['DES', '3DES'],
          short_key_lengths: 3,
          deprecated_protocols: ['SSLv3', 'TLSv1.0']
        }
      },
      compliance_summary: {
        frameworks_tracked: 4,
        overall_compliance_score: 0.94,
        frameworks_compliant: 3,
        active_violations_total: 6,
        audit_readiness_score: 0.89
      },
      security_scans: {
        last_scan_date: new Date().toISOString(),
        scans_completed_today: 3,
        failed_scans: 0,
        scan_coverage_percentage: 97.5
      },
      timestamp: new Date().toISOString()
    }
    
    // Mock alerts
    this.alerts = [
      {
        id: 'alert-001',
        type: 'intrusion',
        severity: 'critical',
        title: 'MD5 Hash Usage Detected',
        message: 'Deprecated MD5 hash algorithm found in authentication service - immediate upgrade to SHA-256 required',
        source: 'cryptographic_scanner',
        timestamp: new Date().toISOString(),
        status: 'active',
        affected_agents: ['auth-service-1', 'auth-service-2'],
        metadata: {
          component: 'authentication-service',
          remediation_priority: 'critical',
          enterprise_impact: 'blocks_deployment'
        }
      },
      {
        id: 'alert-002',
        type: 'authentication',
        severity: 'high',
        title: 'Multiple Failed Login Attempts',
        message: 'Suspicious authentication pattern detected from IP 192.168.1.100',
        source: 'auth_monitor',
        timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
        status: 'investigating',
        metadata: {
          source_ip: '192.168.1.100',
          attempt_count: 15,
          user_accounts_targeted: 5
        }
      }
    ]
    
    this.requestUpdate()
  }

  private handleSecurityEvent(data: any) {
    console.log('Security event received:', data)
    
    // Add new alert if it's a security event
    if (data.alert) {
      this.alerts = [data.alert, ...this.alerts.slice(0, 9)] // Keep only last 10
    }
    
    // Update metrics if provided
    if (data.metrics) {
      this.metrics = { ...this.metrics, ...data.metrics }
    }
    
    this.requestUpdate()
  }

  private handleVulnerabilityUpdate(data: any) {
    console.log('Vulnerability update received:', data)
    
    if (data.vulnerabilities) {
      this.vulnerabilityData = data.vulnerabilities
    }
    
    if (data.metrics?.vulnerability_tracking) {
      this.metrics = {
        ...this.metrics,
        vulnerability_tracking: data.metrics.vulnerability_tracking
      }
    }
    
    this.requestUpdate()
  }

  private handleComplianceAlert(data: any) {
    console.log('Compliance alert received:', data)
    
    if (data.frameworks) {
      this.complianceData = data.frameworks
    }
    
    if (data.metrics?.compliance_summary) {
      this.metrics = {
        ...this.metrics,
        compliance_summary: data.metrics.compliance_summary
      }
    }
    
    this.requestUpdate()
  }

  private handleConnectionStatus(data: any) {
    console.log('Connection status changed:', data)
    
    if (!data.connected && this.realtime) {
      // Show connection warning
      this.dispatchEvent(new CustomEvent('connection-warning', {
        detail: { message: 'Real-time updates disconnected' },
        bubbles: true,
        composed: true
      }))
    }
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
  
  private renderVulnerabilityTracking() {
    if (!this.metrics?.vulnerability_tracking) return this.renderLoadingState()
    
    const { vulnerability_tracking } = this.metrics
    const trendColor = vulnerability_tracking.trend_7d === 'improving' ? '#10b981' : 
                      vulnerability_tracking.trend_7d === 'degrading' ? '#ef4444' : '#f59e0b'
    const slaCompliant = vulnerability_tracking.sla_compliant_percentage >= 95
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card ${vulnerability_tracking.critical_count > 0 ? 'critical' : vulnerability_tracking.high_count > 10 ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Total Vulnerabilities</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
            </svg>
          </div>
          <div class="metric-value ${vulnerability_tracking.critical_count > 0 ? 'critical' : vulnerability_tracking.high_count > 10 ? 'high' : 'healthy'}">
            ${vulnerability_tracking.total_vulnerabilities}
          </div>
          <div class="metric-description">
            Critical: ${vulnerability_tracking.critical_count} | High: ${vulnerability_tracking.high_count} | Medium: ${vulnerability_tracking.medium_count} | Low: ${vulnerability_tracking.low_count}
          </div>
        </div>
        
        <div class="metric-card ${vulnerability_tracking.trend_7d === 'degrading' ? 'high' : vulnerability_tracking.trend_7d === 'improving' ? 'healthy' : 'neutral'}">
          <div class="metric-header">
            <div class="metric-label">7-Day Trend</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              ${vulnerability_tracking.trend_7d === 'improving' ? 
                html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>` :
                vulnerability_tracking.trend_7d === 'degrading' ?
                html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"/>` :
                html`<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>`
              }
            </svg>
          </div>
          <div class="metric-value" style="color: ${trendColor}">
            ${vulnerability_tracking.trend_7d.toUpperCase()}
          </div>
          <div class="metric-description">
            Vulnerability trend analysis over past week
          </div>
        </div>
        
        <div class="metric-card ${!slaCompliant ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">SLA Compliance</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <div class="metric-value ${!slaCompliant ? 'high' : 'healthy'}">
            ${Math.round(vulnerability_tracking.sla_compliant_percentage)}%
          </div>
          <div class="metric-description">
            Avg Resolution: ${vulnerability_tracking.avg_remediation_time_days}d | Overdue: ${vulnerability_tracking.overdue_count}
          </div>
        </div>
      </div>
      
      ${this.vulnerabilityData.length > 0 ? html`
        <div class="vulnerability-list">
          <h4>Active Vulnerabilities</h4>
          ${this.vulnerabilityData.slice(0, 5).map(vuln => html`
            <div class="vulnerability-item ${vuln.severity}" @click=${() => this.handleVulnerabilityClick(vuln)}>
              <div class="vulnerability-header">
                <span class="vulnerability-title">${vuln.title}</span>
                <span class="vulnerability-severity ${vuln.severity}">${vuln.severity.toUpperCase()}</span>
                <span class="vulnerability-cvss">CVSS: ${vuln.cvss_score}</span>
              </div>
              <div class="vulnerability-details">
                <span>Component: ${vuln.affected_component}</span>
                <span>Age: ${vuln.age_days} days</span>
                <span class="sla-status ${vuln.sla_compliant ? 'compliant' : 'overdue'}">
                  ${vuln.sla_compliant ? '✓ SLA Compliant' : '⚠ SLA Overdue'}
                </span>
              </div>
              ${vuln.remediation_status === 'open' && vuln.severity === 'critical' ? html`
                <div class="vulnerability-action">
                  <button class="remediation-button" @click=${(e: Event) => {
                    e.stopPropagation()
                    this.handleVulnerabilityRemediation(vuln)
                  }}>
                    Start Remediation
                  </button>
                </div>
              ` : ''}
            </div>
          `)}
        </div>
      ` : ''}
    `
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
  
  private renderSecurityScans() {
    if (!this.metrics?.security_scans) return this.renderLoadingState()
    
    const { security_scans } = this.metrics
    const lastScanDate = new Date(security_scans.last_scan_date)
    const daysSinceLastScan = Math.floor((Date.now() - lastScanDate.getTime()) / (1000 * 60 * 60 * 24))
    const scanHealthy = daysSinceLastScan <= 1 && security_scans.failed_scans === 0
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card ${!scanHealthy ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Last Security Scan</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <div class="metric-value ${!scanHealthy ? 'high' : 'healthy'}">
            ${daysSinceLastScan === 0 ? 'Today' : `${daysSinceLastScan}d ago`}
          </div>
          <div class="metric-description">
            Coverage: ${security_scans.scan_coverage_percentage}% | Completed today: ${security_scans.scans_completed_today}
          </div>
        </div>
        
        <div class="metric-card ${security_scans.failed_scans > 0 ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Failed Scans</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
            </svg>
          </div>
          <div class="metric-value ${security_scans.failed_scans > 0 ? 'high' : 'healthy'}">
            ${security_scans.failed_scans}
          </div>
          <div class="metric-description">
            Failed security scans requiring attention
          </div>
        </div>
        
        <div class="metric-card">
          <div class="metric-header">
            <div class="metric-label">Scan History</div>
            <button class="scan-history-toggle" @click=${this.toggleScanHistory}>
              <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
              </svg>
            </button>
          </div>
          <div class="metric-value">
            View Details
          </div>
          <div class="metric-description">
            Historical scan results and trends
          </div>
        </div>
      </div>
      
      ${this.scanHistoryVisible && this.securityScans.length > 0 ? html`
        <div class="scan-history">
          <h4>Recent Security Scans</h4>
          ${this.securityScans.slice(0, 5).map(scan => html`
            <div class="scan-item ${scan.status}">
              <div class="scan-header">
                <span class="scan-type">${scan.scan_type.toUpperCase()}</span>
                <span class="scan-status ${scan.status}">${scan.status.toUpperCase()}</span>
                <span class="scan-date">${new Date(scan.started_at).toLocaleDateString()}</span>
              </div>
              <div class="scan-results">
                <div class="vulnerability-counts">
                  <span class="count critical">Critical: ${scan.critical_count}</span>
                  <span class="count high">High: ${scan.high_count}</span>
                  <span class="count medium">Medium: ${scan.medium_count}</span>
                  <span class="count low">Low: ${scan.low_count}</span>
                </div>
                ${scan.baseline_comparison ? html`
                  <div class="baseline-comparison">
                    <span class="trend ${scan.baseline_comparison.trend}">
                      ${scan.baseline_comparison.trend === 'improving' ? '↓' : scan.baseline_comparison.trend === 'degrading' ? '↑' : '→'}
                      New: ${scan.baseline_comparison.vulnerabilities_new} | Fixed: ${scan.baseline_comparison.vulnerabilities_fixed}
                    </span>
                  </div>
                ` : ''}
              </div>
            </div>
          `)}
        </div>
      ` : ''}
    `
  }

  private renderCryptographicHealth() {
    if (!this.metrics?.cryptographic_health) return this.renderLoadingState()
    
    const { cryptographic_health } = this.metrics
    const md5Issues = cryptographic_health.detected_issues.md5_usage_count > 0
    const certIssues = cryptographic_health.certificate_status.expired > 0 || 
                      cryptographic_health.certificate_status.expiring_30_days > 0
    const keyRotationIssues = cryptographic_health.key_rotation_status.overdue_rotation > 0
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card ${md5Issues ? 'critical' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">MD5 Usage Detection</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
            </svg>
          </div>
          <div class="metric-value ${md5Issues ? 'critical' : 'healthy'}">
            ${cryptographic_health.detected_issues.md5_usage_count}
          </div>
          <div class="metric-description">
            ${md5Issues ? '⚠ MD5 usage detected - requires SHA-256 upgrade' : '✓ No deprecated MD5 usage found'}
          </div>
        </div>
        
        <div class="metric-card ${certIssues ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Certificate Health</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
            </svg>
          </div>
          <div class="metric-value ${certIssues ? 'high' : 'healthy'}">
            ${cryptographic_health.certificate_status.expired + cryptographic_health.certificate_status.expiring_30_days}
          </div>
          <div class="metric-description">
            Expired: ${cryptographic_health.certificate_status.expired} | Expiring (30d): ${cryptographic_health.certificate_status.expiring_30_days}
          </div>
        </div>
        
        <div class="metric-card ${keyRotationIssues ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Key Rotation</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
            </svg>
          </div>
          <div class="metric-value ${keyRotationIssues ? 'high' : 'healthy'}">
            ${cryptographic_health.key_rotation_status.rotation_compliant}/${cryptographic_health.key_rotation_status.total_keys}
          </div>
          <div class="metric-description">
            Compliant keys | Overdue: ${cryptographic_health.key_rotation_status.overdue_rotation}
          </div>
        </div>
        
        <div class="metric-card ${cryptographic_health.encryption_compliance.overall_score < 0.9 ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Encryption Compliance</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
            </svg>
          </div>
          <div class="metric-value ${cryptographic_health.encryption_compliance.overall_score < 0.9 ? 'high' : 'healthy'}">
            ${Math.round(cryptographic_health.encryption_compliance.overall_score * 100)}%
          </div>
          <div class="metric-description">
            Overall encryption standard compliance score
          </div>
        </div>
      </div>
      
      ${cryptographic_health.detected_issues.weak_ciphers.length > 0 || 
        cryptographic_health.detected_issues.deprecated_protocols.length > 0 ? html`
        <div class="crypto-issues">
          <h4>⚠ Cryptographic Issues Detected</h4>
          ${cryptographic_health.detected_issues.weak_ciphers.length > 0 ? html`
            <div class="issue-category">
              <strong>Weak Ciphers:</strong> ${cryptographic_health.detected_issues.weak_ciphers.join(', ')}
            </div>
          ` : ''}
          ${cryptographic_health.detected_issues.deprecated_protocols.length > 0 ? html`
            <div class="issue-category">
              <strong>Deprecated Protocols:</strong> ${cryptographic_health.detected_issues.deprecated_protocols.join(', ')}
            </div>
          ` : ''}
          ${cryptographic_health.detected_issues.short_key_lengths > 0 ? html`
            <div class="issue-category">
              <strong>Short Key Lengths:</strong> ${cryptographic_health.detected_issues.short_key_lengths} instances
            </div>
          ` : ''}
        </div>
      ` : ''}
    `
  }

  private renderComplianceStatus() {
    if (!this.metrics?.compliance_summary) return this.renderLoadingState()
    
    const { compliance_summary } = this.metrics
    const overallHealthy = compliance_summary.overall_compliance_score >= 0.9 && 
                          compliance_summary.active_violations_total === 0
    
    return html`
      <div class="metrics-grid">
        <div class="metric-card ${!overallHealthy ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Overall Compliance</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <div class="metric-value ${!overallHealthy ? 'high' : 'healthy'}">
            ${Math.round(compliance_summary.overall_compliance_score * 100)}%
          </div>
          <div class="metric-description">
            ${compliance_summary.frameworks_compliant}/${compliance_summary.frameworks_tracked} frameworks compliant
          </div>
        </div>
        
        <div class="metric-card ${compliance_summary.active_violations_total > 0 ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Active Violations</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
            </svg>
          </div>
          <div class="metric-value ${compliance_summary.active_violations_total > 0 ? 'high' : 'healthy'}">
            ${compliance_summary.active_violations_total}
          </div>
          <div class="metric-description">
            Compliance violations requiring remediation
          </div>
        </div>
        
        <div class="metric-card ${compliance_summary.audit_readiness_score < 0.85 ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Audit Readiness</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"/>
            </svg>
          </div>
          <div class="metric-value ${compliance_summary.audit_readiness_score < 0.85 ? 'high' : 'healthy'}">
            ${Math.round(compliance_summary.audit_readiness_score * 100)}%
          </div>
          <div class="metric-description">
            Readiness for compliance audits
          </div>
        </div>
      </div>
      
      ${this.complianceData.length > 0 ? html`
        <div class="compliance-frameworks">
          <h4>Compliance Framework Status</h4>
          ${this.complianceData.map(framework => html`
            <div class="framework-item ${framework.status}" @click=${() => this.handleComplianceClick(framework)}>
              <div class="framework-header">
                <span class="framework-name">${framework.framework.toUpperCase()}</span>
                <span class="framework-score">${Math.round(framework.overall_score * 100)}%</span>
                <span class="framework-status ${framework.status}">${framework.status.replace('_', ' ').toUpperCase()}</span>
              </div>
              <div class="framework-details">
                <span>Controls: ${framework.controls.compliant}/${framework.controls.total}</span>
                <span>Violations: ${framework.active_violations.length}</span>
                <span>Next Assessment: ${new Date(framework.next_assessment).toLocaleDateString()}</span>
              </div>
              ${framework.active_violations.length > 0 ? html`
                <div class="framework-violations">
                  <strong>Active Violations:</strong>
                  ${framework.active_violations.slice(0, 2).map(violation => html`
                    <div class="violation-item">
                      <span class="violation-severity ${violation.severity}">${violation.severity.toUpperCase()}</span>
                      <span class="violation-description">${violation.description}</span>
                      <span class="violation-deadline">Due: ${new Date(violation.deadline).toLocaleDateString()}</span>
                    </div>
                  `)}
                  ${framework.active_violations.length > 2 ? html`
                    <div class="more-violations">+${framework.active_violations.length - 2} more violations</div>
                  ` : ''}
                </div>
              ` : ''}
            </div>
          `)}
        </div>
      ` : ''}
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
        
        <div class="metric-card ${authentication.anomaly_detection_alerts > 0 ? 'high' : 'healthy'}">
          <div class="metric-header">
            <div class="metric-label">Anomaly Alerts</div>
            <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
            </svg>
          </div>
          <div class="metric-value ${authentication.anomaly_detection_alerts > 0 ? 'high' : 'healthy'}">
            ${authentication.anomaly_detection_alerts}
          </div>
          <div class="metric-description">
            Authentication anomalies detected | Geographic: ${authentication.geographic_anomalies}
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
      case 'vulnerabilities':
        return this.renderVulnerabilityTracking()
      case 'scans':
        return this.renderSecurityScans()
      case 'crypto':
        return this.renderCryptographicHealth()
      case 'compliance':
        return this.renderComplianceStatus()
      case 'threats':
        return this.renderThreatDetection()
      case 'auth':
        return this.renderAuthentication()
      case 'alerts':
        return this.renderAlerts()
      default:
        return html`
          ${this.renderVulnerabilityTracking()}
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

  private toggleScanHistory() {
    this.scanHistoryVisible = !this.scanHistoryVisible
    this.requestUpdate()
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
    
    if (this.autoRefresh) {
      // Re-enable real-time updates
      securityMonitoringService.connectWebSocket().catch(error => {
        console.error('Failed to reconnect WebSocket:', error)
      })
    } else {
      // Disable real-time updates
      securityMonitoringService.disconnectWebSocket()
    }
    
    this.dispatchEvent(new CustomEvent('auto-refresh-toggled', {
      detail: { enabled: this.autoRefresh },
      bubbles: true,
      composed: true
    }))
  }

  /**
   * Refresh security data manually
   */
  async refreshSecurityData() {
    try {
      await this.loadSecurityData()
      this.lastUpdate = new Date()
      
      this.dispatchEvent(new CustomEvent('data-refreshed', {
        detail: { timestamp: this.lastUpdate },
        bubbles: true,
        composed: true
      }))
    } catch (error) {
      console.error('Failed to refresh security data:', error)
      
      this.dispatchEvent(new CustomEvent('refresh-error', {
        detail: { error: error.message },
        bubbles: true,
        composed: true
      }))
    }
  }

  /**
   * Handle vulnerability item click for detailed investigation
   */
  private handleVulnerabilityClick(vulnerability: VulnerabilityItem) {
    this.dispatchEvent(new CustomEvent('vulnerability-selected', {
      detail: { vulnerability },
      bubbles: true,
      composed: true
    }))
  }

  /**
   * Handle compliance framework click for detailed view
   */
  private handleComplianceClick(framework: ComplianceStatus) {
    this.dispatchEvent(new CustomEvent('compliance-framework-selected', {
      detail: { framework },
      bubbles: true,
      composed: true
    }))
  }
  
  updated(changedProperties: Map<string, any>) {
    if ((changedProperties.has('metrics') || changedProperties.has('alerts')) && (this.metrics || this.alerts.length)) {
      this.lastUpdate = new Date()
      
      // Check if we should enter emergency mode
      const criticalThreats = this.criticalAlertCount
      const criticalVulnerabilities = this.metrics?.vulnerability_tracking?.critical_count || 0
      const md5Usage = this.metrics?.cryptographic_health?.detected_issues?.md5_usage_count || 0
      
      if ((criticalThreats > 0 || criticalVulnerabilities > 0 || md5Usage > 0) && !this.emergencyMode) {
        this.emergencyMode = true
        
        // Broadcast emergency alert
        this.dispatchEvent(new CustomEvent('emergency-mode-activated', {
          detail: {
            reasons: {
              criticalThreats,
              criticalVulnerabilities,
              md5Usage
            }
          },
          bubbles: true,
          composed: true
        }))
      }
    }
  }

  /**
   * Handle vulnerability remediation action
   */
  private async handleVulnerabilityRemediation(vulnerability: VulnerabilityItem) {
    try {
      // In production, would initiate remediation workflow
      console.log('Starting remediation for vulnerability:', vulnerability.vulnerability_id)
      
      this.dispatchEvent(new CustomEvent('remediation-started', {
        detail: { vulnerability },
        bubbles: true,
        composed: true
      }))
    } catch (error) {
      console.error('Failed to start remediation:', error)
    }
  }

  /**
   * Cleanup WebSocket connections when component is removed
   */
  disconnectedCallback() {
    super.disconnectedCallback()
    
    // Remove event listeners
    securityMonitoringService.removeEventListener('security_event', this.handleSecurityEvent)
    securityMonitoringService.removeEventListener('vulnerability_update', this.handleVulnerabilityUpdate)
    securityMonitoringService.removeEventListener('compliance_alert', this.handleComplianceAlert)
    securityMonitoringService.removeEventListener('connection_status', this.handleConnectionStatus)
    
    // Disconnect WebSocket
    securityMonitoringService.disconnectWebSocket()
  }
  
  render() {
    const threatLevel = this.metrics?.threat_detection?.threat_level || 'minimal'
    const md5Issues = this.metrics?.cryptographic_health?.detected_issues?.md5_usage_count || 0
    const criticalVulns = this.metrics?.vulnerability_tracking?.critical_count || 0
    
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
          ${md5Issues > 0 ? html`<span class="md5-warning">⚠ MD5 DETECTED</span>` : ''}
          ${criticalVulns > 0 ? html`<span class="critical-vuln-warning">${criticalVulns} CRITICAL</span>` : ''}
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
            title="${this.autoRefresh ? 'Disable' : 'Enable'} real-time updates"
          >
            ${this.autoRefresh ? '🔴 LIVE' : '⏸️ OFFLINE'}
          </button>
          <button 
            class="control-button"
            @click=${this.refreshSecurityData}
            title="Refresh security data"
          >
            🔄
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
            class="tab-button ${this.selectedView === 'vulnerabilities' ? 'active' : ''}"
            @click=${() => this.handleTabChange('vulnerabilities')}
          >
            Vulnerabilities
            ${this.metrics?.vulnerability_tracking?.critical_count > 0 ? html`
              <span class="alert-count">${this.metrics.vulnerability_tracking.critical_count}</span>
            ` : ''}
          </button>
          <button 
            class="tab-button ${this.selectedView === 'scans' ? 'active' : ''}"
            @click=${() => this.handleTabChange('scans')}
          >
            Security Scans
          </button>
          <button 
            class="tab-button ${this.selectedView === 'crypto' ? 'active' : ''}"
            @click=${() => this.handleTabChange('crypto')}
          >
            Cryptographic
            ${this.metrics?.cryptographic_health?.detected_issues?.md5_usage_count > 0 ? html`
              <span class="alert-count">MD5</span>
            ` : ''}
          </button>
          <button 
            class="tab-button ${this.selectedView === 'compliance' ? 'active' : ''}"
            @click=${() => this.handleTabChange('compliance')}
          >
            Compliance
            ${this.metrics?.compliance_summary?.active_violations_total > 0 ? html`
              <span class="alert-count">${this.metrics.compliance_summary.active_violations_total}</span>
            ` : ''}
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