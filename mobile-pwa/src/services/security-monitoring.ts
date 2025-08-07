/**
 * Security Monitoring Service
 * 
 * Provides comprehensive security monitoring integration with backend APIs
 * for vulnerability tracking, compliance monitoring, and threat detection.
 */

import { BaseService } from './base-service'
import type { SecurityMetrics, SecurityAlert, VulnerabilityScan, VulnerabilityItem, ComplianceStatus } from '../components/dashboard/security-monitoring-panel'

export interface SecurityDashboardQuery {
  time_window_hours?: number
  include_resolved?: boolean
  threat_level_filter?: string
  event_type_filter?: string
}

export interface ComplianceReportRequest {
  framework: string
  report_type?: string
  output_format?: string
  include_evidence?: boolean
}

export interface ThreatInvestigationRequest {
  incident_id?: string
  agent_id?: string
  time_range_hours?: number
  include_behavioral_analysis?: boolean
}

export interface SecurityWebSocketMessage {
  type: 'security_event' | 'vulnerability_update' | 'compliance_alert' | 'connection_established' | 'filter_updated'
  data?: any
  timestamp: string
  message?: string
}

export class SecurityMonitoringService extends BaseService {
  private websocket: WebSocket | null = null
  private eventListeners: Map<string, Set<(data: any) => void>> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 5000

  /**
   * Initialize WebSocket connection for real-time security events
   */
  async connectWebSocket(): Promise<void> {
    try {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${wsProtocol}//${window.location.host}/security/ws/security-events`
      
      this.websocket = new WebSocket(wsUrl)
      
      this.websocket.onopen = () => {
        console.log('Security monitoring WebSocket connected')
        this.reconnectAttempts = 0
        this.notifyListeners('connection_status', { connected: true })
      }
      
      this.websocket.onmessage = (event) => {
        try {
          const message: SecurityWebSocketMessage = JSON.parse(event.data)
          this.handleWebSocketMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      this.websocket.onclose = () => {
        console.log('Security monitoring WebSocket disconnected')
        this.notifyListeners('connection_status', { connected: false })
        this.handleWebSocketDisconnect()
      }
      
      this.websocket.onerror = (error) => {
        console.error('Security monitoring WebSocket error:', error)
        this.notifyListeners('websocket_error', { error })
      }
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      throw error
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnectWebSocket(): void {
    if (this.websocket) {
      this.websocket.close()
      this.websocket = null
    }
  }

  /**
   * Get comprehensive security dashboard overview
   */
  async getSecurityOverview(query: SecurityDashboardQuery = {}): Promise<SecurityMetrics> {
    try {
      const params = new URLSearchParams({
        time_window_hours: (query.time_window_hours || 24).toString(),
        include_resolved: (query.include_resolved || false).toString(),
        ...(query.threat_level_filter && { threat_level_filter: query.threat_level_filter }),
        ...(query.event_type_filter && { event_type_filter: query.event_type_filter })
      })

      const response = await this.request<SecurityMetrics>('/security/dashboard/overview', {
        method: 'GET',
        params
      })

      return response
    } catch (error) {
      console.error('Failed to fetch security overview:', error)
      throw error
    }
  }

  /**
   * Get detailed vulnerability tracking data
   */
  async getVulnerabilities(includeResolved = false): Promise<VulnerabilityItem[]> {
    try {
      const response = await this.request<{ vulnerabilities: VulnerabilityItem[] }>('/security/vulnerabilities', {
        method: 'GET',
        params: { include_resolved: includeResolved.toString() }
      })

      return response.vulnerabilities
    } catch (error) {
      console.error('Failed to fetch vulnerabilities:', error)
      // Return mock data for development
      return this.getMockVulnerabilities()
    }
  }

  /**
   * Get security scan results and history
   */
  async getSecurityScans(limit = 10): Promise<VulnerabilityScan[]> {
    try {
      const response = await this.request<{ scans: VulnerabilityScan[] }>('/security/scans', {
        method: 'GET',
        params: { limit: limit.toString() }
      })

      return response.scans
    } catch (error) {
      console.error('Failed to fetch security scans:', error)
      // Return mock data for development
      return this.getMockSecurityScans()
    }
  }

  /**
   * Get compliance status for all frameworks
   */
  async getComplianceStatus(): Promise<ComplianceStatus[]> {
    try {
      const response = await this.request<{ compliance_frameworks: ComplianceStatus[] }>('/security/compliance/dashboard', {
        method: 'GET'
      })

      return response.compliance_frameworks
    } catch (error) {
      console.error('Failed to fetch compliance status:', error)
      // Return mock data for development
      return this.getMockComplianceStatus()
    }
  }

  /**
   * Get detailed threat analysis dashboard
   */
  async getThreatDashboard(query: SecurityDashboardQuery = {}): Promise<any> {
    try {
      const params = new URLSearchParams({
        time_window_hours: (query.time_window_hours || 24).toString(),
        include_resolved: (query.include_resolved || false).toString(),
        ...(query.threat_level_filter && { threat_level_filter: query.threat_level_filter }),
        ...(query.event_type_filter && { event_type_filter: query.event_type_filter })
      })

      const response = await this.request('/security/dashboard/threats', {
        method: 'GET',
        params
      })

      return response
    } catch (error) {
      console.error('Failed to fetch threat dashboard:', error)
      throw error
    }
  }

  /**
   * Get agent behavior analysis
   */
  async getAgentBehaviorAnalysis(timeWindowHours = 24, riskThreshold = 0.5): Promise<any> {
    try {
      const response = await this.request('/security/agents/behavior', {
        method: 'GET',
        params: {
          time_window_hours: timeWindowHours.toString(),
          risk_threshold: riskThreshold.toString()
        }
      })

      return response
    } catch (error) {
      console.error('Failed to fetch agent behavior analysis:', error)
      throw error
    }
  }

  /**
   * Initiate security incident investigation
   */
  async investigateIncident(request: ThreatInvestigationRequest): Promise<any> {
    try {
      const response = await this.request('/security/incidents/investigate', {
        method: 'POST',
        body: JSON.stringify(request)
      })

      return response
    } catch (error) {
      console.error('Failed to initiate investigation:', error)
      throw error
    }
  }

  /**
   * Get investigation status
   */
  async getInvestigationStatus(investigationId: string): Promise<any> {
    try {
      const response = await this.request(`/security/incidents/investigate/${investigationId}`, {
        method: 'GET'
      })

      return response
    } catch (error) {
      console.error('Failed to get investigation status:', error)
      throw error
    }
  }

  /**
   * Generate compliance report
   */
  async generateComplianceReport(request: ComplianceReportRequest): Promise<any> {
    try {
      const response = await this.request('/security/compliance/reports/generate', {
        method: 'POST',
        body: JSON.stringify(request)
      })

      return response
    } catch (error) {
      console.error('Failed to generate compliance report:', error)
      throw error
    }
  }

  /**
   * Get compliance report status
   */
  async getComplianceReportStatus(reportId: string): Promise<any> {
    try {
      const response = await this.request(`/security/compliance/reports/${reportId}`, {
        method: 'GET'
      })

      return response
    } catch (error) {
      console.error('Failed to get compliance report status:', error)
      throw error
    }
  }

  /**
   * Add event listener for real-time updates
   */
  addEventListener(eventType: string, callback: (data: any) => void): void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, new Set())
    }
    this.eventListeners.get(eventType)!.add(callback)
  }

  /**
   * Remove event listener
   */
  removeEventListener(eventType: string, callback: (data: any) => void): void {
    const listeners = this.eventListeners.get(eventType)
    if (listeners) {
      listeners.delete(callback)
      if (listeners.size === 0) {
        this.eventListeners.delete(eventType)
      }
    }
  }

  /**
   * Send filter updates to WebSocket
   */
  updateWebSocketFilters(filters: any): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'update_filters',
        filters
      }))
    }
  }

  /**
   * Get security system health check
   */
  async getSecurityHealth(): Promise<any> {
    try {
      const response = await this.request('/security/health', {
        method: 'GET'
      })

      return response
    } catch (error) {
      console.error('Failed to fetch security health:', error)
      throw error
    }
  }

  // Private methods

  private handleWebSocketMessage(message: SecurityWebSocketMessage): void {
    switch (message.type) {
      case 'security_event':
        this.notifyListeners('security_event', message.data)
        break
      case 'vulnerability_update':
        this.notifyListeners('vulnerability_update', message.data)
        break
      case 'compliance_alert':
        this.notifyListeners('compliance_alert', message.data)
        break
      case 'connection_established':
        this.notifyListeners('connection_established', message.data)
        break
      case 'filter_updated':
        this.notifyListeners('filter_updated', message.data)
        break
      default:
        console.warn('Unknown WebSocket message type:', message.type)
    }
  }

  private handleWebSocketDisconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
      
      setTimeout(() => {
        this.connectWebSocket().catch(error => {
          console.error('WebSocket reconnection failed:', error)
        })
      }, this.reconnectInterval * this.reconnectAttempts)
    } else {
      console.error('Max WebSocket reconnection attempts reached')
      this.notifyListeners('reconnection_failed', { attempts: this.reconnectAttempts })
    }
  }

  private notifyListeners(eventType: string, data: any): void {
    const listeners = this.eventListeners.get(eventType)
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error('Error in security event listener:', error)
        }
      })
    }
  }

  // Mock data methods for development
  private getMockVulnerabilities(): VulnerabilityItem[] {
    return [
      {
        vulnerability_id: 'vuln-001',
        cve_id: 'CVE-2024-12345',
        severity: 'critical',
        title: 'MD5 Hash Algorithm Usage Detected',
        description: 'Application using deprecated MD5 hash algorithm for security-sensitive operations',
        affected_component: 'authentication-service',
        cvss_score: 9.1,
        age_days: 15,
        sla_compliant: false,
        remediation_status: 'open',
        remediation_timeline: '2024-08-20T00:00:00Z',
        patch_available: true,
        exploit_available: true
      },
      {
        vulnerability_id: 'vuln-002',
        cve_id: 'CVE-2024-67890',
        severity: 'high',
        title: 'Weak TLS Configuration',
        description: 'TLS configuration allows weak cipher suites',
        affected_component: 'web-server',
        cvss_score: 7.5,
        age_days: 8,
        sla_compliant: true,
        remediation_status: 'in_progress',
        patch_available: true,
        exploit_available: false
      },
      {
        vulnerability_id: 'vuln-003',
        severity: 'medium',
        title: 'Outdated Dependency',
        description: 'Third-party library with known security issues',
        affected_component: 'api-gateway',
        cvss_score: 5.3,
        age_days: 3,
        sla_compliant: true,
        remediation_status: 'open',
        patch_available: true,
        exploit_available: false
      }
    ]
  }

  private getMockSecurityScans(): VulnerabilityScan[] {
    const now = new Date()
    const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000)
    
    return [
      {
        scan_id: 'scan-001',
        scan_type: 'infrastructure',
        started_at: now.toISOString(),
        completed_at: new Date(now.getTime() + 30 * 60 * 1000).toISOString(),
        status: 'completed',
        vulnerabilities_found: 12,
        critical_count: 1,
        high_count: 3,
        medium_count: 4,
        low_count: 4,
        remediation_recommendations: [
          'Upgrade MD5 usage to SHA-256',
          'Update TLS configuration',
          'Apply security patches to dependencies'
        ],
        baseline_comparison: {
          previous_scan_date: yesterday.toISOString(),
          vulnerabilities_new: 2,
          vulnerabilities_fixed: 3,
          trend: 'improving'
        }
      },
      {
        scan_id: 'scan-002',
        scan_type: 'application',
        started_at: yesterday.toISOString(),
        completed_at: new Date(yesterday.getTime() + 45 * 60 * 1000).toISOString(),
        status: 'completed',
        vulnerabilities_found: 8,
        critical_count: 2,
        high_count: 2,
        medium_count: 2,
        low_count: 2,
        remediation_recommendations: [
          'Fix SQL injection vulnerabilities',
          'Implement input validation',
          'Update authentication mechanisms'
        ]
      }
    ]
  }

  private getMockComplianceStatus(): ComplianceStatus[] {
    return [
      {
        framework: 'soc2_type2',
        overall_score: 0.98,
        status: 'compliant',
        last_assessment: '2024-08-01T00:00:00Z',
        next_assessment: '2024-09-01T00:00:00Z',
        controls: {
          total: 64,
          compliant: 63,
          non_compliant: 1,
          not_assessed: 0
        },
        active_violations: [
          {
            violation_id: 'viol-001',
            control_id: 'CC6.1',
            severity: 'medium',
            description: 'Security awareness training completion rate below threshold',
            deadline: '2024-08-15T00:00:00Z',
            status: 'remediation_in_progress'
          }
        ],
        audit_readiness: {
          score: 0.95,
          missing_evidence: ['Security training records for Q2'],
          documentation_gaps: [],
          recommendations: ['Complete Q2 training documentation']
        }
      },
      {
        framework: 'iso27001',
        overall_score: 0.92,
        status: 'compliant',
        last_assessment: '2024-07-25T00:00:00Z',
        next_assessment: '2024-10-25T00:00:00Z',
        controls: {
          total: 114,
          compliant: 105,
          non_compliant: 9,
          not_assessed: 0
        },
        active_violations: [
          {
            violation_id: 'viol-002',
            control_id: 'A.12.6.1',
            severity: 'high',
            description: 'Vulnerability management process gaps identified',
            deadline: '2024-08-10T00:00:00Z',
            status: 'open'
          }
        ],
        audit_readiness: {
          score: 0.88,
          missing_evidence: [
            'Vulnerability scan reports',
            'Incident response test results'
          ],
          documentation_gaps: [
            'Security policy version control',
            'Business continuity test documentation'
          ],
          recommendations: [
            'Complete vulnerability management documentation',
            'Conduct incident response testing'
          ]
        }
      }
    ]
  }
}

// Export singleton instance
export const securityMonitoringService = new SecurityMonitoringService()