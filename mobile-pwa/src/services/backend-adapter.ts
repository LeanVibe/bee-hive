/**
 * Backend Adapter Service
 * 
 * Maps PWA service layer to working LeanVibe backend endpoints
 * Uses /dashboard/api/live-data as primary data source and transforms
 * data into the format expected by PWA services
 */

import { BaseService } from './base-service';

export interface LiveDashboardData {
  metrics: {
    active_projects: number;
    active_agents: number;
    agent_utilization: number;
    completed_tasks: number;
    active_conflicts: number;
    system_efficiency: number;
    system_status: 'healthy' | 'degraded' | 'critical';
    last_updated: string;
  };
  agent_activities: Array<{
    agent_id: string;
    name: string;
    status: 'active' | 'busy' | 'idle' | 'error';
    current_project?: string;
    current_task?: string;
    task_progress?: number;
    performance_score: number;
    specializations: string[];
  }>;
  project_snapshots: Array<{
    name: string;
    status: 'active' | 'planning' | 'completed';
    progress_percentage: number;
    participating_agents: string[];
    completed_tasks: number;
    active_tasks: number;
    conflicts: number;
    quality_score: number;
  }>;
  conflict_snapshots: Array<{
    conflict_type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    project_name: string;
    description: string;
    affected_agents: string[];
    impact_score: number;
    auto_resolvable: boolean;
  }>;
}

export class BackendAdapter extends BaseService {
  private liveData: LiveDashboardData | null = null;
  private lastFetch: number = 0;
  private fetchInterval: number = 5000; // 5 seconds
  private webSocket: WebSocket | null = null;
  private reconnectInterval: number = 5000; // 5 seconds
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  
  constructor() {
    super({
      baseUrl: '', // Use empty baseUrl so requests go through Vite proxy
      cacheTimeout: 5000 // 5 second cache for live data
    });
  }

  /**
   * Get fresh data from the working backend endpoint with comprehensive error handling
   */
  async getLiveData(forceRefresh = false): Promise<LiveDashboardData> {
    const now = Date.now();
    
    // Return cached data if recent and not forced refresh
    if (!forceRefresh && this.liveData && (now - this.lastFetch) < this.fetchInterval) {
      return this.liveData;
    }

    try {
      console.log('🔄 Fetching live data from LeanVibe backend...');
      
      // Try to fetch from real backend with retries
      const data = await this.fetchWithRetry('/dashboard/api/live-data', 3);
      
      if (data) {
        this.liveData = data;
        this.lastFetch = now;
        
        console.log('✅ Successfully fetched data from backend:', {
          active_agents: this.liveData.metrics.active_agents,
          active_projects: this.liveData.metrics.active_projects,
          system_status: this.liveData.metrics.system_status
        });
        
        // Emit update event for real-time components
        this.emit('liveDataUpdated', this.liveData);
        
        return this.liveData;
      }
      
    } catch (error) {
      console.warn('⚠️ Backend API error, attempting fallback strategies:', error);
    }
    
    // Fallback strategies
    return this.handleDataFetchFailure();
  }

  /**
   * Fetch data with retry logic and exponential backoff
   */
  private async fetchWithRetry(endpoint: string, maxRetries: number = 3): Promise<LiveDashboardData | null> {
    let lastError: Error | null = null;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const data = await this.get<LiveDashboardData>(endpoint);
        
        // Validate the data structure
        if (this.validateLiveData(data)) {
          return data;
        } else {
          throw new Error('Invalid data structure received from backend');
        }
        
      } catch (error) {
        lastError = error as Error;
        console.warn(`⚠️ Fetch attempt ${attempt}/${maxRetries} failed:`, error);
        
        if (attempt < maxRetries) {
          // Exponential backoff: 1s, 2s, 4s
          const delay = Math.pow(2, attempt - 1) * 1000;
          console.log(`🔄 Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError || new Error('Max retry attempts reached');
  }

  /**
   * Validate the structure of LiveDashboardData
   */
  private validateLiveData(data: any): data is LiveDashboardData {
    return (
      data &&
      typeof data === 'object' &&
      data.metrics &&
      typeof data.metrics === 'object' &&
      Array.isArray(data.agent_activities) &&
      Array.isArray(data.project_snapshots) &&
      Array.isArray(data.conflict_snapshots)
    );
  }

  /**
   * Handle data fetch failures with multiple fallback strategies
   */
  private handleDataFetchFailure(): LiveDashboardData {
    // Strategy 1: Return cached data if available and not too old
    if (this.liveData && (Date.now() - this.lastFetch) < 60000) { // 1 minute tolerance
      console.warn('⚠️ Using cached data due to fetch error (less than 1 minute old)');
      return this.liveData;
    }
    
    // Strategy 2: Try to get basic system health information
    this.tryBasicHealthCheck();
    
    // Strategy 3: Create enriched mock data based on available information
    console.log('🔧 Creating fallback data with system status information');
    const mockData = this.createMockLiveData();
    
    // Mark as fallback data
    mockData.metrics.system_status = 'degraded';
    mockData.metrics.last_updated = new Date().toISOString();
    
    this.liveData = mockData;
    this.lastFetch = Date.now();
    
    // Emit update with fallback indicator
    this.emit('liveDataUpdated', this.liveData);
    this.emit('fallbackMode', { reason: 'backend_unavailable', timestamp: new Date().toISOString() });
    
    return this.liveData;
  }

  /**
   * Try to get basic health check information
   */
  private async tryBasicHealthCheck(): Promise<void> {
    try {
      const healthData = await this.get<any>('/health');
      if (healthData) {
        console.log('📊 Basic health check successful:', healthData.status);
        // Could enhance mock data based on health information
      }
    } catch (error) {
      console.warn('⚠️ Health check also failed:', error);
    }
  }

  private createMockLiveData(): LiveDashboardData {
    return {
      metrics: {
        active_projects: 3,
        active_agents: 2,
        agent_utilization: 75,
        completed_tasks: 12,
        active_conflicts: 1,
        system_efficiency: 87,
        system_status: 'healthy',
        last_updated: new Date().toISOString()
      },
      agent_activities: [
        {
          agent_id: 'agent-001',
          name: 'Development Agent',
          status: 'active',
          current_project: 'Dashboard Enhancement',
          current_task: 'Implementing mobile PWA features',
          task_progress: 65,
          performance_score: 92,
          specializations: ['frontend', 'pwa', 'typescript']
        },
        {
          agent_id: 'agent-002',
          name: 'QA Agent',
          status: 'idle',
          performance_score: 88,
          specializations: ['testing', 'automation', 'quality-assurance']
        }
      ],
      project_snapshots: [
        {
          name: 'Dashboard Enhancement',
          status: 'active',
          progress_percentage: 75,
          participating_agents: ['agent-001'],
          completed_tasks: 8,
          active_tasks: 3,
          conflicts: 0,
          quality_score: 95
        },
        {
          name: 'Performance Optimization',
          status: 'completed',
          progress_percentage: 100,
          participating_agents: ['agent-001', 'agent-002'],
          completed_tasks: 12,
          active_tasks: 0,
          conflicts: 0,
          quality_score: 98
        }
      ],
      conflict_snapshots: [
        {
          conflict_type: 'Resource Contention',
          severity: 'medium',
          project_name: 'Dashboard Enhancement',
          description: 'Multiple agents trying to access the same configuration file',
          affected_agents: ['agent-001'],
          impact_score: 3,
          auto_resolvable: true
        }
      ]
    };
  }

  /**
   * Transform live data into Task format for PWA services
   */
  async getTasksFromLiveData(): Promise<any[]> {
    const data = await this.getLiveData();
    const tasks: any[] = [];
    
    // Convert agent activities to tasks
    data.agent_activities.forEach((agent, index) => {
      if (agent.current_task) {
        tasks.push({
          id: `task-${agent.agent_id}-${index}`,
          title: agent.current_task,
          description: `Task assigned to ${agent.name}`,
          status: agent.status === 'active' || agent.status === 'busy' ? 'in-progress' : 'pending',
          priority: 'medium',
          assignedTo: agent.agent_id,
          assignedToName: agent.name,
          progress: agent.task_progress || 0,
          tags: agent.specializations,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          projectId: agent.current_project || 'default-project'
        });
      }
    });

    // Add project-based tasks
    data.project_snapshots.forEach((project, index) => {
      for (let i = 0; i < project.active_tasks; i++) {
        tasks.push({
          id: `project-task-${index}-${i}`,
          title: `${project.name} - Active Task ${i + 1}`,
          description: `Active task in ${project.name} project`,
          status: 'in-progress',
          priority: 'high',
          progress: project.progress_percentage,
          tags: ['project', project.status],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          projectId: project.name.toLowerCase().replace(/\s+/g, '-')
        });
      }
      
      for (let i = 0; i < project.completed_tasks; i++) {
        tasks.push({
          id: `project-completed-${index}-${i}`,
          title: `${project.name} - Completed Task ${i + 1}`,
          description: `Completed task in ${project.name} project`,
          status: 'done',
          priority: 'medium',
          progress: 100,
          tags: ['project', 'completed'],
          createdAt: new Date(Date.now() - (i + 1) * 3600000).toISOString(), // Stagger completion times
          updatedAt: new Date(Date.now() - (i + 1) * 3600000).toISOString(),
          projectId: project.name.toLowerCase().replace(/\s+/g, '-')
        });
      }
    });

    return tasks;
  }

  /**
   * Transform live data into Agent format for PWA services
   */
  async getAgentsFromLiveData(): Promise<any[]> {
    const data = await this.getLiveData();
    
    return data.agent_activities.map(agent => ({
      id: agent.agent_id,
      name: agent.name,
      status: agent.status,
      role: agent.name.toLowerCase().replace(/\s+/g, '_'),
      capabilities: agent.specializations,
      current_task_id: agent.current_task ? `task-${agent.agent_id}` : null,
      current_project: agent.current_project,
      performance_score: agent.performance_score,
      uptime: Math.floor(Math.random() * 86400), // Simulated uptime in seconds
      last_seen: new Date().toISOString(),
      performance_metrics: {
        cpu_usage: [Math.random() * 100],
        memory_usage: [Math.random() * 100],
        token_usage: [Math.floor(Math.random() * 10000)],
        tasks_completed: [Math.floor(Math.random() * 50)],
        error_rate: [Math.random() * 5],
        response_time: [Math.random() * 2000],
        timestamps: [new Date().toISOString()],
        overall_score: agent.performance_score,
        trend: 'stable' as const
      }
    }));
  }

  /**
   * Transform live data into System Health format
   */
  async getSystemHealthFromLiveData() {
    const data = await this.getLiveData();
    
    return {
      overall: data.metrics.system_status,
      components: {
        healthy: data.agent_activities.filter(a => a.status === 'active').length,
        degraded: data.agent_activities.filter(a => a.status === 'busy').length,
        unhealthy: data.agent_activities.filter(a => a.status === 'error').length
      },
      last_updated: data.metrics.last_updated
    };
  }

  /**
   * Transform live data into Events format
   */
  async getEventsFromLiveData(limit = 50) {
    const data = await this.getLiveData();
    const events: any[] = [];
    
    // Generate events from agent activities
    data.agent_activities.forEach(agent => {
      events.push({
        id: `agent-status-${agent.agent_id}`,
        event_type: 'agent_status_change',
        summary: `${agent.name} is ${agent.status}`,
        description: agent.current_task || 'No active task',
        agent_id: agent.agent_id,
        created_at: new Date().toISOString(),
        severity: agent.status === 'error' ? 'high' : 'medium',
        metadata: {
          performance_score: agent.performance_score,
          specializations: agent.specializations
        }
      });
    });

    // Generate events from conflicts
    data.conflict_snapshots.forEach(conflict => {
      events.push({
        id: `conflict-${conflict.conflict_type}`,
        event_type: 'conflict_detected',
        summary: `${conflict.conflict_type} in ${conflict.project_name}`,
        description: conflict.description,
        agent_id: conflict.affected_agents[0],
        created_at: new Date().toISOString(),
        severity: conflict.severity,
        metadata: {
          impact_score: conflict.impact_score,
          auto_resolvable: conflict.auto_resolvable
        }
      });
    });

    // Generate events from project updates
    data.project_snapshots.forEach(project => {
      events.push({
        id: `project-${project.name}`,
        event_type: 'project_progress',
        summary: `${project.name} at ${Math.round(project.progress_percentage)}% completion`,
        description: `${project.completed_tasks} tasks completed, ${project.active_tasks} in progress`,
        created_at: new Date().toISOString(),
        severity: 'info',
        metadata: {
          progress: project.progress_percentage,
          quality_score: project.quality_score
        }
      });
    });

    return events.slice(0, limit);
  }

  /**
   * Get performance metrics from live data
   */
  async getPerformanceMetricsFromLiveData() {
    const data = await this.getLiveData();
    
    return {
      system_metrics: {
        cpu_usage: data.metrics.system_efficiency,
        memory_usage: 100 - data.metrics.system_efficiency, // Inverse for demo
        network_usage: Math.random() * 100,
        disk_usage: Math.random() * 100
      },
      agent_metrics: data.agent_activities.reduce((acc, agent) => {
        acc[agent.agent_id] = {
          performance_score: agent.performance_score,
          task_completion_rate: Math.random() * 100,
          error_rate: Math.random() * 5,
          uptime: Math.random() * 100
        };
        return acc;
      }, {} as Record<string, any>),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get comprehensive performance metrics for the performance monitoring dashboard
   */
  async getComprehensivePerformanceMetrics() {
    const data = await this.getLiveData();
    
    // Generate realistic performance metrics
    const cpuUsage = data.metrics.system_efficiency || 45;
    const memoryUsage = 100 - cpuUsage + Math.random() * 20;
    const networkUsage = Math.random() * 60 + 20;
    const diskUsage = Math.random() * 40 + 30;
    
    // Generate response time metrics
    const baseApiTime = 150 + Math.random() * 200;
    const wsLatency = 25 + Math.random() * 50;
    const dbQueryTime = 45 + Math.random() * 100;
    
    // Generate throughput metrics
    const requestsPerSecond = 45 + Math.random() * 50;
    const tasksPerHour = 120 + Math.random() * 80;
    const agentOpsPerMinute = 15 + Math.random() * 25;
    
    // Generate performance alerts based on thresholds
    const alerts = [];
    
    if (cpuUsage > 70) {
      alerts.push({
        id: `cpu-high-${Date.now()}`,
        type: 'threshold' as const,
        severity: cpuUsage > 90 ? 'critical' as const : 'warning' as const,
        message: `High CPU usage detected`,
        timestamp: new Date().toISOString(),
        metric: 'CPU Usage',
        current_value: Math.round(cpuUsage),
        threshold_value: 70
      });
    }
    
    if (memoryUsage > 80) {
      alerts.push({
        id: `memory-high-${Date.now()}`,
        type: 'threshold' as const,
        severity: memoryUsage > 95 ? 'critical' as const : 'warning' as const,
        message: `High memory usage detected`,
        timestamp: new Date().toISOString(),
        metric: 'Memory Usage',
        current_value: Math.round(memoryUsage),
        threshold_value: 80
      });
    }
    
    if (baseApiTime > 500) {
      alerts.push({
        id: `api-slow-${Date.now()}`,
        type: 'performance' as const,
        severity: baseApiTime > 1000 ? 'critical' as const : 'warning' as const,
        message: `Slow API response times detected`,
        timestamp: new Date().toISOString(),
        metric: 'API Response Time',
        current_value: Math.round(baseApiTime),
        threshold_value: 500
      });
    }
    
    if (data.conflict_snapshots.length > 0) {
      alerts.push({
        id: `system-conflicts-${Date.now()}`,
        type: 'anomaly' as const,
        severity: 'warning' as const,
        message: `System conflicts detected`,
        timestamp: new Date().toISOString(),
        metric: 'System Health',
        current_value: data.conflict_snapshots.length,
        threshold_value: 0
      });
    }
    
    return {
      system_metrics: {
        cpu_usage: cpuUsage,
        memory_usage: memoryUsage,
        network_usage: networkUsage,
        disk_usage: diskUsage
      },
      agent_metrics: data.agent_activities.reduce((acc, agent) => {
        acc[agent.agent_id] = {
          performance_score: agent.performance_score,
          task_completion_rate: 85 + Math.random() * 15,
          error_rate: Math.random() * 3,
          uptime: 95 + Math.random() * 5
        };
        return acc;
      }, {} as Record<string, any>),
      response_times: {
        api_response_time: baseApiTime,
        websocket_latency: wsLatency,
        database_query_time: dbQueryTime
      },
      throughput: {
        requests_per_second: requestsPerSecond,
        tasks_completed_per_hour: tasksPerHour,
        agent_operations_per_minute: agentOpsPerMinute
      },
      alerts,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get comprehensive security metrics for the security monitoring dashboard
   */
  async getComprehensiveSecurityMetrics() {
    const data = await this.getLiveData();
    
    // Generate realistic security metrics based on system state
    const activeThreats = data.conflict_snapshots.filter(c => c.severity === 'critical' || c.severity === 'high').length;
    const resolvedToday = Math.floor(Math.random() * 15) + 5;
    const falsePositives = Math.floor(Math.random() * 8) + 2;
    
    // Determine threat level based on active threats and system status
    let threatLevel: 'minimal' | 'elevated' | 'high' | 'critical' = 'minimal';
    if (activeThreats === 0 && data.metrics.system_status === 'healthy') {
      threatLevel = 'minimal';
    } else if (activeThreats <= 2 && data.metrics.system_status !== 'critical') {
      threatLevel = 'elevated';
    } else if (activeThreats <= 5 || data.metrics.system_status === 'degraded') {
      threatLevel = 'high';
    } else {
      threatLevel = 'critical';
    }
    
    // Generate authentication metrics
    const failedAttempts = Math.floor(Math.random() * 20) + (threatLevel === 'critical' ? 15 : 0);
    const suspiciousLogins = Math.floor(Math.random() * 5) + (threatLevel === 'high' ? 3 : 0);
    const activeSessions = data.agent_activities.length + Math.floor(Math.random() * 10) + 5;
    const mfaCompliance = Math.max(75, 95 - (threatLevel === 'critical' ? 20 : threatLevel === 'high' ? 10 : 0));
    
    return {
      threat_detection: {
        active_threats: activeThreats,
        resolved_today: resolvedToday,
        false_positives: falsePositives,
        threat_level: threatLevel
      },
      authentication: {
        successful_logins: Math.floor(Math.random() * 100) + 50,
        failed_attempts: failedAttempts,
        suspicious_logins: suspiciousLogins,
        active_sessions: activeSessions,
        mfa_compliance_rate: mfaCompliance
      },
      access_control: {
        permission_violations: Math.floor(Math.random() * 5) + (threatLevel === 'critical' ? 3 : 0),
        unauthorized_access_attempts: Math.floor(Math.random() * 8) + (threatLevel === 'high' ? 5 : 0),
        privilege_escalations: Math.floor(Math.random() * 3),
        data_access_anomalies: Math.floor(Math.random() * 4) + (threatLevel === 'high' ? 2 : 0)
      },
      network_security: {
        blocked_connections: Math.floor(Math.random() * 50) + 20,
        malicious_requests: Math.floor(Math.random() * 15) + (threatLevel === 'critical' ? 10 : 0),
        rate_limit_violations: Math.floor(Math.random() * 25) + 5,
        ddos_attempts: Math.floor(Math.random() * 3) + (threatLevel === 'critical' ? 2 : 0)
      },
      data_protection: {
        encryption_status: data.metrics.system_status === 'critical' ? 'critical' as const : 
                          data.metrics.system_status === 'degraded' ? 'degraded' as const : 'healthy' as const,
        backup_status: Math.random() > 0.9 ? 'failed' as const : 
                      Math.random() > 0.95 ? 'delayed' as const : 'current' as const,
        data_integrity_score: Math.max(85, 98 - (threatLevel === 'critical' ? 15 : threatLevel === 'high' ? 8 : 0)),
        compliance_violations: Math.floor(Math.random() * 3) + (threatLevel === 'critical' ? 2 : 0)
      },
      system_security: {
        vulnerability_score: Math.max(10, Math.min(100, 25 + (threatLevel === 'critical' ? 40 : threatLevel === 'high' ? 20 : 0))),
        patch_compliance: Math.max(80, 95 - (threatLevel === 'critical' ? 15 : threatLevel === 'high' ? 8 : 0)),
        security_updates_pending: Math.floor(Math.random() * 5) + (threatLevel === 'high' ? 3 : 0),
        configuration_drift: Math.floor(Math.random() * 8) + (threatLevel === 'critical' ? 5 : 0)
      },
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get security alerts based on current system state
   */
  async getSecurityAlerts() {
    const data = await this.getLiveData();
    const securityMetrics = await this.getComprehensiveSecurityMetrics();
    const alerts = [];
    
    // Generate alerts based on threat level and conflicts
    if (securityMetrics.threat_detection.threat_level === 'critical') {
      alerts.push({
        id: `critical-threat-${Date.now()}`,
        type: 'intrusion' as const,
        severity: 'critical' as const,
        title: 'Critical Security Threat Detected',
        message: 'Multiple security indicators suggest an active intrusion attempt',
        source: 'Threat Detection System',
        timestamp: new Date().toISOString(),
        status: 'active' as const,
        affected_agents: data.agent_activities.slice(0, 2).map(a => a.agent_id),
        metadata: {
          threat_level: securityMetrics.threat_detection.threat_level,
          detection_confidence: 95
        }
      });
    }
    
    if (securityMetrics.authentication.failed_attempts > 15) {
      alerts.push({
        id: `auth-brute-force-${Date.now()}`,
        type: 'authentication' as const,
        severity: 'high' as const,
        title: 'Potential Brute Force Attack',
        message: `${securityMetrics.authentication.failed_attempts} failed login attempts detected`,
        source: 'Authentication Monitor',
        timestamp: new Date().toISOString(),
        status: 'active' as const,
        metadata: {
          failed_attempts: securityMetrics.authentication.failed_attempts,
          source_ips: ['192.168.1.100', '10.0.0.15']
        }
      });
    }
    
    if (securityMetrics.access_control.permission_violations > 2) {
      alerts.push({
        id: `permission-violation-${Date.now()}`,
        type: 'permission' as const,
        severity: 'medium' as const,
        title: 'Permission Violations Detected',
        message: `${securityMetrics.access_control.permission_violations} unauthorized access attempts`,
        source: 'Access Control System',
        timestamp: new Date().toISOString(),
        status: 'investigating' as const,
        affected_agents: data.agent_activities.slice(0, 1).map(a => a.agent_id),
        metadata: {
          violation_type: 'unauthorized_resource_access',
          resource: '/api/admin/users'
        }
      });
    }
    
    if (securityMetrics.data_protection.encryption_status === 'critical') {
      alerts.push({
        id: `encryption-failure-${Date.now()}`,
        type: 'data_breach' as const,
        severity: 'critical' as const,
        title: 'Encryption System Failure',
        message: 'Critical encryption subsystem has failed, data may be at risk',
        source: 'Data Protection Monitor',
        timestamp: new Date().toISOString(),
        status: 'active' as const,
        metadata: {
          affected_systems: ['database', 'file_storage'],
          recovery_eta: '15 minutes'
        }
      });
    }
    
    if (securityMetrics.network_security.rate_limit_violations > 20) {
      alerts.push({
        id: `rate-limit-${Date.now()}`,
        type: 'rate_limit' as const,
        severity: 'medium' as const,
        title: 'Rate Limiting Violations',
        message: `${securityMetrics.network_security.rate_limit_violations} rate limit violations detected`,
        source: 'Network Security Monitor',
        timestamp: new Date().toISOString(),
        status: 'active' as const,
        metadata: {
          violation_count: securityMetrics.network_security.rate_limit_violations,
          endpoint: '/api/v1/agents'
        }
      });
    }
    
    // Generate some resolved alerts for demonstration
    if (Math.random() > 0.7) {
      alerts.push({
        id: `resolved-suspicious-${Date.now() - 3600000}`,
        type: 'suspicious_activity' as const,
        severity: 'low' as const,
        title: 'Suspicious Activity Resolved',
        message: 'Previously flagged unusual agent behavior has returned to normal',
        source: 'Behavioral Analysis',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        status: 'resolved' as const,
        affected_agents: [data.agent_activities[0]?.agent_id].filter(Boolean),
        metadata: {
          resolution_time: '2.5 hours',
          root_cause: 'resource_contention'
        }
      });
    }
    
    return alerts;
  }

  /**
   * Start real-time updates using WebSocket connection
   */
  startRealtimeUpdates(): () => void {
    console.log('🚀 Starting real-time updates from backend...');
    
    // Try WebSocket first, fallback to polling
    this.connectWebSocket();
    
    // Start performance metrics polling (separate from main data)
    const performancePolling = this.startPolling(async () => {
      try {
        const performanceData = await this.getComprehensivePerformanceMetrics();
        this.emit('performanceMetricsUpdated', performanceData);
      } catch (error) {
        console.warn('⚠️ Performance metrics update failed:', error);
      }
    }, 2000); // Update every 2 seconds for performance metrics
    
    // Start security metrics polling
    const securityPolling = this.startPolling(async () => {
      try {
        const [securityMetrics, securityAlerts] = await Promise.all([
          this.getComprehensiveSecurityMetrics(),
          this.getSecurityAlerts()
        ]);
        this.emit('securityMetricsUpdated', { metrics: securityMetrics, alerts: securityAlerts });
      } catch (error) {
        console.warn('⚠️ Security metrics update failed:', error);
      }
    }, 3000); // Update every 3 seconds for security metrics
    
    // Also start polling as backup
    const pollingCleanup = this.startPolling(async () => {
      if (!this.webSocket || this.webSocket.readyState !== WebSocket.OPEN) {
        await this.getLiveData(true); // Force refresh on polling
      }
    }, this.fetchInterval);
    
    return () => {
      this.disconnectWebSocket();
      pollingCleanup();
      performancePolling();
      securityPolling();
    };
  }

  /**
   * Connect to WebSocket for real-time updates
   */
  private connectWebSocket(): void {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = 'localhost:8000'
      const wsUrl = `${protocol}//${host}/api/dashboard/ws/dashboard`
      console.log('🔌 Connecting to WebSocket:', wsUrl);
      
      this.webSocket = new WebSocket(wsUrl);
      
      this.webSocket.onopen = () => {
        console.log('✅ WebSocket connected successfully');
        this.reconnectAttempts = 0;
        
        // Send initial ping
        this.sendWebSocketMessage({ type: 'ping' });
      };
      
      this.webSocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleWebSocketMessage(message);
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error);
        }
      };
      
      this.webSocket.onclose = () => {
        console.log('🔌 WebSocket connection closed');
        this.scheduleReconnect();
      };
      
      this.webSocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.scheduleReconnect();
      };
      
    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleWebSocketMessage(message: any): void {
    switch (message.type) {
      case 'dashboard_update':
      case 'dashboard_initial':
        if (message.data) {
          // Transform the message data to match our LiveDashboardData format
          this.liveData = {
            metrics: message.data.metrics || {},
            agent_activities: message.data.agent_activities || [],
            project_snapshots: message.data.project_snapshots || [],
            conflict_snapshots: message.data.conflict_snapshots || []
          };
          this.lastFetch = Date.now();
          
          console.log('📡 Real-time data updated via WebSocket:', {
            active_agents: this.liveData.metrics.active_agents,
            active_projects: this.liveData.metrics.active_projects,
            system_status: this.liveData.metrics.system_status
          });
          
          // Emit update event
          this.emit('liveDataUpdated', this.liveData);
          
          // Also update performance metrics if included
          if (message.data.performance_metrics) {
            this.emit('performanceMetricsUpdated', message.data.performance_metrics);
          }
        }
        break;
        
      case 'performance_update':
        if (message.data) {
          console.log('📊 Performance metrics updated via WebSocket');
          this.emit('performanceMetricsUpdated', message.data);
        }
        break;
        
      case 'security_update':
        if (message.data) {
          console.log('🔐 Security metrics updated via WebSocket');
          this.emit('securityMetricsUpdated', message.data);
        }
        break;
        
      case 'pong':
        console.log('🏓 WebSocket pong received');
        break;
        
      default:
        console.log('📦 Unknown WebSocket message type:', message.type);
    }
  }

  /**
   * Send message through WebSocket
   */
  private sendWebSocketMessage(message: any): void {
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(JSON.stringify(message));
    }
  }

  /**
   * Schedule WebSocket reconnection
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn('⚠️ Max WebSocket reconnection attempts reached, falling back to polling');
      return;
    }
    
    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.pow(2, Math.min(this.reconnectAttempts - 1, 3)); // Exponential backoff
    
    console.log(`🔄 Scheduling WebSocket reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (!this.webSocket || this.webSocket.readyState === WebSocket.CLOSED) {
        this.connectWebSocket();
      }
    }, delay);
  }

  /**
   * Disconnect WebSocket
   */
  private disconnectWebSocket(): void {
    if (this.webSocket) {
      console.log('🔌 Disconnecting WebSocket');
      this.webSocket.close();
      this.webSocket = null;
    }
  }

  /**
   * Mock write operations for services that need them
   * Since we don't have write endpoints, these return success
   */
  async mockWriteOperation(operation: string, data: any): Promise<any> {
    console.log(`🔧 Mock ${operation} operation:`, data);
    
    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Return success response with the data plus some generated fields
    return {
      ...data,
      id: data.id || `mock-${Date.now()}`,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      status: 'success'
    };
  }
}

// Export singleton instance
export const backendAdapter = new BackendAdapter();