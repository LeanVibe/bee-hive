/**
 * Metrics Service for LeanVibe Agent Hive
 * 
 * Provides comprehensive performance monitoring and analytics including:
 * - Real-time system performance metrics
 * - Agent performance analytics and trends
 * - Resource utilization monitoring
 * - Historical data for trend analysis
 * - Performance alerts and thresholds
 * - Dashboard-ready chart data
 */

import { BaseService } from './base-service';
import type {
  SystemPerformanceMetrics,
  AgentMetrics,
  MetricDataPoint,
  MetricSeries,
  AgentPerformanceMetrics,
  ServiceConfig,
  Subscription,
  EventListener
} from '../types/api';

export interface PerformanceSnapshot {
  timestamp: string;
  cpu: number;
  memory: number;
  disk: number;
  network: { in: number; out: number };
  agents: {
    total: number;
    active: number;
    busy: number;
    idle: number;
  };
  tasks: {
    pending: number;
    inProgress: number;
    completed: number;
    failed: number;
  };
}

export interface PerformanceTrend {
  metric: string;
  trend: 'increasing' | 'decreasing' | 'stable';
  change: number; // percentage change
  timeframe: string;
  significance: 'high' | 'medium' | 'low';
}

export interface PerformanceAlert {
  id: string;
  metric: string;
  threshold: number;
  currentValue: number;
  severity: 'warning' | 'critical';
  message: string;
  timestamp: string;
  resolved: boolean;
}

export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    color: string;
    fill?: boolean;
  }[];
}

export class MetricsService extends BaseService {
  private performanceHistory: PerformanceSnapshot[] = [];
  private agentMetricsHistory: Map<string, AgentPerformanceMetrics[]> = new Map();
  private pollingStopFn: (() => void) | null = null;
  private alerts: Map<string, PerformanceAlert> = new Map();
  private maxHistorySize = 500; // Keep last 500 data points

  // Performance thresholds
  private readonly thresholds = {
    cpu: { warning: 80, critical: 95 },
    memory: { warning: 85, critical: 95 },
    disk: { warning: 85, critical: 95 },
    response_time: { warning: 2000, critical: 5000 },
    error_rate: { warning: 5, critical: 10 }
  };

  constructor(config: Partial<ServiceConfig> = {}) {
    super({
      pollingInterval: 10000, // 10 seconds for metrics
      cacheTimeout: 5000, // 5 second cache for metrics
      ...config
    });
  }

  // ===== SYSTEM PERFORMANCE METRICS =====

  /**
   * Get current system performance snapshot
   */
  async getCurrentPerformance(): Promise<PerformanceSnapshot> {
    try {
      // In real implementation, this would call multiple API endpoints:
      // - /api/v1/metrics/system
      // - /api/v1/agents/status
      // - /api/v1/tasks (with status filters)
      
      // Simulated data for demonstration
      const snapshot: PerformanceSnapshot = {
        timestamp: new Date().toISOString(),
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        disk: Math.random() * 100,
        network: {
          in: Math.random() * 1000,
          out: Math.random() * 1000
        },
        agents: {
          total: 5,
          active: Math.floor(Math.random() * 5) + 1,
          busy: Math.floor(Math.random() * 3),
          idle: Math.floor(Math.random() * 2)
        },
        tasks: {
          pending: Math.floor(Math.random() * 10),
          inProgress: Math.floor(Math.random() * 5),
          completed: Math.floor(Math.random() * 20) + 10,
          failed: Math.floor(Math.random() * 3)
        }
      };

      // Add to history
      this.addPerformanceSnapshot(snapshot);
      
      // Check for alerts
      this.checkPerformanceAlerts(snapshot);

      this.emit('performanceUpdated', snapshot);
      return snapshot;

    } catch (error) {
      this.emit('performanceLoadFailed', error);
      throw error;
    }
  }

  /**
   * Get system performance metrics over time
   */
  async getSystemMetrics(timeframe: '1h' | '6h' | '24h' | '7d' = '1h'): Promise<SystemPerformanceMetrics> {
    try {
      const now = Date.now();
      let cutoff: number;
      
      switch (timeframe) {
        case '1h': cutoff = now - 60 * 60 * 1000; break;
        case '6h': cutoff = now - 6 * 60 * 60 * 1000; break;
        case '24h': cutoff = now - 24 * 60 * 60 * 1000; break;
        case '7d': cutoff = now - 7 * 24 * 60 * 60 * 1000; break;
      }

      const relevantData = this.performanceHistory.filter(
        snapshot => new Date(snapshot.timestamp).getTime() > cutoff
      );

      const metrics: SystemPerformanceMetrics = {
        cpu: this.createMetricSeries('CPU Usage', relevantData.map(d => ({
          timestamp: d.timestamp,
          value: d.cpu
        })), '%'),
        memory: this.createMetricSeries('Memory Usage', relevantData.map(d => ({
          timestamp: d.timestamp,
          value: d.memory
        })), '%'),
        disk: this.createMetricSeries('Disk Usage', relevantData.map(d => ({
          timestamp: d.timestamp,
          value: d.disk
        })), '%'),
        network: {
          in: this.createMetricSeries('Network In', relevantData.map(d => ({
            timestamp: d.timestamp,
            value: d.network.in
          })), 'KB/s'),
          out: this.createMetricSeries('Network Out', relevantData.map(d => ({
            timestamp: d.timestamp,
            value: d.network.out
          })), 'KB/s')
        },
        agents: {
          active_count: this.createMetricSeries('Active Agents', relevantData.map(d => ({
            timestamp: d.timestamp,
            value: d.agents.active
          })), 'count'),
          task_completion_rate: this.createMetricSeries('Task Completion Rate', relevantData.map(d => ({
            timestamp: d.timestamp,
            value: d.tasks.completed / (d.tasks.completed + d.tasks.failed + d.tasks.inProgress + d.tasks.pending) * 100
          })), '%'),
          error_rate: this.createMetricSeries('Error Rate', relevantData.map(d => ({
            timestamp: d.timestamp,
            value: d.tasks.failed / (d.tasks.completed + d.tasks.failed) * 100 || 0
          })), '%')
        }
      };

      this.emit('systemMetricsLoaded', metrics);
      return metrics;

    } catch (error) {
      this.emit('systemMetricsLoadFailed', { timeframe, error });
      throw error;
    }
  }

  // ===== AGENT METRICS =====

  /**
   * Get metrics for a specific agent
   */
  async getAgentMetrics(agentId: string, timeframe: '1h' | '6h' | '24h' = '1h'): Promise<AgentMetrics> {
    try {
      // In real implementation: await this.get<AgentMetrics>(`/api/v1/agents/${agentId}/metrics?timeframe=${timeframe}`)
      
      const history = this.agentMetricsHistory.get(agentId) || [];
      const now = Date.now();
      let cutoff: number;
      
      switch (timeframe) {
        case '1h': cutoff = now - 60 * 60 * 1000; break;
        case '6h': cutoff = now - 6 * 60 * 60 * 1000; break;
        case '24h': cutoff = now - 24 * 60 * 60 * 1000; break;
      }

      // Simulate agent metrics
      const currentMetrics: AgentPerformanceMetrics = {
        tasks_completed: Math.floor(Math.random() * 50),
        tasks_failed: Math.floor(Math.random() * 5),
        average_completion_time: Math.random() * 3600, // seconds
        cpu_usage: Math.random() * 100,
        memory_usage: Math.random() * 100,
        success_rate: 85 + Math.random() * 15,
        uptime: Math.random() * 86400 // seconds
      };

      const metrics: AgentMetrics = {
        agent_id: agentId,
        role: 'backend_developer' as any, // Would come from agent data
        performance: currentMetrics,
        resource_usage: {
          cpu: this.createMetricSeries('CPU Usage', [], '%'),
          memory: this.createMetricSeries('Memory Usage', [], '%')
        },
        task_metrics: {
          completed: this.createMetricSeries('Completed Tasks', [], 'count'),
          failed: this.createMetricSeries('Failed Tasks', [], 'count'),
          completion_time: this.createMetricSeries('Completion Time', [], 'seconds')
        }
      };

      this.emit('agentMetricsLoaded', metrics);
      return metrics;

    } catch (error) {
      this.emit('agentMetricsLoadFailed', { agentId, timeframe, error });
      throw error;
    }
  }

  /**
   * Get metrics for all agents
   */
  async getAllAgentMetrics(timeframe: '1h' | '6h' | '24h' = '1h'): Promise<AgentMetrics[]> {
    try {
      // In real implementation: await this.get<AgentMetrics[]>(`/api/v1/agents/metrics?timeframe=${timeframe}`)
      
      // Return empty array for simulation
      return [];

    } catch (error) {
      this.emit('allAgentMetricsLoadFailed', { timeframe, error });
      throw error;
    }
  }

  // ===== PERFORMANCE ANALYTICS =====

  /**
   * Get performance trends analysis
   */
  getPerformanceTrends(timeframe: '1h' | '6h' | '24h' | '7d' = '24h'): PerformanceTrend[] {
    if (this.performanceHistory.length < 10) {
      return []; // Need sufficient data for trend analysis
    }

    const now = Date.now();
    let cutoff: number;
    
    switch (timeframe) {
      case '1h': cutoff = now - 60 * 60 * 1000; break;
      case '6h': cutoff = now - 6 * 60 * 60 * 1000; break;
      case '24h': cutoff = now - 24 * 60 * 60 * 1000; break;
      case '7d': cutoff = now - 7 * 24 * 60 * 60 * 1000; break;
    }

    const relevantData = this.performanceHistory.filter(
      snapshot => new Date(snapshot.timestamp).getTime() > cutoff
    );

    if (relevantData.length < 5) {
      return [];
    }

    const trends: PerformanceTrend[] = [];

    // Analyze CPU trend
    trends.push(this.analyzeTrend('cpu', relevantData.map(d => d.cpu), timeframe));
    
    // Analyze Memory trend
    trends.push(this.analyzeTrend('memory', relevantData.map(d => d.memory), timeframe));
    
    // Analyze Task completion rate trend
    const completionRates = relevantData.map(d => 
      d.tasks.completed / (d.tasks.completed + d.tasks.failed + d.tasks.inProgress + d.tasks.pending) * 100
    );
    trends.push(this.analyzeTrend('task_completion_rate', completionRates, timeframe));

    return trends.filter(trend => trend.significance !== 'low');
  }

  /**
   * Get current performance alerts
   */
  getPerformanceAlerts(): PerformanceAlert[] {
    return Array.from(this.alerts.values())
      .filter(alert => !alert.resolved)
      .sort((a, b) => {
        // Sort by severity (critical first) then by timestamp
        if (a.severity === 'critical' && b.severity === 'warning') return -1;
        if (a.severity === 'warning' && b.severity === 'critical') return 1;
        return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
      });
  }

  /**
   * Acknowledge a performance alert
   */
  acknowledgeAlert(alertId: string): void {
    const alert = this.alerts.get(alertId);
    if (alert) {
      alert.resolved = true;
      this.alerts.set(alertId, alert);
      this.emit('alertAcknowledged', alert);
    }
  }

  // ===== CHART DATA FORMATTING =====

  /**
   * Get chart data for system overview
   */
  getSystemOverviewChartData(timeframe: '1h' | '6h' | '24h' = '1h'): ChartData {
    const now = Date.now();
    let cutoff: number;
    
    switch (timeframe) {
      case '1h': cutoff = now - 60 * 60 * 1000; break;
      case '6h': cutoff = now - 6 * 60 * 60 * 1000; break;
      case '24h': cutoff = now - 24 * 60 * 60 * 1000; break;
    }

    const relevantData = this.performanceHistory.filter(
      snapshot => new Date(snapshot.timestamp).getTime() > cutoff
    );

    const labels = relevantData.map(d => 
      new Date(d.timestamp).toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
      })
    );

    return {
      labels,
      datasets: [
        {
          label: 'CPU Usage',
          data: relevantData.map(d => d.cpu),
          color: '#ef4444',
          fill: false
        },
        {
          label: 'Memory Usage',
          data: relevantData.map(d => d.memory),
          color: '#3b82f6',
          fill: false
        },
        {
          label: 'Active Agents',
          data: relevantData.map(d => d.agents.active * 20), // Scale for visibility
          color: '#10b981',
          fill: false
        }
      ]
    };
  }

  /**
   * Get chart data for task metrics
   */
  getTaskMetricsChartData(): ChartData {
    const recent = this.performanceHistory.slice(-20); // Last 20 data points
    
    return {
      labels: recent.map((_, i) => `T-${20 - i}`),
      datasets: [
        {
          label: 'Completed',
          data: recent.map(d => d.tasks.completed),
          color: '#10b981',
          fill: true
        },
        {
          label: 'In Progress',
          data: recent.map(d => d.tasks.inProgress),
          color: '#f59e0b',
          fill: true
        },
        {
          label: 'Failed',
          data: recent.map(d => d.tasks.failed),
          color: '#ef4444',
          fill: true
        }
      ]
    };
  }

  // ===== REAL-TIME MONITORING =====

  /**
   * Start real-time metrics collection
   */
  startMonitoring(): void {
    if (this.pollingStopFn) {
      this.stopMonitoring();
    }

    this.pollingStopFn = this.startPolling(async () => {
      try {
        await this.getCurrentPerformance();
      } catch (error) {
        // Polling errors are handled by base class
      }
    }, this.config.pollingInterval);

    this.emit('metricsMonitoringStarted');
  }

  /**
   * Stop real-time metrics collection
   */
  stopMonitoring(): void {
    if (this.pollingStopFn) {
      this.pollingStopFn();
      this.pollingStopFn = null;
      this.emit('metricsMonitoringStopped');
    }
  }

  /**
   * Check if monitoring is active
   */
  isMonitoring(): boolean {
    return this.pollingStopFn !== null;
  }

  // ===== PRIVATE METHODS =====

  private addPerformanceSnapshot(snapshot: PerformanceSnapshot): void {
    this.performanceHistory.push(snapshot);
    
    // Keep only recent history
    if (this.performanceHistory.length > this.maxHistorySize) {
      this.performanceHistory.shift();
    }
  }

  private createMetricSeries(name: string, data: MetricDataPoint[], unit: string): MetricSeries {
    return {
      name,
      data,
      unit,
      description: `${name} over time`
    };
  }

  private analyzeTrend(metric: string, values: number[], timeframe: string): PerformanceTrend {
    if (values.length < 2) {
      return {
        metric,
        trend: 'stable',
        change: 0,
        timeframe,
        significance: 'low'
      };
    }

    const first = values.slice(0, Math.floor(values.length / 3));
    const last = values.slice(-Math.floor(values.length / 3));
    
    const firstAvg = first.reduce((a, b) => a + b, 0) / first.length;
    const lastAvg = last.reduce((a, b) => a + b, 0) / last.length;
    
    const change = ((lastAvg - firstAvg) / firstAvg) * 100;
    
    let trend: 'increasing' | 'decreasing' | 'stable';
    let significance: 'high' | 'medium' | 'low';
    
    if (Math.abs(change) < 5) {
      trend = 'stable';
      significance = 'low';
    } else if (change > 0) {
      trend = 'increasing';
      significance = Math.abs(change) > 20 ? 'high' : 'medium';
    } else {
      trend = 'decreasing';
      significance = Math.abs(change) > 20 ? 'high' : 'medium';
    }

    return {
      metric,
      trend,
      change: Math.round(change * 100) / 100,
      timeframe,
      significance
    };
  }

  private checkPerformanceAlerts(snapshot: PerformanceSnapshot): void {
    // Check CPU alert
    this.checkThreshold('cpu', snapshot.cpu, snapshot.timestamp);
    
    // Check Memory alert
    this.checkThreshold('memory', snapshot.memory, snapshot.timestamp);
    
    // Check Disk alert
    this.checkThreshold('disk', snapshot.disk, snapshot.timestamp);
    
    // Check task error rate
    const errorRate = snapshot.tasks.failed / (snapshot.tasks.completed + snapshot.tasks.failed) * 100 || 0;
    this.checkThreshold('error_rate', errorRate, snapshot.timestamp);
  }

  private checkThreshold(metric: string, value: number, timestamp: string): void {
    const thresholds = this.thresholds[metric as keyof typeof this.thresholds];
    if (!thresholds) return;

    const alertId = `${metric}_${Date.now()}`;
    let severity: 'warning' | 'critical' | null = null;
    let threshold = 0;

    if (value >= thresholds.critical) {
      severity = 'critical';
      threshold = thresholds.critical;
    } else if (value >= thresholds.warning) {
      severity = 'warning';
      threshold = thresholds.warning;
    }

    if (severity) {
      const alert: PerformanceAlert = {
        id: alertId,
        metric,
        threshold,
        currentValue: value,
        severity,
        message: `${metric.toUpperCase()} usage is ${severity}: ${value.toFixed(1)}% (threshold: ${threshold}%)`,
        timestamp,
        resolved: false
      };

      this.alerts.set(alertId, alert);
      this.emit('performanceAlert', alert);
    }
  }

  // ===== EVENT SUBSCRIPTIONS =====

  public onPerformanceUpdated(listener: EventListener<PerformanceSnapshot>): Subscription {
    return this.subscribe('performanceUpdated', listener);
  }

  public onSystemMetricsLoaded(listener: EventListener<SystemPerformanceMetrics>): Subscription {
    return this.subscribe('systemMetricsLoaded', listener);
  }

  public onAgentMetricsLoaded(listener: EventListener<AgentMetrics>): Subscription {
    return this.subscribe('agentMetricsLoaded', listener);
  }

  public onPerformanceAlert(listener: EventListener<PerformanceAlert>): Subscription {
    return this.subscribe('performanceAlert', listener);
  }

  public onAlertAcknowledged(listener: EventListener<PerformanceAlert>): Subscription {
    return this.subscribe('alertAcknowledged', listener);
  }

  public onMetricsMonitoringStarted(listener: EventListener<void>): Subscription {
    return this.subscribe('metricsMonitoringStarted', listener);
  }

  public onMetricsMonitoringStopped(listener: EventListener<void>): Subscription {
    return this.subscribe('metricsMonitoringStopped', listener);
  }

  // ===== UTILITY METHODS =====

  /**
   * Get performance history
   */
  getPerformanceHistory(limit?: number): PerformanceSnapshot[] {
    const history = [...this.performanceHistory];
    return limit ? history.slice(-limit) : history;
  }

  /**
   * Clear performance history
   */
  clearHistory(): void {
    this.performanceHistory = [];
    this.agentMetricsHistory.clear();
    this.emit('historyCleared');
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.stopMonitoring();
    this.performanceHistory = [];
    this.agentMetricsHistory.clear();
    this.alerts.clear();
    super.destroy();
  }
}

// Singleton instance
let metricsService: MetricsService | null = null;

export function getMetricsService(config?: Partial<ServiceConfig>): MetricsService {
  if (!metricsService) {
    metricsService = new MetricsService(config);
  }
  return metricsService;
}

export function resetMetricsService(): void {
  if (metricsService) {
    metricsService.destroy();
    metricsService = null;
  }
}