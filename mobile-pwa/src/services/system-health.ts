/**
 * System Health Service for LeanVibe Agent Hive
 * 
 * Provides real-time system health monitoring including:
 * - Component health status (database, redis, orchestrator, agents)
 * - System metrics (CPU, memory, disk, network)
 * - Health indicator states for UI components
 * - Automatic polling with smart intervals
 * - Health change event emission
 */

import { BaseService } from './base-service';
import type {
  SystemHealth,
  ComponentHealth,
  SystemMetrics,
  ServiceConfig,
  Subscription,
  EventListener
} from '../types/api';

export interface HealthSummary {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  components: {
    healthy: number;
    degraded: number;
    unhealthy: number;
  };
  alerts: HealthAlert[];
}

export interface HealthAlert {
  id: string;
  component: string;
  severity: 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
}

export class SystemHealthService extends BaseService {
  private pollingStopFn: (() => void) | null = null;
  private currentHealth: SystemHealth | null = null;
  private healthHistory: SystemHealth[] = [];
  private maxHistorySize = 100;

  constructor(config: Partial<ServiceConfig> = {}) {
    super({
      pollingInterval: 10000, // 10 seconds for health checks
      cacheTimeout: 5000, // 5 second cache for health
      ...config
    });
  }

  // ===== PUBLIC API =====

  /**
   * Get current system health status
   */
  async getSystemHealth(fromCache = true): Promise<SystemHealth> {
    const cacheKey = 'system_health';
    
    try {
      const health = await this.get<SystemHealth>(
        '/health',
        {},
        fromCache ? cacheKey : undefined
      );

      this.updateCurrentHealth(health);
      return health;

    } catch (error) {
      // Return degraded status if health check fails
      const fallbackHealth: SystemHealth = {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        components: {
          database: { status: 'unhealthy', lastCheck: new Date().toISOString() },
          redis: { status: 'unhealthy', lastCheck: new Date().toISOString() },
          orchestrator: { status: 'unhealthy', lastCheck: new Date().toISOString() },
          agents: { status: 'unhealthy', lastCheck: new Date().toISOString() }
        },
        metrics: {
          cpu_usage: 0,
          memory_usage: 0,
          disk_usage: 0,
          network_io: { in: 0, out: 0 },
          active_connections: 0,
          uptime: 0
        }
      };

      this.updateCurrentHealth(fallbackHealth);
      throw error;
    }
  }

  /**
   * Get health summary for dashboard display
   */
  getHealthSummary(): HealthSummary {
    if (!this.currentHealth) {
      return {
        overall: 'unhealthy',
        components: { healthy: 0, degraded: 0, unhealthy: 0 },
        alerts: []
      };
    }

    const components = Object.values(this.currentHealth.components);
    const summary = {
      healthy: components.filter(c => c.status === 'healthy').length,
      degraded: components.filter(c => c.status === 'degraded').length,
      unhealthy: components.filter(c => c.status === 'unhealthy').length
    };

    const alerts = this.generateHealthAlerts();

    return {
      overall: this.currentHealth.status,
      components: summary,
      alerts
    };
  }

  /**
   * Get specific component health
   */
  getComponentHealth(componentName: string): ComponentHealth | null {
    return this.currentHealth?.components[componentName] || null;
  }

  /**
   * Get system metrics
   */
  getSystemMetrics(): SystemMetrics | null {
    return this.currentHealth?.metrics || null;
  }

  /**
   * Get health history for trend analysis
   */
  getHealthHistory(limit?: number): SystemHealth[] {
    const history = [...this.healthHistory];
    return limit ? history.slice(-limit) : history;
  }

  /**
   * Get health trend analysis
   */
  getHealthTrend(minutes = 60): {
    trend: 'improving' | 'stable' | 'degrading';
    score: number;
    samples: number;
  } {
    const cutoff = Date.now() - (minutes * 60 * 1000);
    const recentHistory = this.healthHistory.filter(h => 
      new Date(h.timestamp).getTime() > cutoff
    );

    if (recentHistory.length < 2) {
      return { trend: 'stable', score: 0, samples: recentHistory.length };
    }

    // Calculate health scores
    const scores = recentHistory.map(h => this.calculateHealthScore(h));
    const firstScore = scores[0];
    const lastScore = scores[scores.length - 1];
    const scoreDiff = lastScore - firstScore;

    let trend: 'improving' | 'stable' | 'degrading';
    if (scoreDiff > 5) trend = 'improving';
    else if (scoreDiff < -5) trend = 'degrading';
    else trend = 'stable';

    return {
      trend,
      score: lastScore,
      samples: scores.length
    };
  }

  // ===== REAL-TIME MONITORING =====

  /**
   * Start real-time health monitoring
   */
  startMonitoring(): void {
    if (this.pollingStopFn) {
      this.stopMonitoring();
    }

    this.pollingStopFn = this.startPolling(async () => {
      try {
        await this.getSystemHealth(false); // Don't use cache for polling
      } catch (error) {
        // Polling errors are handled by base class
      }
    }, this.config.pollingInterval);

    this.emit('monitoringStarted');
  }

  /**
   * Stop real-time health monitoring
   */
  stopMonitoring(): void {
    if (this.pollingStopFn) {
      this.pollingStopFn();
      this.pollingStopFn = null;
      this.emit('monitoringStopped');
    }
  }

  /**
   * Check if monitoring is active
   */
  isMonitoring(): boolean {
    return this.pollingStopFn !== null;
  }

  // ===== EVENT SUBSCRIPTIONS =====

  public onHealthChange(listener: EventListener<SystemHealth>): Subscription {
    return this.subscribe('healthChanged', listener);
  }

  public onHealthAlert(listener: EventListener<HealthAlert>): Subscription {
    return this.subscribe('healthAlert', listener);
  }

  public onMonitoringStarted(listener: EventListener<void>): Subscription {
    return this.subscribe('monitoringStarted', listener);
  }

  public onMonitoringStopped(listener: EventListener<void>): Subscription {
    return this.subscribe('monitoringStopped', listener);
  }

  // ===== PRIVATE METHODS =====

  private updateCurrentHealth(health: SystemHealth): void {
    const previousHealth = this.currentHealth;
    this.currentHealth = health;

    // Add to history
    this.healthHistory.push(health);
    if (this.healthHistory.length > this.maxHistorySize) {
      this.healthHistory.shift();
    }

    // Emit change event
    this.emit('healthChanged', health);

    // Check for new alerts
    if (previousHealth) {
      const newAlerts = this.detectHealthAlerts(previousHealth, health);
      newAlerts.forEach(alert => this.emit('healthAlert', alert));
    }
  }

  private calculateHealthScore(health: SystemHealth): number {
    const weights = {
      database: 25,
      redis: 20,
      orchestrator: 30,
      agents: 25
    };

    let totalScore = 0;
    let totalWeight = 0;

    for (const [component, weight] of Object.entries(weights)) {
      const componentHealth = health.components[component];
      if (componentHealth) {
        let score = 0;
        switch (componentHealth.status) {
          case 'healthy': score = 100; break;
          case 'degraded': score = 50; break;
          case 'unhealthy': score = 0; break;
        }
        totalScore += score * weight;
        totalWeight += weight;
      }
    }

    return totalWeight > 0 ? Math.round(totalScore / totalWeight) : 0;
  }

  private generateHealthAlerts(): HealthAlert[] {
    if (!this.currentHealth) return [];

    const alerts: HealthAlert[] = [];

    // Check component health
    for (const [name, component] of Object.entries(this.currentHealth.components)) {
      if (component.status === 'unhealthy') {
        alerts.push({
          id: `${name}_unhealthy`,
          component: name,
          severity: 'critical',
          message: `${name} component is unhealthy`,
          timestamp: component.lastCheck
        });
      } else if (component.status === 'degraded') {
        alerts.push({
          id: `${name}_degraded`,
          component: name,
          severity: 'warning',
          message: `${name} component is degraded`,
          timestamp: component.lastCheck
        });
      }
    }

    // Check system metrics
    const metrics = this.currentHealth.metrics;
    if (metrics.cpu_usage > 90) {
      alerts.push({
        id: 'high_cpu',
        component: 'system',
        severity: 'error',
        message: `High CPU usage: ${metrics.cpu_usage.toFixed(1)}%`,
        timestamp: this.currentHealth.timestamp
      });
    }

    if (metrics.memory_usage > 90) {
      alerts.push({
        id: 'high_memory',
        component: 'system',
        severity: 'error',
        message: `High memory usage: ${metrics.memory_usage.toFixed(1)}%`,
        timestamp: this.currentHealth.timestamp
      });
    }

    if (metrics.disk_usage > 85) {
      alerts.push({
        id: 'high_disk',
        component: 'system',
        severity: 'warning',
        message: `High disk usage: ${metrics.disk_usage.toFixed(1)}%`,
        timestamp: this.currentHealth.timestamp
      });
    }

    return alerts;
  }

  private detectHealthAlerts(previous: SystemHealth, current: SystemHealth): HealthAlert[] {
    const alerts: HealthAlert[] = [];

    // Detect component status changes
    for (const [name, component] of Object.entries(current.components)) {
      const prevComponent = previous.components[name];
      
      if (prevComponent && prevComponent.status !== component.status) {
        if (component.status === 'unhealthy' && prevComponent.status !== 'unhealthy') {
          alerts.push({
            id: `${name}_became_unhealthy`,
            component: name,
            severity: 'critical',
            message: `${name} component became unhealthy`,
            timestamp: component.lastCheck
          });
        } else if (component.status === 'healthy' && prevComponent.status === 'unhealthy') {
          alerts.push({
            id: `${name}_recovered`,
            component: name,
            severity: 'warning',
            message: `${name} component recovered`,
            timestamp: component.lastCheck
          });
        }
      }
    }

    return alerts;
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.stopMonitoring();
    super.destroy();
  }
}

// Singleton instance
let systemHealthService: SystemHealthService | null = null;

export function getSystemHealthService(config?: Partial<ServiceConfig>): SystemHealthService {
  if (!systemHealthService) {
    systemHealthService = new SystemHealthService(config);
  }
  return systemHealthService;
}

export function resetSystemHealthService(): void {
  if (systemHealthService) {
    systemHealthService.destroy();
    systemHealthService = null;
  }
}