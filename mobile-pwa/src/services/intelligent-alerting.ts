/**
 * Intelligent Alerting System for LeanVibe Agent Hive
 * 
 * Features:
 * - Smart threshold detection with machine learning
 * - Anomaly detection using statistical analysis
 * - Predictive alerting for performance degradation
 * - Alert correlation and deduplication
 * - Contextual recommendations and auto-remediation
 * - Adaptive thresholds based on historical patterns
 */

import { BaseService } from './base-service';
import type {
  ServiceConfig,
  Subscription,
  EventListener
} from '../types/api';

export interface IntelligentAlert {
  id: string;
  type: 'performance' | 'system' | 'agent' | 'predictive' | 'anomaly';
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  metric: string;
  currentValue: number;
  threshold?: number;
  confidence: number; // 0-1 scale
  correlatedAlerts?: string[];
  remediation?: RemediationAction[];
  context: AlertContext;
  timestamp: number;
  resolved: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: number;
  tags: string[];
}

export interface RemediationAction {
  id: string;
  description: string;
  action: 'auto' | 'manual' | 'user_confirm';
  priority: number;
  command?: string;
  estimatedImpact: 'low' | 'medium' | 'high';
  risk: 'safe' | 'moderate' | 'risky';
}

export interface AlertContext {
  source: string;
  component?: string;
  agent?: string;
  environment: 'development' | 'staging' | 'production';
  userImpact: 'none' | 'low' | 'medium' | 'high';
  businessImpact: 'none' | 'low' | 'medium' | 'high';
  additionalData?: Record<string, any>;
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  metric: string;
  condition: AlertCondition;
  severity: IntelligentAlert['severity'];
  enabled: boolean;
  adaptiveThresholds: boolean;
  cooldownPeriod: number; // milliseconds
  tags: string[];
}

export interface AlertCondition {
  type: 'threshold' | 'anomaly' | 'trend' | 'pattern' | 'composite';
  operator?: 'gt' | 'lt' | 'eq' | 'ne' | 'between';
  value?: number;
  lookbackPeriod?: number; // milliseconds
  sensitivity?: number; // 0-1 scale for anomaly detection
  aggregation?: 'avg' | 'max' | 'min' | 'sum' | 'count';
  composite?: {
    logic: 'AND' | 'OR';
    conditions: AlertCondition[];
  };
}

export interface AnomalyDetectionResult {
  isAnomaly: boolean;
  score: number; // How anomalous (0-1)
  expectedRange: { min: number; max: number };
  confidence: number;
  factors: string[]; // What makes it anomalous
}

export interface AlertingStats {
  totalAlerts: number;
  activeAlerts: number;
  resolvedAlerts: number;
  falsePositiveRate: number;
  averageResolutionTime: number;
  alertsByType: Record<string, number>;
  alertsBySeverity: Record<string, number>;
  topAlertingComponents: Array<{ component: string; count: number }>;
  alertingTrends: {
    hourly: number[];
    daily: number[];
    weekly: number[];
  };
}

class SimpleAnomalyDetector {
  private historicalData: Map<string, number[]> = new Map();
  private readonly maxHistorySize = 1000;

  addDataPoint(metric: string, value: number): void {
    if (!this.historicalData.has(metric)) {
      this.historicalData.set(metric, []);
    }
    
    const history = this.historicalData.get(metric)!;
    history.push(value);
    
    if (history.length > this.maxHistorySize) {
      history.shift();
    }
  }

  detectAnomaly(metric: string, value: number, sensitivity = 0.5): AnomalyDetectionResult {
    const history = this.historicalData.get(metric);
    
    if (!history || history.length < 10) {
      return {
        isAnomaly: false,
        score: 0,
        expectedRange: { min: value, max: value },
        confidence: 0,
        factors: []
      };
    }

    const mean = history.reduce((sum, v) => sum + v, 0) / history.length;
    const variance = history.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / history.length;
    const stdDev = Math.sqrt(variance);

    // Z-score based anomaly detection
    const zScore = Math.abs(value - mean) / stdDev;
    const threshold = this.sensitivityToZScore(sensitivity);
    
    const isAnomaly = zScore > threshold;
    const score = Math.min(1, zScore / (threshold * 2));
    
    const factors: string[] = [];
    if (value > mean + (2 * stdDev)) factors.push('Value significantly above normal range');
    if (value < mean - (2 * stdDev)) factors.push('Value significantly below normal range');
    if (stdDev > mean * 0.5) factors.push('High variability detected');

    return {
      isAnomaly,
      score,
      expectedRange: {
        min: mean - (2 * stdDev),
        max: mean + (2 * stdDev)
      },
      confidence: Math.min(1, history.length / 100),
      factors
    };
  }

  private sensitivityToZScore(sensitivity: number): number {
    // Convert 0-1 sensitivity to z-score threshold
    // Higher sensitivity = lower threshold
    return 2 + (1 - sensitivity) * 2; // Range: 2-4
  }
}

export class IntelligentAlertingService extends BaseService {
  private alerts: Map<string, IntelligentAlert> = new Map();
  private alertRules: Map<string, AlertRule> = new Map();
  private alertHistory: IntelligentAlert[] = [];
  private anomalyDetector = new SimpleAnomalyDetector();
  private alertingEnabled = true;
  private correlationWindow = 5 * 60 * 1000; // 5 minutes
  private maxHistorySize = 10000;

  // Alert deduplication tracking
  private recentAlerts: Map<string, number> = new Map();
  private suppressionRules: Map<string, number> = new Map();

  constructor(config: Partial<ServiceConfig> = {}) {
    super({
      pollingInterval: 10000, // Check for alerts every 10 seconds
      cacheTimeout: 5000,
      ...config
    });

    this.initializeDefaultRules();
  }

  // ===== ALERT MANAGEMENT =====

  /**
   * Create a new alert
   */
  createAlert(alertData: Omit<IntelligentAlert, 'id' | 'timestamp' | 'resolved'>): IntelligentAlert {
    const alert: IntelligentAlert = {
      id: this.generateAlertId(),
      timestamp: Date.now(),
      resolved: false,
      ...alertData
    };

    // Check for duplicates and correlation
    const existingAlert = this.findSimilarAlert(alert);
    if (existingAlert && this.shouldSuppressDuplicate(alert, existingAlert)) {
      console.log(`🔇 Alert suppressed due to recent similar alert: ${alert.title}`);
      return existingAlert;
    }

    // Add to active alerts
    this.alerts.set(alert.id, alert);
    this.alertHistory.push(alert);

    // Trim history if needed
    if (this.alertHistory.length > this.maxHistorySize) {
      this.alertHistory.shift();
    }

    // Update anomaly detector
    this.anomalyDetector.addDataPoint(alert.metric, alert.currentValue);

    // Find correlated alerts
    alert.correlatedAlerts = this.findCorrelatedAlerts(alert);

    // Generate remediation suggestions
    if (!alert.remediation) {
      alert.remediation = this.generateRemediationActions(alert);
    }

    console.log(`🚨 New alert created [${alert.severity.toUpperCase()}]: ${alert.title}`);
    this.emit('alertCreated', alert);

    // Auto-resolve if configured
    if (this.shouldAutoResolve(alert)) {
      setTimeout(() => this.tryAutoResolve(alert.id), 30000);
    }

    return alert;
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string, resolvedBy?: string): boolean {
    const alert = this.alerts.get(alertId);
    if (!alert || alert.resolved) {
      return false;
    }

    alert.resolved = true;
    alert.acknowledgedBy = resolvedBy;
    alert.acknowledgedAt = Date.now();

    this.alerts.delete(alertId);
    this.emit('alertResolved', alert);

    console.log(`✅ Alert resolved: ${alert.title}`);
    return true;
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string, acknowledgedBy: string): boolean {
    const alert = this.alerts.get(alertId);
    if (!alert) return false;

    alert.acknowledgedBy = acknowledgedBy;
    alert.acknowledgedAt = Date.now();

    this.emit('alertAcknowledged', alert);
    return true;
  }

  // ===== ANOMALY DETECTION =====

  /**
   * Check if a metric value is anomalous
   */
  checkForAnomalies(metric: string, value: number, context: AlertContext): IntelligentAlert | null {
    const result = this.anomalyDetector.detectAnomaly(metric, value, 0.7);
    
    if (result.isAnomaly && result.confidence > 0.5) {
      return this.createAlert({
        type: 'anomaly',
        severity: result.score > 0.8 ? 'critical' : result.score > 0.5 ? 'error' : 'warning',
        title: `Anomaly detected in ${metric}`,
        message: `Unusual value detected: ${value} (expected: ${result.expectedRange.min.toFixed(2)} - ${result.expectedRange.max.toFixed(2)})`,
        metric,
        currentValue: value,
        confidence: result.confidence,
        context,
        tags: ['anomaly', 'auto-detected']
      });
    }

    return null;
  }

  /**
   * Analyze metric for predictive alerts
   */
  analyzePredictivePattern(
    metric: string, 
    values: number[], 
    context: AlertContext
  ): IntelligentAlert | null {
    if (values.length < 5) return null;

    const trend = this.calculateTrend(values);
    const currentValue = values[values.length - 1];
    
    // Predict future value
    const predictedValue = currentValue + (trend * 10); // Predict 10 time units ahead
    
    // Check if prediction crosses critical thresholds
    const rule = this.getAlertRuleForMetric(metric);
    if (rule && rule.condition.type === 'threshold' && rule.condition.value) {
      const threshold = rule.condition.value;
      const willCrossThreshold = 
        (rule.condition.operator === 'gt' && predictedValue > threshold) ||
        (rule.condition.operator === 'lt' && predictedValue < threshold);

      if (willCrossThreshold) {
        const timeToThreshold = Math.abs((threshold - currentValue) / trend);
        
        return this.createAlert({
          type: 'predictive',
          severity: timeToThreshold < 5 ? 'critical' : 'warning',
          title: `Predicted threshold breach in ${metric}`,
          message: `Current trend suggests ${metric} will reach ${predictedValue.toFixed(2)} (threshold: ${threshold}) in ${timeToThreshold.toFixed(1)} time units`,
          metric,
          currentValue,
          threshold,
          confidence: this.calculateTrendConfidence(values),
          context,
          tags: ['predictive', 'trend-analysis'],
          remediation: [{
            id: 'preventive-action',
            description: 'Take preventive action to avoid threshold breach',
            action: 'manual',
            priority: 1,
            estimatedImpact: 'medium',
            risk: 'safe'
          }]
        });
      }
    }

    return null;
  }

  // ===== ALERT RULES =====

  /**
   * Add or update an alert rule
   */
  setAlertRule(rule: AlertRule): void {
    this.alertRules.set(rule.id, rule);
    this.emit('ruleUpdated', rule);
  }

  /**
   * Remove an alert rule
   */
  removeAlertRule(ruleId: string): boolean {
    const removed = this.alertRules.delete(ruleId);
    if (removed) {
      this.emit('ruleRemoved', ruleId);
    }
    return removed;
  }

  /**
   * Get all alert rules
   */
  getAlertRules(): AlertRule[] {
    return Array.from(this.alertRules.values());
  }

  /**
   * Evaluate all rules against current metrics
   */
  evaluateRules(metrics: Record<string, number>, context: AlertContext): IntelligentAlert[] {
    const newAlerts: IntelligentAlert[] = [];

    for (const rule of this.alertRules.values()) {
      if (!rule.enabled) continue;

      const metricValue = metrics[rule.metric];
      if (metricValue === undefined) continue;

      const shouldAlert = this.evaluateCondition(rule.condition, metricValue, rule.metric);
      
      if (shouldAlert && !this.isInCooldown(rule.id)) {
        const alert = this.createAlert({
          type: 'system',
          severity: rule.severity,
          title: rule.name,
          message: `Rule triggered: ${rule.description}`,
          metric: rule.metric,
          currentValue: metricValue,
          threshold: rule.condition.value,
          confidence: 1.0,
          context,
          tags: [...rule.tags, 'rule-based']
        });

        newAlerts.push(alert);
        this.setCooldown(rule.id, rule.cooldownPeriod);
      }
    }

    return newAlerts;
  }

  // ===== STATISTICS AND REPORTING =====

  /**
   * Get alerting statistics
   */
  getAlertingStats(): AlertingStats {
    const now = Date.now();
    const dayMs = 24 * 60 * 60 * 1000;
    const hourMs = 60 * 60 * 1000;

    const recentAlerts = this.alertHistory.filter(a => now - a.timestamp < 7 * dayMs);
    const resolvedAlerts = recentAlerts.filter(a => a.resolved);
    
    const alertsByType: Record<string, number> = {};
    const alertsBySeverity: Record<string, number> = {};
    const componentCounts: Map<string, number> = new Map();

    recentAlerts.forEach(alert => {
      alertsByType[alert.type] = (alertsByType[alert.type] || 0) + 1;
      alertsBySeverity[alert.severity] = (alertsBySeverity[alert.severity] || 0) + 1;
      
      if (alert.context.component) {
        componentCounts.set(
          alert.context.component,
          (componentCounts.get(alert.context.component) || 0) + 1
        );
      }
    });

    const resolutionTimes = resolvedAlerts
      .filter(a => a.acknowledgedAt)
      .map(a => a.acknowledgedAt! - a.timestamp);

    return {
      totalAlerts: recentAlerts.length,
      activeAlerts: this.alerts.size,
      resolvedAlerts: resolvedAlerts.length,
      falsePositiveRate: this.calculateFalsePositiveRate(),
      averageResolutionTime: resolutionTimes.length > 0 
        ? resolutionTimes.reduce((sum, time) => sum + time, 0) / resolutionTimes.length 
        : 0,
      alertsByType,
      alertsBySeverity,
      topAlertingComponents: Array.from(componentCounts.entries())
        .map(([component, count]) => ({ component, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10),
      alertingTrends: this.calculateAlertingTrends()
    };
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): IntelligentAlert[] {
    return Array.from(this.alerts.values())
      .sort((a, b) => {
        // Sort by severity then timestamp
        const severityOrder = { critical: 4, error: 3, warning: 2, info: 1 };
        const severityDiff = severityOrder[b.severity] - severityOrder[a.severity];
        return severityDiff !== 0 ? severityDiff : b.timestamp - a.timestamp;
      });
  }

  /**
   * Get alert history
   */
  getAlertHistory(limit?: number): IntelligentAlert[] {
    const history = [...this.alertHistory].reverse();
    return limit ? history.slice(0, limit) : history;
  }

  // ===== PRIVATE METHODS =====

  private initializeDefaultRules(): void {
    const defaultRules: AlertRule[] = [
      {
        id: 'high_cpu_usage',
        name: 'High CPU Usage',
        description: 'CPU usage exceeds 90%',
        metric: 'cpu_usage',
        condition: {
          type: 'threshold',
          operator: 'gt',
          value: 90
        },
        severity: 'error',
        enabled: true,
        adaptiveThresholds: true,
        cooldownPeriod: 5 * 60 * 1000, // 5 minutes
        tags: ['system', 'performance']
      },
      {
        id: 'low_fps',
        name: 'Low Frame Rate',
        description: 'FPS drops below 20',
        metric: 'fps',
        condition: {
          type: 'threshold',
          operator: 'lt',
          value: 20
        },
        severity: 'warning',
        enabled: true,
        adaptiveThresholds: false,
        cooldownPeriod: 2 * 60 * 1000, // 2 minutes
        tags: ['performance', 'ui']
      },
      {
        id: 'memory_pressure',
        name: 'Memory Pressure',
        description: 'Memory usage exceeds 85%',
        metric: 'memory_usage',
        condition: {
          type: 'threshold',
          operator: 'gt',
          value: 85
        },
        severity: 'warning',
        enabled: true,
        adaptiveThresholds: true,
        cooldownPeriod: 3 * 60 * 1000, // 3 minutes
        tags: ['system', 'memory']
      }
    ];

    defaultRules.forEach(rule => this.setAlertRule(rule));
  }

  private generateAlertId(): string {
    return `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private findSimilarAlert(newAlert: IntelligentAlert): IntelligentAlert | undefined {
    const threshold = 5 * 60 * 1000; // 5 minutes
    const now = Date.now();

    return Array.from(this.alerts.values()).find(alert => 
      alert.metric === newAlert.metric &&
      alert.severity === newAlert.severity &&
      (now - alert.timestamp) < threshold
    );
  }

  private shouldSuppressDuplicate(newAlert: IntelligentAlert, existingAlert: IntelligentAlert): boolean {
    // Suppress if similar alert exists within correlation window
    return (Date.now() - existingAlert.timestamp) < this.correlationWindow;
  }

  private findCorrelatedAlerts(alert: IntelligentAlert): string[] {
    const correlatedIds: string[] = [];
    const correlationWindow = 10 * 60 * 1000; // 10 minutes

    for (const [id, existingAlert] of this.alerts) {
      if (id === alert.id) continue;

      const timeDiff = Math.abs(alert.timestamp - existingAlert.timestamp);
      if (timeDiff < correlationWindow) {
        // Check for correlation patterns
        if (this.isCorrelated(alert, existingAlert)) {
          correlatedIds.push(id);
        }
      }
    }

    return correlatedIds;
  }

  private isCorrelated(alert1: IntelligentAlert, alert2: IntelligentAlert): boolean {
    // Same component or agent
    if (alert1.context.component === alert2.context.component ||
        alert1.context.agent === alert2.context.agent) {
      return true;
    }

    // Related metrics
    const relatedMetrics = [
      ['cpu_usage', 'memory_usage'],
      ['fps', 'render_time'],
      ['network_latency', 'response_time']
    ];

    return relatedMetrics.some(group => 
      group.includes(alert1.metric) && group.includes(alert2.metric)
    );
  }

  private generateRemediationActions(alert: IntelligentAlert): RemediationAction[] {
    const actions: RemediationAction[] = [];

    switch (alert.metric) {
      case 'cpu_usage':
        actions.push({
          id: 'reduce_cpu_load',
          description: 'Reduce CPU intensive operations',
          action: 'auto',
          priority: 1,
          estimatedImpact: 'high',
          risk: 'safe'
        });
        break;

      case 'memory_usage':
        actions.push({
          id: 'garbage_collect',
          description: 'Force garbage collection',
          action: 'auto',
          priority: 1,
          estimatedImpact: 'medium',
          risk: 'safe'
        });
        break;

      case 'fps':
        actions.push({
          id: 'enable_performance_mode',
          description: 'Enable performance mode to reduce visual complexity',
          action: 'user_confirm',
          priority: 1,
          estimatedImpact: 'high',
          risk: 'safe'
        });
        break;
    }

    return actions;
  }

  private shouldAutoResolve(alert: IntelligentAlert): boolean {
    return alert.type === 'anomaly' && alert.confidence < 0.8;
  }

  private async tryAutoResolve(alertId: string): Promise<void> {
    const alert = this.alerts.get(alertId);
    if (!alert || alert.resolved) return;

    // Check if the condition still exists
    const isStillAlerting = await this.verifyAlertCondition(alert);
    if (!isStillAlerting) {
      this.resolveAlert(alertId, 'auto-resolve');
    }
  }

  private async verifyAlertCondition(alert: IntelligentAlert): Promise<boolean> {
    // This would check current metrics against the alert condition
    // For now, return false to allow auto-resolution
    return false;
  }

  private evaluateCondition(condition: AlertCondition, value: number, metric: string): boolean {
    switch (condition.type) {
      case 'threshold':
        if (!condition.value || !condition.operator) return false;
        
        switch (condition.operator) {
          case 'gt': return value > condition.value;
          case 'lt': return value < condition.value;
          case 'eq': return value === condition.value;
          case 'ne': return value !== condition.value;
          case 'between':
            // Assume value is [min, max] for between
            return false; // Simplified for now
          default: return false;
        }

      case 'anomaly':
        const anomalyResult = this.anomalyDetector.detectAnomaly(metric, value, condition.sensitivity || 0.5);
        return anomalyResult.isAnomaly;

      default:
        return false;
    }
  }

  private getAlertRuleForMetric(metric: string): AlertRule | undefined {
    return Array.from(this.alertRules.values()).find(rule => rule.metric === metric);
  }

  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0;

    const n = values.length;
    const sumX = (n * (n - 1)) / 2; // Sum of indices
    const sumY = values.reduce((sum, val) => sum + val, 0);
    const sumXY = values.reduce((sum, val, i) => sum + (i * val), 0);
    const sumXX = (n * (n - 1) * (2 * n - 1)) / 6; // Sum of squares of indices

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return isNaN(slope) ? 0 : slope;
  }

  private calculateTrendConfidence(values: number[]): number {
    if (values.length < 3) return 0;

    // Calculate R-squared for trend line
    const trend = this.calculateTrend(values);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    
    const ssTotal = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
    const ssResidual = values.reduce((sum, val, i) => {
      const predicted = values[0] + (trend * i);
      return sum + Math.pow(val - predicted, 2);
    }, 0);

    const rSquared = 1 - (ssResidual / ssTotal);
    return Math.max(0, Math.min(1, rSquared));
  }

  private isInCooldown(ruleId: string): boolean {
    const lastTriggered = this.suppressionRules.get(ruleId);
    if (!lastTriggered) return false;

    const rule = this.alertRules.get(ruleId);
    if (!rule) return false;

    return (Date.now() - lastTriggered) < rule.cooldownPeriod;
  }

  private setCooldown(ruleId: string, period: number): void {
    this.suppressionRules.set(ruleId, Date.now());
  }

  private calculateFalsePositiveRate(): number {
    // Simplified calculation - would need user feedback in practice
    return 0.05; // Assume 5% false positive rate
  }

  private calculateAlertingTrends(): { hourly: number[]; daily: number[]; weekly: number[] } {
    const now = Date.now();
    const hourMs = 60 * 60 * 1000;
    const dayMs = 24 * hourMs;
    
    const hourly = Array(24).fill(0);
    const daily = Array(7).fill(0);
    const weekly = Array(4).fill(0);

    this.alertHistory.forEach(alert => {
      const age = now - alert.timestamp;
      
      if (age < 24 * hourMs) {
        const hourIndex = Math.floor(age / hourMs);
        hourly[23 - hourIndex]++;
      }
      
      if (age < 7 * dayMs) {
        const dayIndex = Math.floor(age / dayMs);
        daily[6 - dayIndex]++;
      }
      
      if (age < 4 * 7 * dayMs) {
        const weekIndex = Math.floor(age / (7 * dayMs));
        weekly[3 - weekIndex]++;
      }
    });

    return { hourly, daily, weekly };
  }

  // ===== EVENT SUBSCRIPTIONS =====

  public onAlertCreated(listener: EventListener<IntelligentAlert>): Subscription {
    return this.subscribe('alertCreated', listener);
  }

  public onAlertResolved(listener: EventListener<IntelligentAlert>): Subscription {
    return this.subscribe('alertResolved', listener);
  }

  public onAlertAcknowledged(listener: EventListener<IntelligentAlert>): Subscription {
    return this.subscribe('alertAcknowledged', listener);
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.alerts.clear();
    this.alertRules.clear();
    this.alertHistory = [];
    this.recentAlerts.clear();
    this.suppressionRules.clear();
    super.destroy();
  }
}

// Singleton instance
let intelligentAlertingService: IntelligentAlertingService | null = null;

export function getIntelligentAlertingService(config?: Partial<ServiceConfig>): IntelligentAlertingService {
  if (!intelligentAlertingService) {
    intelligentAlertingService = new IntelligentAlertingService(config);
  }
  return intelligentAlertingService;
}

export function resetIntelligentAlertingService(): void {
  if (intelligentAlertingService) {
    intelligentAlertingService.destroy();
    intelligentAlertingService = null;
  }
}