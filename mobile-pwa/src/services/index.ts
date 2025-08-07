/**
 * LeanVibe Agent Hive API Integration Services
 * 
 * Centralized exports for all API integration services providing:
 * - System health monitoring
 * - Agent lifecycle management  
 * - Task management for Kanban board
 * - Real-time event streaming
 * - Performance metrics and analytics
 * 
 * Usage:
 * ```typescript
 * import { getSystemHealthService, getAgentService } from '@/services';
 * 
 * const healthService = getSystemHealthService();
 * await healthService.getSystemHealth();
 * ```
 */

// Base service and types
export { BaseService } from './base-service';
export type { CacheEntry } from './base-service';

// System Health Service
export { 
  SystemHealthService, 
  getSystemHealthService, 
  resetSystemHealthService 
} from './system-health';
export type { 
  HealthSummary, 
  HealthAlert 
} from './system-health';

// Agent Service
export { 
  AgentService, 
  getAgentService, 
  resetAgentService 
} from './agent';
export type { 
  AgentSummary, 
  TeamComposition, 
  AgentActivationOptions 
} from './agent';

// Task Service
export { 
  TaskService, 
  getTaskService, 
  resetTaskService 
} from './task';
export type { 
  KanbanBoard, 
  KanbanColumn, 
  TaskFilters, 
  TaskStatistics, 
  TaskAssignmentResult 
} from './task';

// Event Service
export { 
  EventService, 
  getEventService, 
  resetEventService 
} from './event';
export type { 
  EventTimeline, 
  EventStatistics, 
  ActivitySummary 
} from './event';

// Metrics Service
export { 
  MetricsService, 
  getMetricsService, 
  resetMetricsService 
} from './metrics';
export type { 
  PerformanceSnapshot, 
  PerformanceTrend, 
  PerformanceAlert, 
  ChartData 
} from './metrics';

// Task Coordination Service
export { 
  TaskCoordinationService, 
  getTaskCoordinationService, 
  resetTaskCoordinationService 
} from './task-coordination';
export type { 
  TaskAssignment, 
  AgentWorkload, 
  TaskCoordinationMetrics, 
  CollaborationRequest, 
  AutoAssignmentRule 
} from './task-coordination';

// Notification Service
export { 
  NotificationService, 
  getNotificationService 
} from './notification';
export type { 
  NotificationData, 
  NotificationAction, 
  PushSubscriptionData 
} from './notification';

// API Types (re-export for convenience)
export type {
  // Base API types
  ApiResponse,
  ApiError,
  PaginatedResponse,
  ServiceConfig,
  Subscription,
  EventListener,
  
  // System Health types
  SystemHealth,
  ComponentHealth,
  SystemMetrics,
  
  // Agent types
  Agent,
  AgentRole,
  AgentStatus,
  AgentSystemStatus,
  AgentActivationRequest,
  AgentActivationResponse,
  AgentPerformanceMetrics,
  
  // Task types
  Task,
  TaskCreate,
  TaskUpdate,
  TaskStatus,
  TaskPriority,
  TaskType,
  TaskListResponse,
  
  // Event types
  SystemEvent,
  EventType,
  EventSeverity,
  EventFilters,
  
  // Metrics types
  SystemPerformanceMetrics,
  AgentMetrics,
  MetricDataPoint,
  MetricSeries,
  
  // WebSocket types
  WebSocketMessage,
  WebSocketEvent,
  WebSocketHealthUpdate,
  WebSocketAgentUpdate,
  WebSocketTaskUpdate
} from '../types/api';

// Service Factory Functions
/**
 * Initialize all services with shared configuration
 */
export function initializeServices(config?: Partial<import('../types/api').ServiceConfig>) {
  const systemHealth = getSystemHealthService(config);
  const agent = getAgentService(config);
  const task = getTaskService(config);
  const event = getEventService(config);
  const metrics = getMetricsService(config);

  return {
    systemHealth,
    agent,
    task,
    event,
    metrics
  };
}

/**
 * Start monitoring for all services
 */
export function startAllMonitoring() {
  const services = initializeServices();
  
  services.systemHealth.startMonitoring();
  services.agent.startMonitoring();
  services.task.startMonitoring();
  services.event.startRealtimeMonitoring();
  services.metrics.startMonitoring();
  
  return services;
}

/**
 * Stop monitoring for all services
 */
export function stopAllMonitoring() {
  const systemHealth = getSystemHealthService();
  const agent = getAgentService();
  const task = getTaskService();
  const event = getEventService();
  const metrics = getMetricsService();
  
  systemHealth.stopMonitoring();
  agent.stopMonitoring();
  task.stopMonitoring();
  event.stopRealtimeMonitoring();
  metrics.stopMonitoring();
}

/**
 * Reset all services (cleanup)
 */
export function resetAllServices() {
  resetSystemHealthService();
  resetAgentService();
  resetTaskService();
  resetEventService();
  resetMetricsService();
}

/**
 * Get service health status
 */
export function getServicesStatus() {
  const systemHealth = getSystemHealthService();
  const agent = getAgentService();
  const task = getTaskService();
  const event = getEventService();
  const metrics = getMetricsService();

  return {
    systemHealth: {
      monitoring: systemHealth.isMonitoring(),
      connected: true // Would check actual connection status
    },
    agent: {
      monitoring: agent.isMonitoring(),
      connected: true
    },
    task: {
      monitoring: task.isMonitoring(),
      connected: true
    },
    event: {
      monitoring: event.isRealtimeMonitoring(),
      connected: true
    },
    metrics: {
      monitoring: metrics.isMonitoring(),
      connected: true
    }
  };
}