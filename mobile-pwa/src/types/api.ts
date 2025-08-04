/**
 * TypeScript interfaces for LeanVibe Agent Hive API integration
 * 
 * These interfaces define the data structures used across all API services
 * and ensure type safety throughout the application.
 */

// ===== BASE TYPES =====

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
  hasMore: boolean;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

// ===== SYSTEM HEALTH TYPES =====

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  components: {
    database: ComponentHealth;
    redis: ComponentHealth;
    orchestrator: ComponentHealth;
    agents: ComponentHealth;
    [key: string]: ComponentHealth;
  };
  metrics: SystemMetrics;
}

export interface ComponentHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency?: number;
  errors?: number;
  details?: Record<string, any>;
  lastCheck: string;
}

export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: {
    in: number;
    out: number;
  };
  active_connections: number;
  uptime: number;
}

// ===== AGENT TYPES =====

export const AgentRole = {
  PRODUCT_MANAGER: 'product_manager',
  ARCHITECT: 'architect', 
  BACKEND_DEVELOPER: 'backend_developer',
  FRONTEND_DEVELOPER: 'frontend_developer',
  QA_ENGINEER: 'qa_engineer',
  DEVOPS_ENGINEER: 'devops_engineer'
} as const;

export type AgentRole = typeof AgentRole[keyof typeof AgentRole];

export const AgentStatus = {
  ACTIVE: 'active',
  IDLE: 'idle',
  BUSY: 'busy',
  ERROR: 'error',
  OFFLINE: 'offline'
} as const;

export type AgentStatus = typeof AgentStatus[keyof typeof AgentStatus];

export interface Agent {
  id: string;
  role: AgentRole;
  status: AgentStatus;
  name: string;
  capabilities: string[];
  created_at: string;
  updated_at: string;
  last_activity: string;
  current_task_id?: string;
  performance_metrics: AgentPerformanceMetrics;
  error_message?: string;
}

export interface AgentPerformanceMetrics {
  tasks_completed: number;
  tasks_failed: number;
  average_completion_time: number;
  cpu_usage: number;
  memory_usage: number;
  success_rate: number;
  uptime: number;
}

export interface AgentActivationRequest {
  team_size?: number;
  roles?: AgentRole[];
  auto_start_tasks?: boolean;
}

export interface AgentActivationResponse {
  success: boolean;
  message: string;
  active_agents: Record<string, Agent>;
  team_composition: Record<string, string>;
}

export interface AgentSystemStatus {
  active: boolean;
  agent_count: number;
  spawner_agents: number;
  orchestrator_agents: number;
  agents: Record<string, Agent>;
  orchestrator_agents_detail: Record<string, Agent>;
  system_ready: boolean;
  hybrid_integration: boolean;
  error?: string;
}

// ===== TASK TYPES =====

export const TaskStatus = {
  PENDING: 'pending',
  ASSIGNED: 'assigned',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled'
} as const;

export type TaskStatus = typeof TaskStatus[keyof typeof TaskStatus];

export const TaskPriority = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical'
} as const;

export type TaskPriority = typeof TaskPriority[keyof typeof TaskPriority];

export const TaskType = {
  FEATURE: 'feature',
  BUG_FIX: 'bug_fix',
  REFACTOR: 'refactor',
  TEST: 'test',
  DOCUMENTATION: 'documentation',
  DEPLOYMENT: 'deployment'
} as const;

export type TaskType = typeof TaskType[keyof typeof TaskType];

export interface Task {
  id: string;
  title: string;
  description: string;
  task_type: TaskType;
  priority: TaskPriority;
  status: TaskStatus;
  required_capabilities: string[];
  estimated_effort?: number;
  actual_effort?: number;
  assigned_agent_id?: string;
  context: Record<string, any>;
  result?: Record<string, any>;
  error_message?: string;
  retry_count: number;
  max_retries: number;
  created_at: string;
  updated_at: string;
  assigned_at?: string;
  started_at?: string;
  completed_at?: string;
}

export interface TaskCreate {
  title: string;
  description: string;
  task_type: TaskType;
  priority: TaskPriority;
  required_capabilities?: string[];
  estimated_effort?: number;
  context?: Record<string, any>;
}

export interface TaskUpdate {
  title?: string;
  description?: string;
  priority?: TaskPriority;
  status?: TaskStatus;
  context?: Record<string, any>;
}

export interface TaskListResponse {
  tasks: Task[];
  total: number;
  offset: number;
  limit: number;
}

// ===== EVENT TYPES =====

export const EventType = {
  AGENT_ACTIVATED: 'agent_activated',
  AGENT_DEACTIVATED: 'agent_deactivated',
  TASK_CREATED: 'task_created',
  TASK_ASSIGNED: 'task_assigned',
  TASK_STARTED: 'task_started',
  TASK_COMPLETED: 'task_completed',
  TASK_FAILED: 'task_failed',
  SYSTEM_ERROR: 'system_error',
  PERFORMANCE_ALERT: 'performance_alert',
  HEALTH_CHECK: 'health_check'
} as const;

export type EventType = typeof EventType[keyof typeof EventType];

export const EventSeverity = {
  INFO: 'info',
  WARNING: 'warning',
  ERROR: 'error',
  CRITICAL: 'critical'
} as const;

export type EventSeverity = typeof EventSeverity[keyof typeof EventSeverity];

export interface SystemEvent {
  id: string;
  type: EventType;
  severity: EventSeverity;
  title: string;
  description: string;
  source: string;
  agent_id?: string;
  task_id?: string;
  data: Record<string, any>;
  timestamp: string;
  acknowledged: boolean;
}

export interface EventFilters {
  type?: EventType[];
  severity?: EventSeverity[];
  agent_id?: string;
  start_date?: string;
  end_date?: string;
  acknowledged?: boolean;
}

// ===== METRICS TYPES =====

export interface MetricDataPoint {
  timestamp: string;
  value: number;
  labels?: Record<string, string>;
}

export interface MetricSeries {
  name: string;
  data: MetricDataPoint[];
  unit: string;
  description: string;
}

export interface SystemPerformanceMetrics {
  cpu: MetricSeries;
  memory: MetricSeries;
  disk: MetricSeries;
  network: {
    in: MetricSeries;
    out: MetricSeries;
  };
  agents: {
    active_count: MetricSeries;
    task_completion_rate: MetricSeries;
    error_rate: MetricSeries;
  };
}

export interface AgentMetrics {
  agent_id: string;
  role: AgentRole;
  performance: AgentPerformanceMetrics;
  resource_usage: {
    cpu: MetricSeries;
    memory: MetricSeries;
  };
  task_metrics: {
    completed: MetricSeries;
    failed: MetricSeries;
    completion_time: MetricSeries;
  };
}

// ===== WEBSOCKET TYPES =====

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
  correlation_id?: string;
}

export interface WebSocketEvent extends WebSocketMessage {
  type: 'event';
  data: SystemEvent;
}

export interface WebSocketHealthUpdate extends WebSocketMessage {
  type: 'health_update';
  data: SystemHealth;
}

export interface WebSocketAgentUpdate extends WebSocketMessage {
  type: 'agent_update';
  data: {
    agent_id: string;
    status: AgentStatus;
    current_task_id?: string;
    performance_metrics: AgentPerformanceMetrics;
  };
}

export interface WebSocketTaskUpdate extends WebSocketMessage {
  type: 'task_update';
  data: {
    task_id: string;
    status: TaskStatus;
    assigned_agent_id?: string;
    progress?: number;
  };
}

// ===== SERVICE CONFIGURATION TYPES =====

export interface ServiceConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  cacheTimeout: number;
  pollingInterval: number;
}

export interface PollingConfig {
  enabled: boolean;
  interval: number;
  maxRetries: number;
  backoffMultiplier: number;
}

export interface CacheConfig {
  enabled: boolean;
  ttl: number;
  maxSize: number;
}

export interface LoadingState {
  isLoading: boolean;
  error?: ApiError;
  lastUpdated?: string;
}

// ===== UTILITY TYPES =====

export type EventListener<T = any> = (data: T) => void;

export interface Subscription {
  unsubscribe: () => void;
}

export interface RetryOptions {
  maxAttempts: number;
  delay: number;
  backoffMultiplier: number;
  shouldRetry?: (error: Error) => boolean;
}