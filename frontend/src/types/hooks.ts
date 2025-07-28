/**
 * TypeScript interfaces and types for Hook Lifecycle System dashboard integration
 */

export enum HookType {
  PRE_TOOL_USE = "PreToolUse",
  POST_TOOL_USE = "PostToolUse",
  STOP = "Stop",
  NOTIFICATION = "Notification",
  SUBAGENT_STOP = "SubagentStop",
  AGENT_START = "AgentStart",
  AGENT_STOP = "AgentStop",
  ERROR = "Error",
}

export enum SecurityRisk {
  SAFE = "SAFE",
  LOW = "LOW",
  MEDIUM = "MEDIUM",
  HIGH = "HIGH",
  CRITICAL = "CRITICAL",
}

export enum ControlDecision {
  ALLOW = "ALLOW",
  DENY = "DENY",
  REQUIRE_APPROVAL = "REQUIRE_APPROVAL",
  ESCALATE = "ESCALATE",
}

export interface HookEvent {
  hook_type: HookType
  agent_id: string
  session_id?: string
  timestamp: string
  payload: Record<string, any>
  correlation_id?: string
  priority: number // 1=highest, 10=lowest
  metadata: Record<string, any>
  event_id?: string
}

export interface HookProcessingResult {
  success: boolean
  processing_time_ms: number
  error?: string
  security_decision?: ControlDecision
  blocked_reason?: string
  event_id?: string
}

export interface SecurityAlert {
  id: string
  event_id: string
  agent_id: string
  session_id?: string
  timestamp: string
  risk_level: SecurityRisk
  command: string
  reason: string
  blocked: boolean
  requires_approval: boolean
  approved?: boolean
  approved_by?: string
  approved_at?: string
}

export interface DangerousCommand {
  pattern: string
  risk_level: SecurityRisk
  description: string
  block_execution: boolean
  require_approval: boolean
}

export interface EventFilter {
  agent_ids?: string[]
  session_ids?: string[]
  hook_types?: HookType[]
  from_time?: string
  to_time?: string
  min_priority?: number
  search_query?: string
  risk_levels?: SecurityRisk[]
  only_blocked?: boolean
  only_errors?: boolean
}

export interface SessionInfo {
  session_id: string
  agent_ids: string[]
  start_time: string
  end_time?: string
  event_count: number
  error_count: number
  blocked_count: number
  status: 'active' | 'completed' | 'error' | 'terminated'
  color?: string // For visualization
}

export interface AgentInfo {
  agent_id: string
  session_ids: string[]
  first_seen: string
  last_seen: string
  event_count: number
  tool_usage_count: number
  error_count: number
  blocked_count: number
  status: 'active' | 'idle' | 'error' | 'blocked'
  color?: string // For visualization
}

export interface HookPerformanceMetrics {
  total_hooks_processed: number
  hooks_blocked: number
  processing_errors: number
  avg_processing_time_ms: number
  performance_threshold_violations: number
  security_validator_metrics: {
    validations_performed: number
    commands_blocked: number
    approvals_required: number
    cache_hits: number
    avg_validation_time_ms: number
  }
  event_aggregator_metrics: {
    events_aggregated: number
    batches_processed: number
    aggregation_rules_applied: number
    avg_batch_size: number
    flush_operations: number
  }
  websocket_streamer_metrics: {
    active_connections: number
    messages_sent: number
    connection_errors: number
    filtered_messages: number
    avg_broadcast_time_ms: number
  }
}

export interface EventTimelineData {
  timestamp: string
  events: HookEvent[]
  agent_count: number
  session_count: number
  security_alerts: number
  performance_score: number
}

export interface SecurityDashboardData {
  active_alerts: SecurityAlert[]
  recent_blocks: SecurityAlert[]
  dangerous_patterns: DangerousCommand[]
  risk_distribution: Record<SecurityRisk, number>
  blocked_commands_timeline: Array<{
    timestamp: string
    count: number
    risk_level: SecurityRisk
  }>
  approval_queue: SecurityAlert[]
}

export interface WebSocketHookMessage {
  type: 'hook_event' | 'security_alert' | 'performance_metric' | 'system_status'
  data: HookEvent | SecurityAlert | HookPerformanceMetrics | any
  timestamp: string
}

// Event payload type definitions for different hook types
export interface PreToolUsePayload {
  tool_name: string
  parameters: Record<string, any>
  timestamp: string
}

export interface PostToolUsePayload {
  tool_name: string
  success: boolean
  result?: any
  error?: string
  execution_time_ms?: number
  timestamp: string
  result_truncated?: boolean
  full_result_size?: number
}

export interface StopPayload {
  reason: string
  details?: Record<string, any>
  timestamp: string
}

export interface NotificationPayload {
  level: 'info' | 'warning' | 'error' | 'critical'
  message: string
  details?: Record<string, any>
  timestamp: string
}

export interface ErrorPayload {
  error_type: string
  error_message: string
  stack_trace?: string
  context?: Record<string, any>
  timestamp: string
}

// Dashboard component props interfaces
export interface HookEventTimelineProps {
  height?: number
  maxEvents?: number
  autoScroll?: boolean
  showFilters?: boolean
  initialFilters?: EventFilter
}

export interface SecurityDashboardProps {
  showApprovalQueue?: boolean
  alertLimit?: number
  autoRefresh?: boolean
  refreshInterval?: number
}

export interface EventFilterPanelProps {
  filters: EventFilter
  availableAgents?: AgentInfo[]
  availableSessions?: SessionInfo[]
  onFiltersChange: (filters: EventFilter) => void
  onClearFilters: () => void
}

export interface SessionVisualizationProps {
  sessions: SessionInfo[]
  maxSessions?: number
  showInactive?: boolean
  colorScheme?: 'default' | 'status' | 'activity'
}

export interface EventDetailsModalProps {
  event?: HookEvent
  isOpen: boolean
  onClose: () => void
  showSecurityInfo?: boolean
  showPerformanceInfo?: boolean
}

export interface PerformanceMonitoringDashboardProps {
  refreshInterval?: number
  showHistoricalData?: boolean
  timeRange?: '1h' | '6h' | '24h' | '7d'
}

// Utility type for color generation
export interface ColorPalette {
  primary: string[]
  status: {
    active: string
    idle: string
    error: string
    blocked: string
    completed: string
    terminated: string
  }
  risk: {
    [SecurityRisk.SAFE]: string
    [SecurityRisk.LOW]: string
    [SecurityRisk.MEDIUM]: string
    [SecurityRisk.HIGH]: string
    [SecurityRisk.CRITICAL]: string
  }
  hookTypes: {
    [HookType.PRE_TOOL_USE]: string
    [HookType.POST_TOOL_USE]: string
    [HookType.STOP]: string
    [HookType.NOTIFICATION]: string
    [HookType.AGENT_START]: string
    [HookType.AGENT_STOP]: string
    [HookType.ERROR]: string
    [HookType.SUBAGENT_STOP]: string
  }
}

// Chart data interfaces for visualization components
export interface TimelineChartData {
  x: string // timestamp
  y: number // event count
  category: string // event type or status
  color?: string
  metadata?: Record<string, any>
}

export interface PerformanceChartData {
  timestamp: string
  processing_time_ms: number
  throughput: number
  error_rate: number
  security_blocks: number
}

export interface DistributionChartData {
  label: string
  value: number
  percentage: number
  color: string
}

// API response interfaces
export interface HookEventsResponse {
  events: HookEvent[]
  pagination: {
    limit: number
    offset: number
    total: number
    hasNext: boolean
    hasPrev: boolean
  }
  filters_applied: EventFilter
}

export interface SecurityAlertsResponse {
  alerts: SecurityAlert[]
  summary: {
    total_alerts: number
    active_alerts: number
    blocked_commands: number
    pending_approvals: number
  }
}

export interface HookMetricsResponse {
  metrics: HookPerformanceMetrics
  timestamp: string
  system_status: 'healthy' | 'degraded' | 'critical'
}