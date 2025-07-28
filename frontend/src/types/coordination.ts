/**
 * Type definitions for Multi-Agent Coordination Dashboard
 */

// Base interfaces for agent and session data
export interface AgentInfo {
  agent_id: string
  status: 'active' | 'idle' | 'sleeping' | 'error' | 'blocked'
  session_ids: string[]
  event_count: number
  blocked_count: number
  first_seen: string
  last_seen: string
  capabilities?: string[]
  metadata?: Record<string, any>
}

export interface SessionInfo {
  id: string
  label: string
  agentCount: number
  lastActivity?: string
  createdAt?: string
  status?: 'active' | 'completed' | 'error'
}

// Graph-related types
export interface GraphNode {
  id: string
  type: 'agent' | 'tool' | 'context' | 'session'
  label: string
  status: 'active' | 'idle' | 'sleeping' | 'error' | 'blocked'
  position?: { x: number; y: number }
  metadata: {
    session_id: string
    agent_type?: string
    current_task?: string
    uptime?: number
    performance?: number
    memory_usage?: number
    connections?: string[]
  }
  last_updated: string
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  type: 'communication' | 'collaboration' | 'dependency' | 'data_flow'
  weight: number
  timestamp: string
  metadata: {
    event_type: string
    success: boolean
    duration_ms?: number
    message_count?: number
  }
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  stats: {
    total_nodes: number
    total_edges: number
    active_sessions: number
    message_throughput: number
  }
  sessionColors: Record<string, string>
}

// Conversation and transcript types
export interface ConversationEvent {
  id: string
  session_id: string
  timestamp: string
  event_type: ConversationEventType
  source_agent: string
  target_agent?: string
  message_type: MessageType
  content: string
  context_shared: boolean
  tool_calls: ToolCall[]
  context_references: string[]
  metadata: Record<string, any>
  response_time_ms?: number
  embedding_vector?: number[]
}

export enum ConversationEventType {
  MESSAGE_SENT = 'message_sent',
  MESSAGE_RECEIVED = 'message_received',
  TOOL_INVOCATION = 'tool_invocation',
  CONTEXT_SHARING = 'context_sharing',
  ERROR_OCCURRED = 'error_occurred',
  COORDINATION_REQUEST = 'coordination_request',
  STATUS_UPDATE = 'status_update',
  TASK_DELEGATION = 'task_delegation',
  COLLABORATION_START = 'collaboration_start',
  COLLABORATION_END = 'collaboration_end'
}

export enum MessageType {
  TASK_ASSIGNMENT = 'task_assignment',
  STATUS_UPDATE = 'status_update',
  COMPLETION = 'completion',
  ERROR = 'error',
  COLLABORATION = 'collaboration',
  COORDINATION = 'coordination',
  TOOL_RESULT = 'tool_result',
  CONTEXT_SHARE = 'context_share'
}

export interface ToolCall {
  name: string
  description?: string
  parameters?: Record<string, any>
  result?: any
  duration_ms?: number
  success: boolean
}

export interface TranscriptData {
  events: ConversationEvent[]
  totalEvents: number
  metadata: {
    event_types: Record<string, number>
    agents_involved: string[]
    time_span_minutes: number
    tool_usage_count: number
    context_sharing_count: number
  }
  agentSummary: Record<string, AgentActivity>
}

export interface AgentActivity {
  total_messages: number
  tool_calls: number
  contexts_shared: number
  first_activity: string
  last_activity: string
  error_count?: number
  avg_response_time?: number
}

// Pattern detection and analysis types
export interface DetectedPattern {
  id: string
  name: string
  description: string
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  occurrences: number
  affectedAgents: string[]
  confidence: number
  pattern_type?: PatternType
  recommendation?: string
  first_detected?: string
  last_detected?: string
}

export enum PatternType {
  INFINITE_LOOP = 'infinite_loop',
  ERROR_CASCADE = 'error_cascade',
  BOTTLENECK = 'bottleneck',
  DEADLOCK = 'deadlock',
  CONTEXT_EXPLOSION = 'context_explosion',
  PERFORMANCE_DEGRADATION = 'performance_degradation',
  COORDINATION_FAILURE = 'coordination_failure'
}

export interface AnalysisData {
  patterns: DetectedPattern[]
  metrics: PerformanceMetrics
  recommendations: OptimizationRecommendation[]
  insights?: AnalysisInsight[]
}

export interface AnalysisInsight {
  id: string
  type: 'performance' | 'behavior' | 'error' | 'optimization'
  title: string
  description: string
  impact: 'low' | 'medium' | 'high'
  confidence: number
  data?: any
}

// Performance and monitoring types
export interface PerformanceMetrics {
  responseTime: number
  errorRate: number
  throughput: number
  agentMetrics: Record<string, AgentPerformanceMetrics>
  p95ResponseTime?: number
  maxResponseTime?: number
  totalErrors?: number
  messageFrequency?: number
}

export interface AgentPerformanceMetrics {
  id: string
  name: string
  avgResponseTime: number
  errorRate: number
  messageCount: number
  uptime: number
  performance: number
  health: 'healthy' | 'warning' | 'critical'
}

export interface OptimizationRecommendation {
  id: string
  type: 'performance' | 'reliability' | 'scalability' | 'security'
  title: string
  description: string
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  estimatedImpact: string
  implementation?: string
  effort?: 'low' | 'medium' | 'high'
}

// Dashboard state management
export interface DashboardState {
  selectedSession: string
  activeFilters: DashboardFilters
  graphData: GraphData
  transcriptData: TranscriptData
  analysisData: AnalysisData
  performance: PerformanceMetrics
  connections: ConnectionStatus
}

export interface DashboardFilters {
  sessionIds: string[]
  agentIds: string[]
  eventTypes: string[]
  timeRange: {
    start?: string
    end?: string
  }
  includeInactive: boolean
  severityFilter?: ('LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL')[]
  patternTypes?: PatternType[]
}

export interface ConnectionStatus {
  websocket: boolean
  api: boolean
  lastUpdate: Date | null
  reconnectAttempts?: number
  latency?: number
}

// Event system types
export interface DashboardEvent {
  type: DashboardEventType
  source: DashboardComponent
  target: DashboardComponent | 'all'
  data: any
  timestamp: string
  correlation_id?: string
}

export enum DashboardEventType {
  SESSION_CHANGED = 'session_changed',
  FILTERS_CHANGED = 'filters_changed',
  DATA_UPDATED = 'data_updated',
  REALTIME_UPDATE = 'realtime_update',
  NODE_SELECTED = 'node_selected',
  EVENT_SELECTED = 'event_selected',
  PATTERN_DETECTED = 'pattern_detected',
  ERROR_OCCURRED = 'error_occurred',
  PERFORMANCE_ALERT = 'performance_alert'
}

export enum DashboardComponent {
  GRAPH = 'graph',
  TRANSCRIPT = 'transcript',
  ANALYSIS = 'analysis',
  MONITORING = 'monitoring',
  SERVICE = 'service'
}

// WebSocket message types
export interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
  session_id?: string
  correlation_id?: string
}

export interface GraphUpdateMessage extends WebSocketMessage {
  type: 'graph_update'
  data: {
    nodes?: GraphNode[]
    edges?: GraphEdge[]
    action: 'add' | 'update' | 'remove'
    affected_ids: string[]
  }
}

export interface TranscriptUpdateMessage extends WebSocketMessage {
  type: 'transcript_update'
  data: {
    events?: ConversationEvent[]
    action: 'add' | 'update'
    session_id: string
  }
}

export interface AnalysisUpdateMessage extends WebSocketMessage {
  type: 'analysis_update'
  data: {
    patterns?: DetectedPattern[]
    insights?: AnalysisInsight[]
    action: 'add' | 'update' | 'resolve'
  }
}

// Search and filtering types
export interface SearchQuery {
  query: string
  filters: DashboardFilters
  searchType: 'semantic' | 'keyword' | 'pattern' | 'hybrid'
  limit?: number
  offset?: number
}

export interface SearchResult {
  id: string
  type: 'agent' | 'event' | 'pattern' | 'session'
  relevance: number
  data: any
  highlights?: string[]
  metadata?: Record<string, any>
}

export interface SearchResponse {
  results: SearchResult[]
  totalMatches: number
  searchTime: number
  suggestions: string[]
  facets: Record<string, Record<string, number>>
}

// Dashboard configuration types
export interface DashboardConfig {
  refreshInterval: number
  autoRefresh: boolean
  maxCacheSize: number
  cacheTTL: number
  webSocketReconnectDelay: number
  defaultFilters: DashboardFilters
  visualizationSettings: VisualizationSettings
}

export interface VisualizationSettings {
  graphLayout: 'force' | 'circle' | 'grid' | 'hierarchical'
  colorScheme: 'session' | 'performance' | 'status' | 'type'
  showInactiveNodes: boolean
  showSystemEvents: boolean
  animationSpeed: number
  nodeSize: 'fixed' | 'scaled' | 'performance'
}

// Error handling types
export interface DashboardError {
  id: string
  type: 'network' | 'data' | 'websocket' | 'parsing' | 'unknown'
  message: string
  component: DashboardComponent
  timestamp: string
  details?: any
  recoverable: boolean
}

export interface ErrorBoundaryState {
  hasError: boolean
  error?: DashboardError
  fallbackComponent?: string
  retryCount: number
  lastRetry?: string
}

// Export utility types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}

export type RequireAtLeastOne<T, Keys extends keyof T = keyof T> = Pick<T, Exclude<keyof T, Keys>> & {
  [K in Keys]-?: Required<Pick<T, K>> & Partial<Pick<T, Exclude<Keys, K>>>
}[Keys]

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

// Dashboard theme and styling types
export interface ThemeColors {
  primary: string
  secondary: string
  success: string
  warning: string
  error: string
  info: string
  background: string
  surface: string
  text: string
}

export interface ComponentStyle {
  backgroundColor?: string
  borderColor?: string
  textColor?: string
  opacity?: number
  scale?: number
}

export interface DashboardTheme {
  name: string
  colors: ThemeColors
  isDark: boolean
  componentStyles: Record<string, ComponentStyle>
}