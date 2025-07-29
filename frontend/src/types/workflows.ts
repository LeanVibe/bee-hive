/**
 * TypeScript type definitions for multi-agent workflow system
 */

export interface WorkflowCommand {
  id: string
  name: string
  version: string
  description: string
  category: string
  tags: string[]
  workflow: WorkflowStep[]
  parameters: Record<string, WorkflowParameter>
  requirements: WorkflowRequirements
  metadata: Record<string, any>
  isEnabled: boolean
  createdAt: Date
  updatedAt: Date
}

export interface WorkflowStep {
  id: string
  name: string
  description: string
  type: 'task' | 'condition' | 'parallel' | 'sequence' | 'loop'
  agent_role?: string
  required_capabilities: string[]
  estimated_duration: number
  dependencies: string[]
  conditions?: WorkflowCondition[]
  parameters: Record<string, any>
  retry_policy?: RetryPolicy
  timeout_seconds?: number
  on_success?: string[]
  on_failure?: string[]
}

export interface WorkflowParameter {
  name: string
  type: 'string' | 'number' | 'boolean' | 'array' | 'object'
  description: string
  required: boolean
  default_value?: any
  validation?: ParameterValidation
}

export interface ParameterValidation {
  min?: number
  max?: number
  pattern?: string
  allowed_values?: any[]
  custom_validator?: string
}

export interface WorkflowRequirements {
  min_agents: number
  max_agents: number
  required_capabilities: string[]
  preferred_capabilities: string[]
  resource_limits?: ResourceLimits
  quality_gates?: QualityGate[]
}

export interface ResourceLimits {
  max_memory_mb: number
  max_cpu_cores: number
  max_execution_time_seconds: number
  max_parallel_tasks: number
}

export interface QualityGate {
  id: string
  name: string
  type: 'pre_execution' | 'step_completion' | 'post_execution'
  criteria: QualityCriteria[]
  required: boolean
}

export interface QualityCriteria {
  metric: string
  operator: 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'neq'
  value: number | string
  weight: number
}

export interface WorkflowCondition {
  type: 'expression' | 'agent_available' | 'resource_available' | 'custom'
  expression?: string
  agent_id?: string
  resource_type?: string
  custom_evaluator?: string
}

export interface RetryPolicy {
  max_attempts: number
  delay_seconds: number
  backoff_multiplier: number
  max_delay_seconds: number
  retry_on_errors: string[]
}

export interface Workflow {
  id: string
  commandId: string
  name: string
  description: string
  status: WorkflowStatus
  steps: WorkflowNode[]
  edges: WorkflowEdge[]
  agentAssignments: Map<string, string[]>
  parameters: Record<string, any>
  createdAt: Date
  startedAt?: Date
  completedAt?: Date
  metadata: Record<string, any>
}

export interface WorkflowNode {
  id: string
  stepId: string
  name: string
  type: 'task' | 'condition' | 'parallel' | 'sequence' | 'loop' | 'start' | 'end'
  status: NodeStatus
  assignedAgentId?: string
  position: NodePosition
  data: NodeData
  progress?: number
  startTime?: Date
  endTime?: Date
  duration?: number
  errors: WorkflowError[]
  logs: WorkflowLog[]
}

export interface NodePosition {
  x: number
  y: number
}

export interface NodeData {
  label: string
  description: string
  capabilities: string[]
  parameters: Record<string, any>
  metadata: Record<string, any>
}

export type NodeStatus = 
  | 'pending' 
  | 'ready' 
  | 'running' 
  | 'completed' 
  | 'failed' 
  | 'cancelled' 
  | 'blocked'
  | 'skipped'

export interface WorkflowEdge {
  id: string
  source: string
  target: string
  type: 'sequence' | 'condition' | 'parallel' | 'dependency'
  condition?: string
  label?: string
  animated?: boolean
  style?: EdgeStyle
}

export interface EdgeStyle {
  stroke?: string
  strokeWidth?: number
  strokeDasharray?: string
}

export type WorkflowStatus = 
  | 'draft' 
  | 'ready' 
  | 'running' 
  | 'paused' 
  | 'completed' 
  | 'failed' 
  | 'cancelled'

export interface WorkflowExecution {
  id: string
  commandName: string
  status: ExecutionStatus
  parameters: Record<string, any>
  startTime: Date
  endTime?: Date | null
  duration?: number
  steps: StepExecutionResult[]
  agentAssignments: AgentAssignment[]
  metrics: ExecutionMetrics
  logs: WorkflowLog[]
  errors: WorkflowError[]
  progress?: number
  currentStep?: string
  estimatedCompletion?: Date | null
}

export type ExecutionStatus = 
  | 'queued'
  | 'starting' 
  | 'running' 
  | 'paused'
  | 'completed' 
  | 'failed' 
  | 'cancelled'
  | 'timeout'

export interface StepExecutionResult {
  stepId: string
  agentId: string
  status: ExecutionStatus
  startTime: Date
  endTime?: Date
  duration?: number
  result?: any
  errors: WorkflowError[]
  logs: WorkflowLog[]
  retryCount: number
  performance: StepPerformance
}

export interface StepPerformance {
  executionTime: number
  memoryUsage: number
  cpuUsage: number
  throughput: number
  errorRate: number
}

export interface AgentAssignment {
  id: string
  agentId: string
  executionId: string
  taskName: string
  status: AssignmentStatus
  assignedAt: Date
  startedAt?: Date | null
  completedAt?: Date | null
  estimatedDuration: number
  actualDuration?: number
  capabilities: string[]
  metadata: Record<string, any>
}

export type AssignmentStatus = 
  | 'assigned'
  | 'accepted' 
  | 'running' 
  | 'completed' 
  | 'failed' 
  | 'cancelled'
  | 'reassigned'

export interface ExecutionMetrics {
  totalExecutions: number
  successfulExecutions: number
  failedExecutions: number
  averageExecutionTime: number
  activeExecutions: number
  agentUtilization: number
  systemThroughput: number
  lastUpdated: Date
}

export interface WorkflowError {
  id?: string
  message: string
  type: ErrorType
  stepId?: string
  agentId?: string
  timestamp: string
  severity: ErrorSeverity
  details?: Record<string, any>
  stackTrace?: string
}

export type ErrorType = 
  | 'validation_error'
  | 'execution_error' 
  | 'timeout_error'
  | 'resource_error'
  | 'agent_error'
  | 'system_error'
  | 'network_error'

export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical' | 'info'

export interface WorkflowLog {
  id?: string
  timestamp: string
  level: LogLevel
  message: string
  stepId?: string
  agentId?: string
  executionId?: string
  metadata?: Record<string, any>
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'trace'

// Visualization types
export interface WorkflowGraphData {
  nodes: WorkflowGraphNode[]
  edges: WorkflowGraphEdge[]
  layout: GraphLayout
  metadata: GraphMetadata
}

export interface WorkflowGraphNode {
  id: string
  label: string
  type: NodeType
  status: NodeStatus
  position: NodePosition
  size: NodeSize
  style: NodeStyle
  data: NodeData
  agentId?: string
  progress?: number
  metrics?: NodeMetrics
}

export interface NodeSize {
  width: number
  height: number
}

export interface NodeStyle {
  fill?: string
  stroke?: string
  strokeWidth?: number
  opacity?: number
  shape?: 'rectangle' | 'circle' | 'diamond' | 'ellipse'
}

export interface NodeMetrics {
  executionTime?: number
  memoryUsage?: number
  successRate?: number
  lastExecution?: Date
}

export type NodeType = 
  | 'start'
  | 'end' 
  | 'task'
  | 'condition' 
  | 'parallel'
  | 'sequence'
  | 'loop'
  | 'agent'
  | 'gateway'

export interface WorkflowGraphEdge {
  id: string
  source: string
  target: string
  type: EdgeType
  label?: string
  style: EdgeStyle
  animated?: boolean
  condition?: string
  weight?: number
}

export type EdgeType = 
  | 'sequence'
  | 'conditional' 
  | 'parallel'
  | 'dependency'
  | 'data_flow'
  | 'control_flow'

export interface GraphLayout {
  type: LayoutType
  options: LayoutOptions
}

export type LayoutType = 
  | 'force'
  | 'hierarchical' 
  | 'circular'
  | 'grid'
  | 'dagre'
  | 'custom'

export interface LayoutOptions {
  direction?: 'TB' | 'BT' | 'LR' | 'RL'
  spacing?: number
  nodeSpacing?: number
  edgeSpacing?: number
  iterations?: number
  springLength?: number
  springCoeff?: number
  dragCoeff?: number
  gravity?: number
  theta?: number
  alpha?: number
}

export interface GraphMetadata {
  title: string
  description: string
  version: string
  author: string
  createdAt: Date
  updatedAt: Date
  tags: string[]
  complexity: GraphComplexity
}

export interface GraphComplexity {
  nodeCount: number
  edgeCount: number
  depth: number
  parallelism: number
  cyclomaticComplexity: number
}

// Control panel types
export interface WorkflowControlOptions {
  canStart: boolean
  canPause: boolean
  canStop: boolean
  canCancel: boolean
  canRestart: boolean
  canDebug: boolean
  emergencyStop: boolean
}

export interface WorkflowControlAction {
  type: ControlActionType
  executionId?: string
  parameters?: Record<string, any>
  reason?: string
  confirmation?: boolean
}

export type ControlActionType = 
  | 'start'
  | 'pause' 
  | 'resume'
  | 'stop'
  | 'cancel'
  | 'restart'
  | 'emergency_stop'
  | 'debug'
  | 'step_into'
  | 'step_over'

// Real-time update types
export interface WorkflowUpdateMessage {
  type: WorkflowUpdateType
  executionId: string
  data: any
  timestamp: string
}

export type WorkflowUpdateType = 
  | 'execution_started'
  | 'execution_progress'
  | 'execution_completed'
  | 'execution_failed'
  | 'execution_cancelled'
  | 'step_started'
  | 'step_completed'
  | 'step_failed'
  | 'agent_assigned'
  | 'agent_unassigned'
  | 'error_occurred'
  | 'metrics_updated'

// Template and command designer types
export interface WorkflowTemplate {
  id: string
  name: string
  description: string
  category: string
  tags: string[]
  workflow: WorkflowStep[]
  parameters: Record<string, WorkflowParameter>
  requirements: WorkflowRequirements
  isPublic: boolean
  authorId: string
  version: string
  downloads: number
  rating: number
  createdAt: Date
  updatedAt: Date
}

export interface CommandDesignerState {
  mode: DesignerMode
  workflow: WorkflowStep[]
  selectedStep: string | null
  draggedStep: WorkflowStep | null
  clipboard: WorkflowStep[]
  history: DesignerHistoryEntry[]
  historyIndex: number
  validation: ValidationResult
  preview: PreviewData
}

export type DesignerMode = 'visual' | 'code' | 'preview'

export interface DesignerHistoryEntry {
  id: string
  action: string
  timestamp: Date
  state: Partial<CommandDesignerState>
  description: string
}

export interface ValidationResult {
  isValid: boolean
  errors: ValidationError[]
  warnings: ValidationWarning[]
  suggestions: ValidationSuggestion[]
}

export interface ValidationError {
  id: string
  message: string
  location: string
  severity: 'error' | 'warning'
  fix?: string
}

export interface ValidationWarning {
  id: string
  message: string
  location: string
  suggestion?: string
}

export interface ValidationSuggestion {
  id: string
  message: string
  action: string
  impact: 'low' | 'medium' | 'high'
}

export interface PreviewData {
  yaml: string
  json: string
  visualization: WorkflowGraphData
  estimatedDuration: number
  resourceRequirements: ResourceLimits
  agentRequirements: string[]
}

// Performance and monitoring types
export interface PerformanceMetrics {
  dashboard: DashboardPerformance
  workflow: WorkflowPerformance
  agent: AgentPerformance
  system: SystemPerformance
}

export interface DashboardPerformance {
  loadTime: number
  renderTime: number
  updateLatency: number
  memoryUsage: number
  fps: number
  errorRate: number
}

export interface WorkflowPerformance {
  executionTime: number
  throughput: number
  successRate: number
  resourceUtilization: number
  bottlenecks: PerformanceBottleneck[]
}

export interface AgentPerformance {
  utilization: number
  responseTime: number
  taskCompletionRate: number
  errorRate: number
  availability: number
}

export interface SystemPerformance {
  cpuUsage: number
  memoryUsage: number
  networkLatency: number
  diskIo: number
  concurrentExecutions: number
}

export interface PerformanceBottleneck {
  id: string
  type: 'cpu' | 'memory' | 'network' | 'agent' | 'dependency'
  location: string
  impact: number
  description: string
  recommendations: string[]
}