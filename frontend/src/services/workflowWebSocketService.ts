/**
 * Workflow WebSocket Service
 * 
 * Extends the UnifiedWebSocketManager with workflow-specific functionality
 * for real-time workflow execution monitoring, command status updates,
 * and multi-agent coordination events.
 */

import { ref, reactive, computed } from 'vue'
import type {
  WorkflowExecutionUpdateMessage,
  NodeStatusUpdateMessage,
  AgentAssignmentUpdateMessage,
  CommandUpdateMessage,
  SystemMetricsUpdateMessage,
  ResourceUsageUpdateMessage,
  WebSocketMessage
} from '@/types/coordination'
import type {
  WorkflowExecution,
  ExecutionStatus,
  AgentAssignment,
  ExecutionMetrics
} from '@/types/workflows'
import { useUnifiedWebSocket } from './unifiedWebSocketManager'
import { useWorkflowStore } from '@/stores/workflows'

export interface WorkflowWebSocketConfig {
  autoReconnect: boolean
  maxReconnectAttempts: number
  reconnectDelay: number
  heartbeatInterval: number
  subscriptionTimeout: number
}

export interface WorkflowSubscription {
  id: string
  type: WorkflowSubscriptionType
  executionId?: string
  agentId?: string
  commandName?: string
  callback: (data: any) => void
  active: boolean
  createdAt: Date
}

export type WorkflowSubscriptionType = 
  | 'execution_updates'
  | 'node_status'
  | 'agent_assignments'
  | 'command_updates'
  | 'system_metrics'
  | 'resource_usage'
  | 'all_workflow_events'

class WorkflowWebSocketService {
  private static instance: WorkflowWebSocketService
  private unifiedWebSocket = useUnifiedWebSocket()
  private workflowStore = useWorkflowStore()
  
  // Configuration
  private config: WorkflowWebSocketConfig = {
    autoReconnect: true,
    maxReconnectAttempts: 10,
    reconnectDelay: 5000,
    heartbeatInterval: 30000,
    subscriptionTimeout: 60000
  }

  // State management
  private state = reactive({
    isConnected: false,
    lastUpdate: null as Date | null,
    subscriptions: new Map<string, WorkflowSubscription>(),
    messageQueue: [] as WebSocketMessage[],
    statistics: {
      messagesReceived: 0,
      messagesProcessed: 0,
      errors: 0,
      subscriptions: 0,
      lastError: null as Error | null
    }
  })

  // WebSocket endpoint configurations
  private endpoints = {
    workflow_execution: {
      id: 'workflow_execution',
      url: 'ws://localhost:8000/ws/workflow/executions',
      protocols: ['workflow-v1'],
      component: 'WORKFLOW' as any,
      priority: 'high' as const,
      autoReconnect: true,
      maxReconnectAttempts: this.config.maxReconnectAttempts,
      reconnectDelay: this.config.reconnectDelay
    },
    command_management: {
      id: 'command_management',
      url: 'ws://localhost:8000/ws/commands',
      protocols: ['command-v1'],
      component: 'COMMAND' as any,
      priority: 'medium' as const,
      autoReconnect: true,
      maxReconnectAttempts: this.config.maxReconnectAttempts,
      reconnectDelay: this.config.reconnectDelay
    },
    system_metrics: {
      id: 'system_metrics',
      url: 'ws://localhost:8000/ws/metrics',
      protocols: ['metrics-v1'],
      component: 'METRICS' as any,
      priority: 'low' as const,
      autoReconnect: true,
      maxReconnectAttempts: this.config.maxReconnectAttempts,
      reconnectDelay: this.config.reconnectDelay
    }
  }

  private constructor() {
    this.initialize()
  }

  static getInstance(): WorkflowWebSocketService {
    if (!WorkflowWebSocketService.instance) {
      WorkflowWebSocketService.instance = new WorkflowWebSocketService()
    }
    return WorkflowWebSocketService.instance
  }

  // Public API
  public readonly isConnected = computed(() => this.state.isConnected)
  public readonly statistics = computed(() => this.state.statistics)
  public readonly subscriptions = computed(() => Array.from(this.state.subscriptions.values()))

  /**
   * Initialize the workflow WebSocket service
   */
  private async initialize(): Promise<void> {
    try {
      // Register workflow endpoints
      Object.values(this.endpoints).forEach(endpoint => {
        this.unifiedWebSocket.registerEndpoint(endpoint)
      })

      // Set up message handlers
      this.setupMessageHandlers()

      // Connect to primary workflow endpoint
      await this.connect()

      console.log('Workflow WebSocket service initialized')
    } catch (error) {
      console.error('Failed to initialize workflow WebSocket service:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Connect to workflow WebSocket endpoints
   */
  public async connect(): Promise<void> {
    try {
      // Connect to workflow execution endpoint (primary)
      await this.unifiedWebSocket.connect('workflow_execution')
      
      // Connect to command management endpoint
      await this.unifiedWebSocket.connect('command_management')
      
      // Connect to system metrics endpoint
      await this.unifiedWebSocket.connect('system_metrics')

      this.state.isConnected = true
      this.state.lastUpdate = new Date()

      console.log('Connected to workflow WebSocket endpoints')
    } catch (error) {
      console.error('Failed to connect to workflow WebSocket endpoints:', error)
      this.handleError(error as Error)
      throw error
    }
  }

  /**
   * Disconnect from all workflow WebSocket endpoints
   */
  public disconnect(): void {
    try {
      // Disconnect from all workflow endpoints
      Object.keys(this.endpoints).forEach(endpointId => {
        this.unifiedWebSocket.disconnect(endpointId)
      })

      // Clear active subscriptions
      this.state.subscriptions.clear()
      this.state.isConnected = false

      console.log('Disconnected from workflow WebSocket endpoints')
    } catch (error) {
      console.error('Error disconnecting from workflow WebSocket:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Subscribe to specific workflow events
   */
  public subscribe<T = any>(
    type: WorkflowSubscriptionType,
    callback: (data: T) => void,
    options?: {
      executionId?: string
      agentId?: string
      commandName?: string
    }
  ): string {
    const subscriptionId = this.generateSubscriptionId()
    
    const subscription: WorkflowSubscription = {
      id: subscriptionId,
      type,
      executionId: options?.executionId,
      agentId: options?.agentId,
      commandName: options?.commandName,
      callback,
      active: true,
      createdAt: new Date()
    }

    this.state.subscriptions.set(subscriptionId, subscription)
    this.state.statistics.subscriptions++

    // Send subscription message to appropriate endpoint
    this.sendSubscriptionMessage(subscription)

    console.log(`Created workflow subscription: ${type} (${subscriptionId})`)
    return subscriptionId
  }

  /**
   * Unsubscribe from workflow events
   */
  public unsubscribe(subscriptionId: string): boolean {
    const subscription = this.state.subscriptions.get(subscriptionId)
    if (!subscription) {
      return false
    }

    // Send unsubscription message
    this.sendUnsubscriptionMessage(subscription)

    // Remove from local state
    this.state.subscriptions.delete(subscriptionId)
    this.state.statistics.subscriptions--

    console.log(`Removed workflow subscription: ${subscriptionId}`)
    return true
  }

  /**
   * Subscribe to execution-specific updates
   */
  public subscribeToExecution(
    executionId: string,
    callback: (execution: WorkflowExecution) => void
  ): string {
    return this.subscribe('execution_updates', callback, { executionId })
  }

  /**
   * Subscribe to agent assignment updates
   */
  public subscribeToAgentUpdates(
    agentId: string,
    callback: (assignments: AgentAssignment[]) => void
  ): string {
    return this.subscribe('agent_assignments', callback, { agentId })
  }

  /**
   * Subscribe to command lifecycle events
   */
  public subscribeToCommandUpdates(
    commandName: string,
    callback: (update: any) => void
  ): string {
    return this.subscribe('command_updates', callback, { commandName })
  }

  /**
   * Subscribe to system-wide metrics updates
   */
  public subscribeToSystemMetrics(
    callback: (metrics: ExecutionMetrics) => void
  ): string {
    return this.subscribe('system_metrics', callback)
  }

  /**
   * Send control message to workflow execution
   */
  public sendExecutionControl(
    executionId: string,
    action: 'pause' | 'resume' | 'cancel' | 'emergency_stop',
    reason?: string
  ): boolean {
    const message: WebSocketMessage = {
      type: 'execution_control',
      data: {
        execution_id: executionId,
        action,
        reason,
        timestamp: new Date().toISOString()
      },
      timestamp: new Date().toISOString()
    }

    return this.unifiedWebSocket.send('workflow_execution', message)
  }

  /**
   * Request execution status update
   */
  public requestExecutionStatus(executionId: string): boolean {
    const message: WebSocketMessage = {
      type: 'get_execution_status',
      data: { execution_id: executionId },
      timestamp: new Date().toISOString()
    }

    return this.unifiedWebSocket.send('workflow_execution', message)
  }

  /**
   * Request agent workload update
   */
  public requestAgentWorkload(agentId?: string): boolean {
    const message: WebSocketMessage = {
      type: 'get_agent_workload',
      data: agentId ? { agent_id: agentId } : {},
      timestamp: new Date().toISOString()
    }

    return this.unifiedWebSocket.send('workflow_execution', message)
  }

  // Private methods

  /**
   * Set up message handlers for workflow-specific messages
   */
  private setupMessageHandlers(): void {
    // Workflow execution updates
    this.unifiedWebSocket.onMessage('workflow_execution_update', (message: WorkflowExecutionUpdateMessage) => {
      this.handleExecutionUpdate(message)
    })

    // Node status updates
    this.unifiedWebSocket.onMessage('node_status_update', (message: NodeStatusUpdateMessage) => {
      this.handleNodeStatusUpdate(message)
    })

    // Agent assignment updates
    this.unifiedWebSocket.onMessage('agent_assignment_update', (message: AgentAssignmentUpdateMessage) => {
      this.handleAgentAssignmentUpdate(message)
    })

    // Command updates
    this.unifiedWebSocket.onMessage('command_update', (message: CommandUpdateMessage) => {
      this.handleCommandUpdate(message)
    })

    // System metrics updates
    this.unifiedWebSocket.onMessage('system_metrics_update', (message: SystemMetricsUpdateMessage) => {
      this.handleSystemMetricsUpdate(message)
    })

    // Resource usage updates
    this.unifiedWebSocket.onMessage('resource_usage_update', (message: ResourceUsageUpdateMessage) => {
      this.handleResourceUsageUpdate(message)
    })

    // Generic workflow events
    this.unifiedWebSocket.onMessage('workflow_event', (message: WebSocketMessage) => {
      this.handleGenericWorkflowEvent(message)
    })
  }

  /**
   * Handle workflow execution updates
   */
  private handleExecutionUpdate(message: WorkflowExecutionUpdateMessage): void {
    try {
      const { execution_id, status, progress, current_step, step_results, agent_assignments, logs, errors } = message.data

      // Update store
      const execution = this.workflowStore.state.executions.get(execution_id)
      if (execution) {
        execution.status = status as ExecutionStatus
        execution.progress = progress
        execution.currentStep = current_step
        
        if (step_results) execution.steps = step_results
        if (agent_assignments) execution.agentAssignments = agent_assignments
        if (logs) execution.logs.push(...logs)
        if (errors) execution.errors.push(...errors)
      }

      // Notify subscribers
      this.notifySubscribers('execution_updates', execution, { executionId: execution_id })

      this.state.statistics.messagesProcessed++
      this.state.lastUpdate = new Date()

    } catch (error) {
      console.error('Error handling execution update:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Handle node status updates
   */
  private handleNodeStatusUpdate(message: NodeStatusUpdateMessage): void {
    try {
      const { node_id, status, progress, agent_id, execution_id } = message.data

      // Notify subscribers
      this.notifySubscribers('node_status', message.data)

      this.state.statistics.messagesProcessed++
      this.state.lastUpdate = new Date()

    } catch (error) {
      console.error('Error handling node status update:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Handle agent assignment updates
   */
  private handleAgentAssignmentUpdate(message: AgentAssignmentUpdateMessage): void {
    try {
      const { agent_id, assignments, workload, status, utilization } = message.data

      // Update store
      this.workflowStore.state.agentAssignments.set(agent_id, assignments)

      // Notify subscribers
      this.notifySubscribers('agent_assignments', assignments, { agentId: agent_id })

      this.state.statistics.messagesProcessed++
      this.state.lastUpdate = new Date()

    } catch (error) {
      console.error('Error handling agent assignment update:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Handle command updates
   */
  private handleCommandUpdate(message: CommandUpdateMessage): void {
    try {
      const { command_name, action, command_data } = message.data

      // Update store based on action
      if (action === 'created' || action === 'updated') {
        if (command_data) {
          this.workflowStore.state.commands.set(command_name, command_data)
        }
      } else if (action === 'deleted') {
        this.workflowStore.state.commands.delete(command_name)
      }

      // Notify subscribers
      this.notifySubscribers('command_updates', message.data, { commandName: command_name })

      this.state.statistics.messagesProcessed++
      this.state.lastUpdate = new Date()

    } catch (error) {
      console.error('Error handling command update:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Handle system metrics updates
   */
  private handleSystemMetricsUpdate(message: SystemMetricsUpdateMessage): void {
    try {
      const metrics = message.data

      // Update store
      this.workflowStore.state.metrics = {
        ...this.workflowStore.state.metrics,
        ...metrics,
        lastUpdated: new Date()
      }

      // Notify subscribers
      this.notifySubscribers('system_metrics', metrics)

      this.state.statistics.messagesProcessed++
      this.state.lastUpdate = new Date()

    } catch (error) {
      console.error('Error handling system metrics update:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Handle resource usage updates
   */
  private handleResourceUsageUpdate(message: ResourceUsageUpdateMessage): void {
    try {
      const { execution_id, usage } = message.data

      // Notify subscribers
      this.notifySubscribers('resource_usage', { executionId: execution_id, usage })

      this.state.statistics.messagesProcessed++
      this.state.lastUpdate = new Date()

    } catch (error) {
      console.error('Error handling resource usage update:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Handle generic workflow events
   */
  private handleGenericWorkflowEvent(message: WebSocketMessage): void {
    try {
      // Notify all subscribers
      this.notifySubscribers('all_workflow_events', message.data)

      this.state.statistics.messagesProcessed++
      this.state.lastUpdate = new Date()

    } catch (error) {
      console.error('Error handling generic workflow event:', error)
      this.handleError(error as Error)
    }
  }

  /**
   * Notify subscribers of updates
   */
  private notifySubscribers(
    type: WorkflowSubscriptionType,
    data: any,
    context?: { executionId?: string; agentId?: string; commandName?: string }
  ): void {
    for (const subscription of this.state.subscriptions.values()) {
      if (!subscription.active || subscription.type !== type) {
        continue
      }

      // Check context filters
      if (context?.executionId && subscription.executionId && subscription.executionId !== context.executionId) {
        continue
      }
      if (context?.agentId && subscription.agentId && subscription.agentId !== context.agentId) {
        continue
      }
      if (context?.commandName && subscription.commandName && subscription.commandName !== context.commandName) {
        continue
      }

      try {
        subscription.callback(data)
      } catch (error) {
        console.error(`Error in subscription callback ${subscription.id}:`, error)
        this.handleError(error as Error)
      }
    }
  }

  /**
   * Send subscription message to appropriate endpoint
   */
  private sendSubscriptionMessage(subscription: WorkflowSubscription): void {
    const endpointId = this.getEndpointForSubscription(subscription.type)
    
    const message: WebSocketMessage = {
      type: 'subscribe',
      data: {
        subscription_type: subscription.type,
        subscription_id: subscription.id,
        filters: {
          execution_id: subscription.executionId,
          agent_id: subscription.agentId,
          command_name: subscription.commandName
        }
      },
      timestamp: new Date().toISOString()
    }

    this.unifiedWebSocket.send(endpointId, message)
  }

  /**
   * Send unsubscription message
   */
  private sendUnsubscriptionMessage(subscription: WorkflowSubscription): void {
    const endpointId = this.getEndpointForSubscription(subscription.type)
    
    const message: WebSocketMessage = {
      type: 'unsubscribe',
      data: {
        subscription_id: subscription.id
      },
      timestamp: new Date().toISOString()
    }

    this.unifiedWebSocket.send(endpointId, message)
  }

  /**
   * Get appropriate endpoint for subscription type
   */
  private getEndpointForSubscription(type: WorkflowSubscriptionType): string {
    switch (type) {
      case 'execution_updates':
      case 'node_status':
      case 'agent_assignments':
        return 'workflow_execution'
      case 'command_updates':
        return 'command_management'
      case 'system_metrics':
      case 'resource_usage':
        return 'system_metrics'
      case 'all_workflow_events':
        return 'workflow_execution'
      default:
        return 'workflow_execution'
    }
  }

  /**
   * Generate unique subscription ID
   */
  private generateSubscriptionId(): string {
    return `wf_sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  /**
   * Handle errors
   */
  private handleError(error: Error): void {
    this.state.statistics.errors++
    this.state.statistics.lastError = error
    console.error('Workflow WebSocket Service Error:', error)
  }

  /**
   * Get service statistics
   */
  public getStatistics() {
    return {
      ...this.state.statistics,
      subscriptions: this.state.subscriptions.size,
      endpoints: Object.keys(this.endpoints).length,
      uptime: this.state.lastUpdate ? Date.now() - this.state.lastUpdate.getTime() : 0
    }
  }

  /**
   * Cleanup and destroy service
   */
  public destroy(): void {
    // Clear all subscriptions
    this.state.subscriptions.clear()
    
    // Disconnect from endpoints
    this.disconnect()
    
    // Unregister endpoints
    Object.keys(this.endpoints).forEach(endpointId => {
      this.unifiedWebSocket.unregisterEndpoint(endpointId)
    })
    
    console.log('Workflow WebSocket service destroyed')
  }
}

// Export singleton instance
export const workflowWebSocketService = WorkflowWebSocketService.getInstance()

// Vue composable for easy integration
export function useWorkflowWebSocket() {
  return {
    // State
    isConnected: workflowWebSocketService.isConnected,
    statistics: workflowWebSocketService.statistics,
    subscriptions: workflowWebSocketService.subscriptions,
    
    // Connection management
    connect: workflowWebSocketService.connect.bind(workflowWebSocketService),
    disconnect: workflowWebSocketService.disconnect.bind(workflowWebSocketService),
    
    // Subscription management
    subscribe: workflowWebSocketService.subscribe.bind(workflowWebSocketService),
    unsubscribe: workflowWebSocketService.unsubscribe.bind(workflowWebSocketService),
    
    // Specific subscriptions
    subscribeToExecution: workflowWebSocketService.subscribeToExecution.bind(workflowWebSocketService),
    subscribeToAgentUpdates: workflowWebSocketService.subscribeToAgentUpdates.bind(workflowWebSocketService),
    subscribeToCommandUpdates: workflowWebSocketService.subscribeToCommandUpdates.bind(workflowWebSocketService),
    subscribeToSystemMetrics: workflowWebSocketService.subscribeToSystemMetrics.bind(workflowWebSocketService),
    
    // Control operations
    sendExecutionControl: workflowWebSocketService.sendExecutionControl.bind(workflowWebSocketService),
    requestExecutionStatus: workflowWebSocketService.requestExecutionStatus.bind(workflowWebSocketService),
    requestAgentWorkload: workflowWebSocketService.requestAgentWorkload.bind(workflowWebSocketService),
    
    // Utilities
    getStatistics: workflowWebSocketService.getStatistics.bind(workflowWebSocketService)
  }
}

export default workflowWebSocketService