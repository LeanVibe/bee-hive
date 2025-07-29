/**
 * Workflow State Management Store
 * 
 * Manages multi-agent workflow state with real-time synchronization,
 * optimistic updates, rollback capabilities, and performance optimization.
 */

import { defineStore } from 'pinia'
import { ref, computed, reactive } from 'vue'
import type {
  Workflow,
  WorkflowExecution,
  WorkflowCommand,
  WorkflowNode,
  WorkflowEdge,
  WorkflowStatus,
  AgentAssignment,
  ExecutionMetrics
} from '@/types/workflows'
import { api } from '@/services/api'
import { useUnifiedWebSocket } from '@/services/unifiedWebSocketManager'

export interface WorkflowState {
  // Command definitions
  commands: Map<string, WorkflowCommand>
  commandCategories: string[]
  commandTags: string[]
  
  // Active workflows
  activeWorkflows: Map<string, Workflow>
  
  // Workflow executions
  executions: Map<string, WorkflowExecution>
  executionHistory: WorkflowExecution[]
  
  // Agent assignments
  agentAssignments: Map<string, AgentAssignment[]>
  
  // System metrics
  metrics: ExecutionMetrics
  
  // UI state
  selectedWorkflow: string | null
  selectedExecution: string | null
  
  // Cache and performance
  lastUpdate: Date | null
  isLoading: boolean
  pendingOperations: Map<string, any>
}

export const useWorkflowStore = defineStore('workflows', () => {
  // Core state
  const state = reactive<WorkflowState>({
    commands: new Map(),
    commandCategories: [],
    commandTags: [],
    activeWorkflows: new Map(),
    executions: new Map(),
    executionHistory: [],
    agentAssignments: new Map(),
    metrics: {
      totalExecutions: 0,
      successfulExecutions: 0,
      failedExecutions: 0,
      averageExecutionTime: 0,
      activeExecutions: 0,
      agentUtilization: 0,
      systemThroughput: 0,
      lastUpdated: new Date()
    },
    selectedWorkflow: null,
    selectedExecution: null,
    lastUpdate: null,
    isLoading: false,
    pendingOperations: new Map()
  })

  // WebSocket integration
  const webSocket = useUnifiedWebSocket()

  // Computed properties
  const commands = computed(() => Array.from(state.commands.values()))
  const activeWorkflows = computed(() => Array.from(state.activeWorkflows.values()))
  const executions = computed(() => Array.from(state.executions.values()))
  const runningExecutions = computed(() => 
    executions.value.filter(exec => exec.status === 'running')
  )
  const completedExecutions = computed(() =>
    executions.value.filter(exec => exec.status === 'completed')
  )
  const failedExecutions = computed(() =>
    executions.value.filter(exec => exec.status === 'failed')
  )

  // Command management
  const loadCommands = async (filters?: { category?: string; tag?: string; limit?: number }) => {
    try {
      state.isLoading = true
      
      const params = new URLSearchParams()
      if (filters?.category) params.append('category', filters.category)
      if (filters?.tag) params.append('tag', filters.tag)
      if (filters?.limit) params.append('limit', filters.limit.toString())
      
      const response = await api.get(`/custom-commands/commands?${params}`)
      const { commands: commandsData, categories, tags } = response.data
      
      // Update commands map
      state.commands.clear()
      commandsData.forEach((cmd: any) => {
        state.commands.set(cmd.name, {
          id: cmd.name,
          name: cmd.name,
          version: cmd.version,
          description: cmd.description,
          category: cmd.category,
          tags: cmd.tags || [],
          workflow: cmd.workflow || [],
          parameters: cmd.parameters || {},
          requirements: cmd.requirements || {},
          metadata: cmd.metadata || {},
          isEnabled: cmd.is_enabled !== false,
          createdAt: new Date(cmd.created_at),
          updatedAt: new Date(cmd.updated_at)
        })
      })
      
      state.commandCategories = categories || []
      state.commandTags = tags || []
      state.lastUpdate = new Date()
      
    } catch (error) {
      console.error('Failed to load commands:', error)
      throw error
    } finally {
      state.isLoading = false
    }
  }

  const createCommand = async (command: Partial<WorkflowCommand>): Promise<string> => {
    try {
      const operationId = generateOperationId()
      state.pendingOperations.set(operationId, { type: 'create_command', data: command })
      
      const response = await api.post('/custom-commands/commands', {
        definition: {
          name: command.name,
          version: command.version || '1.0.0',
          description: command.description,
          category: command.category || 'general',
          tags: command.tags || [],
          workflow: command.workflow || [],
          parameters: command.parameters || {},
          requirements: command.requirements || {}
        },
        validate_agents: true,
        dry_run: false
      })
      
      if (response.data.success) {
        // Optimistic update
        const newCommand: WorkflowCommand = {
          id: command.name!,
          name: command.name!,
          version: command.version || '1.0.0',
          description: command.description || '',
          category: command.category || 'general',
          tags: command.tags || [],
          workflow: command.workflow || [],
          parameters: command.parameters || {},
          requirements: command.requirements || {},
          metadata: {},
          isEnabled: true,
          createdAt: new Date(),
          updatedAt: new Date()
        }
        
        state.commands.set(newCommand.id, newCommand)
        state.pendingOperations.delete(operationId)
        
        return newCommand.id
      }
      
      throw new Error('Command creation failed')
      
    } catch (error) {
      console.error('Failed to create command:', error)
      // Rollback optimistic update if needed
      rollbackOperation('create_command', command.name!)
      throw error
    }
  }

  const updateCommand = async (commandId: string, updates: Partial<WorkflowCommand>): Promise<void> => {
    try {
      const operationId = generateOperationId()
      const originalCommand = state.commands.get(commandId)
      
      if (!originalCommand) {
        throw new Error(`Command ${commandId} not found`)
      }
      
      state.pendingOperations.set(operationId, { 
        type: 'update_command', 
        commandId, 
        original: { ...originalCommand } 
      })
      
      // Optimistic update
      const updatedCommand = { ...originalCommand, ...updates, updatedAt: new Date() }
      state.commands.set(commandId, updatedCommand)
      
      const response = await api.put(`/custom-commands/commands/${commandId}`, {
        definition: {
          name: updatedCommand.name,
          version: updatedCommand.version,
          description: updatedCommand.description,
          category: updatedCommand.category,
          tags: updatedCommand.tags,
          workflow: updatedCommand.workflow,
          parameters: updatedCommand.parameters,
          requirements: updatedCommand.requirements
        }
      })
      
      if (response.data.success) {
        state.pendingOperations.delete(operationId)
      } else {
        throw new Error('Command update failed')
      }
      
    } catch (error) {
      console.error('Failed to update command:', error)
      rollbackOperation('update_command', commandId)
      throw error
    }
  }

  const deleteCommand = async (commandId: string): Promise<void> => {
    try {
      const operationId = generateOperationId()
      const originalCommand = state.commands.get(commandId)
      
      if (!originalCommand) {
        throw new Error(`Command ${commandId} not found`)
      }
      
      state.pendingOperations.set(operationId, { 
        type: 'delete_command', 
        commandId, 
        original: { ...originalCommand } 
      })
      
      // Optimistic update
      state.commands.delete(commandId)
      
      await api.delete(`/custom-commands/commands/${commandId}`)
      state.pendingOperations.delete(operationId)
      
    } catch (error) {
      console.error('Failed to delete command:', error)
      rollbackOperation('delete_command', commandId)
      throw error
    }
  }

  // Workflow execution management
  const executeWorkflow = async (
    commandName: string, 
    parameters: Record<string, any> = {},
    options?: { priority?: 'high' | 'medium' | 'low'; timeout?: number }
  ): Promise<string> => {
    try {
      const executionRequest = {
        command_name: commandName,
        parameters,
        execution_options: {
          priority: options?.priority || 'medium',
          max_execution_time_seconds: options?.timeout || 3600,
          retry_on_failure: true,
          parallel_execution: true
        }
      }
      
      const response = await api.post('/custom-commands/execute', executionRequest)
      const execution = response.data
      
      // Add to executions
      const workflowExecution: WorkflowExecution = {
        id: execution.execution_id,
        commandName: execution.command_name,
        status: execution.status,
        parameters: execution.parameters,
        startTime: new Date(execution.start_time),
        endTime: execution.end_time ? new Date(execution.end_time) : null,
        duration: execution.total_execution_time_seconds,
        steps: execution.step_results || [],
        agentAssignments: execution.agent_assignments || [],
        metrics: execution.execution_metrics || {},
        logs: execution.logs || [],
        errors: execution.errors || []
      }
      
      state.executions.set(workflowExecution.id, workflowExecution)
      updateMetrics()
      
      return workflowExecution.id
      
    } catch (error) {
      console.error('Failed to execute workflow:', error)
      throw error
    }
  }

  const cancelExecution = async (executionId: string, reason = 'User requested'): Promise<void> => {
    try {
      await api.post(`/custom-commands/executions/${executionId}/cancel`, { reason })
      
      // Update local state
      const execution = state.executions.get(executionId)
      if (execution) {
        execution.status = 'cancelled'
        execution.endTime = new Date()
        execution.errors.push({
          message: `Execution cancelled: ${reason}`,
          timestamp: new Date().toISOString(),
          severity: 'info'
        })
      }
      
      updateMetrics()
      
    } catch (error) {
      console.error('Failed to cancel execution:', error)
      throw error
    }
  }

  const getExecutionStatus = async (executionId: string): Promise<WorkflowExecution | null> => {
    try {
      const response = await api.get(`/custom-commands/executions/${executionId}/status`)
      const statusData = response.data
      
      const execution: WorkflowExecution = {
        id: statusData.execution_id,
        commandName: statusData.command_name || '',
        status: statusData.status,
        parameters: {},
        startTime: new Date(),
        endTime: null,
        duration: 0,
        steps: [],
        agentAssignments: [],
        metrics: {},
        logs: statusData.logs || [],
        errors: [],
        progress: statusData.progress_percentage,
        currentStep: statusData.current_step,
        estimatedCompletion: statusData.estimated_completion ? 
          new Date(statusData.estimated_completion) : null
      }
      
      // Update local state
      state.executions.set(executionId, execution)
      return execution
      
    } catch (error) {
      console.error('Failed to get execution status:', error)
      return null
    }
  }

  // Agent assignment management
  const loadAgentAssignments = async (): Promise<void> => {
    try {
      const response = await api.get('/custom-commands/agents/workload')
      const workloadData = response.data
      
      state.agentAssignments.clear()
      Object.entries(workloadData).forEach(([agentId, data]: [string, any]) => {
        const assignments: AgentAssignment[] = (data.current_tasks || []).map((task: any) => ({
          id: task.task_id || generateId(),
          agentId,
          executionId: task.execution_id || '',
          taskName: task.task_name || '',
          status: task.status || 'assigned',
          assignedAt: new Date(task.assigned_at || Date.now()),
          startedAt: task.started_at ? new Date(task.started_at) : null,
          completedAt: task.completed_at ? new Date(task.completed_at) : null,
          estimatedDuration: task.estimated_duration || 0,
          actualDuration: task.actual_duration || 0,
          capabilities: task.required_capabilities || [],
          metadata: task.metadata || {}
        }))
        
        state.agentAssignments.set(agentId, assignments)
      })
      
    } catch (error) {
      console.error('Failed to load agent assignments:', error)
      throw error
    }
  }

  // Real-time updates via WebSocket
  const subscribeToUpdates = (): void => {
    // Subscribe to workflow execution updates
    webSocket.onMessage('workflow_execution_update', (message) => {
      const { execution_id, status, progress, step_results } = message.data
      
      const execution = state.executions.get(execution_id)
      if (execution) {
        execution.status = status
        execution.progress = progress
        if (step_results) {
          execution.steps = step_results
        }
        
        if (status === 'completed' || status === 'failed') {
          execution.endTime = new Date()
          execution.duration = execution.endTime.getTime() - execution.startTime.getTime()
        }
        
        updateMetrics()
      }
    })
    
    // Subscribe to agent assignment updates
    webSocket.onMessage('agent_assignment_update', (message) => {
      const { agent_id, assignments } = message.data
      state.agentAssignments.set(agent_id, assignments)
    })
    
    // Subscribe to command updates
    webSocket.onMessage('command_update', (message) => {
      const { command_name, action, command_data } = message.data
      
      if (action === 'created' || action === 'updated') {
        state.commands.set(command_name, command_data)
      } else if (action === 'deleted') {
        state.commands.delete(command_name)
      }
    })
    
    // Subscribe to system metrics updates
    webSocket.onMessage('system_metrics_update', (message) => {
      state.metrics = { ...state.metrics, ...message.data }
    })
  }

  // Utility functions
  const generateOperationId = (): string => {
    return `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  const generateId = (): string => {
    return `id_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  const rollbackOperation = (operationType: string, commandId: string): void => {
    const operations = Array.from(state.pendingOperations.entries())
    const operation = operations.find(([_, op]) => 
      op.type === operationType && (op.commandId === commandId || op.data?.name === commandId)
    )
    
    if (operation) {
      const [operationId, operationData] = operation
      
      switch (operationType) {
        case 'create_command':
          state.commands.delete(commandId)
          break
        case 'update_command':
          if (operationData.original) {
            state.commands.set(commandId, operationData.original)
          }
          break
        case 'delete_command':
          if (operationData.original) {
            state.commands.set(commandId, operationData.original)
          }
          break
      }
      
      state.pendingOperations.delete(operationId)
    }
  }

  const updateMetrics = (): void => {
    const totalExecutions = state.executions.size
    const successfulExecutions = executions.value.filter(e => e.status === 'completed').length
    const failedExecutions = executions.value.filter(e => e.status === 'failed').length
    const activeExecutions = runningExecutions.value.length
    
    const completedExecutionsWithDuration = executions.value.filter(e => 
      e.status === 'completed' && e.duration
    )
    const averageExecutionTime = completedExecutionsWithDuration.length > 0
      ? completedExecutionsWithDuration.reduce((sum, e) => sum + (e.duration || 0), 0) / completedExecutionsWithDuration.length
      : 0
    
    // Calculate agent utilization
    const totalAgents = state.agentAssignments.size
    const busyAgents = Array.from(state.agentAssignments.values())
      .filter(assignments => assignments.some(a => a.status === 'running')).length
    const agentUtilization = totalAgents > 0 ? (busyAgents / totalAgents) * 100 : 0
    
    state.metrics = {
      totalExecutions,
      successfulExecutions,
      failedExecutions,
      averageExecutionTime,
      activeExecutions,
      agentUtilization,
      systemThroughput: totalExecutions > 0 ? successfulExecutions / totalExecutions : 0,
      lastUpdated: new Date()
    }
  }

  // Selection management
  const selectWorkflow = (workflowId: string | null): void => {
    state.selectedWorkflow = workflowId
  }

  const selectExecution = (executionId: string | null): void => {
    state.selectedExecution = executionId
  }

  // Data refresh
  const refreshAll = async (): Promise<void> => {
    await Promise.all([
      loadCommands(),
      loadAgentAssignments()
    ])
  }

  // Cleanup
  const reset = (): void => {
    state.commands.clear()
    state.activeWorkflows.clear()
    state.executions.clear()
    state.executionHistory = []
    state.agentAssignments.clear()
    state.selectedWorkflow = null
    state.selectedExecution = null
    state.pendingOperations.clear()
    updateMetrics()
  }

  return {
    // State
    state,
    
    // Computed
    commands,
    activeWorkflows,
    executions,
    runningExecutions,
    completedExecutions,
    failedExecutions,
    
    // Command management
    loadCommands,
    createCommand,
    updateCommand,
    deleteCommand,
    
    // Execution management
    executeWorkflow,
    cancelExecution,
    getExecutionStatus,
    
    // Agent assignments
    loadAgentAssignments,
    
    // Real-time updates
    subscribeToUpdates,
    
    // Selection
    selectWorkflow,
    selectExecution,
    
    // Utility
    refreshAll,
    reset
  }
})