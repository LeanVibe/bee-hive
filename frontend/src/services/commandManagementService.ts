/**
 * Command Management Service
 * 
 * Service layer for integrating with Phase 6.1 Custom Commands API
 * Provides comprehensive command CRUD operations, execution management,
 * validation, and real-time status monitoring.
 */

import { api } from './api'
import type { 
  WorkflowCommand, 
  WorkflowExecution, 
  ExecutionMetrics,
  ValidationResult,
  AgentAssignment
} from '@/types/workflows'

export interface CommandListOptions {
  category?: string
  tag?: string
  agent_role?: string
  limit?: number
  offset?: number
}

export interface CommandExecutionOptions {
  priority?: 'high' | 'medium' | 'low'
  max_execution_time_seconds?: number
  retry_on_failure?: boolean
  parallel_execution?: boolean
  agent_preferences?: string[]
}

export interface CommandCreateRequest {
  definition: {
    name: string
    version: string
    description: string
    category: string
    tags: string[]
    workflow: any[]
    parameters: Record<string, any>
    requirements: Record<string, any>
  }
  validate_agents?: boolean
  dry_run?: boolean
}

export interface CommandExecutionRequest {
  command_name: string
  parameters: Record<string, any>
  execution_options?: CommandExecutionOptions
}

export interface CommandStatusResponse {
  execution_id: string
  status: string
  progress_percentage: number
  current_step?: string
  estimated_completion?: string
  logs: any[]
}

export interface SystemMetricsResponse {
  timestamp: string
  execution_statistics: {
    active_executions: number
    total_executions: number
    successful_executions: number
    failed_executions: number
    average_execution_time: number
    peak_concurrent_executions: number
  }
  distribution_statistics: {
    tasks_distributed: number
    agents_utilized: number
    distribution_efficiency: number
  }
  agent_workload: Record<string, {
    utilization: number
    current_tasks: number
    capacity: number
    performance_score: number
    status: string
  }>
  system_health: {
    active_executions: number
    success_rate: number
    average_execution_time: number
    peak_concurrent_executions: number
  }
}

export class CommandManagementService {
  private static instance: CommandManagementService
  private cache: Map<string, any> = new Map()
  private cacheTimeout = 60000 // 1 minute

  static getInstance(): CommandManagementService {
    if (!CommandManagementService.instance) {
      CommandManagementService.instance = new CommandManagementService()
    }
    return CommandManagementService.instance
  }

  /**
   * Command CRUD Operations
   */

  async listCommands(options: CommandListOptions = {}): Promise<{
    commands: WorkflowCommand[]
    total: number
    categories: string[]
    tags: string[]
  }> {
    try {
      const params = new URLSearchParams()
      if (options.category) params.append('category', options.category)
      if (options.tag) params.append('tag', options.tag)
      if (options.agent_role) params.append('agent_role', options.agent_role)
      if (options.limit) params.append('limit', options.limit.toString())
      if (options.offset) params.append('offset', options.offset.toString())

      const cacheKey = `commands_${params.toString()}`
      if (this.isCached(cacheKey)) {
        return this.getFromCache(cacheKey)
      }

      const response = await api.get(`/custom-commands/commands?${params}`)
      const data = response.data

      const result = {
        commands: data.commands.map((cmd: any) => this.transformCommand(cmd)),
        total: data.total,
        categories: data.categories || [],
        tags: data.tags || []
      }

      this.setCache(cacheKey, result)
      return result

    } catch (error) {
      console.error('Failed to list commands:', error)
      throw new Error(`Failed to load commands: ${error.message}`)
    }
  }

  async getCommand(commandName: string, version?: string): Promise<WorkflowCommand | null> {
    try {
      const params = version ? `?version=${version}` : ''
      const cacheKey = `command_${commandName}_${version || 'latest'}`
      
      if (this.isCached(cacheKey)) {
        return this.getFromCache(cacheKey)
      }

      const response = await api.get(`/custom-commands/commands/${commandName}${params}`)
      const data = response.data

      if (!data.command_definition) {
        return null
      }

      const command = this.transformCommand(data.command_definition)
      this.setCache(cacheKey, command)
      return command

    } catch (error) {
      if (error.response?.status === 404) {
        return null
      }
      console.error('Failed to get command:', error)
      throw new Error(`Failed to load command: ${error.message}`)
    }
  }

  async createCommand(request: CommandCreateRequest): Promise<{
    success: boolean
    command_name: string
    validation_result: ValidationResult
  }> {
    try {
      const response = await api.post('/custom-commands/commands', request)
      const data = response.data

      // Invalidate related caches
      this.invalidateCache('commands_')
      this.invalidateCache(`command_${request.definition.name}_`)

      return {
        success: data.success,
        command_name: data.command_name,
        validation_result: this.transformValidationResult(data.validation_result)
      }

    } catch (error) {
      console.error('Failed to create command:', error)
      if (error.response?.status === 400) {
        throw new Error(`Command validation failed: ${error.response.data.detail.message}`)
      }
      throw new Error(`Failed to create command: ${error.message}`)
    }
  }

  async updateCommand(commandName: string, updates: Partial<WorkflowCommand>): Promise<{
    success: boolean
    message: string
  }> {
    try {
      const request = {
        definition: {
          name: updates.name || commandName,
          version: updates.version,
          description: updates.description,
          category: updates.category,
          tags: updates.tags,
          workflow: updates.workflow,
          parameters: updates.parameters,
          requirements: updates.requirements
        }
      }

      const response = await api.put(`/custom-commands/commands/${commandName}`, request)
      const data = response.data

      // Invalidate related caches
      this.invalidateCache('commands_')
      this.invalidateCache(`command_${commandName}_`)

      return {
        success: data.success,
        message: data.message
      }

    } catch (error) {
      console.error('Failed to update command:', error)
      throw new Error(`Failed to update command: ${error.message}`)
    }
  }

  async deleteCommand(commandName: string, version?: string): Promise<{
    success: boolean
    message: string
  }> {
    try {
      const params = version ? `?version=${version}` : ''
      const response = await api.delete(`/custom-commands/commands/${commandName}${params}`)
      const data = response.data

      // Invalidate related caches
      this.invalidateCache('commands_')
      this.invalidateCache(`command_${commandName}_`)

      return {
        success: data.success,
        message: data.message
      }

    } catch (error) {
      console.error('Failed to delete command:', error)
      if (error.response?.status === 404) {
        throw new Error('Command not found or access denied')
      }
      throw new Error(`Failed to delete command: ${error.message}`)
    }
  }

  async validateCommand(command: Partial<WorkflowCommand>, validateAgents = true): Promise<ValidationResult> {
    try {
      const request = {
        name: command.name,
        version: command.version || '1.0.0',
        description: command.description,
        category: command.category,
        tags: command.tags,
        workflow: command.workflow,
        parameters: command.parameters,
        requirements: command.requirements
      }

      const response = await api.post(`/custom-commands/validate?validate_agents=${validateAgents}`, request)
      return this.transformValidationResult(response.data)

    } catch (error) {
      console.error('Failed to validate command:', error)
      throw new Error(`Command validation failed: ${error.message}`)
    }
  }

  /**
   * Command Execution Operations
   */

  async executeCommand(request: CommandExecutionRequest): Promise<WorkflowExecution> {
    try {
      const response = await api.post('/custom-commands/execute', request)
      const data = response.data

      return this.transformExecution(data)

    } catch (error) {
      console.error('Failed to execute command:', error)
      if (error.response?.status === 400) {
        throw new Error(`Execution validation failed: ${error.response.data.detail}`)
      }
      if (error.response?.status === 429) {
        throw new Error('System overloaded. Please try again later.')
      }
      throw new Error(`Failed to execute command: ${error.message}`)
    }
  }

  async getExecutionStatus(executionId: string): Promise<CommandStatusResponse | null> {
    try {
      const cacheKey = `execution_status_${executionId}`
      
      // Don't cache execution status - it changes frequently
      const response = await api.get(`/custom-commands/executions/${executionId}/status`)
      return response.data

    } catch (error) {
      if (error.response?.status === 404) {
        return null
      }
      console.error('Failed to get execution status:', error)
      throw new Error(`Failed to get execution status: ${error.message}`)
    }
  }

  async cancelExecution(executionId: string, reason = 'User requested'): Promise<{
    success: boolean
    message: string
  }> {
    try {
      const response = await api.post(`/custom-commands/executions/${executionId}/cancel`, { reason })
      return response.data

    } catch (error) {
      console.error('Failed to cancel execution:', error)
      if (error.response?.status === 404) {
        throw new Error('Execution not found or already completed')
      }
      throw new Error(`Failed to cancel execution: ${error.message}`)
    }
  }

  async listActiveExecutions(): Promise<WorkflowExecution[]> {
    try {
      const response = await api.get('/custom-commands/executions')
      const executions = response.data

      return executions.map((exec: any) => this.transformExecution(exec))

    } catch (error) {
      console.error('Failed to list active executions:', error)
      throw new Error(`Failed to load active executions: ${error.message}`)
    }
  }

  /**
   * System Monitoring Operations
   */

  async getSystemMetrics(): Promise<SystemMetricsResponse> {
    try {
      const cacheKey = 'system_metrics'
      
      // Short cache for metrics (30 seconds)
      if (this.isCached(cacheKey, 30000)) {
        return this.getFromCache(cacheKey)
      }

      const response = await api.get('/custom-commands/metrics')
      const data = response.data

      this.setCache(cacheKey, data, 30000)
      return data

    } catch (error) {
      console.error('Failed to get system metrics:', error)
      throw new Error(`Failed to load system metrics: ${error.message}`)
    }
  }

  async getCommandMetrics(commandName: string): Promise<ExecutionMetrics> {
    try {
      const cacheKey = `command_metrics_${commandName}`
      
      if (this.isCached(cacheKey)) {
        return this.getFromCache(cacheKey)
      }

      const response = await api.get(`/custom-commands/commands/${commandName}/metrics`)
      const data = response.data

      const metrics = this.transformExecutionMetrics(data)
      this.setCache(cacheKey, metrics)
      return metrics

    } catch (error) {
      if (error.response?.status === 404) {
        throw new Error('No metrics found for this command')
      }
      console.error('Failed to get command metrics:', error)
      throw new Error(`Failed to load command metrics: ${error.message}`)
    }
  }

  async getAgentWorkload(): Promise<Record<string, AgentAssignment[]>> {
    try {
      const cacheKey = 'agent_workload'
      
      // Short cache for workload (15 seconds)
      if (this.isCached(cacheKey, 15000)) {
        return this.getFromCache(cacheKey)
      }

      const response = await api.get('/custom-commands/agents/workload')
      const data = response.data

      const workload: Record<string, AgentAssignment[]> = {}
      Object.entries(data).forEach(([agentId, agentData]: [string, any]) => {
        workload[agentId] = (agentData.current_tasks || []).map((task: any) => 
          this.transformAgentAssignment(task, agentId)
        )
      })

      this.setCache(cacheKey, workload, 15000)
      return workload

    } catch (error) {
      console.error('Failed to get agent workload:', error)
      throw new Error(`Failed to load agent workload: ${error.message}`)
    }
  }

  async optimizeDistributionStrategy(historicalData: Record<string, any>): Promise<{
    recommended_strategy: string
    message: string
  }> {
    try {
      const response = await api.post('/custom-commands/distribution/optimize', historicalData)
      return response.data

    } catch (error) {
      if (error.response?.status === 403) {
        throw new Error('Admin access required for optimization')
      }
      console.error('Failed to optimize distribution strategy:', error)
      throw new Error(`Failed to optimize distribution: ${error.message}`)
    }
  }

  async getHealthStatus(): Promise<{
    status: string
    version: string
    components: Record<string, any>
    uptime_seconds: number
  }> {
    try {
      const response = await api.get('/custom-commands/health')
      return response.data

    } catch (error) {
      console.error('Failed to get health status:', error)
      // Return degraded status if health endpoint fails
      return {
        status: 'unhealthy',
        version: 'unknown',
        components: {},
        uptime_seconds: 0
      }
    }
  }

  /**
   * Transform methods to convert API responses to frontend types
   */

  private transformCommand(apiCommand: any): WorkflowCommand {
    return {
      id: apiCommand.name,
      name: apiCommand.name,
      version: apiCommand.version || '1.0.0',
      description: apiCommand.description || '',
      category: apiCommand.category || 'general',
      tags: apiCommand.tags || [],
      workflow: apiCommand.workflow || [],
      parameters: apiCommand.parameters || {},
      requirements: apiCommand.requirements || {},
      metadata: apiCommand.metadata || {},
      isEnabled: apiCommand.is_enabled !== false,
      createdAt: new Date(apiCommand.created_at || Date.now()),
      updatedAt: new Date(apiCommand.updated_at || Date.now())
    }
  }

  private transformExecution(apiExecution: any): WorkflowExecution {
    return {
      id: apiExecution.execution_id,
      commandName: apiExecution.command_name,
      status: apiExecution.status,
      parameters: apiExecution.parameters || {},
      startTime: new Date(apiExecution.start_time),
      endTime: apiExecution.end_time ? new Date(apiExecution.end_time) : null,
      duration: apiExecution.total_execution_time_seconds,
      steps: apiExecution.step_results || [],
      agentAssignments: (apiExecution.agent_assignments || []).map((assignment: any) => 
        this.transformAgentAssignment(assignment, assignment.agent_id)
      ),
      metrics: this.transformExecutionMetrics(apiExecution.execution_metrics),
      logs: apiExecution.logs || [],
      errors: apiExecution.errors || [],
      progress: apiExecution.progress_percentage,
      currentStep: apiExecution.current_step,
      estimatedCompletion: apiExecution.estimated_completion ? 
        new Date(apiExecution.estimated_completion) : null
    }
  }

  private transformAgentAssignment(apiAssignment: any, agentId: string): AgentAssignment {
    return {
      id: apiAssignment.task_id || `assignment_${Date.now()}`,
      agentId,
      executionId: apiAssignment.execution_id || '',
      taskName: apiAssignment.task_name || 'Unknown Task',
      status: apiAssignment.status || 'assigned',
      assignedAt: new Date(apiAssignment.assigned_at || Date.now()),
      startedAt: apiAssignment.started_at ? new Date(apiAssignment.started_at) : null,
      completedAt: apiAssignment.completed_at ? new Date(apiAssignment.completed_at) : null,
      estimatedDuration: apiAssignment.estimated_duration || 0,
      actualDuration: apiAssignment.actual_duration || 0,
      capabilities: apiAssignment.required_capabilities || [],
      metadata: apiAssignment.metadata || {}
    }
  }

  private transformExecutionMetrics(apiMetrics: any): ExecutionMetrics {
    return {
      totalExecutions: apiMetrics?.total_executions || 0,
      successfulExecutions: apiMetrics?.successful_executions || 0,
      failedExecutions: apiMetrics?.failed_executions || 0,
      averageExecutionTime: apiMetrics?.average_execution_time || 0,
      activeExecutions: apiMetrics?.active_executions || 0,
      agentUtilization: apiMetrics?.agent_utilization || 0,
      systemThroughput: apiMetrics?.system_throughput || 0,
      lastUpdated: new Date()
    }
  }

  private transformValidationResult(apiResult: any): ValidationResult {
    return {
      isValid: apiResult?.is_valid || false,
      errors: apiResult?.errors || [],
      warnings: apiResult?.warnings || [],
      suggestions: apiResult?.suggestions || []
    }
  }

  /**
   * Cache management
   */

  private isCached(key: string, timeout = this.cacheTimeout): boolean {
    const entry = this.cache.get(key)
    if (!entry) return false
    
    return Date.now() - entry.timestamp < timeout
  }

  private getFromCache(key: string): any {
    const entry = this.cache.get(key)
    return entry ? entry.data : null
  }

  private setCache(key: string, data: any, timeout = this.cacheTimeout): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    })

    // Clean up expired entries periodically
    if (this.cache.size > 100) {
      this.cleanupCache()
    }
  }

  private invalidateCache(keyPattern: string): void {
    for (const key of this.cache.keys()) {
      if (key.startsWith(keyPattern)) {
        this.cache.delete(key)
      }
    }
  }

  private cleanupCache(): void {
    const now = Date.now()
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.cacheTimeout) {
        this.cache.delete(key)
      }
    }
  }

  /**
   * Utility methods
   */

  clearCache(): void {
    this.cache.clear()
  }

  getCacheStats(): {
    size: number
    hitRate: number
    memoryUsage: number
  } {
    return {
      size: this.cache.size,
      hitRate: 0, // Would need to track hits/misses
      memoryUsage: JSON.stringify(Array.from(this.cache.values())).length
    }
  }
}

// Export singleton instance
export const commandManagementService = CommandManagementService.getInstance()
export default commandManagementService