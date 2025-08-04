/**
 * Task Coordination Service for LeanVibe Agent Hive
 * 
 * Provides real-time task coordination between agents including:
 * - Multi-agent task assignment and workload balancing
 * - Real-time task status synchronization
 * - Agent collaboration and dependency management
 * - Performance monitoring and optimization
 * - Sprint coordination and autonomous task distribution
 */

import { BaseService } from './base-service'
import { getAgentService } from './agent'
import {
  Task,
  TaskStatus,
  TaskPriority,
  ServiceConfig,
  EventListener,
  Subscription
} from '../types/api'
import type { Agent, AgentRole } from '../types/api'

export interface TaskAssignment {
  taskId: string
  agentId: string
  assignedAt: string
  estimatedCompletionTime?: string
  priority: TaskPriority
  dependencies: string[]
}

export interface AgentWorkload {
  agentId: string
  assignedTasks: TaskAssignment[]
  currentLoad: number
  capacity: number
  utilization: number
  averageCompletionTime: number
  efficiency: number
}

export interface TaskCoordinationMetrics {
  totalTasks: number
  activeTasks: number
  completedTasks: number
  overdueTasks: number
  averageCompletionTime: number
  teamVelocity: number
  bottlenecks: Array<{
    agentId: string
    reason: string
    impact: 'low' | 'medium' | 'high'
  }>
}

export interface CollaborationRequest {
  id: string
  taskId: string
  requestingAgentId: string
  targetAgentId: string
  type: 'review' | 'assistance' | 'handoff' | 'consultation'
  message: string
  priority: TaskPriority
  status: 'pending' | 'accepted' | 'declined' | 'completed'
  createdAt: string
  respondedAt?: string
}

export interface AutoAssignmentRule {
  id: string
  name: string
  enabled: boolean
  criteria: {
    taskType?: string[]
    priority?: TaskPriority[]
    agentRoles?: AgentRole[]
    complexity?: 'low' | 'medium' | 'high'
    keywords?: string[]
  }
  assignment: {
    strategy: 'round-robin' | 'least-loaded' | 'skill-match' | 'priority-based'
    maxTasksPerAgent?: number
    requiredSkills?: string[]
    excludeAgents?: string[]
  }
}

export class TaskCoordinationService extends BaseService {
  private agentService = getAgentService()
  private taskAssignments: Map<string, TaskAssignment> = new Map()
  private agentWorkloads: Map<string, AgentWorkload> = new Map()
  private collaborationRequests: Map<string, CollaborationRequest> = new Map()
  private autoAssignmentRules: AutoAssignmentRule[] = []
  private coordinationPolling: (() => void) | null = null

  constructor(config: Partial<ServiceConfig> = {}) {
    super({
      pollingInterval: 3000, // 3 seconds for task coordination
      cacheTimeout: 2000, // 2 second cache for coordination data
      ...config
    })
    
    this.initializeDefaultRules()
  }

  // ===== TASK ASSIGNMENT & COORDINATION =====

  /**
   * Assign a task to a specific agent
   */
  async assignTask(taskId: string, agentId: string, priority: TaskPriority = 'medium'): Promise<{
    success: boolean
    assignment: TaskAssignment | null
    message: string
  }> {
    try {
      const agent = this.agentService.getAgent(agentId)
      if (!agent) {
        throw new Error(`Agent ${agentId} not found`)
      }

      const assignment: TaskAssignment = {
        taskId,
        agentId,
        assignedAt: new Date().toISOString(),
        priority,
        dependencies: []
      }

      // In real implementation, would call backend API
      const response = await this.post<{
        success: boolean
        assignment: TaskAssignment
        message: string
      }>('/api/tasks/assign', {
        task_id: taskId,
        agent_id: agentId,
        priority
      })

      this.taskAssignments.set(taskId, assignment)
      this.updateAgentWorkload(agentId)
      
      this.emit('taskAssigned', { taskId, agentId, assignment })
      
      return {
        success: true,
        assignment,
        message: `Task assigned to ${agent.name}`
      }

    } catch (error) {
      this.emit('taskAssignmentFailed', { taskId, agentId, error })
      throw error
    }
  }

  /**
   * Auto-assign tasks based on workload balancing and agent capabilities
   */
  async autoAssignTasks(taskIds: string[]): Promise<{
    success: boolean
    assignments: Record<string, string> // taskId -> agentId
    unassigned: string[]
    message: string
  }> {
    const assignments: Record<string, string> = {}
    const unassigned: string[] = []
    
    try {
      const agents = this.agentService.getAgents()
      const activeAgents = agents.filter(agent => 
        agent.status === 'active' || agent.status === 'idle'
      )

      if (activeAgents.length === 0) {
        return {
          success: false,
          assignments: {},
          unassigned: taskIds,
          message: 'No active agents available for assignment'
        }
      }

      for (const taskId of taskIds) {
        const bestAgent = await this.findBestAgentForTask(taskId, activeAgents)
        
        if (bestAgent) {
          await this.assignTask(taskId, bestAgent.id)
          assignments[taskId] = bestAgent.id
        } else {
          unassigned.push(taskId)
        }
      }

      this.emit('autoAssignmentCompleted', {
        assignments,
        unassigned,
        totalTasks: taskIds.length,
        assignedCount: Object.keys(assignments).length
      })

      return {
        success: Object.keys(assignments).length > 0,
        assignments,
        unassigned,
        message: `Assigned ${Object.keys(assignments).length} of ${taskIds.length} tasks`
      }

    } catch (error) {
      this.emit('autoAssignmentFailed', { taskIds, error })
      throw error
    }
  }

  /**
   * Reassign task based on workload balancing
   */
  async rebalanceTaskLoad(): Promise<{
    success: boolean
    reassignments: Array<{ taskId: string; fromAgent: string; toAgent: string }>
    message: string
  }> {
    try {
      const agents = this.agentService.getAgents()
      const workloads = Array.from(this.agentWorkloads.values())
      
      // Find overloaded and underloaded agents
      const overloaded = workloads.filter(w => w.utilization > 85)
      const underloaded = workloads.filter(w => w.utilization < 50)
      
      const reassignments: Array<{ taskId: string; fromAgent: string; toAgent: string }> = []
      
      for (const overloadedAgent of overloaded) {
        const availableAgent = underloaded.find(a => 
          agents.find(agent => agent.id === a.agentId)?.role === 
          agents.find(agent => agent.id === overloadedAgent.agentId)?.role
        )
        
        if (availableAgent && overloadedAgent.assignedTasks.length > 0) {
          // Move lowest priority task
          const taskToMove = overloadedAgent.assignedTasks
            .sort((a, b) => {
              const priorityOrder = { low: 1, medium: 2, high: 3 }
              return priorityOrder[a.priority] - priorityOrder[b.priority]
            })[0]
          
          await this.assignTask(taskToMove.taskId, availableAgent.agentId, taskToMove.priority)
          
          reassignments.push({
            taskId: taskToMove.taskId,
            fromAgent: overloadedAgent.agentId,
            toAgent: availableAgent.agentId
          })
        }
      }
      
      this.emit('workloadRebalanced', { reassignments })
      
      return {
        success: reassignments.length > 0,
        reassignments,
        message: `Rebalanced ${reassignments.length} task assignments`
      }

    } catch (error) {
      this.emit('rebalanceFailed', { error })
      throw error
    }
  }

  // ===== AGENT COLLABORATION =====

  /**
   * Request collaboration between agents
   */
  async requestCollaboration(
    taskId: string,
    requestingAgentId: string,
    targetAgentId: string,
    type: CollaborationRequest['type'],
    message: string,
    priority: TaskPriority = 'medium'
  ): Promise<{ success: boolean; request: CollaborationRequest; message: string }> {
    try {
      const request: CollaborationRequest = {
        id: `collab-${Date.now()}`,
        taskId,
        requestingAgentId,
        targetAgentId,
        type,
        message,
        priority,
        status: 'pending',
        createdAt: new Date().toISOString()
      }

      this.collaborationRequests.set(request.id, request)
      
      this.emit('collaborationRequested', request)
      
      return {
        success: true,
        request,
        message: 'Collaboration request sent'
      }

    } catch (error) {
      this.emit('collaborationRequestFailed', { taskId, requestingAgentId, targetAgentId, error })
      throw error
    }
  }

  /**
   * Respond to collaboration request
   */
  async respondToCollaboration(
    requestId: string,
    response: 'accepted' | 'declined',
    responseMessage?: string
  ): Promise<{ success: boolean; message: string }> {
    try {
      const request = this.collaborationRequests.get(requestId)
      if (!request) {
        throw new Error(`Collaboration request ${requestId} not found`)
      }

      request.status = response
      request.respondedAt = new Date().toISOString()
      
      this.emit('collaborationResponded', { request, response, responseMessage })
      
      return {
        success: true,
        message: `Collaboration request ${response}`
      }

    } catch (error) {
      this.emit('collaborationResponseFailed', { requestId, response, error })
      throw error
    }
  }

  // ===== WORKLOAD MONITORING =====

  /**
   * Get current workload for all agents
   */
  getAgentWorkloads(): AgentWorkload[] {
    return Array.from(this.agentWorkloads.values())
  }

  /**
   * Get workload for specific agent
   */
  getAgentWorkload(agentId: string): AgentWorkload | null {
    return this.agentWorkloads.get(agentId) || null
  }

  /**
   * Update agent workload metrics
   */
  private updateAgentWorkload(agentId: string) {
    const assignments = Array.from(this.taskAssignments.values())
      .filter(a => a.agentId === agentId)
    
    const agent = this.agentService.getAgent(agentId)
    if (!agent) return

    const currentLoad = assignments.length
    const capacity = 8 // Default capacity: 8 concurrent tasks
    const utilization = Math.round((currentLoad / capacity) * 100)
    
    const workload: AgentWorkload = {
      agentId,
      assignedTasks: assignments,
      currentLoad,
      capacity,
      utilization,
      averageCompletionTime: agent.performance_metrics?.average_completion_time || 0,
      efficiency: agent.performance_metrics?.success_rate || 0
    }

    this.agentWorkloads.set(agentId, workload)
    this.emit('workloadUpdated', { agentId, workload })
  }

  /**
   * Get coordination metrics
   */
  async getCoordinationMetrics(): Promise<TaskCoordinationMetrics> {
    const assignments = Array.from(this.taskAssignments.values())
    const workloads = Array.from(this.agentWorkloads.values())
    
    // Calculate bottlenecks
    const bottlenecks = workloads
      .filter(w => w.utilization > 90)
      .map(w => ({
        agentId: w.agentId,
        reason: 'High utilization',
        impact: w.utilization > 95 ? 'high' as const : 'medium' as const
      }))

    const metrics: TaskCoordinationMetrics = {
      totalTasks: assignments.length,
      activeTasks: assignments.filter(a => a.priority !== 'low').length,
      completedTasks: 0, // Would be calculated from completed tasks
      overdueTasks: 0, // Would be calculated from due dates
      averageCompletionTime: workloads.reduce((sum, w) => sum + w.averageCompletionTime, 0) / workloads.length,
      teamVelocity: workloads.reduce((sum, w) => sum + w.efficiency, 0) / workloads.length,
      bottlenecks
    }

    return metrics
  }

  // ===== AUTO-ASSIGNMENT RULES =====

  /**
   * Find best agent for a task based on rules and workload
   */
  private async findBestAgentForTask(taskId: string, availableAgents: Agent[]): Promise<Agent | null> {
    // Sort agents by utilization (least loaded first)
    const agentsByLoad = availableAgents.sort((a, b) => {
      const workloadA = this.getAgentWorkload(a.id)
      const workloadB = this.getAgentWorkload(b.id)
      
      const utilizationA = workloadA?.utilization || 0
      const utilizationB = workloadB?.utilization || 0
      
      return utilizationA - utilizationB
    })

    // Return least loaded agent that's not at capacity
    const bestAgent = agentsByLoad.find(agent => {
      const workload = this.getAgentWorkload(agent.id)
      return !workload || workload.utilization < 80
    })

    return bestAgent || null
  }

  /**
   * Initialize default auto-assignment rules
   */
  private initializeDefaultRules() {
    this.autoAssignmentRules = [
      {
        id: 'backend-tasks',
        name: 'Backend Development Tasks',
        enabled: true,
        criteria: {
          agentRoles: ['backend_developer', 'fullstack_developer'],
          taskType: ['backend', 'api', 'database']
        },
        assignment: {
          strategy: 'skill-match',
          maxTasksPerAgent: 5
        }
      },
      {
        id: 'frontend-tasks',
        name: 'Frontend Development Tasks',
        enabled: true,
        criteria: {
          agentRoles: ['frontend_developer', 'fullstack_developer'],
          taskType: ['frontend', 'ui', 'ux']
        },
        assignment: {
          strategy: 'least-loaded',
          maxTasksPerAgent: 6
        }
      },
      {
        id: 'high-priority',
        name: 'High Priority Tasks',
        enabled: true,
        criteria: {
          priority: ['high']
        },
        assignment: {
          strategy: 'priority-based',
          maxTasksPerAgent: 3
        }
      }
    ]
  }

  // ===== REAL-TIME COORDINATION =====

  /**
   * Start real-time task coordination monitoring
   */
  startCoordination(): void {
    if (this.coordinationPolling) {
      this.stopCoordination()
    }

    this.coordinationPolling = this.startPolling(async () => {
      try {
        // Update workloads for all agents
        const agents = this.agentService.getAgents()
        agents.forEach(agent => this.updateAgentWorkload(agent.id))
        
        // Check for auto-assignment opportunities
        await this.checkAutoAssignmentOpportunities()
        
        // Monitor collaboration requests
        this.monitorCollaborationRequests()
        
      } catch (error) {
        // Coordination errors are handled by base class
      }
    }, this.config.pollingInterval)

    this.emit('coordinationStarted')
  }

  /**
   * Stop real-time coordination monitoring
   */
  stopCoordination(): void {
    if (this.coordinationPolling) {
      this.coordinationPolling()
      this.coordinationPolling = null
      this.emit('coordinationStopped')
    }
  }

  /**
   * Check for auto-assignment opportunities
   */
  private async checkAutoAssignmentOpportunities() {
    // This would check for unassigned tasks and apply auto-assignment rules
    // Implementation would depend on task management integration
  }

  /**
   * Monitor collaboration requests for timeouts and escalation
   */
  private monitorCollaborationRequests() {
    const now = new Date()
    const timeout = 30 * 60 * 1000 // 30 minutes

    this.collaborationRequests.forEach(request => {
      if (request.status === 'pending') {
        const age = now.getTime() - new Date(request.createdAt).getTime()
        
        if (age > timeout) {
          // Auto-decline timed out requests
          request.status = 'declined'
          request.respondedAt = now.toISOString()
          
          this.emit('collaborationTimedOut', { request })
        }
      }
    })
  }

  // ===== EVENT SUBSCRIPTIONS =====

  public onTaskAssigned(listener: EventListener<{ taskId: string; agentId: string; assignment: TaskAssignment }>): Subscription {
    return this.subscribe('taskAssigned', listener)
  }

  public onWorkloadUpdated(listener: EventListener<{ agentId: string; workload: AgentWorkload }>): Subscription {
    return this.subscribe('workloadUpdated', listener)
  }

  public onCollaborationRequested(listener: EventListener<CollaborationRequest>): Subscription {
    return this.subscribe('collaborationRequested', listener)
  }

  public onAutoAssignmentCompleted(listener: EventListener<{
    assignments: Record<string, string>
    unassigned: string[]
    totalTasks: number
    assignedCount: number
  }>): Subscription {
    return this.subscribe('autoAssignmentCompleted', listener)
  }

  public onWorkloadRebalanced(listener: EventListener<{
    reassignments: Array<{ taskId: string; fromAgent: string; toAgent: string }>
  }>): Subscription {
    return this.subscribe('workloadRebalanced', listener)
  }

  public onCoordinationStarted(listener: EventListener<void>): Subscription {
    return this.subscribe('coordinationStarted', listener)
  }

  public onCoordinationStopped(listener: EventListener<void>): Subscription {
    return this.subscribe('coordinationStopped', listener)
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.stopCoordination()
    this.taskAssignments.clear()
    this.agentWorkloads.clear()
    this.collaborationRequests.clear()
    this.autoAssignmentRules = []
    super.destroy()
  }
}

// Singleton instance
let taskCoordinationService: TaskCoordinationService | null = null

export function getTaskCoordinationService(config?: Partial<ServiceConfig>): TaskCoordinationService {
  if (!taskCoordinationService) {
    taskCoordinationService = new TaskCoordinationService(config)
  }
  return taskCoordinationService
}

export function resetTaskCoordinationService(): void {
  if (taskCoordinationService) {
    taskCoordinationService.destroy()
    taskCoordinationService = null
  }
}