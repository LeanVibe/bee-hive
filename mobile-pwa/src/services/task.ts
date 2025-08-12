/**
 * Task Service for LeanVibe Agent Hive
 * 
 * Provides comprehensive task management for Kanban board including:
 * - CRUD operations for tasks
 * - Real-time task status updates
 * - Task assignment and lifecycle management
 * - Kanban board data structure
 * - Task filtering and search
 * - Offline caching with sync indicators
 */

import { BaseService } from './base-service';
import {
  TaskStatus,
  TaskPriority,
  TaskType
} from '../types/api';
import type {
  Task,
  TaskCreate,
  TaskUpdate,
  TaskListResponse,
  AgentRole,
  ServiceConfig,
  Subscription,
  EventListener
} from '../types/api';
import { OfflineService } from './offline';

export interface KanbanBoard {
  columns: KanbanColumn[];
  totalTasks: number;
  lastUpdated: string;
}

export interface KanbanColumn {
  id: TaskStatus;
  title: string;
  tasks: Task[];
  count: number;
  color: string;
}

export interface TaskFilters {
  status?: TaskStatus[];
  priority?: TaskPriority[];
  taskType?: TaskType[];
  assignedAgentId?: string;
  searchQuery?: string;
  dateRange?: {
    start: string;
    end: string;
  };
  sprintId?: string;
  labels?: string[];
  workloadBalance?: boolean;
}

export interface SprintPlan {
  id: string;
  name: string;
  description: string;
  startDate: string;
  endDate: string;
  status: 'planning' | 'active' | 'completed' | 'cancelled';
  tasks: Task[];
  velocity: {
    planned: number;
    completed: number;
    capacity: number;
  };
  teamComposition: string[];
  burndownData: BurndownPoint[];
}

export interface BurndownPoint {
  date: string;
  remainingWork: number;
  idealLine: number;
  completedWork: number;
}

export interface TaskTemplate {
  id: string;
  name: string;
  description: string;
  defaultTaskType: TaskType;
  defaultPriority: TaskPriority;
  estimatedEffort: number;
  requiredSkills: string[];
  checklistItems: string[];
  associatedRoles: AgentRole[];
}

export interface AgentWorkload {
  agentId: string;
  agentName: string;
  role: AgentRole;
  currentLoad: number;
  capacity: number;
  utilization: number;
  assignedTasks: Task[];
  capabilities: string[];
  availability: 'available' | 'busy' | 'overloaded';
}

export interface TaskAssignmentRecommendation {
  agentId: string;
  agentName: string;
  role: AgentRole;
  score: number;
  reasoning: string[];
  estimatedCompletionTime: number;
  currentWorkload: number;
}

export interface TaskStatistics {
  total: number;
  byStatus: Record<TaskStatus, number>;
  byPriority: Record<TaskPriority, number>;
  byType: Record<TaskType, number>;
  completionRate: number;
  averageCompletionTime: number;
  overdueTasks: number;
}

export interface TaskAssignmentResult {
  success: boolean;
  task: Task;
  message: string;
}

export class TaskService extends BaseService {
  private pollingStopFn: (() => void) | null = null;
  private tasks: Map<string, Task> = new Map();
  private lastSync: string | null = null;
  private syncInProgress = false;
  private currentSprint: SprintPlan | null = null;
  private templates: Map<string, TaskTemplate> = new Map();
  private workloadCache: Map<string, AgentWorkload> = new Map();
  private velocityHistory: Map<string, number[]> = new Map();
  private analyticsData: TaskAnalytics | null = null;

  // Kanban column configuration
  private readonly kanbanColumns: Omit<KanbanColumn, 'tasks' | 'count'>[] = [
    { id: TaskStatus.PENDING, title: 'Backlog', color: '#6b7280' },
    { id: TaskStatus.ASSIGNED, title: 'Assigned', color: '#3b82f6' },
    { id: TaskStatus.IN_PROGRESS, title: 'In Progress', color: '#f59e0b' },
    { id: TaskStatus.COMPLETED, title: 'Completed', color: '#10b981' },
    { id: TaskStatus.FAILED, title: 'Failed', color: '#ef4444' },
    { id: TaskStatus.CANCELLED, title: 'Cancelled', color: '#6b7280' }
  ];

  constructor(config: Partial<ServiceConfig> = {}) {
    super({
      pollingInterval: 2000, // 2 seconds for task updates
      cacheTimeout: 10000, // 10 second cache for tasks
      ...config
    });
  }

  // ===== TASK CRUD OPERATIONS =====

  /**
   * Create a new task
   */
  async createTask(taskData: TaskCreate): Promise<Task> {
    try {
      const task = await this.post<Task>('/api/v1/tasks/', taskData);
      
      // Update local cache
      this.tasks.set(task.id, task);
      this.clearCache('tasks'); // Invalidate task list cache
      
      this.emit('taskCreated', task);
      return task;

    } catch (error) {
      // Network/offline fallback: queue and optimistic update
      try {
        if (!navigator.onLine || (error as any)?.code === 'NETWORK_ERROR' || (error as any)?.code === 'TIMEOUT') {
          const optimistic: Task = {
            ...(taskData as any),
            id: `tmp_${Date.now()}`,
            status: taskData.status || 'pending',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          } as Task
          this.tasks.set(optimistic.id, optimistic);
          this.emit('taskCreated', optimistic);
          await OfflineService.getInstance().queueSync('create', 'tasks', taskData);
          return optimistic;
        }
      } catch {}
      this.emit('taskCreateFailed', { taskData, error });
      throw error;
    }
  }

  /**
   * Get all tasks with optional filtering
   */
  async getTasks(
    filters: TaskFilters = {},
    limit = 50,
    offset = 0,
    fromCache = true
  ): Promise<TaskListResponse> {
    const params: Record<string, any> = {
      limit,
      offset
    };

    // Apply filters
    if (filters.status && filters.status.length > 0) {
      params.status = filters.status[0]; // API expects single status
    }
    if (filters.priority && filters.priority.length > 0) {
      params.priority = filters.priority[0]; // API expects single priority
    }
    if (filters.taskType && filters.taskType.length > 0) {
      params.task_type = filters.taskType[0]; // API expects single task_type
    }
    if (filters.assignedAgentId) {
      params.assigned_agent_id = filters.assignedAgentId;
    }

    const queryString = this.buildQueryString(params);
    const cacheKey = fromCache ? `tasks${queryString}` : undefined;
    
    try {
      const response = await this.get<TaskListResponse>(
        `/api/v1/tasks/${queryString}`,
        {},
        cacheKey
      );

      // Update local task cache
      response.tasks.forEach(task => {
        this.tasks.set(task.id, task);
      });

      // Apply client-side filtering for multi-value filters
      let filteredTasks = response.tasks;
      
      if (filters.searchQuery) {
        const query = filters.searchQuery.toLowerCase();
        filteredTasks = filteredTasks.filter(task =>
          task.title.toLowerCase().includes(query) ||
          task.description.toLowerCase().includes(query)
        );
      }

      if (filters.dateRange) {
        const start = new Date(filters.dateRange.start);
        const end = new Date(filters.dateRange.end);
        filteredTasks = filteredTasks.filter(task => {
          const created = new Date(task.created_at);
          return created >= start && created <= end;
        });
      }

      const filteredResponse: TaskListResponse = {
        ...response,
        tasks: filteredTasks,
        total: filteredTasks.length
      };

      this.emit('tasksLoaded', filteredResponse);
      return filteredResponse;

    } catch (error) {
      this.emit('tasksLoadFailed', { filters, error });
      throw error;
    }
  }

  /**
   * Get a specific task by ID
   */
  async getTask(taskId: string, fromCache = true): Promise<Task> {
    // Check local cache first
    if (fromCache) {
      const cached = this.tasks.get(taskId);
      if (cached) {
        return cached;
      }
    }

    try {
      const task = await this.get<Task>(`/api/v1/tasks/${taskId}`);
      
      // Update local cache
      this.tasks.set(task.id, task);
      
      return task;

    } catch (error) {
      this.emit('taskLoadFailed', { taskId, error });
      throw error;
    }
  }

  /**
   * Update an existing task
   */
  async updateTask(taskId: string, updates: TaskUpdate): Promise<Task> {
    try {
      const task = await this.put<Task>(`/api/v1/tasks/${taskId}`, updates);
      
      // Update local cache
      this.tasks.set(task.id, task);
      this.clearCache('tasks'); // Invalidate task list cache
      
      this.emit('taskUpdated', task);
      return task;

    } catch (error) {
      // Offline/optimistic path
      try {
        if (!navigator.onLine || (error as any)?.code === 'NETWORK_ERROR' || (error as any)?.code === 'TIMEOUT') {
          const local = this.tasks.get(taskId) as Task | undefined;
          const optimistic = { ...(local as any), ...updates, id: taskId } as Task;
          this.tasks.set(taskId, optimistic);
          this.emit('taskUpdated', optimistic);
          await OfflineService.getInstance().queueSync('update', 'tasks', { id: taskId, ...updates });
          return optimistic;
        }
      } catch {}
      this.emit('taskUpdateFailed', { taskId, updates, error });
      throw error;
    }
  }

  /**
   * Delete (cancel) a task
   */
  async deleteTask(taskId: string): Promise<void> {
    try {
      await this.delete(`/api/v1/tasks/${taskId}`);
      
      // Update local cache
      const task = this.tasks.get(taskId);
      if (task) {
        task.status = TaskStatus.CANCELLED;
        task.completed_at = new Date().toISOString();
        this.tasks.set(taskId, task);
      }
      
      this.clearCache('tasks'); // Invalidate task list cache
      
      this.emit('taskDeleted', taskId);

    } catch (error) {
      try {
        if (!navigator.onLine || (error as any)?.code === 'NETWORK_ERROR' || (error as any)?.code === 'TIMEOUT') {
          await OfflineService.getInstance().queueSync('delete', 'tasks', { id: taskId });
          this.tasks.delete(taskId);
          this.emit('taskDeleted', taskId);
          return;
        }
      } catch {}
      this.emit('taskDeleteFailed', { taskId, error });
      throw error;
    }
  }

  // ===== TASK ASSIGNMENT & LIFECYCLE =====

  /**
   * Assign a task to an agent
   */
  async assignTask(taskId: string, agentId: string): Promise<TaskAssignmentResult> {
    try {
      const task = await this.post<Task>(`/api/v1/tasks/${taskId}/assign/${agentId}`);
      
      // Update local cache
      this.tasks.set(task.id, task);
      this.clearCache('tasks');
      
      const result: TaskAssignmentResult = {
        success: true,
        task,
        message: `Task assigned to agent ${agentId}`
      };
      
      this.emit('taskAssigned', result);
      return result;

    } catch (error) {
      const result: TaskAssignmentResult = {
        success: false,
        task: this.tasks.get(taskId)!,
        message: error instanceof Error ? error.message : 'Assignment failed'
      };
      
      this.emit('taskAssignmentFailed', { taskId, agentId, error });
      throw error;
    }
  }

  /**
   * Start task execution
   */
  async startTask(taskId: string): Promise<Task> {
    try {
      const task = await this.post<Task>(`/api/v1/tasks/${taskId}/start`);
      
      // Update local cache
      this.tasks.set(task.id, task);
      this.clearCache('tasks');
      
      this.emit('taskStarted', task);
      return task;

    } catch (error) {
      this.emit('taskStartFailed', { taskId, error });
      throw error;
    }
  }

  /**
   * Complete a task with result data
   */
  async completeTask(taskId: string, result: Record<string, any>): Promise<Task> {
    try {
      const task = await this.post<Task>(`/api/v1/tasks/${taskId}/complete`, { result });
      
      // Update local cache
      this.tasks.set(task.id, task);
      this.clearCache('tasks');
      
      this.emit('taskCompleted', task);
      return task;

    } catch (error) {
      this.emit('taskCompletionFailed', { taskId, result, error });
      throw error;
    }
  }

  /**
   * Mark a task as failed
   */
  async failTask(taskId: string, errorMessage: string, canRetry = true): Promise<Task> {
    try {
      const task = await this.post<Task>(`/api/v1/tasks/${taskId}/fail`, {
        error_message: errorMessage,
        can_retry: canRetry
      });
      
      // Update local cache
      this.tasks.set(task.id, task);
      this.clearCache('tasks');
      
      this.emit('taskFailed', task);
      return task;

    } catch (error) {
      this.emit('taskFailureFailed', { taskId, errorMessage, error });
      throw error;
    }
  }

  // ===== KANBAN BOARD OPERATIONS =====

  /**
   * Get Kanban board data structure
   */
  async getKanbanBoard(filters: TaskFilters = {}): Promise<KanbanBoard> {
    try {
      const tasksResponse = await this.getTasks(filters, 1000); // Get more tasks for board
      
      const columns: KanbanColumn[] = this.kanbanColumns.map(columnConfig => {
        const columnTasks = tasksResponse.tasks.filter(task => task.status === columnConfig.id);
        
        return {
          ...columnConfig,
          tasks: columnTasks.sort((a, b) => {
            // Sort by priority, then by creation date
            const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
            const aPriority = priorityOrder[a.priority];
            const bPriority = priorityOrder[b.priority];
            
            if (aPriority !== bPriority) {
              return bPriority - aPriority;
            }
            
            return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
          }),
          count: columnTasks.length
        };
      });

      const board: KanbanBoard = {
        columns,
        totalTasks: tasksResponse.total,
        lastUpdated: new Date().toISOString()
      };

      this.emit('kanbanBoardUpdated', board);
      return board;

    } catch (error) {
      this.emit('kanbanBoardLoadFailed', { filters, error });
      throw error;
    }
  }

  /**
   * Move task to different status (for Kanban drag & drop)
   */
  async moveTask(taskId: string, newStatus: TaskStatus): Promise<Task> {
    return this.updateTask(taskId, { status: newStatus });
  }

  // ===== AGENT TASK MANAGEMENT =====

  /**
   * Get tasks for a specific agent
   */
  async getAgentTasks(
    agentId: string,
    status?: TaskStatus,
    limit = 50,
    offset = 0
  ): Promise<TaskListResponse> {
    const params: Record<string, any> = {
      limit,
      offset
    };

    if (status) {
      params.status = status;
    }

    const queryString = this.buildQueryString(params);
    
    try {
      return await this.get<TaskListResponse>(`/api/v1/tasks/agent/${agentId}${queryString}`);
    } catch (error) {
      this.emit('agentTasksLoadFailed', { agentId, error });
      throw error;
    }
  }

  // ===== SPRINT PLANNING & BACKLOG MANAGEMENT =====

  /**
   * Create a new sprint plan
   */
  async createSprint(sprintData: {
    name: string;
    description: string;
    startDate: string;
    endDate: string;
    teamComposition: string[];
  }): Promise<SprintPlan> {
    try {
      const sprint: SprintPlan = {
        id: `sprint_${Date.now()}`,
        ...sprintData,
        status: 'planning',
        tasks: [],
        velocity: {
          planned: 0,
          completed: 0,
          capacity: sprintData.teamComposition.length * 40 // 40 hours per agent
        },
        burndownData: []
      };

      // In real implementation, would call backend API
      // For now, storing in cache
      this.setInCache(`sprint_${sprint.id}`, sprint);
      
      this.emit('sprintCreated', sprint);
      return sprint;

    } catch (error) {
      this.emit('sprintCreateFailed', { sprintData, error });
      throw error;
    }
  }

  /**
   * Add tasks to sprint backlog
   */
  async addTasksToSprint(sprintId: string, taskIds: string[]): Promise<SprintPlan> {
    try {
      const sprint = this.getFromCache(`sprint_${sprintId}`) as SprintPlan;
      if (!sprint) {
        throw new Error(`Sprint ${sprintId} not found`);
      }

      const tasks = await Promise.all(
        taskIds.map(id => this.getTask(id))
      );

      sprint.tasks.push(...tasks);
      sprint.velocity.planned = tasks.reduce((sum, task) => sum + (task.estimated_effort || 0), 0);
      
      this.setInCache(`sprint_${sprintId}`, sprint);
      this.emit('sprintUpdated', sprint);
      
      return sprint;

    } catch (error) {
      this.emit('sprintUpdateFailed', { sprintId, taskIds, error });
      throw error;
    }
  }

  /**
   * Get sprint burndown data
   */
  async getSprintBurndown(sprintId: string): Promise<BurndownPoint[]> {
    try {
      const sprint = this.getFromCache(`sprint_${sprintId}`) as SprintPlan;
      if (!sprint) {
        throw new Error(`Sprint ${sprintId} not found`);
      }

      // Calculate burndown based on task completion
      const startDate = new Date(sprint.startDate);
      const endDate = new Date(sprint.endDate);
      const totalWork = sprint.velocity.planned;
      const sprintDays = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
      
      const burndownData: BurndownPoint[] = [];
      
      for (let day = 0; day <= sprintDays; day++) {
        const currentDate = new Date(startDate);
        currentDate.setDate(startDate.getDate() + day);
        
        // Calculate completed work up to this date
        const completedWork = sprint.tasks
          .filter(task => task.status === TaskStatus.COMPLETED && 
                         new Date(task.completed_at || '') <= currentDate)
          .reduce((sum, task) => sum + (task.actual_effort || task.estimated_effort || 0), 0);
        
        const remainingWork = totalWork - completedWork;
        const idealLine = totalWork - (totalWork / sprintDays) * day;
        
        burndownData.push({
          date: currentDate.toISOString().split('T')[0],
          remainingWork: Math.max(0, remainingWork),
          idealLine: Math.max(0, idealLine),
          completedWork
        });
      }
      
      return burndownData;

    } catch (error) {
      this.emit('burndownLoadFailed', { sprintId, error });
      throw error;
    }
  }

  // ===== TASK TEMPLATES =====

  /**
   * Create task from template
   */
  async createTaskFromTemplate(templateId: string, customData: Partial<TaskCreate>): Promise<Task> {
    const template = this.getTaskTemplate(templateId);
    if (!template) {
      throw new Error(`Template ${templateId} not found`);
    }

    const taskData: TaskCreate = {
      title: template.name,
      description: template.description,
      task_type: template.defaultTaskType,
      priority: template.defaultPriority,
      estimated_effort: template.estimatedEffort,
      required_skills: template.requiredSkills,
      checklist: template.checklistItems,
      ...customData
    };

    return this.createTask(taskData);
  }

  /**
   * Get available task templates
   */
  getTaskTemplates(): TaskTemplate[] {
    return [
      {
        id: 'bug_fix',
        name: 'Bug Fix',
        description: 'Standard bug fix template',
        defaultTaskType: TaskType.BUG_FIX,
        defaultPriority: TaskPriority.HIGH,
        estimatedEffort: 2,
        requiredSkills: ['debugging', 'testing'],
        checklistItems: [
          'Reproduce the bug',
          'Identify root cause',
          'Implement fix',
          'Write test cases',
          'Verify fix'
        ],
        associatedRoles: [AgentRole.BACKEND_DEVELOPER, AgentRole.QA_ENGINEER]
      },
      {
        id: 'feature_development',
        name: 'Feature Development',
        description: 'New feature development template',
        defaultTaskType: TaskType.FEATURE,
        defaultPriority: TaskPriority.MEDIUM,
        estimatedEffort: 8,
        requiredSkills: ['development', 'design', 'testing'],
        checklistItems: [
          'Review requirements',
          'Design implementation',
          'Develop feature',
          'Write tests',
          'Documentation',
          'Code review'
        ],
        associatedRoles: [AgentRole.PRODUCT_MANAGER, AgentRole.BACKEND_DEVELOPER, AgentRole.FRONTEND_DEVELOPER]
      },
      {
        id: 'code_review',
        name: 'Code Review',
        description: 'Code review template',
        defaultTaskType: TaskType.REVIEW,
        defaultPriority: TaskPriority.MEDIUM,
        estimatedEffort: 1,
        requiredSkills: ['code_review', 'best_practices'],
        checklistItems: [
          'Review code quality',
          'Check test coverage',
          'Verify requirements',
          'Provide feedback'
        ],
        associatedRoles: [AgentRole.ARCHITECT, AgentRole.QA_ENGINEER]
      }
    ];
  }

  /**
   * Get specific task template
   */
  getTaskTemplate(templateId: string): TaskTemplate | null {
    return this.getTaskTemplates().find(t => t.id === templateId) || null;
  }

  // ===== INTELLIGENT TASK ASSIGNMENT =====

  /**
   * Get agent workload information
   */
  async getAgentWorkloads(agentIds?: string[]): Promise<AgentWorkload[]> {
    try {
      const tasks = await this.getTasks({}, 1000);
      const activeTasks = tasks.tasks.filter(task => 
        task.status === TaskStatus.ASSIGNED || task.status === TaskStatus.IN_PROGRESS
      );

      // Get agent information from agent service
      const { getAgentService } = await import('./agent');
      const agentService = getAgentService();
      const agents = agentService.getAgents();
      
      const workloads: AgentWorkload[] = agents
        .filter(agent => !agentIds || agentIds.includes(agent.id))
        .map(agent => {
          const agentTasks = activeTasks.filter(task => task.assigned_agent_id === agent.id);
          const currentLoad = agentTasks.reduce((sum, task) => 
            sum + (task.estimated_effort || 0) - (task.actual_effort || 0), 0
          );
          const capacity = 40; // 40 hours per week capacity
          const utilization = capacity > 0 ? (currentLoad / capacity) * 100 : 0;
          
          return {
            agentId: agent.id,
            agentName: agent.name || `Agent ${agent.id}`,
            role: agent.role,
            currentLoad,
            capacity,
            utilization,
            assignedTasks: agentTasks,
            capabilities: agent.capabilities || [],
            availability: utilization > 90 ? 'overloaded' : utilization > 70 ? 'busy' : 'available'
          };
        });

      return workloads;

    } catch (error) {
      this.emit('workloadLoadFailed', { agentIds, error });
      throw error;
    }
  }

  /**
   * Get task assignment recommendations
   */
  async getTaskAssignmentRecommendations(taskId: string): Promise<TaskAssignmentRecommendation[]> {
    try {
      const task = await this.getTask(taskId);
      const workloads = await this.getAgentWorkloads();
      
      const recommendations: TaskAssignmentRecommendation[] = workloads
        .map(workload => {
          let score = 0;
          const reasoning: string[] = [];
          
          // Score based on availability (40%)
          const availabilityScore = workload.availability === 'available' ? 40 : 
                                   workload.availability === 'busy' ? 20 : 0;
          score += availabilityScore;
          reasoning.push(`Availability: ${workload.availability} (${availabilityScore} points)`);
          
          // Score based on role match (35%)
          const taskTemplate = this.getTaskTemplates().find(t => t.defaultTaskType === task.task_type);
          const roleMatch = taskTemplate?.associatedRoles.includes(workload.role) ? 35 : 0;
          score += roleMatch;
          reasoning.push(`Role match: ${roleMatch > 0 ? 'Yes' : 'No'} (${roleMatch} points)`);
          
          // Score based on capabilities (25%)
          const requiredSkills = task.required_skills || taskTemplate?.requiredSkills || [];
          const skillMatches = requiredSkills.filter(skill => 
            workload.capabilities.some(cap => cap.toLowerCase().includes(skill.toLowerCase()))
          ).length;
          const capabilityScore = requiredSkills.length > 0 ? 
            (skillMatches / requiredSkills.length) * 25 : 15;
          score += capabilityScore;
          reasoning.push(`Skill match: ${skillMatches}/${requiredSkills.length} (${Math.round(capabilityScore)} points)`);
          
          const estimatedCompletionTime = task.estimated_effort || 2;
          
          return {
            agentId: workload.agentId,
            agentName: workload.agentName,
            role: workload.role,
            score,
            reasoning,
            estimatedCompletionTime,
            currentWorkload: workload.currentLoad
          };
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, 5); // Top 5 recommendations
      
      return recommendations;

    } catch (error) {
      this.emit('assignmentRecommendationsFailed', { taskId, error });
      throw error;
    }
  }

  /**
   * Auto-assign tasks based on intelligent matching
   */
  async autoAssignTasks(taskIds: string[], balanceWorkload = true): Promise<{
    assignments: Array<{ taskId: string; agentId: string; confidence: number }>;
    unassigned: string[];
    summary: {
      total: number;
      assigned: number;
      unassigned: number;
    };
  }> {
    const assignments: Array<{ taskId: string; agentId: string; confidence: number }> = [];
    const unassigned: string[] = [];
    
    for (const taskId of taskIds) {
      try {
        const recommendations = await this.getTaskAssignmentRecommendations(taskId);
        
        if (recommendations.length > 0) {
          let selectedAgent = recommendations[0];
          
          // If workload balancing is enabled, prefer less loaded agents
          if (balanceWorkload && recommendations.length > 1) {
            const availableAgents = recommendations.filter(r => r.currentWorkload < 30);
            if (availableAgents.length > 0) {
              selectedAgent = availableAgents[0];
            }
          }
          
          await this.assignTask(taskId, selectedAgent.agentId);
          assignments.push({
            taskId,
            agentId: selectedAgent.agentId,
            confidence: selectedAgent.score
          });
        } else {
          unassigned.push(taskId);
        }
      } catch (error) {
        unassigned.push(taskId);
      }
    }
    
    const result = {
      assignments,
      unassigned,
      summary: {
        total: taskIds.length,
        assigned: assignments.length,
        unassigned: unassigned.length
      }
    };
    
    this.emit('autoAssignmentCompleted', result);
    return result;
  }

  // ===== BULK OPERATIONS =====

  /**
   * Perform bulk operations on multiple tasks
   */
  async performBulkOperation(
    taskIds: string[],
    operation: 'assign' | 'move' | 'delete' | 'prioritize' | 'add_label',
    options: {
      agentId?: string;
      newStatus?: TaskStatus;
      priority?: TaskPriority;
      label?: string;
    } = {}
  ): Promise<{
    success: boolean;
    results: Record<string, { success: boolean; message: string }>;
    summary: {
      total: number;
      successful: number;
      failed: number;
    };
  }> {
    const results: Record<string, { success: boolean; message: string }> = {};
    let successful = 0;
    let failed = 0;
    
    for (const taskId of taskIds) {
      try {
        switch (operation) {
          case 'assign':
            if (options.agentId) {
              await this.assignTask(taskId, options.agentId);
              results[taskId] = { success: true, message: 'Task assigned successfully' };
            } else {
              throw new Error('Agent ID required for assignment');
            }
            break;
            
          case 'move':
            if (options.newStatus) {
              await this.moveTask(taskId, options.newStatus);
              results[taskId] = { success: true, message: 'Task moved successfully' };
            } else {
              throw new Error('New status required for move operation');
            }
            break;
            
          case 'delete':
            await this.deleteTask(taskId);
            results[taskId] = { success: true, message: 'Task deleted successfully' };
            break;
            
          case 'prioritize':
            if (options.priority) {
              await this.updateTask(taskId, { priority: options.priority });
              results[taskId] = { success: true, message: 'Task priority updated' };
            } else {
              throw new Error('Priority required for prioritize operation');
            }
            break;
            
          case 'add_label':
            if (options.label) {
              const task = await this.getTask(taskId);
              const labels = task.labels || [];
              if (!labels.includes(options.label)) {
                labels.push(options.label);
                await this.updateTask(taskId, { labels });
              }
              results[taskId] = { success: true, message: 'Label added successfully' };
            } else {
              throw new Error('Label required for add_label operation');
            }
            break;
            
          default:
            throw new Error(`Unknown operation: ${operation}`);
        }
        successful++;
      } catch (error) {
        results[taskId] = {
          success: false,
          message: error instanceof Error ? error.message : 'Operation failed'
        };
        failed++;
      }
    }
    
    const result = {
      success: failed === 0,
      results,
      summary: {
        total: taskIds.length,
        successful,
        failed
      }
    };
    
    this.emit('bulkOperationCompleted', { operation, taskIds, results: result });
    return result;
  }

  // ===== TASK STATISTICS & ANALYTICS =====

  /**
   * Get task statistics
   */
  getTaskStatistics(): TaskStatistics {
    const tasks = Array.from(this.tasks.values());
    
    const stats: TaskStatistics = {
      total: tasks.length,
      byStatus: {} as Record<TaskStatus, number>,
      byPriority: {} as Record<TaskPriority, number>,
      byType: {} as Record<TaskType, number>,
      completionRate: 0,
      averageCompletionTime: 0,
      overdueTasks: 0
    };

    // Initialize counters
    Object.values(TaskStatus).forEach(status => {
      stats.byStatus[status] = 0;
    });
    Object.values(TaskPriority).forEach(priority => {
      stats.byPriority[priority] = 0;
    });
    Object.values(TaskType).forEach(type => {
      stats.byType[type] = 0;
    });

    // Calculate statistics
    let totalCompletionTime = 0;
    let completedTasksWithTime = 0;
    
    tasks.forEach(task => {
      stats.byStatus[task.status]++;
      stats.byPriority[task.priority]++;
      stats.byType[task.task_type]++;

      // Calculate completion time for completed tasks
      if (task.actual_effort) {
        totalCompletionTime += task.actual_effort;
        completedTasksWithTime++;
      }

      // Count overdue tasks (simplified logic)
      if (task.status === TaskStatus.IN_PROGRESS && task.estimated_effort) {
        const startTime = task.started_at ? new Date(task.started_at).getTime() : 0;
        const currentTime = Date.now();
        const elapsedMinutes = (currentTime - startTime) / (1000 * 60);
        
        if (elapsedMinutes > task.estimated_effort * 1.5) { // 50% overtime threshold
          stats.overdueTasks++;
        }
      }
    });

    // Calculate completion rate
    const completedTasks = stats.byStatus[TaskStatus.COMPLETED];
    const totalFinishedTasks = completedTasks + stats.byStatus[TaskStatus.FAILED] + stats.byStatus[TaskStatus.CANCELLED];
    stats.completionRate = totalFinishedTasks > 0 ? (completedTasks / totalFinishedTasks) * 100 : 0;

    // Calculate average completion time
    stats.averageCompletionTime = completedTasksWithTime > 0 ? totalCompletionTime / completedTasksWithTime : 0;

    return stats;
  }

  // ===== REAL-TIME MONITORING =====

  /**
   * Start real-time task monitoring
   */
  startMonitoring(): void {
    if (this.pollingStopFn) {
      this.stopMonitoring();
    }

    this.pollingStopFn = this.startPolling(async () => {
      try {
        // Refresh tasks without cache
        await this.getTasks({}, 100, 0, false);
      } catch (error) {
        // Polling errors are handled by base class
      }
    }, this.config.pollingInterval);

    this.emit('monitoringStarted');
  }

  /**
   * Stop real-time task monitoring
   */
  stopMonitoring(): void {
    if (this.pollingStopFn) {
      this.pollingStopFn();
      this.pollingStopFn = null;
      this.emit('monitoringStopped');
    }
  }

  /**
   * Check if monitoring is active
   */
  isMonitoring(): boolean {
    return this.pollingStopFn !== null;
  }

  /**
   * Get velocity tracking data
   */
  async getVelocityTracking(timeframeDays = 30): Promise<{
    dailyCompletion: Array<{ date: string; completed: number; planned: number }>;
    weeklyVelocity: Array<{ week: string; storyPoints: number; tasksCompleted: number }>;
    trends: {
      completionTrend: 'improving' | 'declining' | 'stable';
      velocityTrend: 'increasing' | 'decreasing' | 'stable';
      predictedCapacity: number;
    };
  }> {
    try {
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(endDate.getDate() - timeframeDays);
      
      const tasksResponse = await this.getTasks({
        dateRange: {
          start: startDate.toISOString(),
          end: endDate.toISOString()
        }
      }, 1000);
      
      const tasks = tasksResponse.tasks;
      const dailyCompletion: Array<{ date: string; completed: number; planned: number }> = [];
      const weeklyData: Map<string, { completed: number; storyPoints: number }> = new Map();
      
      // Generate daily completion data
      for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
        const dateStr = d.toISOString().split('T')[0];
        const completedTasks = tasks.filter(task => 
          task.status === TaskStatus.COMPLETED &&
          task.completed_at &&
          task.completed_at.startsWith(dateStr)
        );
        
        const plannedTasks = tasks.filter(task => 
          task.created_at.startsWith(dateStr)
        );
        
        dailyCompletion.push({
          date: dateStr,
          completed: completedTasks.length,
          planned: plannedTasks.length
        });
        
        // Aggregate weekly data
        const weekKey = `${d.getFullYear()}-W${Math.ceil(d.getDate() / 7)}`;
        const weekData = weeklyData.get(weekKey) || { completed: 0, storyPoints: 0 };
        weekData.completed += completedTasks.length;
        weekData.storyPoints += completedTasks.reduce((sum, task) => 
          sum + (task.estimated_effort || 1), 0
        );
        weeklyData.set(weekKey, weekData);
      }
      
      const weeklyVelocity = Array.from(weeklyData.entries()).map(([week, data]) => ({
        week,
        storyPoints: data.storyPoints,
        tasksCompleted: data.completed
      }));
      
      // Calculate trends
      const recentWeeks = weeklyVelocity.slice(-4);
      const olderWeeks = weeklyVelocity.slice(-8, -4);
      
      const avgRecentVelocity = recentWeeks.reduce((sum, week) => sum + week.storyPoints, 0) / Math.max(recentWeeks.length, 1);
      const avgOlderVelocity = olderWeeks.reduce((sum, week) => sum + week.storyPoints, 0) / Math.max(olderWeeks.length, 1);
      
      const velocityTrend = avgRecentVelocity > avgOlderVelocity * 1.1 ? 'increasing' :
                           avgRecentVelocity < avgOlderVelocity * 0.9 ? 'decreasing' : 'stable';
      
      const avgRecentCompletion = recentWeeks.reduce((sum, week) => sum + week.tasksCompleted, 0) / Math.max(recentWeeks.length, 1);
      const avgOlderCompletion = olderWeeks.reduce((sum, week) => sum + week.tasksCompleted, 0) / Math.max(olderWeeks.length, 1);
      
      const completionTrend = avgRecentCompletion > avgOlderCompletion * 1.1 ? 'improving' :
                             avgRecentCompletion < avgOlderCompletion * 0.9 ? 'declining' : 'stable';
      
      return {
        dailyCompletion,
        weeklyVelocity,
        trends: {
          completionTrend,
          velocityTrend,
          predictedCapacity: avgRecentVelocity
        }
      };
      
    } catch (error) {
      this.emit('velocityTrackingFailed', { timeframeDays, error });
      throw error;
    }
  }

  /**
   * Identify bottlenecks in the development process
   */
  async identifyBottlenecks(): Promise<{
    statusBottlenecks: Array<{
      status: TaskStatus;
      averageTime: number;
      taskCount: number;
      impact: 'high' | 'medium' | 'low';
    }>;
    agentBottlenecks: Array<{
      agentId: string;
      agentName: string;
      overloadFactor: number;
      blockedTasks: number;
      suggestions: string[];
    }>;
    systemBottlenecks: {
      totalBlockedTasks: number;
      averageResolutionTime: number;
      criticalPath: string[];
    };
  }> {
    try {
      const tasksResponse = await this.getTasks({}, 1000);
      const tasks = tasksResponse.tasks;
      const workloads = await this.getAgentWorkloads();
      
      // Analyze status bottlenecks
      const statusTimes: Map<TaskStatus, number[]> = new Map();
      
      tasks.forEach(task => {
        if (task.status === TaskStatus.COMPLETED && task.started_at && task.completed_at) {
          const startTime = new Date(task.started_at).getTime();
          const endTime = new Date(task.completed_at).getTime();
          const duration = (endTime - startTime) / (1000 * 60 * 60); // hours
          
          const statusKey = task.status;
          if (!statusTimes.has(statusKey)) {
            statusTimes.set(statusKey, []);
          }
          statusTimes.get(statusKey)!.push(duration);
        }
      });
      
      const statusBottlenecks = Array.from(statusTimes.entries()).map(([status, times]) => {
        const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length;
        const impact = averageTime > 48 ? 'high' : averageTime > 24 ? 'medium' : 'low';
        
        return {
          status,
          averageTime,
          taskCount: times.length,
          impact
        };
      }).sort((a, b) => b.averageTime - a.averageTime);
      
      // Analyze agent bottlenecks
      const agentBottlenecks = workloads
        .filter(workload => workload.availability === 'overloaded' || workload.utilization > 80)
        .map(workload => {
          const suggestions: string[] = [];
          
          if (workload.utilization > 100) {
            suggestions.push('Redistribute tasks to other agents');
          }
          if (workload.assignedTasks.length > 5) {
            suggestions.push('Limit concurrent task assignments');
          }
          if (workload.assignedTasks.some(task => task.priority === TaskPriority.CRITICAL)) {
            suggestions.push('Prioritize critical tasks first');
          }
          
          return {
            agentId: workload.agentId,
            agentName: workload.agentName,
            overloadFactor: workload.utilization / 100,
            blockedTasks: workload.assignedTasks.filter(task => 
              task.status === TaskStatus.ASSIGNED && 
              new Date(task.created_at).getTime() < Date.now() - (24 * 60 * 60 * 1000)
            ).length,
            suggestions
          };
        });
      
      // System-level bottlenecks
      const blockedTasks = tasks.filter(task => 
        task.status === TaskStatus.ASSIGNED &&
        new Date(task.created_at).getTime() < Date.now() - (48 * 60 * 60 * 1000)
      );
      
      const completedTasks = tasks.filter(task => 
        task.status === TaskStatus.COMPLETED &&
        task.started_at &&
        task.completed_at
      );
      
      const averageResolutionTime = completedTasks.length > 0 ?
        completedTasks.reduce((sum, task) => {
          const start = new Date(task.started_at!).getTime();
          const end = new Date(task.completed_at!).getTime();
          return sum + (end - start) / (1000 * 60 * 60);
        }, 0) / completedTasks.length : 0;
      
      return {
        statusBottlenecks,
        agentBottlenecks,
        systemBottlenecks: {
          totalBlockedTasks: blockedTasks.length,
          averageResolutionTime,
          criticalPath: blockedTasks
            .filter(task => task.priority === TaskPriority.CRITICAL)
            .map(task => task.id)
        }
      };
      
    } catch (error) {
      this.emit('bottleneckAnalysisFailed', { error });
      throw error;
    }
  }

  // ===== EVENT SUBSCRIPTIONS =====

  public onTaskCreated(listener: EventListener<Task>): Subscription {
    return this.subscribe('taskCreated', listener);
  }

  public onTaskUpdated(listener: EventListener<Task>): Subscription {
    return this.subscribe('taskUpdated', listener);
  }

  public onTaskDeleted(listener: EventListener<string>): Subscription {
    return this.subscribe('taskDeleted', listener);
  }

  public onTaskAssigned(listener: EventListener<TaskAssignmentResult>): Subscription {
    return this.subscribe('taskAssigned', listener);
  }

  public onTaskStarted(listener: EventListener<Task>): Subscription {
    return this.subscribe('taskStarted', listener);
  }

  public onTaskCompleted(listener: EventListener<Task>): Subscription {
    return this.subscribe('taskCompleted', listener);
  }

  public onTaskFailed(listener: EventListener<Task>): Subscription {
    return this.subscribe('taskFailed', listener);
  }

  public onKanbanBoardUpdated(listener: EventListener<KanbanBoard>): Subscription {
    return this.subscribe('kanbanBoardUpdated', listener);
  }

  public onTasksLoaded(listener: EventListener<TaskListResponse>): Subscription {
    return this.subscribe('tasksLoaded', listener);
  }

  public onMonitoringStarted(listener: EventListener<void>): Subscription {
    return this.subscribe('monitoringStarted', listener);
  }

  public onMonitoringStopped(listener: EventListener<void>): Subscription {
    return this.subscribe('monitoringStopped', listener);
  }

  public onSprintCreated(listener: EventListener<SprintPlan>): Subscription {
    return this.subscribe('sprintCreated', listener);
  }

  public onSprintUpdated(listener: EventListener<SprintPlan>): Subscription {
    return this.subscribe('sprintUpdated', listener);
  }

  public onAutoAssignmentCompleted(listener: EventListener<any>): Subscription {
    return this.subscribe('autoAssignmentCompleted', listener);
  }

  public onBulkOperationCompleted(listener: EventListener<any>): Subscription {
    return this.subscribe('bulkOperationCompleted', listener);
  }

  // ===== UTILITY METHODS =====

  /**
   * Get local task by ID (cache only)
   */
  getLocalTask(taskId: string): Task | null {
    return this.tasks.get(taskId) || null;
  }

  /**
   * Get all local tasks (cache only)
   */
  getLocalTasks(): Task[] {
    return Array.from(this.tasks.values());
  }

  /**
   * Get sync status
   */
  getSyncStatus(): {
    lastSync: string | null;
    syncInProgress: boolean;
    cachedTaskCount: number;
  } {
    return {
      lastSync: this.lastSync,
      syncInProgress: this.syncInProgress,
      cachedTaskCount: this.tasks.size
    };
  }

  // ===== ENHANCED TASK MANAGEMENT =====

  /**
   * Multi-agent task assignment with load balancing
   */
  async assignTaskToAgent(taskId: string, agentId: string, options: {
    balanceLoad?: boolean;
    considerCapabilities?: boolean;
    priority?: TaskPriority;
  } = {}): Promise<TaskAssignmentResult> {
    try {
      const task = this.getTask(taskId);
      if (!task) {
        throw new Error(`Task ${taskId} not found`);
      }

      // Check agent workload if load balancing is enabled
      if (options.balanceLoad) {
        const workload = await this.getAgentWorkload(agentId);
        if (workload.currentLoad > 0.8) { // 80% capacity threshold
          const alternativeAgent = await this.findAlternativeAgent(task, agentId);
          if (alternativeAgent) {
            agentId = alternativeAgent;
          }
        }
      }

      // Update task assignment
      const updatedTask = {
        ...task,
        assignedAgentId: agentId,
        status: TaskStatus.ASSIGNED,
        assignedAt: new Date().toISOString(),
        priority: options.priority || task.priority
      };

      // Update local state
      this.tasks.set(taskId, updatedTask);
      
      // Update workload cache
      this.updateAgentWorkload(agentId, updatedTask);

      // Emit assignment event
      this.emit('taskAssigned', { taskId, agentId, task: updatedTask });

      return {
        success: true,
        task: updatedTask,
        message: `Task assigned to agent ${agentId} successfully`
      };

    } catch (error) {
      this.emit('taskAssignmentFailed', { taskId, agentId, error });
      throw error;
    }
  }

  /**
   * Bulk task operations for multi-task management
   */
  async performBulkTaskOperation(taskIds: string[], operation: {
    type: 'assign' | 'move' | 'delete' | 'prioritize' | 'sprint';
    agentId?: string;
    status?: TaskStatus;
    priority?: TaskPriority;
    sprintId?: string;
  }): Promise<{
    success: boolean;
    results: Record<string, { success: boolean; message: string }>;
    summary: {
      total: number;
      successful: number;
      failed: number;
    };
  }> {
    const results: Record<string, { success: boolean; message: string }> = {};
    let successful = 0;
    let failed = 0;

    for (const taskId of taskIds) {
      try {
        switch (operation.type) {
          case 'assign':
            if (operation.agentId) {
              await this.assignTaskToAgent(taskId, operation.agentId, { balanceLoad: true });
              results[taskId] = { success: true, message: 'Task assigned successfully' };
            } else {
              throw new Error('Agent ID required for assignment');
            }
            break;

          case 'move':
            if (operation.status) {
              await this.updateTaskStatus(taskId, operation.status);
              results[taskId] = { success: true, message: 'Task status updated successfully' };
            } else {
              throw new Error('Status required for move operation');
            }
            break;

          case 'delete':
            await this.deleteTask(taskId);
            results[taskId] = { success: true, message: 'Task deleted successfully' };
            break;

          case 'prioritize':
            if (operation.priority) {
              const task = this.getTask(taskId);
              if (task) {
                await this.updateTask(taskId, { priority: operation.priority });
                results[taskId] = { success: true, message: 'Task priority updated successfully' };
              } else {
                throw new Error('Task not found');
              }
            } else {
              throw new Error('Priority required for prioritize operation');
            }
            break;

          case 'sprint':
            if (operation.sprintId) {
              await this.addTaskToSprint(taskId, operation.sprintId);
              results[taskId] = { success: true, message: 'Task added to sprint successfully' };
            } else {
              throw new Error('Sprint ID required for sprint operation');
            }
            break;

          default:
            throw new Error(`Unknown operation type: ${operation.type}`);
        }
        successful++;
      } catch (error) {
        results[taskId] = {
          success: false,
          message: error instanceof Error ? error.message : 'Operation failed'
        };
        failed++;
      }
    }

    this.emit('bulkTaskOperationCompleted', {
      operation,
      taskIds,
      results,
      summary: { total: taskIds.length, successful, failed }
    });

    return {
      success: failed === 0,
      results,
      summary: {
        total: taskIds.length,
        successful,
        failed
      }
    };
  }

  /**
   * Sprint planning and management with duration
   */
  async createSprintWithDuration(sprintData: {
    name: string;
    description: string;
    duration: number; // in days
    teamComposition: string[];
    capacity: number;
  }): Promise<SprintPlan> {
    const startDate = new Date();
    const endDate = new Date(startDate.getTime() + sprintData.duration * 24 * 60 * 60 * 1000);

    const sprint: SprintPlan = {
      id: `sprint-${Date.now()}`,
      name: sprintData.name,
      description: sprintData.description,
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
      status: 'planning',
      tasks: [],
      velocity: {
        planned: 0,
        completed: 0,
        capacity: sprintData.capacity
      },
      teamComposition: sprintData.teamComposition,
      burndownData: []
    };

    this.currentSprint = sprint;
    this.emit('sprintCreated', sprint);

    return sprint;
  }

  /**
   * Add task to sprint with capacity validation
   */
  async addTaskToSprint(taskId: string, sprintId: string): Promise<void> {
    const task = this.getTask(taskId);
    if (!task) {
      throw new Error(`Task ${taskId} not found`);
    }

    if (this.currentSprint && this.currentSprint.id === sprintId) {
      // Check sprint capacity
      const currentLoad = this.currentSprint.tasks.reduce((sum, t) => sum + (t.estimatedEffort || 1), 0);
      const taskEffort = task.estimatedEffort || 1;

      if (currentLoad + taskEffort > this.currentSprint.velocity.capacity) {
        throw new Error('Sprint capacity exceeded');
      }

      // Add task to sprint
      this.currentSprint.tasks.push(task);
      this.currentSprint.velocity.planned += taskEffort;

      // Update task with sprint assignment
      const updatedTask = { ...task, sprintId, addedToSprintAt: new Date().toISOString() };
      this.tasks.set(taskId, updatedTask);

      this.emit('taskAddedToSprint', { task: updatedTask, sprint: this.currentSprint });
    } else {
      throw new Error('Sprint not found or not active');
    }
  }

  /**
   * Get agent workload for load balancing
   */
  async getAgentWorkload(agentId: string): Promise<AgentWorkload> {
    // Check cache first
    if (this.workloadCache.has(agentId)) {
      const cached = this.workloadCache.get(agentId)!;
      // Return cached if less than 30 seconds old
      if (Date.now() - new Date(cached.lastUpdated).getTime() < 30000) {
        return cached;
      }
    }

    // Calculate workload from current tasks
    const agentTasks = Array.from(this.tasks.values()).filter(task => 
      task.assignedAgentId === agentId && 
      [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS].includes(task.status)
    );

    const totalEffort = agentTasks.reduce((sum, task) => sum + (task.estimatedEffort || 1), 0);
    const maxCapacity = 8; // 8 story points max capacity

    const workload: AgentWorkload = {
      agentId,
      agentName: `Agent ${agentId}`,
      role: 'backend_developer' as AgentRole, // This would come from agent service
      currentLoad: Math.min(totalEffort / maxCapacity, 1.0),
      taskCount: agentTasks.length,
      totalEffort,
      capacity: maxCapacity,
      efficiency: 0.85, // Default efficiency
      lastUpdated: new Date().toISOString()
    };

    // Cache the result
    this.workloadCache.set(agentId, workload);

    return workload;
  }

  /**
   * Find alternative agent for load balancing
   */
  private async findAlternativeAgent(task: Task, originalAgentId: string): Promise<string | null> {
    // This would integrate with the agent service from Day 1
    // For now, return null to keep original assignment
    return null;
  }

  /**
   * Update agent workload cache
   */
  private updateAgentWorkload(agentId: string, task: Task): void {
    const cached = this.workloadCache.get(agentId);
    if (cached) {
      cached.taskCount += 1;
      cached.totalEffort += task.estimatedEffort || 1;
      cached.currentLoad = Math.min(cached.totalEffort / cached.capacity, 1.0);
      cached.lastUpdated = new Date().toISOString();
    }
  }

  /**
   * Get task templates from template registry
   */
  getRegisteredTaskTemplates(): TaskTemplate[] {
    return Array.from(this.templates.values());
  }

  /**
   * Create task from registered template with customizations
   */
  async createTaskFromRegisteredTemplate(templateId: string, customizations: {
    title?: string;
    description?: string;
    assignedAgentId?: string;
    priority?: TaskPriority;
    sprintId?: string;
  } = {}): Promise<Task> {
    const template = this.templates.get(templateId);
    if (!template) {
      throw new Error(`Template ${templateId} not found`);
    }

    const taskData: TaskCreate = {
      title: customizations.title || template.name,
      description: customizations.description || template.description,
      taskType: template.defaultTaskType,
      priority: customizations.priority || template.defaultPriority,
      estimatedEffort: template.estimatedEffort,
      requiredSkills: template.requiredSkills,
      assignedAgentId: customizations.assignedAgentId,
      sprintId: customizations.sprintId,
      labels: [`template:${templateId}`],
      checklist: template.checklistItems.map(item => ({ item, completed: false }))
    };

    return this.createTask(taskData);
  }

  /**
   * Initialize default task templates
   */
  private initializeTaskTemplates(): void {
    const defaultTemplates: TaskTemplate[] = [
      {
        id: 'feature-implementation',
        name: 'Feature Implementation',
        description: 'Implement a new feature with full testing',
        defaultTaskType: TaskType.FEATURE,
        defaultPriority: TaskPriority.MEDIUM,
        estimatedEffort: 5,
        requiredSkills: ['development', 'testing'],
        checklistItems: [
          'Analyze requirements',
          'Design implementation approach',
          'Write code',
          'Write unit tests',
          'Write integration tests',
          'Code review',
          'Documentation update'
        ],
        associatedRoles: ['backend_developer', 'frontend_developer', 'qa_engineer'] as AgentRole[]
      },
      {
        id: 'bug-fix',
        name: 'Bug Fix',
        description: 'Fix a reported bug with root cause analysis',
        defaultTaskType: TaskType.BUG,
        defaultPriority: TaskPriority.HIGH,
        estimatedEffort: 3,
        requiredSkills: ['debugging', 'testing'],
        checklistItems: [
          'Reproduce the bug',
          'Identify root cause',
          'Implement fix',
          'Write regression test',
          'Verify fix works',
          'Update documentation'
        ],
        associatedRoles: ['backend_developer', 'frontend_developer'] as AgentRole[]
      },
      {
        id: 'refactoring',
        name: 'Code Refactoring',
        description: 'Refactor code for better maintainability',
        defaultTaskType: TaskType.TECHNICAL,
        defaultPriority: TaskPriority.LOW,
        estimatedEffort: 4,
        requiredSkills: ['refactoring', 'architecture'],
        checklistItems: [
          'Identify refactoring scope',
          'Ensure test coverage',
          'Refactor incrementally',
          'Run all tests',
          'Performance validation',
          'Code review'
        ],
        associatedRoles: ['architect', 'backend_developer', 'frontend_developer'] as AgentRole[]
      }
    ];

    defaultTemplates.forEach(template => {
      this.templates.set(template.id, template);
    });
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.stopMonitoring();
    this.tasks.clear();
    this.workloadCache.clear();
    this.velocityHistory.clear();
    this.templates.clear();
    this.lastSync = null;
    this.syncInProgress = false;
    this.currentSprint = null;
    this.analyticsData = null;
    super.destroy();
  }
}

// Singleton instance
let taskService: TaskService | null = null;

export function getTaskService(config?: Partial<ServiceConfig>): TaskService {
  if (!taskService) {
    taskService = new TaskService(config);
  }
  return taskService;
}

export function resetTaskService(): void {
  if (taskService) {
    taskService.destroy();
    taskService = null;
  }
}