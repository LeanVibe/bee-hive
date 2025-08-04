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

  // ===== CLEANUP =====

  public destroy(): void {
    this.stopMonitoring();
    this.tasks.clear();
    this.lastSync = null;
    this.syncInProgress = false;
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