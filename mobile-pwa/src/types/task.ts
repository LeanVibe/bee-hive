export type TaskStatus = 'todo' | 'pending' | 'in-progress' | 'review' | 'done'
export type TaskPriority = 'low' | 'medium' | 'high' | 'critical'
export type TaskType = 'feature' | 'bug' | 'enhancement' | 'documentation' | 'testing' | 'refactoring'
export type SyncStatus = 'synced' | 'pending' | 'error'

export interface Task {
  id: string
  title: string
  description?: string
  status: TaskStatus
  priority: TaskPriority
  type: TaskType
  agent: string
  tags?: string[]
  createdAt: string
  updatedAt: string
  syncStatus?: SyncStatus
  estimatedHours?: number
  actualHours?: number
  dependencies?: string[]
  assignee?: string
  dueDate?: string
  acceptanceCriteria?: string[]
  metadata?: Record<string, any>
}

export interface TaskFilter {
  search?: string
  agent?: string
  priority?: TaskPriority
  status?: TaskStatus
  tags?: string[]
  dateRange?: {
    start: string
    end: string
  }
}

export interface TaskUpdate {
  id?: string
  status?: TaskStatus
  priority?: TaskPriority
  type?: TaskType
  title?: string
  description?: string
  tags?: string[]
  estimatedHours?: number
  actualHours?: number
  assignee?: string
  dueDate?: string
  acceptanceCriteria?: string[]
  dependencies?: string[]
  updated_at?: Date
  metadata?: Record<string, any>
}

export interface TaskCreate {
  title: string
  description?: string
  status: TaskStatus
  priority: TaskPriority
  type: TaskType
  assignee?: string
  tags?: string[]
  estimatedHours?: number
  dependencies?: string[]
  acceptanceCriteria?: string[]
  dueDate?: string
  created_at: Date
  updated_at: Date
  metadata?: Record<string, any>
}

export interface TaskMoveEvent {
  taskId: string
  newStatus: TaskStatus
  newIndex: number
  offline: boolean
}

export interface TaskStats {
  total: number
  byStatus: Record<TaskStatus, number>
  byPriority: Record<TaskPriority, number>
  byAgent: Record<string, number>
  completionRate: number
  averageHours: number
}