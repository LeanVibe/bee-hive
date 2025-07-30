export type TaskStatus = 'pending' | 'in-progress' | 'review' | 'done'
export type TaskPriority = 'low' | 'medium' | 'high'
export type SyncStatus = 'synced' | 'pending' | 'error'

export interface Task {
  id: string
  title: string
  description?: string
  status: TaskStatus
  priority: TaskPriority
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
  id: string
  status?: TaskStatus
  priority?: TaskPriority
  title?: string
  description?: string
  tags?: string[]
  estimatedHours?: number
  actualHours?: number
  dueDate?: string
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