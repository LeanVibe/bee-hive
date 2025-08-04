import { Task, TaskStatus } from '../../src/types/task'

/**
 * Mock data for testing dashboard functionality
 */
export const mockTasks: Task[] = [
  {
    id: 'task-1',
    title: 'Implement user authentication',
    description: 'Add JWT-based authentication system',
    status: 'pending' as TaskStatus,
    priority: 'high',
    agent: 'agent-1',
    createdAt: '2025-01-01T10:00:00Z',
    updatedAt: '2025-01-01T10:00:00Z',
    tags: ['backend', 'security'],
    estimatedHours: 8,
    actualHours: 0
  },
  {
    id: 'task-2',
    title: 'Create dashboard UI components',
    description: 'Build responsive dashboard with real-time updates',
    status: 'in-progress' as TaskStatus,
    priority: 'high',
    agent: 'agent-2',
    createdAt: '2025-01-01T09:00:00Z',
    updatedAt: '2025-01-01T11:00:00Z',
    tags: ['frontend', 'ui'],
    estimatedHours: 12,
    actualHours: 6
  },
  {
    id: 'task-3',
    title: 'Set up automated testing',
    description: 'Configure Playwright E2E testing framework',
    status: 'review' as TaskStatus,
    priority: 'medium',
    agent: 'agent-3',
    createdAt: '2025-01-01T08:00:00Z',
    updatedAt: '2025-01-01T12:00:00Z',
    tags: ['testing', 'automation'],
    estimatedHours: 6,
    actualHours: 5
  },
  {
    id: 'task-4',
    title: 'Deploy to production',
    description: 'Set up CI/CD pipeline and deploy application',
    status: 'done' as TaskStatus,
    priority: 'high',
    agent: 'agent-1',
    createdAt: '2025-01-01T07:00:00Z',
    updatedAt: '2025-01-01T13:00:00Z',
    tags: ['devops', 'deployment'],
    estimatedHours: 4,
    actualHours: 3
  },
  {
    id: 'task-5',
    title: 'Optimize performance',
    description: 'Improve application load times and responsiveness',
    status: 'pending' as TaskStatus,
    priority: 'low',
    agent: 'agent-2',
    createdAt: '2025-01-01T06:00:00Z',
    updatedAt: '2025-01-01T06:00:00Z',
    tags: ['performance', 'optimization'],
    estimatedHours: 10,
    actualHours: 0
  }
]

export const mockAgents = [
  {
    id: 'agent-1',
    name: 'Backend Developer',
    status: 'active',
    uptime: 3600,
    lastSeen: '2025-01-01T13:00:00Z',
    currentTask: 'task-1',
    capabilities: ['backend', 'database', 'api'],
    performance: {
      score: 92,
      trend: 'up',
      tasksCompleted: 15,
      averageTime: 4.2
    },
    metrics: {
      cpuUsage: [20, 25, 30, 28, 22],
      memoryUsage: [45, 48, 52, 49, 46],
      tokenUsage: [1200, 1350, 1100, 1250, 1180],
      tasksCompleted: [2, 3, 1, 4, 2],
      errorRate: [0, 0, 1, 0, 0],
      responseTime: [1.2, 0.8, 1.5, 1.1, 0.9],
      timestamps: ['10:00', '11:00', '12:00', '13:00', '14:00']
    }
  },
  {
    id: 'agent-2',
    name: 'Frontend Developer',
    status: 'active',
    uptime: 2800,
    lastSeen: '2025-01-01T12:30:00Z',
    currentTask: 'task-2',
    capabilities: ['frontend', 'ui', 'ux'],
    performance: {
      score: 88,
      trend: 'stable',
      tasksCompleted: 12,
      averageTime: 3.8
    },
    metrics: {
      cpuUsage: [15, 20, 18, 22, 19],
      memoryUsage: [38, 42, 40, 45, 41],
      tokenUsage: [800, 920, 750, 850, 780],
      tasksCompleted: [1, 2, 2, 3, 1],
      errorRate: [0, 1, 0, 0, 0],
      responseTime: [0.9, 1.3, 1.0, 0.7, 1.1],
      timestamps: ['10:00', '11:00', '12:00', '13:00', '14:00']
    }
  },
  {
    id: 'agent-3',
    name: 'QA Engineer',
    status: 'idle',
    uptime: 1200,
    lastSeen: '2025-01-01T12:00:00Z',
    currentTask: null,
    capabilities: ['testing', 'automation', 'quality'],
    performance: {
      score: 95,
      trend: 'up',
      tasksCompleted: 8,
      averageTime: 2.5
    },
    metrics: {
      cpuUsage: [10, 12, 8, 15, 11],
      memoryUsage: [25, 28, 23, 32, 27],
      tokenUsage: [500, 580, 420, 600, 480],
      tasksCompleted: [1, 1, 2, 1, 1],
      errorRate: [0, 0, 0, 0, 0],
      responseTime: [0.5, 0.8, 0.6, 0.4, 0.7],
      timestamps: ['10:00', '11:00', '12:00', '13:00', '14:00']
    }
  },
  {
    id: 'agent-4',
    name: 'DevOps Engineer',
    status: 'error',
    uptime: 600,
    lastSeen: '2025-01-01T11:00:00Z',
    currentTask: null,
    capabilities: ['devops', 'deployment', 'monitoring'],
    performance: {
      score: 75,
      trend: 'down',
      tasksCompleted: 5,
      averageTime: 6.2
    },
    metrics: {
      cpuUsage: [30, 45, 60, 35, 25],
      memoryUsage: [55, 70, 85, 60, 50],
      tokenUsage: [1500, 1800, 2200, 1600, 1400],
      tasksCompleted: [1, 0, 0, 2, 1],
      errorRate: [0, 1, 2, 1, 0],
      responseTime: [2.5, 4.0, 5.5, 3.2, 2.8],
      timestamps: ['10:00', '11:00', '12:00', '13:00', '14:00']
    }
  }
]

export const mockEvents = [
  {
    id: 'event-1',
    type: 'task-created',
    title: 'New task created',
    description: 'Task "Implement user authentication" was created',
    agent: 'agent-1',
    timestamp: '2025-01-01T10:00:00Z',
    severity: 'info',
    metadata: { taskId: 'task-1', priority: 'high' }
  },
  {
    id: 'event-2',
    type: 'task-completed',
    title: 'Task completed',
    description: 'Task "Deploy to production" was completed',
    agent: 'agent-1',
    timestamp: '2025-01-01T13:00:00Z',
    severity: 'info',
    metadata: { taskId: 'task-4', duration: '3h' }
  },
  {
    id: 'event-3',
    type: 'agent-error',
    title: 'Agent error occurred',
    description: 'DevOps Engineer encountered an error during deployment',
    agent: 'agent-4',
    timestamp: '2025-01-01T11:30:00Z',
    severity: 'error',
    metadata: { errorCode: 'DEPLOY_FAILED', message: 'Connection timeout' }
  },
  {
    id: 'event-4',
    type: 'system-alert',
    title: 'High CPU usage detected',
    description: 'System CPU usage exceeded 80% threshold',
    agent: null,
    timestamp: '2025-01-01T12:45:00Z',
    severity: 'warning',
    metadata: { cpuUsage: 85, threshold: 80 }
  }
]

export const mockSystemHealth = {
  overall: 'healthy',
  components: {
    healthy: 8,
    degraded: 1,
    unhealthy: 0
  },
  services: {
    database: { status: 'healthy', latency: 45 },
    redis: { status: 'healthy', latency: 12 },
    api: { status: 'healthy', latency: 150 },
    websocket: { status: 'degraded', latency: 800 }
  },
  metrics: {
    uptime: 99.8,
    responseTime: 120,
    errorRate: 0.02,
    throughput: 450
  }
}

export const mockPerformanceMetrics = {
  system_metrics: {
    cpu_usage: 35.5,
    memory_usage: 68.2,
    disk_usage: 45.8,
    network_io: 125.3
  },
  application_metrics: {
    active_connections: 25,
    requests_per_second: 15.2,
    average_response_time: 180,
    error_rate: 0.8
  },
  timestamp: '2025-01-01T13:00:00Z'
}