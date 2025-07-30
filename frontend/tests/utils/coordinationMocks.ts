/**
 * Mock utilities for coordination dashboard testing
 * 
 * Provides comprehensive mock data and utilities for testing
 * the enhanced coordination features.
 */

export interface MockAgent {
  agent_id: string
  name: string
  type: string
  status: 'active' | 'idle' | 'busy' | 'sleeping' | 'error'
  current_workload: number
  available_capacity: number
  capabilities: Array<{
    name: string
    confidence_level: number
    description?: string
  }>
  active_tasks: number
  completed_today: number
  average_response_time_ms: number
  last_heartbeat: string
  performance_score: number
}

export interface MockTask {
  id: string
  task_title: string
  task_description: string
  task_type: string
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  required_capabilities: string[]
  estimated_effort_hours?: number
  deadline?: string
  status: string
  dependencies?: string[]
  context_data?: Record<string, any>
}

export interface MockPerformanceMetrics {
  system_efficiency_score: number
  agent_utilization_percentage: number
  tasks_completed_today: number
  average_task_completion_time_hours: number
  total_agents: number
  active_agents: number
  total_tasks: number
  error_count: number
  bottlenecks: Array<{
    id: string
    title: string
    description: string
    severity: 'low' | 'medium' | 'high' | 'critical'
    impact: number
  }>
  top_performing_agents: Array<{
    agent_id: string
    name: string
    tasks_completed: number
    avg_response_time: number
    performance_score: number
  }>
}

// Agent mock factory
export function mockAgent(overrides: Partial<MockAgent> = {}): MockAgent {
  const baseAgent: MockAgent = {
    agent_id: `agent-${Math.random().toString(36).substr(2, 9)}`,
    name: `Agent ${Math.floor(Math.random() * 100)}`,
    type: 'CLAUDE',
    status: 'active',
    current_workload: Math.random() * 0.8,
    available_capacity: 0.8,
    capabilities: [
      {
        name: 'Vue.js',
        confidence_level: 0.8 + Math.random() * 0.2,
        description: 'Frontend framework expertise'
      },
      {
        name: 'TypeScript',
        confidence_level: 0.7 + Math.random() * 0.3,
        description: 'Type-safe JavaScript development'
      },
      {
        name: 'Python',
        confidence_level: 0.6 + Math.random() * 0.4,
        description: 'Backend development'
      },
      {
        name: 'API Development',
        confidence_level: 0.7 + Math.random() * 0.3,
        description: 'RESTful API design and implementation'
      }
    ],
    active_tasks: Math.floor(Math.random() * 5),
    completed_today: Math.floor(Math.random() * 20),
    average_response_time_ms: 200 + Math.random() * 800,
    last_heartbeat: new Date().toISOString(),
    performance_score: 0.6 + Math.random() * 0.4
  }

  return { ...baseAgent, ...overrides }
}

// Task mock factory
export function mockTask(overrides: Partial<MockTask> = {}): MockTask {
  const taskTypes = [
    'FRONTEND_DEVELOPMENT',
    'BACKEND_DEVELOPMENT',
    'API_INTEGRATION',
    'TESTING',
    'CODE_REVIEW',
    'DOCUMENTATION',
    'BUG_FIX'
  ]

  const priorities: Array<'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'> = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
  
  const capabilitySets = [
    ['Vue.js', 'TypeScript', 'CSS'],
    ['Python', 'FastAPI', 'Database Design'],
    ['JavaScript', 'Node.js', 'Express.js'],
    ['Testing', 'Jest', 'Cypress'],
    ['Documentation', 'Technical Writing'],
    ['Debugging', 'Problem Solving']
  ]

  const baseTask: MockTask = {
    id: `task-${Math.random().toString(36).substr(2, 9)}`,
    task_title: `Task ${Math.floor(Math.random() * 1000)}`,
    task_description: 'This is a sample task description for testing purposes.',
    task_type: taskTypes[Math.floor(Math.random() * taskTypes.length)],
    priority: priorities[Math.floor(Math.random() * priorities.length)],
    required_capabilities: capabilitySets[Math.floor(Math.random() * capabilitySets.length)],
    estimated_effort_hours: 1 + Math.random() * 20,
    deadline: new Date(Date.now() + Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
    status: 'pending',
    dependencies: [],
    context_data: {
      project: 'LeanVibe Agent Hive',
      priority_reason: 'Feature enhancement request'
    }
  }

  return { ...baseTask, ...overrides }
}

// Performance metrics mock factory
export function mockPerformanceMetrics(overrides: Partial<MockPerformanceMetrics> = {}): MockPerformanceMetrics {
  const baseMetrics: MockPerformanceMetrics = {
    system_efficiency_score: 0.75 + Math.random() * 0.25,
    agent_utilization_percentage: 60 + Math.random() * 30,
    tasks_completed_today: Math.floor(Math.random() * 100),
    average_task_completion_time_hours: 1 + Math.random() * 8,
    total_agents: 5 + Math.floor(Math.random() * 15),
    active_agents: 3 + Math.floor(Math.random() * 10),
    total_tasks: 50 + Math.floor(Math.random() * 200),
    error_count: Math.floor(Math.random() * 10),
    bottlenecks: Math.random() > 0.7 ? [
      {
        id: 'bottleneck-1',
        title: 'High Memory Usage',
        description: 'Agent memory usage exceeding 90%',
        severity: 'high',
        impact: 0.7 + Math.random() * 0.3
      },
      {
        id: 'bottleneck-2',
        title: 'Slow Response Times',
        description: 'Average response time above threshold',
        severity: 'medium',
        impact: 0.5 + Math.random() * 0.3
      }
    ] : [],
    top_performing_agents: Array.from({ length: 5 }, (_, i) => ({
      agent_id: `agent-${i}`,
      name: `Top Agent ${i + 1}`,
      tasks_completed: 10 + Math.floor(Math.random() * 20),
      avg_response_time: 100 + Math.random() * 300,
      performance_score: 0.8 + Math.random() * 0.2
    }))
  }

  return { ...baseMetrics, ...overrides }
}

// Agent capability matching mock
export function mockAgentMatch(agent: MockAgent, taskRequirements: string[]) {
  const agentCapabilities = agent.capabilities.map(c => c.name.toLowerCase())
  const requiredCapabilities = taskRequirements.map(r => r.toLowerCase())
  
  const matchedCapabilities = requiredCapabilities.filter(req => 
    agentCapabilities.some(agent => agent.includes(req) || req.includes(agent))
  )
  
  const capabilityScore = matchedCapabilities.length / requiredCapabilities.length
  const workloadScore = 1 - agent.current_workload
  const performanceScore = agent.performance_score
  
  const overallMatch = (capabilityScore * 0.5) + (workloadScore * 0.3) + (performanceScore * 0.2)
  
  return {
    agent,
    overallMatch,
    capabilityMatches: Object.fromEntries(
      requiredCapabilities.map(capability => [
        capability,
        {
          score: agentCapabilities.some(ac => ac.includes(capability)) ? 0.7 + Math.random() * 0.3 : 0,
          confidence: 0.6 + Math.random() * 0.4,
          hasCapability: agentCapabilities.some(ac => ac.includes(capability))
        }
      ])
    ),
    reasoning: [
      capabilityScore > 0.8 ? 'Strong skill match' : 'Partial skill match',
      workloadScore > 0.7 ? 'Low workload' : 'Moderate workload',
      performanceScore > 0.8 ? 'High performance' : 'Good performance'
    ].slice(0, 2)
  }
}

// WebSocket message mocks
export function mockTaskAssignmentMessage(taskId: string, agentId: string) {
  return {
    type: 'task_assignment',
    data: {
      task_id: taskId,
      agent_id: agentId,
      agent_name: `Agent ${agentId.substr(-3)}`,
      task_title: `Task ${taskId.substr(-3)}`,
      priority: 'HIGH',
      assigned_at: new Date().toISOString(),
      confidence_score: 0.8 + Math.random() * 0.2,
      estimated_completion: new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString()
    },
    timestamp: new Date().toISOString()
  }
}

export function mockAgentWorkloadMessage(agentId: string) {
  return {
    type: 'agent_workload',
    data: {
      agent_id: agentId,
      agent_name: `Agent ${agentId.substr(-3)}`,
      old_workload: 0.5,
      new_workload: 0.7,
      active_tasks: 3,
      available_capacity: 0.3,
      updated_at: new Date().toISOString()
    },
    timestamp: new Date().toISOString()
  }
}

export function mockPerformanceMetricsMessage() {
  return {
    type: 'performance_metrics',
    data: {
      system_efficiency: 0.82,
      average_utilization: 0.75,
      task_throughput: 45,
      error_rate: 0.03,
      timestamp: new Date().toISOString(),
      agent_metrics: {
        'agent-1': { performance: 0.9, workload: 0.6 },
        'agent-2': { performance: 0.8, workload: 0.8 }
      }
    },
    timestamp: new Date().toISOString()
  }
}

export function mockBottleneckAlert() {
  return {
    type: 'bottleneck_detected',
    data: {
      id: `bottleneck-${Date.now()}`,
      title: 'High CPU Usage Detected',
      description: 'System CPU usage has exceeded 90% for the past 5 minutes',
      severity: 'high' as const,
      impact: 0.8,
      category: 'performance',
      affected_agents: ['agent-1', 'agent-2'],
      detected_at: new Date().toISOString(),
      auto_resolve: false
    },
    timestamp: new Date().toISOString()
  }
}

// Mock WebSocket implementation
export function createMockWebSocket() {
  const mockWs = {
    readyState: 1, // WebSocket.OPEN
    onopen: null as ((event: Event) => void) | null,
    onclose: null as ((event: CloseEvent) => void) | null,
    onmessage: null as ((event: MessageEvent) => void) | null,
    onerror: null as ((event: Event) => void) | null,
    send: vi.fn(),
    close: vi.fn(),
    
    // Test utilities
    simulateOpen() {
      this.readyState = 1
      if (this.onopen) {
        this.onopen(new Event('open'))
      }
    },
    
    simulateMessage(data: any) {
      if (this.onmessage) {
        this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }))
      }
    },
    
    simulateClose(code = 1000, reason = '') {
      this.readyState = 3 // WebSocket.CLOSED
      if (this.onclose) {
        this.onclose(new CloseEvent('close', { code, reason }))
      }
    },
    
    simulateError(error?: any) {
      if (this.onerror) {
        this.onerror(new ErrorEvent('error', { error }))
      }
    }
  }

  // Auto-simulate connection after creation
  setTimeout(() => mockWs.simulateOpen(), 10)
  
  return mockWs
}

// Chart.js mock utilities
export function mockChartInstance() {
  return {
    data: {
      labels: [],
      datasets: []
    },
    options: {},
    destroy: vi.fn(),
    update: vi.fn(),
    reset: vi.fn(),
    render: vi.fn(),
    resize: vi.fn(),
    clear: vi.fn(),
    stop: vi.fn(),
    toBase64Image: vi.fn(() => 'data:image/png;base64,mock'),
    generateLegend: vi.fn(() => '<div>Legend</div>'),
    getElementAtEvent: vi.fn(() => []),
    getElementsAtEvent: vi.fn(() => []),
    getDatasetAtEvent: vi.fn(() => [])
  }
}

// Performance testing utilities
export function createPerformanceTestData(size: number) {
  return {
    agents: Array.from({ length: size }, () => mockAgent()),
    tasks: Array.from({ length: size * 2 }, () => mockTask()),
    metrics: Array.from({ length: size / 10 }, () => mockPerformanceMetrics())
  }
}

// Mobile testing utilities
export function mockMobileEnvironment() {
  // Mock touch events
  Object.defineProperty(window, 'ontouchstart', {
    value: null,
    writable: true
  })
  
  // Mock mobile viewport
  Object.defineProperty(window, 'innerWidth', {
    value: 375,
    writable: true
  })
  
  Object.defineProperty(window, 'innerHeight', {
    value: 667,
    writable: true
  })
  
  // Mock user agent
  Object.defineProperty(window.navigator, 'userAgent', {
    value: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1',
    writable: true
  })
  
  // Mock touch support
  Object.defineProperty(window, 'TouchEvent', {
    value: class TouchEvent extends Event {
      touches: Touch[]
      targetTouches: Touch[]
      changedTouches: Touch[]
      
      constructor(type: string, options: any = {}) {
        super(type, options)
        this.touches = options.touches || []
        this.targetTouches = options.targetTouches || []
        this.changedTouches = options.changedTouches || []
      }
    }
  })
  
  // Mock vibration API
  Object.defineProperty(window.navigator, 'vibrate', {
    value: vi.fn(),
    writable: true
  })
}

// Test data builders
export class CoordinationTestDataBuilder {
  private agents: MockAgent[] = []
  private tasks: MockTask[] = []
  private metrics: MockPerformanceMetrics | null = null
  
  withAgents(count: number, overrides?: Partial<MockAgent>) {
    this.agents = Array.from({ length: count }, () => mockAgent(overrides))
    return this
  }
  
  withTasks(count: number, overrides?: Partial<MockTask>) {
    this.tasks = Array.from({ length: count }, () => mockTask(overrides))
    return this
  }
  
  withMetrics(overrides?: Partial<MockPerformanceMetrics>) {
    this.metrics = mockPerformanceMetrics(overrides)
    return this
  }
  
  withHighWorkloadAgents(count: number) {
    const highWorkloadAgents = Array.from({ length: count }, () => 
      mockAgent({ current_workload: 0.8 + Math.random() * 0.2 })
    )
    this.agents.push(...highWorkloadAgents)
    return this
  }
  
  withCriticalTasks(count: number) {
    const criticalTasks = Array.from({ length: count }, () =>
      mockTask({ priority: 'CRITICAL' })
    )
    this.tasks.push(...criticalTasks)
    return this
  }
  
  withBottlenecks(count: number) {
    if (!this.metrics) this.metrics = mockPerformanceMetrics()
    
    this.metrics.bottlenecks = Array.from({ length: count }, (_, i) => ({
      id: `bottleneck-${i}`,
      title: `Bottleneck ${i + 1}`,
      description: `System bottleneck description ${i + 1}`,
      severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any,
      impact: Math.random()
    }))
    
    return this
  }
  
  build() {
    return {
      agents: this.agents,
      tasks: this.tasks,
      metrics: this.metrics || mockPerformanceMetrics()
    }
  }
}

// Export test data builder factory
export function createTestDataBuilder() {
  return new CoordinationTestDataBuilder()
}

// Common test scenarios
export const testScenarios = {
  // Basic coordination scenario
  basicCoordination() {
    return createTestDataBuilder()
      .withAgents(5)
      .withTasks(10)
      .withMetrics()
      .build()
  },
  
  // High load scenario
  highLoad() {
    return createTestDataBuilder()
      .withAgents(20)
      .withTasks(100)
      .withHighWorkloadAgents(15)
      .withCriticalTasks(20)
      .withBottlenecks(3)
      .build()
  },
  
  // Low activity scenario
  lowActivity() {
    return createTestDataBuilder()
      .withAgents(3, { current_workload: 0.1 })
      .withTasks(5, { priority: 'LOW' })
      .withMetrics({ system_efficiency_score: 0.6 })
      .build()
  },
  
  // Error conditions
  errorConditions() {
    return createTestDataBuilder()
      .withAgents(5, { status: 'error' })
      .withTasks(10, { status: 'failed' })
      .withMetrics({ error_count: 15 })
      .withBottlenecks(5)
      .build()
  }
}

// API response mocks
export const apiMocks = {
  successfulTaskAssignment: {
    data: {
      task_id: 'task-123',
      assigned_agent_id: 'agent-456',
      agent_name: 'Test Agent',
      assignment_confidence: 0.85,
      estimated_completion_time: new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString(),
      capability_match_details: {
        capability_score: 0.9,
        workload_factor: 0.8,
        performance_factor: 0.85
      },
      workload_impact: 0.2
    }
  },
  
  agentRegistration: {
    data: {
      agent_id: 'agent-new',
      name: 'New Agent',
      type: 'CLAUDE',
      status: 'active',
      capabilities: [],
      performance_score: 0.8
    }
  },
  
  coordinationMetrics: {
    data: mockPerformanceMetrics()
  }
}

export default {
  mockAgent,
  mockTask,
  mockPerformanceMetrics,
  mockAgentMatch,
  createMockWebSocket,
  mockChartInstance,
  createPerformanceTestData,
  mockMobileEnvironment,
  createTestDataBuilder,
  testScenarios,
  apiMocks
}