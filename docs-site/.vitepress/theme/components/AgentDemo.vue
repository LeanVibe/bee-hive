<template>
  <div class="agent-demo glass-card">
    <div class="demo-header">
      <h3 class="demo-title">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
        Live Multi-Agent Coordination
      </h3>
      <div class="demo-controls">
        <select v-model="selectedScenario" @change="loadScenario" class="scenario-selector">
          <option value="feature-development">Feature Development</option>
          <option value="bug-fix">Bug Fix Workflow</option>
          <option value="code-review">Code Review Process</option>
          <option value="deployment">Deployment Pipeline</option>
        </select>
        <button @click="startDemo" :disabled="isRunning" class="start-button">
          <svg v-if="!isRunning" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          <svg v-else class="w-4 h-4 spinning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          {{ isRunning ? 'Running...' : 'Start Demo' }}
        </button>
      </div>
    </div>

    <div class="demo-content">
      <!-- Agent Network Visualization -->
      <div class="agent-network">
        <div class="network-header">
          <h4>Agent Network</h4>
          <div class="network-stats">
            <span class="stat">
              <span class="stat-value">{{ activeAgents }}</span>
              <span class="stat-label">Active</span>
            </span>
            <span class="stat">
              <span class="stat-value">{{ completedTasks }}</span>
              <span class="stat-label">Tasks</span>
            </span>
          </div>
        </div>
        
        <div class="agent-grid">
          <div 
            v-for="agent in agents" 
            :key="agent.id"
            :class="['agent-node', `status-${agent.status}`]"
            @click="selectAgent(agent)"
          >
            <div class="agent-avatar">
              <component :is="getAgentIcon(agent.type)" class="w-6 h-6" />
            </div>
            <div class="agent-info">
              <div class="agent-name">{{ agent.name }}</div>
              <div class="agent-type">{{ agent.type }}</div>
              <div class="agent-status">{{ agent.status }}</div>
            </div>
            <div v-if="agent.currentTask" class="agent-task">
              {{ agent.currentTask }}
            </div>
            <div class="agent-progress">
              <div 
                class="progress-bar" 
                :style="{ width: `${agent.progress}%` }"
              ></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Real-time Activity Feed -->
      <div class="activity-feed">
        <div class="feed-header">
          <h4>Live Activity</h4>
          <button @click="clearActivity" class="clear-button">Clear</button>
        </div>
        
        <div class="feed-content" ref="feedContent">
          <TransitionGroup name="activity" tag="div">
            <div 
              v-for="activity in activities" 
              :key="activity.id"
              :class="['activity-item', `type-${activity.type}`]"
            >
              <div class="activity-timestamp">
                {{ formatTime(activity.timestamp) }}
              </div>
              <div class="activity-agent">
                <component :is="getAgentIcon(activity.agentType)" class="w-4 h-4" />
                {{ activity.agentName }}
              </div>
              <div class="activity-message">
                {{ activity.message }}
              </div>
              <div v-if="activity.details" class="activity-details">
                {{ activity.details }}
              </div>
            </div>
          </TransitionGroup>
        </div>
      </div>
    </div>

    <!-- Selected Agent Details -->
    <div v-if="selectedAgent" class="agent-details">
      <div class="details-header">
        <h4>{{ selectedAgent.name }} Details</h4>
        <button @click="selectedAgent = null" class="close-button">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      <div class="details-content">
        <div class="detail-section">
          <h5>Capabilities</h5>
          <div class="capability-tags">
            <span 
              v-for="capability in selectedAgent.capabilities" 
              :key="capability"
              class="capability-tag"
            >
              {{ capability }}
            </span>
          </div>
        </div>
        
        <div class="detail-section">
          <h5>Performance Metrics</h5>
          <div class="metrics-grid">
            <div class="metric">
              <span class="metric-label">Success Rate</span>
              <span class="metric-value">{{ selectedAgent.successRate }}%</span>
            </div>
            <div class="metric">
              <span class="metric-label">Avg Duration</span>
              <span class="metric-value">{{ selectedAgent.avgDuration }}s</span>
            </div>
            <div class="metric">
              <span class="metric-label">Tasks Completed</span>
              <span class="metric-value">{{ selectedAgent.tasksCompleted }}</span>
            </div>
          </div>
        </div>
        
        <div class="detail-section">
          <h5>Recent Tasks</h5>
          <div class="recent-tasks">
            <div 
              v-for="task in selectedAgent.recentTasks" 
              :key="task.id"
              class="task-item"
            >
              <span class="task-name">{{ task.name }}</span>
              <span :class="['task-status', `status-${task.status}`]">
                {{ task.status }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="demo-footer">
      <div class="demo-info">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        This demo shows real-time multi-agent coordination in action.
        <a href="/learn/fundamentals/coordination" class="info-link">Learn more about coordination</a>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

// Icons for different agent types
const AgentIcons = {
  architect: () => h('svg', { class: 'w-6 h-6', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
    h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4' })
  ]),
  developer: () => h('svg', { class: 'w-6 h-6', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
    h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4' })
  ]),
  tester: () => h('svg', { class: 'w-6 h-6', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
    h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' })
  ]),
  reviewer: () => h('svg', { class: 'w-6 h-6', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' }, [
    h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M15 12a3 3 0 11-6 0 3 3 0 016 0z' })
  ])
}

// Component state
const selectedScenario = ref('feature-development')
const isRunning = ref(false)
const selectedAgent = ref(null)
const feedContent = ref(null)

// Agent data
const agents = ref([
  {
    id: 'arch-1',
    name: 'Alice',
    type: 'architect',
    status: 'idle',
    progress: 0,
    currentTask: null,
    capabilities: ['system_design', 'api_design', 'architecture_review'],
    successRate: 94,
    avgDuration: 45,
    tasksCompleted: 127,
    recentTasks: [
      { id: 1, name: 'Design user authentication', status: 'completed' },
      { id: 2, name: 'Review API specifications', status: 'completed' },
      { id: 3, name: 'Architecture planning', status: 'in_progress' }
    ]
  },
  {
    id: 'dev-1',
    name: 'Bob',
    type: 'developer',
    status: 'idle',
    progress: 0,
    currentTask: null,
    capabilities: ['python', 'javascript', 'database_design', 'api_development'],
    successRate: 91,
    avgDuration: 32,
    tasksCompleted: 203,
    recentTasks: [
      { id: 1, name: 'Implement user registration', status: 'completed' },
      { id: 2, name: 'Fix authentication bug', status: 'completed' },
      { id: 3, name: 'Add password validation', status: 'pending' }
    ]
  },
  {
    id: 'test-1',
    name: 'Carol',
    type: 'tester',
    status: 'idle',
    progress: 0,
    currentTask: null,
    capabilities: ['unit_testing', 'integration_testing', 'performance_testing'],
    successRate: 97,
    avgDuration: 28,
    tasksCompleted: 89,
    recentTasks: [
      { id: 1, name: 'Test login functionality', status: 'completed' },
      { id: 2, name: 'Performance benchmarks', status: 'completed' },
      { id: 3, name: 'Security testing', status: 'in_progress' }
    ]
  },
  {
    id: 'rev-1',
    name: 'David',
    type: 'reviewer',
    status: 'idle',
    progress: 0,
    currentTask: null,
    capabilities: ['code_review', 'security_audit', 'best_practices'],
    successRate: 96,
    avgDuration: 18,
    tasksCompleted: 156,
    recentTasks: [
      { id: 1, name: 'Review authentication code', status: 'completed' },
      { id: 2, name: 'Security audit report', status: 'completed' },
      { id: 3, name: 'Code quality assessment', status: 'pending' }
    ]
  }
])

// Activity feed
const activities = ref([])
let activityCounter = 0
let demoInterval = null

// Computed properties
const activeAgents = computed(() => 
  agents.value.filter(agent => agent.status === 'working').length
)

const completedTasks = computed(() => 
  agents.value.reduce((total, agent) => total + agent.tasksCompleted, 0)
)

// Demo scenarios
const scenarios = {
  'feature-development': {
    name: 'Feature Development',
    steps: [
      { agent: 'arch-1', task: 'Analyzing feature requirements', duration: 3000 },
      { agent: 'arch-1', task: 'Creating system design', duration: 4000 },
      { agent: 'dev-1', task: 'Implementing backend logic', duration: 6000 },
      { agent: 'dev-1', task: 'Creating API endpoints', duration: 4000 },
      { agent: 'test-1', task: 'Writing unit tests', duration: 3000 },
      { agent: 'test-1', task: 'Running integration tests', duration: 2000 },
      { agent: 'rev-1', task: 'Code review and feedback', duration: 3000 }
    ]
  },
  'bug-fix': {
    name: 'Bug Fix Workflow',
    steps: [
      { agent: 'test-1', task: 'Reproducing bug report', duration: 2000 },
      { agent: 'dev-1', task: 'Investigating root cause', duration: 4000 },
      { agent: 'dev-1', task: 'Implementing fix', duration: 3000 },
      { agent: 'test-1', task: 'Verifying bug fix', duration: 2000 },
      { agent: 'rev-1', task: 'Reviewing fix implementation', duration: 2000 }
    ]
  }
}

// Methods
const getAgentIcon = (type) => AgentIcons[type] || AgentIcons.developer

const loadScenario = () => {
  if (isRunning.value) return
  // Reset agent states
  agents.value.forEach(agent => {
    agent.status = 'idle'
    agent.progress = 0
    agent.currentTask = null
  })
  activities.value = []
}

const startDemo = async () => {
  if (isRunning.value) return
  
  isRunning.value = true
  const scenario = scenarios[selectedScenario.value]
  
  addActivity('system', 'System', `Starting ${scenario.name} demo`, 'Initializing agents...')
  
  for (const step of scenario.steps) {
    const agent = agents.value.find(a => a.id === step.agent)
    if (!agent) continue
    
    // Start task
    agent.status = 'working'
    agent.currentTask = step.task
    agent.progress = 0
    
    addActivity(agent.type, agent.name, `Started: ${step.task}`, `Estimated duration: ${step.duration/1000}s`)
    
    // Simulate progress
    const progressInterval = setInterval(() => {
      if (agent.progress < 100) {
        agent.progress += Math.random() * 20
        if (agent.progress > 100) agent.progress = 100
      }
    }, step.duration / 10)
    
    // Wait for completion
    await new Promise(resolve => {
      setTimeout(() => {
        clearInterval(progressInterval)
        agent.progress = 100
        agent.status = 'idle'
        agent.currentTask = null
        agent.tasksCompleted++
        
        addActivity(agent.type, agent.name, `Completed: ${step.task}`, 'Task finished successfully')
        resolve()
      }, step.duration)
    })
  }
  
  addActivity('system', 'System', `${scenario.name} completed successfully!`, 'All agents returned to idle state')
  isRunning.value = false
}

const selectAgent = (agent) => {
  selectedAgent.value = agent
}

const addActivity = (type, agentName, message, details = null) => {
  const activity = {
    id: activityCounter++,
    timestamp: new Date(),
    type,
    agentType: type,
    agentName,
    message,
    details
  }
  
  activities.value.unshift(activity)
  
  // Keep only last 20 activities
  if (activities.value.length > 20) {
    activities.value.pop()
  }
  
  // Auto-scroll to top
  setTimeout(() => {
    if (feedContent.value) {
      feedContent.value.scrollTop = 0
    }
  }, 100)
}

const clearActivity = () => {
  activities.value = []
}

const formatTime = (timestamp) => {
  return timestamp.toLocaleTimeString('en-US', { 
    hour12: false, 
    hour: '2-digit', 
    minute: '2-digit', 
    second: '2-digit' 
  })
}

// Lifecycle
onMounted(() => {
  // Add some initial activity
  addActivity('system', 'System', 'Demo environment initialized', 'Ready for multi-agent coordination')
})

onUnmounted(() => {
  if (demoInterval) {
    clearInterval(demoInterval)
  }
})
</script>

<style scoped>
.agent-demo {
  margin: 2rem 0;
  border-radius: 12px;
  overflow: hidden;
}

.demo-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: rgba(16, 185, 129, 0.1);
  border-bottom: 1px solid rgba(16, 185, 129, 0.2);
}

.demo-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.demo-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.scenario-selector {
  padding: 0.5rem;
  border: 1px solid var(--vp-c-border);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.875rem;
}

.start-button {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--lv-gradient-secondary);
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.start-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.start-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.demo-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: var(--vp-c-border);
}

.agent-network, .activity-feed {
  background: var(--vp-c-bg);
  padding: 1.5rem;
}

.network-header, .feed-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.network-header h4, .feed-header h4 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.network-stats {
  display: flex;
  gap: 1rem;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--lv-primary);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.agent-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.agent-node {
  padding: 1rem;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
}

.agent-node:hover {
  border-color: var(--lv-primary);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

.agent-node.status-working {
  border-color: var(--lv-secondary);
  background: rgba(16, 185, 129, 0.05);
}

.agent-avatar {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 3rem;
  height: 3rem;
  background: var(--lv-gradient-primary);
  color: white;
  border-radius: 50%;
  margin-bottom: 0.75rem;
}

.agent-info {
  text-align: center;
  margin-bottom: 0.5rem;
}

.agent-name {
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.agent-type {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  text-transform: capitalize;
}

.agent-status {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.agent-task {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  text-align: center;
  margin-bottom: 0.5rem;
  min-height: 2.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.agent-progress {
  height: 4px;
  background: var(--vp-c-bg-soft);
  border-radius: 2px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: var(--lv-gradient-secondary);
  transition: width 0.3s ease;
}

.feed-content {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  padding: 1rem;
}

.activity-item {
  padding: 0.75rem;
  border-left: 3px solid var(--vp-c-border);
  margin-bottom: 0.75rem;
  border-radius: 0 6px 6px 0;
  background: var(--vp-c-bg-soft);
}

.activity-item.type-system {
  border-left-color: var(--lv-accent);
}

.activity-item.type-architect {
  border-left-color: var(--lv-primary);
}

.activity-item.type-developer {
  border-left-color: var(--lv-secondary);
}

.activity-item.type-tester {
  border-left-color: #f59e0b;
}

.activity-item.type-reviewer {
  border-left-color: #8b5cf6;
}

.activity-timestamp {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  margin-bottom: 0.25rem;
}

.activity-agent {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 0.25rem;
}

.activity-message {
  color: var(--vp-c-text-2);
  margin-bottom: 0.25rem;
}

.activity-details {
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
  font-style: italic;
}

.clear-button {
  padding: 0.25rem 0.5rem;
  background: transparent;
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  color: var(--vp-c-text-2);
  cursor: pointer;
  font-size: 0.875rem;
}

.clear-button:hover {
  background: var(--vp-c-bg-soft);
}

.agent-details {
  grid-column: 1 / -1;
  background: var(--vp-c-bg);
  border-top: 1px solid var(--vp-c-border);
  padding: 1.5rem;
}

.details-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.details-header h4 {
  margin: 0;
  color: var(--vp-c-text-1);
}

.close-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  background: transparent;
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  color: var(--vp-c-text-2);
  cursor: pointer;
}

.close-button:hover {
  background: var(--vp-c-bg-soft);
}

.details-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
}

.detail-section h5 {
  margin: 0 0 0.75rem 0;
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--vp-c-text-2);
}

.capability-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.capability-tag {
  padding: 0.25rem 0.5rem;
  background: var(--lv-gradient-primary);
  color: white;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
}

.metric {
  text-align: center;
  padding: 1rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
}

.metric-label {
  display: block;
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
  margin-bottom: 0.5rem;
}

.metric-value {
  display: block;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--lv-primary);
}

.recent-tasks {
  space-y: 0.5rem;
}

.task-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 6px;
  margin-bottom: 0.5rem;
}

.task-name {
  color: var(--vp-c-text-1);
}

.task-status {
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
}

.task-status.status-completed {
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
}

.task-status.status-in_progress {
  background: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}

.task-status.status-pending {
  background: rgba(107, 114, 128, 0.1);
  color: #6b7280;
}

.demo-footer {
  padding: 1rem 1.5rem;
  background: rgba(0, 0, 0, 0.02);
  border-top: 1px solid var(--vp-c-border);
}

.demo-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.info-link {
  color: var(--lv-primary);
  text-decoration: none;
  font-weight: 500;
}

.info-link:hover {
  text-decoration: underline;
}

/* Animations */
.activity-enter-active, .activity-leave-active {
  transition: all 0.3s ease;
}

.activity-enter-from {
  opacity: 0;
  transform: translateY(-10px);
}

.activity-leave-to {
  opacity: 0;
  transform: translateX(10px);
}

/* Responsive design */
@media (max-width: 768px) {
  .demo-content {
    grid-template-columns: 1fr;
  }
  
  .demo-header {
    flex-direction: column;
    gap: 1rem;
    align-items: stretch;
  }
  
  .demo-controls {
    justify-content: space-between;
  }
  
  .agent-grid {
    grid-template-columns: 1fr;
  }
  
  .details-content {
    grid-template-columns: 1fr;
  }
}
</style>