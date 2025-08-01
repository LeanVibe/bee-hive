<template>
  <div class="feature-showcase">
    <div class="showcase-header">
      <h2>Experience Autonomous Development</h2>
      <p>Interactive demonstrations of our advanced AI coordination capabilities</p>
    </div>
    
    <div class="feature-tabs">
      <button 
        v-for="feature in features" 
        :key="feature.id"
        :class="['tab-button', { active: activeFeature === feature.id }]"
        @click="setActiveFeature(feature.id)"
      >
        <span class="tab-icon">{{ feature.icon }}</span>
        <span class="tab-label">{{ feature.name }}</span>
      </button>
    </div>
    
    <Transition name="feature" mode="out-in">
      <div class="feature-content" :key="activeFeature">
        <div class="feature-demo">
          <div class="demo-visualization">
            <component :is="currentFeature.component" />
          </div>
          
          <div class="demo-description">
            <h3>{{ currentFeature.name }}</h3>
            <p>{{ currentFeature.description }}</p>
            
            <div class="feature-highlights">
              <div 
                v-for="highlight in currentFeature.highlights" 
                :key="highlight"
                class="highlight-item"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
                {{ highlight }}
              </div>
            </div>
            
            <div class="feature-actions">
              <a :href="currentFeature.learnMore" class="action-button primary">
                Learn More
              </a>
              <a :href="currentFeature.tryNow" class="action-button secondary">
                Try It Now
              </a>
            </div>
          </div>
        </div>
        
        <div class="feature-stats">
          <div 
            v-for="stat in currentFeature.stats" 
            :key="stat.label"
            class="stat-item"
          >
            <div class="stat-value">{{ stat.value }}</div>
            <div class="stat-label">{{ stat.label }}</div>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

// Demo components for each feature
const MultiAgentVisualization = {
  template: `
    <div class="agent-visualization">
      <div class="coordination-flow">
        <div v-for="agent in agents" :key="agent.id" :class="['agent-node', agent.status]">
          <div class="agent-avatar">{{ agent.icon }}</div>
          <div class="agent-label">{{ agent.name }}</div>
          <div class="agent-task">{{ agent.currentTask }}</div>
          <div class="progress-ring">
            <circle :stroke-dasharray="'${agent.progress} 100'" />
          </div>
        </div>
        
        <!-- Connection lines between agents -->
        <svg class="connection-lines" viewBox="0 0 400 200">
          <path d="M100,50 Q200,25 300,50" stroke="var(--lv-primary)" stroke-width="2" fill="none" opacity="0.6" />
          <path d="M100,150 Q200,175 300,150" stroke="var(--lv-primary)" stroke-width="2" fill="none" opacity="0.6" />
          <path d="M50,100 L350,100" stroke="var(--lv-primary)" stroke-width="2" fill="none" opacity="0.4" />
        </svg>
      </div>
      
      <div class="coordination-status">
        <div class="status-item">
          <span class="status-dot active"></span>
          <span>Real-time coordination active</span>
        </div>
        <div class="status-item">
          <span class="status-dot"></span>
          <span>Conflict resolution: Auto</span>
        </div>
        <div class="status-item">
          <span class="status-dot"></span>
          <span>Load balancing: Optimal</span>
        </div>
      </div>
    </div>
  `,
  data() {
    return {
      agents: [
        { id: 1, name: 'Alice', icon: '🏗️', status: 'active', currentTask: 'Designing API', progress: 75 },
        { id: 2, name: 'Bob', icon: '💻', status: 'active', currentTask: 'Writing code', progress: 45 },
        { id: 3, name: 'Carol', icon: '🧪', status: 'waiting', currentTask: 'Preparing tests', progress: 20 },
        { id: 4, name: 'David', icon: '👁️', status: 'idle', currentTask: 'Ready for review', progress: 0 }
      ]
    }
  },
  mounted() {
    // Simulate progress updates
    setInterval(() => {
      this.agents.forEach(agent => {
        if (agent.status === 'active' && agent.progress < 100) {
          agent.progress += Math.random() * 5
          if (agent.progress >= 100) {
            agent.progress = 100
            agent.status = 'completed'
          }
        }
      })
    }, 2000)
  }
}

const CommandVisualization = {
  template: `
    <div class="command-visualization">
      <div class="command-editor">
        <div class="editor-header">
          <span class="editor-title">Custom Command Editor</span>
          <span class="editor-status">✅ Syntax Valid</span>
        </div>
        <pre class="yaml-content">{{ yamlCommand }}</pre>
      </div>
      
      <div class="execution-flow">
        <div class="flow-header">Execution Pipeline</div>
        <div class="pipeline-steps">
          <div v-for="step in executionSteps" :key="step.id" :class="['pipeline-step', step.status]">
            <div class="step-icon">{{ step.icon }}</div>
            <div class="step-content">
              <div class="step-name">{{ step.name }}</div>
              <div class="step-description">{{ step.description }}</div>
            </div>
            <div class="step-status">{{ step.statusText }}</div>
          </div>
        </div>
      </div>
    </div>
  `,
  data() {
    return {
      yamlCommand: \`name: feature-development
description: Complete feature implementation
agents:
  - architect
  - developer  
  - tester
steps:
  - name: design
    agent: architect
    action: create_design
  - name: implement
    agent: developer
    action: write_code
    depends_on: [design]
  - name: test
    agent: tester
    action: run_tests
    depends_on: [implement]\`,
      executionSteps: [
        { id: 1, name: 'Validate Command', description: 'Syntax and schema validation', icon: '✓', status: 'completed', statusText: 'Passed' },
        { id: 2, name: 'Agent Assignment', description: 'Assign tasks to available agents', icon: '🤖', status: 'completed', statusText: 'Done' },
        { id: 3, name: 'Execute Steps', description: 'Run command steps in sequence', icon: '⚡', status: 'active', statusText: 'Running' },
        { id: 4, name: 'Monitor Progress', description: 'Track execution and handle conflicts', icon: '📊', status: 'pending', statusText: 'Waiting' }
      ]
    }
  }
}

const ContextVisualization = {
  template: `
    <div class="context-visualization">
      <div class="memory-graph">
        <div class="graph-header">Context Memory Network</div>
        <div class="memory-nodes">
          <div v-for="node in memoryNodes" :key="node.id" :class="['memory-node', node.type]">
            <div class="node-content">
              <div class="node-icon">{{ node.icon }}</div>
              <div class="node-label">{{ node.label }}</div>
              <div class="node-connections">{{ node.connections }} connections</div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="learning-progress">
        <div class="progress-header">Learning Insights</div>
        <div class="insights-list">
          <div v-for="insight in learningInsights" :key="insight.id" class="insight-item">
            <div class="insight-icon">{{ insight.icon }}</div>
            <div class="insight-content">
              <div class="insight-text">{{ insight.text }}</div>
              <div class="insight-confidence">Confidence: {{ insight.confidence }}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  data() {
    return {
      memoryNodes: [
        { id: 1, type: 'pattern', icon: '🧠', label: 'Code Patterns', connections: 23 },
        { id: 2, type: 'preference', icon: '⚙️', label: 'User Preferences', connections: 15 },
        { id: 3, type: 'context', icon: '📝', label: 'Project Context', connections: 31 },
        { id: 4, type: 'error', icon: '🐛', label: 'Error Solutions', connections: 12 }
      ],
      learningInsights: [
        { id: 1, icon: '🎯', text: 'Prefers functional programming style', confidence: 94 },
        { id: 2, icon: '🔧', text: 'Uses TypeScript for new projects', confidence: 87 },
        { id: 3, icon: '📊', text: 'Favors comprehensive testing coverage', confidence: 91 },
        { id: 4, icon: '🚀', text: 'Values performance optimization', confidence: 78 }
      ]
    }
  }
}

// Feature definitions
const features = ref([
  {
    id: 'coordination',
    name: 'Multi-Agent Coordination',
    icon: '🤖',
    description: 'Watch specialized AI agents collaborate seamlessly on complex development tasks with intelligent task distribution and real-time conflict resolution.',
    component: MultiAgentVisualization,
    highlights: [
      'Real-time agent coordination',
      'Automatic conflict resolution', 
      'Intelligent load balancing',
      'Progress synchronization'
    ],
    stats: [
      { label: 'Agent Types', value: '8+' },
      { label: 'Coordination Efficiency', value: '94%' },
      { label: 'Conflict Resolution', value: '<500ms' }
    ],
    learnMore: '/learn/fundamentals/coordination',
    tryNow: '/demo'
  },
  {
    id: 'commands',
    name: 'Custom Commands',
    icon: '⚡',
    description: 'Create sophisticated multi-step workflows with declarative YAML definitions. Eight advanced commands handle every development scenario.',
    component: CommandVisualization,
    highlights: [
      'Declarative YAML syntax',
      'Complex dependency management',
      'Real-time execution monitoring',
      'Failure recovery and retry'
    ],
    stats: [
      { label: 'Available Commands', value: '8' },
      { label: 'Lines of Code', value: '5,365' },
      { label: 'Success Rate', value: '96.7%' }
    ],
    learnMore: '/api/commands/',
    tryNow: '/learn/fundamentals/commands'
  },
  {
    id: 'memory',
    name: 'Context Memory',
    icon: '🧠',
    description: 'Agents learn and remember your project patterns, coding styles, and preferences for increasingly intelligent automation over time.',
    component: ContextVisualization,
    highlights: [
      'Persistent learning memory',
      'Pattern recognition',
      'Preference adaptation', 
      'Context-aware suggestions'
    ],
    stats: [
      { label: 'Memory Nodes', value: '1000+' },
      { label: 'Learning Accuracy', value: '91%' },
      { label: 'Context Retention', value: '30 days' }
    ],
    learnMore: '/learn/fundamentals/memory',
    tryNow: '/api/observability/'
  }
])

// Component state
const activeFeature = ref('coordination')

// Computed properties
const currentFeature = computed(() => 
  features.value.find(f => f.id === activeFeature.value) || features.value[0]
)

// Methods
const setActiveFeature = (featureId: string) => {
  activeFeature.value = featureId
}
</script>

<style scoped>
.feature-showcase {
  margin: 3rem 0;
  padding: 2rem;
  background: var(--lv-glass-bg);
  backdrop-filter: blur(10px);
  border: 1px solid var(--lv-glass-border);
  border-radius: 16px;
  box-shadow: var(--lv-shadow-xl);
}

.showcase-header {
  text-align: center;
  margin-bottom: 2rem;
}

.showcase-header h2 {
  margin: 0 0 0.5rem 0;
  font-size: 2rem;
  font-weight: 700;
  background: var(--lv-gradient-primary);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.showcase-header p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 1.1rem;
}

.feature-tabs {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--vp-c-border);
}

.tab-button {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem 1.5rem;
  background: transparent;
  border: none;
  border-bottom: 3px solid transparent;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;
}

.tab-button:hover {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
}

.tab-button.active {
  color: var(--lv-primary);
  border-bottom-color: var(--lv-primary);
  background: rgba(99, 102, 241, 0.05);
}

.tab-icon {
  font-size: 1.2rem;
}

.tab-label {
  font-size: 0.95rem;
}

.feature-content {
  min-height: 400px;
}

.feature-demo {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  align-items: start;
  margin-bottom: 2rem;
}

.demo-visualization {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 1.5rem;
  min-height: 300px;
}

.demo-description h3 {
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-1);
  font-size: 1.5rem;
}

.demo-description p {
  margin: 0 0 1.5rem 0;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.feature-highlights {
  margin-bottom: 2rem;
}

.highlight-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
  color: var(--vp-c-text-2);
}

.highlight-item svg {
  color: var(--lv-secondary);
  flex-shrink: 0;
}

.feature-actions {
  display: flex;
  gap: 1rem;
}

.action-button {
  padding: 0.75rem 1.5rem;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.2s ease;
  display: inline-block;
}

.action-button.primary {
  background: var(--lv-gradient-primary);
  color: white;
}

.action-button.secondary {
  background: transparent;
  color: var(--lv-primary);
  border: 2px solid var(--lv-primary);
}

.action-button:hover {
  transform: translateY(-2px);
  box-shadow: var(--lv-shadow-lg);
}

.feature-stats {
  display: flex;
  justify-content: center;
  gap: 2rem;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
}

.stat-item {
  text-align: center;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--lv-primary);
  margin-bottom: 0.25rem;
}

.stat-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

/* Visualization specific styles */
.agent-visualization {
  text-align: center;
}

.coordination-flow {
  position: relative;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.agent-node {
  padding: 1rem;
  border-radius: 8px;
  border: 2px solid var(--vp-c-border);
  transition: all 0.3s ease;
}

.agent-node.active {
  border-color: var(--lv-secondary);
  background: rgba(16, 185, 129, 0.05);
}

.agent-node.completed {
  border-color: var(--lv-primary);
  background: rgba(99, 102, 241, 0.05);
}

.agent-avatar {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}

.agent-label {
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 0.25rem;
}

.agent-task {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.coordination-status {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--vp-c-border);
}

.status-dot.active {
  background: var(--lv-secondary);
  animation: pulse 2s infinite;
}

/* Transitions */
.feature-enter-active, .feature-leave-active {
  transition: all 0.3s ease;
}

.feature-enter-from, .feature-leave-to {
  opacity: 0;
  transform: translateY(10px);
}

@media (max-width: 1024px) {
  .feature-demo {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
  
  .feature-stats {
    flex-direction: column;
    gap: 1rem;
  }
}

@media (max-width: 768px) {
  .feature-tabs {
    flex-direction: column;
    align-items: stretch;
  }
  
  .tab-button {
    justify-content: center;
    padding: 0.75rem 1rem;
  }
  
  .feature-actions {
    flex-direction: column;
  }
  
  .coordination-flow {
    grid-template-columns: 1fr;
  }
}
</style>