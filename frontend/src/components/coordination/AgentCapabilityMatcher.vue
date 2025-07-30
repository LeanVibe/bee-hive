<template>
  <div class="agent-capability-matcher">
    <!-- Header -->
    <div class="matcher-header mb-6">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 class="text-2xl font-bold text-slate-900 dark:text-white">
            Agent Capability Matcher
          </h2>
          <p class="text-slate-600 dark:text-slate-400 mt-1">
            Intelligent matching of agents to task requirements with visual capability analysis
          </p>
        </div>
        <div class="flex items-center space-x-3 mt-4 sm:mt-0">
          <!-- View Mode Toggle -->
          <div class="view-toggle bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
            <button
              @click="viewMode = 'grid'"
              :class="viewMode === 'grid' ? 'btn-toggle-active' : 'btn-toggle'"
            >
              <Squares2X2Icon class="w-4 h-4" />
            </button>
            <button
              @click="viewMode = 'matrix'"
              :class="viewMode === 'matrix' ? 'btn-toggle-active' : 'btn-toggle'"
            >
              <TableCellsIcon class="w-4 h-4" />
            </button>
            <button
              @click="viewMode = 'radar'"
              :class="viewMode === 'radar' ? 'btn-toggle-active' : 'btn-toggle'"
            >
              <ChartPieIcon class="w-4 h-4" />
            </button>
          </div>
          
          <!-- Refresh Button -->
          <button
            @click="refreshData"
            :disabled="loading"
            class="btn-secondary flex items-center"
          >
            <ArrowPathIcon 
              class="w-4 h-4 mr-2" 
              :class="{ 'animate-spin': loading }"
            />
            Refresh
          </button>
        </div>
      </div>
    </div>

    <!-- Task Requirements Panel -->
    <div class="requirements-panel glass-card rounded-xl p-6 mb-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Task Requirements
        </h3>
        <button
          @click="showRequirementsEditor"
          class="btn-primary text-sm"
        >
          <PencilIcon class="w-4 h-4 mr-1" />
          Edit Requirements
        </button>
      </div>

      <div v-if="taskRequirements" class="requirements-content">
        <div class="flex items-start justify-between mb-4">
          <div class="flex-1">
            <h4 class="font-medium text-slate-900 dark:text-white mb-1">
              {{ taskRequirements.title }}
            </h4>
            <p class="text-sm text-slate-600 dark:text-slate-400">
              {{ taskRequirements.description }}
            </p>
          </div>
          <span 
            class="priority-badge px-2 py-1 text-xs rounded-full font-medium"
            :class="getPriorityClass(taskRequirements.priority)"
          >
            {{ taskRequirements.priority }}
          </span>
        </div>

        <!-- Required Capabilities -->
        <div class="required-capabilities">
          <p class="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Required Capabilities:
          </p>
          <div class="capability-tags flex flex-wrap gap-2">
            <span 
              v-for="capability in taskRequirements.requiredCapabilities"
              :key="capability.name"
              class="capability-tag inline-flex items-center px-3 py-1 text-sm rounded-full"
              :class="getCapabilityImportanceClass(capability.importance)"
            >
              {{ capability.name }}
              <span class="ml-1 text-xs opacity-75">
                ({{ capability.importance }})
              </span>
            </span>
          </div>
        </div>
      </div>

      <!-- No Requirements State -->
      <div 
        v-else
        class="no-requirements text-center py-8 text-slate-500 dark:text-slate-400"
      >
        <ClipboardDocumentIcon class="w-12 h-12 mx-auto mb-3 opacity-50" />
        <p>No task requirements defined</p>
        <p class="text-sm mt-1">Add requirements to start capability matching</p>
      </div>
    </div>

    <!-- Matching Results -->
    <div v-if="taskRequirements" class="matching-results">
      
      <!-- Grid View -->
      <div v-if="viewMode === 'grid'" class="grid-view">
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AgentMatchCard
            v-for="match in sortedMatches"
            :key="match.agent.agent_id"
            :agent="match.agent"
            :match-data="match"
            :task-requirements="taskRequirements"
            @agent-select="selectAgent"
            @assign-task="assignTaskToAgent"
          />
        </div>
      </div>

      <!-- Matrix View -->
      <div v-if="viewMode === 'matrix'" class="matrix-view glass-card rounded-xl p-6">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          Capability Matching Matrix
        </h3>
        
        <div class="matrix-container overflow-x-auto">
          <table class="matrix-table w-full text-sm">
            <thead>
              <tr class="border-b border-slate-200 dark:border-slate-700">
                <th class="matrix-header-cell text-left">Agent</th>
                <th 
                  v-for="capability in taskRequirements.requiredCapabilities"
                  :key="capability.name"
                  class="matrix-header-cell text-center min-w-24"
                >
                  <div class="capability-header">
                    <div class="capability-name">{{ capability.name }}</div>
                    <div class="capability-importance text-xs opacity-75">
                      {{ capability.importance }}
                    </div>
                  </div>
                </th>
                <th class="matrix-header-cell text-center">Overall</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="match in sortedMatches"
                :key="match.agent.agent_id"
                class="matrix-row border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                <td class="matrix-cell">
                  <div class="flex items-center space-x-3">
                    <div 
                      class="agent-avatar w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-medium"
                      :style="{ backgroundColor: getAgentColor(match.agent.agent_id) }"
                    >
                      {{ getAgentInitials(match.agent.name) }}
                    </div>
                    <div>
                      <div class="font-medium text-slate-900 dark:text-white">
                        {{ match.agent.name }}
                      </div>
                      <div class="text-xs text-slate-500 dark:text-slate-400">
                        {{ match.agent.type }}
                      </div>
                    </div>
                  </div>
                </td>
                <td 
                  v-for="capability in taskRequirements.requiredCapabilities"
                  :key="capability.name"
                  class="matrix-cell text-center"
                >
                  <CapabilityMatchIndicator
                    :match-score="getCapabilityMatch(match, capability.name)"
                    :confidence="getCapabilityConfidence(match, capability.name)"
                  />
                </td>
                <td class="matrix-cell text-center">
                  <div class="overall-score flex items-center justify-center space-x-2">
                    <div 
                      class="score-circle w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white"
                      :class="getOverallScoreColor(match.overallMatch)"
                    >
                      {{ Math.round(match.overallMatch * 100) }}
                    </div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Radar Chart View -->
      <div v-if="viewMode === 'radar'" class="radar-view">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div
            v-for="match in sortedMatches.slice(0, 4)"
            :key="match.agent.agent_id"
            class="radar-card glass-card rounded-xl p-6"
          >
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center space-x-3">
                <div 
                  class="agent-avatar w-10 h-10 rounded-full flex items-center justify-center text-white font-medium"
                  :style="{ backgroundColor: getAgentColor(match.agent.agent_id) }"
                >
                  {{ getAgentInitials(match.agent.name) }}
                </div>
                <div>
                  <h4 class="font-medium text-slate-900 dark:text-white">
                    {{ match.agent.name }}
                  </h4>
                  <p class="text-sm text-slate-600 dark:text-slate-400">
                    {{ Math.round(match.overallMatch * 100) }}% match
                  </p>
                </div>
              </div>
              <button
                @click="assignTaskToAgent(match.agent, match)"
                class="btn-primary text-sm"
              >
                Assign
              </button>
            </div>
            
            <!-- Radar Chart Container -->
            <div class="radar-chart-container h-64">
              <canvas 
                :ref="`radarChart-${match.agent.agent_id}`"
                class="w-full h-full"
              ></canvas>
            </div>
          </div>
        </div>
      </div>

      <!-- No Matches State -->
      <div 
        v-if="!sortedMatches.length && !loading"
        class="no-matches text-center py-12 text-slate-500 dark:text-slate-400"
      >
        <UserGroupIcon class="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p class="text-lg font-medium">No suitable agents found</p>
        <p class="text-sm mt-1">Try adjusting the task requirements or register more agents</p>
      </div>
    </div>

    <!-- Requirements Editor Modal -->
    <RequirementsEditorModal
      v-if="showRequirementsModal"
      :requirements="taskRequirements"
      @close="showRequirementsModal = false"
      @save="updateRequirements"
    />

    <!-- Agent Selection Modal -->
    <AgentSelectionModal
      v-if="showAgentModal"
      :agent="selectedAgent"
      :match-data="selectedMatch"
      :task-requirements="taskRequirements"
      @close="showAgentModal = false"
      @assign="confirmAssignment"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { Chart, registerables } from 'chart.js'

// Register Chart.js components
Chart.register(...registerables)

// Icons
import {
  Squares2X2Icon,
  TableCellsIcon,
  ChartPieIcon,
  ArrowPathIcon,
  PencilIcon,
  ClipboardDocumentIcon,
  UserGroupIcon
} from '@heroicons/vue/24/outline'

// Components
import AgentMatchCard from './AgentMatchCard.vue'
import CapabilityMatchIndicator from './CapabilityMatchIndicator.vue'
import RequirementsEditorModal from './RequirementsEditorModal.vue'
import AgentSelectionModal from './AgentSelectionModal.vue'

// Services and composables
import { useSessionColors } from '@/utils/SessionColorManager'
import { api } from '@/services/api'

// Types
interface TaskRequirements {
  title: string
  description: string
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  requiredCapabilities: Array<{
    name: string
    importance: 'critical' | 'high' | 'medium' | 'low'
    minConfidence?: number
  }>
  estimatedEffort?: number
  deadline?: string
}

interface Agent {
  agent_id: string
  name: string
  type: string
  status: string
  capabilities: Array<{
    name: string
    confidence_level: number
    description?: string
  }>
  performance_score: number
  current_workload: number
}

interface AgentMatch {
  agent: Agent
  overallMatch: number
  capabilityMatches: Record<string, {
    score: number
    confidence: number
    hasCapability: boolean
  }>
  reasoning: string[]
  recommendedScore: number
}

const { getAgentColor } = useSessionColors()

// State
const loading = ref(false)
const viewMode = ref<'grid' | 'matrix' | 'radar'>('grid')
const showRequirementsModal = ref(false)
const showAgentModal = ref(false)
const selectedAgent = ref<Agent | null>(null)
const selectedMatch = ref<AgentMatch | null>(null)

// Data
const taskRequirements = ref<TaskRequirements | null>(null)
const agents = ref<Agent[]>([])
const matches = ref<AgentMatch[]>([])

// Chart instances
const radarCharts = new Map<string, Chart>()

// Computed
const sortedMatches = computed(() => {
  return matches.value
    .sort((a, b) => b.overallMatch - a.overallMatch)
    .slice(0, 20) // Limit to top 20 matches
})

// Methods
const refreshData = async () => {
  loading.value = true
  
  try {
    await Promise.all([
      loadAgents(),
      calculateMatches()
    ])
  } catch (error) {
    console.error('Failed to refresh data:', error)
  } finally {
    loading.value = false
  }
}

const loadAgents = async () => {
  try {
    const response = await api.get('/team-coordination/agents')
    agents.value = response.data || []
  } catch (error) {
    console.error('Failed to load agents:', error)
  }
}

const calculateMatches = async () => {
  if (!taskRequirements.value || !agents.value.length) {
    matches.value = []
    return
  }

  const agentMatches: AgentMatch[] = []

  for (const agent of agents.value) {
    if (agent.status === 'sleeping' || agent.current_workload >= 1.0) {
      continue // Skip unavailable agents
    }

    const match = calculateAgentMatch(agent, taskRequirements.value)
    agentMatches.push(match)
  }

  matches.value = agentMatches
  
  // Update radar charts if in radar view
  if (viewMode.value === 'radar') {
    await nextTick()
    updateRadarCharts()
  }
}

const calculateAgentMatch = (agent: Agent, requirements: TaskRequirements): AgentMatch => {
  const capabilityMatches: Record<string, any> = {}
  let totalScore = 0
  let totalWeightedScore = 0
  let totalWeight = 0
  const reasoning: string[] = []

  // Calculate individual capability matches
  for (const reqCapability of requirements.requiredCapabilities) {
    const weight = getCapabilityWeight(reqCapability.importance)
    totalWeight += weight

    // Find matching agent capabilities
    const agentCapabilities = agent.capabilities.filter(ac => 
      ac.name.toLowerCase().includes(reqCapability.name.toLowerCase()) ||
      reqCapability.name.toLowerCase().includes(ac.name.toLowerCase())
    )

    let bestMatch = { score: 0, confidence: 0, hasCapability: false }

    if (agentCapabilities.length > 0) {
      // Use the best matching capability
      const bestCapability = agentCapabilities.reduce((best, current) => 
        current.confidence_level > best.confidence_level ? current : best
      )

      bestMatch = {
        score: bestCapability.confidence_level,
        confidence: bestCapability.confidence_level,
        hasCapability: true
      }

      if (bestCapability.confidence_level >= 0.8) {
        reasoning.push(`Strong ${reqCapability.name} expertise`)
      } else if (bestCapability.confidence_level >= 0.6) {
        reasoning.push(`Good ${reqCapability.name} knowledge`)
      } else {
        reasoning.push(`Basic ${reqCapability.name} familiarity`)
      }
    } else {
      reasoning.push(`No ${reqCapability.name} experience`)
    }

    capabilityMatches[reqCapability.name] = bestMatch
    totalScore += bestMatch.score
    totalWeightedScore += bestMatch.score * weight
  }

  // Calculate overall match score
  const averageScore = totalScore / requirements.requiredCapabilities.length
  const weightedScore = totalWeightedScore / totalWeight

  // Apply bonuses/penalties
  let finalScore = (averageScore + weightedScore) / 2

  // Performance bonus
  if (agent.performance_score > 0.8) {
    finalScore *= 1.1
    reasoning.push('High performance history')
  } else if (agent.performance_score < 0.5) {
    finalScore *= 0.9
    reasoning.push('Performance concerns')
  }

  // Workload penalty
  if (agent.current_workload > 0.8) {
    finalScore *= 0.8
    reasoning.push('High current workload')
  } else if (agent.current_workload < 0.3) {
    finalScore *= 1.05
    reasoning.push('Available capacity')
  }

  return {
    agent,
    overallMatch: Math.min(finalScore, 1),
    capabilityMatches,
    reasoning: reasoning.slice(0, 3), // Keep top 3 reasons
    recommendedScore: finalScore >= 0.7 ? 1 : finalScore >= 0.5 ? 0.7 : 0.3
  }
}

const updateRadarCharts = () => {
  // Clear existing charts
  radarCharts.forEach(chart => chart.destroy())
  radarCharts.clear()

  if (!taskRequirements.value) return

  // Create radar charts for top matches
  for (const match of sortedMatches.value.slice(0, 4)) {
    const canvas = document.querySelector(`[data-ref="radarChart-${match.agent.agent_id}"]`) as HTMLCanvasElement
    if (canvas) {
      createRadarChart(canvas, match)
    }
  }
}

const createRadarChart = (canvas: HTMLCanvasElement, match: AgentMatch) => {
  const ctx = canvas.getContext('2d')
  if (!ctx || !taskRequirements.value) return

  const capabilities = taskRequirements.value.requiredCapabilities
  const data = capabilities.map(cap => 
    match.capabilityMatches[cap.name]?.score * 100 || 0
  )

  const chart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: capabilities.map(cap => cap.name),
      datasets: [{
        label: match.agent.name,
        data: data,
        backgroundColor: getAgentColor(match.agent.agent_id) + '20',
        borderColor: getAgentColor(match.agent.agent_id),
        borderWidth: 2,
        pointBackgroundColor: getAgentColor(match.agent.agent_id),
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          grid: {
            color: 'rgba(148, 163, 184, 0.2)'
          },
          angleLines: {
            color: 'rgba(148, 163, 184, 0.2)'
          },
          pointLabels: {
            font: {
              size: 10
            },
            color: 'rgb(100, 116, 139)'
          },
          ticks: {
            display: false
          }
        }
      }
    }
  })

  radarCharts.set(match.agent.agent_id, chart)
}

const showRequirementsEditor = () => {
  showRequirementsModal.value = true
}

const updateRequirements = (requirements: TaskRequirements) => {
  taskRequirements.value = requirements
  showRequirementsModal.value = false
  calculateMatches()
}

const selectAgent = (agent: Agent, matchData: AgentMatch) => {
  selectedAgent.value = agent
  selectedMatch.value = matchData
  showAgentModal.value = true
}

const assignTaskToAgent = (agent: Agent, matchData: AgentMatch) => {
  selectAgent(agent, matchData)
}

const confirmAssignment = async (assignmentData: any) => {
  try {
    // Call the team coordination API to assign the task
    await api.post('/team-coordination/tasks/distribute', {
      ...assignmentData,
      target_agent_id: selectedAgent.value?.agent_id
    })

    showAgentModal.value = false
    selectedAgent.value = null
    selectedMatch.value = null

    // Refresh data to reflect the assignment
    await refreshData()
  } catch (error) {
    console.error('Failed to assign task:', error)
  }
}

// Utility functions
const getCapabilityWeight = (importance: string): number => {
  const weights = {
    critical: 3,
    high: 2,
    medium: 1.5,
    low: 1
  }
  return weights[importance] || 1
}

const getCapabilityMatch = (match: AgentMatch, capabilityName: string): number => {
  return match.capabilityMatches[capabilityName]?.score || 0
}

const getCapabilityConfidence = (match: AgentMatch, capabilityName: string): number => {
  return match.capabilityMatches[capabilityName]?.confidence || 0
}

const getAgentInitials = (name: string): string => {
  return name
    .split(' ')
    .map(n => n.charAt(0))
    .join('')
    .toUpperCase()
    .slice(0, 2)
}

const getPriorityClass = (priority: string): string => {
  const classes = {
    'CRITICAL': 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400',
    'HIGH': 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400',
    'MEDIUM': 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400',
    'LOW': 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
  }
  return classes[priority] || classes.MEDIUM
}

const getCapabilityImportanceClass = (importance: string): string => {
  const classes = {
    critical: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800',
    high: 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 border border-orange-200 dark:border-orange-800',
    medium: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-800',
    low: 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-600'
  }
  return classes[importance] || classes.medium
}

const getOverallScoreColor = (score: number): string => {
  if (score >= 0.8) return 'bg-green-500'
  if (score >= 0.6) return 'bg-blue-500'
  if (score >= 0.4) return 'bg-yellow-500'
  return 'bg-red-500'
}

// Initialize with sample requirements
onMounted(() => {
  // Sample task requirements for demonstration
  taskRequirements.value = {
    title: 'Frontend Dashboard Enhancement',
    description: 'Enhance the coordination dashboard with real-time updates and improved UX',
    priority: 'HIGH',
    requiredCapabilities: [
      { name: 'Vue.js', importance: 'critical' },
      { name: 'TypeScript', importance: 'high' },
      { name: 'WebSocket', importance: 'medium' },
      { name: 'UI/UX Design', importance: 'medium' },
      { name: 'Chart.js', importance: 'low' }
    ],
    estimatedEffort: 16,
    deadline: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString()
  }

  refreshData()
})

onUnmounted(() => {
  // Clean up radar charts
  radarCharts.forEach(chart => chart.destroy())
  radarCharts.clear()
})
</script>

<style scoped>
.agent-capability-matcher {
  @apply min-h-screen bg-slate-50 dark:bg-slate-900;
}

.glass-card {
  @apply bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50;
}

.btn-toggle {
  @apply px-3 py-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors rounded-md;
}

.btn-toggle-active {
  @apply px-3 py-2 bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-sm rounded-md;
}

.btn-primary {
  @apply bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.btn-secondary {
  @apply bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 px-4 py-2 rounded-md font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.capability-tag {
  @apply transition-all duration-150 hover:scale-105;
}

.matrix-table {
  @apply border-collapse;
}

.matrix-header-cell {
  @apply px-4 py-3 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider;
}

.matrix-cell {
  @apply px-4 py-3;
}

.matrix-row {
  @apply transition-colors duration-150;
}

.capability-header {
  @apply text-center;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.radar-card,
.matrix-view {
  animation: slideIn 0.3s ease-out;
}
</style>