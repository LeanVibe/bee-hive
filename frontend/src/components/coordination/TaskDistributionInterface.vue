<template>
  <div class="task-distribution-interface">
    <!-- Header with Controls -->
    <div class="distribution-header mb-6">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 class="text-2xl font-bold text-slate-900 dark:text-white">
            Task Distribution Center
          </h2>
          <p class="text-slate-600 dark:text-slate-400 mt-1">
            Intelligently assign tasks to optimal agents with drag-and-drop simplicity
          </p>
        </div>
        <div class="flex items-center space-x-3 mt-4 sm:mt-0">
          <button
            @click="createNewTask"
            class="btn-primary flex items-center"
          >
            <PlusIcon class="w-4 h-4 mr-2" />
            New Task
          </button>
          <button
            @click="refreshAgents"
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

    <!-- Main Distribution Layout -->
    <div class="distribution-layout grid grid-cols-1 xl:grid-cols-3 gap-6">
      
      <!-- Task Queue (Left Panel) -->
      <div class="task-queue-panel glass-card rounded-xl p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Task Queue
          </h3>
          <div class="flex items-center space-x-2">
            <select
              v-model="selectedPriority"
              @change="filterTasks"
              class="input-field text-sm"
            >
              <option value="all">All Priority</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MEDIUM">Medium</option>
              <option value="LOW">Low</option>
            </select>
            
            <span 
              class="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full"
            >
              {{ filteredTasks.length }} tasks
            </span>
          </div>
        </div>

        <!-- Task List with Drag-and-Drop -->
        <div 
          ref="taskQueueContainer"
          class="task-queue-container space-y-3 max-h-96 overflow-y-auto"
        >
          <TaskCard
            v-for="task in filteredTasks"
            :key="task.id"
            :task="task"
            :draggable="true"
            :is-dragging="draggedTask?.id === task.id"
            @drag-start="onTaskDragStart"
            @drag-end="onTaskDragEnd"
            @task-click="selectTask"
            @edit-task="editTask"
          />
          
          <!-- Empty State -->
          <div 
            v-if="!filteredTasks.length && !loading"
            class="text-center py-8 text-slate-500 dark:text-slate-400"
          >
            <ClipboardDocumentListIcon class="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No tasks in queue</p>
            <p class="text-sm mt-1">Create a new task to get started</p>
          </div>
          
          <!-- Loading State -->
          <div 
            v-if="loading"
            class="text-center py-8"
          >
            <div class="animate-spin w-6 h-6 border-2 border-primary-600 border-t-transparent rounded-full mx-auto mb-3"></div>
            <p class="text-slate-500 dark:text-slate-400">Loading tasks...</p>
          </div>
        </div>
      </div>

      <!-- Agent Grid (Center Panel) -->
      <div class="agent-grid-panel glass-card rounded-xl p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Active Agents
          </h3>
          <div class="flex items-center space-x-2">
            <select
              v-model="selectedAgentFilter"
              @change="filterAgents"
              class="input-field text-sm"
            >
              <option value="all">All Agents</option>
              <option value="available">Available</option>
              <option value="busy">Busy</option>
              <option value="idle">Idle</option>
            </select>
            
            <span 
              class="px-2 py-1 text-xs bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full"
            >
              {{ filteredAgents.length }} active
            </span>
          </div>
        </div>

        <!-- Agent Cards Grid -->
        <div class="agent-grid grid grid-cols-1 lg:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
          <AgentCard
            v-for="agent in filteredAgents"
            :key="agent.agent_id"
            :agent="agent"
            :drop-target="true"
            :is-drop-target="dropTargetAgent?.agent_id === agent.agent_id"
            :suggested-match="getSuggestedMatch(agent)"
            @drop="onTaskDrop"
            @dragover="onAgentDragOver"
            @dragleave="onAgentDragLeave"
            @agent-click="selectAgent"
          />
          
          <!-- Empty State -->
          <div 
            v-if="!filteredAgents.length && !loading"
            class="col-span-full text-center py-8 text-slate-500 dark:text-slate-400"
          >
            <UserGroupIcon class="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No active agents</p>
            <p class="text-sm mt-1">Register agents to start distributing tasks</p>
          </div>
        </div>
      </div>

      <!-- Assignment Details (Right Panel) -->
      <div class="assignment-details-panel glass-card rounded-xl p-6">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          Assignment Details
        </h3>

        <!-- Selected Task Details -->
        <div v-if="selectedTask" class="selected-task mb-6">
          <h4 class="font-medium text-slate-900 dark:text-white mb-3">Selected Task</h4>
          <div class="bg-slate-50 dark:bg-slate-800 rounded-lg p-4">
            <div class="flex items-start justify-between mb-2">
              <h5 class="font-medium text-sm">{{ selectedTask.task_title }}</h5>
              <span 
                class="px-2 py-1 text-xs rounded-full"
                :class="getPriorityClass(selectedTask.priority)"
              >
                {{ selectedTask.priority }}
              </span>
            </div>
            <p class="text-xs text-slate-600 dark:text-slate-400 mb-3">
              {{ selectedTask.task_description }}
            </p>
            
            <!-- Required Capabilities -->
            <div class="mb-3">
              <p class="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                Required Capabilities:
              </p>
              <div class="flex flex-wrap gap-1">
                <span 
                  v-for="capability in selectedTask.required_capabilities"
                  :key="capability"
                  class="inline-block px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded"
                >
                  {{ capability }}
                </span>
              </div>
            </div>
            
            <!-- Task Metadata -->
            <div class="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span class="text-slate-500 dark:text-slate-400">Effort:</span>
                <span class="ml-1">{{ selectedTask.estimated_effort_hours || 'N/A' }}h</span>
              </div>
              <div>
                <span class="text-slate-500 dark:text-slate-400">Deadline:</span>
                <span class="ml-1">{{ formatDeadline(selectedTask.deadline) }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Smart Matching Suggestions -->
        <div v-if="selectedTask && matchingSuggestions.length" class="matching-suggestions mb-6">
          <h4 class="font-medium text-slate-900 dark:text-white mb-3">Smart Suggestions</h4>
          <div class="space-y-2">
            <div 
              v-for="suggestion in matchingSuggestions"
              :key="suggestion.agent_id"
              class="suggestion-item p-3 bg-gradient-to-r from-primary-50 to-blue-50 dark:from-primary-900/20 dark:to-blue-900/20 rounded-lg border border-primary-200 dark:border-primary-700"
            >
              <div class="flex items-center justify-between mb-2">
                <div class="flex items-center space-x-2">
                  <div 
                    class="w-3 h-3 rounded-full"
                    :style="{ backgroundColor: getAgentColor(suggestion.agent_id) }"
                  ></div>
                  <span class="font-medium text-sm text-slate-900 dark:text-white">
                    {{ suggestion.name }}
                  </span>
                </div>
                <div class="flex items-center space-x-2">
                  <span class="text-xs text-slate-600 dark:text-slate-400">
                    {{ Math.round(suggestion.match_score * 100) }}% match
                  </span>
                  <button
                    @click="assignTaskToAgent(selectedTask, suggestion)"
                    class="px-2 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
                  >
                    Assign
                  </button>
                </div>
              </div>
              <div class="text-xs text-slate-600 dark:text-slate-400">
                {{ suggestion.reasoning }}
              </div>
            </div>
          </div>
        </div>

        <!-- Assignment History -->
        <div v-if="recentAssignments.length" class="assignment-history">
          <h4 class="font-medium text-slate-900 dark:text-white mb-3">Recent Assignments</h4>
          <div class="space-y-2">
            <div 
              v-for="assignment in recentAssignments"
              :key="assignment.id"
              class="assignment-item p-2 bg-slate-50 dark:bg-slate-800 rounded text-xs"
            >
              <div class="flex justify-between items-start">
                <div>
                  <span class="font-medium">{{ assignment.task_title }}</span>
                  <span class="text-slate-500 dark:text-slate-400 ml-2">→</span>
                  <span class="text-primary-600 dark:text-primary-400 ml-2">
                    {{ assignment.agent_name }}
                  </span>
                </div>
                <span class="text-slate-400 dark:text-slate-500">
                  {{ formatTime(assignment.assigned_at) }}
                </span>
              </div>
              <div class="mt-1 flex items-center space-x-2">
                <span 
                  class="px-1 py-0.5 rounded text-xs"
                  :class="getStatusClass(assignment.status)"
                >
                  {{ assignment.status }}
                </span>
                <span class="text-slate-500 dark:text-slate-400">
                  {{ Math.round(assignment.confidence * 100) }}% confidence
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- No Selection State -->
        <div 
          v-if="!selectedTask && !selectedAgent"
          class="text-center py-8 text-slate-500 dark:text-slate-400"
        >
          <CursorArrowRaysIcon class="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>Select a task or agent</p>
          <p class="text-sm mt-1">to see assignment details</p>
        </div>
      </div>
    </div>

    <!-- Task Creation Modal -->
    <TaskCreationModal
      v-if="showTaskModal"
      :task="editingTask"
      @close="showTaskModal = false"
      @save="saveTask"
    />

    <!-- Assignment Confirmation Modal -->
    <AssignmentConfirmationModal
      v-if="showAssignmentModal"
      :task="pendingAssignment?.task"
      :agent="pendingAssignment?.agent"
      :confidence="pendingAssignment?.confidence"
      @confirm="confirmAssignment"
      @cancel="cancelAssignment"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { formatDistanceToNow, format } from 'date-fns'

// Icons
import {
  PlusIcon,
  ArrowPathIcon,
  ClipboardDocumentListIcon,
  UserGroupIcon,
  CursorArrowRaysIcon
} from '@heroicons/vue/24/outline'

// Components
import TaskCard from './TaskCard.vue'
import AgentCard from './AgentCard.vue'
import TaskCreationModal from './TaskCreationModal.vue'
import AssignmentConfirmationModal from './AssignmentConfirmationModal.vue'

// Services and composables
import { useCoordinationService } from '@/services/coordinationService'
import { useUnifiedWebSocket } from '@/services/unifiedWebSocketManager'
import { useSessionColors } from '@/utils/SessionColorManager'
import { api } from '@/services/api'

// Types
interface Task {
  id: string
  task_title: string
  task_description: string
  task_type: string
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  required_capabilities: string[]
  estimated_effort_hours?: number
  deadline?: string
  status: string
  context_data?: Record<string, any>
}

interface Agent {
  agent_id: string
  name: string
  type: string
  status: 'active' | 'idle' | 'busy' | 'sleeping'
  current_workload: number
  available_capacity: number
  capabilities: Array<{ name: string; confidence_level: number }>
  active_tasks: number
  performance_score: number
}

interface Assignment {
  id: string
  task_title: string
  agent_name: string
  status: string
  confidence: number
  assigned_at: string
}

interface MatchingSuggestion {
  agent_id: string
  name: string
  match_score: number
  reasoning: string
  workload_impact: number
}

const coordinationService = useCoordinationService()
const webSocketManager = useUnifiedWebSocket()
const { getAgentColor } = useSessionColors()

// State
const loading = ref(false)
const draggedTask = ref<Task | null>(null)
const dropTargetAgent = ref<Agent | null>(null)
const selectedTask = ref<Task | null>(null)
const selectedAgent = ref<Agent | null>(null)
const selectedPriority = ref('all')
const selectedAgentFilter = ref('all')

// Modal states
const showTaskModal = ref(false)
const showAssignmentModal = ref(false)
const editingTask = ref<Task | null>(null)
const pendingAssignment = ref<{ task: Task; agent: Agent; confidence: number } | null>(null)

// Data
const tasks = ref<Task[]>([])
const agents = ref<Agent[]>([])
const matchingSuggestions = ref<MatchingSuggestion[]>([])
const recentAssignments = ref<Assignment[]>([])

// Computed
const filteredTasks = computed(() => {
  if (selectedPriority.value === 'all') {
    return tasks.value.filter(task => task.status === 'pending' || task.status === 'created')
  }
  return tasks.value.filter(task => 
    task.priority === selectedPriority.value && 
    (task.status === 'pending' || task.status === 'created')
  )
})

const filteredAgents = computed(() => {
  if (selectedAgentFilter.value === 'all') {
    return agents.value.filter(agent => agent.status !== 'sleeping')
  }
  
  switch (selectedAgentFilter.value) {
    case 'available':
      return agents.value.filter(agent => 
        agent.status === 'active' && agent.current_workload < 0.8
      )
    case 'busy':
      return agents.value.filter(agent => 
        agent.status === 'active' && agent.current_workload >= 0.8
      )
    case 'idle':
      return agents.value.filter(agent => agent.status === 'idle')
    default:
      return agents.value
  }
})

// Methods
const loadTasks = async () => {
  loading.value = true
  try {
    const response = await api.get('/tasks', {
      params: {
        status: 'pending,created',
        limit: 100
      }
    })
    tasks.value = response.data.tasks || []
  } catch (error) {
    console.error('Failed to load tasks:', error)
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

const loadRecentAssignments = async () => {
  try {
    const response = await api.get('/tasks/assignments/recent', {
      params: { limit: 10 }
    })
    recentAssignments.value = response.data.assignments || []
  } catch (error) {
    console.error('Failed to load recent assignments:', error)
  }
}

const refreshAgents = async () => {
  await Promise.all([
    loadAgents(),
    loadTasks(),
    loadRecentAssignments()
  ])
}

const createNewTask = () => {
  editingTask.value = null
  showTaskModal.value = true
}

const editTask = (task: Task) => {
  editingTask.value = task
  showTaskModal.value = true
}

const saveTask = async (taskData: Partial<Task>) => {
  try {
    if (editingTask.value) {
      // Update existing task
      await api.put(`/tasks/${editingTask.value.id}`, taskData)
    } else {
      // Create new task
      await api.post('/tasks', taskData)
    }
    
    await loadTasks()
    showTaskModal.value = false
  } catch (error) {
    console.error('Failed to save task:', error)
  }
}

const selectTask = (task: Task) => {
  selectedTask.value = task
  selectedAgent.value = null
  generateMatchingSuggestions(task)
}

const selectAgent = (agent: Agent) => {
  selectedAgent.value = agent
  selectedTask.value = null
  matchingSuggestions.value = []
}

const generateMatchingSuggestions = async (task: Task) => {
  if (!task.required_capabilities?.length) {
    matchingSuggestions.value = []
    return
  }

  // Calculate capability matches for all available agents
  const suggestions: MatchingSuggestion[] = []

  for (const agent of filteredAgents.value) {
    if (agent.current_workload >= 1.0) continue // Skip fully loaded agents

    const agentCapabilities = agent.capabilities?.map(c => c.name.toLowerCase()) || []
    const requiredCapabilities = task.required_capabilities.map(c => c.toLowerCase())
    
    // Calculate match score
    const matchedCapabilities = requiredCapabilities.filter(req => 
      agentCapabilities.some(ac => ac.includes(req) || req.includes(ac))
    )
    
    const capabilityScore = matchedCapabilities.length / requiredCapabilities.length
    const workloadScore = 1 - agent.current_workload
    const performanceScore = agent.performance_score || 0.5
    
    const compositeScore = (capabilityScore * 0.5) + (workloadScore * 0.3) + (performanceScore * 0.2)
    
    if (compositeScore > 0.3) { // Only suggest if decent match
      suggestions.push({
        agent_id: agent.agent_id,
        name: agent.name,
        match_score: compositeScore,
        reasoning: generateReasoningText(capabilityScore, workloadScore, performanceScore, matchedCapabilities),
        workload_impact: 1 - agent.available_capacity
      })
    }
  }

  // Sort by match score and take top 3
  matchingSuggestions.value = suggestions
    .sort((a, b) => b.match_score - a.match_score)
    .slice(0, 3)
}

const generateReasoningText = (
  capabilityScore: number, 
  workloadScore: number, 
  performanceScore: number,
  matchedCapabilities: string[]
): string => {
  const reasons = []
  
  if (capabilityScore > 0.8) {
    reasons.push(`Strong capability match (${matchedCapabilities.length} skills)`)
  } else if (capabilityScore > 0.5) {
    reasons.push(`Good capability match`)
  }
  
  if (workloadScore > 0.7) {
    reasons.push('Low current workload')
  } else if (workloadScore > 0.4) {
    reasons.push('Moderate workload')
  }
  
  if (performanceScore > 0.8) {
    reasons.push('High performance history')
  }
  
  return reasons.join(', ') || 'Basic compatibility'
}

const getSuggestedMatch = (agent: Agent) => {
  if (!selectedTask.value) return null
  return matchingSuggestions.value.find(s => s.agent_id === agent.agent_id)
}

// Drag and Drop handlers
const onTaskDragStart = (task: Task) => {
  draggedTask.value = task
  selectedTask.value = task
  generateMatchingSuggestions(task)
}

const onTaskDragEnd = () => {
  draggedTask.value = null
  dropTargetAgent.value = null
}

const onAgentDragOver = (agent: Agent) => {
  if (draggedTask.value) {
    dropTargetAgent.value = agent
  }
}

const onAgentDragLeave = () => {
  dropTargetAgent.value = null
}

const onTaskDrop = (agent: Agent) => {
  if (draggedTask.value) {
    const suggestion = getSuggestedMatch(agent)
    const confidence = suggestion?.match_score || 0.5
    
    pendingAssignment.value = {
      task: draggedTask.value,
      agent,
      confidence
    }
    showAssignmentModal.value = true
  }
  
  draggedTask.value = null
  dropTargetAgent.value = null
}

const assignTaskToAgent = (task: Task, suggestion: MatchingSuggestion) => {
  const agent = agents.value.find(a => a.agent_id === suggestion.agent_id)
  if (agent) {
    pendingAssignment.value = {
      task,
      agent,
      confidence: suggestion.match_score
    }
    showAssignmentModal.value = true
  }
}

const confirmAssignment = async () => {
  if (!pendingAssignment.value) return

  try {
    const { task, agent } = pendingAssignment.value
    
    await api.post('/team-coordination/tasks/distribute', {
      task_title: task.task_title,
      task_description: task.task_description,
      task_type: task.task_type,
      priority: task.priority,
      required_capabilities: task.required_capabilities,
      estimated_effort_hours: task.estimated_effort_hours,
      deadline: task.deadline,
      context_data: task.context_data,
      target_agent_id: agent.agent_id
    })

    // Refresh data
    await Promise.all([
      loadTasks(),
      loadAgents(),
      loadRecentAssignments()
    ])

    showAssignmentModal.value = false
    pendingAssignment.value = null
    selectedTask.value = null
    matchingSuggestions.value = []

  } catch (error) {
    console.error('Failed to assign task:', error)
  }
}

const cancelAssignment = () => {
  showAssignmentModal.value = false
  pendingAssignment.value = null
}

const filterTasks = () => {
  // Filtering is handled by computed property
}

const filterAgents = () => {
  // Filtering is handled by computed property
}

// Utility functions
const formatTime = (dateString: string) => {
  return formatDistanceToNow(new Date(dateString), { addSuffix: true })
}

const formatDeadline = (deadline?: string) => {
  if (!deadline) return 'No deadline'
  return format(new Date(deadline), 'MMM dd, HH:mm')
}

const getPriorityClass = (priority: string) => {
  const classes = {
    'CRITICAL': 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400',
    'HIGH': 'bg-orange-100 dark:bg-orange-900 text-orange-600 dark:text-orange-400',
    'MEDIUM': 'bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400',
    'LOW': 'bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400'
  }
  return classes[priority] || 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-400'
}

const getStatusClass = (status: string) => {
  const classes = {
    'assigned': 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400',
    'in_progress': 'bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400',
    'completed': 'bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400',
    'failed': 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400'
  }
  return classes[status] || 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-400'
}

// WebSocket integration
const setupWebSocketListeners = () => {
  coordinationService.on('task_update', (event) => {
    if (event.type === 'task_assigned' || event.type === 'task_completed') {
      loadTasks()
      loadRecentAssignments()
    }
  })

  coordinationService.on('agent_update', (event) => {
    if (event.type === 'status_changed' || event.type === 'workload_changed') {
      loadAgents()
    }
  })
}

// Lifecycle
onMounted(async () => {
  await refreshAgents()
  setupWebSocketListeners()
})

onUnmounted(() => {
  coordinationService.off('task_update', () => {})
  coordinationService.off('agent_update', () => {})
})

// Watch for task selection changes to update suggestions
watch(selectedTask, (newTask) => {
  if (newTask) {
    generateMatchingSuggestions(newTask)
  } else {
    matchingSuggestions.value = []
  }
})
</script>

<style scoped>
.task-distribution-interface {
  @apply min-h-screen bg-slate-50 dark:bg-slate-900;
}

.glass-card {
  @apply bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50;
}

.input-field {
  @apply bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent;
}

.btn-primary {
  @apply bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.btn-secondary {
  @apply bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 px-4 py-2 rounded-md font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.task-queue-container::-webkit-scrollbar,
.agent-grid::-webkit-scrollbar {
  @apply w-2;
}

.task-queue-container::-webkit-scrollbar-track,
.agent-grid::-webkit-scrollbar-track {
  @apply bg-slate-100 dark:bg-slate-800 rounded;
}

.task-queue-container::-webkit-scrollbar-thumb,
.agent-grid::-webkit-scrollbar-thumb {
  @apply bg-slate-300 dark:bg-slate-600 rounded hover:bg-slate-400 dark:hover:bg-slate-500;
}

.suggestion-item {
  @apply transform transition-all duration-200 hover:scale-105;
}

.assignment-item {
  @apply transform transition-all duration-200 hover:bg-slate-100 dark:hover:bg-slate-700;
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

.suggestion-item,
.assignment-item {
  animation: slideIn 0.3s ease-out;
}
</style>