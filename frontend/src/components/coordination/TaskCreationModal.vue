<template>
  <div class="task-creation-modal fixed inset-0 z-50 overflow-y-auto">
    <!-- Backdrop -->
    <div 
      class="modal-backdrop fixed inset-0 bg-black bg-opacity-50 transition-opacity"
      @click="$emit('close')"
    ></div>

    <!-- Modal -->
    <div class="modal-container flex items-center justify-center min-h-screen px-4 py-6">
      <div class="modal-content glass-card rounded-xl w-full max-w-2xl max-h-screen overflow-y-auto">
        <!-- Header -->
        <div class="modal-header flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700">
          <h2 class="text-xl font-semibold text-slate-900 dark:text-white">
            {{ task ? 'Edit Task' : 'Create New Task' }}
          </h2>
          <button
            @click="$emit('close')"
            class="close-button p-2 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
          >
            <XMarkIcon class="w-5 h-5 text-slate-400" />
          </button>
        </div>

        <!-- Form -->
        <form @submit.prevent="handleSubmit" class="modal-body p-6 space-y-6">
          <!-- Task Title -->
          <div class="form-group">
            <label for="task-title" class="form-label">
              Task Title *
            </label>
            <input
              id="task-title"
              v-model="formData.task_title"
              type="text"
              class="form-input"
              placeholder="Enter a descriptive task title"
              required
            />
            <p v-if="errors.task_title" class="form-error">
              {{ errors.task_title }}
            </p>
          </div>

          <!-- Task Description -->
          <div class="form-group">
            <label for="task-description" class="form-label">
              Task Description *
            </label>
            <textarea
              id="task-description"
              v-model="formData.task_description"
              class="form-textarea"
              rows="4"
              placeholder="Provide detailed task requirements and context"
              required
            ></textarea>
            <p v-if="errors.task_description" class="form-error">
              {{ errors.task_description }}
            </p>
          </div>

          <!-- Task Type and Priority Row -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Task Type -->
            <div class="form-group">
              <label for="task-type" class="form-label">
                Task Type *
              </label>
              <select
                id="task-type"
                v-model="formData.task_type"
                class="form-select"
                required
              >
                <option value="">Select task type</option>
                <option value="FRONTEND_DEVELOPMENT">Frontend Development</option>
                <option value="BACKEND_DEVELOPMENT">Backend Development</option>
                <option value="API_INTEGRATION">API Integration</option>
                <option value="TESTING">Testing</option>
                <option value="CODE_REVIEW">Code Review</option>
                <option value="DOCUMENTATION">Documentation</option>
                <option value="BUG_FIX">Bug Fix</option>
                <option value="FEATURE_ENHANCEMENT">Feature Enhancement</option>
                <option value="DEPLOYMENT">Deployment</option>
                <option value="RESEARCH">Research</option>
              </select>
              <p v-if="errors.task_type" class="form-error">
                {{ errors.task_type }}
              </p>
            </div>

            <!-- Priority -->
            <div class="form-group">
              <label for="priority" class="form-label">
                Priority *
              </label>
              <select
                id="priority"
                v-model="formData.priority"
                class="form-select"
                required
              >
                <option value="">Select priority</option>
                <option value="LOW">Low</option>
                <option value="MEDIUM">Medium</option>
                <option value="HIGH">High</option>
                <option value="CRITICAL">Critical</option>
              </select>
              <p v-if="errors.priority" class="form-error">
                {{ errors.priority }}
              </p>
            </div>
          </div>

          <!-- Required Capabilities -->
          <div class="form-group">
            <label class="form-label">
              Required Capabilities *
            </label>
            <div class="capabilities-input">
              <div class="flex items-center space-x-2 mb-3">
                <input
                  v-model="newCapability"
                  type="text"
                  class="form-input flex-1"
                  placeholder="Enter a required skill or capability"
                  @keydown.enter.prevent="addCapability"
                />
                <button
                  type="button"
                  @click="addCapability"
                  class="btn-primary"
                  :disabled="!newCapability.trim()"
                >
                  <PlusIcon class="w-4 h-4" />
                </button>
              </div>
              
              <!-- Capability Tags -->
              <div v-if="formData.required_capabilities.length" class="capability-tags flex flex-wrap gap-2 mb-2">
                <span
                  v-for="(capability, index) in formData.required_capabilities"
                  :key="index"
                  class="capability-tag inline-flex items-center px-3 py-1 text-sm bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-full"
                >
                  {{ capability }}
                  <button
                    type="button"
                    @click="removeCapability(index)"
                    class="ml-2 text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-200"
                  >
                    <XMarkIcon class="w-3 h-3" />
                  </button>
                </span>
              </div>
              
              <!-- Suggested Capabilities -->
              <div v-if="suggestedCapabilities.length" class="suggested-capabilities">
                <p class="text-xs text-slate-600 dark:text-slate-400 mb-2">Suggested capabilities:</p>
                <div class="flex flex-wrap gap-1">
                  <button
                    v-for="suggestion in suggestedCapabilities"
                    :key="suggestion"
                    type="button"
                    @click="addSuggestedCapability(suggestion)"
                    class="suggestion-tag px-2 py-1 text-xs bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
                  >
                    + {{ suggestion }}
                  </button>
                </div>
              </div>
            </div>
            <p v-if="errors.required_capabilities" class="form-error">
              {{ errors.required_capabilities }}
            </p>
          </div>

          <!-- Effort and Deadline Row -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Estimated Effort -->
            <div class="form-group">
              <label for="estimated-effort" class="form-label">
                Estimated Effort (hours)
              </label>
              <input
                id="estimated-effort"
                v-model.number="formData.estimated_effort_hours"
                type="number"
                min="0.1"
                max="200"
                step="0.5"
                class="form-input"
                placeholder="0.0"
              />
              <p class="form-help">
                Leave empty if effort is unknown
              </p>
            </div>

            <!-- Deadline -->
            <div class="form-group">
              <label for="deadline" class="form-label">
                Deadline
              </label>
              <input
                id="deadline"
                v-model="formData.deadline"
                type="datetime-local"
                class="form-input"
              />
              <p class="form-help">
                Optional deadline for task completion
              </p>
            </div>
          </div>

          <!-- Dependencies -->
          <div class="form-group">
            <label class="form-label">
              Task Dependencies
            </label>
            <div class="dependencies-section">
              <select
                v-model="selectedDependency"
                class="form-select mb-2"
              >
                <option value="">Select a dependency task</option>
                <option
                  v-for="availableTask in availableTasks"
                  :key="availableTask.id"
                  :value="availableTask.id"
                >
                  {{ availableTask.task_title }}
                </option>
              </select>
              
              <button
                type="button"
                @click="addDependency"
                class="btn-secondary text-sm mb-3"
                :disabled="!selectedDependency"
              >
                <LinkIcon class="w-4 h-4 mr-1" />
                Add Dependency
              </button>

              <!-- Dependency List -->
              <div v-if="formData.dependencies.length" class="dependency-list space-y-2">
                <div
                  v-for="(dependencyId, index) in formData.dependencies"
                  :key="dependencyId"
                  class="dependency-item flex items-center justify-between p-2 bg-slate-50 dark:bg-slate-800 rounded"
                >
                  <span class="text-sm">
                    {{ getDependencyTaskTitle(dependencyId) }}
                  </span>
                  <button
                    type="button"
                    @click="removeDependency(index)"
                    class="text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-200"
                  >
                    <XMarkIcon class="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Context Data (Advanced) -->
          <div class="form-group">
            <div class="flex items-center justify-between mb-2">
              <label class="form-label">
                Additional Context
              </label>
              <button
                type="button"
                @click="showAdvanced = !showAdvanced"
                class="text-sm text-primary-600 dark:text-primary-400 hover:underline"
              >
                {{ showAdvanced ? 'Hide Advanced' : 'Show Advanced' }}
              </button>
            </div>
            
            <div v-show="showAdvanced" class="advanced-section">
              <textarea
                v-model="contextDataText"
                class="form-textarea"
                rows="4"
                placeholder="Enter additional context as JSON or plain text"
              ></textarea>
              <p class="form-help">
                Additional context data for the task (JSON format supported)
              </p>
            </div>
          </div>
        </form>

        <!-- Footer -->
        <div class="modal-footer flex items-center justify-end space-x-3 p-6 border-t border-slate-200 dark:border-slate-700">
          <button
            type="button"
            @click="$emit('close')"
            class="btn-secondary"
          >
            Cancel
          </button>
          <button
            @click="handleSubmit"
            class="btn-primary"
            :disabled="loading || !isFormValid"
          >
            <span v-if="loading" class="flex items-center">
              <div class="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
              {{ task ? 'Updating...' : 'Creating...' }}
            </span>
            <span v-else>
              {{ task ? 'Update Task' : 'Create Task' }}
            </span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'

// Icons
import {
  XMarkIcon,
  PlusIcon,
  LinkIcon
} from '@heroicons/vue/24/outline'

interface Task {
  id?: string
  task_title: string
  task_description: string
  task_type: string
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  required_capabilities: string[]
  estimated_effort_hours?: number
  deadline?: string
  dependencies?: string[]
  context_data?: Record<string, any>
}

interface Props {
  task?: Task | null
  availableTasks?: Array<{ id: string; task_title: string }>
}

interface Emits {
  (e: 'close'): void
  (e: 'save', task: Partial<Task>): void
}

const props = withDefaults(defineProps<Props>(), {
  task: null,
  availableTasks: () => []
})

const emit = defineEmits<Emits>()

// State
const loading = ref(false)
const showAdvanced = ref(false)
const newCapability = ref('')
const selectedDependency = ref('')
const contextDataText = ref('')

// Form data
const formData = ref<Partial<Task>>({
  task_title: '',
  task_description: '',
  task_type: '',
  priority: 'MEDIUM',
  required_capabilities: [],
  estimated_effort_hours: undefined,
  deadline: '',
  dependencies: [],
  context_data: {}
})

// Form validation
const errors = ref<Record<string, string>>({})

// Suggested capabilities based on task type
const suggestedCapabilities = computed(() => {
  const suggestions: Record<string, string[]> = {
    FRONTEND_DEVELOPMENT: ['Vue.js', 'React', 'TypeScript', 'CSS', 'HTML', 'JavaScript'],
    BACKEND_DEVELOPMENT: ['Python', 'FastAPI', 'Node.js', 'Database Design', 'API Development'],
    API_INTEGRATION: ['REST APIs', 'GraphQL', 'Authentication', 'HTTP Protocols'],
    TESTING: ['Unit Testing', 'Integration Testing', 'Test Automation', 'QA'],
    CODE_REVIEW: ['Code Analysis', 'Best Practices', 'Security Review'],
    DOCUMENTATION: ['Technical Writing', 'API Documentation', 'User Guides'],
    BUG_FIX: ['Debugging', 'Problem Solving', 'Root Cause Analysis'],
    FEATURE_ENHANCEMENT: ['Feature Design', 'UX/UI', 'Performance Optimization'],
    DEPLOYMENT: ['DevOps', 'CI/CD', 'Infrastructure', 'Docker'],
    RESEARCH: ['Research Skills', 'Analysis', 'Documentation']
  }

  const taskType = formData.value.task_type
  if (!taskType) return []

  const available = suggestions[taskType] || []
  return available.filter(cap => 
    !formData.value.required_capabilities?.includes(cap) &&
    cap.toLowerCase().includes(newCapability.value.toLowerCase())
  )
})

// Form validation
const isFormValid = computed(() => {
  return formData.value.task_title &&
         formData.value.task_description &&
         formData.value.task_type &&
         formData.value.priority &&
         formData.value.required_capabilities &&
         formData.value.required_capabilities.length > 0
})

// Methods
const addCapability = () => {
  const capability = newCapability.value.trim()
  if (capability && !formData.value.required_capabilities?.includes(capability)) {
    if (!formData.value.required_capabilities) {
      formData.value.required_capabilities = []
    }
    formData.value.required_capabilities.push(capability)
    newCapability.value = ''
  }
}

const removeCapability = (index: number) => {
  formData.value.required_capabilities?.splice(index, 1)
}

const addSuggestedCapability = (capability: string) => {
  if (!formData.value.required_capabilities?.includes(capability)) {
    if (!formData.value.required_capabilities) {
      formData.value.required_capabilities = []
    }
    formData.value.required_capabilities.push(capability)
  }
}

const addDependency = () => {
  if (selectedDependency.value && !formData.value.dependencies?.includes(selectedDependency.value)) {
    if (!formData.value.dependencies) {
      formData.value.dependencies = []
    }
    formData.value.dependencies.push(selectedDependency.value)
    selectedDependency.value = ''
  }
}

const removeDependency = (index: number) => {
  formData.value.dependencies?.splice(index, 1)
}

const getDependencyTaskTitle = (taskId: string): string => {
  const task = props.availableTasks.find(t => t.id === taskId)
  return task?.task_title || 'Unknown task'
}

const validateForm = (): boolean => {
  errors.value = {}

  if (!formData.value.task_title?.trim()) {
    errors.value.task_title = 'Task title is required'
  }

  if (!formData.value.task_description?.trim()) {
    errors.value.task_description = 'Task description is required'
  }

  if (!formData.value.task_type) {
    errors.value.task_type = 'Task type is required'
  }

  if (!formData.value.priority) {
    errors.value.priority = 'Priority is required'
  }

  if (!formData.value.required_capabilities?.length) {
    errors.value.required_capabilities = 'At least one capability is required'
  }

  return Object.keys(errors.value).length === 0
}

const handleSubmit = async () => {
  if (!validateForm()) {
    return
  }

  loading.value = true

  try {
    // Parse context data if provided
    let contextData = formData.value.context_data
    if (contextDataText.value.trim()) {
      try {
        contextData = JSON.parse(contextDataText.value)
      } catch {
        contextData = { notes: contextDataText.value }
      }
    }

    const taskData: Partial<Task> = {
      ...formData.value,
      context_data: contextData,
      deadline: formData.value.deadline || undefined
    }

    emit('save', taskData)
  } catch (error) {
    console.error('Failed to save task:', error)
  } finally {
    loading.value = false
  }
}

// Initialize form data when task prop changes
watch(() => props.task, (newTask) => {
  if (newTask) {
    formData.value = {
      ...newTask,
      deadline: newTask.deadline ? new Date(newTask.deadline).toISOString().slice(0, -1) : ''
    }
    
    if (newTask.context_data) {
      contextDataText.value = JSON.stringify(newTask.context_data, null, 2)
    }
  } else {
    // Reset form for new task
    formData.value = {
      task_title: '',
      task_description: '',
      task_type: '',
      priority: 'MEDIUM',
      required_capabilities: [],
      estimated_effort_hours: undefined,
      deadline: '',
      dependencies: [],
      context_data: {}
    }
    contextDataText.value = ''
  }
}, { immediate: true })

onMounted(() => {
  // Focus on title input when modal opens
  const titleInput = document.getElementById('task-title')
  if (titleInput) {
    setTimeout(() => titleInput.focus(), 100)
  }
})
</script>

<style scoped>
.modal-backdrop {
  backdrop-filter: blur(4px);
}

.glass-card {
  @apply bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg border border-slate-200/50 dark:border-slate-700/50 shadow-xl;
}

.form-label {
  @apply block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2;
}

.form-input,
.form-textarea,
.form-select {
  @apply w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-colors;
}

.form-textarea {
  @apply resize-y;
}

.form-error {
  @apply mt-1 text-sm text-red-600 dark:text-red-400;
}

.form-help {
  @apply mt-1 text-xs text-slate-500 dark:text-slate-400;
}

.btn-primary {
  @apply bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white px-4 py-2 rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.btn-secondary {
  @apply bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 px-4 py-2 rounded-md font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.capability-tag {
  @apply transition-all duration-150 hover:scale-105;
}

.suggestion-tag {
  @apply transition-all duration-150 hover:scale-105;
}

.close-button {
  @apply transition-all duration-150 hover:scale-110;
}

.advanced-section {
  @apply space-y-3 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.modal-content {
  animation: slideIn 0.2s ease-out;
}
</style>