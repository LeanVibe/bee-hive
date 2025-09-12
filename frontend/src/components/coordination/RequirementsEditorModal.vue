<template>
  <div 
    v-if="isVisible"
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    @click.self="close"
  >
    <div 
      class="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden"
      @click.stop
    >
      <!-- Header -->
      <div class="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
          Edit Task Requirements
        </h2>
        <button
          @click="close"
          class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>
      
      <!-- Content -->
      <div class="p-6 overflow-y-auto max-h-[calc(90vh-8rem)]">
        <form @submit.prevent="save" class="space-y-6">
          <!-- Basic Requirements -->
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Min Agents
              </label>
              <input
                v-model.number="formData.min_agents"
                type="number"
                min="1"
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Max Agents
              </label>
              <input
                v-model.number="formData.max_agents"
                type="number"
                :min="formData.min_agents"
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
          
          <!-- Required Capabilities -->
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Required Capabilities
            </label>
            <div class="space-y-2">
              <div 
                v-for="(capability, index) in formData.required_capabilities"
                :key="index"
                class="flex items-center space-x-2"
              >
                <input
                  v-model="capability.name"
                  type="text"
                  placeholder="Capability name"
                  class="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <select
                  v-model="capability.level"
                  class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="basic">Basic</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                  <option value="expert">Expert</option>
                </select>
                <button
                  type="button"
                  @click="removeRequiredCapability(index)"
                  class="p-2 text-red-600 hover:text-red-800 hover:bg-red-50 dark:hover:bg-red-900 rounded"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                  </svg>
                </button>
              </div>
              <button
                type="button"
                @click="addRequiredCapability"
                class="w-full py-2 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-md text-gray-500 dark:text-gray-400 hover:border-primary-500 hover:text-primary-500 transition-colors"
              >
                + Add Required Capability
              </button>
            </div>
          </div>
          
          <!-- Preferred Capabilities -->
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Preferred Capabilities (Optional)
            </label>
            <div class="space-y-2">
              <div 
                v-for="(capability, index) in formData.preferred_capabilities"
                :key="index"
                class="flex items-center space-x-2"
              >
                <input
                  v-model="capability.name"
                  type="text"
                  placeholder="Capability name"
                  class="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <select
                  v-model="capability.level"
                  class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="basic">Basic</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                  <option value="expert">Expert</option>
                </select>
                <input
                  v-model.number="capability.weight"
                  type="number"
                  min="0"
                  max="100"
                  placeholder="Weight"
                  class="w-20 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <button
                  type="button"
                  @click="removePreferredCapability(index)"
                  class="p-2 text-red-600 hover:text-red-800 hover:bg-red-50 dark:hover:bg-red-900 rounded"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                  </svg>
                </button>
              </div>
              <button
                type="button"
                @click="addPreferredCapability"
                class="w-full py-2 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-md text-gray-500 dark:text-gray-400 hover:border-primary-500 hover:text-primary-500 transition-colors"
              >
                + Add Preferred Capability
              </button>
            </div>
          </div>
          
          <!-- Resource Requirements -->
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Memory Requirement (MB)
              </label>
              <input
                v-model.number="formData.resource_requirements.memory_mb"
                type="number"
                min="0"
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                CPU Cores
              </label>
              <input
                v-model.number="formData.resource_requirements.cpu_cores"
                type="number"
                min="0"
                step="0.1"
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
          
          <!-- Execution Context -->
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Priority
              </label>
              <select
                v-model="formData.execution_context.priority"
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Timeout (minutes)
              </label>
              <input
                v-model.number="formData.execution_context.timeout_minutes"
                type="number"
                min="1"
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
        </form>
      </div>
      
      <!-- Footer -->
      <div class="flex items-center justify-end space-x-3 p-6 border-t border-gray-200 dark:border-gray-700">
        <button
          @click="close"
          type="button"
          class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        >
          Cancel
        </button>
        <button
          @click="save"
          type="button"
          class="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-md transition-colors"
        >
          Save Requirements
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, reactive } from 'vue'

interface Capability {
  name: string
  level: string
  weight?: number
}

interface ResourceRequirements {
  memory_mb: number
  cpu_cores: number
}

interface ExecutionContext {
  priority: string
  timeout_minutes: number
}

interface Requirements {
  min_agents: number
  max_agents: number
  required_capabilities: Capability[]
  preferred_capabilities: Capability[]
  resource_requirements: ResourceRequirements
  execution_context: ExecutionContext
}

interface Props {
  isVisible: boolean
  requirements?: Requirements | null
}

const props = withDefaults(defineProps<Props>(), {
  requirements: null
})

const emit = defineEmits<{
  close: []
  save: [requirements: Requirements]
}>()

const formData = reactive<Requirements>({
  min_agents: 1,
  max_agents: 5,
  required_capabilities: [],
  preferred_capabilities: [],
  resource_requirements: {
    memory_mb: 512,
    cpu_cores: 1
  },
  execution_context: {
    priority: 'medium',
    timeout_minutes: 30
  }
})

watch(() => props.requirements, (newRequirements) => {
  if (newRequirements) {
    Object.assign(formData, JSON.parse(JSON.stringify(newRequirements)))
  }
}, { immediate: true })

function addRequiredCapability() {
  formData.required_capabilities.push({ name: '', level: 'basic' })
}

function removeRequiredCapability(index: number) {
  formData.required_capabilities.splice(index, 1)
}

function addPreferredCapability() {
  formData.preferred_capabilities.push({ name: '', level: 'basic', weight: 50 })
}

function removePreferredCapability(index: number) {
  formData.preferred_capabilities.splice(index, 1)
}

function save() {
  // Validate and clean up data
  const cleanedData = {
    ...formData,
    required_capabilities: formData.required_capabilities.filter(cap => cap.name.trim()),
    preferred_capabilities: formData.preferred_capabilities.filter(cap => cap.name.trim())
  }
  
  emit('save', cleanedData)
  close()
}

function close() {
  emit('close')
}
</script>

<style scoped>
/* Modal styles */
</style>