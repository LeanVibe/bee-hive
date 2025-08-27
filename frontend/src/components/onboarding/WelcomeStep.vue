<template>
  <div class="flex flex-col lg:flex-row h-full">
    <!-- Left side - Hero content -->
    <div class="flex-1 p-8 lg:p-12 flex flex-col justify-center">
      <div class="max-w-lg mx-auto lg:mx-0">
        <!-- Animated logo/icon -->
        <div class="mb-8 flex justify-center lg:justify-start">
          <div class="relative">
            <div class="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center">
              <svg class="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <!-- Pulsing animation -->
            <div class="absolute -inset-4 rounded-2xl bg-gradient-to-r from-blue-500/20 to-purple-600/20 animate-pulse"></div>
          </div>
        </div>

        <h1 class="text-4xl lg:text-5xl font-bold text-gray-900 mb-6 text-center lg:text-left">
          Welcome to
          <span class="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            LeanVibe Agent Hive
          </span>
        </h1>
        
        <p class="text-xl text-gray-600 mb-8 text-center lg:text-left leading-relaxed">
          Your AI-powered orchestration platform that transforms complex workflows into simple, intelligent automation.
          Let's get you set up in just a few minutes.
        </p>

        <!-- Value propositions -->
        <div class="space-y-4 mb-8">
          <div class="flex items-start space-x-3">
            <div class="flex-shrink-0 w-8 h-8 bg-green-100 rounded-full flex items-center justify-center mt-0.5">
              <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <h3 class="font-semibold text-gray-900">Autonomous AI Agents</h3>
              <p class="text-gray-600">Create intelligent agents that handle complex tasks independently</p>
            </div>
          </div>
          
          <div class="flex items-start space-x-3">
            <div class="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
              <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h3 class="font-semibold text-gray-900">Real-time Analytics</h3>
              <p class="text-gray-600">Monitor performance and optimize workflows with live insights</p>
            </div>
          </div>
          
          <div class="flex items-start space-x-3">
            <div class="flex-shrink-0 w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center mt-0.5">
              <svg class="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
              </svg>
            </div>
            <div>
              <h3 class="font-semibold text-gray-900">Enterprise Security</h3>
              <p class="text-gray-600">Bank-grade security with granular access controls and audit trails</p>
            </div>
          </div>
        </div>

        <!-- User input section -->
        <div class="space-y-4">
          <div>
            <label for="fullName" class="block text-sm font-medium text-gray-700 mb-2">
              What should we call you?
            </label>
            <input
              id="fullName"
              v-model="formData.name"
              type="text"
              placeholder="Enter your name"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              @input="handleInputChange"
            >
          </div>
          
          <div>
            <label for="role" class="block text-sm font-medium text-gray-700 mb-2">
              What's your primary role?
            </label>
            <select
              id="role"
              v-model="formData.role"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              @change="handleInputChange"
            >
              <option value="">Select your role</option>
              <option value="developer">Developer</option>
              <option value="manager">Manager</option>
              <option value="analyst">Business Analyst</option>
              <option value="ops">Operations</option>
              <option value="executive">Executive</option>
              <option value="other">Other</option>
            </select>
          </div>

          <!-- Goals selection -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-3">
              What do you want to achieve? (Select all that apply)
            </label>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label 
                v-for="goal in availableGoals" 
                :key="goal.id"
                class="flex items-center space-x-3 p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 cursor-pointer transition-all"
              >
                <input
                  type="checkbox"
                  :value="goal.id"
                  v-model="formData.goals"
                  class="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  @change="handleInputChange"
                >
                <div class="flex-1">
                  <div class="font-medium text-gray-900">{{ goal.title }}</div>
                  <div class="text-sm text-gray-600">{{ goal.description }}</div>
                </div>
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right side - Visual/video -->
    <div class="flex-1 bg-gradient-to-br from-blue-100 to-purple-100 p-8 lg:p-12 flex items-center justify-center">
      <div class="max-w-md">
        <!-- Animated dashboard preview or video -->
        <div class="bg-white rounded-2xl shadow-2xl overflow-hidden">
          <div class="bg-gray-900 px-4 py-2 flex items-center space-x-2">
            <div class="w-3 h-3 bg-red-400 rounded-full"></div>
            <div class="w-3 h-3 bg-yellow-400 rounded-full"></div>
            <div class="w-3 h-3 bg-green-400 rounded-full"></div>
          </div>
          
          <div class="p-6">
            <!-- Mock dashboard interface -->
            <div class="space-y-4">
              <div class="h-8 bg-gradient-to-r from-blue-400 to-purple-400 rounded-lg animate-pulse"></div>
              
              <div class="grid grid-cols-2 gap-3">
                <div class="h-20 bg-green-100 rounded-lg p-3">
                  <div class="h-2 bg-green-400 rounded mb-2 animate-pulse"></div>
                  <div class="h-6 bg-green-300 rounded animate-pulse" style="animation-delay: 0.2s"></div>
                </div>
                <div class="h-20 bg-blue-100 rounded-lg p-3">
                  <div class="h-2 bg-blue-400 rounded mb-2 animate-pulse" style="animation-delay: 0.1s"></div>
                  <div class="h-6 bg-blue-300 rounded animate-pulse" style="animation-delay: 0.3s"></div>
                </div>
              </div>
              
              <div class="space-y-2">
                <div class="h-3 bg-gray-200 rounded animate-pulse" style="animation-delay: 0.4s"></div>
                <div class="h-3 bg-gray-200 rounded w-3/4 animate-pulse" style="animation-delay: 0.5s"></div>
                <div class="h-3 bg-gray-200 rounded w-1/2 animate-pulse" style="animation-delay: 0.6s"></div>
              </div>
            </div>
          </div>
        </div>
        
        <p class="text-center text-gray-600 mt-6 text-sm">
          Get started with your personalized dashboard in under 5 minutes
        </p>
      </div>
    </div>

    <!-- Action buttons - desktop -->
    <div class="absolute bottom-8 left-8 right-8 flex justify-between items-center lg:static lg:hidden">
      <!-- Hidden for first step -->
    </div>
    
    <!-- Action buttons - positioned for desktop -->
    <div class="hidden lg:flex absolute bottom-12 right-12">
      <button
        @click="handleContinue"
        :disabled="!isFormValid"
        class="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
      >
        <span class="font-semibold">Let's Get Started</span>
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'

interface Props {
  userData?: any
  stepData?: any
}

interface FormData {
  name: string
  role: string
  goals: string[]
}

const props = defineProps<Props>()
const emit = defineEmits(['next', 'update-data', 'analytics-event'])

// Form data
const formData = ref<FormData>({
  name: props.userData?.name || '',
  role: props.userData?.role || '',
  goals: props.userData?.goals || []
})

// Available goals
const availableGoals = ref([
  {
    id: 'automate_workflows',
    title: 'Automate Workflows',
    description: 'Streamline repetitive tasks'
  },
  {
    id: 'improve_efficiency',
    title: 'Improve Efficiency',
    description: 'Optimize team productivity'
  },
  {
    id: 'scale_operations',
    title: 'Scale Operations',
    description: 'Handle increased workload'
  },
  {
    id: 'reduce_errors',
    title: 'Reduce Errors',
    description: 'Minimize human mistakes'
  },
  {
    id: 'data_insights',
    title: 'Get Data Insights',
    description: 'Make data-driven decisions'
  },
  {
    id: 'cost_reduction',
    title: 'Reduce Costs',
    description: 'Optimize resource usage'
  }
])

// Computed properties
const isFormValid = computed(() => {
  return formData.value.name.trim().length >= 2 && 
         formData.value.role.length > 0 &&
         formData.value.goals.length > 0
})

// Event handlers
const handleInputChange = () => {
  emit('update-data', formData.value)
  
  // Track input analytics
  emit('analytics-event', 'form_interaction', {
    field: 'user_data',
    hasName: formData.value.name.length > 0,
    hasRole: formData.value.role.length > 0,
    goalCount: formData.value.goals.length
  })
}

const handleContinue = () => {
  if (isFormValid.value) {
    emit('analytics-event', 'welcome_completed', {
      name: formData.value.name,
      role: formData.value.role,
      goals: formData.value.goals,
      timeOnStep: Date.now() - stepStartTime.value
    })
    emit('next')
  }
}

// Track time on step
const stepStartTime = ref<number>(Date.now())

// Watch for external data updates
watch(() => props.userData, (newData) => {
  if (newData) {
    formData.value = {
      name: newData.name || '',
      role: newData.role || '',
      goals: newData.goals || []
    }
  }
}, { immediate: true })

// Lifecycle
onMounted(() => {
  stepStartTime.value = Date.now()
  
  // Focus first input for accessibility
  setTimeout(() => {
    const firstInput = document.getElementById('fullName')
    firstInput?.focus()
  }, 100)
  
  emit('analytics-event', 'welcome_step_viewed', {
    timestamp: new Date().toISOString()
  })
})

// Keyboard shortcuts
const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && event.metaKey && isFormValid.value) {
    handleContinue()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

import { onBeforeUnmount } from 'vue'
onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped>
/* Custom animations for the dashboard preview */
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

.floating {
  animation: float 3s ease-in-out infinite;
}

/* Form input focus states */
input:focus,
select:focus {
  outline: none;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Checkbox custom styles */
input[type="checkbox"]:checked {
  background-color: #3b82f6;
  border-color: #3b82f6;
}

/* Mobile responsiveness */
@media (max-width: 1024px) {
  .absolute.bottom-12.right-12 {
    position: static;
    display: flex;
    justify-content: center;
    margin-top: 2rem;
  }
}

/* Reduce motion for accessibility */
@media (prefers-reduced-motion: reduce) {
  .animate-pulse,
  .floating,
  .transform.hover\\:scale-105 {
    animation: none;
    transform: none;
  }
}
</style>