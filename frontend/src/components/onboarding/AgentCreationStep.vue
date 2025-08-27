<template>
  <div class="flex flex-col lg:flex-row h-full">
    <!-- Left side - Form content -->
    <div class="flex-1 p-8 lg:p-12 flex flex-col justify-center">
      <div class="max-w-2xl mx-auto lg:mx-0">
        <!-- Header -->
        <div class="mb-8">
          <h1 class="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
            Create Your First AI Agent
          </h1>
          <p class="text-lg text-gray-600 mb-6">
            AI Agents are autonomous workers that can handle tasks independently. 
            Let's create one tailored to your needs based on what you told us.
          </p>
          
          <!-- Progress indicator -->
          <div class="flex items-center space-x-2 text-sm text-gray-500">
            <svg class="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
            </svg>
            <span>Based on your goals: {{ formattedGoals }}</span>
          </div>
        </div>

        <!-- Agent Template Selection -->
        <div class="mb-8">
          <h3 class="text-lg font-semibold text-gray-900 mb-4">Choose an Agent Template</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div 
              v-for="template in recommendedTemplates" 
              :key="template.id"
              @click="selectTemplate(template)"
              class="relative p-6 border-2 rounded-xl cursor-pointer transition-all hover:border-blue-300 hover:shadow-lg"
              :class="selectedTemplate?.id === template.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'"
            >
              <!-- Selection indicator -->
              <div 
                v-if="selectedTemplate?.id === template.id"
                class="absolute top-4 right-4 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center"
              >
                <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
              </div>

              <!-- Template content -->
              <div class="flex items-start space-x-4">
                <div class="flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center text-2xl" :style="{ backgroundColor: template.color + '20' }">
                  {{ template.icon }}
                </div>
                <div class="flex-1">
                  <h4 class="font-semibold text-gray-900 mb-1">{{ template.name }}</h4>
                  <p class="text-sm text-gray-600 mb-3">{{ template.description }}</p>
                  <div class="flex flex-wrap gap-2">
                    <span 
                      v-for="capability in template.capabilities.slice(0, 3)" 
                      :key="capability"
                      class="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded-full"
                    >
                      {{ capability }}
                    </span>
                    <span 
                      v-if="template.capabilities.length > 3"
                      class="px-2 py-1 text-xs bg-gray-100 text-gray-500 rounded-full"
                    >
                      +{{ template.capabilities.length - 3 }} more
                    </span>
                  </div>
                </div>
              </div>
              
              <!-- Recommended badge -->
              <div 
                v-if="template.recommended"
                class="absolute top-2 left-2 px-2 py-1 text-xs bg-green-100 text-green-700 rounded-full"
              >
                Recommended
              </div>
            </div>
          </div>
        </div>

        <!-- Agent Customization Form -->
        <div v-if="selectedTemplate" class="space-y-6">
          <div class="bg-gray-50 rounded-lg p-6">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">Customize Your Agent</h3>
            
            <div class="space-y-4">
              <div>
                <label for="agentName" class="block text-sm font-medium text-gray-700 mb-2">
                  Agent Name
                </label>
                <input
                  id="agentName"
                  v-model="agentForm.name"
                  type="text"
                  :placeholder="selectedTemplate.name"
                  class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  @input="handleFormChange"
                >
                <p class="mt-1 text-xs text-gray-500">Give your agent a memorable name</p>
              </div>

              <div>
                <label for="agentDescription" class="block text-sm font-medium text-gray-700 mb-2">
                  What should this agent focus on? (Optional)
                </label>
                <textarea
                  id="agentDescription"
                  v-model="agentForm.description"
                  rows="3"
                  :placeholder="selectedTemplate.defaultPrompt"
                  class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
                  @input="handleFormChange"
                ></textarea>
              </div>

              <!-- Capability Selection -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-3">
                  Capabilities (Select what this agent can do)
                </label>
                <div class="grid grid-cols-2 gap-3">
                  <label 
                    v-for="capability in selectedTemplate.capabilities" 
                    :key="capability"
                    class="flex items-center space-x-3 p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 cursor-pointer transition-all"
                    :class="agentForm.capabilities.includes(capability) ? 'border-blue-500 bg-blue-50' : ''"
                  >
                    <input
                      type="checkbox"
                      :value="capability"
                      v-model="agentForm.capabilities"
                      class="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      @change="handleFormChange"
                    >
                    <span class="text-sm text-gray-700">{{ capability }}</span>
                  </label>
                </div>
              </div>

              <!-- Priority Level -->
              <div>
                <label for="priority" class="block text-sm font-medium text-gray-700 mb-2">
                  Default Priority Level
                </label>
                <select
                  id="priority"
                  v-model="agentForm.priority"
                  class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  @change="handleFormChange"
                >
                  <option value="low">Low - Background tasks</option>
                  <option value="medium" selected>Medium - Regular operations</option>
                  <option value="high">High - Urgent tasks</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <!-- Creation Status -->
        <div v-if="isCreating" class="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div class="flex items-center space-x-3">
            <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
            <span class="text-blue-700">Creating your AI agent...</span>
          </div>
        </div>

        <!-- Error Display -->
        <div v-if="error" class="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div class="flex items-center space-x-2">
            <svg class="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
            </svg>
            <span class="text-red-700">{{ error }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Right side - Preview -->
    <div class="flex-1 bg-gradient-to-br from-purple-100 to-blue-100 p-8 lg:p-12 flex items-center justify-center">
      <div class="max-w-md w-full">
        <!-- Agent Preview Card -->
        <div class="bg-white rounded-2xl shadow-2xl overflow-hidden">
          <div class="bg-gradient-to-r from-blue-500 to-purple-600 px-6 py-4">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center text-2xl">
                {{ selectedTemplate?.icon || '🤖' }}
              </div>
              <div class="text-white">
                <h3 class="font-semibold">{{ agentForm.name || selectedTemplate?.name || 'Your Agent' }}</h3>
                <p class="text-sm opacity-90">{{ selectedTemplate?.type || 'AI Agent' }}</p>
              </div>
            </div>
          </div>
          
          <div class="p-6">
            <!-- Status -->
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center space-x-2">
                <div class="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                <span class="text-sm text-gray-600">Ready to work</span>
              </div>
              <span class="text-xs text-gray-400">{{ selectedTemplate?.type || 'General Purpose' }}</span>
            </div>

            <!-- Capabilities Preview -->
            <div class="mb-4">
              <h4 class="text-sm font-medium text-gray-700 mb-2">Capabilities</h4>
              <div class="flex flex-wrap gap-1">
                <span 
                  v-for="capability in agentForm.capabilities.slice(0, 4)" 
                  :key="capability"
                  class="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded-full"
                >
                  {{ capability }}
                </span>
                <span 
                  v-if="agentForm.capabilities.length > 4"
                  class="px-2 py-1 text-xs bg-gray-100 text-gray-500 rounded-full"
                >
                  +{{ agentForm.capabilities.length - 4 }}
                </span>
              </div>
            </div>

            <!-- Mock activity -->
            <div class="space-y-2">
              <h4 class="text-sm font-medium text-gray-700">Recent Activity</h4>
              <div class="space-y-1">
                <div class="flex items-center space-x-2 text-xs text-gray-500">
                  <div class="w-1 h-1 bg-green-400 rounded-full"></div>
                  <span>Initialized successfully</span>
                  <span class="ml-auto">just now</span>
                </div>
                <div class="flex items-center space-x-2 text-xs text-gray-500">
                  <div class="w-1 h-1 bg-blue-400 rounded-full"></div>
                  <span>Awaiting first task</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <p class="text-center text-gray-600 mt-6 text-sm">
          This is how your agent will appear in the dashboard
        </p>
      </div>
    </div>

    <!-- Action buttons -->
    <div class="absolute bottom-8 left-8 right-8 flex justify-between items-center lg:hidden">
      <button
        @click="handleBack"
        class="flex items-center space-x-2 px-6 py-3 text-gray-600 hover:text-gray-800 transition-colors"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
        </svg>
        <span>Back</span>
      </button>
      
      <button
        @click="handleCreateAgent"
        :disabled="!canCreateAgent || isCreating"
        class="flex items-center space-x-2 px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed hover:from-blue-600 hover:to-purple-700 transition-all"
      >
        <span>{{ isCreating ? 'Creating...' : 'Create Agent' }}</span>
        <svg v-if="!isCreating" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
      </button>
    </div>
    
    <!-- Desktop action buttons -->
    <div class="hidden lg:flex absolute bottom-12 left-12 right-12 justify-between">
      <button
        @click="handleBack"
        class="flex items-center space-x-2 px-6 py-3 text-gray-600 hover:text-gray-800 transition-colors"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
        </svg>
        <span class="font-medium">Back</span>
      </button>
      
      <button
        @click="handleCreateAgent"
        :disabled="!canCreateAgent || isCreating"
        class="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
      >
        <span class="font-semibold">{{ isCreating ? 'Creating Agent...' : 'Create Agent' }}</span>
        <svg v-if="!isCreating" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useOnboardingStore } from '@/stores/onboarding'

interface Props {
  userData?: any
  stepData?: any
}

interface AgentTemplate {
  id: string
  name: string
  type: string
  icon: string
  color: string
  description: string
  capabilities: string[]
  defaultPrompt: string
  recommended?: boolean
}

interface AgentForm {
  name: string
  description: string
  capabilities: string[]
  priority: string
  templateId: string
}

const props = defineProps<Props>()
const emit = defineEmits(['next', 'back', 'update-data', 'analytics-event'])

const onboardingStore = useOnboardingStore()

// State
const selectedTemplate = ref<AgentTemplate | null>(null)
const isCreating = ref(false)
const error = ref<string | null>(null)
const stepStartTime = ref<number>(Date.now())

// Form data
const agentForm = ref<AgentForm>({
  name: '',
  description: '',
  capabilities: [],
  priority: 'medium',
  templateId: ''
})

// Agent templates based on user goals
const allTemplates = ref<AgentTemplate[]>([
  {
    id: 'workflow_automator',
    name: 'Workflow Automator',
    type: 'Process Automation',
    icon: '⚡',
    color: '#3B82F6',
    description: 'Streamlines repetitive tasks and automates complex workflows',
    capabilities: ['Task Automation', 'Process Monitoring', 'Error Handling', 'Scheduling', 'Integration Management'],
    defaultPrompt: 'Automate repetitive tasks and optimize workflow efficiency',
    recommended: true
  },
  {
    id: 'data_analyst',
    name: 'Data Analyst',
    type: 'Analytics & Insights',
    icon: '📊',
    color: '#10B981',
    description: 'Analyzes data patterns and generates actionable insights',
    capabilities: ['Data Analysis', 'Report Generation', 'Trend Identification', 'Dashboard Creation', 'Predictive Modeling'],
    defaultPrompt: 'Analyze data trends and provide business insights'
  },
  {
    id: 'operations_manager',
    name: 'Operations Manager',
    type: 'Operations & Scaling',
    icon: '🎯',
    color: '#8B5CF6',
    description: 'Manages operational tasks and helps scale business processes',
    capabilities: ['Resource Management', 'Performance Monitoring', 'Capacity Planning', 'Quality Assurance', 'Team Coordination'],
    defaultPrompt: 'Optimize operations and manage scaling requirements'
  },
  {
    id: 'quality_controller',
    name: 'Quality Controller',
    type: 'Quality Assurance',
    icon: '✅',
    color: '#F59E0B',
    description: 'Monitors processes and reduces errors through systematic checks',
    capabilities: ['Quality Monitoring', 'Error Detection', 'Process Validation', 'Compliance Checking', 'Issue Resolution'],
    defaultPrompt: 'Ensure quality standards and minimize operational errors'
  },
  {
    id: 'business_intelligence',
    name: 'Business Intelligence Agent',
    type: 'Strategic Analysis',
    icon: '🧠',
    color: '#EF4444',
    description: 'Provides strategic insights and supports data-driven decisions',
    capabilities: ['Business Analysis', 'Strategic Planning', 'Market Research', 'KPI Tracking', 'Competitive Analysis'],
    defaultPrompt: 'Generate business intelligence and strategic recommendations'
  },
  {
    id: 'cost_optimizer',
    name: 'Cost Optimizer',
    type: 'Resource Optimization',
    icon: '💰',
    color: '#06B6D4',
    description: 'Identifies cost-saving opportunities and optimizes resource usage',
    capabilities: ['Cost Analysis', 'Resource Optimization', 'Budget Management', 'Efficiency Tracking', 'ROI Calculation'],
    defaultPrompt: 'Optimize costs and improve resource utilization'
  }
])

// Computed properties
const recommendedTemplates = computed(() => {
  const userGoals = props.userData?.goals || []
  
  // Score templates based on user goals
  const scoredTemplates = allTemplates.value.map(template => {
    let score = 0
    let recommended = false
    
    // Goal-based scoring
    if (userGoals.includes('automate_workflows') && template.id === 'workflow_automator') {
      score += 10
      recommended = true
    }
    if (userGoals.includes('data_insights') && template.id === 'data_analyst') {
      score += 10
      recommended = true
    }
    if (userGoals.includes('scale_operations') && template.id === 'operations_manager') {
      score += 10
      recommended = true
    }
    if (userGoals.includes('reduce_errors') && template.id === 'quality_controller') {
      score += 10
      recommended = true
    }
    if (userGoals.includes('improve_efficiency')) {
      if (['workflow_automator', 'operations_manager'].includes(template.id)) {
        score += 7
        recommended = true
      }
    }
    if (userGoals.includes('cost_reduction') && template.id === 'cost_optimizer') {
      score += 10
      recommended = true
    }
    
    return { ...template, score, recommended }
  })
  
  // Sort by score and return top 4
  return scoredTemplates
    .sort((a, b) => b.score - a.score)
    .slice(0, 4)
})

const formattedGoals = computed(() => {
  const goals = props.userData?.goals || []
  const goalLabels: { [key: string]: string } = {
    'automate_workflows': 'Automate Workflows',
    'improve_efficiency': 'Improve Efficiency',
    'scale_operations': 'Scale Operations',
    'reduce_errors': 'Reduce Errors',
    'data_insights': 'Get Data Insights',
    'cost_reduction': 'Reduce Costs'
  }
  
  return goals.map((goal: string) => goalLabels[goal] || goal).join(', ')
})

const canCreateAgent = computed(() => {
  return selectedTemplate.value && 
         agentForm.value.name.trim().length >= 2 &&
         agentForm.value.capabilities.length > 0
})

// Event handlers
const selectTemplate = (template: AgentTemplate) => {
  selectedTemplate.value = template
  
  // Pre-populate form with template defaults
  agentForm.value = {
    name: template.name,
    description: template.defaultPrompt,
    capabilities: [...template.capabilities.slice(0, 3)], // Pre-select first 3 capabilities
    priority: 'medium',
    templateId: template.id
  }
  
  emit('analytics-event', 'template_selected', {
    templateId: template.id,
    templateName: template.name,
    recommended: template.recommended
  })
}

const handleFormChange = () => {
  emit('update-data', {
    agentTemplate: selectedTemplate.value?.id,
    agentForm: agentForm.value
  })
  
  emit('analytics-event', 'agent_form_updated', {
    templateId: selectedTemplate.value?.id,
    hasName: agentForm.value.name.length > 0,
    hasDescription: agentForm.value.description.length > 0,
    capabilityCount: agentForm.value.capabilities.length
  })
}

const handleCreateAgent = async () => {
  if (!canCreateAgent.value) return
  
  isCreating.value = true
  error.value = null
  
  try {
    const agentData = {
      name: agentForm.value.name.trim(),
      type: selectedTemplate.value!.type,
      description: agentForm.value.description || selectedTemplate.value!.defaultPrompt,
      capabilities: agentForm.value.capabilities,
      priority: agentForm.value.priority,
      template_id: selectedTemplate.value!.id,
      metadata: {
        created_via: 'onboarding',
        template_name: selectedTemplate.value!.name,
        user_goals: props.userData?.goals
      }
    }
    
    const createdAgent = await onboardingStore.createAgent(agentData)
    
    emit('analytics-event', 'agent_created', {
      agentId: createdAgent.id,
      templateId: selectedTemplate.value!.id,
      timeSpent: Date.now() - stepStartTime.value,
      capabilities: agentForm.value.capabilities.length
    })
    
    emit('update-data', { createdAgent })
    emit('next')
    
  } catch (err: any) {
    error.value = err.message || 'Failed to create agent. Please try again.'
    
    emit('analytics-event', 'agent_creation_error', {
      error: error.value,
      templateId: selectedTemplate.value?.id
    })
  } finally {
    isCreating.value = false
  }
}

const handleBack = () => {
  emit('back')
}

// Lifecycle
onMounted(() => {
  stepStartTime.value = Date.now()
  
  // Auto-select recommended template if there's a clear winner
  const topRecommended = recommendedTemplates.value.find(t => t.recommended)
  if (topRecommended && recommendedTemplates.value.filter(t => t.recommended).length === 1) {
    setTimeout(() => selectTemplate(topRecommended), 500)
  }
  
  emit('analytics-event', 'agent_creation_viewed', {
    recommendedTemplateCount: recommendedTemplates.value.filter(t => t.recommended).length,
    userGoals: props.userData?.goals
  })
})

// Watch for template selection to auto-scroll on mobile
watch(selectedTemplate, (newTemplate) => {
  if (newTemplate && window.innerWidth < 1024) {
    setTimeout(() => {
      const customization = document.querySelector('.bg-gray-50')
      customization?.scrollIntoView({ behavior: 'smooth' })
    }, 300)
  }
})
</script>

<style scoped>
/* Custom animations */
@keyframes bounce-in {
  0% {
    opacity: 0;
    transform: scale(0.3);
  }
  50% {
    opacity: 1;
    transform: scale(1.05);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
}

.template-selected {
  animation: bounce-in 0.6s ease-out;
}

/* Mobile scrolling optimization */
@media (max-width: 1024px) {
  .space-y-6 > div {
    scroll-margin-top: 2rem;
  }
}

/* Accessibility improvements */
input:focus,
select:focus,
textarea:focus {
  outline: none;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Checkbox custom styling */
input[type="checkbox"]:checked {
  background-color: #3b82f6;
  border-color: #3b82f6;
}

/* Template selection hover effects */
.cursor-pointer:hover {
  transform: translateY(-2px);
}

/* Loading spinner for agent creation */
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.animate-spin {
  animation: spin 1s linear infinite;
}
</style>