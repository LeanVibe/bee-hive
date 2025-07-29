<template>
  <div class="test-runner p-6 max-w-6xl mx-auto">
    <!-- Header -->
    <div class="flex items-center justify-between mb-8">
      <div>
        <h1 class="text-3xl font-bold text-slate-900 dark:text-white">
          User Testing Framework
        </h1>
        <p class="mt-2 text-slate-600 dark:text-slate-400">
          Automated testing for accessibility, performance, and mobile responsiveness
        </p>
      </div>
      
      <div class="flex items-center space-x-4">
        <button
          @click="runAllTests"
          :disabled="isRunning"
          class="btn-primary"
          :class="{ 'opacity-50 cursor-not-allowed': isRunning }"
        >
          <PlayIcon v-if="!isRunning" class="w-4 h-4 mr-2" />
          <ArrowPathIcon v-else class="w-4 h-4 mr-2 animate-spin" />
          {{ isRunning ? 'Running Tests...' : 'Run All Tests' }}
        </button>
        
        <button
          @click="clearResults"
          :disabled="isRunning || results.length === 0"
          class="btn-secondary"
        >
          <TrashIcon class="w-4 h-4 mr-2" />
          Clear Results
        </button>
      </div>
    </div>

    <!-- Test Configuration -->
    <div class="glass-card rounded-xl p-6 mb-8">
      <h2 class="text-xl font-semibold mb-4 text-slate-900 dark:text-white">
        Test Configuration
      </h2>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <!-- Device Selection -->
        <div>
          <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Test Devices
          </label>
          <div class="space-y-2">
            <label
              v-for="device in deviceTargets"
              :key="device.name"
              class="flex items-center space-x-2"
            >
              <input
                type="checkbox"
                v-model="selectedDevices"
                :value="device.name"
                class="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              <span class="text-sm">{{ device.name }}</span>
              <span class="text-xs text-slate-500">({{ device.width }}x{{ device.height }})</span>
            </label>
          </div>
        </div>
        
        <!-- Test Categories -->
        <div>
          <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Test Categories
          </label>
          <div class="space-y-2">
            <label class="flex items-center space-x-2">
              <input
                type="checkbox"
                v-model="testAccessibility"
                class="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              <span class="text-sm">Accessibility (WCAG)</span>
            </label>
            <label class="flex items-center space-x-2">
              <input
                type="checkbox"
                v-model="testPerformance"
                class="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              <span class="text-sm">Performance</span>
            </label>
            <label class="flex items-center space-x-2">
              <input
                type="checkbox"
                v-model="testMobile"
                class="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              <span class="text-sm">Mobile Responsiveness</span>
            </label>
          </div>
        </div>
        
        <!-- Scenario Selection -->
        <div>
          <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Test Scenarios
          </label>
          <div class="space-y-2 max-h-32 overflow-y-auto">
            <label
              v-for="scenario in scenarios"
              :key="scenario.id"
              class="flex items-center space-x-2"
            >
              <input
                type="checkbox"
                v-model="selectedScenarios"
                :value="scenario.id"
                class="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              <span class="text-sm">{{ scenario.name }}</span>
              <span
                class="text-xs px-2 py-1 rounded"
                :class="getPriorityClass(scenario.priority)"
              >
                {{ scenario.priority }}
              </span>
            </label>
          </div>
        </div>
      </div>
    </div>

    <!-- Test Progress -->
    <div v-if="isRunning" class="glass-card rounded-xl p-6 mb-8">
      <h3 class="text-lg font-semibold mb-4 text-slate-900 dark:text-white">
        Test Progress
      </h3>
      
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <span class="text-sm text-slate-600 dark:text-slate-400">
            {{ currentTest || 'Initializing...' }}
          </span>
          <span class="text-sm font-medium">
            {{ completedTests }}/{{ totalTests }} tests
          </span>
        </div>
        
        <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
          <div
            class="bg-primary-600 h-2 rounded-full transition-all duration-300"
            :style="{ width: `${progressPercentage}%` }"
          ></div>
        </div>
      </div>
    </div>

    <!-- Test Results Summary -->
    <div v-if="results.length > 0 && !isRunning" class="glass-card rounded-xl p-6 mb-8">
      <h3 class="text-lg font-semibold mb-4 text-slate-900 dark:text-white">
        Test Results Summary
      </h3>
      
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <div class="text-center">
          <div class="text-3xl font-bold text-slate-900 dark:text-white">
            {{ summary.total }}
          </div>
          <div class="text-sm text-slate-600 dark:text-slate-400">Total Tests</div>
        </div>
        
        <div class="text-center">
          <div class="text-3xl font-bold text-green-600">
            {{ summary.passed }}
          </div>
          <div class="text-sm text-slate-600 dark:text-slate-400">Passed</div>
        </div>
        
        <div class="text-center">
          <div class="text-3xl font-bold text-red-600">
            {{ summary.failed }}
          </div>
          <div class="text-sm text-slate-600 dark:text-slate-400">Failed</div>
        </div>
        
        <div class="text-center">
          <div class="text-3xl font-bold text-blue-600">
            {{ summary.passRate.toFixed(1) }}%
          </div>
          <div class="text-sm text-slate-600 dark:text-slate-400">Pass Rate</div>
        </div>
      </div>
      
      <!-- Performance Metrics -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="text-center p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
          <div class="text-xl font-semibold text-slate-900 dark:text-white">
            {{ summary.avgDuration.toFixed(0) }}ms
          </div>
          <div class="text-sm text-slate-600 dark:text-slate-400">Avg Duration</div>
        </div>
        
        <div class="text-center p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
          <div class="text-xl font-semibold text-slate-900 dark:text-white">
            {{ summary.avgAccessibilityScore.toFixed(0) }}/100
          </div>
          <div class="text-sm text-slate-600 dark:text-slate-400">Accessibility Score</div>
        </div>
        
        <div class="text-center p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
          <div class="text-xl font-semibold text-slate-900 dark:text-white">
            {{ devicesCovered }}
          </div>
          <div class="text-sm text-slate-600 dark:text-slate-400">Devices Tested</div>
        </div>
      </div>
    </div>

    <!-- Detailed Results -->
    <div v-if="results.length > 0" class="space-y-6">
      <h3 class="text-xl font-semibold text-slate-900 dark:text-white">
        Detailed Results
      </h3>
      
      <div
        v-for="result in results"
        :key="`${result.scenarioId}-${result.timestamp}`"
        class="glass-card rounded-xl p-6"
      >
        <div class="flex items-start justify-between mb-4">
          <div>
            <h4 class="text-lg font-semibold text-slate-900 dark:text-white">
              {{ getScenarioName(result.scenarioId) }}
            </h4>
            <p class="text-sm text-slate-600 dark:text-slate-400">
              {{ formatDate(result.timestamp) }} • {{ result.duration.toFixed(0) }}ms
            </p>
          </div>
          
          <div class="flex items-center space-x-2">
            <span
              class="px-3 py-1 rounded-full text-sm font-medium"
              :class="getStatusClass(result.status)"
            >
              {{ result.status.toUpperCase() }}
            </span>
          </div>
        </div>
        
        <!-- Errors -->
        <div v-if="result.errors.length > 0" class="mb-4">
          <h5 class="text-sm font-medium text-red-700 dark:text-red-400 mb-2">
            Errors ({{ result.errors.length }})
          </h5>
          <div class="space-y-2">
            <div
              v-for="error in result.errors"
              :key="error.step"
              class="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg"
            >
              <div class="text-sm font-medium text-red-800 dark:text-red-200">
                Step {{ error.step }}: {{ error.message }}
              </div>
              <div v-if="error.stack" class="text-xs text-red-600 dark:text-red-400 mt-1 font-mono">
                {{ error.stack.split('\n')[0] }}
              </div>
            </div>
          </div>
        </div>
        
        <!-- Accessibility Score -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div class="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div class="text-sm font-medium text-slate-700 dark:text-slate-300">
              Accessibility Score
            </div>
            <div
              class="text-xl font-bold"
              :class="getAccessibilityScoreClass(result.accessibilityScore.score)"
            >
              {{ result.accessibilityScore.score }}/100
            </div>
          </div>
          
          <div class="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div class="text-sm font-medium text-slate-700 dark:text-slate-300">
              Violations
            </div>
            <div class="text-xl font-bold text-red-600">
              {{ result.accessibilityScore.violations.length }}
            </div>
          </div>
          
          <div class="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div class="text-sm font-medium text-slate-700 dark:text-slate-300">
              Warnings
            </div>
            <div class="text-xl font-bold text-yellow-600">
              {{ result.accessibilityScore.warnings.length }}
            </div>
          </div>
        </div>
        
        <!-- Accessibility Violations -->
        <div v-if="result.accessibilityScore.violations.length > 0" class="mb-4">
          <details class="group">
            <summary class="cursor-pointer text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Accessibility Violations ({{ result.accessibilityScore.violations.length }})
            </summary>
            <div class="space-y-2 mt-2">
              <div
                v-for="violation in result.accessibilityScore.violations"
                :key="violation.rule"
                class="p-3 border rounded-lg"
                :class="getViolationClass(violation.impact)"
              >
                <div class="flex items-start justify-between mb-2">
                  <div class="font-medium">{{ violation.rule }}</div>
                  <span
                    class="px-2 py-1 rounded text-xs font-medium"
                    :class="getImpactClass(violation.impact)"
                  >
                    {{ violation.impact }}
                  </span>
                </div>
                <div class="text-sm text-slate-600 dark:text-slate-400 mb-1">
                  {{ violation.description }}
                </div>
                <div class="text-xs text-slate-500 dark:text-slate-500">
                  Element: {{ violation.element }}
                </div>
                <div class="text-xs text-primary-600 dark:text-primary-400 mt-1">
                  💡 {{ violation.help }}
                </div>
              </div>
            </div>
          </details>
        </div>
        
        <!-- Performance Metrics -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div class="text-slate-600 dark:text-slate-400">Load Time</div>
            <div class="font-medium">{{ result.metrics.loadTime.toFixed(0) }}ms</div>
          </div>
          <div>
            <div class="text-slate-600 dark:text-slate-400">First Paint</div>
            <div class="font-medium">{{ result.metrics.firstContentfulPaint.toFixed(0) }}ms</div>
          </div>
          <div>
            <div class="text-slate-600 dark:text-slate-400">Memory</div>
            <div class="font-medium">{{ formatBytes(result.metrics.memoryUsage) }}</div>
          </div>
          <div>
            <div class="text-slate-600 dark:text-slate-400">Requests</div>
            <div class="font-medium">{{ result.metrics.networkRequests }}</div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Recommendations -->
    <div v-if="recommendations.length > 0" class="glass-card rounded-xl p-6 mt-8">
      <h3 class="text-lg font-semibold mb-4 text-slate-900 dark:text-white">
        Recommendations
      </h3>
      <ul class="space-y-2">
        <li
          v-for="recommendation in recommendations"
          :key="recommendation"
          class="flex items-start space-x-2"
        >
          <LightBulbIcon class="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
          <span class="text-sm text-slate-700 dark:text-slate-300">
            {{ recommendation }}
          </span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import UserTestingFramework, { type TestResult, type TestScenario, type DeviceTarget } from './UserTestingFramework'
import {
  PlayIcon,
  ArrowPathIcon,
  TrashIcon,
  LightBulbIcon,
} from '@heroicons/vue/24/outline'

// Testing framework instance
const testingFramework = new UserTestingFramework()

// Reactive state
const isRunning = ref(false)
const currentTest = ref('')
const completedTests = ref(0)
const totalTests = ref(0)
const results = ref<TestResult[]>([])
const scenarios = ref<TestScenario[]>([])
const deviceTargets = ref<DeviceTarget[]>([])

// Configuration
const selectedDevices = ref<string[]>(['Desktop', 'iPhone 12'])
const selectedScenarios = ref<string[]>([])
const testAccessibility = ref(true)
const testPerformance = ref(true)
const testMobile = ref(true)

// Computed properties
const progressPercentage = computed(() => {
  return totalTests.value > 0 ? (completedTests.value / totalTests.value) * 100 : 0
})

const summary = computed(() => {
  if (results.value.length === 0) {
    return {
      total: 0,
      passed: 0,
      failed: 0,
      passRate: 0,
      avgDuration: 0,
      avgAccessibilityScore: 0
    }
  }
  
  const total = results.value.length
  const passed = results.value.filter(r => r.status === 'passed').length
  const failed = results.value.filter(r => r.status === 'failed').length
  const avgDuration = results.value.reduce((sum, r) => sum + r.duration, 0) / total
  const avgAccessibilityScore = results.value.reduce((sum, r) => sum + r.accessibilityScore.score, 0) / total
  
  return {
    total,
    passed,
    failed,
    passRate: (passed / total) * 100,
    avgDuration,
    avgAccessibilityScore
  }
})

const devicesCovered = computed(() => {
  const devices = new Set(selectedDevices.value)
  return devices.size
})

const recommendations = computed(() => {
  const report = testingFramework.generateReport()
  return report.recommendations || []
})

// Methods
const runAllTests = async () => {
  if (isRunning.value) return
  
  isRunning.value = true
  results.value = []
  completedTests.value = 0
  totalTests.value = selectedScenarios.value.length * selectedDevices.value.length
  
  try {
    for (const scenarioId of selectedScenarios.value) {
      for (const deviceName of selectedDevices.value) {
        const device = deviceTargets.value.find(d => d.name === deviceName)
        if (!device) continue
        
        currentTest.value = `${getScenarioName(scenarioId)} on ${deviceName}`
        
        const result = await testingFramework.runScenario(scenarioId, device)
        results.value.push(result)
        completedTests.value++
      }
    }
  } catch (error) {
    console.error('Test execution failed:', error)
  } finally {
    isRunning.value = false
    currentTest.value = ''
  }
}

const clearResults = () => {
  results.value = []
  testingFramework.clearResults()
}

const getScenarioName = (scenarioId: string): string => {
  const scenario = scenarios.value.find(s => s.id === scenarioId)
  return scenario?.name || scenarioId
}

const getPriorityClass = (priority: string): string => {
  switch (priority) {
    case 'high':
      return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
    case 'medium':
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
    case 'low':
      return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
    default:
      return 'bg-slate-100 text-slate-800 dark:bg-slate-900/20 dark:text-slate-400'
  }
}

const getStatusClass = (status: string): string => {
  switch (status) {
    case 'passed':
      return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
    case 'failed':
      return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
    case 'skipped':
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
    default:
      return 'bg-slate-100 text-slate-800 dark:bg-slate-900/20 dark:text-slate-400'
  }
}

const getAccessibilityScoreClass = (score: number): string => {
  if (score >= 90) return 'text-green-600'
  if (score >= 70) return 'text-yellow-600'
  return 'text-red-600'
}

const getViolationClass = (impact: string): string => {
  switch (impact) {
    case 'critical':
      return 'border-red-300 bg-red-50 dark:border-red-700 dark:bg-red-900/20'
    case 'serious':
      return 'border-orange-300 bg-orange-50 dark:border-orange-700 dark:bg-orange-900/20'
    case 'moderate':
      return 'border-yellow-300 bg-yellow-50 dark:border-yellow-700 dark:bg-yellow-900/20'
    case 'minor':
      return 'border-blue-300 bg-blue-50 dark:border-blue-700 dark:bg-blue-900/20'
    default:
      return 'border-slate-300 bg-slate-50 dark:border-slate-700 dark:bg-slate-800'
  }
}

const getImpactClass = (impact: string): string => {
  switch (impact) {
    case 'critical':
      return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
    case 'serious':
      return 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400'
    case 'moderate':
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
    case 'minor':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400'
    default:
      return 'bg-slate-100 text-slate-800 dark:bg-slate-900/20 dark:text-slate-400'
  }
}

const formatDate = (timestamp: number): string => {
  return new Date(timestamp).toLocaleString()
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// Lifecycle
onMounted(() => {
  scenarios.value = testingFramework.getAllScenarios()
  deviceTargets.value = testingFramework.getDeviceTargets()
  
  // Select all scenarios by default
  selectedScenarios.value = scenarios.value.map(s => s.id)
})
</script>

<style scoped>
.btn-primary {
  @apply bg-primary-600 hover:bg-primary-700 focus:ring-primary-500 text-white;
  @apply px-4 py-2 rounded-lg font-medium transition-colors duration-200;
  @apply focus:outline-none focus:ring-2 focus:ring-offset-2;
  @apply flex items-center;
}

.btn-secondary {
  @apply bg-slate-200 hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600;
  @apply text-slate-700 dark:text-slate-200;
  @apply px-4 py-2 rounded-lg font-medium transition-colors duration-200;
  @apply focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-slate-500;
  @apply flex items-center;
}

.glass-card {
  @apply bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-white/20 dark:border-slate-700/50;
  @apply shadow-glass hover:shadow-glass-hover transition-all duration-300;
}
</style>