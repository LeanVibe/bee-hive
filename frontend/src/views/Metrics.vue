<template>
  <div class="p-6">
    <div class="mb-6">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white">Metrics</h1>
      <p class="text-gray-600 dark:text-gray-300">System performance and business metrics</p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- HTTP Metrics -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h2 class="text-xl font-semibold mb-4">HTTP Metrics</h2>
        <div class="space-y-4">
          <div v-for="metric in httpMetrics" :key="metric.name" class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-300">{{ metric.label }}</span>
            <span class="font-mono text-sm">{{ metric.value }}</span>
          </div>
        </div>
      </div>

      <!-- Agent Metrics -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h2 class="text-xl font-semibold mb-4">Agent Metrics</h2>
        <div class="space-y-4">
          <div v-for="metric in agentMetrics" :key="metric.name" class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-300">{{ metric.label }}</span>
            <span class="font-mono text-sm">{{ metric.value }}</span>
          </div>
        </div>
      </div>

      <!-- System Metrics -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h2 class="text-xl font-semibold mb-4">System Resources</h2>
        <div class="space-y-4">
          <div v-for="metric in systemMetrics" :key="metric.name" class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-300">{{ metric.label }}</span>
            <span class="font-mono text-sm">{{ metric.value }}</span>
          </div>
        </div>
      </div>

      <!-- Business Metrics -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h2 class="text-xl font-semibold mb-4">Business Metrics</h2>
        <div class="space-y-4">
          <div v-for="metric in businessMetrics" :key="metric.name" class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-300">{{ metric.label }}</span>
            <span class="font-mono text-sm">{{ metric.value }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

interface Metric {
  name: string
  label: string
  value: string
}

const httpMetrics = ref<Metric[]>([])
const agentMetrics = ref<Metric[]>([])
const systemMetrics = ref<Metric[]>([])
const businessMetrics = ref<Metric[]>([])

onMounted(() => {
  // Mock metrics data
  httpMetrics.value = [
    { name: 'requests_total', label: 'Total Requests', value: '12,345' },
    { name: 'error_rate', label: 'Error Rate', value: '0.12%' },
    { name: 'response_time_p95', label: 'P95 Response Time', value: '125ms' }
  ]

  agentMetrics.value = [
    { name: 'active_agents', label: 'Active Agents', value: '8' },
    { name: 'tool_executions', label: 'Tool Executions', value: '2,456' },
    { name: 'success_rate', label: 'Success Rate', value: '98.7%' }
  ]

  systemMetrics.value = [
    { name: 'cpu_usage', label: 'CPU Usage', value: '42%' },
    { name: 'memory_usage', label: 'Memory Usage', value: '68%' },
    { name: 'disk_usage', label: 'Disk Usage', value: '35%' }
  ]

  businessMetrics.value = [
    { name: 'sessions_total', label: 'Total Sessions', value: '1,234' },
    { name: 'avg_session_duration', label: 'Avg Session Duration', value: '8m 32s' },
    { name: 'tasks_completed', label: 'Tasks Completed', value: '9,876' }
  ]
})
</script>