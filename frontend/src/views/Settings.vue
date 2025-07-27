<template>
  <div class="p-6">
    <div class="mb-6">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white">Settings</h1>
      <p class="text-gray-600 dark:text-gray-300">Configure observability dashboard preferences</p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Display Settings -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h2 class="text-xl font-semibold mb-4">Display Settings</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Theme
            </label>
            <select 
              v-model="settings.theme"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="auto">Auto</option>
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Refresh Interval
            </label>
            <select 
              v-model="settings.refreshInterval"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
              <option :value="1000">1 second</option>
              <option :value="5000">5 seconds</option>
              <option :value="10000">10 seconds</option>
              <option :value="30000">30 seconds</option>
            </select>
          </div>

          <div>
            <label class="flex items-center">
              <input 
                type="checkbox" 
                v-model="settings.soundEnabled"
                class="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              >
              <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable sound notifications</span>
            </label>
          </div>
        </div>
      </div>

      <!-- Alert Settings -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h2 class="text-xl font-semibold mb-4">Alert Settings</h2>
        <div class="space-y-4">
          <div>
            <label class="flex items-center">
              <input 
                type="checkbox" 
                v-model="settings.alertsEnabled"
                class="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              >
              <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable desktop notifications</span>
            </label>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Minimum Alert Severity
            </label>
            <select 
              v-model="settings.minAlertSeverity"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="critical">Critical</option>
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Alert Retention (days)
            </label>
            <input 
              type="number" 
              v-model="settings.alertRetentionDays"
              min="1" 
              max="365"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
          </div>
        </div>
      </div>

      <!-- Performance Settings -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h2 class="text-xl font-semibold mb-4">Performance Settings</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Max Events to Display
            </label>
            <input 
              type="number" 
              v-model="settings.maxEventsDisplay"
              min="10" 
              max="1000"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Chart Animation Duration (ms)
            </label>
            <input 
              type="number" 
              v-model="settings.chartAnimationDuration"
              min="0" 
              max="2000"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
          </div>

          <div>
            <label class="flex items-center">
              <input 
                type="checkbox" 
                v-model="settings.autoScroll"
                class="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              >
              <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Auto-scroll to new events</span>
            </label>
          </div>
        </div>
      </div>

      <!-- Export Settings -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h2 class="text-xl font-semibold mb-4">Export Settings</h2>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Default Export Format
            </label>
            <select 
              v-model="settings.exportFormat"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
              <option value="json">JSON</option>
              <option value="csv">CSV</option>
              <option value="xlsx">Excel</option>
            </select>
          </div>

          <div class="flex space-x-4">
            <button 
              @click="exportSettings"
              class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Export Settings
            </button>
            <button 
              @click="importSettings"
              class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
            >
              Import Settings
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Save Changes -->
    <div class="mt-6 flex justify-end space-x-4">
      <button 
        @click="resetSettings"
        class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
      >
        Reset to Defaults
      </button>
      <button 
        @click="saveSettings"
        class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
      >
        Save Changes
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

interface Settings {
  theme: string
  refreshInterval: number
  soundEnabled: boolean
  alertsEnabled: boolean
  minAlertSeverity: string
  alertRetentionDays: number
  maxEventsDisplay: number
  chartAnimationDuration: number
  autoScroll: boolean
  exportFormat: string
}

const settings = ref<Settings>({
  theme: 'auto',
  refreshInterval: 5000,
  soundEnabled: false,
  alertsEnabled: true,
  minAlertSeverity: 'warning',
  alertRetentionDays: 30,
  maxEventsDisplay: 100,
  chartAnimationDuration: 300,
  autoScroll: true,
  exportFormat: 'json'
})

const saveSettings = () => {
  localStorage.setItem('observability-settings', JSON.stringify(settings.value))
  alert('Settings saved successfully!')
}

const resetSettings = () => {
  settings.value = {
    theme: 'auto',
    refreshInterval: 5000,
    soundEnabled: false,
    alertsEnabled: true,
    minAlertSeverity: 'warning',
    alertRetentionDays: 30,
    maxEventsDisplay: 100,
    chartAnimationDuration: 300,
    autoScroll: true,
    exportFormat: 'json'
  }
}

const exportSettings = () => {
  const blob = new Blob([JSON.stringify(settings.value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'observability-settings.json'
  a.click()
  URL.revokeObjectURL(url)
}

const importSettings = () => {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.json'
  input.onchange = (event) => {
    const file = (event.target as HTMLInputElement).files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const imported = JSON.parse(e.target?.result as string)
          settings.value = { ...settings.value, ...imported }
          alert('Settings imported successfully!')
        } catch (error) {
          alert('Error importing settings: Invalid JSON file')
        }
      }
      reader.readAsText(file)
    }
  }
  input.click()
}

onMounted(() => {
  const saved = localStorage.getItem('observability-settings')
  if (saved) {
    try {
      settings.value = { ...settings.value, ...JSON.parse(saved) }
    } catch (error) {
      console.error('Error loading saved settings:', error)
    }
  }
})
</script>