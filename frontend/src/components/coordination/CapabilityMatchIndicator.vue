<template>
  <div class="capability-match-indicator">
    <div class="flex items-center justify-between mb-2">
      <span class="text-sm font-medium text-gray-700 dark:text-gray-300">
        {{ capability }}
      </span>
      <div class="flex items-center space-x-2">
        <div 
          class="px-2 py-1 rounded text-xs font-medium"
          :class="getMatchLevelClass(matchLevel)"
        >
          {{ getMatchLevelText(matchLevel) }}
        </div>
        <span class="text-sm text-gray-500 dark:text-gray-400">
          {{ matchScore }}%
        </span>
      </div>
    </div>
    
    <!-- Progress Bar -->
    <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-3">
      <div 
        class="h-2 rounded-full transition-all duration-300"
        :class="getProgressBarClass(matchLevel)"
        :style="{ width: matchScore + '%' }"
      ></div>
    </div>
    
    <!-- Required vs Available -->
    <div v-if="showDetails" class="space-y-2 text-xs">
      <div class="flex justify-between">
        <span class="text-gray-600 dark:text-gray-400">Required:</span>
        <span class="font-medium text-gray-900 dark:text-gray-100">
          {{ requiredLevel || 'Any' }}
        </span>
      </div>
      <div class="flex justify-between">
        <span class="text-gray-600 dark:text-gray-400">Available:</span>
        <span class="font-medium text-gray-900 dark:text-gray-100">
          {{ availableLevel || 'None' }}
        </span>
      </div>
      <div v-if="description" class="text-gray-500 dark:text-gray-400 text-xs">
        {{ description }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

type MatchLevel = 'perfect' | 'good' | 'partial' | 'poor' | 'none'

interface Props {
  capability: string
  matchScore: number
  matchLevel?: MatchLevel
  requiredLevel?: string
  availableLevel?: string
  description?: string
  showDetails?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showDetails: false
})

const matchLevel = computed(() => {
  if (props.matchLevel) return props.matchLevel
  
  if (props.matchScore >= 90) return 'perfect'
  if (props.matchScore >= 75) return 'good'
  if (props.matchScore >= 50) return 'partial'
  if (props.matchScore >= 25) return 'poor'
  return 'none'
})

function getMatchLevelText(level: MatchLevel): string {
  switch (level) {
    case 'perfect': return 'Perfect'
    case 'good': return 'Good'
    case 'partial': return 'Partial'
    case 'poor': return 'Poor'
    case 'none': return 'None'
    default: return 'Unknown'
  }
}

function getMatchLevelClass(level: MatchLevel): string {
  switch (level) {
    case 'perfect': return 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100'
    case 'good': return 'bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-100'
    case 'partial': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100'
    case 'poor': return 'bg-orange-100 text-orange-800 dark:bg-orange-800 dark:text-orange-100'
    case 'none': return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100'
    default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100'
  }
}

function getProgressBarClass(level: MatchLevel): string {
  switch (level) {
    case 'perfect': return 'bg-green-500'
    case 'good': return 'bg-blue-500'
    case 'partial': return 'bg-yellow-500'
    case 'poor': return 'bg-orange-500'
    case 'none': return 'bg-gray-400'
    default: return 'bg-gray-400'
  }
}
</script>

<style scoped>
.capability-match-indicator {
  transition: all 0.2s ease-in-out;
}
</style>