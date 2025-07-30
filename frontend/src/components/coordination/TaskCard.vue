<template>
  <div
    class="task-card"
    :class="{
      'dragging': isDragging,
      'selected': isSelected,
      'cursor-grab': draggable && !isDragging,
      'cursor-grabbing': isDragging
    }"
    :draggable="draggable"
    @dragstart="onDragStart"
    @dragend="onDragEnd"
    @click="onClick"
  >
    <!-- Task Header -->
    <div class="task-header flex items-start justify-between mb-3">
      <div class="flex-1">
        <h4 class="task-title font-medium text-sm text-slate-900 dark:text-white">
          {{ task.task_title }}
        </h4>
        <p class="task-type text-xs text-slate-500 dark:text-slate-400 mt-1">
          {{ formatTaskType(task.task_type) }}
        </p>
      </div>
      
      <div class="task-actions flex items-center space-x-2">
        <!-- Priority Badge -->
        <span 
          class="priority-badge px-2 py-1 text-xs rounded-full font-medium"
          :class="getPriorityClass(task.priority)"
        >
          {{ task.priority }}
        </span>
        
        <!-- More Options -->
        <div class="relative">
          <button
            @click.stop="toggleOptions"
            class="options-button p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
          >
            <EllipsisVerticalIcon class="w-4 h-4 text-slate-400" />
          </button>
          
          <!-- Options Dropdown -->
          <div 
            v-if="showOptions"
            class="options-dropdown absolute right-0 top-full mt-1 bg-white dark:bg-slate-800 rounded-md shadow-lg border border-slate-200 dark:border-slate-700 z-10"
          >
            <button
              @click.stop="editTask"
              class="option-item w-full text-left px-3 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 flex items-center"
            >
              <PencilIcon class="w-4 h-4 mr-2" />
              Edit
            </button>
            <button
              @click.stop="duplicateTask"
              class="option-item w-full text-left px-3 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 flex items-center"
            >
              <DocumentDuplicateIcon class="w-4 h-4 mr-2" />
              Duplicate
            </button>
            <button
              @click.stop="deleteTask"
              class="option-item w-full text-left px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center"
            >
              <TrashIcon class="w-4 h-4 mr-2" />
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Task Description -->
    <p class="task-description text-xs text-slate-600 dark:text-slate-400 mb-3 line-clamp-2">
      {{ task.task_description }}
    </p>

    <!-- Required Capabilities -->
    <div v-if="task.required_capabilities?.length" class="capabilities-section mb-3">
      <p class="capabilities-label text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
        Required Skills:
      </p>
      <div class="capabilities-list flex flex-wrap gap-1">
        <span 
          v-for="capability in task.required_capabilities.slice(0, 3)"
          :key="capability"
          class="capability-tag inline-block px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded"
        >
          {{ capability }}
        </span>
        <span 
          v-if="task.required_capabilities.length > 3"
          class="capability-tag inline-block px-2 py-1 text-xs bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 rounded"
        >
          +{{ task.required_capabilities.length - 3 }} more
        </span>
      </div>
    </div>

    <!-- Task Metadata -->
    <div class="task-metadata grid grid-cols-2 gap-2 text-xs">
      <div class="metadata-item">
        <span class="metadata-label text-slate-500 dark:text-slate-400">Effort:</span>
        <span class="metadata-value ml-1 font-medium">
          {{ task.estimated_effort_hours ? `${task.estimated_effort_hours}h` : 'N/A' }}
        </span>
      </div>
      <div class="metadata-item">
        <span class="metadata-label text-slate-500 dark:text-slate-400">Deadline:</span>
        <span class="metadata-value ml-1 font-medium" :class="getDeadlineClass(task.deadline)">
          {{ formatDeadline(task.deadline) }}
        </span>
      </div>
    </div>

    <!-- Dependencies Indicator -->
    <div v-if="task.dependencies?.length" class="dependencies-indicator mt-2 pt-2 border-t border-slate-200 dark:border-slate-700">
      <div class="flex items-center text-xs text-slate-500 dark:text-slate-400">
        <LinkIcon class="w-3 h-3 mr-1" />
        <span>{{ task.dependencies.length }} dependencies</span>
      </div>
    </div>

    <!-- Drag Handle -->
    <div 
      v-if="draggable"
      class="drag-handle absolute top-2 left-2 opacity-0 group-hover:opacity-100 transition-opacity"
    >
      <Bars3Icon class="w-4 h-4 text-slate-400" />
    </div>

    <!-- Selection Indicator -->
    <div 
      v-if="isSelected"
      class="selection-indicator absolute -inset-0.5 bg-primary-500 rounded-lg opacity-20"
    ></div>

    <!-- Drag Preview -->
    <div 
      v-if="isDragging"
      class="drag-preview absolute inset-0 bg-primary-100 dark:bg-primary-900/20 rounded-lg border-2 border-primary-400 dark:border-primary-600"
    ></div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { format, isAfter, isBefore, addDays } from 'date-fns'

// Icons
import {
  EllipsisVerticalIcon,
  PencilIcon,
  DocumentDuplicateIcon,
  TrashIcon,
  LinkIcon,
  Bars3Icon
} from '@heroicons/vue/24/outline'

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
  dependencies?: string[]
  context_data?: Record<string, any>
}

interface Props {
  task: Task
  draggable?: boolean
  isDragging?: boolean
  isSelected?: boolean
}

interface Emits {
  (e: 'drag-start', task: Task): void
  (e: 'drag-end'): void
  (e: 'task-click', task: Task): void
  (e: 'edit-task', task: Task): void
  (e: 'duplicate-task', task: Task): void
  (e: 'delete-task', task: Task): void
}

const props = withDefaults(defineProps<Props>(), {
  draggable: false,
  isDragging: false,
  isSelected: false
})

const emit = defineEmits<Emits>()

// State
const showOptions = ref(false)

// Methods
const onDragStart = (event: DragEvent) => {
  if (!props.draggable) return
  
  event.dataTransfer?.setData('text/plain', props.task.id)
  emit('drag-start', props.task)
}

const onDragEnd = () => {
  emit('drag-end')
}

const onClick = () => {
  emit('task-click', props.task)
}

const toggleOptions = () => {
  showOptions.value = !showOptions.value
}

const editTask = () => {
  showOptions.value = false
  emit('edit-task', props.task)
}

const duplicateTask = () => {
  showOptions.value = false
  emit('duplicate-task', props.task)
}

const deleteTask = () => {
  showOptions.value = false
  emit('delete-task', props.task)
}

const formatTaskType = (taskType: string) => {
  return taskType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const formatDeadline = (deadline?: string) => {
  if (!deadline) return 'No deadline'
  
  const deadlineDate = new Date(deadline)
  const now = new Date()
  
  if (isBefore(deadlineDate, now)) {
    return 'Overdue'
  } else if (isBefore(deadlineDate, addDays(now, 1))) {
    return 'Today'
  } else if (isBefore(deadlineDate, addDays(now, 2))) {
    return 'Tomorrow'
  } else if (isBefore(deadlineDate, addDays(now, 7))) {
    return format(deadlineDate, 'EEE')
  } else {
    return format(deadlineDate, 'MMM dd')
  }
}

const getPriorityClass = (priority: string) => {
  const classes = {
    'CRITICAL': 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800',
    'HIGH': 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 border border-orange-200 dark:border-orange-800',
    'MEDIUM': 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 border border-yellow-200 dark:border-yellow-800', 
    'LOW': 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800'
  }
  return classes[priority] || 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-400'
}

const getDeadlineClass = (deadline?: string) => {
  if (!deadline) return 'text-slate-500 dark:text-slate-400'
  
  const deadlineDate = new Date(deadline)
  const now = new Date()
  
  if (isBefore(deadlineDate, now)) {
    return 'text-red-600 dark:text-red-400 font-medium'
  } else if (isBefore(deadlineDate, addDays(now, 1))) {
    return 'text-orange-600 dark:text-orange-400 font-medium'
  } else if (isBefore(deadlineDate, addDays(now, 3))) {
    return 'text-yellow-600 dark:text-yellow-400'
  } else {
    return 'text-slate-600 dark:text-slate-300'
  }
}

// Close options dropdown when clicking outside
const handleClickOutside = (event: Event) => {
  const target = event.target as Element
  if (!target.closest('.options-dropdown') && !target.closest('.options-button')) {
    showOptions.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.task-card {
  @apply relative p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 hover:shadow-md transition-all duration-200 group;
}

.task-card.dragging {
  @apply opacity-50 transform rotate-3 scale-105 shadow-lg;
}

.task-card.selected {
  @apply ring-2 ring-primary-500 border-primary-300 dark:border-primary-600;
}

.task-card:hover {
  @apply transform -translate-y-1 shadow-lg border-slate-300 dark:border-slate-600;
}

.line-clamp-2 {
  overflow: hidden;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.options-dropdown {
  @apply min-w-32;
}

.option-item {
  @apply transition-colors duration-150;
}

.option-item:first-child {
  @apply rounded-t-md;  
}

.option-item:last-child {
  @apply rounded-b-md;
}

.capability-tag {
  @apply transition-all duration-150 hover:scale-105;
}

.selection-indicator {
  pointer-events: none;
}

.drag-preview {
  pointer-events: none;
}

@keyframes wiggle {
  0%, 100% { transform: rotate(0deg); }
  25% { transform: rotate(1deg); }
  75% { transform: rotate(-1deg); }
}

.task-card.dragging {
  animation: wiggle 0.5s ease-in-out infinite;
}
</style>