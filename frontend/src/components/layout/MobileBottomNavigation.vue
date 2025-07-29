<template>
  <nav 
    class="fixed bottom-0 left-0 right-0 z-50 glass-card border-t border-slate-200/50 dark:border-slate-700/50"
    :class="{ 'safe-area-bottom': hasSafeArea }"
    role="navigation"
    aria-label="Main navigation"
  >
    <div class="grid grid-cols-5 h-16">
      <router-link
        v-for="item in navigationItems"
        :key="item.name"
        :to="item.to"
        class="flex flex-col items-center justify-center space-y-1 touch-44 transition-all duration-200"
        :class="[
          $route.path === item.to
            ? 'text-primary-600 dark:text-primary-400 bg-primary-50/50 dark:bg-primary-900/20'
            : 'text-slate-500 dark:text-slate-400 hover:text-primary-600 dark:hover:text-primary-400'
        ]"
        :aria-label="`Navigate to ${item.name}`"
        @click="handleNavigation(item)"
      >
        <component 
          :is="item.icon" 
          class="w-5 h-5 flex-shrink-0"
          :class="{
            'scale-110': $route.path === item.to
          }"
        />
        <span class="text-xs font-medium truncate max-w-[60px]">
          {{ item.name }}
        </span>
        
        <!-- Badge for notifications -->
        <div 
          v-if="item.badge && item.badge > 0"
          class="absolute -top-1 -right-1 min-w-[18px] h-[18px] bg-danger-500 text-white text-xs font-bold rounded-full flex items-center justify-center"
          :aria-label="`${item.badge} ${item.badgeLabel || 'notifications'}`"
        >
          {{ item.badge > 99 ? '99+' : item.badge }}
        </div>
      </router-link>
    </div>
    
    <!-- Haptic feedback indicator -->
    <div 
      v-if="showHapticFeedback"
      class="absolute inset-0 bg-primary-500/10 rounded-t-lg pointer-events-none"
      :class="{ 'animate-pulse': hapticActive }"
    ></div>
  </nav>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useNotificationStore } from '@/stores/notifications'
import {
  HomeIcon,
  ShareIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CogIcon,
} from '@heroicons/vue/24/outline'

interface NavigationItem {
  name: string
  to: string
  icon: any
  badge?: number
  badgeLabel?: string
}

const router = useRouter()
const notificationStore = useNotificationStore()

// State
const showHapticFeedback = ref(false)
const hapticActive = ref(false)
const hasSafeArea = ref(false)

// Navigation items with computed badges
const navigationItems = computed<NavigationItem[]>(() => [
  { 
    name: 'Home', 
    to: '/', 
    icon: HomeIcon 
  },
  { 
    name: 'Graph', 
    to: '/agent-graph', 
    icon: ShareIcon 
  },
  { 
    name: 'Metrics', 
    to: '/metrics', 
    icon: ChartBarIcon 
  },
  { 
    name: 'Events', 
    to: '/events', 
    icon: ExclamationTriangleIcon,
    badge: notificationStore.unreadCount,
    badgeLabel: 'unread events'
  },
  { 
    name: 'Settings', 
    to: '/settings', 
    icon: CogIcon 
  },
])

// Haptic feedback support
const triggerHapticFeedback = () => {
  if ('vibrate' in navigator) {
    navigator.vibrate(50) // Light vibration
  }
  
  // Visual feedback
  hapticActive.value = true
  setTimeout(() => {
    hapticActive.value = false
  }, 150)
}

// Navigation handler with haptic feedback
const handleNavigation = (item: NavigationItem) => {
  if (showHapticFeedback.value) {
    triggerHapticFeedback()
  }
  
  // Track navigation for analytics
  console.log(`📱 Mobile navigation to ${item.name}`)
}

// Check for safe area support
const checkSafeAreaSupport = () => {
  // Check if device has safe area insets (iPhone X+ style notches)
  const testElement = document.createElement('div')
  testElement.style.paddingBottom = 'env(safe-area-inset-bottom)'
  document.body.appendChild(testElement)
  
  const computedStyle = window.getComputedStyle(testElement)
  hasSafeArea.value = computedStyle.paddingBottom !== '0px'
  
  document.body.removeChild(testElement)
}

// Lifecycle
onMounted(() => {
  checkSafeAreaSupport()
  
  // Enable haptic feedback on touch devices
  showHapticFeedback.value = 'ontouchstart' in window
})
</script>

<style scoped>
/* Safe area utilities */
.safe-area-bottom {
  padding-bottom: max(1rem, env(safe-area-inset-bottom));
}

/* Active state animations */
.router-link-active {
  transform: scale(1.05);
}

/* Touch optimization */
@media (hover: none) and (pointer: coarse) {
  .touch-44:active {
    transform: scale(0.95);
    background-color: rgba(0, 0, 0, 0.05);
  }
}
</style>