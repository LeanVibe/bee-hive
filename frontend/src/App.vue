<template>
  <div id="app" class="min-h-screen">
    <!-- Navigation -->
    <nav class="glass-card sticky top-0 z-50 border-b border-slate-200/50 dark:border-slate-700/50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <!-- Logo and title -->
          <div class="flex items-center space-x-4">
            <div class="flex-shrink-0">
              <h1 class="text-xl font-bold text-slate-900 dark:text-white">
                LeanVibe Agent Hive
              </h1>
            </div>
            <div class="hidden md:block">
              <div class="flex items-center space-x-4">
                <router-link
                  v-for="link in navigation"
                  :key="link.name"
                  :to="link.to"
                  class="nav-link"
                  :class="{ 'nav-link-active': $route.path === link.to }"
                >
                  <component :is="link.icon" class="w-4 h-4 inline mr-2" />
                  {{ link.name }}
                </router-link>
              </div>
            </div>
          </div>
          
          <!-- Status indicator and controls -->
          <div class="flex items-center space-x-4">
            <!-- Connection status -->
            <div class="flex items-center space-x-2">
              <div
                :class="[
                  'status-dot',
                  connectionStore.isConnected ? 'status-healthy' : 'status-danger'
                ]"
              ></div>
              <span class="text-sm font-medium">
                {{ connectionStore.isConnected ? 'Connected' : 'Disconnected' }}
              </span>
            </div>
            
            <!-- Theme toggle -->
            <button
              @click="toggleTheme"
              class="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
              :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
            >
              <SunIcon v-if="isDark" class="w-5 h-5" />
              <MoonIcon v-else class="w-5 h-5" />
            </button>
            
            <!-- Mobile menu button -->
            <button
              @click="mobileMenuOpen = !mobileMenuOpen"
              class="md:hidden p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700"
            >
              <Bars3Icon v-if="!mobileMenuOpen" class="w-5 h-5" />
              <XMarkIcon v-else class="w-5 h-5" />
            </button>
          </div>
        </div>
        
        <!-- Mobile menu -->
        <div v-show="mobileMenuOpen" class="md:hidden py-4 border-t border-slate-200/50 dark:border-slate-700/50">
          <div class="flex flex-col space-y-2">
            <router-link
              v-for="link in navigation"
              :key="link.name"
              :to="link.to"
              class="nav-link"
              :class="{ 'nav-link-active': $route.path === link.to }"
              @click="mobileMenuOpen = false"
            >
              <component :is="link.icon" class="w-4 h-4 inline mr-2" />
              {{ link.name }}
            </router-link>
          </div>
        </div>
      </div>
    </nav>
    
    <!-- Main content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <router-view />
    </main>
    
    <!-- Global notifications -->
    <NotificationContainer />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useConnectionStore } from '@/stores/connection'
import { useNotificationStore } from '@/stores/notifications'
import NotificationContainer from '@/components/notifications/NotificationContainer.vue'
import {
  HomeIcon,
  ChartBarIcon,
  CogIcon,
  ExclamationTriangleIcon,
  SunIcon,
  MoonIcon,
  Bars3Icon,
  XMarkIcon,
  ShareIcon,
} from '@heroicons/vue/24/outline'

// Stores
const connectionStore = useConnectionStore()
const notificationStore = useNotificationStore()

// Local state
const mobileMenuOpen = ref(false)
const isDark = ref(false)

// Navigation links
const navigation = [
  { name: 'Dashboard', to: '/', icon: HomeIcon },
  { name: 'Agent Graph', to: '/agent-graph', icon: ShareIcon },
  { name: 'Metrics', to: '/metrics', icon: ChartBarIcon },
  { name: 'Events', to: '/events', icon: ExclamationTriangleIcon },
  { name: 'Settings', to: '/settings', icon: CogIcon },
]

// Theme management
const toggleTheme = () => {
  isDark.value = !isDark.value
  document.documentElement.classList.toggle('dark', isDark.value)
  localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
}

// Initialize theme
const initTheme = () => {
  const savedTheme = localStorage.getItem('theme')
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  
  isDark.value = savedTheme === 'dark' || (!savedTheme && prefersDark)
  document.documentElement.classList.toggle('dark', isDark.value)
}

// Lifecycle
onMounted(() => {
  initTheme()
  
  // Initialize WebSocket connection
  connectionStore.connect()
  
  // Show welcome notification
  notificationStore.addNotification({
    type: 'info',
    title: 'Dashboard Loaded',
    message: 'Real-time observability dashboard is ready',
    duration: 3000,
  })
})
</script>