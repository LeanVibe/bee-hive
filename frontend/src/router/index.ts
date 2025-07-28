import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

// Lazy load components for better performance
const Dashboard = () => import('@/views/Dashboard.vue')
const CoordinationDashboard = () => import('@/views/CoordinationDashboard.vue')
const AgentGraphDashboard = () => import('@/views/AgentGraphDashboard.vue')
const Metrics = () => import('@/views/Metrics.vue')
const Events = () => import('@/views/Events.vue')
const Settings = () => import('@/views/Settings.vue')
const NotFound = () => import('@/views/NotFound.vue')

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Dashboard',
    component: Dashboard,
    meta: {
      title: 'Dashboard - LeanVibe Agent Hive',
      description: 'Real-time overview of agent activities and system health',
    },
  },
  {
    path: '/coordination',
    name: 'Coordination',
    component: CoordinationDashboard,
    meta: {
      title: 'Coordination Dashboard - LeanVibe Agent Hive',
      description: 'Unified multi-agent coordination, communication analysis, and system monitoring',
    },
  },
  {
    path: '/coordination/:tab',
    name: 'CoordinationWithTab',
    component: CoordinationDashboard,
    meta: {
      title: 'Coordination Dashboard - LeanVibe Agent Hive',
      description: 'Unified multi-agent coordination, communication analysis, and system monitoring',
    },
  },
  {
    path: '/agent-graph',
    name: 'AgentGraph',
    component: AgentGraphDashboard,
    meta: {
      title: 'Agent Graph - LeanVibe Agent Hive',
      description: 'Real-time multi-agent coordination and performance visualization',
    },
  },
  {
    path: '/metrics',
    name: 'Metrics',
    component: Metrics,
    meta: {
      title: 'Metrics - LeanVibe Agent Hive',
      description: 'Performance metrics and system analytics',
    },
  },
  {
    path: '/events',
    name: 'Events',
    component: Events,
    meta: {
      title: 'Events - LeanVibe Agent Hive',
      description: 'Event timeline and detailed logs',
    },
  },
  {
    path: '/settings',
    name: 'Settings',
    component: Settings,
    meta: {
      title: 'Settings - LeanVibe Agent Hive',
      description: 'Dashboard configuration and preferences',
    },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: NotFound,
    meta: {
      title: 'Page Not Found - LeanVibe Agent Hive',
    },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(_to, _from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    } else {
      return { top: 0 }
    }
  },
})

// Route guards
router.beforeEach((to, _from, next) => {
  // Update document title
  if (to.meta.title) {
    document.title = to.meta.title as string
  }
  
  // Update meta description
  if (to.meta.description) {
    const metaDescription = document.querySelector('meta[name="description"]')
    if (metaDescription) {
      metaDescription.setAttribute('content', to.meta.description as string)
    }
  }
  
  next()
})

export default router