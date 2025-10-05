<template>
  <div class="security-dashboard bg-white dark:bg-gray-800 rounded-lg shadow-sm">
    <!-- Header -->
    <div class="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
      <div>
        <h3 class="text-xl font-semibold text-gray-900 dark:text-white">
          Security Dashboard
        </h3>
        <p class="text-sm text-gray-500 dark:text-gray-400">
          Real-time threat monitoring and command blocking
        </p>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Security status indicator -->
        <div class="flex items-center space-x-2">
          <div 
            :class="[
              'w-3 h-3 rounded-full',
              getSecurityStatusColor()
            ]"
          ></div>
          <span class="text-sm font-medium" :class="getSecurityStatusTextColor()">
            {{ getSecurityStatusText() }}
          </span>
        </div>

        <!-- Auto refresh toggle -->
        <button
          @click="autoRefresh = !autoRefresh"
          :class="[
            'px-3 py-1 text-xs rounded-full transition-colors',
            autoRefresh
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
              : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
          ]"
        >
          Auto-refresh {{ autoRefresh ? 'ON' : 'OFF' }}
        </button>

        <!-- Refresh button -->
        <button
          @click="refreshData"
          :disabled="loading"
          class="px-3 py-1 text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors disabled:opacity-50"
        >
          {{ loading ? 'Refreshing...' : 'Refresh' }}
        </button>
      </div>
    </div>

    <!-- Security overview cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-6 border-b border-gray-200 dark:border-gray-700">
      <!-- Active alerts -->
      <div class="security-card">
        <div class="security-card-header">
          <div class="security-card-icon bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-400">
            <ExclamationTriangleIcon class="w-5 h-5" />
          </div>
          <h4 class="security-card-title">Active Alerts</h4>
        </div>
        <div class="security-card-value text-red-600 dark:text-red-400">
          {{ activeAlertsCount }}
        </div>
        <div class="security-card-description">
          Critical security events requiring attention
        </div>
      </div>

      <!-- Blocked commands -->
      <div class="security-card">
        <div class="security-card-header">
          <div class="security-card-icon bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-400">
            <ShieldExclamationIcon class="w-5 h-5" />
          </div>
          <h4 class="security-card-title">Blocked Commands</h4>
        </div>
        <div class="security-card-value text-orange-600 dark:text-orange-400">
          {{ blockedCommandsCount }}
        </div>
        <div class="security-card-description">
          Commands blocked in the last hour
        </div>
      </div>

      <!-- Pending approvals -->
      <div class="security-card">
        <div class="security-card-header">
          <div class="security-card-icon bg-yellow-100 text-yellow-600 dark:bg-yellow-900 dark:text-yellow-400">
            <ClockIcon class="w-5 h-5" />
          </div>
          <h4 class="security-card-title">Pending Approvals</h4>
        </div>
        <div class="security-card-value text-yellow-600 dark:text-yellow-400">
          {{ pendingApprovalsCount }}
        </div>
        <div class="security-card-description">
          Commands waiting for manual approval
        </div>
      </div>

      <!-- Risk score -->
      <div class="security-card">
        <div class="security-card-header">
          <div class="security-card-icon bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400">
            <ChartBarIcon class="w-5 h-5" />
          </div>
          <h4 class="security-card-title">Risk Score</h4>
        </div>
        <div class="security-card-value" :class="getRiskScoreColor()">
          {{ currentRiskScore }}/100
        </div>
        <div class="security-card-description">
          Current system security risk level
        </div>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
      <!-- Recent security alerts -->
      <div class="security-section">
        <div class="flex items-center justify-between mb-4">
          <h4 class="text-lg font-semibold text-gray-900 dark:text-white">
            Recent Security Alerts
          </h4>
          <button
            v-if="securityAlerts.length > alertLimit"
            @click="showAllAlerts = !showAllAlerts"
            class="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            {{ showAllAlerts ? 'Show less' : `Show all (${securityAlerts.length})` }}
          </button>
        </div>

        <div class="space-y-3 max-h-96 overflow-y-auto">
          <div
            v-for="alert in displayedAlerts"
            :key="alert.id"
            class="security-alert"
            :class="getAlertSeverityClass(alert.risk_level)"
          >
            <div class="flex items-start justify-between">
              <div class="flex items-start space-x-3">
                <div 
                  :class="[
                    'flex-shrink-0 w-2 h-2 rounded-full mt-2',
                    getRiskLevelColor(alert.risk_level)
                  ]"
                ></div>
                <div class="flex-1 min-w-0">
                  <div class="flex items-center space-x-2 mb-1">
                    <span 
                      class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
                      :class="getRiskLevelBadgeClass(alert.risk_level)"
                    >
                      {{ alert.risk_level }}
                    </span>
                    <span class="text-xs text-gray-500 dark:text-gray-400">
                      Agent {{ alert.agent_id.substring(0, 8) }}
                    </span>
                    <span class="text-xs text-gray-500 dark:text-gray-400">
                      {{ formatTimestamp(alert.timestamp) }}
                    </span>
                  </div>
                  <p class="text-sm text-gray-900 dark:text-white font-medium mb-1">
                    {{ alert.reason }}
                  </p>
                  <div class="text-xs font-mono bg-gray-100 dark:bg-gray-700 rounded px-2 py-1 text-gray-600 dark:text-gray-400 break-all">
                    {{ alert.command }}
                  </div>
                </div>
              </div>
              
              <div class="flex items-center space-x-2 ml-4">
                <span 
                  v-if="alert.blocked"
                  class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                >
                  Blocked
                </span>
                <span 
                  v-else-if="alert.requires_approval"
                  class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
                >
                  Pending
                </span>
                <span 
                  v-else-if="alert.approved"
                  class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                >
                  Approved
                </span>
              </div>
            </div>

            <!-- Approval actions -->
            <div 
              v-if="alert.requires_approval && !alert.approved && !alert.blocked"
              class="mt-3 flex items-center justify-end space-x-2"
            >
              <button
                @click="approveCommand(alert)"
                class="px-3 py-1 text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded hover:bg-green-200 dark:hover:bg-green-800 transition-colors"
              >
                Approve
              </button>
              <button
                @click="rejectCommand(alert)"
                class="px-3 py-1 text-xs bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200 rounded hover:bg-red-200 dark:hover:bg-red-800 transition-colors"
              >
                Reject
              </button>
            </div>
          </div>

          <!-- Empty state -->
          <div 
            v-if="securityAlerts.length === 0"
            class="text-center py-8"
          >
            <ShieldCheckIcon class="w-12 h-12 text-green-400 mx-auto mb-3" />
            <h5 class="text-lg font-medium text-gray-900 dark:text-white">
              All Clear
            </h5>
            <p class="text-gray-500 dark:text-gray-400">
              No security alerts detected
            </p>
          </div>
        </div>
      </div>

      <!-- Risk distribution chart -->
      <div class="security-section">
        <h4 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Risk Level Distribution
        </h4>
        
        <div class="space-y-4">
          <div
            v-for="(count, riskLevel) in riskDistribution"
            :key="riskLevel"
            class="risk-distribution-item"
          >
            <div class="flex items-center justify-between mb-1">
              <span class="text-sm font-medium text-gray-900 dark:text-white capitalize">
                {{ riskLevel.toLowerCase() }}
              </span>
              <span class="text-sm text-gray-500 dark:text-gray-400">
                {{ count }} events
              </span>
            </div>
            <div class="risk-distribution-bar">
              <div 
                class="risk-distribution-fill"
                :class="getRiskLevelColor(riskLevel as SecurityRisk)"
                :style="{ 
                  width: `${getRiskDistributionPercentage(count)}%` 
                }"
              ></div>
            </div>
          </div>
        </div>

        <!-- Dangerous patterns -->
        <div class="mt-6">
          <h5 class="text-md font-semibold text-gray-900 dark:text-white mb-3">
            Active Dangerous Patterns
          </h5>
          <div class="space-y-2">
            <div
              v-for="pattern in dangerousPatterns.slice(0, 5)"
              :key="pattern.pattern"
              class="dangerous-pattern"
            >
              <div class="flex items-center justify-between">
                <span class="text-sm font-mono text-gray-600 dark:text-gray-400">
                  {{ pattern.pattern }}
                </span>
                <span 
                  class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
                  :class="getRiskLevelBadgeClass(pattern.risk_level)"
                >
                  {{ pattern.risk_level }}
                </span>
              </div>
              <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {{ pattern.description }}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Security timeline (if showApprovalQueue is enabled) -->
    <div v-if="showApprovalQueue && pendingApprovals.length > 0" class="border-t border-gray-200 dark:border-gray-700 p-6">
      <h4 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Approval Queue
      </h4>
      
      <div class="space-y-3">
        <div
          v-for="approval in pendingApprovals"
          :key="approval.id"
          class="approval-item"
        >
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-yellow-100 dark:bg-yellow-900 rounded-full flex items-center justify-center">
                  <ClockIcon class="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                </div>
              </div>
              <div>
                <p class="text-sm font-medium text-gray-900 dark:text-white">
                  {{ approval.reason }}
                </p>
                <div class="text-xs text-gray-500 dark:text-gray-400">
                  Agent {{ approval.agent_id.substring(0, 8) }} • 
                  {{ formatTimestamp(approval.timestamp) }}
                </div>
                <div class="text-xs font-mono bg-gray-100 dark:bg-gray-700 rounded px-2 py-1 text-gray-600 dark:text-gray-400 mt-1 inline-block">
                  {{ approval.command }}
                </div>
              </div>
            </div>
            
            <div class="flex items-center space-x-2">
              <button
                @click="approveCommand(approval)"
                class="px-4 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
              >
                Approve
              </button>
              <button
                @click="rejectCommand(approval)"
                class="px-4 py-2 text-sm bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
              >
                Reject
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useEventsStore } from '@/stores/events'
import { useNotificationStore } from '@/stores/notifications'
import { 
  SecurityRisk,
  type SecurityAlert,
  type SecurityDashboardProps,
  type DangerousCommand
} from '@/types/hooks'
import {
  ExclamationTriangleIcon,
  ShieldExclamationIcon,
  ShieldCheckIcon,
  ClockIcon,
  ChartBarIcon
} from '@heroicons/vue/24/outline'
import { format } from 'date-fns'

// Props
interface Props extends SecurityDashboardProps {}

const props = withDefaults(defineProps<Props>(), {
  showApprovalQueue: true,
  alertLimit: 10,
  autoRefresh: true,
  refreshInterval: 30000 // 30 seconds
})

// Store
const eventsStore = useEventsStore()
const notificationStore = useNotificationStore()

// Local state
const loading = ref(false)
const autoRefresh = ref(props.autoRefresh)
const showAllAlerts = ref(false)
const refreshTimer = ref<number | null>(null)

// Mock data (in real implementation, this would come from API)
const mockDangerousPatterns: DangerousCommand[] = [
  {
    pattern: 'rm\\s+-rf\\s*/',
    risk_level: SecurityRisk.CRITICAL,
    description: 'Recursive delete from root directory',
    block_execution: true,
    require_approval: false
  },
  {
    pattern: 'sudo\\s+rm\\s+-rf',
    risk_level: SecurityRisk.CRITICAL,
    description: 'Sudo recursive delete',
    block_execution: true,
    require_approval: false
  },
  {
    pattern: 'sudo\\s+',
    risk_level: SecurityRisk.HIGH,
    description: 'Sudo commands',
    block_execution: false,
    require_approval: true
  },
  {
    pattern: 'chmod\\s+777',
    risk_level: SecurityRisk.HIGH,
    description: 'Dangerous permission changes',
    block_execution: false,
    require_approval: true
  },
  {
    pattern: 'curl.*\\|\\s*sh',
    risk_level: SecurityRisk.HIGH,
    description: 'Download and execute scripts',
    block_execution: false,
    require_approval: true
  }
]

// Computed
const { securityAlerts } = eventsStore

const activeAlertsCount = computed(() => {
  return securityAlerts.filter(alert => 
    !alert.approved && 
    (alert.blocked || alert.requires_approval)
  ).length
})

const blockedCommandsCount = computed(() => {
  const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000)
  return securityAlerts.filter(alert => 
    alert.blocked && 
    new Date(alert.timestamp) > oneHourAgo
  ).length
})

const pendingApprovalsCount = computed(() => {
  return securityAlerts.filter(alert => 
    alert.requires_approval && 
    !alert.approved && 
    !alert.blocked
  ).length
})

const currentRiskScore = computed(() => {
  const criticalWeight = 40
  const highWeight = 20
  const mediumWeight = 10
  const lowWeight = 5

  const recentAlerts = securityAlerts.filter(alert => {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000)
    return new Date(alert.timestamp) > oneHourAgo
  })

  let score = 0
  recentAlerts.forEach(alert => {
    switch (alert.risk_level) {
      case SecurityRisk.CRITICAL:
        score += criticalWeight
        break
      case SecurityRisk.HIGH:
        score += highWeight
        break
      case SecurityRisk.MEDIUM:
        score += mediumWeight
        break
      case SecurityRisk.LOW:
        score += lowWeight
        break
    }
  })

  return Math.min(100, score)
})

const displayedAlerts = computed(() => {
  const alerts = securityAlerts.slice().sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  )
  
  return showAllAlerts.value ? alerts : alerts.slice(0, props.alertLimit)
})

const pendingApprovals = computed(() => {
  return securityAlerts.filter(alert => 
    alert.requires_approval && 
    !alert.approved && 
    !alert.blocked
  ).slice(0, 5)
})

const riskDistribution = computed(() => {
  const distribution = {
    [SecurityRisk.CRITICAL]: 0,
    [SecurityRisk.HIGH]: 0,
    [SecurityRisk.MEDIUM]: 0,
    [SecurityRisk.LOW]: 0,
    [SecurityRisk.SAFE]: 0
  }

  securityAlerts.forEach(alert => {
    distribution[alert.risk_level]++
  })

  return distribution
})

const dangerousPatterns = computed(() => {
  return mockDangerousPatterns
})

// Methods
const formatTimestamp = (timestamp: string) => {
  return format(new Date(timestamp), 'HH:mm:ss')
}

const getSecurityStatusColor = () => {
  if (currentRiskScore.value >= 70) return 'bg-red-500'
  if (currentRiskScore.value >= 40) return 'bg-yellow-500'
  return 'bg-green-500'
}

const getSecurityStatusTextColor = () => {
  if (currentRiskScore.value >= 70) return 'text-red-600 dark:text-red-400'
  if (currentRiskScore.value >= 40) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-green-600 dark:text-green-400'
}

const getSecurityStatusText = () => {
  if (currentRiskScore.value >= 70) return 'High Risk'
  if (currentRiskScore.value >= 40) return 'Medium Risk'
  return 'Secure'
}

const getRiskScoreColor = () => {
  if (currentRiskScore.value >= 70) return 'text-red-600 dark:text-red-400'
  if (currentRiskScore.value >= 40) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-green-600 dark:text-green-400'
}

const getRiskLevelColor = (riskLevel: SecurityRisk) => {
  const colorMap = {
    [SecurityRisk.CRITICAL]: 'bg-red-500',
    [SecurityRisk.HIGH]: 'bg-orange-500',
    [SecurityRisk.MEDIUM]: 'bg-yellow-500',
    [SecurityRisk.LOW]: 'bg-blue-500',
    [SecurityRisk.SAFE]: 'bg-green-500'
  }
  return colorMap[riskLevel] || 'bg-gray-500'
}

const getRiskLevelBadgeClass = (riskLevel: SecurityRisk) => {
  const classMap = {
    [SecurityRisk.CRITICAL]: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    [SecurityRisk.HIGH]: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    [SecurityRisk.MEDIUM]: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    [SecurityRisk.LOW]: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    [SecurityRisk.SAFE]: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
  }
  return classMap[riskLevel] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getAlertSeverityClass = (riskLevel: SecurityRisk) => {
  const classMap = {
    [SecurityRisk.CRITICAL]: 'border-l-4 border-red-500 bg-red-50 dark:bg-red-900/10',
    [SecurityRisk.HIGH]: 'border-l-4 border-orange-500 bg-orange-50 dark:bg-orange-900/10',
    [SecurityRisk.MEDIUM]: 'border-l-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-900/10',
    [SecurityRisk.LOW]: 'border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/10',
    [SecurityRisk.SAFE]: 'border-l-4 border-green-500 bg-green-50 dark:bg-green-900/10'
  }
  return classMap[riskLevel] || 'border-l-4 border-gray-500 bg-gray-50 dark:bg-gray-900/10'
}

const getRiskDistributionPercentage = (count: number) => {
  const total = Object.values(riskDistribution.value).reduce((sum, c) => sum + c, 0)
  return total > 0 ? (count / total) * 100 : 0
}

const refreshData = async () => {
  loading.value = true
  try {
    // In real implementation, fetch security data from API
    await new Promise(resolve => setTimeout(resolve, 500))
  } catch (error) {
    console.error('Failed to refresh security data:', error)
  } finally {
    loading.value = false
  }
}

const approveCommand = async (alert: SecurityAlert) => {
  try {
    // In real implementation, send approval to API
    alert.approved = true
    alert.approved_by = 'current_user'
    alert.approved_at = new Date().toISOString()
    
    notificationStore.addNotification({
      type: 'success',
      title: 'Command Approved',
      message: `Security alert ${alert.id} has been approved`,
      duration: 3000
    })
  } catch (error) {
    console.error('Failed to approve command:', error)
    notificationStore.addNotification({
      type: 'error',
      title: 'Approval Failed',
      message: 'Failed to approve the command',
      duration: 5000
    })
  }
}

const rejectCommand = async (alert: SecurityAlert) => {
  try {
    // In real implementation, send rejection to API
    alert.blocked = true
    
    notificationStore.addNotification({
      type: 'warning',
      title: 'Command Rejected',
      message: `Security alert ${alert.id} has been rejected and blocked`,
      duration: 3000
    })
  } catch (error) {
    console.error('Failed to reject command:', error)
    notificationStore.addNotification({
      type: 'error',
      title: 'Rejection Failed',
      message: 'Failed to reject the command',
      duration: 5000
    })
  }
}

// Auto-refresh functionality
const startAutoRefresh = () => {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
  }
  
  if (autoRefresh.value) {
    refreshTimer.value = setInterval(() => {
      refreshData()
    }, props.refreshInterval)
  }
}

// Lifecycle
onMounted(() => {
  refreshData()
  startAutoRefresh()
  
  // Subscribe to security alerts
  eventsStore.onSecurityAlert((alert) => {
    // Handle new security alert
    notificationStore.addNotification({
      type: alert.risk_level === SecurityRisk.CRITICAL ? 'error' : 'warning',
      title: 'Security Alert',
      message: `${alert.risk_level} risk detected: ${alert.reason}`,
      duration: alert.risk_level === SecurityRisk.CRITICAL ? 0 : 10000
    })
  })
})

onUnmounted(() => {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
  }
})

// Watch auto-refresh changes
watch(autoRefresh, () => {
  startAutoRefresh()
})
</script>

<style scoped>
.security-dashboard {
  min-height: 600px;
}

.security-card {
  @apply bg-white dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600;
}

.security-card-header {
  @apply flex items-center space-x-3 mb-3;
}

.security-card-icon {
  @apply w-10 h-10 rounded-lg flex items-center justify-center;
}

.security-card-title {
  @apply text-sm font-medium text-gray-900 dark:text-white;
}

.security-card-value {
  @apply text-2xl font-bold mb-1;
}

.security-card-description {
  @apply text-xs text-gray-500 dark:text-gray-400;
}

.security-section {
  @apply bg-gray-50 dark:bg-gray-900 rounded-lg p-6;
}

.security-alert {
  @apply p-4 rounded-lg transition-all duration-200;
}

.security-alert:hover {
  @apply shadow-sm;
}

.risk-distribution-item {
  @apply mb-3;
}

.risk-distribution-bar {
  @apply w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2;
}

.risk-distribution-fill {
  @apply h-2 rounded-full transition-all duration-300;
}

.dangerous-pattern {
  @apply p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700;
}

.approval-item {
  @apply p-4 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600;
}

/* Responsive design */
@media (max-width: 768px) {
  .security-card-header {
    @apply flex-col items-start space-x-0 space-y-2;
  }
  
  .security-card-icon {
    @apply w-8 h-8;
  }
  
  .approval-item .flex-items-center {
    @apply flex-col items-start space-x-0 space-y-3;
  }
}
</style>