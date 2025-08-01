<template>
  <div class="metrics-widget">
    <div class="widget-header">
      <h3>Live System Metrics</h3>
      <div class="widget-status" :class="status">
        <div class="status-dot"></div>
        <span>{{ statusText }}</span>
      </div>
    </div>
    
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-icon">🤖</div>
        <div class="metric-content">
          <div class="metric-value">{{ metrics.activeAgents }}</div>
          <div class="metric-label">Active Agents</div>
          <div class="metric-change positive">+{{ metrics.agentChange }} today</div>
        </div>
      </div>
      
      <div class="metric-card">
        <div class="metric-icon">⚡</div>
        <div class="metric-content">
          <div class="metric-value">{{ metrics.commandsExecuted }}</div>
          <div class="metric-label">Commands Executed</div>
          <div class="metric-change positive">+{{ metrics.commandChange }}% this week</div>
        </div>
      </div>
      
      <div class="metric-card">
        <div class="metric-icon">🎯</div>
        <div class="metric-content">
          <div class="metric-value">{{ metrics.successRate }}%</div>
          <div class="metric-label">Success Rate</div>
          <div class="metric-change positive">+{{ metrics.successChange }}% improvement</div>
        </div>
      </div>
      
      <div class="metric-card">
        <div class="metric-icon">⏱️</div>
        <div class="metric-content">
          <div class="metric-value">{{ metrics.avgResponseTime }}ms</div>
          <div class="metric-label">Avg Response Time</div>
          <div class="metric-change negative">-{{ metrics.responseChange }}ms faster</div>
        </div>
      </div>
    </div>
    
    <div class="metrics-footer">
      <div class="update-time">
        Last updated: {{ lastUpdate }}
      </div>
      <a href="/api/observability/metrics" class="view-all-link">
        View All Metrics →
      </a>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

// Component state
const status = ref('healthy')
const statusText = ref('All Systems Operational')
const lastUpdate = ref('')
const updateInterval = ref(null)

// Metrics data
const metrics = ref({
  activeAgents: 8,
  agentChange: 2,
  commandsExecuted: 1247,
  commandChange: 23,
  successRate: 96.7,
  successChange: 2.1,
  avgResponseTime: 89,
  responseChange: 15
})

// Simulate real-time updates
const updateMetrics = () => {
  // Simulate small fluctuations in metrics
  metrics.value.activeAgents = Math.max(6, Math.min(12, metrics.value.activeAgents + (Math.random() - 0.5) * 2))
  metrics.value.commandsExecuted += Math.floor(Math.random() * 5)
  metrics.value.successRate = Math.max(85, Math.min(100, metrics.value.successRate + (Math.random() - 0.5) * 0.5))
  metrics.value.avgResponseTime = Math.max(50, Math.min(200, metrics.value.avgResponseTime + (Math.random() - 0.5) * 10))
  
  // Update timestamp
  lastUpdate.value = new Date().toLocaleTimeString()
  
  // Update status based on metrics
  if (metrics.value.successRate > 95 && metrics.value.avgResponseTime < 100) {
    status.value = 'healthy'
    statusText.value = 'All Systems Operational'
  } else if (metrics.value.successRate > 90) {
    status.value = 'warning'
    statusText.value = 'Performance Degraded'
  } else {
    status.value = 'error'
    statusText.value = 'System Issues Detected'
  }
}

// Lifecycle
onMounted(() => {
  updateMetrics()
  updateInterval.value = setInterval(updateMetrics, 5000) // Update every 5 seconds
})

onUnmounted(() => {
  if (updateInterval.value) {
    clearInterval(updateInterval.value)
  }
})
</script>

<style scoped>
.metrics-widget {
  margin: 2rem 0;
  padding: 1.5rem;
  background: var(--lv-glass-bg);
  backdrop-filter: blur(10px);
  border: 1px solid var(--lv-glass-border);
  border-radius: 12px;
  box-shadow: var(--lv-shadow-lg);
}

.widget-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.widget-header h3 {
  margin: 0;
  color: var(--vp-c-text-1);
  font-size: 1.25rem;
}

.widget-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  font-weight: 500;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--lv-secondary);
  animation: pulse 2s infinite;
}

.widget-status.warning .status-dot {
  background: var(--lv-accent);
}

.widget-status.error .status-dot {
  background: #ef4444;
}

.widget-status.healthy {
  color: var(--lv-secondary);
}

.widget-status.warning {
  color: var(--lv-accent);
}

.widget-status.error {
  color: #ef4444;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.metric-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  transition: all 0.3s ease;
}

.metric-card:hover {
  border-color: var(--lv-primary);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

.metric-icon {
  font-size: 2rem;
  flex-shrink: 0;
}

.metric-content {
  flex: 1;
  min-width: 0;
}

.metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  line-height: 1.2;
}

.metric-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  margin: 0.25rem 0;
}

.metric-change {
  font-size: 0.75rem;
  font-weight: 500;
}

.metric-change.positive {
  color: var(--lv-secondary);
}

.metric-change.negative {
  color: var(--lv-secondary); /* Negative here means improvement (faster response time) */
}

.metrics-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--vp-c-border);
}

.update-time {
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
}

.view-all-link {
  color: var(--lv-primary);
  text-decoration: none;
  font-weight: 500;
  font-size: 0.875rem;
}

.view-all-link:hover {
  text-decoration: underline;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

@media (max-width: 768px) {
  .metrics-grid {
    grid-template-columns: 1fr;
  }
  
  .metric-card {
    flex-direction: column;
    text-align: center;
    gap: 0.5rem;
  }
  
  .metrics-footer {
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }
  
  .widget-header {
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }
}
</style>