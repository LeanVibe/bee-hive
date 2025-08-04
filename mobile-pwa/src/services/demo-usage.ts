/**
 * Demo Usage Examples for LeanVibe Agent Hive API Services
 * 
 * This file demonstrates how to use the API integration services
 * in various scenarios common to the dashboard application.
 */

import { 
  initializeServices, 
  startAllMonitoring, 
  getSystemHealthService,
  getAgentService,
  getTaskService,
  getEventService,
  getMetricsService,
  AgentRole,
  TaskStatus,
  TaskPriority,
  TaskType
} from './index';

/**
 * Example 1: Initialize and start monitoring all services
 */
export async function setupDashboardServices() {
  console.log('🚀 Setting up LeanVibe Agent Hive dashboard services...');

  // Initialize all services with shared configuration
  const services = initializeServices({
    baseUrl: 'http://localhost:8000',
    timeout: 10000,
    retryAttempts: 3,
    pollingInterval: 5000
  });

  // Start real-time monitoring
  const monitoringServices = startAllMonitoring();

  console.log('✅ All services initialized and monitoring started');
  return services;
}

/**
 * Example 2: System Health Dashboard Integration
 */
export async function systemHealthDashboardExample() {
  console.log('📊 System Health Dashboard Example');

  const healthService = getSystemHealthService();

  try {
    // Get current system health
    const health = await healthService.getSystemHealth();
    console.log(`System Status: ${health.status}`);

    // Get health summary for UI indicators
    const summary = healthService.getHealthSummary();
    console.log(`Components: ${summary.components.healthy} healthy, ${summary.components.unhealthy} unhealthy`);
    console.log(`Active Alerts: ${summary.alerts.length}`);

    // Listen for health changes
    healthService.onHealthChange((newHealth) => {
      console.log('Health status changed:', newHealth.status);
      // Update UI components here
    });

    // Listen for critical alerts
    healthService.onHealthAlert((alert) => {
      console.log('Health alert:', alert.message);
      // Show notification to user
    });

  } catch (error) {
    console.error('Health check failed:', error);
  }
}

/**
 * Example 3: Agent Management Integration
 */
export async function agentManagementExample() {
  console.log('🤖 Agent Management Example');

  const agentService = getAgentService();

  try {
    // Get current agent status
    const status = await agentService.getAgentSystemStatus();
    console.log(`Agents: ${status.agent_count} total, ${status.active ? 'system active' : 'system inactive'}`);

    // Activate agent system if needed
    if (!status.active) {
      console.log('Activating agent system...');
      const activation = await agentService.activateAgentSystem({
        teamSize: 5,
        roles: [AgentRole.BACKEND_DEVELOPER, AgentRole.FRONTEND_DEVELOPER, AgentRole.QA_ENGINEER],
        autoStartTasks: true
      });
      console.log('Agent activation result:', activation.message);
    }

    // Get agent summary for dashboard
    const summary = agentService.getAgentSummary();
    console.log(`Agent Summary: ${summary.total} total, ${summary.active} active, ${summary.busy} busy`);

    // Get team composition
    const composition = agentService.getTeamComposition();
    console.log('Team Composition:', Object.keys(composition));

    // Listen for agent changes
    agentService.onAgentStatusChanged((newStatus) => {
      console.log('Agent status updated:', newStatus.agent_count, 'agents');
      // Update agent dashboard components
    });

  } catch (error) {
    console.error('Agent management failed:', error);
  }
}

/**
 * Example 4: Kanban Board Integration
 */
export async function kanbanBoardExample() {
  console.log('📋 Kanban Board Example');

  const taskService = getTaskService();
  const agentService = getAgentService();

  try {
    // Create a new task
    const newTask = await taskService.createTask({
      title: 'Implement user authentication API',
      description: 'Create REST API endpoints for user login, registration, and token refresh',
      task_type: TaskType.FEATURE,
      priority: TaskPriority.HIGH,
      required_capabilities: ['api_development', 'security', 'database'],
      estimated_effort: 240 // 4 hours
    });
    console.log('Created task:', newTask.title);

    // Get Kanban board data
    const board = await taskService.getKanbanBoard();
    console.log(`Kanban Board: ${board.totalTasks} total tasks across ${board.columns.length} columns`);

    // Display column summary
    board.columns.forEach(column => {
      console.log(`  ${column.title}: ${column.count} tasks`);
    });

    // Assign task to agent
    const agents = agentService.getAgents();
    const backendDeveloper = agents.find(a => a.role === AgentRole.BACKEND_DEVELOPER);
    
    if (backendDeveloper) {
      const assignment = await taskService.assignTask(newTask.id, backendDeveloper.id);
      console.log('Task assigned to:', backendDeveloper.name);

      // Start the task
      await taskService.startTask(newTask.id);
      console.log('Task started');
    }

    // Listen for task updates
    taskService.onTaskUpdated((updatedTask) => {
      console.log('Task updated:', updatedTask.title, 'status:', updatedTask.status);
      // Update Kanban board UI
    });

    // Listen for Kanban board changes
    taskService.onKanbanBoardUpdated((updatedBoard) => {
      console.log('Kanban board updated:', updatedBoard.totalTasks, 'tasks');
      // Refresh board display
    });

  } catch (error) {
    console.error('Task management failed:', error);
  }
}

/**
 * Example 5: Real-time Event Timeline
 */
export async function eventTimelineExample() {
  console.log('⚡ Event Timeline Example');

  const eventService = getEventService();

  try {
    // Get recent events
    const timeline = await eventService.getRecentEvents({}, 20);
    console.log(`Event Timeline: ${timeline.events.length} recent events`);

    // Display recent events
    timeline.events.slice(0, 5).forEach(event => {
      console.log(`  ${event.timestamp}: ${event.title} (${event.severity})`);
    });

    // Get event statistics
    const stats = eventService.getEventStatistics();
    console.log(`Event Stats: ${stats.total} total, ${stats.unacknowledged} unacknowledged`);
    console.log(`Recent Activity: ${stats.recentActivity.last5min} in last 5 min`);

    // Get activity summary for dashboard
    const activity = eventService.getActivitySummary(24);
    console.log(`Activity Summary: ${activity.events.length} events, ${activity.agentActivity.length} active agents`);

    // Start real-time monitoring
    eventService.startRealtimeMonitoring();

    // Listen for new events
    eventService.onNewEvent((event) => {
      console.log('New event:', event.title);
      // Add to timeline UI
    });

    // Listen for critical events
    eventService.onCriticalEvent((event) => {
      console.log('CRITICAL EVENT:', event.title);
      // Show urgent notification
    });

  } catch (error) {
    console.error('Event monitoring failed:', error);
  }
}

/**
 * Example 6: Performance Metrics Dashboard
 */
export async function performanceMetricsExample() {
  console.log('📈 Performance Metrics Example');

  const metricsService = getMetricsService();

  try {
    // Get current performance snapshot
    const performance = await metricsService.getCurrentPerformance();
    console.log(`Performance: CPU ${performance.cpu.toFixed(1)}%, Memory ${performance.memory.toFixed(1)}%`);
    console.log(`Agents: ${performance.agents.active} active, ${performance.agents.busy} busy`);
    console.log(`Tasks: ${performance.tasks.inProgress} in progress, ${performance.tasks.completed} completed`);

    // Get system metrics for charts
    const metrics = await metricsService.getSystemMetrics('1h');
    console.log(`System Metrics: ${metrics.cpu.data.length} data points over 1 hour`);

    // Get performance trends
    const trends = metricsService.getPerformanceTrends('24h');
    console.log(`Performance Trends: ${trends.length} significant trends identified`);
    trends.forEach(trend => {
      console.log(`  ${trend.metric}: ${trend.trend} by ${trend.change.toFixed(1)}%`);
    });

    // Get current alerts
    const alerts = metricsService.getPerformanceAlerts();
    console.log(`Performance Alerts: ${alerts.length} active alerts`);
    alerts.forEach(alert => {
      console.log(`  ${alert.severity.toUpperCase()}: ${alert.message}`);
    });

    // Get chart data for dashboard
    const chartData = metricsService.getSystemOverviewChartData('1h');
    console.log(`Chart Data: ${chartData.labels.length} time points, ${chartData.datasets.length} series`);

    // Listen for performance updates
    metricsService.onPerformanceUpdated((snapshot) => {
      console.log('Performance updated:', snapshot.timestamp);
      // Update dashboard charts
    });

    // Listen for performance alerts
    metricsService.onPerformanceAlert((alert) => {
      console.log('Performance alert:', alert.message);
      // Show alert notification
    });

  } catch (error) {
    console.error('Performance monitoring failed:', error);
  }
}

/**
 * Example 7: Complete Dashboard Setup
 */
export async function completeDashboardSetup() {
  console.log('🎯 Complete Dashboard Setup');

  try {
    // Initialize all services
    await setupDashboardServices();

    // Wait a moment for services to initialize
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Run all dashboard integrations
    await Promise.all([
      systemHealthDashboardExample(),
      agentManagementExample(),
      kanbanBoardExample(),
      eventTimelineExample(),
      performanceMetricsExample()
    ]);

    console.log('✅ Complete dashboard setup successful!');
    console.log('');
    console.log('The LeanVibe Agent Hive dashboard is now fully integrated with:');
    console.log('  🏥 Real-time system health monitoring');
    console.log('  🤖 Agent lifecycle management');
    console.log('  📋 Kanban task board with live updates');
    console.log('  ⚡ Real-time event timeline');
    console.log('  📈 Performance metrics and analytics');
    console.log('');
    console.log('All services are monitoring and will provide real-time updates to the UI.');

  } catch (error) {
    console.error('Dashboard setup failed:', error);
  }
}

// Export for easy testing
export {
  setupDashboardServices,
  systemHealthDashboardExample,
  agentManagementExample,
  kanbanBoardExample,
  eventTimelineExample,
  performanceMetricsExample,
  completeDashboardSetup
};