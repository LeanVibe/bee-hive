import { test, expect } from '@playwright/test';
import { 
  AuthHelpers, 
  NavigationHelpers, 
  loadTestData,
  takeTimestampedScreenshot,
  waitForNetworkIdle,
  waitForElementWithRetry
} from '../utils/test-helpers';

/**
 * Task Management Workflow E2E Tests
 * Tests complete task lifecycle management in PWA
 */

test.describe('Task Management Workflows', () => {
  let authHelpers: AuthHelpers;
  let navHelpers: NavigationHelpers;
  let testData: any;

  test.beforeEach(async ({ page }) => {
    authHelpers = new AuthHelpers(page);
    navHelpers = new NavigationHelpers(page);
    testData = loadTestData();
    
    // Log in as admin user for task management
    const adminUser = testData.users.find((u: any) => u.role === 'admin');
    await page.goto('/');
    await authHelpers.login(adminUser);
  });

  test.describe('Task Creation', () => {
    test('should create a new monitoring task', async ({ page }) => {
      // Navigate to tasks page
      await navHelpers.goToTasks();
      await expect(page.locator('[data-testid="tasks-page"]')).toBeVisible();

      // Click create new task button
      await page.click('[data-testid="create-task-button"]');
      await expect(page.locator('[data-testid="task-form"]')).toBeVisible();

      // Fill out task creation form
      await page.fill('[data-testid="task-title-input"]', 'Monitor Database Performance');
      await page.selectOption('[data-testid="task-type-select"]', 'monitoring');
      await page.fill('[data-testid="task-description-input"]', 'Continuously monitor database performance metrics and alert on anomalies');

      // Set task priority and deadline
      await page.selectOption('[data-testid="task-priority-select"]', 'high');
      await page.fill('[data-testid="task-deadline-input"]', '2024-12-31');

      // Assign to agent
      await page.selectOption('[data-testid="assigned-agent-select"]', testData.agents[0].id);

      // Configure task parameters
      await page.fill('[data-testid="monitoring-threshold-input"]', '95');
      await page.check('[data-testid="enable-notifications"]');
      await page.fill('[data-testid="notification-email-input"]', 'alerts@leanvibe.test');

      // Add tags
      await page.fill('[data-testid="task-tags-input"]', 'database, performance, critical');

      // Submit form
      await page.click('[data-testid="create-task-submit"]');

      // Wait for task to be created and redirected to task detail
      await waitForNetworkIdle(page);
      await expect(page.locator('[data-testid="task-detail-page"]')).toBeVisible();

      // Verify task details
      await expect(page.locator('[data-testid="task-title"]')).toContainText('Monitor Database Performance');
      await expect(page.locator('[data-testid="task-status"]')).toContainText('pending');
      await expect(page.locator('[data-testid="task-priority"]')).toContainText('high');

      await takeTimestampedScreenshot(page, 'task-created');
    });

    test('should create a data collection task', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="create-task-button"]');

      // Fill out data collection task form
      await page.fill('[data-testid="task-title-input"]', 'Collect User Engagement Metrics');
      await page.selectOption('[data-testid="task-type-select"]', 'data-collection');
      await page.fill('[data-testid="task-description-input"]', 'Gather comprehensive user engagement data for analytics');

      // Set collection parameters
      await page.check('[data-testid="collect-page-views"]');
      await page.check('[data-testid="collect-click-events"]');
      await page.check('[data-testid="collect-session-duration"]');
      await page.fill('[data-testid="collection-interval-input"]', '300'); // 5 minutes

      // Set data retention and export settings
      await page.selectOption('[data-testid="data-retention-select"]', '90-days');
      await page.check('[data-testid="auto-export"]');
      await page.selectOption('[data-testid="export-format-select"]', 'json');

      // Assign to agent
      await page.selectOption('[data-testid="assigned-agent-select"]', testData.agents[1].id);

      await page.click('[data-testid="create-task-submit"]');
      await waitForNetworkIdle(page);

      // Verify creation
      await expect(page.locator('[data-testid="task-title"]')).toContainText('Collect User Engagement Metrics');
      await expect(page.locator('[data-testid="task-type"]')).toContainText('data-collection');

      await takeTimestampedScreenshot(page, 'data-collection-task-created');
    });

    test('should validate required fields in task creation', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="create-task-button"]');

      // Try to submit without filling required fields
      await page.click('[data-testid="create-task-submit"]');

      // Should show validation errors
      await expect(page.locator('[data-testid="task-title-error"]')).toBeVisible();
      await expect(page.locator('[data-testid="task-type-error"]')).toBeVisible();
      await expect(page.locator('[data-testid="assigned-agent-error"]')).toBeVisible();

      // Test individual field validation
      await page.fill('[data-testid="task-title-input"]', 'A'); // Too short
      await page.blur('[data-testid="task-title-input"]');
      await expect(page.locator('[data-testid="task-title-error"]')).toContainText('at least 5 characters');

      // Test deadline validation
      await page.fill('[data-testid="task-deadline-input"]', '2020-01-01'); // Past date
      await page.blur('[data-testid="task-deadline-input"]');
      await expect(page.locator('[data-testid="task-deadline-error"]')).toContainText('future date');

      await takeTimestampedScreenshot(page, 'task-validation-errors');
    });

    test('should create recurring task', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="create-task-button"]');

      // Fill basic task info
      await page.fill('[data-testid="task-title-input"]', 'Daily Health Check');
      await page.selectOption('[data-testid="task-type-select"]', 'monitoring');
      await page.fill('[data-testid="task-description-input"]', 'Daily system health verification');

      // Enable recurring schedule
      await page.check('[data-testid="enable-recurring"]');
      await expect(page.locator('[data-testid="recurring-options"]')).toBeVisible();

      // Configure recurrence
      await page.selectOption('[data-testid="recurrence-type-select"]', 'daily');
      await page.fill('[data-testid="recurrence-time-input"]', '09:00');
      await page.selectOption('[data-testid="recurrence-timezone-select"]', 'America/New_York');

      // Set end condition
      await page.check('[data-testid="recurrence-end-date"]');
      await page.fill('[data-testid="recurrence-end-date-input"]', '2024-12-31');

      await page.selectOption('[data-testid="assigned-agent-select"]', testData.agents[0].id);

      await page.click('[data-testid="create-task-submit"]');
      await waitForNetworkIdle(page);

      // Verify recurring task created
      await expect(page.locator('[data-testid="task-recurrence"]')).toContainText('Daily at 09:00');
      await expect(page.locator('[data-testid="next-execution"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'recurring-task-created');
    });
  });

  test.describe('Task Management', () => {
    test('should edit an existing task', async ({ page }) => {
      await navHelpers.goToTasks();
      
      // Click on first task
      await page.click('[data-testid="task-card"]:first-child');
      await expect(page.locator('[data-testid="task-detail-page"]')).toBeVisible();

      // Click edit button
      await page.click('[data-testid="edit-task-button"]');
      await expect(page.locator('[data-testid="task-form"]')).toBeVisible();

      // Modify task details
      await page.fill('[data-testid="task-title-input"]', 'Updated Task Title');
      await page.selectOption('[data-testid="task-priority-select"]', 'medium');
      await page.fill('[data-testid="task-description-input"]', 'Updated task description with more details');

      // Update deadline
      await page.fill('[data-testid="task-deadline-input"]', '2025-01-15');

      // Add additional tags
      const existingTags = await page.inputValue('[data-testid="task-tags-input"]');
      await page.fill('[data-testid="task-tags-input"]', `${existingTags}, updated, modified`);

      // Save changes
      await page.click('[data-testid="save-task-button"]');
      await waitForNetworkIdle(page);

      // Verify changes saved
      await expect(page.locator('[data-testid="task-title"]')).toContainText('Updated Task Title');
      await expect(page.locator('[data-testid="task-priority"]')).toContainText('medium');
      await expect(page.locator('[data-testid="task-description"]')).toContainText('Updated task description');

      await takeTimestampedScreenshot(page, 'task-edited');
    });

    test('should assign task to different agent', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="task-card"]:first-child');

      // Get current assigned agent
      const currentAgent = await page.locator('[data-testid="assigned-agent-name"]').textContent();

      // Click reassign button
      await page.click('[data-testid="reassign-task-button"]');
      await expect(page.locator('[data-testid="reassign-modal"]')).toBeVisible();

      // Select different agent
      const availableAgents = testData.agents.filter((a: any) => a.name !== currentAgent);
      await page.selectOption('[data-testid="new-agent-select"]', availableAgents[0].id);

      // Add reassignment reason
      await page.fill('[data-testid="reassignment-reason-input"]', 'Better suited for this agent type');

      // Confirm reassignment
      await page.click('[data-testid="confirm-reassign-button"]');
      await waitForNetworkIdle(page);

      // Verify reassignment
      await expect(page.locator('[data-testid="assigned-agent-name"]')).toContainText(availableAgents[0].name);
      await expect(page.locator('[data-testid="task-history"]')).toContainText('reassigned');

      await takeTimestampedScreenshot(page, 'task-reassigned');
    });

    test('should duplicate an existing task', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="task-card"]:first-child');

      // Get original task name for verification
      const originalTitle = await page.locator('[data-testid="task-title"]').textContent();

      // Duplicate the task
      await page.click('[data-testid="duplicate-task-button"]');
      await expect(page.locator('[data-testid="task-form"]')).toBeVisible();

      // Verify form is pre-filled with original task data
      const duplicateTitle = await page.inputValue('[data-testid="task-title-input"]');
      expect(duplicateTitle).toContain('Copy of');

      // Modify the duplicate
      await page.fill('[data-testid="task-title-input"]', 'Duplicate - Enhanced Monitoring');
      await page.fill('[data-testid="task-description-input"]', 'Enhanced version of the original monitoring task');

      // Submit the duplicate
      await page.click('[data-testid="create-task-submit"]');
      await waitForNetworkIdle(page);

      // Verify duplicate was created
      await expect(page.locator('[data-testid="task-title"]')).toContainText('Duplicate - Enhanced Monitoring');

      await takeTimestampedScreenshot(page, 'task-duplicated');
    });
  });

  test.describe('Task Execution', () => {
    test('should start and monitor task execution', async ({ page }) => {
      await navHelpers.goToTasks();
      
      // Find a pending task
      const pendingTask = page.locator('[data-testid="task-card"]').filter({
        has: page.locator('[data-testid="task-status"]:has-text("pending")')
      }).first();
      
      await pendingTask.click();
      await expect(page.locator('[data-testid="task-detail-page"]')).toBeVisible();

      // Start the task
      await page.click('[data-testid="start-task-button"]');
      await waitForNetworkIdle(page);

      // Verify task started
      await expect(page.locator('[data-testid="task-status"]')).toContainText('running');
      await expect(page.locator('[data-testid="start-task-button"]')).not.toBeVisible();
      await expect(page.locator('[data-testid="stop-task-button"]')).toBeVisible();

      // Check execution progress
      await page.click('[data-testid="execution-tab"]');
      await expect(page.locator('[data-testid="execution-details"]')).toBeVisible();
      await expect(page.locator('[data-testid="execution-start-time"]')).toBeVisible();
      await expect(page.locator('[data-testid="execution-progress"]')).toBeVisible();

      // Wait for some execution data
      await page.waitForTimeout(3000);

      // Verify execution metrics
      await expect(page.locator('[data-testid="execution-duration"]')).toBeVisible();
      await expect(page.locator('[data-testid="execution-metrics"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'task-execution');
    });

    test('should pause and resume task execution', async ({ page }) => {
      await navHelpers.goToTasks();
      
      // Find a running task
      const runningTask = page.locator('[data-testid="task-card"]').filter({
        has: page.locator('[data-testid="task-status"]:has-text("running")')
      }).first();
      
      await runningTask.click();

      // Pause the task
      await page.click('[data-testid="pause-task-button"]');
      await waitForNetworkIdle(page);

      // Verify task paused
      await expect(page.locator('[data-testid="task-status"]')).toContainText('paused');
      await expect(page.locator('[data-testid="resume-task-button"]')).toBeVisible();

      // Resume the task
      await page.click('[data-testid="resume-task-button"]');
      await waitForNetworkIdle(page);

      // Verify task resumed
      await expect(page.locator('[data-testid="task-status"]')).toContainText('running');
      await expect(page.locator('[data-testid="pause-task-button"]')).toBeVisible();

      // Check pause/resume history
      await page.click('[data-testid="execution-history-tab"]');
      await expect(page.locator('[data-testid="execution-event"]').filter({
        hasText: 'paused'
      })).toBeVisible();
      await expect(page.locator('[data-testid="execution-event"]').filter({
        hasText: 'resumed'
      })).toBeVisible();

      await takeTimestampedScreenshot(page, 'task-pause-resume');
    });

    test('should handle task completion', async ({ page }) => {
      await navHelpers.goToTasks();
      
      // Start a short-running task for testing completion
      await page.click('[data-testid="create-task-button"]');
      
      // Create a quick test task
      await page.fill('[data-testid="task-title-input"]', 'Quick Test Task');
      await page.selectOption('[data-testid="task-type-select"]', 'monitoring');
      await page.fill('[data-testid="task-description-input"]', 'Quick task for completion testing');
      await page.selectOption('[data-testid="assigned-agent-select"]', testData.agents[0].id);
      
      // Set it to run only once with short duration
      await page.fill('[data-testid="max-execution-time-input"]', '5'); // 5 seconds
      
      await page.click('[data-testid="create-task-submit"]');
      await waitForNetworkIdle(page);

      // Start the task
      await page.click('[data-testid="start-task-button"]');
      
      // Wait for completion (give it some buffer time)
      await page.waitForTimeout(10000);

      // Check for completion
      await page.reload();
      await waitForNetworkIdle(page);
      
      await expect(page.locator('[data-testid="task-status"]')).toContainText('completed');
      
      // Verify completion details
      await page.click('[data-testid="execution-tab"]');
      await expect(page.locator('[data-testid="completion-time"]')).toBeVisible();
      await expect(page.locator('[data-testid="execution-summary"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'task-completed');
    });

    test('should handle task failures', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="task-card"]:first-child');

      // Configure task to fail by setting invalid parameters
      await page.click('[data-testid="edit-task-button"]');
      await page.fill('[data-testid="monitoring-endpoint-input"]', 'http://invalid-endpoint.localhost:99999');
      await page.click('[data-testid="save-task-button"]');
      await waitForNetworkIdle(page);

      // Start task (should fail)
      await page.click('[data-testid="start-task-button"]');
      await page.waitForTimeout(5000);

      // Verify failure status
      await expect(page.locator('[data-testid="task-status"]')).toContainText('failed');
      await expect(page.locator('[data-testid="failure-message"]')).toBeVisible();

      // Check failure details
      await page.click('[data-testid="view-failure-details"]');
      await expect(page.locator('[data-testid="failure-details-modal"]')).toBeVisible();
      await expect(page.locator('[data-testid="error-description"]')).toContainText('connection');

      // Verify retry option is available
      await expect(page.locator('[data-testid="retry-task-button"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'task-failure');
    });
  });

  test.describe('Task Monitoring and Reports', () => {
    test('should display task execution metrics', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="task-card"]:first-child');

      // Go to metrics tab
      await page.click('[data-testid="metrics-tab"]');
      await expect(page.locator('[data-testid="task-metrics-dashboard"]')).toBeVisible();

      // Verify key metrics are displayed
      await expect(page.locator('[data-testid="total-executions"]')).toBeVisible();
      await expect(page.locator('[data-testid="success-rate"]')).toBeVisible();
      await expect(page.locator('[data-testid="average-duration"]')).toBeVisible();
      await expect(page.locator('[data-testid="failure-rate"]')).toBeVisible();

      // Check metric charts
      await expect(page.locator('[data-testid="execution-history-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="duration-trend-chart"]')).toBeVisible();

      // Test time range selector
      await page.selectOption('[data-testid="metrics-time-range"]', '7d');
      await waitForNetworkIdle(page);
      
      await expect(page.locator('[data-testid="time-range-indicator"]')).toContainText('7 days');

      await takeTimestampedScreenshot(page, 'task-metrics');
    });

    test('should generate task execution reports', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="task-card"]:first-child');

      // Go to reports tab
      await page.click('[data-testid="reports-tab"]');
      await expect(page.locator('[data-testid="reports-section"]')).toBeVisible();

      // Generate execution report
      await page.click('[data-testid="generate-report-button"]');
      await expect(page.locator('[data-testid="report-options-modal"]')).toBeVisible();

      // Configure report options
      await page.selectOption('[data-testid="report-type-select"]', 'execution-summary');
      await page.selectOption('[data-testid="report-format-select"]', 'pdf');
      await page.selectOption('[data-testid="report-period-select"]', '30d');
      
      // Include specific metrics
      await page.check('[data-testid="include-performance-metrics"]');
      await page.check('[data-testid="include-error-analysis"]');
      await page.check('[data-testid="include-recommendations"]');

      // Generate report
      await page.click('[data-testid="generate-report-submit"]');
      await waitForNetworkIdle(page, 10000);

      // Verify report generation
      await expect(page.locator('[data-testid="report-generated-message"]')).toBeVisible();
      await expect(page.locator('[data-testid="download-report-button"]')).toBeVisible();

      // Check report preview
      await page.click('[data-testid="preview-report-button"]');
      await expect(page.locator('[data-testid="report-preview-modal"]')).toBeVisible();
      await expect(page.locator('[data-testid="report-content"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'task-report-generated');
    });

    test('should display task execution logs', async ({ page }) => {
      await navHelpers.goToTasks();
      await page.click('[data-testid="task-card"]:first-child');

      // Go to logs tab
      await page.click('[data-testid="task-logs-tab"]');
      await expect(page.locator('[data-testid="task-logs"]')).toBeVisible();

      // Verify log entries
      await expect(page.locator('[data-testid="log-entry"]').first()).toBeVisible();
      
      // Test log filtering by level
      await page.selectOption('[data-testid="log-level-filter"]', 'info');
      await waitForNetworkIdle(page);
      
      const infoLogs = page.locator('[data-testid="log-entry"]');
      if (await infoLogs.count() > 0) {
        await expect(infoLogs.first().locator('[data-testid="log-level"]')).toContainText('info');
      }

      // Test log search
      await page.fill('[data-testid="log-search-input"]', 'execution');
      await page.keyboard.press('Enter');
      await waitForNetworkIdle(page);

      // Test log export
      await page.click('[data-testid="export-logs-button"]');
      await expect(page.locator('[data-testid="export-options-modal"]')).toBeVisible();
      
      await page.selectOption('[data-testid="export-format-select"]', 'json');
      await page.click('[data-testid="export-logs-submit"]');
      
      await expect(page.locator('[data-testid="export-success-message"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'task-logs');
    });
  });

  test.describe('Task Bulk Operations', () => {
    test('should perform bulk task actions', async ({ page }) => {
      await navHelpers.goToTasks();

      // Enable bulk selection mode
      await page.click('[data-testid="bulk-actions-toggle"]');
      await expect(page.locator('[data-testid="bulk-actions-bar"]')).toBeVisible();

      // Select multiple tasks
      await page.check('[data-testid="task-checkbox"]:nth-child(1)');
      await page.check('[data-testid="task-checkbox"]:nth-child(2)');
      await page.check('[data-testid="task-checkbox"]:nth-child(3)');

      // Verify selection count
      await expect(page.locator('[data-testid="selected-count"]')).toContainText('3 selected');

      // Test bulk priority change
      await page.click('[data-testid="bulk-change-priority"]');
      await expect(page.locator('[data-testid="bulk-priority-modal"]')).toBeVisible();
      
      await page.selectOption('[data-testid="new-priority-select"]', 'high');
      await page.click('[data-testid="apply-bulk-priority"]');
      await waitForNetworkIdle(page);

      // Verify changes applied
      await expect(page.locator('[data-testid="bulk-action-success"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'bulk-task-actions');
    });

    test('should bulk delete tasks', async ({ page }) => {
      await navHelpers.goToTasks();

      // Get initial task count
      const initialCount = await page.locator('[data-testid="task-card"]').count();

      // Select tasks for deletion
      await page.click('[data-testid="bulk-actions-toggle"]');
      await page.check('[data-testid="task-checkbox"]:nth-child(1)');
      await page.check('[data-testid="task-checkbox"]:nth-child(2)');

      // Bulk delete
      await page.click('[data-testid="bulk-delete-tasks"]');
      await expect(page.locator('[data-testid="confirm-bulk-delete-modal"]')).toBeVisible();
      
      await page.fill('[data-testid="delete-confirmation-input"]', 'DELETE');
      await page.click('[data-testid="confirm-bulk-delete-button"]');
      await waitForNetworkIdle(page);

      // Verify deletion
      const finalCount = await page.locator('[data-testid="task-card"]').count();
      expect(finalCount).toBe(initialCount - 2);

      await takeTimestampedScreenshot(page, 'bulk-task-deletion');
    });
  });
});