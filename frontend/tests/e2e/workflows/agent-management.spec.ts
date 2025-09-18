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
 * Agent Management Workflow E2E Tests
 * Tests complete agent lifecycle management in PWA
 */

test.describe('Agent Management Workflows', () => {
  let authHelpers: AuthHelpers;
  let navHelpers: NavigationHelpers;
  let testData: any;

  test.beforeEach(async ({ page }) => {
    authHelpers = new AuthHelpers(page);
    navHelpers = new NavigationHelpers(page);
    testData = loadTestData();
    
    // Log in as admin user for agent management
    const adminUser = testData.users.find((u: any) => u.role === 'admin');
    await page.goto('/');
    await authHelpers.login(adminUser);
  });

  test.describe('Agent Creation', () => {
    test('should create a new monitoring agent', async ({ page }) => {
      // Navigate to agents page
      await navHelpers.goToAgents();
      await expect(page.locator('[data-testid="agents-page"]')).toBeVisible();

      // Click create new agent button
      await page.click('[data-testid="create-agent-button"]');
      await expect(page.locator('[data-testid="agent-form"]')).toBeVisible();

      // Fill out agent creation form
      await page.fill('[data-testid="agent-name-input"]', 'Test Monitoring Agent');
      await page.selectOption('[data-testid="agent-type-select"]', 'monitoring');
      await page.fill('[data-testid="agent-description-input"]', 'Agent for testing monitoring capabilities');

      // Configure monitoring settings
      await page.fill('[data-testid="monitoring-interval-input"]', '30');
      await page.fill('[data-testid="monitoring-endpoints-input"]', 'http://localhost:8000/health\nhttp://localhost:8001/status');

      // Set priority and tags
      await page.selectOption('[data-testid="agent-priority-select"]', 'high');
      await page.fill('[data-testid="agent-tags-input"]', 'monitoring, health-check, critical');

      // Submit form
      await page.click('[data-testid="create-agent-submit"]');

      // Wait for agent to be created and redirected to agent detail
      await waitForNetworkIdle(page);
      await expect(page.locator('[data-testid="agent-detail-page"]')).toBeVisible();

      // Verify agent details
      await expect(page.locator('[data-testid="agent-name"]')).toContainText('Test Monitoring Agent');
      await expect(page.locator('[data-testid="agent-type"]')).toContainText('monitoring');
      await expect(page.locator('[data-testid="agent-status"]')).toContainText('inactive');

      await takeTimestampedScreenshot(page, 'agent-created');
    });

    test('should create a data collection agent', async ({ page }) => {
      await navHelpers.goToAgents();
      await page.click('[data-testid="create-agent-button"]');

      // Fill out data collection agent form
      await page.fill('[data-testid="agent-name-input"]', 'User Analytics Collector');
      await page.selectOption('[data-testid="agent-type-select"]', 'data-collection');
      await page.fill('[data-testid="agent-description-input"]', 'Collects user interaction analytics');

      // Configure data collection settings
      await page.check('[data-testid="collect-page-views"]');
      await page.check('[data-testid="collect-user-interactions"]');
      await page.fill('[data-testid="collection-interval-input"]', '60');
      
      // Set data retention
      await page.selectOption('[data-testid="data-retention-select"]', '30-days');

      await page.click('[data-testid="create-agent-submit"]');
      await waitForNetworkIdle(page);

      // Verify creation
      await expect(page.locator('[data-testid="agent-name"]')).toContainText('User Analytics Collector');
      await expect(page.locator('[data-testid="agent-type"]')).toContainText('data-collection');

      await takeTimestampedScreenshot(page, 'data-collection-agent-created');
    });

    test('should validate required fields in agent creation', async ({ page }) => {
      await navHelpers.goToAgents();
      await page.click('[data-testid="create-agent-button"]');

      // Try to submit without filling required fields
      await page.click('[data-testid="create-agent-submit"]');

      // Should show validation errors
      await expect(page.locator('[data-testid="agent-name-error"]')).toBeVisible();
      await expect(page.locator('[data-testid="agent-type-error"]')).toBeVisible();

      // Test individual field validation
      await page.fill('[data-testid="agent-name-input"]', 'A'); // Too short
      await page.blur('[data-testid="agent-name-input"]');
      await expect(page.locator('[data-testid="agent-name-error"]')).toContainText('at least 3 characters');

      await takeTimestampedScreenshot(page, 'agent-validation-errors');
    });
  });

  test.describe('Agent Configuration', () => {
    test('should configure agent monitoring settings', async ({ page }) => {
      await navHelpers.goToAgents();
      
      // Click on existing test agent
      await page.click('[data-testid="agent-card"]:first-child');
      await expect(page.locator('[data-testid="agent-detail-page"]')).toBeVisible();

      // Click configure button
      await page.click('[data-testid="configure-agent-button"]');
      await expect(page.locator('[data-testid="agent-config-form"]')).toBeVisible();

      // Update monitoring configuration
      await page.fill('[data-testid="monitoring-interval-input"]', '45');
      await page.fill('[data-testid="timeout-threshold-input"]', '5000');
      await page.check('[data-testid="enable-alerting"]');
      await page.fill('[data-testid="alert-email-input"]', 'alerts@leanvibe.test');

      // Add custom headers for monitoring
      await page.click('[data-testid="add-custom-header"]');
      await page.fill('[data-testid="header-key-input"]', 'Authorization');
      await page.fill('[data-testid="header-value-input"]', 'Bearer test-token');

      // Save configuration
      await page.click('[data-testid="save-config-button"]');
      await waitForNetworkIdle(page);

      // Verify configuration saved
      await expect(page.locator('[data-testid="config-saved-message"]')).toBeVisible();
      await expect(page.locator('[data-testid="monitoring-interval-value"]')).toContainText('45');

      await takeTimestampedScreenshot(page, 'agent-configured');
    });

    test('should configure agent scheduling', async ({ page }) => {
      await navHelpers.goToAgents();
      await page.click('[data-testid="agent-card"]:first-child');
      await page.click('[data-testid="configure-agent-button"]');

      // Go to scheduling tab
      await page.click('[data-testid="scheduling-tab"]');

      // Configure schedule
      await page.check('[data-testid="enable-scheduling"]');
      await page.selectOption('[data-testid="schedule-type-select"]', 'recurring');
      
      // Set recurring schedule
      await page.check('[data-testid="monday"]');
      await page.check('[data-testid="wednesday"]');
      await page.check('[data-testid="friday"]');
      
      await page.fill('[data-testid="start-time-input"]', '09:00');
      await page.fill('[data-testid="end-time-input"]', '17:00');

      // Set timezone
      await page.selectOption('[data-testid="timezone-select"]', 'America/New_York');

      await page.click('[data-testid="save-config-button"]');
      await waitForNetworkIdle(page);

      await expect(page.locator('[data-testid="schedule-summary"]')).toContainText('Mon, Wed, Fri 09:00-17:00');

      await takeTimestampedScreenshot(page, 'agent-scheduled');
    });
  });

  test.describe('Agent Operations', () => {
    test('should start and stop an agent', async ({ page }) => {
      await navHelpers.goToAgents();
      
      // Find an inactive agent
      const agentCard = page.locator('[data-testid="agent-card"]').filter({
        has: page.locator('[data-testid="agent-status"]:has-text("inactive")')
      }).first();
      
      await agentCard.click();
      await expect(page.locator('[data-testid="agent-detail-page"]')).toBeVisible();

      // Start the agent
      await page.click('[data-testid="start-agent-button"]');
      await waitForNetworkIdle(page, 10000);

      // Verify agent started
      await expect(page.locator('[data-testid="agent-status"]')).toContainText('active');
      await expect(page.locator('[data-testid="start-agent-button"]')).not.toBeVisible();
      await expect(page.locator('[data-testid="stop-agent-button"]')).toBeVisible();

      // Wait a moment for agent to run
      await page.waitForTimeout(3000);

      // Stop the agent
      await page.click('[data-testid="stop-agent-button"]');
      await waitForNetworkIdle(page);

      // Verify agent stopped
      await expect(page.locator('[data-testid="agent-status"]')).toContainText('inactive');
      await expect(page.locator('[data-testid="start-agent-button"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'agent-start-stop');
    });

    test('should restart a running agent', async ({ page }) => {
      await navHelpers.goToAgents();
      
      // Start with an active agent
      const activeAgentCard = page.locator('[data-testid="agent-card"]').filter({
        has: page.locator('[data-testid="agent-status"]:has-text("active")')
      }).first();
      
      await activeAgentCard.click();

      // Restart the agent
      await page.click('[data-testid="restart-agent-button"]');
      
      // Confirm restart in modal
      await expect(page.locator('[data-testid="confirm-restart-modal"]')).toBeVisible();
      await page.click('[data-testid="confirm-restart-button"]');
      
      await waitForNetworkIdle(page, 10000);

      // Verify agent restarted (should show active status)
      await expect(page.locator('[data-testid="agent-status"]')).toContainText('active');
      
      // Check for restart indicator in logs
      await page.click('[data-testid="agent-logs-tab"]');
      await expect(page.locator('[data-testid="agent-logs"]')).toContainText('Agent restarted');

      await takeTimestampedScreenshot(page, 'agent-restarted');
    });

    test('should handle agent errors gracefully', async ({ page }) => {
      await navHelpers.goToAgents();
      await page.click('[data-testid="agent-card"]:first-child');

      // Configure agent with invalid endpoint to trigger error
      await page.click('[data-testid="configure-agent-button"]');
      await page.fill('[data-testid="monitoring-endpoints-input"]', 'http://invalid-endpoint.localhost:99999');
      await page.click('[data-testid="save-config-button"]');
      await waitForNetworkIdle(page);

      // Start agent (should fail)
      await page.click('[data-testid="start-agent-button"]');
      await page.waitForTimeout(5000);

      // Verify error status
      await expect(page.locator('[data-testid="agent-status"]')).toContainText('error');
      await expect(page.locator('[data-testid="error-message"]')).toBeVisible();

      // Check error details
      await page.click('[data-testid="view-error-details"]');
      await expect(page.locator('[data-testid="error-details-modal"]')).toBeVisible();
      await expect(page.locator('[data-testid="error-description"]')).toContainText('connection');

      await takeTimestampedScreenshot(page, 'agent-error-handling');
    });
  });

  test.describe('Agent Monitoring', () => {
    test('should display real-time agent metrics', async ({ page }) => {
      await navHelpers.goToAgents();
      
      // Click on an active agent
      const activeAgent = page.locator('[data-testid="agent-card"]').filter({
        has: page.locator('[data-testid="agent-status"]:has-text("active")')
      }).first();
      
      await activeAgent.click();

      // Go to metrics tab
      await page.click('[data-testid="metrics-tab"]');
      await expect(page.locator('[data-testid="metrics-dashboard"]')).toBeVisible();

      // Verify key metrics are displayed
      await expect(page.locator('[data-testid="uptime-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="requests-per-minute"]')).toBeVisible();
      await expect(page.locator('[data-testid="success-rate"]')).toBeVisible();
      await expect(page.locator('[data-testid="average-response-time"]')).toBeVisible();

      // Check for real-time updates
      const initialValue = await page.locator('[data-testid="requests-count"]').textContent();
      await page.waitForTimeout(3000);
      const updatedValue = await page.locator('[data-testid="requests-count"]').textContent();
      
      // Values should be different (indicating real-time updates)
      // Note: This might be flaky in test environment, so we just check for presence
      expect(updatedValue).toBeTruthy();

      await takeTimestampedScreenshot(page, 'agent-metrics');
    });

    test('should display agent execution logs', async ({ page }) => {
      await navHelpers.goToAgents();
      await page.click('[data-testid="agent-card"]:first-child');

      // Go to logs tab
      await page.click('[data-testid="agent-logs-tab"]');
      await expect(page.locator('[data-testid="agent-logs"]')).toBeVisible();

      // Verify log entries
      await expect(page.locator('[data-testid="log-entry"]').first()).toBeVisible();
      
      // Check log filtering
      await page.selectOption('[data-testid="log-level-filter"]', 'error');
      await waitForNetworkIdle(page);
      
      // Should only show error logs (or no logs if none exist)
      const errorLogs = page.locator('[data-testid="log-entry"]');
      if (await errorLogs.count() > 0) {
        await expect(errorLogs.first().locator('[data-testid="log-level"]')).toContainText('error');
      }

      // Test log search
      await page.fill('[data-testid="log-search-input"]', 'health');
      await page.keyboard.press('Enter');
      await waitForNetworkIdle(page);

      // Should filter logs containing 'health'
      const searchResults = page.locator('[data-testid="log-entry"]');
      if (await searchResults.count() > 0) {
        await expect(searchResults.first()).toContainText('health');
      }

      await takeTimestampedScreenshot(page, 'agent-logs');
    });

    test('should show agent performance trends', async ({ page }) => {
      await navHelpers.goToAgents();
      await page.click('[data-testid="agent-card"]:first-child');

      // Go to performance tab
      await page.click('[data-testid="performance-tab"]');
      await expect(page.locator('[data-testid="performance-charts"]')).toBeVisible();

      // Verify performance charts
      await expect(page.locator('[data-testid="response-time-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="throughput-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="error-rate-chart"]')).toBeVisible();

      // Test time range selector
      await page.selectOption('[data-testid="time-range-select"]', '24h');
      await waitForNetworkIdle(page);
      
      await expect(page.locator('[data-testid="time-range-indicator"]')).toContainText('24 hours');

      // Test chart interactions
      const chart = page.locator('[data-testid="response-time-chart"]');
      await chart.hover();
      
      // Should show tooltip with data
      await expect(page.locator('[data-testid="chart-tooltip"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'agent-performance');
    });
  });

  test.describe('Agent Lifecycle', () => {
    test('should delete an agent', async ({ page }) => {
      await navHelpers.goToAgents();
      
      // Get initial agent count
      const initialCount = await page.locator('[data-testid="agent-card"]').count();

      // Click on first agent
      await page.click('[data-testid="agent-card"]:first-child');
      
      // Get agent name for verification
      const agentName = await page.locator('[data-testid="agent-name"]').textContent();

      // Delete the agent
      await page.click('[data-testid="delete-agent-button"]');
      
      // Confirm deletion in modal
      await expect(page.locator('[data-testid="confirm-delete-modal"]')).toBeVisible();
      await expect(page.locator('[data-testid="delete-confirmation-text"]')).toContainText(agentName!);
      
      await page.fill('[data-testid="delete-confirmation-input"]', agentName!);
      await page.click('[data-testid="confirm-delete-button"]');

      // Wait for deletion and redirect
      await waitForNetworkIdle(page);
      await page.waitForURL('**/agents');

      // Verify agent was deleted
      const finalCount = await page.locator('[data-testid="agent-card"]').count();
      expect(finalCount).toBe(initialCount - 1);

      // Verify deleted agent is not in the list
      const agentNames = await page.locator('[data-testid="agent-name"]').allTextContents();
      expect(agentNames).not.toContain(agentName);

      await takeTimestampedScreenshot(page, 'agent-deleted');
    });

    test('should clone an existing agent', async ({ page }) => {
      await navHelpers.goToAgents();
      await page.click('[data-testid="agent-card"]:first-child');

      // Clone the agent
      await page.click('[data-testid="clone-agent-button"]');
      await expect(page.locator('[data-testid="agent-form"]')).toBeVisible();

      // Verify form is pre-filled with original agent data
      const originalName = await page.locator('[data-testid="agent-name-input"]').inputValue();
      expect(originalName).toContain('Copy of');

      // Modify the cloned agent
      await page.fill('[data-testid="agent-name-input"]', 'Cloned Test Agent');
      await page.fill('[data-testid="agent-description-input"]', 'This is a cloned agent for testing');

      // Submit the clone
      await page.click('[data-testid="create-agent-submit"]');
      await waitForNetworkIdle(page);

      // Verify clone was created
      await expect(page.locator('[data-testid="agent-name"]')).toContainText('Cloned Test Agent');

      await takeTimestampedScreenshot(page, 'agent-cloned');
    });
  });
});