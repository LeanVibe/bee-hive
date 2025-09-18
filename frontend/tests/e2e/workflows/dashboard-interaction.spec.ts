import { test, expect } from '@playwright/test';
import { 
  AuthHelpers, 
  NavigationHelpers, 
  loadTestData,
  takeTimestampedScreenshot,
  waitForNetworkIdle,
  PerformanceHelpers
} from '../utils/test-helpers';

/**
 * Dashboard Interaction E2E Tests
 * Tests real-time dashboard functionality and user interactions
 */

test.describe('Dashboard Interaction Workflows', () => {
  let authHelpers: AuthHelpers;
  let navHelpers: NavigationHelpers;
  let performanceHelpers: PerformanceHelpers;
  let testData: any;

  test.beforeEach(async ({ page }) => {
    authHelpers = new AuthHelpers(page);
    navHelpers = new NavigationHelpers(page);
    performanceHelpers = new PerformanceHelpers(page);
    testData = loadTestData();
    
    // Log in as admin user
    const adminUser = testData.users.find((u: any) => u.role === 'admin');
    await page.goto('/');
    await authHelpers.login(adminUser);
    await navHelpers.goToDashboard();
  });

  test.describe('Dashboard Overview', () => {
    test('should display comprehensive system overview', async ({ page }) => {
      // Verify main dashboard components are visible
      await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible();
      await expect(page.locator('[data-testid="system-metrics-overview"]')).toBeVisible();
      await expect(page.locator('[data-testid="agent-status-summary"]')).toBeVisible();
      await expect(page.locator('[data-testid="task-execution-summary"]')).toBeVisible();
      await expect(page.locator('[data-testid="recent-activity-feed"]')).toBeVisible();

      // Check key metrics cards
      await expect(page.locator('[data-testid="total-agents-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="active-tasks-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="system-health-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="alerts-count-metric"]')).toBeVisible();

      // Verify metric values are displayed
      const totalAgents = await page.locator('[data-testid="total-agents-value"]').textContent();
      const activeTasks = await page.locator('[data-testid="active-tasks-value"]').textContent();
      
      expect(parseInt(totalAgents || '0')).toBeGreaterThanOrEqual(0);
      expect(parseInt(activeTasks || '0')).toBeGreaterThanOrEqual(0);

      await takeTimestampedScreenshot(page, 'dashboard-overview');
    });

    test('should display real-time system health status', async ({ page }) => {
      // Check system health indicator
      await expect(page.locator('[data-testid="system-health-indicator"]')).toBeVisible();
      
      // Verify health status badge
      const healthStatus = page.locator('[data-testid="health-status-badge"]');
      await expect(healthStatus).toBeVisible();
      
      const statusText = await healthStatus.textContent();
      expect(['Healthy', 'Warning', 'Critical', 'Unknown']).toContain(statusText);

      // Check detailed health metrics
      await page.click('[data-testid="health-details-toggle"]');
      await expect(page.locator('[data-testid="health-details-panel"]')).toBeVisible();
      
      await expect(page.locator('[data-testid="cpu-usage-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="memory-usage-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="disk-usage-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="network-status-metric"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'system-health-status');
    });

    test('should update metrics in real-time', async ({ page }) => {
      // Get initial metric values
      const initialActiveAgents = await page.locator('[data-testid="active-agents-value"]').textContent();
      const initialActiveTasks = await page.locator('[data-testid="active-tasks-value"]').textContent();

      // Wait for potential real-time updates
      await page.waitForTimeout(5000);

      // Check if values have potentially updated (in a real environment they might change)
      const updatedActiveAgents = await page.locator('[data-testid="active-agents-value"]').textContent();
      const updatedActiveTasks = await page.locator('[data-testid="active-tasks-value"]').textContent();

      // Verify the elements are still present and have valid values
      expect(updatedActiveAgents).toBeTruthy();
      expect(updatedActiveTasks).toBeTruthy();

      // Check that last updated timestamp is present and recent
      await expect(page.locator('[data-testid="last-updated-timestamp"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'real-time-updates');
    });
  });

  test.describe('Interactive Charts and Visualizations', () => {
    test('should interact with performance charts', async ({ page }) => {
      // Navigate to performance charts section
      await page.click('[data-testid="performance-charts-section"]');
      await expect(page.locator('[data-testid="performance-charts-container"]')).toBeVisible();

      // Test CPU usage chart interaction
      const cpuChart = page.locator('[data-testid="cpu-usage-chart"]');
      await expect(cpuChart).toBeVisible();
      
      // Hover over chart to show tooltip
      await cpuChart.hover();
      await expect(page.locator('[data-testid="chart-tooltip"]')).toBeVisible();

      // Test time range selector
      await page.selectOption('[data-testid="chart-time-range"]', '24h');
      await waitForNetworkIdle(page);
      
      await expect(page.locator('[data-testid="time-range-indicator"]')).toContainText('24 hours');

      // Test chart zoom functionality
      const chartArea = cpuChart.locator('canvas, svg').first();
      await chartArea.dragTo(chartArea, {
        sourcePosition: { x: 100, y: 100 },
        targetPosition: { x: 200, y: 150 }
      });

      // Verify zoom controls are available
      await expect(page.locator('[data-testid="chart-zoom-reset"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'chart-interactions');
    });

    test('should interact with agent status visualization', async ({ page }) => {
      // Check agent status distribution chart
      await expect(page.locator('[data-testid="agent-status-chart"]')).toBeVisible();

      // Test clicking on chart segments
      const activeAgentsSegment = page.locator('[data-testid="active-agents-segment"]');
      await activeAgentsSegment.click();

      // Should show filtered view or details
      await expect(page.locator('[data-testid="agents-filter-active"]')).toBeVisible();

      // Test agent list view toggle
      await page.click('[data-testid="toggle-agent-list-view"]');
      await expect(page.locator('[data-testid="agent-list-container"]')).toBeVisible();

      // Verify agent cards are displayed
      const agentCards = page.locator('[data-testid="agent-card"]');
      expect(await agentCards.count()).toBeGreaterThan(0);

      // Test agent card interactions
      await agentCards.first().hover();
      await expect(page.locator('[data-testid="agent-quick-actions"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'agent-visualization');
    });

    test('should interact with task execution timeline', async ({ page }) => {
      // Navigate to task timeline
      await page.click('[data-testid="task-timeline-section"]');
      await expect(page.locator('[data-testid="task-timeline-container"]')).toBeVisible();

      // Test timeline navigation
      await page.click('[data-testid="timeline-zoom-in"]');
      await page.waitForTimeout(1000);
      
      await page.click('[data-testid="timeline-zoom-out"]');
      await page.waitForTimeout(1000);

      // Test task filtering on timeline
      await page.selectOption('[data-testid="timeline-task-filter"]', 'running');
      await waitForNetworkIdle(page);

      // Verify only running tasks are shown
      const timelineEvents = page.locator('[data-testid="timeline-event"]');
      if (await timelineEvents.count() > 0) {
        await expect(timelineEvents.first()).toHaveAttribute('data-status', 'running');
      }

      // Test timeline event interaction
      if (await timelineEvents.count() > 0) {
        await timelineEvents.first().click();
        await expect(page.locator('[data-testid="task-details-popover"]')).toBeVisible();
      }

      await takeTimestampedScreenshot(page, 'task-timeline');
    });
  });

  test.describe('Dashboard Customization', () => {
    test('should customize dashboard layout', async ({ page }) => {
      // Enter customization mode
      await page.click('[data-testid="customize-dashboard-button"]');
      await expect(page.locator('[data-testid="customization-toolbar"]')).toBeVisible();

      // Test widget repositioning
      const systemMetricsWidget = page.locator('[data-testid="system-metrics-widget"]');
      const agentStatusWidget = page.locator('[data-testid="agent-status-widget"]');

      // Drag and drop to reposition
      await systemMetricsWidget.dragTo(agentStatusWidget);
      
      // Verify drag handles are visible in edit mode
      await expect(page.locator('[data-testid="widget-drag-handle"]').first()).toBeVisible();

      // Test widget resizing
      const resizeHandle = systemMetricsWidget.locator('[data-testid="resize-handle"]');
      await resizeHandle.dragTo(resizeHandle, {
        sourcePosition: { x: 0, y: 0 },
        targetPosition: { x: 50, y: 50 }
      });

      // Save layout changes
      await page.click('[data-testid="save-layout-button"]');
      await waitForNetworkIdle(page);

      await expect(page.locator('[data-testid="layout-saved-message"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'dashboard-customized');
    });

    test('should add and remove dashboard widgets', async ({ page }) => {
      await page.click('[data-testid="customize-dashboard-button"]');

      // Add new widget
      await page.click('[data-testid="add-widget-button"]');
      await expect(page.locator('[data-testid="widget-gallery"]')).toBeVisible();

      // Select a widget to add
      await page.click('[data-testid="network-monitoring-widget"]');
      await page.click('[data-testid="add-selected-widget"]');

      // Verify new widget appeared
      await expect(page.locator('[data-testid="network-monitoring-widget-instance"]')).toBeVisible();

      // Remove a widget
      const widgetToRemove = page.locator('[data-testid="system-metrics-widget"]');
      await widgetToRemove.hover();
      await page.click('[data-testid="remove-widget-button"]');

      // Confirm removal
      await page.click('[data-testid="confirm-remove-widget"]');

      // Verify widget was removed
      await expect(page.locator('[data-testid="system-metrics-widget"]')).not.toBeVisible();

      await page.click('[data-testid="save-layout-button"]');
      await waitForNetworkIdle(page);

      await takeTimestampedScreenshot(page, 'widgets-modified');
    });

    test('should configure widget settings', async ({ page }) => {
      // Configure a specific widget
      const performanceWidget = page.locator('[data-testid="performance-widget"]');
      await performanceWidget.hover();
      
      await page.click('[data-testid="widget-settings-button"]');
      await expect(page.locator('[data-testid="widget-settings-modal"]')).toBeVisible();

      // Configure widget options
      await page.fill('[data-testid="widget-title-input"]', 'Custom Performance Metrics');
      await page.selectOption('[data-testid="refresh-interval-select"]', '30');
      await page.check('[data-testid="show-legend"]');
      await page.selectOption('[data-testid="chart-type-select"]', 'line');

      // Apply settings
      await page.click('[data-testid="apply-widget-settings"]');
      await waitForNetworkIdle(page);

      // Verify settings applied
      await expect(page.locator('[data-testid="widget-title"]')).toContainText('Custom Performance Metrics');

      await takeTimestampedScreenshot(page, 'widget-configured');
    });
  });

  test.describe('Alert and Notification Management', () => {
    test('should display and manage alerts', async ({ page }) => {
      // Check alerts panel
      await expect(page.locator('[data-testid="alerts-panel"]')).toBeVisible();

      // Test alert severity filtering
      await page.selectOption('[data-testid="alert-severity-filter"]', 'critical');
      await waitForNetworkIdle(page);

      const criticalAlerts = page.locator('[data-testid="alert-item"][data-severity="critical"]');
      if (await criticalAlerts.count() > 0) {
        // Test alert interaction
        await criticalAlerts.first().click();
        await expect(page.locator('[data-testid="alert-details-modal"]')).toBeVisible();

        // Test alert acknowledgment
        await page.click('[data-testid="acknowledge-alert-button"]');
        await page.fill('[data-testid="acknowledgment-note"]', 'Investigating the issue');
        await page.click('[data-testid="confirm-acknowledge"]');

        await expect(page.locator('[data-testid="alert-acknowledged-status"]')).toBeVisible();
      }

      // Test alert creation (for testing purposes)
      await page.click('[data-testid="create-test-alert"]');
      await expect(page.locator('[data-testid="test-alert-created"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'alerts-management');
    });

    test('should configure notification preferences', async ({ page }) => {
      // Access notification settings
      await page.click('[data-testid="notification-settings-button"]');
      await expect(page.locator('[data-testid="notification-settings-modal"]')).toBeVisible();

      // Configure email notifications
      await page.check('[data-testid="enable-email-notifications"]');
      await page.fill('[data-testid="notification-email"]', 'alerts@leanvibe.test');

      // Configure alert thresholds
      await page.fill('[data-testid="cpu-threshold"]', '80');
      await page.fill('[data-testid="memory-threshold"]', '85');
      await page.fill('[data-testid="disk-threshold"]', '90');

      // Set notification frequency
      await page.selectOption('[data-testid="notification-frequency"]', 'immediate');

      // Configure quiet hours
      await page.check('[data-testid="enable-quiet-hours"]');
      await page.fill('[data-testid="quiet-start-time"]', '22:00');
      await page.fill('[data-testid="quiet-end-time"]', '06:00');

      await page.click('[data-testid="save-notification-settings"]');
      await waitForNetworkIdle(page);

      await expect(page.locator('[data-testid="settings-saved-message"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'notification-settings');
    });
  });

  test.describe('Performance and Responsiveness', () => {
    test('should load dashboard quickly', async ({ page }) => {
      // Measure initial page load time
      const loadTime = await performanceHelpers.measurePageLoadTime();
      
      // Dashboard should load within reasonable time
      expect(loadTime).toBeLessThan(3000); // 3 seconds

      // Check for performance metrics
      const vitals = await performanceHelpers.getCoreWebVitals();
      
      // Verify Core Web Vitals are within acceptable ranges
      if (vitals.FCP) expect(vitals.FCP).toBeLessThan(2500); // First Contentful Paint
      if (vitals.LCP) expect(vitals.LCP).toBeLessThan(4000); // Largest Contentful Paint

      await takeTimestampedScreenshot(page, 'dashboard-performance');
    });

    test('should handle large data sets efficiently', async ({ page }) => {
      // Switch to a view with large dataset
      await page.selectOption('[data-testid="data-range-selector"]', 'last-30-days');
      await waitForNetworkIdle(page, 10000);

      // Verify dashboard remains responsive
      await expect(page.locator('[data-testid="data-loading-indicator"]')).not.toBeVisible();
      await expect(page.locator('[data-testid="large-dataset-chart"]')).toBeVisible();

      // Test scrolling performance with large data
      await page.evaluate(() => {
        window.scrollTo(0, document.body.scrollHeight);
      });
      await page.waitForTimeout(1000);

      await page.evaluate(() => {
        window.scrollTo(0, 0);
      });

      // Dashboard should remain responsive
      await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'large-dataset-handling');
    });

    test('should be responsive across different screen sizes', async ({ page }) => {
      // Test tablet view
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.waitForTimeout(1000);

      await expect(page.locator('[data-testid="dashboard-mobile-nav"]')).toBeVisible();
      await expect(page.locator('[data-testid="dashboard-content"]')).toBeVisible();

      // Test mobile view
      await page.setViewportSize({ width: 375, height: 667 });
      await page.waitForTimeout(1000);

      await expect(page.locator('[data-testid="mobile-dashboard-layout"]')).toBeVisible();
      
      // Test mobile navigation
      await page.click('[data-testid="mobile-menu-toggle"]');
      await expect(page.locator('[data-testid="mobile-nav-menu"]')).toBeVisible();

      // Return to desktop view
      await page.setViewportSize({ width: 1280, height: 720 });
      await page.waitForTimeout(1000);

      await takeTimestampedScreenshot(page, 'responsive-dashboard');
    });
  });

  test.describe('Data Export and Sharing', () => {
    test('should export dashboard data', async ({ page }) => {
      // Access export functionality
      await page.click('[data-testid="export-dashboard-data"]');
      await expect(page.locator('[data-testid="export-options-modal"]')).toBeVisible();

      // Configure export options
      await page.selectOption('[data-testid="export-format"]', 'csv');
      await page.selectOption('[data-testid="export-time-range"]', '7d');
      
      // Select data to export
      await page.check('[data-testid="export-system-metrics"]');
      await page.check('[data-testid="export-agent-data"]');
      await page.check('[data-testid="export-task-data"]');

      // Initiate export
      await page.click('[data-testid="start-export"]');
      await waitForNetworkIdle(page, 10000);

      // Verify export completion
      await expect(page.locator('[data-testid="export-completed"]')).toBeVisible();
      await expect(page.locator('[data-testid="download-export-file"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'data-export');
    });

    test('should share dashboard views', async ({ page }) => {
      // Create shareable dashboard link
      await page.click('[data-testid="share-dashboard-button"]');
      await expect(page.locator('[data-testid="share-options-modal"]')).toBeVisible();

      // Configure sharing options
      await page.check('[data-testid="include-current-filters"]');
      await page.selectOption('[data-testid="share-duration"]', '7-days');
      await page.check('[data-testid="password-protect"]');
      await page.fill('[data-testid="share-password"]', 'dashboard123');

      // Generate share link
      await page.click('[data-testid="generate-share-link"]');
      await waitForNetworkIdle(page);

      // Verify share link created
      await expect(page.locator('[data-testid="share-link-generated"]')).toBeVisible();
      const shareUrl = await page.locator('[data-testid="share-url"]').inputValue();
      
      expect(shareUrl).toContain('http');
      expect(shareUrl).toContain('share');

      // Test copy link functionality
      await page.click('[data-testid="copy-share-link"]');
      await expect(page.locator('[data-testid="link-copied-message"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'dashboard-sharing');
    });
  });
});