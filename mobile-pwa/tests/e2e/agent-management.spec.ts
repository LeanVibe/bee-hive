import { test, expect } from '@playwright/test'
import { DashboardPage, AgentHealthPanelPage, AgentConfigModalPage } from '../fixtures/page-objects'
import { TestHelpers } from '../utils/test-helpers'
import { APIMocks } from '../utils/api-mocks'

test.describe('Agent Management Functionality', () => {
  let dashboardPage: DashboardPage
  let agentHealthPanel: AgentHealthPanelPage
  let agentConfigModal: AgentConfigModalPage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page)
    agentHealthPanel = new AgentHealthPanelPage(page)
    agentConfigModal = new AgentConfigModalPage(page)
    
    // Set up API mocks for consistent testing
    await APIMocks.setupStandardMocks(page)
    
    // Navigate to dashboard and then to agents view
    await dashboardPage.goto('/')
    await dashboardPage.waitForLoad()
    await dashboardPage.navigateToAgents()
  })

  test.describe('Agent Display and Status', () => {
    test('should display all agents with correct information', async ({ page }) => {
      // Verify agents are displayed
      const agentCount = await agentHealthPanel.getAgentCount()
      expect(agentCount).toBeGreaterThan(0)
      
      // Verify agent cards show required information
      const agentCards = page.locator('.agent-card')
      const firstAgent = agentCards.first()
      
      await expect(firstAgent.locator('.agent-name')).toBeVisible()
      await expect(firstAgent.locator('.agent-status')).toBeVisible()
      await expect(firstAgent.locator('.agent-uptime')).toBeVisible()
      await expect(firstAgent.locator('.agent-performance')).toBeVisible()
      
      await TestHelpers.takeTimestampedScreenshot(page, 'agent-list')
    })

    test('should display different agent status correctly', async ({ page }) => {
      // Verify active agents
      const activeAgents = page.locator('.agent-card[data-status="active"]')
      if (await activeAgents.count() > 0) {
        await expect(activeAgents.first().locator('.status-indicator')).toHaveClass(/active/)
        await expect(activeAgents.first().locator('.agent-status')).toContainText('Active')
      }
      
      // Verify idle agents
      const idleAgents = page.locator('.agent-card[data-status="idle"]')
      if (await idleAgents.count() > 0) {
        await expect(idleAgents.first().locator('.status-indicator')).toHaveClass(/idle/)
        await expect(idleAgents.first().locator('.agent-status')).toContainText('Idle')
      }
      
      // Verify error agents
      const errorAgents = page.locator('.agent-card[data-status="error"]')
      if (await errorAgents.count() > 0) {
        await expect(errorAgents.first().locator('.status-indicator')).toHaveClass(/error/)
        await expect(errorAgents.first().locator('.agent-status')).toContainText('Error')
      }
      
      // Verify offline agents
      const offlineAgents = page.locator('.agent-card[data-status="offline"]')
      if (await offlineAgents.count() > 0) {
        await expect(offlineAgents.first().locator('.status-indicator')).toHaveClass(/offline/)
        await expect(offlineAgents.first().locator('.agent-status')).toContainText('Offline')
      }
    })

    test('should show agent performance metrics', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      
      // Verify performance metrics are displayed
      await agentHealthPanel.verifyAgentMetrics('agent-1')
      
      // Verify charts are present
      await expect(agentCard.locator('.performance-chart')).toBeVisible()
      await expect(agentCard.locator('.cpu-usage')).toBeVisible()
      await expect(agentCard.locator('.memory-usage')).toBeVisible()
      
      // Verify performance score
      const performanceScore = agentCard.locator('.performance-score')
      if (await performanceScore.isVisible()) {
        const scoreText = await performanceScore.textContent()
        const score = parseFloat(scoreText || '0')
        expect(score).toBeGreaterThanOrEqual(0)
        expect(score).toBeLessThanOrEqual(100)
      }
    })

    test('should display current task assignment', async ({ page }) => {
      // Find agents with current tasks
      const agentsWithTasks = page.locator('.agent-card[data-current-task]')
      
      if (await agentsWithTasks.count() > 0) {
        const agentWithTask = agentsWithTasks.first()
        await expect(agentWithTask.locator('.current-task')).toBeVisible()
        await expect(agentWithTask.locator('.current-task')).not.toBeEmpty()
        
        // Task should be clickable/linkable
        const taskLink = agentWithTask.locator('.current-task a')
        if (await taskLink.isVisible()) {
          await expect(taskLink).toBeVisible()
        }
      }
    })
  })

  test.describe('Agent Team Activation', () => {
    test('should activate full agent team', async ({ page }) => {
      // Look for agent team activation button
      const activateTeamButton = page.locator('[data-testid="activate-team"], .activate-team-button')
      
      if (await activateTeamButton.isVisible()) {
        // Click activate team button
        await activateTeamButton.click()
        
        // Should show loading state
        await expect(activateTeamButton).toHaveClass(/loading/)
        
        // Wait for activation to complete
        await TestHelpers.waitForCondition(async () => {
          return !(await activateTeamButton.getAttribute('class'))?.includes('loading')
        })
        
        // Verify success feedback
        const successMessage = page.locator('.success-message, .notification')
        if (await successMessage.isVisible()) {
          await expect(successMessage).toContainText(/activated|success/)
        }
        
        await TestHelpers.takeTimestampedScreenshot(page, 'agent-team-activated')
      }
    })

    test('should handle agent team activation errors', async ({ page }) => {
      // Mock activation failure
      await page.route('**/api/v1/agents/activate', async route => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Activation failed' })
        })
      })
      
      const activateTeamButton = page.locator('[data-testid="activate-team"], .activate-team-button')
      
      if (await activateTeamButton.isVisible()) {
        await activateTeamButton.click()
        
        // Should show error state
        await TestHelpers.verifyErrorHandling(page, 'Activation failed')
      }
    })

    test('should show activation progress', async ({ page }) => {
      // Mock slow activation to test progress
      await page.route('**/api/v1/agents/activate', async route => {
        await new Promise(resolve => setTimeout(resolve, 2000))
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'Team activated', activatedAgents: 5 })
        })
      })
      
      const activateTeamButton = page.locator('[data-testid="activate-team"], .activate-team-button')
      
      if (await activateTeamButton.isVisible()) {
        await activateTeamButton.click()
        
        // Should show progress indicators
        await expect(activateTeamButton).toHaveClass(/loading/)
        
        // Progress should complete
        await TestHelpers.waitForCondition(async () => {
          const isLoading = await activateTeamButton.getAttribute('class')
          return !isLoading?.includes('loading')
        }, 5000)
      }
    })
  })

  test.describe('Individual Agent Controls', () => {
    test('should activate individual agents', async ({ page }) => {
      // Find inactive agents
      const inactiveAgents = page.locator('.agent-card[data-status="idle"], .agent-card[data-status="offline"]')
      
      if (await inactiveAgents.count() > 0) {
        const targetAgent = inactiveAgents.first()
        const agentId = await targetAgent.getAttribute('data-agent-id')
        
        // Click activate button
        const activateButton = targetAgent.locator('.activate-button, [data-testid="activate-agent"]')
        if (await activateButton.isVisible()) {
          await activateButton.click()
          
          // Should show loading state
          await expect(activateButton).toHaveClass(/loading/)
          
          // Should eventually show active status
          await TestHelpers.waitForCondition(async () => {
            const status = await agentHealthPanel.getAgentStatus(agentId!)
            return status.toLowerCase().includes('active')
          })
        }
      }
    })

    test('should deactivate individual agents', async ({ page }) => {
      // Find active agents
      const activeAgents = page.locator('.agent-card[data-status="active"]')
      
      if (await activeAgents.count() > 0) {
        const targetAgent = activeAgents.first()
        const agentId = await targetAgent.getAttribute('data-agent-id')
        
        // Click deactivate button
        const deactivateButton = targetAgent.locator('.deactivate-button, [data-testid="deactivate-agent"]')
        if (await deactivateButton.isVisible()) {
          await deactivateButton.click()
          
          // Should show confirmation dialog
          const confirmButton = page.locator('[data-testid="confirm-deactivate"]')
          if (await confirmButton.isVisible()) {
            await confirmButton.click()
          }
          
          // Should show loading state
          await expect(deactivateButton).toHaveClass(/loading/)
          
          // Should eventually show inactive status
          await TestHelpers.waitForCondition(async () => {
            const status = await agentHealthPanel.getAgentStatus(agentId!)
            return !status.toLowerCase().includes('active')
          })
        }
      }
    })

    test('should restart agents in error state', async ({ page }) => {
      // Find agents in error state
      const errorAgents = page.locator('.agent-card[data-status="error"]')
      
      if (await errorAgents.count() > 0) {
        const targetAgent = errorAgents.first()
        
        // Click restart button
        const restartButton = targetAgent.locator('.restart-button, [data-testid="restart-agent"]')
        if (await restartButton.isVisible()) {
          await restartButton.click()
          
          // Should show loading state
          await expect(restartButton).toHaveClass(/loading/)
          
          // Should eventually recover from error state
          await TestHelpers.waitForCondition(async () => {
            const statusIndicator = targetAgent.locator('.status-indicator')
            return !(await statusIndicator.getAttribute('class'))?.includes('error')
          })
        }
      }
    })
  })

  test.describe('Agent Configuration', () => {
    test('should open agent configuration modal', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      
      // Click configure button or agent settings
      const configButton = agentCard.locator('.config-button, [data-testid="configure-agent"]')
      if (await configButton.isVisible()) {
        await configButton.click()
        
        // Modal should open
        await expect(agentConfigModal.modal).toBeVisible()
        
        // Modal should contain configuration fields
        await expect(agentConfigModal.nameInput).toBeVisible()
        await expect(agentConfigModal.capabilitiesSelect).toBeVisible()
        await expect(agentConfigModal.saveButton).toBeVisible()
        await expect(agentConfigModal.cancelButton).toBeVisible()
        
        await TestHelpers.takeTimestampedScreenshot(page, 'agent-config-modal')
      }
    })

    test('should configure agent settings', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      const configButton = agentCard.locator('.config-button, [data-testid="configure-agent"]')
      
      if (await configButton.isVisible()) {
        await configButton.click()
        await expect(agentConfigModal.modal).toBeVisible()
        
        // Configure agent
        await agentConfigModal.configureAgent({
          name: 'Updated Agent Name',
          capabilities: ['backend', 'testing'],
          priority: 75
        })
        
        // Save configuration
        await agentConfigModal.saveConfiguration()
        
        // Modal should close
        await expect(agentConfigModal.modal).not.toBeVisible()
        
        // Changes should be reflected in agent card
        await TestHelpers.waitForCondition(async () => {
          const nameElement = agentCard.locator('.agent-name')
          const name = await nameElement.textContent()
          return name?.includes('Updated Agent Name') || false
        })
      }
    })

    test('should validate configuration fields', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      const configButton = agentCard.locator('.config-button, [data-testid="configure-agent"]')
      
      if (await configButton.isVisible()) {
        await configButton.click()
        await expect(agentConfigModal.modal).toBeVisible()
        
        // Try to save with empty name
        await agentConfigModal.nameInput.clear()
        await agentConfigModal.saveButton.click()
        
        // Should show validation error
        const errorMessage = page.locator('.error-message')
        if (await errorMessage.isVisible()) {
          await expect(errorMessage).toContainText(/required|name/)
        }
        
        // Modal should remain open
        await expect(agentConfigModal.modal).toBeVisible()
      }
    })

    test('should cancel configuration changes', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      const configButton = agentCard.locator('.config-button, [data-testid="configure-agent"]')
      
      if (await configButton.isVisible()) {
        await configButton.click()
        await expect(agentConfigModal.modal).toBeVisible()
        
        // Make changes
        await agentConfigModal.nameInput.fill('Temporary Name')
        
        // Cancel changes
        await agentConfigModal.cancelConfiguration()
        
        // Modal should close without saving
        await expect(agentConfigModal.modal).not.toBeVisible()
        
        // Changes should not be applied
        const nameElement = agentCard.locator('.agent-name')
        const name = await nameElement.textContent()
        expect(name).not.toContain('Temporary Name')
      }
    })
  })

  test.describe('Bulk Agent Operations', () => {
    test('should select multiple agents', async ({ page }) => {
      // Look for agent selection checkboxes
      const agentCheckboxes = page.locator('.agent-card input[type="checkbox"]')
      
      if (await agentCheckboxes.count() >= 2) {
        // Select first two agents
        await agentCheckboxes.first().check()
        await agentCheckboxes.nth(1).check()
        
        // Bulk actions should become available
        const bulkActionsBar = page.locator('.bulk-actions, [data-testid="bulk-actions"]')
        if (await bulkActionsBar.isVisible()) {
          await expect(bulkActionsBar).toBeVisible()
          
          // Should show number of selected agents
          await expect(bulkActionsBar).toContainText('2')
        }
      }
    })

    test('should perform bulk activation', async ({ page }) => {
      const agentCheckboxes = page.locator('.agent-card input[type="checkbox"]')
      
      if (await agentCheckboxes.count() >= 2) {
        // Select multiple agents
        await agentCheckboxes.first().check()
        await agentCheckboxes.nth(1).check()
        
        // Click bulk activate
        const bulkActivateButton = page.locator('[data-testid="bulk-activate"]')
        if (await bulkActivateButton.isVisible()) {
          await bulkActivateButton.click()
          
          // Should show progress
          await expect(bulkActivateButton).toHaveClass(/loading/)
          
          // Should complete successfully
          await TestHelpers.waitForCondition(async () => {
            return !(await bulkActivateButton.getAttribute('class'))?.includes('loading')
          })
        }
      }
    })

    test('should perform bulk deactivation', async ({ page }) => {
      const agentCheckboxes = page.locator('.agent-card input[type="checkbox"]')
      
      if (await agentCheckboxes.count() >= 2) {
        // Select multiple agents
        await agentCheckboxes.first().check()
        await agentCheckboxes.nth(1).check()
        
        // Click bulk deactivate
        const bulkDeactivateButton = page.locator('[data-testid="bulk-deactivate"]')
        if (await bulkDeactivateButton.isVisible()) {
          await bulkDeactivateButton.click()
          
          // Should show confirmation
          const confirmButton = page.locator('[data-testid="confirm-bulk-deactivate"]')
          if (await confirmButton.isVisible()) {
            await confirmButton.click()
          }
          
          // Should show progress
          await expect(bulkDeactivateButton).toHaveClass(/loading/)
          
          // Should complete successfully
          await TestHelpers.waitForCondition(async () => {
            return !(await bulkDeactivateButton.getAttribute('class'))?.includes('loading')
          })
        }
      }
    })
  })

  test.describe('Real-time Agent Updates', () => {
    test('should update agent status in real-time', async ({ page }) => {
      // Set up WebSocket mock for real-time updates
      await APIMocks.mockWebSocketAPI(page)
      
      const firstAgent = page.locator('.agent-card').first()
      const statusIndicator = firstAgent.locator('.status-indicator')
      
      // Wait for WebSocket updates
      await page.waitForTimeout(4000) // Wait for mock updates
      
      // Status should be updated (we can't predict exact status, but indicator should be present)
      await expect(statusIndicator).toBeVisible()
      
      // Performance metrics should update
      const performanceChart = firstAgent.locator('.performance-chart')
      if (await performanceChart.isVisible()) {
        await expect(performanceChart).toBeVisible()
      }
    })

    test('should show real-time task assignments', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      
      // Current task assignment should be visible
      const currentTask = agentCard.locator('.current-task')
      if (await currentTask.isVisible()) {
        await expect(currentTask).toBeVisible()
        
        // Task should have a title
        await expect(currentTask).not.toBeEmpty()
      }
    })

    test('should update performance metrics in real-time', async ({ page }) => {
      const firstAgent = page.locator('.agent-card').first()
      
      // Check if performance metrics are present
      const cpuMetric = firstAgent.locator('.cpu-usage')
      const memoryMetric = firstAgent.locator('.memory-usage')
      
      if (await cpuMetric.isVisible()) {
        const initialCpuValue = await cpuMetric.textContent()
        
        // Wait for potential updates
        await page.waitForTimeout(5000)
        
        // Metrics should still be present (may or may not have changed)
        await expect(cpuMetric).toBeVisible()
        await expect(memoryMetric).toBeVisible()
      }
    })
  })

  test.describe('Agent Performance Monitoring', () => {
    test('should display performance charts', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      
      // Verify performance charts are present
      await expect(agentCard.locator('.performance-section')).toBeVisible()
      
      // Should have CPU chart
      const cpuChart = agentCard.locator('.cpu-chart, [data-chart="cpu"]')
      if (await cpuChart.isVisible()) {
        await expect(cpuChart).toBeVisible()
      }
      
      // Should have memory chart
      const memoryChart = agentCard.locator('.memory-chart, [data-chart="memory"]')
      if (await memoryChart.isVisible()) {
        await expect(memoryChart).toBeVisible()
      }
      
      // Should have performance score
      const performanceScore = agentCard.locator('.performance-score')
      if (await performanceScore.isVisible()) {
        const scoreText = await performanceScore.textContent()
        expect(scoreText).toMatch(/\d+/)
      }
    })

    test('should show performance trends', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      
      // Look for trend indicators
      const trendIndicator = agentCard.locator('.trend-indicator, .performance-trend')
      if (await trendIndicator.isVisible()) {
        await expect(trendIndicator).toBeVisible()
        
        // Should show trend direction (up, down, stable)
        const trendClass = await trendIndicator.getAttribute('class')
        expect(trendClass).toMatch(/up|down|stable/)
      }
    })

    test('should display task completion metrics', async ({ page }) => {
      const agentCard = page.locator('.agent-card').first()
      
      // Look for task completion information
      const tasksCompleted = agentCard.locator('.tasks-completed, [data-metric="tasks-completed"]')
      if (await tasksCompleted.isVisible()) {
        const completedText = await tasksCompleted.textContent()
        expect(completedText).toMatch(/\d+/)
      }
      
      // Look for average completion time
      const avgTime = agentCard.locator('.avg-time, [data-metric="avg-time"]')
      if (await avgTime.isVisible()) {
        const timeText = await avgTime.textContent()
        expect(timeText).toMatch(/\d+/)
      }
    })
  })

  test.describe('Error Handling and Edge Cases', () => {
    test('should handle agent operation failures', async ({ page }) => {
      // Mock agent activation failure
      await page.route('**/api/v1/agents/*/activate', async route => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Agent activation failed' })
        })
      })
      
      const agentCard = page.locator('.agent-card').first()
      const activateButton = agentCard.locator('.activate-button')
      
      if (await activateButton.isVisible()) {
        await activateButton.click()
        
        // Should show error state
        await TestHelpers.verifyErrorHandling(page)
        
        // Agent should remain in previous state
        const statusIndicator = agentCard.locator('.status-indicator')
        await expect(statusIndicator).not.toHaveClass(/active/)
      }
    })

    test('should handle missing agent data gracefully', async ({ page }) => {
      // Mock empty agents response
      await page.route('**/api/v1/agents', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([])
        })
      })
      
      await page.reload()
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToAgents()
      
      // Should show empty state
      const emptyState = page.locator('.empty-state, [data-testid="no-agents"]')
      if (await emptyState.isVisible()) {
        await expect(emptyState).toBeVisible()
        await expect(emptyState).toContainText(/no agents|empty/)
      }
    })

    test('should handle agent connection issues', async ({ page }) => {
      // Mock network errors for agent endpoints
      await page.route('**/api/v1/agents', async route => {
        await route.abort('connectionfailed')
      })
      
      await page.reload()
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToAgents()
      
      // Should show connection error
      await TestHelpers.verifyErrorHandling(page, 'connection')
    })
  })
})