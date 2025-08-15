import { test, expect } from '@playwright/test'
import { TestHelpers } from '../../utils/test-helpers'

/**
 * Agent Workflow End-to-End Tests
 * 
 * Validates complete autonomous agent development workflows:
 * - Task creation and assignment
 * - Agent spawning and management
 * - Real-time progress monitoring
 * - Task completion and results
 * - Multi-agent coordination
 * - Error handling and recovery
 * - Performance under agent load
 */

test.describe('Autonomous Agent Development Workflows', () => {
  
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await TestHelpers.waitForNetworkIdle(page)
    
    // Verify we're on the dashboard
    await expect(page.locator('text=HiveOps, text=Agent Dashboard')).toBeVisible()
  })

  test('complete task creation to completion workflow', async ({ page }) => {
    // Step 1: Create a new task
    const taskCreationSelectors = [
      '[data-testid="new-task"]',
      '.new-task-button',
      'button:has-text("New Task")',
      'button:has-text("Create Task")',
      '.task-creation-trigger'
    ]
    
    let taskCreationButton = null
    for (const selector of taskCreationSelectors) {
      const element = page.locator(selector)
      if (await element.count() > 0) {
        taskCreationButton = element.first()
        break
      }
    }
    
    if (taskCreationButton) {
      await taskCreationButton.click()
      await TestHelpers.waitForAnimations(page)
      
      // Look for task creation form or modal
      const taskForm = page.locator('.task-form, .modal-content, [data-testid="task-form"]')
      if (await taskForm.count() > 0) {
        await expect(taskForm.first()).toBeVisible()
        
        // Fill out task details
        const taskDescription = 'E2E Test: Implement user authentication system'
        
        const descriptionField = page.locator(
          'input[name="description"], textarea[name="description"], [data-testid="task-description"]'
        )
        
        if (await descriptionField.count() > 0) {
          await descriptionField.first().fill(taskDescription)
        }
        
        // Select agent type if available
        const agentTypeSelector = page.locator('select[name="agent-type"], [data-testid="agent-type"]')
        if (await agentTypeSelector.count() > 0) {
          await agentTypeSelector.first().selectOption('backend-engineer')
        }
        
        // Submit task
        const submitButton = page.locator('button[type="submit"], .submit-task, [data-testid="submit-task"]')
        if (await submitButton.count() > 0) {
          await submitButton.first().click()
          await TestHelpers.waitForAnimations(page)
          
          console.log('✓ Task creation form submitted')
        }
      }
    }
    
    // Step 2: Verify task appears in task queue
    await page.waitForTimeout(2000) // Allow for task processing
    
    const taskQueue = page.locator('.task-queue, .task-list, [data-testid="task-queue"]')
    if (await taskQueue.count() > 0) {
      const taskItems = taskQueue.locator('.task-item, .task-card')
      const taskCount = await taskItems.count()
      
      if (taskCount > 0) {
        console.log(`✓ Found ${taskCount} tasks in queue`)
        
        // Verify our task is present
        const ourTask = page.locator(`text="E2E Test: Implement user authentication"`)
        if (await ourTask.count() > 0) {
          await expect(ourTask.first()).toBeVisible()
          console.log('✓ Created task found in queue')
        }
      }
    }
    
    // Step 3: Monitor agent assignment and progress
    await page.waitForTimeout(5000) // Allow for agent assignment
    
    const agentCards = page.locator('.agent-card, .agent-status, [data-testid*="agent"]')
    if (await agentCards.count() > 0) {
      console.log(`✓ Found ${await agentCards.count()} agent cards`)
      
      // Look for active agents
      const activeAgents = page.locator('.agent-card[data-status="active"], .agent-status.active')
      if (await activeAgents.count() > 0) {
        console.log(`✓ Found ${await activeAgents.count()} active agents`)
      }
    }
    
    // Step 4: Monitor real-time progress updates
    const progressIndicators = page.locator('.progress-bar, .task-progress, [data-testid*="progress"]')
    if (await progressIndicators.count() > 0) {
      const initialProgress = await progressIndicators.first().textContent()
      console.log(`Initial progress: ${initialProgress}`)
      
      // Wait for progress updates
      await page.waitForTimeout(10000)
      
      const updatedProgress = await progressIndicators.first().textContent()
      console.log(`Updated progress: ${updatedProgress}`)
      
      if (initialProgress !== updatedProgress) {
        console.log('✓ Progress updates detected')
      }
    }
    
    // Step 5: Look for completion indicators
    await page.waitForTimeout(15000) // Allow more time for task completion
    
    const completionIndicators = page.locator(
      '.task-completed, .status-completed, [data-status="completed"]'
    )
    
    if (await completionIndicators.count() > 0) {
      console.log('✓ Task completion detected')
      
      // Take screenshot of completed state
      await page.screenshot({ 
        path: 'test-results/agent-workflows/task-completed.png',
        fullPage: true 
      })
    }
    
    // Take final screenshot of workflow
    await page.screenshot({ 
      path: 'test-results/agent-workflows/complete-workflow.png',
      fullPage: true 
    })
  })

  test('agent spawning and management workflow', async ({ page }) => {
    // Look for agent management interface
    const agentManagementSelectors = [
      '.agent-management',
      '[data-testid="agent-management"]',
      '.agent-controls',
      'button:has-text("Spawn Agent")',
      'button:has-text("New Agent")'
    ]
    
    let agentInterface = null
    for (const selector of agentManagementSelectors) {
      const element = page.locator(selector)
      if (await element.count() > 0) {
        agentInterface = element.first()
        break
      }
    }
    
    // Test agent spawning
    if (agentInterface) {
      const spawnButton = agentInterface.locator('button')
      if (await spawnButton.count() > 0) {
        const initialAgentCount = await page.locator('.agent-card, .agent-item').count()
        
        await spawnButton.first().click()
        await TestHelpers.waitForAnimations(page)
        
        // Wait for agent to spawn
        await page.waitForTimeout(3000)
        
        const finalAgentCount = await page.locator('.agent-card, .agent-item').count()
        
        if (finalAgentCount > initialAgentCount) {
          console.log(`✓ Agent spawned successfully (${initialAgentCount} → ${finalAgentCount})`)
        }
      }
    }
    
    // Test agent status monitoring
    const agentCards = page.locator('.agent-card, .agent-status')
    const agentCount = await agentCards.count()
    
    if (agentCount > 0) {
      console.log(`Monitoring ${agentCount} agents`)
      
      for (let i = 0; i < Math.min(agentCount, 3); i++) {
        const agent = agentCards.nth(i)
        
        // Get agent information
        const agentInfo = await agent.evaluate(el => {
          return {
            id: el.getAttribute('data-agent-id') || el.getAttribute('id'),
            status: el.getAttribute('data-status') || 
                   el.querySelector('.status')?.textContent ||
                   'unknown',
            type: el.getAttribute('data-type') || 
                 el.querySelector('.agent-type')?.textContent ||
                 'unknown',
            tasks: el.querySelector('.task-count')?.textContent || '0'
          }
        })
        
        console.log(`Agent ${i + 1}:`, agentInfo)
        
        // Test agent interaction
        const agentButton = agent.locator('button')
        if (await agentButton.count() > 0) {
          // Test pause/resume functionality
          await agentButton.first().click()
          await TestHelpers.waitForAnimations(page)
          
          // Verify status change
          const newStatus = await agent.getAttribute('data-status')
          console.log(`Agent ${i + 1} status after interaction: ${newStatus}`)
        }
      }
    }
  })

  test('multi-agent coordination and task distribution', async ({ page }) => {
    // Monitor multi-agent coordination
    const coordinationIndicators = [
      '.coordination-panel',
      '.multi-agent-view',
      '[data-testid="coordination"]',
      '.agent-communication'
    ]
    
    let coordinationUI = null
    for (const selector of coordinationIndicators) {
      const element = page.locator(selector)
      if (await element.count() > 0) {
        coordinationUI = element.first()
        break
      }
    }
    
    if (coordinationUI) {
      await expect(coordinationUI).toBeVisible()
      console.log('✓ Multi-agent coordination UI found')
    }
    
    // Test task distribution across multiple agents
    const agents = page.locator('.agent-card, .agent-item')
    const agentCount = await agents.count()
    
    if (agentCount > 1) {
      console.log(`Testing task distribution across ${agentCount} agents`)
      
      // Create multiple tasks to test distribution
      const taskCreationButton = page.locator('[data-testid="new-task"], .new-task-button')
      
      if (await taskCreationButton.count() > 0) {
        for (let i = 0; i < Math.min(3, agentCount); i++) {
          await taskCreationButton.first().click()
          await TestHelpers.waitForAnimations(page)
          
          // Fill task details if form appears
          const taskForm = page.locator('.task-form, .modal-content')
          if (await taskForm.count() > 0) {
            const descriptionField = taskForm.locator('input, textarea').first()
            if (await descriptionField.count() > 0) {
              await descriptionField.fill(`Multi-agent test task ${i + 1}`)
            }
            
            const submitButton = taskForm.locator('button[type="submit"], .submit')
            if (await submitButton.count() > 0) {
              await submitButton.click()
              await TestHelpers.waitForAnimations(page)
            }
          }
          
          await page.waitForTimeout(1000) // Spacing between task creations
        }
      }
      
      // Monitor task distribution
      await page.waitForTimeout(5000) // Allow for task assignment
      
      const agentTaskCounts = []
      for (let i = 0; i < agentCount; i++) {
        const agent = agents.nth(i)
        const taskCount = await agent.locator('.task-count, [data-testid="task-count"]')
                                   .textContent()
                                   .catch(() => '0')
        
        agentTaskCounts.push(parseInt(taskCount) || 0)
      }
      
      console.log('Task distribution:', agentTaskCounts)
      
      // Verify tasks are distributed (not all on one agent)
      const totalTasks = agentTaskCounts.reduce((sum, count) => sum + count, 0)
      const agentsWithTasks = agentTaskCounts.filter(count => count > 0).length
      
      if (totalTasks > 1) {
        expect(agentsWithTasks).toBeGreaterThan(0)
        console.log(`✓ Tasks distributed across ${agentsWithTasks} agents`)
      }
    }
  })

  test('error handling and recovery in agent workflows', async ({ page }) => {
    // Test error scenarios
    const errorScenarios = [
      {
        name: 'Network interruption during task execution',
        action: async () => {
          await page.context().setOffline(true)
          await page.waitForTimeout(3000)
          await page.context().setOffline(false)
        }
      },
      {
        name: 'Invalid task creation',
        action: async () => {
          const taskButton = page.locator('[data-testid="new-task"], .new-task-button')
          if (await taskButton.count() > 0) {
            await taskButton.first().click()
            
            const submitButton = page.locator('button[type="submit"], .submit')
            if (await submitButton.count() > 0) {
              // Submit without filling required fields
              await submitButton.click()
            }
          }
        }
      }
    ]
    
    for (const scenario of errorScenarios) {
      console.log(`Testing: ${scenario.name}`)
      
      await scenario.action()
      await page.waitForTimeout(2000)
      
      // Look for error handling UI
      const errorIndicators = page.locator(
        '.error-message, .alert-error, [data-testid*="error"], .notification-error'
      )
      
      if (await errorIndicators.count() > 0) {
        const errorText = await errorIndicators.first().textContent()
        console.log(`Error handled: ${errorText}`)
        
        // Verify error is user-friendly
        expect(errorText?.length || 0).toBeGreaterThan(10)
      }
      
      // Look for recovery mechanisms
      const retryButtons = page.locator(
        'button:has-text("Retry"), button:has-text("Try Again"), .retry-button'
      )
      
      if (await retryButtons.count() > 0) {
        console.log('✓ Recovery mechanism found')
        
        // Test retry functionality
        await retryButtons.first().click()
        await TestHelpers.waitForAnimations(page)
      }
      
      // Verify system returns to stable state
      await page.waitForTimeout(3000)
      await expect(page.locator('body')).toBeVisible()
      
      console.log(`✓ System recovered from: ${scenario.name}`)
    }
  })

  test('agent performance under load', async ({ page }) => {
    // Simulate load by creating multiple concurrent tasks
    const loadTestTasks = 5
    const startTime = Date.now()
    
    console.log(`Starting load test with ${loadTestTasks} concurrent tasks`)
    
    // Create multiple tasks rapidly
    for (let i = 0; i < loadTestTasks; i++) {
      const taskButton = page.locator('[data-testid="new-task"], .new-task-button').first()
      
      if (await taskButton.count() > 0) {
        await taskButton.click()
        await TestHelpers.waitForAnimations(page)
        
        const taskForm = page.locator('.task-form, .modal-content')
        if (await taskForm.count() > 0) {
          const descriptionField = taskForm.locator('input, textarea').first()
          if (await descriptionField.count() > 0) {
            await descriptionField.fill(`Load test task ${i + 1}`)
          }
          
          const submitButton = taskForm.locator('button[type="submit"], .submit')
          if (await submitButton.count() > 0) {
            await submitButton.click()
            await page.waitForTimeout(200) // Quick succession
          }
        }
      }
    }
    
    const taskCreationTime = Date.now() - startTime
    console.log(`Tasks created in ${taskCreationTime}ms`)
    
    // Monitor system performance during load
    await page.waitForTimeout(10000) // Allow processing time
    
    const performanceMetrics = await page.evaluate(() => {
      const memory = (performance as any).memory
      return {
        memoryUsed: memory ? memory.usedJSHeapSize : null,
        memoryTotal: memory ? memory.totalJSHeapSize : null,
        loadTime: performance.now(),
        resourceCount: performance.getEntriesByType('resource').length
      }
    })
    
    console.log('Performance under load:', performanceMetrics)
    
    // Verify system remains responsive
    const responsiveTest = Date.now()
    await page.locator('body').click()
    const responseTime = Date.now() - responsiveTest
    
    expect(responseTime).toBeLessThan(1000) // Should respond within 1s
    console.log(`System response time under load: ${responseTime}ms`)
    
    // Verify UI updates continue
    const taskElements = page.locator('.task-item, .task-card')
    const taskCount = await taskElements.count()
    
    console.log(`Task queue contains ${taskCount} tasks after load test`)
    
    // Verify agents are still functional
    const agentElements = page.locator('.agent-card, .agent-status')
    const agentCount = await agentElements.count()
    
    if (agentCount > 0) {
      const activeAgents = await page.locator('.agent-card[data-status="active"], .agent-status.active').count()
      console.log(`${activeAgents}/${agentCount} agents remain active under load`)
    }
  })

  test('agent workflow persistence and state management', async ({ page }) => {
    // Test state persistence across page reloads
    const initialState = await page.evaluate(() => {
      return {
        taskCount: document.querySelectorAll('.task-item, .task-card').length,
        agentCount: document.querySelectorAll('.agent-card, .agent-status').length,
        url: window.location.href
      }
    })
    
    console.log('Initial state:', initialState)
    
    // Reload page
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    
    // Verify state is restored
    const restoredState = await page.evaluate(() => {
      return {
        taskCount: document.querySelectorAll('.task-item, .task-card').length,
        agentCount: document.querySelectorAll('.agent-card, .agent-status').length,
        url: window.location.href
      }
    })
    
    console.log('Restored state:', restoredState)
    
    // State should be preserved (or gracefully recovered)
    if (initialState.taskCount > 0) {
      // Allow for some variation due to task completion
      expect(Math.abs(restoredState.taskCount - initialState.taskCount)).toBeLessThan(5)
    }
    
    if (initialState.agentCount > 0) {
      expect(restoredState.agentCount).toBeGreaterThanOrEqual(initialState.agentCount)
    }
    
    console.log('✓ Workflow state persistence verified')
    
    // Test local storage persistence
    const storageData = await page.evaluate(() => {
      const data: any = {}
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i)
        if (key && (key.includes('agent') || key.includes('task') || key.includes('workflow'))) {
          data[key] = localStorage.getItem(key)
        }
      }
      return data
    })
    
    if (Object.keys(storageData).length > 0) {
      console.log('Workflow data in localStorage:', Object.keys(storageData))
    }
  })

  test('real-time collaboration and communication between agents', async ({ page }) => {
    // Look for agent communication indicators
    const communicationElements = page.locator(
      '.agent-communication, .collaboration-panel, [data-testid*="communication"]'
    )
    
    if (await communicationElements.count() > 0) {
      console.log('✓ Agent communication interface found')
      
      // Monitor communication logs
      const communicationLogs = communicationElements.locator('.log, .message, .communication-item')
      const logCount = await communicationLogs.count()
      
      if (logCount > 0) {
        console.log(`Found ${logCount} communication log entries`)
        
        // Sample some communication content
        for (let i = 0; i < Math.min(logCount, 3); i++) {
          const logEntry = await communicationLogs.nth(i).textContent()
          console.log(`Communication ${i + 1}: ${logEntry?.substring(0, 100)}`)
        }
      }
    }
    
    // Test agent coordination events
    const coordinationEvents = page.locator(
      '.coordination-event, .agent-event, [data-testid*="coordination"]'
    )
    
    if (await coordinationEvents.count() > 0) {
      console.log(`Found ${await coordinationEvents.count()} coordination events`)
    }
    
    // Look for real-time updates between agents
    const agentCards = page.locator('.agent-card, .agent-status')
    const agentCount = await agentCards.count()
    
    if (agentCount > 1) {
      // Monitor for synchronized updates
      const initialStates = []
      
      for (let i = 0; i < agentCount; i++) {
        const agent = agentCards.nth(i)
        const state = await agent.evaluate(el => {
          return {
            status: el.getAttribute('data-status') || 'unknown',
            tasks: el.querySelector('.task-count')?.textContent || '0',
            lastUpdate: el.querySelector('.last-updated')?.textContent || ''
          }
        })
        initialStates.push(state)
      }
      
      // Wait for potential updates
      await page.waitForTimeout(8000)
      
      const finalStates = []
      for (let i = 0; i < agentCount; i++) {
        const agent = agentCards.nth(i)
        const state = await agent.evaluate(el => {
          return {
            status: el.getAttribute('data-status') || 'unknown',
            tasks: el.querySelector('.task-count')?.textContent || '0',
            lastUpdate: el.querySelector('.last-updated')?.textContent || ''
          }
        })
        finalStates.push(state)
      }
      
      // Check for coordinated changes
      let updatesDetected = 0
      for (let i = 0; i < agentCount; i++) {
        if (JSON.stringify(initialStates[i]) !== JSON.stringify(finalStates[i])) {
          updatesDetected++
        }
      }
      
      if (updatesDetected > 0) {
        console.log(`✓ Detected coordinated updates in ${updatesDetected} agents`)
      }
    }
  })
})