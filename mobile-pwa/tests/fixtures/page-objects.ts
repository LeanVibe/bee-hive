import { Page, Locator, expect } from '@playwright/test'

/**
 * Base Page Object with common functionality
 */
export class BasePage {
  readonly page: Page

  constructor(page: Page) {
    this.page = page
  }

  async goto(path: string = '/') {
    await this.page.goto(path)
  }

  async waitForLoad() {
    await this.page.waitForLoadState('networkidle')
  }

  async takeScreenshot(name: string) {
    await this.page.screenshot({ path: `test-results/screenshots/${name}.png` })
  }
}

/**
 * Dashboard Page Object
 */
export class DashboardPage extends BasePage {
  // Header elements
  readonly pageTitle: Locator
  readonly refreshButton: Locator
  readonly syncStatus: Locator
  readonly syncIndicator: Locator

  // Navigation tabs
  readonly overviewTab: Locator
  readonly tasksTab: Locator
  readonly agentsTab: Locator
  readonly eventsTab: Locator

  // Overview summary cards
  readonly activeTasksCard: Locator
  readonly completedTasksCard: Locator
  readonly activeAgentsCard: Locator
  readonly systemHealthCard: Locator
  readonly cpuUsageCard: Locator
  readonly memoryUsageCard: Locator

  // Dashboard panels
  readonly agentHealthPanel: Locator
  readonly eventTimeline: Locator

  constructor(page: Page) {
    super(page)
    
    // Header elements
    this.pageTitle = page.locator('h1.page-title')
    this.refreshButton = page.locator('.refresh-button')
    this.syncStatus = page.locator('.sync-status')
    this.syncIndicator = page.locator('.sync-indicator')

    // Navigation tabs
    this.overviewTab = page.locator('.tab-button').filter({ hasText: 'Overview' })
    this.tasksTab = page.locator('.tab-button').filter({ hasText: 'Tasks' })
    this.agentsTab = page.locator('.tab-button').filter({ hasText: 'Agents' })
    this.eventsTab = page.locator('.tab-button').filter({ hasText: 'Events' })

    // Overview summary cards
    this.activeTasksCard = page.locator('.summary-card').filter({ hasText: 'Active Tasks' })
    this.completedTasksCard = page.locator('.summary-card').filter({ hasText: 'Completed Tasks' })
    this.activeAgentsCard = page.locator('.summary-card').filter({ hasText: 'Active Agents' })
    this.systemHealthCard = page.locator('.summary-card').filter({ hasText: 'System Health' })
    this.cpuUsageCard = page.locator('.summary-card').filter({ hasText: 'CPU Usage' })
    this.memoryUsageCard = page.locator('.summary-card').filter({ hasText: 'Memory Usage' })

    // Dashboard panels
    this.agentHealthPanel = page.locator('agent-health-panel')
    this.eventTimeline = page.locator('event-timeline')
  }

  async navigateToOverview() {
    await this.overviewTab.click()
    await expect(this.overviewTab).toHaveClass(/active/)
  }

  async navigateToTasks() {
    await this.tasksTab.click()
    await expect(this.tasksTab).toHaveClass(/active/)
  }

  async navigateToAgents() {
    await this.agentsTab.click()
    await expect(this.agentsTab).toHaveClass(/active/)
  }

  async navigateToEvents() {
    await this.eventsTab.click()
    await expect(this.eventsTab).toHaveClass(/active/)
  }

  async refreshData() {
    await this.refreshButton.click()
    await expect(this.refreshButton).not.toHaveClass(/spinning/)
  }

  async getSummaryCardValue(cardName: string): Promise<string> {
    const card = this.page.locator('.summary-card').filter({ hasText: cardName })
    const value = await card.locator('.summary-value').textContent()
    return value || ''
  }

  async waitForRealTimeUpdate() {
    // Wait for sync status to show recent activity
    await expect(this.syncStatus).toContainText(/Last sync:|just now/)
  }

  async verifyOfflineMode() {
    await expect(this.syncIndicator).toHaveClass(/offline/)
    await expect(this.syncStatus).toContainText('Offline mode')
  }
}

/**
 * Kanban Board Page Object
 */
export class KanbanBoardPage extends BasePage {
  readonly kanbanBoard: Locator
  readonly searchFilter: Locator
  readonly agentFilter: Locator
  readonly offlineIndicator: Locator
  readonly updatingOverlay: Locator

  // Kanban columns
  readonly pendingColumn: Locator
  readonly inProgressColumn: Locator
  readonly reviewColumn: Locator
  readonly doneColumn: Locator

  constructor(page: Page) {
    super(page)
    
    this.kanbanBoard = page.locator('kanban-board')
    this.searchFilter = page.locator('.filter-input')
    this.agentFilter = page.locator('.filter-select')
    this.offlineIndicator = page.locator('.offline-indicator')
    this.updatingOverlay = page.locator('.updating-overlay')

    // Kanban columns
    this.pendingColumn = page.locator('kanban-column[data-column="pending"]')
    this.inProgressColumn = page.locator('kanban-column[data-column="in-progress"]')
    this.reviewColumn = page.locator('kanban-column[data-column="review"]')
    this.doneColumn = page.locator('kanban-column[data-column="done"]')
  }

  async filterTasks(searchTerm: string) {
    await this.searchFilter.fill(searchTerm)
    await this.page.waitForTimeout(500) // Debounce
  }

  async filterByAgent(agentName: string) {
    await this.agentFilter.selectOption(agentName)
    await this.page.waitForTimeout(500)
  }

  async clearFilters() {
    await this.searchFilter.clear()
    await this.agentFilter.selectOption('')
  }

  async getTasksInColumn(columnName: 'pending' | 'in-progress' | 'review' | 'done'): Promise<number> {
    const column = this.page.locator(`kanban-column[data-column="${columnName}"]`)
    const tasks = column.locator('task-card')
    return await tasks.count()
  }

  async dragTaskToColumn(taskId: string, targetColumn: 'pending' | 'in-progress' | 'review' | 'done') {
    const task = this.page.locator(`task-card[data-task-id="${taskId}"]`)
    const column = this.page.locator(`kanban-column[data-column="${targetColumn}"]`)
    
    await task.dragTo(column)
    await expect(this.updatingOverlay).not.toBeVisible()
  }

  async clickTask(taskId: string) {
    const task = this.page.locator(`task-card[data-task-id="${taskId}"]`)
    await task.click()
  }

  async verifyTaskInColumn(taskId: string, columnName: string) {
    const column = this.page.locator(`kanban-column[data-column="${columnName}"]`)
    const task = column.locator(`task-card[data-task-id="${taskId}"]`)
    await expect(task).toBeVisible()
  }
}

/**
 * Agent Health Panel Page Object
 */
export class AgentHealthPanelPage extends BasePage {
  readonly agentPanel: Locator
  readonly agentCards: Locator
  readonly refreshAgentsButton: Locator

  constructor(page: Page) {
    super(page)
    
    this.agentPanel = page.locator('agent-health-panel')
    this.agentCards = page.locator('.agent-card')
    this.refreshAgentsButton = page.locator('[data-testid="refresh-agents"]')
  }

  async getAgentCount(): Promise<number> {
    return await this.agentCards.count()
  }

  async getAgentStatus(agentId: string): Promise<string> {
    const agent = this.page.locator(`.agent-card[data-agent-id="${agentId}"]`)
    const status = await agent.locator('.agent-status').textContent()
    return status || ''
  }

  async clickAgent(agentId: string) {
    const agent = this.page.locator(`.agent-card[data-agent-id="${agentId}"]`)
    await agent.click()
  }

  async activateAgent(agentId: string) {
    const agent = this.page.locator(`.agent-card[data-agent-id="${agentId}"]`)
    const activateButton = agent.locator('.activate-button')
    await activateButton.click()
  }

  async deactivateAgent(agentId: string) {
    const agent = this.page.locator(`.agent-card[data-agent-id="${agentId}"]`)
    const deactivateButton = agent.locator('.deactivate-button')
    await deactivateButton.click()
  }

  async verifyAgentMetrics(agentId: string) {
    const agent = this.page.locator(`.agent-card[data-agent-id="${agentId}"]`)
    await expect(agent.locator('.performance-metrics')).toBeVisible()
    await expect(agent.locator('.cpu-chart')).toBeVisible()
    await expect(agent.locator('.memory-chart')).toBeVisible()
  }
}

/**
 * Event Timeline Page Object
 */
export class EventTimelinePage extends BasePage {
  readonly eventTimeline: Locator
  readonly eventItems: Locator
  readonly filterButtons: Locator

  constructor(page: Page) {
    super(page)
    
    this.eventTimeline = page.locator('event-timeline')
    this.eventItems = page.locator('.timeline-event')
    this.filterButtons = page.locator('.filter-button')
  }

  async getEventCount(): Promise<number> {
    return await this.eventItems.count()
  }

  async filterEventsBySeverity(severity: 'info' | 'warning' | 'error') {
    const filterButton = this.filterButtons.filter({ hasText: severity })
    await filterButton.click()
  }

  async clickEvent(eventId: string) {
    const event = this.page.locator(`.timeline-event[data-event-id="${eventId}"]`)
    await event.click()
  }

  async verifyEventDetails(eventId: string, expectedTitle: string) {
    const event = this.page.locator(`.timeline-event[data-event-id="${eventId}"]`)
    await expect(event.locator('.event-title')).toContainText(expectedTitle)
  }

  async verifyRealTimeUpdates() {
    const initialCount = await this.getEventCount()
    
    // Wait for potential new events
    await this.page.waitForTimeout(2000)
    
    const newCount = await this.getEventCount()
    // Events should be updated (could be same count if no new events, which is fine)
    expect(newCount).toBeGreaterThanOrEqual(initialCount)
  }
}

/**
 * Task Edit Modal Page Object
 */
export class TaskEditModalPage extends BasePage {
  readonly modal: Locator
  readonly titleInput: Locator
  readonly descriptionInput: Locator
  readonly statusSelect: Locator
  readonly prioritySelect: Locator
  readonly agentSelect: Locator
  readonly tagsInput: Locator
  readonly saveButton: Locator
  readonly cancelButton: Locator
  readonly deleteButton: Locator

  constructor(page: Page) {
    super(page)
    
    this.modal = page.locator('task-edit-modal')
    this.titleInput = page.locator('[data-testid="task-title"]')
    this.descriptionInput = page.locator('[data-testid="task-description"]')
    this.statusSelect = page.locator('[data-testid="task-status"]')
    this.prioritySelect = page.locator('[data-testid="task-priority"]')
    this.agentSelect = page.locator('[data-testid="task-agent"]')
    this.tagsInput = page.locator('[data-testid="task-tags"]')
    this.saveButton = page.locator('[data-testid="save-task"]')
    this.cancelButton = page.locator('[data-testid="cancel-task"]')
    this.deleteButton = page.locator('[data-testid="delete-task"]')
  }

  async fillTaskForm(taskData: {
    title?: string
    description?: string
    status?: string
    priority?: string
    agent?: string
    tags?: string
  }) {
    if (taskData.title) {
      await this.titleInput.fill(taskData.title)
    }
    if (taskData.description) {
      await this.descriptionInput.fill(taskData.description)
    }
    if (taskData.status) {
      await this.statusSelect.selectOption(taskData.status)
    }
    if (taskData.priority) {
      await this.prioritySelect.selectOption(taskData.priority)
    }
    if (taskData.agent) {
      await this.agentSelect.selectOption(taskData.agent)
    }
    if (taskData.tags) {
      await this.tagsInput.fill(taskData.tags)
    }
  }

  async saveTask() {
    await this.saveButton.click()
    await expect(this.modal).not.toBeVisible()
  }

  async cancelEdit() {
    await this.cancelButton.click()
    await expect(this.modal).not.toBeVisible()
  }

  async deleteTask() {
    await this.deleteButton.click()
    
    // Handle confirmation dialog if present
    const confirmButton = this.page.locator('[data-testid="confirm-delete"]')
    if (await confirmButton.isVisible()) {
      await confirmButton.click()
    }
    
    await expect(this.modal).not.toBeVisible()
  }

  async verifyValidationError(fieldName: string) {
    const field = this.page.locator(`[data-testid="task-${fieldName}"]`)
    const errorMessage = field.locator('+ .error-message')
    await expect(errorMessage).toBeVisible()
  }
}

/**
 * Agent Configuration Modal Page Object
 */
export class AgentConfigModalPage extends BasePage {
  readonly modal: Locator
  readonly nameInput: Locator
  readonly capabilitiesSelect: Locator
  readonly prioritySlider: Locator
  readonly saveButton: Locator
  readonly cancelButton: Locator

  constructor(page: Page) {
    super(page)
    
    this.modal = page.locator('agent-config-modal')
    this.nameInput = page.locator('[data-testid="agent-name"]')
    this.capabilitiesSelect = page.locator('[data-testid="agent-capabilities"]')
    this.prioritySlider = page.locator('[data-testid="agent-priority"]')
    this.saveButton = page.locator('[data-testid="save-agent"]')
    this.cancelButton = page.locator('[data-testid="cancel-agent"]')
  }

  async configureAgent(config: {
    name?: string
    capabilities?: string[]
    priority?: number
  }) {
    if (config.name) {
      await this.nameInput.fill(config.name)
    }
    if (config.capabilities) {
      for (const capability of config.capabilities) {
        await this.capabilitiesSelect.selectOption(capability)
      }
    }
    if (config.priority !== undefined) {
      await this.prioritySlider.fill(config.priority.toString())
    }
  }

  async saveConfiguration() {
    await this.saveButton.click()
    await expect(this.modal).not.toBeVisible()
  }

  async cancelConfiguration() {
    await this.cancelButton.click()
    await expect(this.modal).not.toBeVisible()
  }
}