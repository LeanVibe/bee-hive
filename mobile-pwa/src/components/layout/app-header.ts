import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { getSystemHealthService } from '../../services'
import type { HealthSummary } from '../../services'
import './notification-center'

@customElement('app-header')
export class AppHeader extends LitElement {
  @property() declare currentRoute: string
  @property() declare isOnline: boolean
  @property({ type: Boolean }) declare showMenuButton: boolean
  @state() private declare systemHealthService: any
  @state() private declare healthSummary: HealthSummary | null
  @state() private declare activeAgents: number
  @state() private declare activeTasks: number
  
  constructor() {
    super()
    
    // Initialize reactive properties
    this.currentRoute = '/'
    this.isOnline = true
    this.showMenuButton = true
    this.systemHealthService = getSystemHealthService()
    this.healthSummary = null
    this.activeAgents = 0
    this.activeTasks = 0
  }
  
  static styles = css`
    :host {
      display: block;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(229, 231, 235, 0.8);
      position: sticky;
      top: 0;
      z-index: 40;
    }
    
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem;
      max-width: 100%;
    }
    
    .logo {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-weight: 600;
      color: #1f2937;
    }
    
    .logo-icon {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      border-radius: 0.5rem;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
    }
    
    .title {
      font-size: 1.125rem;
    }
    
    .status-indicators {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .status-indicator {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.75rem;
      padding: 0.25rem 0.5rem;
      border-radius: 0.375rem;
      font-weight: 500;
    }
    
    .status-online {
      background: #d1fae5;
      color: #065f46;
    }
    
    .status-offline {
      background: #fef2f2;
      color: #991b1b;
    }
    
    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
    }
    
    .status-healthy {
      background: #d1fae5;
      color: #065f46;
    }
    
    .status-degraded {
      background: #fef3c7;
      color: #92400e;
    }
    
    .status-unhealthy {
      background: #fef2f2;
      color: #991b1b;
    }
    
    .status-count {
      background: #f3f4f6;
      color: #374151;
      font-size: 0.6875rem;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-weight: 600;
    }

    .menu-button {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      background: none;
      border: none;
      border-radius: 0.5rem;
      cursor: pointer;
      color: #374151;
      transition: all 0.2s;
    }

    .menu-button:hover {
      background: #f3f4f6;
    }

    .menu-icon {
      width: 24px;
      height: 24px;
    }
    
    @media (max-width: 640px) {
      .header {
        padding: 0.75rem 1rem;
      }
      
      .title {
        display: none;
      }
      
      .status-indicator {
        font-size: 0.625rem;
        padding: 0.25rem;
      }
    }
  `
  
  private getPageTitle(): string {
    switch (this.currentRoute) {
      case '/':
      case '/dashboard':
        return 'Dashboard'
      case '/tasks':
        return 'Tasks'
      case '/agents':
        return 'Agents'
      case '/system-health':
        return 'System Health'
      case '/events':
        return 'Events'
      case '/settings':
        return 'Settings'
      default:
        return 'Agent Hive'
    }
  }

  async connectedCallback() {
    super.connectedCallback()
    await this.initializeHealthTracking()
  }
  
  /**
   * Initialize health tracking with periodic updates
   */
  private async initializeHealthTracking() {
    try {
      // Set up event listener for health changes
      this.systemHealthService.addEventListener('healthChanged', this.handleHealthChanged.bind(this))
      
      // Get initial health summary
      await this.updateHealthSummary()
      
      // Set up periodic updates
      setInterval(() => {
        this.updateHealthSummary()
      }, 10000) // Update every 10 seconds
      
    } catch (error) {
      console.error('Failed to initialize health tracking:', error)
    }
  }
  
  /**
   * Update health summary data
   */
  private async updateHealthSummary() {
    try {
      this.healthSummary = await this.systemHealthService.getHealthSummary()
    } catch (error) {
      console.error('Failed to get health summary:', error)
    }
  }
  
  /**
   * Handle real-time health changes
   */
  private handleHealthChanged = (event: CustomEvent) => {
    this.updateHealthStatus()
  }
  
  /**
   * Update status from parent data (called by parent components)
   */
  updateStatus(activeAgents: number, activeTasks: number) {
    this.activeAgents = activeAgents
    this.activeTasks = activeTasks
  }
  
  private async updateHealthStatus() {
    await this.updateHealthSummary()
  }

  private handleMenuClick() {
    this.dispatchEvent(new CustomEvent('menu-toggle', {
      bubbles: true,
      composed: true
    }))
  }
  
  render() {
    return html`
      <header class="header">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
          ${this.showMenuButton ? html`
            <button class="menu-button" @click="${this.handleMenuClick}" aria-label="Open menu">
              <svg class="menu-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
              </svg>
            </button>
          ` : ''}
          
          <div class="logo">
            <div class="logo-icon">🤖</div>
            <span class="title">${this.getPageTitle()}</span>
          </div>
        </div>
        
        <div class="status-indicators">
          <!-- System Health Status -->
          ${this.healthSummary ? html`
            <div class="status-indicator ${
              this.healthSummary.overall === 'healthy' ? 'status-healthy' : 
              this.healthSummary.overall === 'degraded' ? 'status-degraded' : 'status-unhealthy'
            }">
              <div class="status-dot"></div>
              ${this.healthSummary.overall === 'healthy' ? 'Healthy' : 
                this.healthSummary.overall === 'degraded' ? 'Degraded' : 'Unhealthy'}
            </div>
          ` : ''}
          
          <!-- Agent Count -->
          ${this.activeAgents > 0 ? html`
            <div class="status-count">
              🤖 ${this.activeAgents}
            </div>
          ` : ''}
          
          <!-- Task Count -->  
          ${this.activeTasks > 0 ? html`
            <div class="status-count">
              📋 ${this.activeTasks}
            </div>
          ` : ''}
          
          <!-- Notification Center -->
          <notification-center></notification-center>
          
          <!-- Connection Status -->
          <div class="status-indicator ${this.isOnline ? 'status-online' : 'status-offline'}">
            <div class="status-dot"></div>
            ${this.isOnline ? 'Online' : 'Offline'}
          </div>
        </div>
      </header>
    `
  }
}