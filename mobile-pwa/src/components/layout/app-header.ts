import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { getSystemHealthService } from '../../services'
import type { HealthSummary } from '../../services'
import { themeService } from '../../services/theme'
import './notification-center'
import './theme-toggle'

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
      background: var(--glass-bg, rgba(255, 255, 255, 0.95));
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--color-border, rgba(229, 231, 235, 0.8));
      position: sticky;
      top: 0;
      z-index: 40;
      transition: all var(--transition-normal, 0.3s);
      /* Safe area support for mobile */
      padding-top: env(safe-area-inset-top);
    }
    
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: var(--space-4, 1rem);
      max-width: 100%;
      min-height: 60px;
    }
    
    .logo {
      display: flex;
      align-items: center;
      gap: var(--space-3, 0.75rem);
      font-weight: 600;
      color: var(--color-text, #1f2937);
      transition: all var(--transition-normal, 0.3s);
    }
    
    .logo-icon {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, var(--color-primary, #1e40af), var(--color-primary-light, #3b82f6));
      border-radius: var(--radius-lg, 0.5rem);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 700;
      font-size: 0.875rem;
      letter-spacing: -0.02em;
      box-shadow: var(--shadow-sm, 0 1px 2px 0 rgba(0, 0, 0, 0.05));
      transition: all var(--transition-normal, 0.3s);
    }
    
    .logo-icon:hover {
      transform: scale(1.05);
      box-shadow: var(--shadow-md, 0 4px 6px -1px rgba(0, 0, 0, 0.1));
    }
    
    .title {
      font-size: var(--text-lg, 1.125rem);
      font-weight: 700;
    }
    
    .status-indicators {
      display: flex;
      align-items: center;
      gap: var(--space-2, 0.5rem);
      flex-wrap: wrap;
    }
    
    .status-indicator {
      display: flex;
      align-items: center;
      gap: var(--space-1, 0.25rem);
      font-size: var(--text-xs, 0.75rem);
      padding: var(--space-1, 0.25rem) var(--space-2, 0.5rem);
      border-radius: var(--radius-md, 0.375rem);
      font-weight: 600;
      transition: all var(--transition-normal, 0.3s);
      white-space: nowrap;
    }
    
    .status-online {
      background: var(--color-success-light, #d1fae5);
      color: var(--color-success-dark, #065f46);
      border: 1px solid var(--color-success, #10b981);
    }
    
    .status-offline {
      background: var(--color-error-light, #fef2f2);
      color: var(--color-error-dark, #991b1b);
      border: 1px solid var(--color-error, #ef4444);
      animation: pulse 2s infinite;
    }
    
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
      position: relative;
    }
    
    .status-dot::after {
      content: '';
      position: absolute;
      top: -2px;
      left: -2px;
      right: -2px;
      bottom: -2px;
      border: 1px solid currentColor;
      border-radius: 50%;
      opacity: 0;
      animation: ping 2s infinite;
    }
    
    .status-healthy {
      background: var(--color-success-light, #d1fae5);
      color: var(--color-success-dark, #065f46);
      border: 1px solid var(--color-success, #10b981);
    }
    
    .status-degraded {
      background: var(--color-warning-light, #fef3c7);
      color: var(--color-warning-dark, #92400e);
      border: 1px solid var(--color-warning, #f59e0b);
    }
    
    .status-unhealthy {
      background: var(--color-error-light, #fef2f2);
      color: var(--color-error-dark, #991b1b);
      border: 1px solid var(--color-error, #ef4444);
      animation: pulse 2s infinite;
    }
    
    .status-count {
      background: var(--color-surface-secondary, #f3f4f6);
      color: var(--color-text-secondary, #374151);
      font-size: var(--text-xs, 0.6875rem);
      padding: var(--space-1, 0.125rem) var(--space-2, 0.375rem);
      border-radius: var(--radius-md, 0.25rem);
      font-weight: 700;
      border: 1px solid var(--color-border, #e5e7eb);
      transition: all var(--transition-normal, 0.3s);
    }
    
    .status-count:hover {
      background: var(--color-primary-alpha, rgba(59, 130, 246, 0.1));
      border-color: var(--color-primary, #3b82f6);
      color: var(--color-primary, #3b82f6);
    }

    .menu-button {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 44px;
      height: 44px;
      background: var(--color-surface, transparent);
      border: 1px solid var(--color-border, transparent);
      border-radius: var(--radius-lg, 0.5rem);
      cursor: pointer;
      color: var(--color-text-secondary, #374151);
      transition: all var(--transition-normal, 0.3s);
      touch-action: manipulation;
    }

    .menu-button:hover {
      background: var(--color-surface-secondary, #f3f4f6);
      border-color: var(--color-border-focus, #3b82f6);
      transform: scale(1.05);
    }

    .menu-button:active {
      transform: scale(0.95);
    }

    .menu-icon {
      width: 20px;
      height: 20px;
      transition: transform var(--transition-normal, 0.3s);
    }

    .menu-button:hover .menu-icon {
      transform: rotate(90deg);
    }
    
    .controls-section {
      display: flex;
      align-items: center;
      gap: var(--space-2, 0.5rem);
    }
    
    /* Mobile optimizations */
    @media (max-width: 768px) {
      .header {
        padding: var(--space-3, 0.75rem) var(--space-4, 1rem);
        min-height: 56px;
      }
      
      .title {
        display: none;
      }
      
      .status-indicator {
        font-size: var(--text-xs, 0.625rem);
        padding: var(--space-1, 0.25rem);
      }
      
      .status-indicators {
        gap: var(--space-1, 0.25rem);
      }
      
      .menu-button {
        width: 48px;
        height: 48px;
      }
      
      .menu-icon {
        width: 22px;
        height: 22px;
      }
      
      .logo-icon {
        width: 36px;
        height: 36px;
        font-size: 1rem;
      }
      
      /* Hide some status indicators on small screens */
      .status-count {
        display: none;
      }
    }
    
    @media (max-width: 480px) {
      .status-indicators {
        flex-direction: column;
        align-items: flex-end;
        gap: var(--space-1, 0.25rem);
      }
      
      .status-indicator {
        padding: 2px var(--space-1, 0.25rem);
        font-size: 10px;
      }
    }
    
    /* Animations */
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.7;
      }
    }
    
    @keyframes ping {
      75%, 100% {
        opacity: 0;
        transform: scale(1.5);
      }
    }
    
    /* High contrast mode */
    @media (prefers-contrast: high) {
      .status-indicator {
        border-width: 2px;
      }
      
      .menu-button {
        border-width: 2px;
        border-color: var(--color-text, #374151);
      }
      
      .logo-icon {
        border: 2px solid var(--color-text-inverse, white);
      }
    }
    
    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
      .logo-icon,
      .menu-button,
      .menu-icon,
      .status-indicator,
      .status-dot::after {
        transition: none;
        animation: none;
      }
      
      .menu-button:hover .menu-icon {
        transform: none;
      }
    }
    
    /* Dark theme adjustments */
    .theme-dark .status-online {
      background: var(--color-success-light, #064e3b);
      color: var(--color-success-dark, #34d399);
    }
    
    .theme-dark .status-offline {
      background: var(--color-error-light, #451a1a);
      color: var(--color-error-dark, #f87171);
    }
    
    .theme-dark .status-healthy {
      background: var(--color-success-light, #064e3b);
      color: var(--color-success-dark, #34d399);
    }
    
    .theme-dark .status-degraded {
      background: var(--color-warning-light, #451a03);
      color: var(--color-warning-dark, #fbbf24);
    }
    
    .theme-dark .status-unhealthy {
      background: var(--color-error-light, #451a1a);
      color: var(--color-error-dark, #f87171);
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
        return 'HiveOps'
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
            <div class="logo-icon" aria-label="HiveOps brand">HO</div>
            <span class="title">${this.getPageTitle()}</span>
          </div>
        </div>
        
        <div class="status-indicators">
          <!-- System Health Status -->
          ${this.healthSummary ? html`
            <div class="status-indicator ${
              this.healthSummary.overall === 'healthy' ? 'status-healthy' : 
              this.healthSummary.overall === 'degraded' ? 'status-degraded' : 'status-unhealthy'
            }" title="System health status">
              <div class="status-dot"></div>
              ${this.healthSummary.overall === 'healthy' ? 'Healthy' : 
                this.healthSummary.overall === 'degraded' ? 'Degraded' : 'Unhealthy'}
            </div>
          ` : ''}
          
          <!-- Agent Count -->
          ${this.activeAgents > 0 ? html`
            <div class="status-count" title="${this.activeAgents} active agents">
              🤖 ${this.activeAgents}
            </div>
          ` : ''}
          
          <!-- Task Count -->  
          ${this.activeTasks > 0 ? html`
            <div class="status-count" title="${this.activeTasks} active tasks">
              📋 ${this.activeTasks}
            </div>
          ` : ''}
          
          <!-- Theme Toggle -->
          <div class="controls-section">
            <theme-toggle compact></theme-toggle>
          </div>
          
          <!-- Notification Center -->
          <notification-center></notification-center>
          
          <!-- Connection Status -->
          <div class="status-indicator ${this.isOnline ? 'status-online' : 'status-offline'}" title="Connection status">
            <div class="status-dot"></div>
            ${this.isOnline ? 'Live' : 'Offline'}
          </div>
        </div>
      </header>
    `
  }
}