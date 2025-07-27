import { LitElement, html, css } from 'lit'
import { customElement, property } from 'lit/decorators.js'

@customElement('app-header')
export class AppHeader extends LitElement {
  @property() currentRoute: string = '/'
  @property() isOnline: boolean = true
  
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
      case '/events':
        return 'Events'
      case '/settings':
        return 'Settings'
      default:
        return 'Agent Hive'
    }
  }
  
  render() {
    return html`
      <header class="header">
        <div class="logo">
          <div class="logo-icon">🤖</div>
          <span class="title">${this.getPageTitle()}</span>
        </div>
        
        <div class="status-indicators">
          <div class="status-indicator ${this.isOnline ? 'status-online' : 'status-offline'}">
            <div class="status-dot"></div>
            ${this.isOnline ? 'Online' : 'Offline'}
          </div>
        </div>
      </header>
    `
  }
}