import { LitElement, html, css } from 'lit'
import { customElement, property } from 'lit/decorators.js'

@customElement('bottom-navigation')
export class BottomNavigation extends LitElement {
  @property() currentRoute: string = '/'
  
  static styles = css`
    :host {
      display: block;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border-top: 1px solid rgba(229, 231, 235, 0.8);
      position: sticky;
      bottom: 0;
      z-index: 40;
    }
    
    .nav {
      display: flex;
      justify-content: space-around;
      align-items: center;
      padding: 0.5rem 0;
      max-width: 100%;
    }
    
    .nav-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.25rem;
      padding: 0.5rem;
      border-radius: 0.5rem;
      transition: all 0.2s;
      cursor: pointer;
      text-decoration: none;
      color: #6b7280;
      min-width: 60px;
      min-height: 48px;
      justify-content: center;
    }
    
    .nav-item:hover {
      background: rgba(59, 130, 246, 0.1);
      color: #3b82f6;
    }
    
    .nav-item.active {
      color: #3b82f6;
      background: rgba(59, 130, 246, 0.1);
    }
    
    .nav-icon {
      width: 24px;
      height: 24px;
      stroke-width: 1.5;
    }
    
    .nav-label {
      font-size: 0.625rem;
      font-weight: 500;
      text-align: center;
      line-height: 1;
    }
    
    .badge {
      position: absolute;
      top: 0.25rem;
      right: 0.25rem;
      background: #ef4444;
      color: white;
      font-size: 0.625rem;
      font-weight: 600;
      padding: 0.125rem 0.375rem;
      border-radius: 0.75rem;
      min-width: 1rem;
      height: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .nav-item {
      position: relative;
    }
    
    @supports (padding: max(0px)) {
      .nav {
        padding-bottom: max(0.5rem, env(safe-area-inset-bottom));
      }
    }
  `
  
  private navItems = [
    {
      path: '/dashboard',
      label: 'Dashboard',
      icon: html`
        <svg class="nav-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
          <path stroke-linecap="round" stroke-linejoin="round" d="M8 5a2 2 0 012-2h4a2 2 0 012 2v0M8 5a2 2 0 000 4h8a2 2 0 000-4" />
        </svg>
      `
    },
    {
      path: '/tasks',
      label: 'Tasks',
      icon: html`
        <svg class="nav-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
        </svg>
      `
    },
    {
      path: '/agents',
      label: 'Agents',
      icon: html`
        <svg class="nav-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      `
    },
    {
      path: '/events',
      label: 'Events',
      icon: html`
        <svg class="nav-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m3 0H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2z" />
          <path stroke-linecap="round" stroke-linejoin="round" d="M7 8h10M7 12h10m-7 4h7" />
        </svg>
      `
    },
    {
      path: '/settings',
      label: 'Settings',
      icon: html`
        <svg class="nav-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      `
    }
  ]
  
  private handleNavigation(path: string) {
    this.dispatchEvent(new CustomEvent('navigate', {
      detail: { route: path },
      bubbles: true,
      composed: true
    }))
  }
  
  private isActive(path: string): boolean {
    if (path === '/dashboard') {
      return this.currentRoute === '/' || this.currentRoute === '/dashboard'
    }
    return this.currentRoute === path
  }
  
  render() {
    return html`
      <nav class="nav">
        ${this.navItems.map(item => html`
          <div
            class="nav-item ${this.isActive(item.path) ? 'active' : ''}"
            @click=${() => this.handleNavigation(item.path)}
            role="button"
            tabindex="0"
            aria-label=${item.label}
          >
            ${item.icon}
            <span class="nav-label">${item.label}</span>
            ${item.path === '/events' ? html`
              <!-- Example badge for notifications -->
              <!-- <span class="badge">3</span> -->
            ` : ''}
          </div>
        `)}
      </nav>
    `
  }
}