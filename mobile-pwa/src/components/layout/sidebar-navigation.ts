import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export interface NavigationItem {
  id: string
  path: string
  label: string
  icon: string
  badge?: string | number
  section?: 'main' | 'admin'
  children?: NavigationItem[]
}

@customElement('sidebar-navigation')
export class SidebarNavigation extends LitElement {
  @property() declare currentRoute: string
  @property({ type: Boolean }) declare collapsed: boolean
  @property({ type: Boolean }) declare mobile: boolean
  @state() declare private expandedItems: Set<string>

  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
      border-right: 1px solid rgba(148, 163, 184, 0.1);
      color: #f1f5f9;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      z-index: 30;
    }

    :host([collapsed]) {
      width: 72px;
    }

    :host(:not([collapsed])) {
      width: 280px;
    }

    .sidebar-container {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }

    /* Header Section */
    .sidebar-header {
      padding: 1.5rem 1rem;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
      display: flex;
      align-items: center;
      gap: 0.75rem;
      transition: all 0.3s ease;
    }

    .logo {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      border-radius: 0.75rem;
      font-size: 1.5rem;
      box-shadow: 0 8px 16px rgba(59, 130, 246, 0.2);
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }
    
    .logo::before {
      content: '';
      position: absolute;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
      transform: rotate(-45deg);
      transition: all 0.6s ease;
      opacity: 0;
    }
    
    .logo:hover::before {
      opacity: 1;
      animation: shimmer 0.6s ease;
    }
    
    .logo:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 24px rgba(59, 130, 246, 0.3);
    }
    
    @keyframes shimmer {
      0% { transform: translateX(-100%) translateY(-100%) rotate(-45deg); }
      100% { transform: translateX(100%) translateY(100%) rotate(-45deg); }
    }

    .brand-text {
      flex: 1;
      opacity: 1;
      transition: opacity 0.2s ease;
    }

    :host([collapsed]) .brand-text {
      opacity: 0;
      pointer-events: none;
    }

    .brand-title {
      font-size: 1.125rem;
      font-weight: 700;
      color: #f8fafc;
      margin: 0;
      line-height: 1.2;
    }

    .brand-subtitle {
      font-size: 0.75rem;
      color: #94a3b8;
      margin: 0;
      line-height: 1;
    }

    /* Toggle Button */
    .sidebar-toggle {
      position: absolute;
      top: 1.5rem;
      right: -12px;
      width: 24px;
      height: 24px;
      background: #3b82f6;
      border: 2px solid #1e293b;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
      z-index: 10;
    }

    .sidebar-toggle:hover {
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      transform: scale(1.15);
      box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
      border-color: rgba(59, 130, 246, 0.5);
    }

    .sidebar-toggle svg {
      width: 12px;
      height: 12px;
      color: white;
      transition: transform 0.3s ease;
    }

    :host([collapsed]) .sidebar-toggle svg {
      transform: rotate(180deg);
    }

    /* Navigation Section */
    .sidebar-nav {
      flex: 1;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 1rem 0;
      scrollbar-width: none;
      -ms-overflow-style: none;
    }

    .sidebar-nav::-webkit-scrollbar {
      display: none;
    }

    .nav-section {
      margin-bottom: 2rem;
    }

    .nav-section-title {
      padding: 0 1rem 0.5rem 1rem;
      font-size: 0.75rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      opacity: 1;
      transition: opacity 0.2s ease;
    }

    :host([collapsed]) .nav-section-title {
      opacity: 0;
      pointer-events: none;
    }

    .nav-items {
      list-style: none;
      margin: 0;
      padding: 0;
    }

    /* Navigation Items */
    .nav-item {
      margin: 0 0.75rem 0.25rem 0.75rem;
    }

    .nav-link {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.75rem;
      border-radius: 0.5rem;
      text-decoration: none;
      color: #cbd5e1;
      transition: all 0.2s ease;
      cursor: pointer;
      position: relative;
      font-size: 0.875rem;
      font-weight: 500;
      min-height: 44px;
    }

    .nav-link:hover {
      background: rgba(148, 163, 184, 0.1);
      color: #f1f5f9;
      transform: translateX(4px);
      border-left: 2px solid #3b82f6;
      box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .nav-link:hover .nav-icon {
      color: #60a5fa;
      transform: scale(1.1);
    }
    
    .nav-link:hover .nav-badge {
      transform: scale(1.1);
      box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }

    .nav-link.active {
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
      color: white;
      box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
    }

    .nav-link.active::before {
      content: '';
      position: absolute;
      left: -0.75rem;
      top: 50%;
      transform: translateY(-50%);
      width: 3px;
      height: 20px;
      background: #60a5fa;
      border-radius: 0 2px 2px 0;
    }

    .nav-icon {
      width: 20px;
      height: 20px;
      flex-shrink: 0;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      line-height: 1;
    }

    .nav-label {
      flex: 1;
      opacity: 1;
      transition: opacity 0.2s ease;
      white-space: nowrap;
      overflow: hidden;
    }

    :host([collapsed]) .nav-label {
      opacity: 0;
      pointer-events: none;
    }

    .nav-badge {
      background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
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
      opacity: 1;
      transition: all 0.2s ease;
      box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2);
      animation: badgePulse 2s infinite;
    }
    
    @keyframes badgePulse {
      0%, 100% {
        transform: scale(1);
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2);
      }
      50% {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(239, 68, 68, 0.3);
      }
    }

    :host([collapsed]) .nav-badge {
      opacity: 0;
      pointer-events: none;
    }

    /* Tooltip for collapsed state */
    .nav-tooltip {
      position: absolute;
      left: 100%;
      top: 50%;
      transform: translateY(-50%);
      margin-left: 0.75rem;
      background: #1f2937;
      color: #f9fafb;
      padding: 0.5rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      white-space: nowrap;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
      z-index: 50;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    :host([collapsed]) .nav-link:hover .nav-tooltip {
      opacity: 1;
    }

    /* System Status Section */
    .sidebar-footer {
      padding: 1rem;
      border-top: 1px solid rgba(148, 163, 184, 0.1);
      margin-top: auto;
    }

    .system-status {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.75rem;
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 41, 59, 0.4) 100%);
      border-radius: 0.5rem;
      border: 1px solid rgba(148, 163, 184, 0.2);
      backdrop-filter: blur(10px);
      transition: all 0.2s ease;
    }
    
    .system-status:hover {
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
      border-color: rgba(59, 130, 246, 0.3);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .status-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }

    .status-indicator.warning {
      background: #f59e0b;
    }

    .status-indicator.error {
      background: #ef4444;
    }

    .status-text {
      font-size: 0.75rem;
      color: #94a3b8;
      opacity: 1;
      transition: opacity 0.2s ease;
    }

    :host([collapsed]) .status-text {
      opacity: 0;
      pointer-events: none;
    }

    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }

    /* Mobile Adaptations */
    @media (max-width: 768px) {
      :host {
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        z-index: 40;
        transform: translateX(-100%);
        transition: transform 0.3s ease;
      }

      :host([mobile][open]) {
        transform: translateX(0);
      }

      .sidebar-toggle {
        display: none;
      }
    }

    /* High contrast mode */
    @media (prefers-contrast: high) {
      :host {
        border-right: 2px solid #475569;
      }

      .nav-link.active {
        outline: 2px solid #60a5fa;
      }
    }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
      * {
        transition: none !important;
        animation: none !important;
      }
    }

    /* Dark mode enhancements */
    @media (prefers-color-scheme: dark) {
      :host {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
      }

      .sidebar-header {
        border-bottom-color: rgba(71, 85, 105, 0.2);
      }

      .system-status {
        background: rgba(0, 0, 0, 0.3);
        border-color: rgba(71, 85, 105, 0.2);
      }
    }
  `

  private navigationItems: NavigationItem[] = [
    {
      id: 'main',
      path: '',
      label: 'Main',
      icon: '',
      section: 'main',
      children: [
        {
          id: 'dashboard',
          path: '/dashboard',
          label: 'Dashboard',
          icon: '📊'
        },
        {
          id: 'agents',
          path: '/agents',
          label: 'Agents',
          icon: '🤖',
          badge: '3'
        },
        {
          id: 'tasks',
          path: '/tasks',
          label: 'Tasks',
          icon: '✅'
        },
        {
          id: 'system-health',
          path: '/system-health',
          label: 'System Health',
          icon: '💚'
        }
      ]
    },
    {
      id: 'admin',
      path: '',
      label: 'Administration',
      icon: '',
      section: 'admin',
      children: [
        {
          id: 'settings',
          path: '/settings',
          label: 'Settings',
          icon: '⚙️'
        }
      ]
    }
  ]

  constructor() {
    super()
    
    // Initialize state properties
    this.currentRoute = '/'
    this.collapsed = false
    this.mobile = false
    this.expandedItems = new Set()
    
    // Auto-collapse on mobile by default
    if (window.innerWidth < 768) {
      this.mobile = true
    }

    // Listen for window resize
    window.addEventListener('resize', this.handleResize.bind(this))
  }

  private handleResize() {
    const wasMobile = this.mobile
    this.mobile = window.innerWidth < 768
    
    // If switching to desktop, ensure sidebar is not collapsed
    if (wasMobile && !this.mobile) {
      this.collapsed = false
    }
  }

  private handleToggle() {
    this.collapsed = !this.collapsed
    this.dispatchEvent(new CustomEvent('sidebar-toggle', {
      detail: { collapsed: this.collapsed },
      bubbles: true,
      composed: true
    }))
  }

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

  private renderNavigationItems(items: NavigationItem[], isSection: boolean = false): any {
    return items.map(item => {
      if (item.children) {
        return html`
          <div class="nav-section">
            ${!isSection ? html`
              <div class="nav-section-title">${item.label}</div>
            ` : ''}
            <ul class="nav-items">
              ${this.renderNavigationItems(item.children, true)}
            </ul>
          </div>
        `
      }

      return html`
        <li class="nav-item">
          <a
            class="nav-link ${this.isActive(item.path) ? 'active' : ''}"
            @click=${() => this.handleNavigation(item.path)}
            role="button"
            tabindex="0"
            aria-label=${item.label}
          >
            <span class="nav-icon">${item.icon ? item.icon : ''}</span>
            <span class="nav-label">${item.label}</span>
            ${item.badge ? html`
              <span class="nav-badge">${item.badge}</span>
            ` : ''}
            <div class="nav-tooltip">${item.label}</div>
          </a>
        </li>
      `
    })
  }

  render() {
    return html`
      <div class="sidebar-container">
        <!-- Header -->
        <div class="sidebar-header">
          <div class="logo">🤖</div>
          <div class="brand-text">
            <h1 class="brand-title">Agent Hive</h1>
            <p class="brand-subtitle">Autonomous Development</p>
          </div>
        </div>

        <!-- Toggle Button -->
        ${!this.mobile ? html`
          <button 
            class="sidebar-toggle" 
            @click=${this.handleToggle}
            aria-label="${this.collapsed ? 'Expand sidebar' : 'Collapse sidebar'}"
          >
${this.collapsed ? '▶' : '◀'}
          </button>
        ` : ''}

        <!-- Navigation -->
        <nav class="sidebar-nav">
          ${this.renderNavigationItems(this.navigationItems)}
        </nav>

        <!-- Footer -->
        <div class="sidebar-footer">
          <div class="system-status">
            <div class="status-indicator"></div>
            <div class="status-text">All systems operational</div>
          </div>
        </div>
      </div>
    `
  }
}