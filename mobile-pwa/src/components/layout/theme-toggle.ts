import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { themeService, type ThemeMode, getThemeCSS } from '../../services/theme'

@customElement('theme-toggle')
export class ThemeToggle extends LitElement {
  @state() private currentTheme: ThemeMode = 'auto'
  @state() private isDark: boolean = false
  @state() private showOptions: boolean = false
  
  static styles = css`
    :host {
      display: inline-block;
      position: relative;
    }
    
    .toggle-button {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 44px;
      height: 44px;
      border: none;
      border-radius: 50%;
      background: transparent;
      color: var(--color-text-secondary, #64748b);
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
      touch-action: manipulation;
      -webkit-tap-highlight-color: transparent;
    }
    
    .toggle-button:hover {
      background: var(--color-surface-secondary, #f1f5f9);
      color: var(--color-text, #0f172a);
      transform: scale(1.05);
    }
    
    .toggle-button:active {
      transform: scale(0.95);
    }
    
    .toggle-button:focus {
      outline: none;
      box-shadow: 0 0 0 2px var(--color-primary, #1e40af), 0 0 0 4px rgba(30, 64, 175, 0.2);
    }
    
    .toggle-icon {
      width: 20px;
      height: 20px;
      transition: all 0.3s ease;
    }
    
    .toggle-icon.dark {
      transform: rotate(180deg);
    }
    
    .options-menu {
      position: absolute;
      top: 100%;
      right: 0;
      margin-top: 0.5rem;
      background: var(--glass-bg, rgba(255, 255, 255, 0.8));
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid var(--glass-border, rgba(255, 255, 255, 0.2));
      border-radius: 0.75rem;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      min-width: 160px;
      z-index: 50;
      opacity: 0;
      transform: translateY(-10px) scale(0.95);
      transition: all 0.2s ease;
      pointer-events: none;
      padding: 0.5rem;
    }
    
    .options-menu.show {
      opacity: 1;
      transform: translateY(0) scale(1);
      pointer-events: auto;
    }
    
    .option-item {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.75rem;
      border: none;
      background: transparent;
      color: var(--color-text, #0f172a);
      font-size: 0.875rem;
      font-weight: 500;
      width: 100%;
      text-align: left;
      border-radius: 0.5rem;
      cursor: pointer;
      transition: all 0.2s ease;
      touch-action: manipulation;
      min-height: 44px;
    }
    
    .option-item:hover {
      background: var(--color-surface-secondary, #f1f5f9);
    }
    
    .option-item:active {
      transform: scale(0.98);
    }
    
    .option-item.active {
      background: var(--color-primary, #1e40af);
      color: white;
    }
    
    .option-icon {
      width: 16px;
      height: 16px;
      flex-shrink: 0;
    }
    
    .theme-indicator {
      position: absolute;
      top: -2px;
      right: -2px;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--color-success, #10b981);
      border: 2px solid var(--color-background, #ffffff);
      opacity: 0;
      transition: opacity 0.2s ease;
    }
    
    .theme-indicator.show {
      opacity: 1;
    }
    
    /* Mobile optimizations */
    @media (max-width: 640px) {
      .options-menu {
        right: 0;
        left: auto;
        min-width: 180px;
      }
      
      .option-item {
        padding: 1rem 0.75rem;
        font-size: 1rem;
        min-height: 48px;
      }
      
      .toggle-button {
        width: 48px;
        height: 48px;
      }
      
      .toggle-icon {
        width: 22px;
        height: 22px;
      }
    }
    
    /* High contrast mode */
    @media (prefers-contrast: high) {
      .toggle-button {
        border: 2px solid currentColor;
      }
      
      .options-menu {
        border-width: 2px;
        background: var(--color-background);
        backdrop-filter: none;
      }
      
      .option-item.active {
        outline: 2px solid var(--color-text);
        outline-offset: 2px;
      }
    }
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
      .toggle-button,
      .toggle-icon,
      .options-menu,
      .option-item {
        transition: none;
      }
      
      .toggle-icon.dark {
        transform: none;
      }
    }
    
    /* Glass effect in dark mode */
    :host(.theme-dark) .options-menu {
      background: var(--glass-bg, rgba(15, 23, 42, 0.8));
      border-color: var(--glass-border, rgba(255, 255, 255, 0.1));
    }
  `
  
  connectedCallback() {
    super.connectedCallback()
    
    this.currentTheme = themeService.getTheme()
    this.isDark = themeService.isDarkMode()
    
    // Listen for theme changes
    themeService.on('theme-changed', this.handleThemeChange.bind(this))
    themeService.on('system-theme-changed', this.handleSystemThemeChange.bind(this))
    
    // Close menu when clicking outside
    document.addEventListener('click', this.handleDocumentClick.bind(this))
    
    // Update host class for styling
    this.updateHostClass()
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    
    themeService.off('theme-changed', this.handleThemeChange.bind(this))
    themeService.off('system-theme-changed', this.handleSystemThemeChange.bind(this))
    document.removeEventListener('click', this.handleDocumentClick.bind(this))
  }
  
  private handleThemeChange(event: any) {
    this.currentTheme = event.mode
    this.isDark = event.isDark
    this.updateHostClass()
  }
  
  private handleSystemThemeChange() {
    if (this.currentTheme === 'auto') {
      this.isDark = themeService.isDarkMode()
      this.updateHostClass()
    }
  }
  
  private updateHostClass() {
    this.classList.toggle('theme-dark', this.isDark)
    this.classList.toggle('theme-light', !this.isDark)
  }
  
  private handleDocumentClick(event: Event) {
    if (!this.contains(event.target as Node)) {
      this.showOptions = false
    }
  }
  
  private toggleOptions(event: Event) {
    event.stopPropagation()
    this.showOptions = !this.showOptions
  }
  
  private selectTheme(theme: ThemeMode, event: Event) {
    event.stopPropagation()
    themeService.setTheme(theme)
    this.showOptions = false
    
    // Provide haptic feedback on mobile
    if ('vibrate' in navigator) {
      navigator.vibrate(50)
    }
    
    // Announce change for screen readers
    this.announceThemeChange(theme)
  }
  
  private announceThemeChange(theme: ThemeMode) {
    const announcement = `Theme changed to ${theme === 'auto' ? 'system preference' : theme} mode`
    
    const announcer = document.createElement('div')
    announcer.setAttribute('aria-live', 'polite')
    announcer.setAttribute('aria-atomic', 'true')
    announcer.className = 'sr-only'
    announcer.textContent = announcement
    
    document.body.appendChild(announcer)
    setTimeout(() => document.body.removeChild(announcer), 1000)
  }
  
  private quickToggle() {
    // Quick toggle between light and dark, preserving auto if active
    if (this.currentTheme === 'auto') {
      // Switch to opposite of current effective theme
      const newTheme = this.isDark ? 'light' : 'dark'
      themeService.setTheme(newTheme)
    } else {
      // Toggle between current and opposite
      const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark'
      themeService.setTheme(newTheme)
    }
    
    // Provide haptic feedback
    if ('vibrate' in navigator) {
      navigator.vibrate(50)
    }
  }
  
  private getThemeIcon(theme: ThemeMode): string {
    switch (theme) {
      case 'light':
        return `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="5"/>
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
          </svg>
        `
      case 'dark':
        return `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
          </svg>
        `
      case 'auto':
        return `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
            <line x1="8" y1="21" x2="16" y2="21"/>
            <line x1="12" y1="17" x2="12" y2="21"/>
          </svg>
        `
      default:
        return ''
    }
  }
  
  private getCurrentIcon(): string {
    return this.getThemeIcon(this.currentTheme === 'auto' ? (this.isDark ? 'dark' : 'light') : this.currentTheme)
  }
  
  render() {
    return html`
      <button 
        class="toggle-button"
        @click=${this.toggleOptions}
        @dblclick=${this.quickToggle}
        aria-label="Theme settings - ${this.currentTheme} mode"
        aria-expanded=${this.showOptions}
        aria-haspopup="menu"
        title="Change theme (double-tap to toggle)"
      >
        <div class="toggle-icon ${this.isDark ? 'dark' : ''}">${this.getCurrentIcon()}</div>
        <div class="theme-indicator ${this.currentTheme !== 'auto' ? 'show' : ''}"></div>
      </button>
      
      <div class="options-menu ${this.showOptions ? 'show' : ''}" role="menu">
        <button 
          class="option-item ${this.currentTheme === 'light' ? 'active' : ''}"
          @click=${(e: Event) => this.selectTheme('light', e)}
          role="menuitem"
          aria-label="Light theme"
        >
          <div class="option-icon">${this.getThemeIcon('light')}</div>
          Light
        </button>
        
        <button 
          class="option-item ${this.currentTheme === 'dark' ? 'active' : ''}"
          @click=${(e: Event) => this.selectTheme('dark', e)}
          role="menuitem"
          aria-label="Dark theme"
        >
          <div class="option-icon">${this.getThemeIcon('dark')}</div>
          Dark
        </button>
        
        <button 
          class="option-item ${this.currentTheme === 'auto' ? 'active' : ''}"
          @click=${(e: Event) => this.selectTheme('auto', e)}
          role="menuitem"
          aria-label="Auto theme (follows system preference)"
        >
          <div class="option-icon">${this.getThemeIcon('auto')}</div>
          Auto
        </button>
      </div>
    `
  }
}