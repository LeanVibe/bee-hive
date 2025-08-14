import { EventEmitter } from '../utils/event-emitter'

export type ThemeMode = 'light' | 'dark' | 'auto'

export interface ThemeConfig {
  mode: ThemeMode
  accentColor?: string
  reducedMotion?: boolean
  highContrast?: boolean
}

export interface ThemeColors {
  primary: string
  primaryDark: string
  primaryLight: string
  secondary: string
  background: string
  backgroundSecondary: string
  surface: string
  surfaceSecondary: string
  text: string
  textSecondary: string
  textMuted: string
  border: string
  borderLight: string
  error: string
  warning: string
  success: string
  info: string
}

export class ThemeService extends EventEmitter {
  private static instance: ThemeService
  private currentTheme: ThemeMode = 'auto'
  private systemPrefersDark: boolean = false
  private mediaQuery: MediaQuery | null = null
  private storageKey = 'hiveops-theme-config'
  
  static getInstance(): ThemeService {
    if (!ThemeService.instance) {
      ThemeService.instance = new ThemeService()
    }
    return ThemeService.instance
  }
  
  constructor() {
    super()
    this.init()
  }
  
  private init(): void {
    // Set up media query listener for system theme changes
    this.mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    this.systemPrefersDark = this.mediaQuery.matches
    
    // Listen for system theme changes
    this.mediaQuery.addEventListener('change', (e) => {
      this.systemPrefersDark = e.matches
      if (this.currentTheme === 'auto') {
        this.applyTheme()
      }
      this.emit('system-theme-changed', { prefersDark: this.systemPrefersDark })
    })
    
    // Listen for reduced motion preference changes
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    reducedMotionQuery.addEventListener('change', () => {
      this.applyAccessibilityPreferences()
      this.emit('accessibility-changed', { reducedMotion: reducedMotionQuery.matches })
    })
    
    // Listen for high contrast preference changes  
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)')
    highContrastQuery.addEventListener('change', () => {
      this.applyAccessibilityPreferences()
      this.emit('accessibility-changed', { highContrast: highContrastQuery.matches })
    })
    
    // Load saved theme configuration
    this.loadThemeConfig()
    
    // Apply initial theme
    this.applyTheme()
    this.applyAccessibilityPreferences()
  }
  
  private loadThemeConfig(): void {
    try {
      const saved = localStorage.getItem(this.storageKey)
      if (saved) {
        const config: ThemeConfig = JSON.parse(saved)
        this.currentTheme = config.mode || 'auto'
      }
    } catch (error) {
      console.warn('Failed to load theme config:', error)
      this.currentTheme = 'auto'
    }
  }
  
  private saveThemeConfig(): void {
    try {
      const config: ThemeConfig = {
        mode: this.currentTheme,
        reducedMotion: this.getReducedMotionPreference(),
        highContrast: this.getHighContrastPreference()
      }
      localStorage.setItem(this.storageKey, JSON.stringify(config))
    } catch (error) {
      console.warn('Failed to save theme config:', error)
    }
  }
  
  setTheme(mode: ThemeMode): void {
    this.currentTheme = mode
    this.saveThemeConfig()
    this.applyTheme()
    this.emit('theme-changed', { 
      mode, 
      isDark: this.isDarkMode(),
      effectiveTheme: this.getEffectiveTheme()
    })
  }
  
  getTheme(): ThemeMode {
    return this.currentTheme
  }
  
  getEffectiveTheme(): 'light' | 'dark' {
    if (this.currentTheme === 'auto') {
      return this.systemPrefersDark ? 'dark' : 'light'
    }
    return this.currentTheme as 'light' | 'dark'
  }
  
  isDarkMode(): boolean {
    return this.getEffectiveTheme() === 'dark'
  }
  
  toggleTheme(): void {
    const current = this.getEffectiveTheme()
    this.setTheme(current === 'dark' ? 'light' : 'dark')
  }
  
  private applyTheme(): void {
    const isDark = this.isDarkMode()
    const root = document.documentElement
    
    // Apply theme class to root element
    root.classList.remove('theme-light', 'theme-dark')
    root.classList.add(isDark ? 'theme-dark' : 'theme-light')
    
    // Update meta theme-color for mobile browsers
    this.updateThemeColor(isDark)
    
    // Apply CSS custom properties for theme colors
    this.applyCSSVariables(isDark)
    
    // Update manifest theme color dynamically
    this.updateManifestThemeColor(isDark)
  }
  
  private updateThemeColor(isDark: boolean): void {
    const themeColorMeta = document.querySelector('meta[name="theme-color"]')
    const backgroundColor = isDark ? '#0f172a' : '#1e40af'
    
    if (themeColorMeta) {
      themeColorMeta.setAttribute('content', backgroundColor)
    } else {
      const meta = document.createElement('meta')
      meta.name = 'theme-color'
      meta.content = backgroundColor
      document.head.appendChild(meta)
    }
  }
  
  private applyCSSVariables(isDark: boolean): void {
    const root = document.documentElement
    const colors = this.getThemeColors(isDark)
    
    // Apply color variables
    Object.entries(colors).forEach(([key, value]) => {
      const cssVar = `--color-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`
      root.style.setProperty(cssVar, value)
    })
    
    // Apply additional mobile-specific theme variables
    root.style.setProperty('--safe-area-inset-top', 'env(safe-area-inset-top)')
    root.style.setProperty('--safe-area-inset-bottom', 'env(safe-area-inset-bottom)')
    root.style.setProperty('--safe-area-inset-left', 'env(safe-area-inset-left)')
    root.style.setProperty('--safe-area-inset-right', 'env(safe-area-inset-right)')
    
    // Glass effect variables based on theme
    if (isDark) {
      root.style.setProperty('--glass-bg', 'rgba(15, 23, 42, 0.8)')
      root.style.setProperty('--glass-border', 'rgba(255, 255, 255, 0.1)')
      root.style.setProperty('--glass-shadow', 'rgba(0, 0, 0, 0.5)')
    } else {
      root.style.setProperty('--glass-bg', 'rgba(255, 255, 255, 0.8)')
      root.style.setProperty('--glass-border', 'rgba(255, 255, 255, 0.2)')
      root.style.setProperty('--glass-shadow', 'rgba(0, 0, 0, 0.1)')
    }
  }
  
  private getThemeColors(isDark: boolean): ThemeColors {
    if (isDark) {
      return {
        primary: '#3b82f6',
        primaryDark: '#1d4ed8',
        primaryLight: '#60a5fa',
        secondary: '#6366f1',
        background: '#0f172a',
        backgroundSecondary: '#1e293b',
        surface: '#334155',
        surfaceSecondary: '#475569',
        text: '#f8fafc',
        textSecondary: '#e2e8f0',
        textMuted: '#94a3b8',
        border: '#475569',
        borderLight: '#64748b',
        error: '#ef4444',
        warning: '#f59e0b',
        success: '#10b981',
        info: '#3b82f6'
      }
    } else {
      return {
        primary: '#1e40af',
        primaryDark: '#1e3a8a',
        primaryLight: '#3b82f6',
        secondary: '#4f46e5',
        background: '#ffffff',
        backgroundSecondary: '#f8fafc',
        surface: '#ffffff',
        surfaceSecondary: '#f1f5f9',
        text: '#0f172a',
        textSecondary: '#334155',
        textMuted: '#64748b',
        border: '#e2e8f0',
        borderLight: '#f1f5f9',
        error: '#dc2626',
        warning: '#d97706',
        success: '#059669',
        info: '#0284c7'
      }
    }
  }
  
  private updateManifestThemeColor(isDark: boolean): void {
    // Update the manifest theme color dynamically
    // This affects the app's appearance in the task switcher on mobile
    const manifestLink = document.querySelector('link[rel="manifest"]') as HTMLLinkElement
    if (manifestLink) {
      // Create a dynamic manifest with updated theme color
      const themeColor = isDark ? '#0f172a' : '#1e40af'
      const backgroundColor = isDark ? '#0f172a' : '#ffffff'
      
      // Store the original manifest URL if not already stored
      if (!manifestLink.dataset.originalHref) {
        manifestLink.dataset.originalHref = manifestLink.href
      }
      
      // Create a blob URL with the updated manifest
      fetch(manifestLink.dataset.originalHref!)
        .then(response => response.json())
        .then(manifest => {
          const updatedManifest = {
            ...manifest,
            theme_color: themeColor,
            background_color: backgroundColor
          }
          
          const blob = new Blob([JSON.stringify(updatedManifest)], { type: 'application/json' })
          const blobUrl = URL.createObjectURL(blob)
          manifestLink.href = blobUrl
        })
        .catch(error => console.warn('Failed to update manifest theme color:', error))
    }
  }
  
  private applyAccessibilityPreferences(): void {
    const root = document.documentElement
    
    // Apply reduced motion preference
    if (this.getReducedMotionPreference()) {
      root.classList.add('reduce-motion')
    } else {
      root.classList.remove('reduce-motion')
    }
    
    // Apply high contrast preference
    if (this.getHighContrastPreference()) {
      root.classList.add('high-contrast')
    } else {
      root.classList.remove('high-contrast')
    }
  }
  
  private getReducedMotionPreference(): boolean {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  }
  
  private getHighContrastPreference(): boolean {
    return window.matchMedia('(prefers-contrast: high)').matches
  }
  
  // Utility methods for components
  getThemeColors(): ThemeColors {
    return this.getThemeColors(this.isDarkMode())
  }
  
  getCSSVariable(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(`--color-${name}`)
  }
  
  // Mobile-specific theme utilities
  setStatusBarStyle(): void {
    // Set status bar style for mobile browsers and PWA
    const isDark = this.isDarkMode()
    
    // Apple specific
    let statusBarMeta = document.querySelector('meta[name="apple-mobile-web-app-status-bar-style"]')
    if (!statusBarMeta) {
      statusBarMeta = document.createElement('meta')
      statusBarMeta.setAttribute('name', 'apple-mobile-web-app-status-bar-style')
      document.head.appendChild(statusBarMeta)
    }
    
    statusBarMeta.setAttribute('content', isDark ? 'black-translucent' : 'default')
    
    // Microsoft specific
    let msTileColorMeta = document.querySelector('meta[name="msapplication-TileColor"]')
    if (!msTileColorMeta) {
      msTileColorMeta = document.createElement('meta')
      msTileColorMeta.setAttribute('name', 'msapplication-TileColor')
      document.head.appendChild(msTileColorMeta)
    }
    
    msTileColorMeta.setAttribute('content', isDark ? '#0f172a' : '#1e40af')
  }
  
  // Theme persistence for offline use
  exportThemeConfig(): string {
    const config: ThemeConfig = {
      mode: this.currentTheme,
      reducedMotion: this.getReducedMotionPreference(),
      highContrast: this.getHighContrastPreference()
    }
    return JSON.stringify(config)
  }
  
  importThemeConfig(configString: string): boolean {
    try {
      const config: ThemeConfig = JSON.parse(configString)
      this.setTheme(config.mode)
      return true
    } catch (error) {
      console.error('Failed to import theme config:', error)
      return false
    }
  }
  
  // Service Worker integration for theme caching
  notifyServiceWorkerThemeChange(): void {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'THEME_CHANGED',
        theme: this.getEffectiveTheme(),
        config: this.exportThemeConfig()
      })
    }
  }
}

// Global theme utilities
export const themeService = ThemeService.getInstance()

// CSS-in-JS theme helper for Lit components
export function getThemeCSS(isDark?: boolean): string {
  const theme = isDark ?? themeService.isDarkMode()
  const colors = themeService.getThemeColors()
  
  return `
    :host {
      color-scheme: ${theme ? 'dark' : 'light'};
    }
    
    .theme-surface {
      background: ${colors.surface};
      color: ${colors.text};
      border: 1px solid ${colors.border};
    }
    
    .theme-surface-secondary {
      background: ${colors.surfaceSecondary};
      color: ${colors.textSecondary};
    }
    
    .theme-text {
      color: ${colors.text};
    }
    
    .theme-text-secondary {
      color: ${colors.textSecondary};
    }
    
    .theme-text-muted {
      color: ${colors.textMuted};
    }
    
    .theme-primary {
      background: ${colors.primary};
      color: ${theme ? colors.text : '#ffffff'};
    }
    
    .theme-border {
      border-color: ${colors.border};
    }
    
    .theme-glass {
      background: var(--glass-bg);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid var(--glass-border);
      box-shadow: 0 8px 32px var(--glass-shadow);
    }
  `
}