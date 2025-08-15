/**
 * Accessibility utilities for WCAG AA compliance
 * Ensures the PWA is accessible to all users including those using assistive technologies
 */

// ARIA live region manager for screen reader announcements
export class LiveRegionManager {
  private static instance: LiveRegionManager
  private liveRegions: Map<string, HTMLElement> = new Map()

  static getInstance(): LiveRegionManager {
    if (!LiveRegionManager.instance) {
      LiveRegionManager.instance = new LiveRegionManager()
    }
    return LiveRegionManager.instance
  }

  constructor() {
    this.createLiveRegions()
  }

  private createLiveRegions() {
    // Create polite live region for status updates
    const politeRegion = document.createElement('div')
    politeRegion.setAttribute('aria-live', 'polite')
    politeRegion.setAttribute('aria-atomic', 'true')
    politeRegion.className = 'sr-only'
    politeRegion.id = 'live-region-polite'
    document.body.appendChild(politeRegion)
    this.liveRegions.set('polite', politeRegion)

    // Create assertive live region for urgent announcements
    const assertiveRegion = document.createElement('div')
    assertiveRegion.setAttribute('aria-live', 'assertive')
    assertiveRegion.setAttribute('aria-atomic', 'true')
    assertiveRegion.className = 'sr-only'
    assertiveRegion.id = 'live-region-assertive'
    document.body.appendChild(assertiveRegion)
    this.liveRegions.set('assertive', assertiveRegion)
  }

  announce(message: string, priority: 'polite' | 'assertive' = 'polite') {
    const region = this.liveRegions.get(priority)
    if (region) {
      // Clear previous content
      region.textContent = ''
      
      // Use setTimeout to ensure screen readers pick up the change
      setTimeout(() => {
        region.textContent = message
      }, 100)

      // Clear the message after 5 seconds
      setTimeout(() => {
        region.textContent = ''
      }, 5000)
    }
  }
}

// Focus management utilities
export class FocusManager {
  private focusStack: HTMLElement[] = []
  private trapElement: HTMLElement | null = null
  private beforeTrapElement: HTMLElement | null = null

  // Trap focus within an element (for modals, dialogs)
  trapFocus(element: HTMLElement) {
    this.beforeTrapElement = document.activeElement as HTMLElement
    this.trapElement = element
    
    const focusableElements = this.getFocusableElements(element)
    if (focusableElements.length > 0) {
      focusableElements[0].focus()
    }

    document.addEventListener('keydown', this.handleTrapKeydown)
  }

  // Release focus trap
  releaseFocus() {
    document.removeEventListener('keydown', this.handleTrapKeydown)
    
    if (this.beforeTrapElement) {
      this.beforeTrapElement.focus()
    }
    
    this.trapElement = null
    this.beforeTrapElement = null
  }

  private handleTrapKeydown = (event: KeyboardEvent) => {
    if (event.key !== 'Tab' || !this.trapElement) return

    const focusableElements = this.getFocusableElements(this.trapElement)
    if (focusableElements.length === 0) return

    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]

    if (event.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstElement) {
        event.preventDefault()
        lastElement.focus()
      }
    } else {
      // Tab
      if (document.activeElement === lastElement) {
        event.preventDefault()
        firstElement.focus()
      }
    }
  }

  // Get all focusable elements within a container
  private getFocusableElements(container: HTMLElement): HTMLElement[] {
    const focusableSelectors = [
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      'a[href]',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]'
    ].join(', ')

    return Array.from(container.querySelectorAll(focusableSelectors))
      .filter(element => {
        // Check if element is visible
        const style = window.getComputedStyle(element)
        return style.display !== 'none' && 
               style.visibility !== 'hidden' && 
               style.opacity !== '0'
      }) as HTMLElement[]
  }

  // Save current focus to stack
  pushFocus() {
    const activeElement = document.activeElement as HTMLElement
    if (activeElement) {
      this.focusStack.push(activeElement)
    }
  }

  // Restore previous focus from stack
  popFocus() {
    const element = this.focusStack.pop()
    if (element) {
      element.focus()
    }
  }
}

// Color contrast checker for WCAG compliance
export class ContrastChecker {
  // Calculate relative luminance
  private static getLuminance(color: string): number {
    const rgb = this.hexToRgb(color)
    if (!rgb) return 0

    const { r, g, b } = rgb
    const [sR, sG, sB] = [r, g, b].map(c => {
      c = c / 255
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
    })

    return 0.2126 * sR + 0.7152 * sG + 0.0722 * sB
  }

  // Convert hex color to RGB
  private static hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null
  }

  // Check contrast ratio between two colors
  static checkContrast(
    foreground: string, 
    background: string
  ): { ratio: number; isAA: boolean; isAAA: boolean } {
    const fgLuminance = this.getLuminance(foreground)
    const bgLuminance = this.getLuminance(background)
    
    const ratio = (Math.max(fgLuminance, bgLuminance) + 0.05) / 
                  (Math.min(fgLuminance, bgLuminance) + 0.05)

    return {
      ratio,
      isAA: ratio >= 4.5,   // WCAG AA standard
      isAAA: ratio >= 7     // WCAG AAA standard
    }
  }

  // Validate all contrast ratios in the current theme
  static validateThemeContrasts(): Array<{
    property: string
    foreground: string
    background: string
    result: { ratio: number; isAA: boolean; isAAA: boolean }
  }> {
    const rootStyles = getComputedStyle(document.documentElement)
    const results = []

    const textColors = [
      '--color-text',
      '--color-text-secondary', 
      '--color-text-muted'
    ]

    const backgrounds = [
      '--color-background',
      '--color-surface',
      '--color-surface-secondary'
    ]

    textColors.forEach(textColor => {
      backgrounds.forEach(bgColor => {
        const fg = rootStyles.getPropertyValue(textColor).trim()
        const bg = rootStyles.getPropertyValue(bgColor).trim()
        
        if (fg && bg) {
          results.push({
            property: `${textColor} on ${bgColor}`,
            foreground: fg,
            background: bg,
            result: this.checkContrast(fg, bg)
          })
        }
      })
    })

    return results
  }
}

// Keyboard navigation utilities
export class KeyboardNavigationManager {
  private static skipLinksContainer: HTMLElement | null = null

  // Create skip links for keyboard navigation
  static createSkipLinks(links: Array<{ href: string; text: string }>) {
    if (this.skipLinksContainer) {
      this.skipLinksContainer.remove()
    }

    this.skipLinksContainer = document.createElement('div')
    this.skipLinksContainer.className = 'skip-links'
    this.skipLinksContainer.setAttribute('aria-label', 'Skip navigation links')

    links.forEach(link => {
      const skipLink = document.createElement('a')
      skipLink.href = link.href
      skipLink.textContent = link.text
      skipLink.className = 'skip-link'
      this.skipLinksContainer!.appendChild(skipLink)
    })

    document.body.insertBefore(this.skipLinksContainer, document.body.firstChild)
  }

  // Handle roving tabindex for complex widgets
  static setupRovingTabindex(
    container: HTMLElement, 
    itemSelector: string,
    orientation: 'horizontal' | 'vertical' = 'horizontal'
  ) {
    const items = Array.from(container.querySelectorAll(itemSelector)) as HTMLElement[]
    if (items.length === 0) return

    // Set initial tabindex
    items.forEach((item, index) => {
      item.setAttribute('tabindex', index === 0 ? '0' : '-1')
    })

    // Handle keyboard navigation
    container.addEventListener('keydown', (event) => {
      const currentIndex = items.findIndex(item => item === document.activeElement)
      if (currentIndex === -1) return

      let nextIndex = currentIndex

      switch (event.key) {
        case 'ArrowRight':
          if (orientation === 'horizontal') {
            event.preventDefault()
            nextIndex = (currentIndex + 1) % items.length
          }
          break
        case 'ArrowLeft':
          if (orientation === 'horizontal') {
            event.preventDefault()
            nextIndex = currentIndex === 0 ? items.length - 1 : currentIndex - 1
          }
          break
        case 'ArrowDown':
          if (orientation === 'vertical') {
            event.preventDefault()
            nextIndex = (currentIndex + 1) % items.length
          }
          break
        case 'ArrowUp':
          if (orientation === 'vertical') {
            event.preventDefault()
            nextIndex = currentIndex === 0 ? items.length - 1 : currentIndex - 1
          }
          break
        case 'Home':
          event.preventDefault()
          nextIndex = 0
          break
        case 'End':
          event.preventDefault()
          nextIndex = items.length - 1
          break
      }

      if (nextIndex !== currentIndex) {
        // Update tabindex
        items[currentIndex].setAttribute('tabindex', '-1')
        items[nextIndex].setAttribute('tabindex', '0')
        items[nextIndex].focus()
      }
    })
  }
}

// Screen reader utilities
export class ScreenReaderUtilities {
  // Create descriptive text for complex UI elements
  static createDescription(element: HTMLElement, description: string) {
    const descId = `desc-${Math.random().toString(36).substr(2, 9)}`
    
    const descElement = document.createElement('div')
    descElement.id = descId
    descElement.className = 'sr-only'
    descElement.textContent = description
    
    element.parentNode?.insertBefore(descElement, element.nextSibling)
    element.setAttribute('aria-describedby', descId)
  }

  // Update progress indicators for screen readers
  static updateProgress(
    element: HTMLElement, 
    value: number, 
    max: number = 100,
    label?: string
  ) {
    element.setAttribute('role', 'progressbar')
    element.setAttribute('aria-valuenow', value.toString())
    element.setAttribute('aria-valuemax', max.toString())
    element.setAttribute('aria-valuemin', '0')
    
    if (label) {
      element.setAttribute('aria-label', label)
    }

    // Add percentage for better context
    const percentage = Math.round((value / max) * 100)
    element.setAttribute('aria-valuetext', `${percentage}% complete`)
  }

  // Announce status changes
  static announceStatus(message: string, priority: 'polite' | 'assertive' = 'polite') {
    LiveRegionManager.getInstance().announce(message, priority)
  }
}

// Export singleton instances
export const liveRegionManager = LiveRegionManager.getInstance()
export const focusManager = new FocusManager()

// Initialize accessibility features
export function initializeAccessibility() {
  // Create skip links
  KeyboardNavigationManager.createSkipLinks([
    { href: '#main-content', text: 'Skip to main content' },
    { href: '#navigation', text: 'Skip to navigation' },
    { href: '#search', text: 'Skip to search' }
  ])

  // Add global keyboard event listeners
  document.addEventListener('keydown', (event) => {
    // Escape key handling
    if (event.key === 'Escape') {
      // Close any open modals or focus traps
      focusManager.releaseFocus()
    }
  })

  // Validate theme contrasts in development
  if (process.env.NODE_ENV === 'development') {
    setTimeout(() => {
      const contrastResults = ContrastChecker.validateThemeContrasts()
      const failedContrasts = contrastResults.filter(result => !result.result.isAA)
      
      if (failedContrasts.length > 0) {
        console.warn('WCAG AA contrast failures detected:', failedContrasts)
      } else {
        console.log('✅ All theme contrasts meet WCAG AA standards')
      }
    }, 1000)
  }

  console.log('✅ Accessibility features initialized')
}