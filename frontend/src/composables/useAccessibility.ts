import { ref, onMounted, onUnmounted } from 'vue'

export interface AccessibilityOptions {
  announcePageChanges?: boolean
  manageFocus?: boolean
  respectReducedMotion?: boolean
  enforceColorContrast?: boolean
}

export interface AccessibilityState {
  isScreenReaderActive: boolean
  prefersReducedMotion: boolean
  prefersHighContrast: boolean
  fontSize: 'small' | 'medium' | 'large' | 'x-large'
  colorScheme: 'light' | 'dark' | 'auto'
}

export function useAccessibility(options: AccessibilityOptions = {}) {
  const {
    announcePageChanges = true,
    manageFocus = true,
    respectReducedMotion = true,
    enforceColorContrast = true
  } = options

  // State
  const state = ref<AccessibilityState>({
    isScreenReaderActive: false,
    prefersReducedMotion: false,
    prefersHighContrast: false,
    fontSize: 'medium',
    colorScheme: 'auto'
  })

  const liveRegion = ref<HTMLElement | null>(null)
  const focusHistory: HTMLElement[] = []

  // Screen reader detection
  const detectScreenReader = () => {
    // Check for common screen reader indicators
    const indicators = [
      // NVDA
      window.navigator.userAgent.includes('NVDA'),
      // JAWS
      'speechSynthesis' in window && window.speechSynthesis.getVoices().length > 0,
      // VoiceOver (approximate detection)
      /Mac|iPhone|iPad/.test(window.navigator.userAgent) && 'speechSynthesis' in window,
      // High contrast mode (often used with screen readers)
      window.matchMedia('(prefers-contrast: high)').matches
    ]
    
    state.value.isScreenReaderActive = indicators.some(Boolean)
  }

  // Detect user preferences
  const detectPreferences = () => {
    // Reduced motion
    if (respectReducedMotion) {
      state.value.prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    }
    
    // High contrast
    if (enforceColorContrast) {
      state.value.prefersHighContrast = window.matchMedia('(prefers-contrast: high)').matches
    }
    
    // Color scheme
    state.value.colorScheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    
    // Font size (approximate based on zoom level)
    const pixelRatio = window.devicePixelRatio || 1
    if (pixelRatio > 1.5) {
      state.value.fontSize = 'large'
    } else if (pixelRatio > 1.2) {
      state.value.fontSize = 'medium'
    } else {
      state.value.fontSize = 'small'
    }
  }

  // Create live region for announcements
  const createLiveRegion = () => {
    if (!announcePageChanges) return
    
    const region = document.createElement('div')
    region.setAttribute('aria-live', 'polite')
    region.setAttribute('aria-atomic', 'true')
    region.setAttribute('class', 'sr-only')
    region.style.cssText = `
      position: absolute;
      left: -10000px;
      width: 1px;
      height: 1px;
      overflow: hidden;
    `
    
    document.body.appendChild(region)
    liveRegion.value = region
  }

  // Announce to screen readers
  const announce = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (!liveRegion.value) return
    
    // Clear previous message
    liveRegion.value.textContent = ''
    
    // Set priority
    liveRegion.value.setAttribute('aria-live', priority)
    
    // Announce new message after a brief delay
    setTimeout(() => {
      if (liveRegion.value) {
        liveRegion.value.textContent = message
      }
    }, 10)
  }

  // Focus management
  const manageFocusOnRouteChange = (newPageTitle: string) => {
    if (!manageFocus) return
    
    // Announce page change
    if (announcePageChanges) {
      announce(`Navigated to ${newPageTitle}`, 'polite')
    }
    
    // Focus management
    const mainContent = document.querySelector('[role="main"]') || 
                       document.querySelector('main') ||
                       document.querySelector('#main-content')
    
    if (mainContent instanceof HTMLElement) {
      // Make it focusable if not already
      if (!mainContent.getAttribute('tabindex')) {
        mainContent.setAttribute('tabindex', '-1')
      }
      
      // Focus after a brief delay to ensure page is ready
      setTimeout(() => {
        mainContent.focus()
      }, 100)
    }
  }

  // Keyboard navigation helpers
  const trapFocus = (container: HTMLElement) => {
    if (!manageFocus) return () => {}
    
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )
    
    const firstElement = focusableElements[0] as HTMLElement
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement
    
    const handleTabKey = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return
      
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
    
    container.addEventListener('keydown', handleTabKey)
    
    // Return cleanup function
    return () => {
      container.removeEventListener('keydown', handleTabKey)
    }
  }

  // Save and restore focus
  const saveFocus = () => {
    const activeElement = document.activeElement as HTMLElement
    if (activeElement && activeElement !== document.body) {
      focusHistory.push(activeElement)
    }
  }

  const restoreFocus = () => {
    const previousElement = focusHistory.pop()
    if (previousElement && document.contains(previousElement)) {
      previousElement.focus()
    }
  }

  // Skip link functionality
  const addSkipLink = (targetSelector: string = '[role="main"], main, #main-content') => {
    const skipLink = document.createElement('a')
    skipLink.href = '#main-content'
    skipLink.textContent = 'Skip to main content'
    skipLink.className = 'skip-link'
    skipLink.style.cssText = `
      position: absolute;
      left: -10000px;
      top: auto;
      width: 1px;
      height: 1px;
      overflow: hidden;
    `
    
    skipLink.addEventListener('focus', () => {
      skipLink.style.cssText = `
        position: absolute;
        left: 6px;
        top: 7px;
        z-index: 999999;
        padding: 8px 16px;
        background: #000;
        color: #fff;
        text-decoration: none;
        border-radius: 3px;
      `
    })
    
    skipLink.addEventListener('blur', () => {
      skipLink.style.cssText = `
        position: absolute;
        left: -10000px;
        top: auto;
        width: 1px;
        height: 1px;
        overflow: hidden;
      `
    })
    
    skipLink.addEventListener('click', (event) => {
      event.preventDefault()
      const target = document.querySelector(targetSelector) as HTMLElement
      if (target) {
        target.setAttribute('tabindex', '-1')
        target.focus()
      }
    })
    
    document.body.insertBefore(skipLink, document.body.firstChild)
  }

  // Color contrast utilities
  const checkColorContrast = (foreground: string, background: string): number => {
    // Simple contrast ratio calculation
    const getLuminance = (hex: string): number => {
      const rgb = parseInt(hex.slice(1), 16)
      const r = (rgb >> 16) & 0xff
      const g = (rgb >> 8) & 0xff
      const b = rgb & 0xff
      
      const [rs, gs, bs] = [r, g, b].map(c => {
        c = c / 255
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
      })
      
      return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs
    }
    
    const l1 = getLuminance(foreground)
    const l2 = getLuminance(background)
    const lighter = Math.max(l1, l2)
    const darker = Math.min(l1, l2)
    
    return (lighter + 0.05) / (darker + 0.05)
  }

  // Media query listeners
  const setupMediaQueryListeners = () => {
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)')
    const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)')
    
    const updatePreferences = () => {
      detectPreferences()
    }
    
    reducedMotionQuery.addEventListener('change', updatePreferences)
    highContrastQuery.addEventListener('change', updatePreferences)
    darkModeQuery.addEventListener('change', updatePreferences)
    
    return () => {
      reducedMotionQuery.removeEventListener('change', updatePreferences)
      highContrastQuery.removeEventListener('change', updatePreferences)
      darkModeQuery.removeEventListener('change', updatePreferences)
    }
  }

  // Lifecycle
  onMounted(() => {
    detectScreenReader()
    detectPreferences()
    createLiveRegion()
    addSkipLink()
    
    const cleanupListeners = setupMediaQueryListeners()
    
    onUnmounted(() => {
      if (liveRegion.value) {
        document.body.removeChild(liveRegion.value)
      }
      cleanupListeners()
    })
  })

  return {
    state,
    announce,
    manageFocusOnRouteChange,
    trapFocus,
    saveFocus,
    restoreFocus,
    checkColorContrast
  }
}