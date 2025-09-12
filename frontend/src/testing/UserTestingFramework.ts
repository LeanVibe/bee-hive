/**
 * User Testing Framework for LeanVibe Agent Hive Dashboard
 * 
 * This framework provides automated user experience testing capabilities,
 * including accessibility validation, mobile responsiveness, and performance monitoring.
 */

export interface TestScenario {
  id: string
  name: string
  description: string
  steps: TestStep[]
  expectedOutcome: string
  priority: 'high' | 'medium' | 'low'
  devices?: DeviceTarget[]
  accessibility?: boolean
  performance?: boolean
}

export interface TestStep {
  action: 'navigate' | 'click' | 'type' | 'wait' | 'scroll' | 'swipe' | 'verify'
  selector?: string
  value?: string
  timeout?: number
  assertion?: Assertion
}

export interface Assertion {
  type: 'exists' | 'visible' | 'text' | 'attribute' | 'count' | 'style'
  target: string
  expected: any
  tolerance?: number
}

export interface DeviceTarget {
  name: string
  width: number
  height: number
  userAgent: string
  touchEnabled: boolean
}

export interface TestResult {
  scenarioId: string
  status: 'passed' | 'failed' | 'skipped'
  duration: number
  errors: TestError[]
  screenshots: string[]
  metrics: PerformanceMetrics
  accessibilityScore: AccessibilityScore
  timestamp: number
}

export interface TestError {
  step: number
  message: string
  stack?: string
  screenshot?: string
}

export interface PerformanceMetrics {
  loadTime: number
  firstContentfulPaint: number
  largestContentfulPaint: number
  cumulativeLayoutShift: number
  firstInputDelay: number
  memoryUsage: number
  networkRequests: number
}

export interface AccessibilityScore {
  score: number
  violations: AccessibilityViolation[]
  warnings: AccessibilityWarning[]
}

export interface AccessibilityViolation {
  rule: string
  impact: 'critical' | 'serious' | 'moderate' | 'minor'
  description: string
  element: string
  help: string
}

export interface AccessibilityWarning {
  rule: string
  description: string
  element: string
  suggestion: string
}

class UserTestingFramework {
  private scenarios: Map<string, TestScenario> = new Map()
  private results: TestResult[] = []
  private currentTest: TestResult | null = null
  private performanceObserver: PerformanceObserver | null = null
  private mutationObserver: MutationObserver | null = null
  
  // Default device targets
  private deviceTargets: DeviceTarget[] = [
    {
      name: 'iPhone 12',
      width: 390,
      height: 844,
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15',
      touchEnabled: true
    },
    {
      name: 'iPad Air',
      width: 820,
      height: 1180,
      userAgent: 'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15',
      touchEnabled: true
    },
    {
      name: 'Desktop',
      width: 1920,
      height: 1080,
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      touchEnabled: false
    },
    {
      name: 'Tablet',
      width: 768,
      height: 1024,
      userAgent: 'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15',
      touchEnabled: true
    }
  ]

  constructor() {
    this.setupPerformanceMonitoring()
    this.loadDefaultScenarios()
  }

  // Scenario Management
  registerScenario(scenario: TestScenario): void {
    this.scenarios.set(scenario.id, scenario)
  }

  getScenario(id: string): TestScenario | undefined {
    return this.scenarios.get(id)
  }

  getAllScenarios(): TestScenario[] {
    return Array.from(this.scenarios.values())
  }

  // Test Execution
  async runScenario(scenarioId: string, device?: DeviceTarget): Promise<TestResult> {
    const scenario = this.scenarios.get(scenarioId)
    if (!scenario) {
      throw new Error(`Scenario ${scenarioId} not found`)
    }

    console.log(`🧪 Running test scenario: ${scenario.name}`)
    
    const result: TestResult = {
      scenarioId,
      status: 'passed',
      duration: 0,
      errors: [],
      screenshots: [],
      metrics: this.initializeMetrics(),
      accessibilityScore: { score: 0, violations: [], warnings: [] },
      timestamp: Date.now()
    }

    this.currentTest = result
    const startTime = performance.now()

    try {
      // Set up device viewport if specified
      if (device) {
        await this.setViewport(device)
      }

      // Execute test steps
      for (let i = 0; i < scenario.steps.length; i++) {
        const step = scenario.steps[i]
        
        try {
          await this.executeStep(step, i)
          
          // Take screenshot after critical steps
          if (step.action === 'navigate' || step.action === 'click') {
            result.screenshots.push(await this.takeScreenshot())
          }
          
        } catch (error) {
          result.status = 'failed'
          result.errors.push({
            step: i,
            message: error instanceof Error ? error.message : 'Unknown error',
            stack: error instanceof Error ? error.stack : undefined,
            screenshot: await this.takeScreenshot()
          })
          
          console.error(`Step ${i} failed:`, error)
          break
        }
      }

      // Run accessibility tests if enabled
      if (scenario.accessibility) {
        result.accessibilityScore = await this.runAccessibilityTests()
      }

      // Collect performance metrics if enabled
      if (scenario.performance) {
        result.metrics = await this.collectPerformanceMetrics()
      }

    } catch (error) {
      result.status = 'failed'
      result.errors.push({
        step: -1,
        message: error instanceof Error ? error.message : 'Test setup failed',
        stack: error instanceof Error ? error.stack : undefined
      })
    }

    result.duration = performance.now() - startTime
    this.results.push(result)
    this.currentTest = null

    console.log(`✅ Test completed: ${result.status} in ${result.duration.toFixed(2)}ms`)
    
    return result
  }

  async runAllScenarios(device?: DeviceTarget): Promise<TestResult[]> {
    const results: TestResult[] = []
    
    for (const scenario of this.scenarios.values()) {
      const result = await this.runScenario(scenario.id, device)
      results.push(result)
    }

    return results
  }

  async runCrossDeviceTests(scenarioId: string): Promise<Map<string, TestResult>> {
    const results = new Map<string, TestResult>()
    
    for (const device of this.deviceTargets) {
      console.log(`📱 Testing on ${device.name}`)
      const result = await this.runScenario(scenarioId, device)
      results.set(device.name, result)
    }

    return results
  }

  // Step Execution
  private async executeStep(step: TestStep, stepIndex: number): Promise<void> {
    console.log(`Executing step ${stepIndex}: ${step.action}`)

    switch (step.action) {
      case 'navigate':
        await this.navigate(step.value || '/')
        break
        
      case 'click':
        await this.click(step.selector!)
        break
        
      case 'type':
        await this.type(step.selector!, step.value!)
        break
        
      case 'wait':
        await this.wait(step.timeout || 1000)
        break
        
      case 'scroll':
        await this.scroll(step.selector, step.value)
        break
        
      case 'swipe':
        await this.swipe(step.selector!, step.value!)
        break
        
      case 'verify':
        await this.verify(step.assertion!)
        break
        
      default:
        throw new Error(`Unknown step action: ${step.action}`)
    }
  }

  private async navigate(url: string): Promise<void> {
    window.history.pushState({}, '', url)
    await this.waitForLoad()
  }

  private async click(selector: string): Promise<void> {
    const element = document.querySelector(selector)
    if (!element) {
      throw new Error(`Element not found: ${selector}`)
    }

    // Simulate mouse/touch event
    const event = new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      view: window
    })
    
    element.dispatchEvent(event)
    await this.wait(100) // Small delay for UI updates
  }

  private async type(selector: string, value: string): Promise<void> {
    const element = document.querySelector(selector) as HTMLInputElement
    if (!element) {
      throw new Error(`Input element not found: ${selector}`)
    }

    element.focus()
    element.value = value
    
    // Trigger input events
    element.dispatchEvent(new Event('input', { bubbles: true }))
    element.dispatchEvent(new Event('change', { bubbles: true }))
  }

  private async wait(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  private async scroll(selector?: string, direction?: string): Promise<void> {
    const element = selector ? document.querySelector(selector) : window
    if (!element) {
      throw new Error(`Scroll target not found: ${selector}`)
    }

    const scrollOptions: ScrollToOptions = {
      behavior: 'smooth'
    }

    if (direction === 'top') {
      scrollOptions.top = 0
    } else if (direction === 'bottom') {
      scrollOptions.top = document.body.scrollHeight
    } else if (direction && !isNaN(parseInt(direction))) {
      scrollOptions.top = parseInt(direction)
    }

    if (element === window) {
      window.scrollTo(scrollOptions)
    } else {
      (element as Element).scrollTo(scrollOptions)
    }

    await this.wait(500) // Wait for scroll animation
  }

  private async swipe(selector: string, direction: string): Promise<void> {
    const element = document.querySelector(selector)
    if (!element) {
      throw new Error(`Swipe target not found: ${selector}`)
    }

    const rect = element.getBoundingClientRect()
    const centerX = rect.left + rect.width / 2
    const centerY = rect.top + rect.height / 2

    let startX = centerX
    let startY = centerY
    let endX = centerX
    let endY = centerY

    const swipeDistance = 100

    switch (direction) {
      case 'left':
        endX = startX - swipeDistance
        break
      case 'right':
        endX = startX + swipeDistance
        break
      case 'up':
        endY = startY - swipeDistance
        break
      case 'down':
        endY = startY + swipeDistance
        break
    }

    // Simulate touch events
    const touchStart = new TouchEvent('touchstart', {
      touches: [new Touch({
        identifier: 1,
        target: element,
        clientX: startX,
        clientY: startY
      })]
    })

    const touchEnd = new TouchEvent('touchend', {
      touches: [new Touch({
        identifier: 1,
        target: element,
        clientX: endX,
        clientY: endY
      })]
    })

    element.dispatchEvent(touchStart)
    await this.wait(50)
    element.dispatchEvent(touchEnd)
    await this.wait(200)
  }

  private async verify(assertion: Assertion): Promise<void> {
    const element = document.querySelector(assertion.target)

    switch (assertion.type) {
      case 'exists':
        if (!element && assertion.expected) {
          throw new Error(`Element should exist: ${assertion.target}`)
        }
        if (element && !assertion.expected) {
          throw new Error(`Element should not exist: ${assertion.target}`)
        }
        break

      case 'visible':
        if (!element) {
          throw new Error(`Element not found: ${assertion.target}`)
        }
        const isVisible = (element as HTMLElement).offsetWidth > 0 && (element as HTMLElement).offsetHeight > 0
        if (isVisible !== assertion.expected) {
          throw new Error(`Element visibility mismatch: ${assertion.target}`)
        }
        break

      case 'text':
        if (!element) {
          throw new Error(`Element not found: ${assertion.target}`)
        }
        const actualText = element.textContent?.trim()
        if (actualText !== assertion.expected) {
          throw new Error(`Text mismatch. Expected: "${assertion.expected}", Got: "${actualText}"`)
        }
        break

      case 'attribute':
        if (!element) {
          throw new Error(`Element not found: ${assertion.target}`)
        }
        const attrValue = element.getAttribute(assertion.expected.name)
        if (attrValue !== assertion.expected.value) {
          throw new Error(`Attribute mismatch: ${assertion.expected.name}`)
        }
        break

      case 'count':
        const elements = document.querySelectorAll(assertion.target)
        if (elements.length !== assertion.expected) {
          throw new Error(`Element count mismatch. Expected: ${assertion.expected}, Got: ${elements.length}`)
        }
        break

      case 'style':
        if (!element) {
          throw new Error(`Element not found: ${assertion.target}`)
        }
        const style = window.getComputedStyle(element)
        const actualValue = style.getPropertyValue(assertion.expected.property)
        if (actualValue !== assertion.expected.value) {
          throw new Error(`Style mismatch: ${assertion.expected.property}`)
        }
        break
    }
  }

  // Accessibility Testing
  private async runAccessibilityTests(): Promise<AccessibilityScore> {
    const violations: AccessibilityViolation[] = []
    const warnings: AccessibilityWarning[] = []

    // Check for common accessibility issues
    await this.checkColorContrast(violations)
    await this.checkAriaLabels(violations, warnings)
    await this.checkKeyboardNavigation(violations, warnings)
    await this.checkHeadingStructure(violations, warnings)
    await this.checkFormLabels(violations, warnings)
    await this.checkImageAltText(violations, warnings)

    // Calculate score (100 - violations weighted by impact)
    const impactWeights = { critical: 25, serious: 15, moderate: 10, minor: 5 }
    const totalDeductions = violations.reduce((sum, v) => sum + impactWeights[v.impact], 0)
    const score = Math.max(0, 100 - totalDeductions)

    return { score, violations, warnings }
  }

  private async checkColorContrast(violations: AccessibilityViolation[]): Promise<void> {
    const elements = document.querySelectorAll('*')
    
    for (const element of elements) {
      const style = window.getComputedStyle(element)
      const color = style.color
      const backgroundColor = style.backgroundColor
      
      if (color && backgroundColor && color !== 'rgba(0, 0, 0, 0)' && backgroundColor !== 'rgba(0, 0, 0, 0)') {
        const contrast = this.calculateColorContrast(color, backgroundColor)
        
        if (contrast < 4.5) {
          violations.push({
            rule: 'color-contrast',
            impact: 'serious',
            description: `Insufficient color contrast ratio: ${contrast.toFixed(2)}`,
            element: this.getElementSelector(element),
            help: 'Ensure text has sufficient contrast against its background'
          })
        }
      }
    }
  }

  private async checkAriaLabels(violations: AccessibilityViolation[], warnings: AccessibilityWarning[]): Promise<void> {
    // Check interactive elements for ARIA labels
    const interactiveElements = document.querySelectorAll('button, a, input, select, textarea')
    
    for (const element of interactiveElements) {
      const hasAriaLabel = element.hasAttribute('aria-label') || 
                          element.hasAttribute('aria-labelledby') ||
                          element.hasAttribute('aria-describedby')
      
      const hasVisibleText = element.textContent?.trim() || 
                            element.getAttribute('alt') ||
                            element.getAttribute('title')
      
      if (!hasAriaLabel && !hasVisibleText) {
        violations.push({
          rule: 'aria-label',
          impact: 'serious',
          description: 'Interactive element lacks accessible name',
          element: this.getElementSelector(element),
          help: 'Add aria-label, visible text, or other accessible name'
        })
      }
    }
  }

  private async checkKeyboardNavigation(violations: AccessibilityViolation[], warnings: AccessibilityWarning[]): Promise<void> {
    const focusableElements = document.querySelectorAll('button, a, input, select, textarea, [tabindex]')
    
    for (const element of focusableElements) {
      const tabIndex = element.getAttribute('tabindex')
      
      if (tabIndex && parseInt(tabIndex) > 0) {
        warnings.push({
          rule: 'tabindex',
          description: 'Positive tabindex may disrupt keyboard navigation',
          element: this.getElementSelector(element),
          suggestion: 'Use tabindex="0" or rely on natural tab order'
        })
      }
      
      // Check for focus indicators
      const style = window.getComputedStyle(element, ':focus')
      if (!style.outline || style.outline === 'none') {
        warnings.push({
          rule: 'focus-indicator',
          description: 'Element may lack visible focus indicator',
          element: this.getElementSelector(element),
          suggestion: 'Ensure focusable elements have visible focus states'
        })
      }
    }
  }

  private async checkHeadingStructure(violations: AccessibilityViolation[], warnings: AccessibilityWarning[]): Promise<void> {
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6')
    let previousLevel = 0
    
    for (const heading of headings) {
      const level = parseInt(heading.tagName.charAt(1))
      
      if (level > previousLevel + 1) {
        violations.push({
          rule: 'heading-structure',
          impact: 'moderate',
          description: `Heading level ${level} skips level ${previousLevel + 1}`,
          element: this.getElementSelector(heading),
          help: 'Use heading levels in sequential order'
        })
      }
      
      previousLevel = level
    }
  }

  private async checkFormLabels(violations: AccessibilityViolation[], warnings: AccessibilityWarning[]): Promise<void> {
    const formInputs = document.querySelectorAll('input:not([type="hidden"]), select, textarea')
    
    for (const input of formInputs) {
      const id = input.getAttribute('id')
      const hasLabel = id && document.querySelector(`label[for="${id}"]`)
      const hasAriaLabel = input.hasAttribute('aria-label') || input.hasAttribute('aria-labelledby')
      
      if (!hasLabel && !hasAriaLabel) {
        violations.push({
          rule: 'form-label',
          impact: 'serious',
          description: 'Form input lacks associated label',
          element: this.getElementSelector(input),
          help: 'Associate input with label element or add aria-label'
        })
      }
    }
  }

  private async checkImageAltText(violations: AccessibilityViolation[], warnings: AccessibilityWarning[]): Promise<void> {
    const images = document.querySelectorAll('img')
    
    for (const img of images) {
      const alt = img.getAttribute('alt')
      const hasAriaLabel = img.hasAttribute('aria-label') || img.hasAttribute('aria-labelledby')
      
      if (alt === null && !hasAriaLabel) {
        violations.push({
          rule: 'image-alt',
          impact: 'serious',
          description: 'Image lacks alternative text',
          element: this.getElementSelector(img),
          help: 'Add meaningful alt attribute to describe image content'
        })
      } else if (alt === '') {
        // Empty alt is OK for decorative images, but warn if it might be content
        const src = img.getAttribute('src')
        if (src && !src.includes('icon') && !src.includes('decoration')) {
          warnings.push({
            rule: 'image-alt-empty',
            description: 'Image has empty alt attribute - ensure it is decorative',
            element: this.getElementSelector(img),
            suggestion: 'Verify that empty alt is appropriate for decorative images'
          })
        }
      }
    }
  }

  // Performance Monitoring
  private setupPerformanceMonitoring(): void {
    if ('PerformanceObserver' in window) {
      this.performanceObserver = new PerformanceObserver((list) => {
        // Store performance entries for analysis
        const entries = list.getEntries()
        if (this.currentTest) {
          // Update performance metrics during active test
          this.updatePerformanceMetrics(entries)
        }
      })
      
      this.performanceObserver.observe({ entryTypes: ['paint', 'layout-shift', 'first-input'] })
    }
  }

  private async collectPerformanceMetrics(): Promise<PerformanceMetrics> {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    
    return {
      loadTime: navigation.loadEventEnd - navigation.loadEventStart,
      firstContentfulPaint: this.getMetricValue('first-contentful-paint'),
      largestContentfulPaint: this.getMetricValue('largest-contentful-paint'),
      cumulativeLayoutShift: this.getMetricValue('cumulative-layout-shift'),
      firstInputDelay: this.getMetricValue('first-input-delay'),
      memoryUsage: (performance as any).memory?.usedJSHeapSize || 0,
      networkRequests: performance.getEntriesByType('resource').length
    }
  }

  private getMetricValue(name: string): number {
    const entries = performance.getEntriesByName(name)
    return entries.length > 0 ? entries[entries.length - 1].startTime : 0
  }

  private updatePerformanceMetrics(entries: PerformanceEntry[]): void {
    // Update current test metrics
    // This is called during test execution to capture real-time performance
  }

  // Utility Methods
  private initializeMetrics(): PerformanceMetrics {
    return {
      loadTime: 0,
      firstContentfulPaint: 0,
      largestContentfulPaint: 0,
      cumulativeLayoutShift: 0,
      firstInputDelay: 0,
      memoryUsage: 0,
      networkRequests: 0
    }
  }

  private async setViewport(device: DeviceTarget): Promise<void> {
    // In a real browser automation scenario, this would set the viewport
    // For in-browser testing, we simulate by setting CSS
    document.documentElement.style.width = `${device.width}px`
    document.documentElement.style.height = `${device.height}px`
    
    // Update user agent if possible (limited in browser context)
    Object.defineProperty(navigator, 'userAgent', {
      value: device.userAgent,
      configurable: true
    })
  }

  private async takeScreenshot(): Promise<string> {
    // In a real implementation, this would capture actual screenshots
    // For demo purposes, return a placeholder
    return `screenshot_${Date.now()}.png`
  }

  private async waitForLoad(): Promise<void> {
    return new Promise((resolve) => {
      if (document.readyState === 'complete') {
        resolve()
      } else {
        window.addEventListener('load', () => resolve(), { once: true })
      }
    })
  }

  private calculateColorContrast(color1: string, color2: string): number {
    // Simplified color contrast calculation
    // In a real implementation, this would properly parse CSS colors and calculate WCAG contrast ratio
    return 4.5 // Placeholder
  }

  private getElementSelector(element: Element): string {
    // Generate a CSS selector for the element
    if (element.id) {
      return `#${element.id}`
    }
    
    if (element.className) {
      return `.${element.className.split(' ').join('.')}`
    }
    
    return element.tagName.toLowerCase()
  }

  // Default Test Scenarios
  private loadDefaultScenarios(): void {
    // Dashboard navigation test
    this.registerScenario({
      id: 'dashboard-navigation',
      name: 'Dashboard Navigation Test',
      description: 'Tests basic navigation through dashboard sections',
      priority: 'high',
      accessibility: true,
      performance: true,
      steps: [
        { action: 'navigate', value: '/' },
        { action: 'verify', assertion: { type: 'exists', target: '[role="main"]', expected: true } },
        { action: 'click', selector: 'a[href="/metrics"]' },
        { action: 'verify', assertion: { type: 'text', target: 'h1', expected: 'Metrics Dashboard' } },
        { action: 'click', selector: 'a[href="/events"]' },
        { action: 'verify', assertion: { type: 'exists', target: '.events-section', expected: true } }
      ],
      expectedOutcome: 'User can navigate between main dashboard sections'
    })

    // Mobile responsiveness test
    this.registerScenario({
      id: 'mobile-responsiveness',
      name: 'Mobile Responsiveness Test',
      description: 'Tests dashboard behavior on mobile devices',
      priority: 'high',
      accessibility: true,
      devices: this.deviceTargets.filter(d => d.touchEnabled),
      steps: [
        { action: 'navigate', value: '/' },
        { action: 'verify', assertion: { type: 'exists', target: '.mobile-navigation', expected: true } },
        { action: 'swipe', selector: '.metric-card', value: 'left' },
        { action: 'scroll', value: 'bottom' },
        { action: 'verify', assertion: { type: 'visible', target: '.bottom-navigation', expected: true } }
      ],
      expectedOutcome: 'Dashboard is fully functional on mobile devices'
    })

    // Accessibility test
    this.registerScenario({
      id: 'accessibility-compliance',
      name: 'Accessibility Compliance Test',
      description: 'Tests WCAG compliance and screen reader compatibility',
      priority: 'high',
      accessibility: true,
      steps: [
        { action: 'navigate', value: '/' },
        { action: 'verify', assertion: { type: 'exists', target: '[role="main"]', expected: true } },
        { action: 'verify', assertion: { type: 'exists', target: 'h1', expected: true } },
        { action: 'verify', assertion: { type: 'attribute', target: 'img', expected: { name: 'alt', value: true } } }
      ],
      expectedOutcome: 'Dashboard meets WCAG AA accessibility standards'
    })

    // Performance test
    this.registerScenario({
      id: 'performance-baseline',
      name: 'Performance Baseline Test',
      description: 'Measures core performance metrics',
      priority: 'medium',
      performance: true,
      steps: [
        { action: 'navigate', value: '/' },
        { action: 'wait', timeout: 3000 },
        { action: 'verify', assertion: { type: 'exists', target: '.dashboard-loaded', expected: true } }
      ],
      expectedOutcome: 'Dashboard loads within performance budgets'
    })
  }

  // Reporting
  generateReport(): any {
    const passedTests = this.results.filter(r => r.status === 'passed').length
    const failedTests = this.results.filter(r => r.status === 'failed').length
    const totalTests = this.results.length

    const avgDuration = this.results.reduce((sum, r) => sum + r.duration, 0) / totalTests
    const avgAccessibilityScore = this.results.reduce((sum, r) => sum + r.accessibilityScore.score, 0) / totalTests

    return {
      summary: {
        total: totalTests,
        passed: passedTests,
        failed: failedTests,
        passRate: totalTests > 0 ? (passedTests / totalTests) * 100 : 0,
        avgDuration,
        avgAccessibilityScore
      },
      results: this.results,
      recommendations: this.generateRecommendations()
    }
  }

  private generateRecommendations(): string[] {
    const recommendations: string[] = []
    
    // Analyze results and generate actionable recommendations
    const failedTests = this.results.filter(r => r.status === 'failed')
    const lowAccessibilityScores = this.results.filter(r => r.accessibilityScore.score < 80)
    const slowTests = this.results.filter(r => r.duration > 5000)

    if (failedTests.length > 0) {
      recommendations.push(`Fix ${failedTests.length} failing test scenarios`)
    }

    if (lowAccessibilityScores.length > 0) {
      recommendations.push('Improve accessibility scores - focus on color contrast and ARIA labels')
    }

    if (slowTests.length > 0) {
      recommendations.push('Optimize performance for better user experience')
    }

    return recommendations
  }

  // Public API
  getResults(): TestResult[] {
    return this.results
  }

  clearResults(): void {
    this.results = []
  }

  getDeviceTargets(): DeviceTarget[] {
    return this.deviceTargets
  }

  addDeviceTarget(device: DeviceTarget): void {
    this.deviceTargets.push(device)
  }
}

export default UserTestingFramework