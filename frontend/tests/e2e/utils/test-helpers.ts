import { Page, BrowserContext, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';

/**
 * Common test helpers for PWA E2E testing
 * Provides reusable functions for complex testing scenarios
 */

export interface TestUser {
  id: string;
  email: string;
  role: string;
  permissions: string[];
}

export interface TestAgent {
  id: string;
  name: string;
  status: string;
  type: string;
  configuration: Record<string, any>;
}

export interface TestTask {
  id: string;
  title: string;
  description: string;
  assignedAgent: string;
  status: string;
  priority: string;
}

/**
 * Load test data fixtures
 */
export function loadTestData(): {
  users: TestUser[];
  agents: TestAgent[];
  tasks: TestTask[];
} {
  const fixturesPath = path.join(process.cwd(), 'tests/e2e/fixtures/test-data.json');
  const data = fs.readFileSync(fixturesPath, 'utf-8');
  return JSON.parse(data);
}

/**
 * PWA specific helpers
 */
export class PWAHelpers {
  constructor(private page: Page, private context: BrowserContext) {}

  /**
   * Check if PWA is installable
   */
  async isPWAInstallable(): Promise<boolean> {
    try {
      const beforeInstallPrompt = await this.page.evaluate(() => {
        return new Promise((resolve) => {
          if ('serviceWorker' in navigator) {
            window.addEventListener('beforeinstallprompt', () => resolve(true));
            setTimeout(() => resolve(false), 1000);
          } else {
            resolve(false);
          }
        });
      });
      return beforeInstallPrompt as boolean;
    } catch {
      return false;
    }
  }

  /**
   * Check service worker registration
   */
  async checkServiceWorkerRegistration(): Promise<boolean> {
    return await this.page.evaluate(() => {
      return new Promise((resolve) => {
        if ('serviceWorker' in navigator) {
          navigator.serviceWorker.getRegistration().then(registration => {
            resolve(!!registration);
          }).catch(() => resolve(false));
        } else {
          resolve(false);
        }
      });
    });
  }

  /**
   * Test offline functionality
   */
  async testOfflineMode(): Promise<void> {
    // Set offline mode
    await this.context.setOffline(true);
    
    // Verify page still loads (from cache)
    await this.page.reload();
    await expect(this.page.locator('body')).toBeVisible();
    
    // Restore online mode
    await this.context.setOffline(false);
  }

  /**
   * Check PWA manifest
   */
  async checkManifest(): Promise<any> {
    const manifest = await this.page.evaluate(async () => {
      const manifestLink = document.querySelector('link[rel="manifest"]');
      if (!manifestLink) return null;
      
      try {
        const response = await fetch((manifestLink as HTMLLinkElement).href);
        return await response.json();
      } catch {
        return null;
      }
    });
    
    return manifest;
  }

  /**
   * Check if running as PWA (installed)
   */
  async isRunningAsPWA(): Promise<boolean> {
    return await this.page.evaluate(() => {
      return window.matchMedia('(display-mode: standalone)').matches ||
             window.matchMedia('(display-mode: fullscreen)').matches ||
             window.matchMedia('(display-mode: minimal-ui)').matches ||
             (window.navigator as any).standalone === true;
    });
  }
}

/**
 * Authentication helpers
 */
export class AuthHelpers {
  constructor(private page: Page) {}

  /**
   * Perform login with test user
   */
  async login(user: TestUser): Promise<void> {
    // Navigate to login if not already there
    if (!this.page.url().includes('/login')) {
      await this.page.goto('/login');
    }

    // Fill login form
    await this.page.fill('[data-testid="email-input"]', user.email);
    await this.page.fill('[data-testid="password-input"]', 'test-password');
    
    // Submit login
    await this.page.click('[data-testid="login-submit"]');
    
    // Wait for navigation to dashboard
    await this.page.waitForURL('**/dashboard', { timeout: 10000 });
    
    // Verify user is logged in
    await expect(this.page.locator('[data-testid="user-menu"]')).toBeVisible();
  }

  /**
   * Perform logout
   */
  async logout(): Promise<void> {
    // Click user menu
    await this.page.click('[data-testid="user-menu"]');
    
    // Click logout
    await this.page.click('[data-testid="logout-button"]');
    
    // Wait for redirect to login
    await this.page.waitForURL('**/login', { timeout: 10000 });
  }

  /**
   * Check if user is authenticated
   */
  async isAuthenticated(): Promise<boolean> {
    try {
      await this.page.locator('[data-testid="user-menu"]').waitFor({ 
        state: 'visible', 
        timeout: 2000 
      });
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Navigation helpers
 */
export class NavigationHelpers {
  constructor(private page: Page) {}

  /**
   * Navigate to main sections
   */
  async goToDashboard(): Promise<void> {
    await this.page.click('[data-testid="nav-dashboard"]');
    await this.page.waitForURL('**/dashboard');
  }

  async goToAgents(): Promise<void> {
    await this.page.click('[data-testid="nav-agents"]');
    await this.page.waitForURL('**/agents');
  }

  async goToTasks(): Promise<void> {
    await this.page.click('[data-testid="nav-tasks"]');
    await this.page.waitForURL('**/tasks');
  }

  async goToSettings(): Promise<void> {
    await this.page.click('[data-testid="nav-settings"]');
    await this.page.waitForURL('**/settings');
  }
}

/**
 * Wait for network idle (useful for SPA navigation)
 */
export async function waitForNetworkIdle(page: Page, timeout = 5000): Promise<void> {
  await page.waitForLoadState('networkidle', { timeout });
}

/**
 * Take screenshot with timestamp
 */
export async function takeTimestampedScreenshot(
  page: Page, 
  name: string
): Promise<void> {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `${name}-${timestamp}.png`;
  await page.screenshot({ 
    path: path.join('test-results/screenshots', filename),
    fullPage: true 
  });
}

/**
 * Check for console errors
 */
export async function checkConsoleErrors(page: Page): Promise<string[]> {
  const errors: string[] = [];
  
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });

  return errors;
}

/**
 * Wait for element with retry logic
 */
export async function waitForElementWithRetry(
  page: Page,
  selector: string,
  options: { timeout?: number; retries?: number } = {}
): Promise<void> {
  const { timeout = 5000, retries = 3 } = options;
  
  for (let i = 0; i < retries; i++) {
    try {
      await page.locator(selector).waitFor({ 
        state: 'visible', 
        timeout: timeout / retries 
      });
      return;
    } catch (error) {
      if (i === retries - 1) throw error;
      await page.waitForTimeout(500);
    }
  }
}

/**
 * Performance helpers
 */
export class PerformanceHelpers {
  constructor(private page: Page) {}

  /**
   * Measure page load time
   */
  async measurePageLoadTime(): Promise<number> {
    const startTime = Date.now();
    await this.page.waitForLoadState('domcontentloaded');
    return Date.now() - startTime;
  }

  /**
   * Get Core Web Vitals
   */
  async getCoreWebVitals(): Promise<{
    FCP?: number;
    LCP?: number;
    FID?: number;
    CLS?: number;
  }> {
    return await this.page.evaluate(() => {
      return new Promise((resolve) => {
        const vitals: any = {};
        
        // First Contentful Paint
        new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          entries.forEach((entry) => {
            if (entry.name === 'first-contentful-paint') {
              vitals.FCP = entry.startTime;
            }
          });
        }).observe({ entryTypes: ['paint'] });

        // Largest Contentful Paint
        new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          const lastEntry = entries[entries.length - 1];
          vitals.LCP = lastEntry.startTime;
        }).observe({ entryTypes: ['largest-contentful-paint'] });

        // Cumulative Layout Shift
        new PerformanceObserver((entryList) => {
          let clsValue = 0;
          for (const entry of entryList.getEntries()) {
            if (!(entry as any).hadRecentInput) {
              clsValue += (entry as any).value;
            }
          }
          vitals.CLS = clsValue;
        }).observe({ entryTypes: ['layout-shift'] });

        // Return vitals after a short delay
        setTimeout(() => resolve(vitals), 3000);
      });
    });
  }
}

/**
 * Accessibility helpers
 */
export class AccessibilityHelpers {
  constructor(private page: Page) {}

  /**
   * Check keyboard navigation
   */
  async testKeyboardNavigation(): Promise<boolean> {
    try {
      // Start from first focusable element
      await this.page.keyboard.press('Tab');
      const firstFocus = await this.page.evaluate(() => document.activeElement?.tagName);
      
      // Tab through several elements
      for (let i = 0; i < 5; i++) {
        await this.page.keyboard.press('Tab');
        await this.page.waitForTimeout(100);
      }
      
      // Check if focus changed
      const currentFocus = await this.page.evaluate(() => document.activeElement?.tagName);
      
      return firstFocus !== currentFocus;
    } catch {
      return false;
    }
  }

  /**
   * Check aria labels and roles
   */
  async checkAriaCompliance(): Promise<{
    missingLabels: string[];
    invalidRoles: string[];
  }> {
    return await this.page.evaluate(() => {
      const missingLabels: string[] = [];
      const invalidRoles: string[] = [];
      
      // Check for missing aria-labels on interactive elements
      const interactiveElements = document.querySelectorAll(
        'button, a, input, select, textarea, [role="button"], [role="link"]'
      );
      
      interactiveElements.forEach((element, index) => {
        const hasLabel = element.hasAttribute('aria-label') ||
                         element.hasAttribute('aria-labelledby') ||
                         element.textContent?.trim();
        
        if (!hasLabel) {
          missingLabels.push(`Element ${index}: ${element.tagName}`);
        }
      });

      // Check for invalid ARIA roles
      const elementsWithRoles = document.querySelectorAll('[role]');
      const validRoles = [
        'button', 'link', 'menuitem', 'tab', 'option', 'checkbox', 'radio',
        'textbox', 'combobox', 'grid', 'listbox', 'tree', 'dialog', 'alertdialog'
      ];

      elementsWithRoles.forEach((element, index) => {
        const role = element.getAttribute('role');
        if (role && !validRoles.includes(role)) {
          invalidRoles.push(`Element ${index}: role="${role}"`);
        }
      });

      return { missingLabels, invalidRoles };
    });
  }
}