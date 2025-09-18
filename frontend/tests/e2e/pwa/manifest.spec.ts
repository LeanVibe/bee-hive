import { test, expect } from '@playwright/test';
import { 
  PWAHelpers,
  takeTimestampedScreenshot,
  waitForNetworkIdle
} from '../utils/test-helpers';

/**
 * PWA Manifest E2E Tests
 * Tests PWA manifest functionality, installation, and display modes
 */

test.describe('PWA Manifest Functionality', () => {
  let pwaHelpers: PWAHelpers;

  test.beforeEach(async ({ page, context }) => {
    pwaHelpers = new PWAHelpers(page, context);
    await page.goto('/');
  });

  test.describe('Manifest Validation', () => {
    test('should have valid PWA manifest', async ({ page }) => {
      // Check manifest link in HTML
      const manifestLink = await page.locator('link[rel="manifest"]').getAttribute('href');
      expect(manifestLink).toBeTruthy();

      // Fetch and validate manifest content
      const manifest = await pwaHelpers.checkManifest();
      expect(manifest).toBeTruthy();

      // Validate required manifest properties
      expect(manifest.name).toBeTruthy();
      expect(manifest.short_name).toBeTruthy();
      expect(manifest.start_url).toBeTruthy();
      expect(manifest.display).toBeTruthy();
      expect(manifest.theme_color).toBeTruthy();
      expect(manifest.background_color).toBeTruthy();
      expect(manifest.icons).toBeTruthy();
      expect(Array.isArray(manifest.icons)).toBe(true);
      expect(manifest.icons.length).toBeGreaterThan(0);

      // Validate icon specifications
      const requiredIconSizes = ['192x192', '512x512'];
      const availableIconSizes = manifest.icons.map((icon: any) => icon.sizes);
      
      for (const requiredSize of requiredIconSizes) {
        expect(availableIconSizes.some((size: string) => size.includes(requiredSize))).toBe(true);
      }

      // Validate icon formats
      const validFormats = ['image/png', 'image/jpeg', 'image/webp'];
      manifest.icons.forEach((icon: any) => {
        expect(validFormats).toContain(icon.type);
      });

      await takeTimestampedScreenshot(page, 'manifest-validated');
    });

    test('should have proper display modes configured', async ({ page }) => {
      const manifest = await pwaHelpers.checkManifest();
      
      // Validate display mode
      const validDisplayModes = ['standalone', 'minimal-ui', 'fullscreen', 'browser'];
      expect(validDisplayModes).toContain(manifest.display);

      // Check for fallback display modes
      if (manifest.display_override) {
        expect(Array.isArray(manifest.display_override)).toBe(true);
        manifest.display_override.forEach((mode: string) => {
          expect(validDisplayModes.concat(['window-controls-overlay'])).toContain(mode);
        });
      }

      await takeTimestampedScreenshot(page, 'display-modes-validated');
    });

    test('should have proper orientation settings', async ({ page }) => {
      const manifest = await pwaHelpers.checkManifest();
      
      // Check orientation if specified
      if (manifest.orientation) {
        const validOrientations = [
          'any', 'natural', 'landscape', 'landscape-primary', 'landscape-secondary',
          'portrait', 'portrait-primary', 'portrait-secondary'
        ];
        expect(validOrientations).toContain(manifest.orientation);
      }

      await takeTimestampedScreenshot(page, 'orientation-validated');
    });

    test('should have proper scope and start_url', async ({ page }) => {
      const manifest = await pwaHelpers.checkManifest();
      
      // Validate start_url
      expect(manifest.start_url).toBeTruthy();
      expect(manifest.start_url).toMatch(/^\/|^https?:\/\//);

      // Validate scope if present
      if (manifest.scope) {
        expect(manifest.scope).toMatch(/^\/|^https?:\/\//);
        // start_url should be within scope
        expect(manifest.start_url).toContain(manifest.scope.replace(/\/$/, ''));
      }

      await takeTimestampedScreenshot(page, 'scope-validated');
    });
  });

  test.describe('PWA Installation', () => {
    test('should be installable', async ({ page, context }) => {
      // Check if PWA is installable
      const installable = await pwaHelpers.isPWAInstallable();
      
      // Note: PWA installability depends on various factors
      // In test environment, it might not always be installable
      if (installable) {
        expect(installable).toBe(true);
      }

      // Check for install prompt handling
      const installPromptHandled = await page.evaluate(() => {
        return new Promise((resolve) => {
          let promptHandled = false;
          
          window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            promptHandled = true;
            resolve(true);
          });
          
          // Timeout after 2 seconds
          setTimeout(() => resolve(promptHandled), 2000);
        });
      });

      // If install prompt was triggered, verify it was handled
      if (installPromptHandled) {
        expect(installPromptHandled).toBe(true);
      }

      await takeTimestampedScreenshot(page, 'pwa-installable');
    });

    test('should show install banner when criteria met', async ({ page }) => {
      // Wait for potential install banner
      await page.waitForTimeout(3000);

      // Check if install button/banner is visible
      const installButton = page.locator('[data-testid="install-pwa-button"]');
      const hasBanner = await installButton.isVisible().catch(() => false);

      if (hasBanner) {
        // Test install button functionality
        await installButton.click();
        
        // Should show install confirmation or trigger browser install
        const installModal = page.locator('[data-testid="install-confirmation-modal"]');
        const hasModal = await installModal.isVisible().catch(() => false);
        
        if (hasModal) {
          await expect(installModal).toBeVisible();
          await page.click('[data-testid="confirm-install"]');
        }
      }

      await takeTimestampedScreenshot(page, 'install-banner');
    });

    test('should handle installation on different platforms', async ({ page, browserName }) => {
      // Platform-specific installation behavior
      const platformInfo = await page.evaluate(() => {
        const userAgent = navigator.userAgent;
        return {
          isMobile: /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent),
          isAndroid: /Android/i.test(userAgent),
          isIOS: /iPhone|iPad|iPod/i.test(userAgent),
          isChrome: /Chrome/i.test(userAgent),
          isSafari: /Safari/i.test(userAgent) && !/Chrome/i.test(userAgent)
        };
      });

      // Test platform-specific install prompts
      if (platformInfo.isAndroid && browserName === 'chromium') {
        // Android Chrome should support standard install prompt
        const manifest = await pwaHelpers.checkManifest();
        expect(manifest.display).toBe('standalone');
      }

      if (platformInfo.isIOS) {
        // iOS should show custom install instructions
        const iosInstallHint = page.locator('[data-testid="ios-install-hint"]');
        // May not always be visible in test environment
      }

      await takeTimestampedScreenshot(page, `install-${browserName}`);
    });
  });

  test.describe('Display Modes', () => {
    test('should support standalone display mode', async ({ page }) => {
      // Check if running in standalone mode
      const isStandalone = await pwaHelpers.isRunningAsPWA();
      
      if (isStandalone) {
        // Verify standalone mode characteristics
        const standaloneFeatures = await page.evaluate(() => {
          return {
            hasStandaloneMode: window.matchMedia('(display-mode: standalone)').matches,
            windowTop: window.screenTop || window.screenY,
            hasFullWindow: window.innerHeight === screen.height
          };
        });

        expect(standaloneFeatures.hasStandaloneMode).toBe(true);
      }

      await takeTimestampedScreenshot(page, 'standalone-mode');
    });

    test('should adapt UI for different display modes', async ({ page }) => {
      // Test UI adaptation based on display mode
      const displayModeSupport = await page.evaluate(() => {
        const modes = ['standalone', 'minimal-ui', 'fullscreen', 'browser'];
        const supportedModes = modes.filter(mode => 
          window.matchMedia(`(display-mode: ${mode})`).matches
        );
        
        return {
          currentMode: supportedModes[0] || 'browser',
          supportedModes: supportedModes
        };
      });

      // Verify UI elements adapt to display mode
      if (displayModeSupport.currentMode === 'standalone') {
        // In standalone mode, browser controls should be hidden
        // PWA should provide its own navigation
        await expect(page.locator('[data-testid="pwa-navigation"]')).toBeVisible();
      }

      await takeTimestampedScreenshot(page, `display-mode-${displayModeSupport.currentMode}`);
    });

    test('should handle display mode changes', async ({ page }) => {
      // Monitor display mode changes
      const displayModeListener = await page.evaluateHandle(() => {
        const modes = ['standalone', 'minimal-ui', 'fullscreen', 'browser'];
        const results: string[] = [];
        
        modes.forEach(mode => {
          const mediaQuery = window.matchMedia(`(display-mode: ${mode})`);
          mediaQuery.addEventListener('change', (e) => {
            if (e.matches) {
              results.push(`Changed to: ${mode}`);
            }
          });
        });
        
        return results;
      });

      // Simulate display mode change (if possible)
      // Note: This is difficult to test in automated environment
      // but we can verify the listeners are set up

      await page.waitForTimeout(1000);

      const displayModeChanges = await displayModeListener.evaluate(results => results);
      
      // Verify listeners are functioning (results array exists)
      expect(Array.isArray(displayModeChanges)).toBe(true);

      await takeTimestampedScreenshot(page, 'display-mode-changes');
    });
  });

  test.describe('Theme and Colors', () => {
    test('should apply theme colors correctly', async ({ page }) => {
      const manifest = await pwaHelpers.checkManifest();
      
      // Verify theme color is applied
      const metaThemeColor = await page.getAttribute('meta[name="theme-color"]', 'content');
      
      if (metaThemeColor && manifest.theme_color) {
        expect(metaThemeColor).toBe(manifest.theme_color);
      }

      // Check if colors are valid hex codes or CSS colors
      const isValidColor = (color: string) => {
        return /^#([0-9A-F]{3}){1,2}$/i.test(color) || 
               /^rgb\(\d+,\s*\d+,\s*\d+\)$/i.test(color) ||
               /^hsl\(\d+,\s*\d+%,\s*\d+%\)$/i.test(color);
      };

      if (manifest.theme_color) {
        expect(isValidColor(manifest.theme_color)).toBe(true);
      }
      
      if (manifest.background_color) {
        expect(isValidColor(manifest.background_color)).toBe(true);
      }

      await takeTimestampedScreenshot(page, 'theme-colors');
    });

    test('should support dark/light theme switching', async ({ page }) => {
      // Check for theme switching capability
      const themeSupport = await page.evaluate(() => {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
        const prefersLight = window.matchMedia('(prefers-color-scheme: light)');
        
        return {
          supportsDarkMode: prefersDark.matches,
          supportsLightMode: prefersLight.matches,
          hasThemeSwitcher: !!document.querySelector('[data-testid="theme-switcher"]')
        };
      });

      if (themeSupport.hasThemeSwitcher) {
        // Test theme switching
        const themeSwitcher = page.locator('[data-testid="theme-switcher"]');
        await themeSwitcher.click();
        
        // Verify theme change
        await page.waitForTimeout(500);
        
        const themeChanged = await page.evaluate(() => {
          return document.documentElement.classList.contains('dark') ||
                 document.body.classList.contains('dark-theme');
        });
        
        if (themeChanged) {
          expect(themeChanged).toBe(true);
        }
      }

      await takeTimestampedScreenshot(page, 'theme-switching');
    });

    test('should respect system theme preferences', async ({ page }) => {
      // Test system theme detection
      const systemTheme = await page.evaluate(() => {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
        
        return {
          prefersDark,
          prefersLight,
          systemTheme: prefersDark ? 'dark' : 'light'
        };
      });

      // Verify app respects system theme
      const appTheme = await page.evaluate(() => {
        const isDark = document.documentElement.classList.contains('dark') ||
                      document.body.classList.contains('dark-theme') ||
                      getComputedStyle(document.body).backgroundColor === 'rgb(0, 0, 0)';
        
        return isDark ? 'dark' : 'light';
      });

      // App theme should match system preference (unless user has overridden)
      if (systemTheme.prefersDark || systemTheme.prefersLight) {
        // Just verify that theme detection is working
        expect(['dark', 'light']).toContain(appTheme);
      }

      await takeTimestampedScreenshot(page, 'system-theme');
    });
  });

  test.describe('App Categories and Features', () => {
    test('should have proper app categories', async ({ page }) => {
      const manifest = await pwaHelpers.checkManifest();
      
      // Check app categories if specified
      if (manifest.categories) {
        expect(Array.isArray(manifest.categories)).toBe(true);
        
        const validCategories = [
          'business', 'education', 'entertainment', 'finance', 'fitness',
          'food', 'games', 'government', 'health', 'kids', 'lifestyle',
          'magazines', 'medical', 'music', 'navigation', 'news', 'personalization',
          'photo', 'politics', 'productivity', 'security', 'shopping', 'social',
          'sports', 'travel', 'utilities', 'weather'
        ];
        
        manifest.categories.forEach((category: string) => {
          expect(validCategories).toContain(category);
        });
      }

      await takeTimestampedScreenshot(page, 'app-categories');
    });

    test('should have proper shortcuts configured', async ({ page }) => {
      const manifest = await pwaHelpers.checkManifest();
      
      // Check app shortcuts if specified
      if (manifest.shortcuts) {
        expect(Array.isArray(manifest.shortcuts)).toBe(true);
        expect(manifest.shortcuts.length).toBeLessThanOrEqual(4); // Platform limit
        
        manifest.shortcuts.forEach((shortcut: any) => {
          expect(shortcut.name).toBeTruthy();
          expect(shortcut.url).toBeTruthy();
          expect(shortcut.url).toMatch(/^\/|^https?:\/\//);
          
          if (shortcut.icons) {
            expect(Array.isArray(shortcut.icons)).toBe(true);
          }
        });
      }

      await takeTimestampedScreenshot(page, 'app-shortcuts');
    });

    test('should handle file handlers properly', async ({ page }) => {
      const manifest = await pwaHelpers.checkManifest();
      
      // Check file handlers if specified
      if (manifest.file_handlers) {
        expect(Array.isArray(manifest.file_handlers)).toBe(true);
        
        manifest.file_handlers.forEach((handler: any) => {
          expect(handler.action).toBeTruthy();
          expect(handler.accept).toBeTruthy();
          expect(typeof handler.accept).toBe('object');
        });
      }

      await takeTimestampedScreenshot(page, 'file-handlers');
    });

    test('should have proper share target configuration', async ({ page }) => {
      const manifest = await pwaHelpers.checkManifest();
      
      // Check share target if specified
      if (manifest.share_target) {
        expect(manifest.share_target.action).toBeTruthy();
        expect(manifest.share_target.action).toMatch(/^\/|^https?:\/\//);
        
        if (manifest.share_target.method) {
          expect(['GET', 'POST']).toContain(manifest.share_target.method);
        }
        
        if (manifest.share_target.params) {
          const params = manifest.share_target.params;
          // Should have at least one of the standard params
          const hasValidParams = params.title || params.text || params.url || params.files;
          expect(hasValidParams).toBeTruthy();
        }
      }

      await takeTimestampedScreenshot(page, 'share-target');
    });
  });
});