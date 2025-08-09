import { test, expect } from '@playwright/test'

// Minimal smoke: verify shell loads and title shows HiveOps
// Assumes backend is running on 8000 (as per README quickstart)

test('dashboard shell loads and shows HiveOps title', async ({ page }) => {
  await page.goto('/')
  // Header title component
  await expect(page.locator('header')).toBeVisible()
  await expect(page.locator('text=HiveOps')).toBeVisible()
  // Also assert one deterministic metric label exists (static UI element)
  await expect(page.locator('text=Agent Dashboard')).toBeVisible()
})
