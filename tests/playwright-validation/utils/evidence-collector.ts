import { Page } from '@playwright/test';
import * as fs from 'fs/promises';
import * as path from 'path';

/**
 * Evidence Collector Utility
 * 
 * Collects comprehensive evidence during Playwright tests to build trust and validate claims.
 * Captures screenshots, API responses, system states, performance metrics, and other artifacts.
 */

export class EvidenceCollector {
  private page: Page;
  private testCategory: string;
  private collectionId: string;
  private evidenceDir: string;
  private startTime: number;
  private evidence: Array<{
    timestamp: string;
    type: string;
    name: string;
    data: any;
    filePath?: string;
  }>;

  constructor(page: Page, testCategory: string) {
    this.page = page;
    this.testCategory = testCategory;
    this.collectionId = '';
    this.evidenceDir = '';
    this.startTime = 0;
    this.evidence = [];
  }

  async startCollection(collectionId: string): Promise<void> {
    this.collectionId = collectionId;
    this.startTime = Date.now();
    
    // Create evidence directory structure
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.evidenceDir = path.join(
      process.cwd(),
      'reports',
      'evidence',
      this.testCategory,
      `${timestamp}-${collectionId}`
    );
    
    await fs.mkdir(this.evidenceDir, { recursive: true });
    
    // Create subdirectories
    await fs.mkdir(path.join(this.evidenceDir, 'screenshots'), { recursive: true });
    await fs.mkdir(path.join(this.evidenceDir, 'api-responses'), { recursive: true });
    await fs.mkdir(path.join(this.evidenceDir, 'data'), { recursive: true });
    await fs.mkdir(path.join(this.evidenceDir, 'logs'), { recursive: true });
    
    console.log(`📊 Evidence collection started: ${this.evidenceDir}`);
  }

  async captureScreenshot(name: string, options?: {
    fullPage?: boolean;
    clip?: { x: number; y: number; width: number; height: number };
  }): Promise<string> {
    const timestamp = new Date().toISOString();
    const filename = `${name.replace(/[^a-zA-Z0-9-]/g, '_')}.png`;
    const screenshotPath = path.join(this.evidenceDir, 'screenshots', filename);
    
    await this.page.screenshot({
      path: screenshotPath,
      fullPage: options?.fullPage || false,
      clip: options?.clip
    });
    
    this.evidence.push({
      timestamp,
      type: 'screenshot',
      name: name,
      data: { path: screenshotPath, filename },
      filePath: screenshotPath
    });
    
    return screenshotPath;
  }

  async captureApiResponse(endpoint: string, responseData: any): Promise<void> {
    const timestamp = new Date().toISOString();
    const filename = `${endpoint.replace(/[^a-zA-Z0-9-]/g, '_')}.json`;
    const responsePath = path.join(this.evidenceDir, 'api-responses', filename);
    
    const evidenceData = {
      timestamp,
      endpoint,
      responseData,
      metadata: {
        userAgent: await this.page.evaluate(() => navigator.userAgent),
        url: this.page.url(),
        viewport: await this.page.viewportSize()
      }
    };
    
    await fs.writeFile(responsePath, JSON.stringify(evidenceData, null, 2));
    
    this.evidence.push({
      timestamp,
      type: 'api_response',
      name: endpoint,
      data: evidenceData,
      filePath: responsePath
    });
  }

  async captureData(name: string, data: any): Promise<void> {
    const timestamp = new Date().toISOString();
    const filename = `${name.replace(/[^a-zA-Z0-9-]/g, '_')}.json`;
    const dataPath = path.join(this.evidenceDir, 'data', filename);
    
    const evidenceData = {
      timestamp,
      name,
      data,
      metadata: {
        testCategory: this.testCategory,
        collectionId: this.collectionId,
        pageUrl: this.page.url(),
        testDuration: Date.now() - this.startTime
      }
    };
    
    await fs.writeFile(dataPath, JSON.stringify(evidenceData, null, 2));
    
    this.evidence.push({
      timestamp,
      type: 'data',
      name: name,
      data: evidenceData,
      filePath: dataPath
    });
  }

  async capturePageSource(name: string): Promise<string> {
    const timestamp = new Date().toISOString();
    const filename = `${name.replace(/[^a-zA-Z0-9-]/g, '_')}.html`;
    const sourcePath = path.join(this.evidenceDir, 'data', filename);
    
    const pageSource = await this.page.content();
    
    const evidenceData = {
      timestamp,
      name,
      url: this.page.url(),
      title: await this.page.title(),
      source: pageSource
    };
    
    await fs.writeFile(sourcePath, JSON.stringify(evidenceData, null, 2));
    
    this.evidence.push({
      timestamp,
      type: 'page_source',
      name: name,
      data: { sourceLength: pageSource.length, url: this.page.url() },
      filePath: sourcePath
    });
    
    return sourcePath;
  }

  async captureConsoleLog(): Promise<void> {
    const timestamp = new Date().toISOString();
    const filename = `console-log-${timestamp.replace(/[:.]/g, '-')}.json`;
    const logPath = path.join(this.evidenceDir, 'logs', filename);
    
    // Capture console messages
    const consoleLogs: any[] = [];
    
    this.page.on('console', msg => {
      consoleLogs.push({
        timestamp: new Date().toISOString(),
        type: msg.type(),
        text: msg.text(),
        location: msg.location()
      });
    });
    
    // Wait a moment to collect logs
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    if (consoleLogs.length > 0) {
      await fs.writeFile(logPath, JSON.stringify(consoleLogs, null, 2));
      
      this.evidence.push({
        timestamp,
        type: 'console_log',
        name: 'console_messages',
        data: { logCount: consoleLogs.length, logs: consoleLogs },
        filePath: logPath
      });
    }
  }

  async captureNetworkActivity(): Promise<void> {
    const timestamp = new Date().toISOString();
    const filename = `network-activity-${timestamp.replace(/[:.]/g, '-')}.json`;
    const networkPath = path.join(this.evidenceDir, 'logs', filename);
    
    const networkActivity: any[] = [];
    
    this.page.on('request', request => {
      networkActivity.push({
        timestamp: new Date().toISOString(),
        type: 'request',
        method: request.method(),
        url: request.url(),
        headers: request.headers(),
        resourceType: request.resourceType()
      });
    });
    
    this.page.on('response', response => {
      networkActivity.push({
        timestamp: new Date().toISOString(),
        type: 'response',
        status: response.status(),
        url: response.url(),
        headers: response.headers(),
        ok: response.ok()
      });
    });
    
    // Wait for network activity
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    if (networkActivity.length > 0) {
      await fs.writeFile(networkPath, JSON.stringify(networkActivity, null, 2));
      
      this.evidence.push({
        timestamp,
        type: 'network_activity',
        name: 'network_requests_responses',
        data: { activityCount: networkActivity.length, activity: networkActivity },
        filePath: networkPath
      });
    }
  }

  async capturePerformanceMetrics(): Promise<void> {
    const timestamp = new Date().toISOString();
    
    // Capture web vitals and performance metrics
    const performanceMetrics = await this.page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const paint = performance.getEntriesByType('paint');
      
      return {
        timing: {
          domContentLoaded: navigation?.domContentLoadedEventEnd - navigation?.fetchStart,
          loadComplete: navigation?.loadEventEnd - navigation?.fetchStart,
          firstPaint: paint.find(p => p.name === 'first-paint')?.startTime,
          firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime
        },
        memory: (performance as any).memory ? {
          usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
          totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
          jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit
        } : null,
        resources: performance.getEntriesByType('resource').length
      };
    });
    
    this.evidence.push({
      timestamp,
      type: 'performance_metrics',
      name: 'page_performance',
      data: performanceMetrics
    });
  }

  async getScreenshotPath(name: string): Promise<string> {
    const filename = `${name.replace(/[^a-zA-Z0-9-]/g, '_')}.png`;
    return path.join(this.evidenceDir, 'screenshots', filename);
  }

  async finishCollection(): Promise<void> {
    const timestamp = new Date().toISOString();
    const duration = Date.now() - this.startTime;
    
    // Capture final performance metrics
    await this.capturePerformanceMetrics();
    
    // Generate collection summary
    const summary = {
      collectionId: this.collectionId,
      testCategory: this.testCategory,
      startTime: new Date(this.startTime).toISOString(),
      endTime: timestamp,
      duration: duration,
      evidenceCount: this.evidence.length,
      evidence: this.evidence,
      evidenceByType: this.evidence.reduce((acc, item) => {
        acc[item.type] = (acc[item.type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    };
    
    // Save collection summary
    const summaryPath = path.join(this.evidenceDir, 'collection-summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2));
    
    // Generate HTML evidence report
    await this.generateHtmlReport(summary);
    
    console.log(`✅ Evidence collection completed: ${this.evidence.length} items in ${duration}ms`);
    console.log(`📋 Evidence directory: ${this.evidenceDir}`);
  }

  private async generateHtmlReport(summary: any): Promise<void> {
    const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evidence Report - ${summary.collectionId}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 20px; }
        .section { margin: 20px 0; }
        .evidence-item { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #007bff; }
        .screenshot { max-width: 300px; border: 1px solid #ddd; border-radius: 4px; }
        .metadata { font-size: 0.9em; color: #666; }
        .summary-stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat-card { background: #e9ecef; padding: 15px; border-radius: 4px; text-align: center; flex: 1; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Evidence Report: ${summary.collectionId}</h1>
            <p class="metadata">
                Category: ${summary.testCategory} | 
                Duration: ${(summary.duration / 1000).toFixed(2)}s | 
                Generated: ${summary.endTime}
            </p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <h3>${summary.evidenceCount}</h3>
                <p>Total Evidence Items</p>
            </div>
            <div class="stat-card">
                <h3>${summary.evidenceByType.screenshot || 0}</h3>
                <p>Screenshots</p>
            </div>
            <div class="stat-card">
                <h3>${summary.evidenceByType.api_response || 0}</h3>
                <p>API Responses</p>
            </div>
            <div class="stat-card">
                <h3>${summary.evidenceByType.data || 0}</h3>
                <p>Data Captures</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Evidence Items</h2>
            ${summary.evidence.map((item: any, index: number) => `
                <div class="evidence-item">
                    <h3>${item.name} (${item.type})</h3>
                    <p class="metadata">Timestamp: ${item.timestamp}</p>
                    ${item.type === 'screenshot' ? `
                        <img src="${path.basename(item.filePath || '')}" alt="${item.name}" class="screenshot">
                    ` : `
                        <pre style="background: #f1f1f1; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.9em;">
${JSON.stringify(item.data, null, 2).substring(0, 500)}${JSON.stringify(item.data, null, 2).length > 500 ? '...' : ''}
                        </pre>
                    `}
                </div>
            `).join('')}
        </div>
        
        <div class="section">
            <h2>Collection Summary</h2>
            <pre style="background: #f8f9fa; padding: 20px; border-radius: 4px; overflow-x: auto;">
${JSON.stringify(summary, null, 2)}
            </pre>
        </div>
    </div>
</body>
</html>
    `;
    
    const reportPath = path.join(this.evidenceDir, 'evidence-report.html');
    await fs.writeFile(reportPath, htmlContent);
  }
}