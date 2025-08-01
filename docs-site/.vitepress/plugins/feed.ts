import type { Plugin } from 'vite'
import { Feed } from 'feed'
import { writeFileSync } from 'fs'
import { resolve } from 'path'

/**
 * RSS/Atom feed generation plugin for LeanVibe Agent Hive documentation
 */
export function feedPlugin(): Plugin {
  return {
    name: 'leanvibe-feed',
    buildEnd: async (config: any) => {
      await generateFeeds(config)
    }
  }
}

/**
 * Generate RSS and Atom feeds for documentation updates
 */
async function generateFeeds(config: any) {
  console.log('📡 Generating RSS/Atom feeds...')
  
  const feed = new Feed({
    title: 'LeanVibe Agent Hive Documentation',
    description: 'Latest updates, tutorials, and announcements for autonomous software development',
    id: 'https://docs.leanvibe.dev/',
    link: 'https://docs.leanvibe.dev/',
    language: 'en',
    image: 'https://docs.leanvibe.dev/images/logo.svg',
    favicon: 'https://docs.leanvibe.dev/favicon.ico',
    copyright: 'Copyright © 2025 LeanVibe. MIT Licensed.',
    updated: new Date(),
    generator: 'LeanVibe Agent Hive VitePress',
    feedLinks: {
      rss2: 'https://docs.leanvibe.dev/feed.xml',
      atom: 'https://docs.leanvibe.dev/feed.atom',
      json: 'https://docs.leanvibe.dev/feed.json'
    },
    author: {
      name: 'LeanVibe Team',
      email: 'team@leanvibe.dev',
      link: 'https://leanvibe.dev'
    }
  })
  
  // Add recent documentation updates
  const recentUpdates = await getRecentUpdates()
  
  recentUpdates.forEach(update => {
    feed.addItem({
      title: update.title,
      id: update.url,
      link: update.url,
      description: update.description,
      content: update.content,
      author: [
        {
          name: update.author,
          email: 'team@leanvibe.dev'
        }
      ],
      date: update.date,
      category: update.categories.map(cat => ({ name: cat }))
    })
  })
  
  // Write feed files
  const distPath = resolve(config.outDir || '.vitepress/dist')
  
  try {
    writeFileSync(resolve(distPath, 'feed.xml'), feed.rss2())
    writeFileSync(resolve(distPath, 'feed.atom'), feed.atom1())
    writeFileSync(resolve(distPath, 'feed.json'), feed.json1())
    
    console.log('✅ Generated RSS/Atom feeds successfully')
  } catch (error) {
    console.warn('⚠️ Failed to generate feeds:', error)
  }
}

/**
 * Get recent documentation updates
 */
async function getRecentUpdates(): Promise<FeedItem[]> {
  // In a real implementation, this would:
  // 1. Parse git commits for documentation changes
  // 2. Extract frontmatter from markdown files
  // 3. Track release notes and announcements
  // 4. Monitor community contributions
  
  // Mock data for demonstration
  return [
    {
      title: 'New Interactive Code Playground Released',
      url: 'https://docs.leanvibe.dev/learn/fundamentals/commands',
      description: 'Try autonomous development workflows directly in your browser with our new interactive code playground.',
      content: `
        <p>We're excited to announce the release of our interactive code playground, allowing you to test custom commands and multi-agent workflows directly in the documentation.</p>
        
        <h3>Key Features:</h3>
        <ul>
          <li>Real-time code execution</li>
          <li>Multiple command templates</li>
          <li>Live agent coordination demos</li>
          <li>Syntax highlighting and validation</li>
        </ul>
        
        <p>The playground is now available throughout our learning guides and API documentation.</p>
      `,
      author: 'LeanVibe Team',
      date: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000), // 1 day ago
      categories: ['Features', 'Documentation', 'Interactive']
    },
    {
      title: 'Multi-Agent Coordination Guide Updated',
      url: 'https://docs.leanvibe.dev/learn/fundamentals/coordination',
      description: 'Comprehensive updates to our multi-agent coordination guide with new examples and best practices.',
      content: `
        <p>We've significantly updated our multi-agent coordination guide with new examples, best practices, and troubleshooting tips.</p>
        
        <h3>What's New:</h3>
        <ul>
          <li>Advanced conflict resolution strategies</li>
          <li>Performance optimization techniques</li>
          <li>Real-world use case examples</li>
          <li>Interactive agent visualization</li>
        </ul>
        
        <p>The guide now includes live demonstrations of agent coordination in action.</p>
      `,
      author: 'Sarah Chen',
      date: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000), // 3 days ago
      categories: ['Documentation', 'Multi-Agent', 'Coordination']
    },
    {
      title: 'API Reference v2.0 Released',
      url: 'https://docs.leanvibe.dev/api/',
      description: 'Complete API reference documentation with interactive examples and SDK information.',
      content: `
        <p>Our API reference has been completely rewritten with v2.0, featuring interactive examples and comprehensive SDK documentation.</p>
        
        <h3>Improvements:</h3>
        <ul>
          <li>Interactive API explorer</li>
          <li>Code examples in multiple languages</li>
          <li>Real-time testing environment</li>
          <li>SDK documentation for Python, JavaScript, and more</li>
        </ul>
        
        <p>Developers can now test API endpoints directly from the documentation.</p>
      `,
      author: 'Mike Rodriguez',
      date: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5 days ago
      categories: ['API', 'Documentation', 'SDK']
    },
    {
      title: 'Enterprise Deployment Guide',
      url: 'https://docs.leanvibe.dev/enterprise/deployment/',
      description: 'New comprehensive guide for enterprise deployment and scaling strategies.',
      content: `
        <p>We've published a comprehensive enterprise deployment guide covering production-ready configurations and scaling strategies.</p>
        
        <h3>Covered Topics:</h3>
        <ul>
          <li>Production architecture patterns</li>
          <li>Security and compliance requirements</li>
          <li>Horizontal scaling strategies</li>
          <li>Monitoring and observability setup</li>
        </ul>
        
        <p>Enterprise customers can now deploy with confidence using our battle-tested recommendations.</p>
      `,
      author: 'David Park',
      date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
      categories: ['Enterprise', 'Deployment', 'Production']
    },
    {
      title: 'Community Showcase: E-commerce Automation',
      url: 'https://docs.leanvibe.dev/examples/showcase/featured',
      description: 'Community member builds complete e-commerce platform using autonomous agents.',
      content: `
        <p>Community member Anna Thompson has built an impressive e-commerce automation system using LeanVibe Agent Hive.</p>
        
        <h3>Project Highlights:</h3>
        <ul>
          <li>Automated product catalog management</li>
          <li>Dynamic pricing optimization</li>
          <li>Customer service automation</li>
          <li>Inventory management workflows</li>
        </ul>
        
        <p>The project demonstrates the power of multi-agent coordination in real-world business applications.</p>
      `,
      author: 'Anna Thompson',
      date: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000), // 10 days ago
      categories: ['Community', 'Showcase', 'E-commerce']
    }
  ]
}

/**
 * Feed item interface
 */
interface FeedItem {
  title: string
  url: string
  description: string
  content: string
  author: string
  date: Date
  categories: string[]
}

/**
 * Generate sitemap.xml for SEO
 */
export async function generateSitemap(config: any) {
  console.log('🗺️ Generating sitemap...')
  
  const baseUrl = 'https://docs.leanvibe.dev'
  const pages = await getDocumentationPages()
  
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map(page => `  <url>
    <loc>${baseUrl}${page.url}</loc>
    <lastmod>${page.lastmod}</lastmod>
    <changefreq>${page.changefreq}</changefreq>
    <priority>${page.priority}</priority>
  </url>`).join('\n')}
</urlset>`
  
  const distPath = resolve(config.outDir || '.vitepress/dist')
  
  try {
    writeFileSync(resolve(distPath, 'sitemap.xml'), sitemap)
    console.log('✅ Generated sitemap.xml successfully')
  } catch (error) {
    console.warn('⚠️ Failed to generate sitemap:', error)
  }
}

/**
 * Get all documentation pages for sitemap
 */
async function getDocumentationPages(): Promise<SitemapPage[]> {
  // In a real implementation, this would scan all markdown files
  // and extract URLs and metadata
  
  return [
    {
      url: '/',
      lastmod: new Date().toISOString().split('T')[0],
      changefreq: 'weekly',
      priority: '1.0'
    },
    {
      url: '/learn/',
      lastmod: new Date().toISOString().split('T')[0],
      changefreq: 'weekly',
      priority: '0.9'
    },
    {
      url: '/api/',
      lastmod: new Date().toISOString().split('T')[0],
      changefreq: 'weekly',
      priority: '0.9'
    },
    {
      url: '/examples/',
      lastmod: new Date().toISOString().split('T')[0],
      changefreq: 'weekly',
      priority: '0.8'
    },
    {
      url: '/community/',
      lastmod: new Date().toISOString().split('T')[0],
      changefreq: 'weekly',
      priority: '0.8'
    },
    {
      url: '/enterprise/',
      lastmod: new Date().toISOString().split('T')[0],
      changefreq: 'monthly',
      priority: '0.7'
    }
  ]
}

interface SitemapPage {
  url: string
  lastmod: string
  changefreq: string
  priority: string
}