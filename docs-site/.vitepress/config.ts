import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'LeanVibe Agent Hive',
  description: 'Autonomous Software Development Platform - Transform development with AI agents that actually work',
  
  // Theme configuration
  themeConfig: {
    logo: '/images/logo.svg',
    siteTitle: 'LeanVibe Agent Hive',
    
    // Navigation
    nav: [
      { 
        text: 'Learn', 
        link: '/learn/',
        activeMatch: '/learn/'
      },
      { 
        text: 'API Reference', 
        link: '/api/',
        activeMatch: '/api/'
      },
      { 
        text: 'Examples', 
        link: '/examples/',
        activeMatch: '/examples/'
      },
      { 
        text: 'Community', 
        link: '/community/',
        activeMatch: '/community/'
      },
      {
        text: 'Enterprise',
        items: [
          { text: 'Overview', link: '/enterprise/' },
          { text: 'Security', link: '/enterprise/security/' },
          { text: 'Deployment', link: '/enterprise/deployment/' },
          { text: 'ROI Calculator', link: '/enterprise/roi/' }
        ]
      },
      {
        text: 'Live Demo',
        link: '/demo',
        target: '_blank'
      }
    ],

    // Sidebar configuration
    sidebar: {
      '/learn/': [
        {
          text: 'Getting Started',
          collapsed: false,
          items: [
            { text: 'Introduction', link: '/learn/' },
            { text: 'Quick Setup', link: '/learn/getting-started/setup' },
            { text: 'First Demo', link: '/learn/getting-started/first-demo' },
            { text: 'Core Concepts', link: '/learn/getting-started/concepts' }
          ]
        },
        {
          text: 'Fundamentals',
          collapsed: false,
          items: [
            { text: 'Agent Coordination', link: '/learn/fundamentals/coordination' },
            { text: 'Custom Commands', link: '/learn/fundamentals/commands' },
            { text: 'Workflow Design', link: '/learn/fundamentals/workflows' },
            { text: 'System Monitoring', link: '/learn/fundamentals/monitoring' }
          ]
        },
        {
          text: 'Advanced Topics',
          collapsed: true,
          items: [
            { text: 'Enterprise Deployment', link: '/learn/advanced/deployment' },
            { text: 'Custom Integrations', link: '/learn/advanced/integrations' },
            { text: 'Performance Tuning', link: '/learn/advanced/performance' },
            { text: 'Troubleshooting', link: '/learn/advanced/troubleshooting' }
          ]
        },
        {
          text: 'Mastery',
          collapsed: true,
          items: [
            { text: 'Architecture Deep Dive', link: '/learn/mastery/architecture' },
            { text: 'Contributing', link: '/learn/mastery/contributing' },
            { text: 'Plugin Development', link: '/learn/mastery/plugins' },
            { text: 'Community Leadership', link: '/learn/mastery/leadership' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Overview',
          items: [
            { text: 'Introduction', link: '/api/' },
            { text: 'Authentication', link: '/api/auth' },
            { text: 'Rate Limits', link: '/api/limits' },
            { text: 'Error Handling', link: '/api/errors' }
          ]
        },
        {
          text: 'Custom Commands',
          items: [
            { text: 'Overview', link: '/api/commands/' },
            { text: 'Command Creation', link: '/api/commands/creation' },
            { text: 'Execution', link: '/api/commands/execution' },
            { text: 'Monitoring', link: '/api/commands/monitoring' }
          ]
        },
        {
          text: 'Multi-Agent Coordination',
          items: [
            { text: 'Overview', link: '/api/coordination/' },
            { text: 'Agent Management', link: '/api/coordination/agents' },
            { text: 'Project Coordination', link: '/api/coordination/projects' },
            { text: 'Conflict Resolution', link: '/api/coordination/conflicts' }
          ]
        },
        {
          text: 'Observability',
          items: [
            { text: 'Metrics API', link: '/api/observability/metrics' },
            { text: 'Events API', link: '/api/observability/events' },
            { text: 'WebSocket Streams', link: '/api/observability/websockets' }
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Use Cases',
          items: [
            { text: 'Overview', link: '/examples/' },
            { text: 'Web Development', link: '/examples/use-cases/web-dev' },
            { text: 'API Development', link: '/examples/use-cases/api-dev' },
            { text: 'DevOps Automation', link: '/examples/use-cases/devops' },
            { text: 'Data Processing', link: '/examples/use-cases/data' }
          ]
        },
        {
          text: 'Integrations',
          items: [
            { text: 'GitHub Integration', link: '/examples/integrations/github' },
            { text: 'CI/CD Pipelines', link: '/examples/integrations/cicd' },
            { text: 'Monitoring Systems', link: '/examples/integrations/monitoring' },
            { text: 'Custom Tools', link: '/examples/integrations/custom' }
          ]
        },
        {
          text: 'Community Showcase',
          items: [
            { text: 'Featured Projects', link: '/examples/showcase/featured' },
            { text: 'Success Stories', link: '/examples/showcase/stories' },
            { text: 'Case Studies', link: '/examples/showcase/studies' }
          ]
        }
      ],
      '/community/': [
        {
          text: 'Getting Involved',
          items: [
            { text: 'Overview', link: '/community/' },
            { text: 'Code of Conduct', link: '/community/code-of-conduct' },
            { text: 'How to Contribute', link: '/community/contributing' },
            { text: 'Governance', link: '/community/governance' }
          ]
        },
        {
          text: 'Resources',
          items: [
            { text: 'Discord Community', link: '/community/discord' },
            { text: 'GitHub Discussions', link: '/community/discussions' },
            { text: 'Office Hours', link: '/community/office-hours' },
            { text: 'Events & Webinars', link: '/community/events' }
          ]
        },
        {
          text: 'Support',
          items: [
            { text: 'Getting Help', link: '/community/support' },
            { text: 'Bug Reports', link: '/community/bugs' },
            { text: 'Feature Requests', link: '/community/features' },
            { text: 'Security Reports', link: '/community/security' }
          ]
        }
      ],
      '/enterprise/': [
        {
          text: 'Overview',
          items: [
            { text: 'Enterprise Features', link: '/enterprise/' },
            { text: 'Business Value', link: '/enterprise/value' },
            { text: 'ROI Calculator', link: '/enterprise/roi' },
            { text: 'Success Stories', link: '/enterprise/stories' }
          ]
        },
        {
          text: 'Security & Compliance',
          items: [
            { text: 'Security Overview', link: '/enterprise/security/' },
            { text: 'Data Protection', link: '/enterprise/security/data' },
            { text: 'Access Control', link: '/enterprise/security/access' },
            { text: 'Audit & Compliance', link: '/enterprise/security/audit' }
          ]
        },
        {
          text: 'Deployment',
          items: [
            { text: 'Deployment Options', link: '/enterprise/deployment/' },
            { text: 'On-Premises', link: '/enterprise/deployment/on-premises' },
            { text: 'Cloud Deployment', link: '/enterprise/deployment/cloud' },
            { text: 'Hybrid Solutions', link: '/enterprise/deployment/hybrid' }
          ]
        }
      ]
    },

    // Social links
    socialLinks: [
      { icon: 'github', link: 'https://github.com/LeanVibe/bee-hive' },
      { icon: 'discord', link: 'https://discord.gg/leanvibe' },
      { icon: 'twitter', link: 'https://twitter.com/leanvibe' }
    ],

    // Footer
    footer: {
      message: 'Built with ❤️ by the LeanVibe team • Powered by autonomous AI agents',
      copyright: 'Copyright © 2025 LeanVibe. MIT Licensed.'
    },

    // Edit link
    editLink: {
      pattern: 'https://github.com/LeanVibe/bee-hive/edit/main/docs-site/:path',
      text: 'Edit this page on GitHub'
    },

    // Last update
    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },

    // Search
    search: {
      provider: 'local',
      options: {
        miniSearch: {
          searchOptions: {
            fuzzy: 0.2,
            prefix: true,
            boost: { title: 4, text: 2, titles: 1 }
          }
        }
      }
    },

    // Carbon ads (optional)
    carbonAds: {
      code: 'your-carbon-code',
      placement: 'your-carbon-placement'
    }
  },

  // Site configuration
  lang: 'en-US',
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    ['link', { rel: 'icon', type: 'image/png', href: '/favicon.png' }],
    ['meta', { name: 'theme-color', content: '#646cff' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'en' }],
    ['meta', { name: 'og:site_name', content: 'LeanVibe Agent Hive' }],
    ['meta', { name: 'og:image', content: '/images/og-image.jpg' }],
    ['script', { src: 'https://cdn.usefathom.com/script.js', 'data-site': 'ABCDEFGH', defer: '' }]
  ],

  // Clean URLs
  cleanUrls: true,

  // Markdown configuration
  markdown: {
    lineNumbers: true
  },

  // Build configuration
  vite: {
    optimizeDeps: {
      include: ['vue', 'algoliasearch']
    },
    build: {
      chunkSizeWarningLimit: 1000
    }
  },

  // Site map
  sitemap: {
    hostname: 'https://docs.leanvibe.dev'
  }
})