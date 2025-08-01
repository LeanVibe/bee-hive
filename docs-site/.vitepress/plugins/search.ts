import type { Plugin } from 'vite'

/**
 * VitePress search plugin for LeanVibe Agent Hive documentation
 * Provides enhanced search capabilities with AI-powered suggestions
 */
export function searchPlugin(): Plugin {
  return {
    name: 'leanvibe-search',
    configResolved(config) {
      // Add search configuration
      config.define = config.define || {}
      config.define.__SEARCH_ENABLED__ = true
    },
    buildStart() {
      console.log('🔍 Initializing search indexing...')
    },
    generateBundle() {
      console.log('📚 Building search index...')
      // In a real implementation, this would:
      // 1. Parse all markdown files
      // 2. Extract content and metadata
      // 3. Create search index
      // 4. Generate search data files
    }
  }
}

/**
 * Search index builder for documentation content
 */
export class SearchIndexBuilder {
  private index: Map<string, any> = new Map()
  
  async buildIndex(pages: string[]) {
    console.log(`Building search index for ${pages.length} pages...`)
    
    // Process each page
    for (const page of pages) {
      await this.indexPage(page)
    }
    
    return this.generateSearchData()
  }
  
  private async indexPage(pagePath: string) {
    // Extract content, headings, and metadata
    const content = await this.extractContent(pagePath)
    const headings = this.extractHeadings(content)
    const keywords = this.extractKeywords(content)
    
    this.index.set(pagePath, {
      title: this.extractTitle(content),
      content: this.stripMarkdown(content),
      headings,
      keywords,
      path: pagePath
    })
  }
  
  private async extractContent(pagePath: string): Promise<string> {
    // In a real implementation, read and parse the markdown file
    return `# Sample Content\nThis is sample content for ${pagePath}`
  }
  
  private extractHeadings(content: string): string[] {
    const headingRegex = /^#{1,6}\s+(.+)$/gm
    const headings: string[] = []
    let match
    
    while ((match = headingRegex.exec(content)) !== null) {
      headings.push(match[1].trim())
    }
    
    return headings
  }
  
  private extractKeywords(content: string): string[] {
    // Simple keyword extraction - in reality, use NLP
    const words = content.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 3)
    
    // Get unique words and sort by frequency
    const wordCount = new Map<string, number>()
    words.forEach(word => {
      wordCount.set(word, (wordCount.get(word) || 0) + 1)
    })
    
    return Array.from(wordCount.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([word]) => word)
  }
  
  private extractTitle(content: string): string {
    const titleMatch = content.match(/^#\s+(.+)$/m)
    return titleMatch ? titleMatch[1] : 'Untitled'
  }
  
  private stripMarkdown(content: string): string {
    return content
      .replace(/#{1,6}\s+/g, '') // Remove heading markers
      .replace(/\*\*(.+?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.+?)\*/g, '$1') // Remove italic
      .replace(/`(.+?)`/g, '$1') // Remove inline code
      .replace(/\[(.+?)\]\(.+?\)/g, '$1') // Remove links, keep text
      .trim()
  }
  
  private generateSearchData() {
    const searchData = Array.from(this.index.entries()).map(([path, data]) => ({
      id: path,
      title: data.title,
      text: data.content.substring(0, 200), // First 200 chars for preview
      headings: data.headings,
      keywords: data.keywords,
      href: path.replace(/\.md$/, '.html')
    }))
    
    return {
      searchData,
      meta: {
        total: searchData.length,
        indexed: new Date().toISOString()
      }
    }
  }
}

/**
 * Client-side search functionality
 */
export class ClientSearch {
  private searchData: any[] = []
  private fuse: any = null
  
  async initialize(searchDataUrl: string) {
    try {
      const response = await fetch(searchDataUrl)
      const data = await response.json()
      this.searchData = data.searchData
      
      // Initialize Fuse.js for fuzzy search
      if (typeof window !== 'undefined') {
        const { default: Fuse } = await import('fuse.js')
        this.fuse = new Fuse(this.searchData, {
          keys: [
            { name: 'title', weight: 0.4 },
            { name: 'text', weight: 0.3 },
            { name: 'headings', weight: 0.2 },
            { name: 'keywords', weight: 0.1 }
          ],
          threshold: 0.4,
          includeScore: true,
          includeMatches: true
        })
      }
    } catch (error) {
      console.error('Failed to initialize search:', error)
    }
  }
  
  search(query: string, limit = 10) {
    if (!this.fuse || !query.trim()) {
      return []
    }
    
    const results = this.fuse.search(query, { limit })
    
    return results.map((result: any) => ({
      ...result.item,
      score: result.score,
      matches: result.matches
    }))
  }
  
  suggest(query: string) {
    // Simple suggestion implementation
    if (!query.trim() || query.length < 2) {
      return []
    }
    
    const suggestions = new Set<string>()
    const lowerQuery = query.toLowerCase()
    
    this.searchData.forEach(item => {
      // Check title
      if (item.title.toLowerCase().includes(lowerQuery)) {
        suggestions.add(item.title)
      }
      
      // Check keywords
      item.keywords?.forEach((keyword: string) => {
        if (keyword.toLowerCase().includes(lowerQuery)) {
          suggestions.add(keyword)
        }
      })
    })
    
    return Array.from(suggestions).slice(0, 5)
  }
}