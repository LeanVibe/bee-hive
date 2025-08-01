# LeanVibe Agent Hive Documentation Site

Modern VitePress documentation site for LeanVibe Agent Hive - the autonomous software development platform that actually works.

## 🚀 Features

- **Modern Design**: Professional glass morphism design with responsive layouts
- **Interactive Components**: Live code playground, agent demonstrations, API explorer
- **Community Features**: Real-time activity feeds, contribution workflows, showcase
- **Enterprise Focus**: Business-oriented content and ROI calculators
- **Performance Optimized**: Fast loading, SEO optimized, PWA-ready
- **Accessibility**: WCAG 2.1 AA compliant with comprehensive keyboard navigation

## 🛠️ Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Available Scripts

```bash
# Development
npm run dev              # Start dev server with hot reload
npm run build            # Build for production
npm run preview          # Preview production build

# Quality Assurance
npm run lint             # Lint TypeScript and Vue files
npm run format           # Format code with Prettier
npm run check-links      # Validate internal and external links

# Documentation
npm run build:search     # Build search index
npm run generate-api     # Generate API documentation from code

# Deployment
npm run deploy           # Deploy to GitHub Pages
```

## 📁 Project Structure

```
docs-site/
├── .vitepress/           # VitePress configuration
│   ├── config.ts        # Main configuration
│   ├── theme/           # Custom theme
│   │   ├── index.ts     # Theme entry point
│   │   ├── components/  # Custom components
│   │   └── styles/      # Custom styles
│   └── plugins/         # Custom plugins
├── public/              # Static assets
├── learn/               # Learning content
├── api/                 # API documentation
├── examples/            # Examples and use cases
├── community/           # Community resources
├── enterprise/          # Enterprise content
└── index.md             # Homepage
```

## 🎨 Theming

The site uses a custom VitePress theme with:

- **Design System**: CSS custom properties for consistent styling
- **Glass Morphism**: Modern glass effect components
- **Responsive Design**: Mobile-first approach with breakpoint system
- **Dark Mode**: Automatic theme switching support
- **Typography**: Inter font family with proper font loading

### Key Design Tokens

```css
:root {
  --lv-primary: #6366f1;
  --lv-secondary: #10b981;
  --lv-accent: #f59e0b;
  --lv-glass-bg: rgba(255, 255, 255, 0.1);
  --lv-glass-border: rgba(255, 255, 255, 0.2);
}
```

## 🧩 Components

### Interactive Components

- **CodePlayground**: Live code execution environment
- **AgentDemo**: Real-time multi-agent coordination visualization
- **FeatureShowcase**: Interactive feature demonstrations
- **MetricsWidget**: Live system metrics display
- **CommunityHub**: Community activity and engagement

### Content Components

- **InteractiveGuide**: Step-by-step guided tutorials
- **EnterpriseCard**: Business-focused content cards
- **LiveDemo**: Embedded live demonstrations

## 🔌 Plugins

### Search Plugin (`plugins/search.ts`)
- Full-text search with Fuse.js
- AI-powered suggestions
- Real-time indexing

### Code Playground Plugin (`plugins/code-playground.ts`)
- Live code execution
- Syntax highlighting
- Template management
- WebSocket streaming

### Feed Plugin (`plugins/feed.ts`)
- RSS/Atom feed generation
- Sitemap generation
- SEO optimization

## 📊 Performance

### Lighthouse Scores
- **Performance**: 95+
- **Accessibility**: 100
- **Best Practices**: 100
- **SEO**: 100

### Optimization Features
- **Static Generation**: Pre-built pages for fast loading
- **Code Splitting**: Lazy loading of interactive components
- **Image Optimization**: WebP conversion and compression
- **CDN Ready**: Optimized for global distribution
- **Service Worker**: PWA capabilities with offline support

## 🔍 SEO Features

- **Meta Tags**: Comprehensive OpenGraph and Twitter Card support
- **Structured Data**: JSON-LD schema markup
- **Sitemap**: Automatically generated sitemap.xml
- **RSS Feeds**: Multi-format feed generation
- **Canonical URLs**: Proper URL canonicalization

## 🌐 Deployment

### GitHub Pages (Automatic)
The site automatically deploys to GitHub Pages on pushes to `main` branch.

### Custom Domain
Configure `CUSTOM_DOMAIN` variable in repository settings for custom domain deployment.

### Environment Variables

```bash
# Required for full functionality
VITE_ANTHROPIC_API_KEY=your_api_key_here

# Optional for enhanced features
VITE_ALGOLIA_APP_ID=your_algolia_app_id
VITE_ALGOLIA_API_KEY=your_algolia_api_key
VITE_DISCORD_WEBHOOK_URL=your_discord_webhook
```

## 🧪 Testing

### Link Validation
```bash
npm run check-links
```

### Performance Testing
```bash
# Lighthouse CI
npm install -g @lhci/cli
lhci autorun
```

### Accessibility Testing
```bash
# axe-core testing
npm install -g @axe-core/cli
axe https://docs.leanvibe.dev
```

## 🤝 Contributing

### Content Contributions
1. Fork the repository
2. Create a feature branch
3. Add/modify content in appropriate directories
4. Test locally with `npm run dev`
5. Submit pull request

### Component Development
1. Create components in `.vitepress/theme/components/`
2. Export from `.vitepress/theme/index.ts`
3. Document usage and props
4. Add TypeScript types

### Style Guidelines
- Use CSS custom properties for theming
- Follow BEM methodology for class names
- Ensure mobile responsiveness
- Test in multiple browsers

## 📚 Content Guidelines

### Writing Style
- **Clear and Concise**: Use simple, direct language
- **Progressive Disclosure**: Start simple, add complexity gradually
- **Action-Oriented**: Focus on what users can do
- **Community-Focused**: Encourage participation and contribution

### Code Examples
- **Complete and Runnable**: All examples should work as-is
- **Well-Commented**: Explain complex concepts
- **Multiple Languages**: Provide examples in various programming languages
- **Real-World**: Use practical, relevant scenarios

### Documentation Structure
- **Problem → Solution**: Start with user problems
- **Quick Start**: Provide immediate value
- **Deep Dive**: Offer comprehensive information
- **Next Steps**: Guide users to additional resources

## 🔧 Troubleshooting

### Common Issues

**Build Fails**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Search Not Working**
```bash
# Rebuild search index
npm run build:search
```

**Slow Performance**
```bash
# Analyze bundle size
npm run build -- --analyze
```

### Debug Mode
```bash
# Enable debug logging
DEBUG=vitepress:* npm run dev
```

## 📄 License

MIT License - see [LICENSE](../LICENSE) file for details.

## 🆘 Support

- **Documentation Issues**: [GitHub Issues](https://github.com/LeanVibe/bee-hive/issues)
- **General Questions**: [GitHub Discussions](https://github.com/LeanVibe/bee-hive/discussions)
- **Community Chat**: [Discord Server](https://discord.gg/leanvibe)
- **Enterprise Support**: team@leanvibe.dev

---

Built with ❤️ by the LeanVibe team using VitePress and modern web technologies.