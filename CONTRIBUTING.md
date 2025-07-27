# Contributing to LeanVibe Agent Hive

We love your input! We want to make contributing to LeanVibe Agent Hive as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### We Use [GitHub Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- PostgreSQL 15+ (or use Docker)
- Redis 7+ (or use Docker)

### Setup Steps

```bash
# 1. Clone your fork
git clone https://github.com/yourusername/bee-hive.git
cd bee-hive

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# 3. Start services
docker-compose up -d postgres redis

# 4. Run migrations
alembic upgrade head

# 5. Set up frontend (choose one or both)
# Vue.js Web Dashboard
cd frontend && npm install && cd ..

# Mobile PWA Dashboard
cd mobile-pwa && npm install && cd ..

# 6. Create .env file
cp .env.example .env
# Edit .env with your configuration
```

### Running the Application

```bash
# Backend
uvicorn app.main:app --reload

# Frontend (Vue.js)
cd frontend && npm run dev

# Mobile PWA
cd mobile-pwa && npm run dev
```

## Code Quality Standards

### Python

We use several tools to maintain code quality:

```bash
# Format code
black app/ tests/

# Check linting
ruff check app/ tests/

# Type checking
mypy app/

# Run tests
pytest -v --cov=app
```

**Code Style Guidelines:**
- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Use async/await for all I/O operations
- Maximum line length: 88 characters (Black default)

### TypeScript/JavaScript

For both frontends:

```bash
# Mobile PWA
cd mobile-pwa
npm run lint        # ESLint
npm run type-check  # TypeScript checking
npm test           # Unit tests

# Vue.js Frontend
cd frontend
npm run lint
npm run type-check
npm test
```

**Frontend Guidelines:**
- Use TypeScript strict mode
- Follow established component patterns
- Write unit tests for complex logic
- Use semantic HTML for accessibility
- Follow mobile-first responsive design principles

## Testing Requirements

### Minimum Coverage

- **Backend**: 90% test coverage required
- **Frontend**: 80% test coverage required
- **E2E**: Critical user flows must be covered

### Testing Strategy

```bash
# Backend unit tests
pytest tests/unit/

# Backend integration tests
pytest tests/integration/

# Backend E2E tests
pytest tests/e2e/

# Frontend unit tests
cd frontend && npm test

# Mobile PWA unit tests
cd mobile-pwa && npm test

# Mobile PWA E2E tests
cd mobile-pwa && npm run test:e2e
```

### Writing Tests

**Backend Tests:**
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_agent(client: AsyncClient):
    response = await client.post("/api/v1/agents/", json={
        "name": "test-agent",
        "role": "developer"
    })
    assert response.status_code == 201
    assert response.json()["name"] == "test-agent"
```

**Frontend Tests:**
```typescript
import { render, screen } from '@testing-library/react'
import { AgentCard } from './AgentCard'

test('renders agent name', () => {
  render(<AgentCard name="test-agent" status="active" />)
  expect(screen.getByText('test-agent')).toBeInTheDocument()
})
```

## Documentation

### Code Documentation

- **Python**: Use Google-style docstrings
- **TypeScript**: Use JSDoc comments for complex functions
- **API**: Update OpenAPI schemas when changing endpoints

### User Documentation

When adding features that affect user experience:

1. Update relevant documentation in `docs/`
2. Add examples to README if applicable
3. Update API documentation if endpoints change

## Submitting Changes

### Pull Request Guidelines

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what changes you made and why
3. **Testing**: Describe how you tested your changes
4. **Screenshots**: Include screenshots for UI changes
5. **Breaking Changes**: Clearly mark any breaking changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## Issue Reporting

### Bug Reports

When filing an issue, make sure to answer these questions:

1. What version of the software are you using?
2. What operating system and processor architecture are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

Use this template:

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. macOS, Windows, Linux]
 - Browser [e.g. chrome, safari] (for frontend issues)
 - Version [e.g. 22]

**Additional context**
Add any other context about the problem here.
```

### Feature Requests

```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Security Issues

Please do not report security vulnerabilities through public GitHub issues.

Instead, please send an email to security@leanvibe.com with:
- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any possible mitigations

We'll respond as quickly as possible and work with you to resolve the issue.

## Community Guidelines

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

### Getting Help

- **Documentation**: Check the [docs/](docs/) directory first
- **GitHub Discussions**: For questions and community discussion
- **GitHub Issues**: For bug reports and feature requests
- **Discord**: Join our community Discord server (link in README)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Annual contributor highlights

Thank you for contributing to LeanVibe Agent Hive! 🚀