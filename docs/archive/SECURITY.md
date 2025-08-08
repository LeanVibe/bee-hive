# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT create a public GitHub issue

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

### 2. Send a private report

Instead, please send an email to: **security@leanvibe.com**

Include the following information:
- A clear description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any possible mitigations you've identified
- Your contact information for follow-up questions

### 3. Response Timeline

- **Initial Response**: We'll acknowledge receipt within 24 hours
- **Investigation**: We'll investigate and provide an initial assessment within 72 hours
- **Updates**: We'll provide regular updates on our progress every 5 business days
- **Resolution**: We aim to resolve critical vulnerabilities within 7 days

### 4. Coordinated Disclosure

- We follow responsible disclosure practices
- We'll work with you to understand the issue and develop a fix
- We'll credit you in the security advisory (if desired)
- We'll coordinate the public disclosure after a fix is available

## Security Measures

### Authentication & Authorization

- **JWT-based authentication** with secure token rotation
- **Role-based access control (RBAC)** with principle of least privilege
- **WebAuthn support** for passwordless authentication
- **Session timeout** and automatic logout
- **Rate limiting** on authentication endpoints
- **Account lockout** after failed login attempts

### Data Protection

- **Encryption at rest** for sensitive data in PostgreSQL
- **TLS 1.3** for all data in transit
- **Secure cookie settings** with HttpOnly, Secure, and SameSite flags
- **Input validation** and sanitization on all endpoints
- **SQL injection prevention** through parameterized queries
- **XSS protection** with Content Security Policy headers

### Infrastructure Security

- **Container security** with minimal base images and non-root users
- **Secrets management** with environment variables and external secret stores
- **Regular dependency updates** with automated vulnerability scanning
- **Network isolation** between services
- **Audit logging** for all security-relevant events
- **Monitoring and alerting** for suspicious activities

### API Security

- **CORS configuration** with explicit origin allowlists
- **Request size limits** to prevent DoS attacks
- **Rate limiting** with sliding window counters
- **Input validation** with Pydantic schemas
- **Error handling** that doesn't leak sensitive information
- **API versioning** for backward compatibility

### Frontend Security

- **Content Security Policy (CSP)** headers
- **Subresource Integrity (SRI)** for external resources
- **Secure storage** using Web Crypto API where possible
- **XSS prevention** through proper escaping and sanitization
- **CSRF protection** with SameSite cookies and CSRF tokens

## Security Best Practices for Contributors

### Code Review

- All code changes require review from at least two maintainers
- Security-sensitive changes require review from a security-aware maintainer
- Automated security scanning runs on all pull requests
- Dependencies are regularly audited for known vulnerabilities

### Development Environment

- Use the provided Docker Compose setup for consistency
- Keep development dependencies up to date
- Never commit secrets or API keys to the repository
- Use `.env` files (which are gitignored) for local configuration

### Testing

- Include security test cases for new features
- Test authentication and authorization flows thoroughly
- Verify input validation and error handling
- Test rate limiting and DoS protection measures

## Vulnerability Management

### Automated Scanning

We use automated tools to scan for vulnerabilities:

- **Dependabot** for dependency vulnerabilities
- **CodeQL** for static code analysis
- **npm audit** for Node.js dependencies
- **Safety** for Python dependencies
- **Docker image scanning** for container vulnerabilities

### Manual Security Reviews

- Security reviews for all major feature releases
- Regular penetration testing by external security firms
- Code audits for authentication and authorization logic
- Architecture reviews for new components

## Security Configuration

### Environment Variables

Never use default values in production:

```bash
# Example of secure production configuration
DEBUG=false
JWT_SECRET_KEY=<cryptographically-strong-random-key>
DATABASE_URL=postgresql://user:password@secure-host:5432/db
CORS_ORIGINS=["https://yourdomain.com"]
```

### TLS Configuration

Minimum TLS 1.2, prefer TLS 1.3:

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
```

### Database Security

- Use connection pooling with secure defaults
- Enable audit logging for sensitive operations
- Use prepared statements to prevent SQL injection
- Regular security updates for PostgreSQL and extensions

## Incident Response

### Detection

- Automated monitoring for security events
- Log analysis for suspicious patterns
- Performance monitoring for potential DoS attacks
- User reports through security@leanvibe.com

### Response Process

1. **Triage**: Assess severity and impact within 1 hour
2. **Containment**: Implement immediate mitigations
3. **Investigation**: Determine root cause and scope
4. **Remediation**: Deploy fixes and verify effectiveness
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Document and improve processes

### Communication

- Internal stakeholders notified immediately
- Users notified within 24 hours for high-severity issues
- Public post-mortem published after resolution
- Security advisory published with CVE if applicable

## Security Resources

### Training

- OWASP Top 10 awareness for all developers
- Secure coding training for backend developers
- Security architecture training for senior developers
- Regular security awareness sessions

### Tools and Resources

- [OWASP](https://owasp.org/) - Web Application Security
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE - Common Weakness Enumeration](https://cwe.mitre.org/)
- [CVE - Common Vulnerabilities and Exposures](https://cve.mitre.org/)

## Contact

For security-related questions or concerns:

- **Email**: security@leanvibe.com
- **PGP Key**: Available upon request
- **Bug Bounty**: We don't currently have a formal bug bounty program, but we appreciate responsible disclosure

---

**Note**: This security policy is a living document and is updated regularly to reflect our current security practices and threat landscape.