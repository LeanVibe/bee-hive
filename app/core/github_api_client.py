"""
GitHub API Client for LeanVibe Agent Hive 2.0

Comprehensive GitHub REST/GraphQL API client with authentication, rate limiting,
retry mechanisms, and webhook integration for multi-agent development workflows.
"""

import asyncio
import json
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from urllib.parse import urljoin
import logging

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet

from ..core.config import get_settings
from ..models.github_integration import GitHubRepository, PullRequest, GitHubIssue


logger = logging.getLogger(__name__)
try:
    settings = get_settings()
except Exception:
    # During CI pytest collection, fall back to minimal defaults
    class _Minimal:
        GITHUB_TOKEN = None
        GITHUB_API_URL = "https://api.github.com"
    settings = _Minimal()


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: Dict = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        super().__init__(self.message)


class RateLimitInfo(BaseModel):
    """Rate limit information from GitHub API."""
    limit: int
    remaining: int
    reset_time: datetime
    used: int
    resource: str = "core"


class GitHubAPIRateLimiter:
    """
    Advanced rate limiting with exponential backoff and intelligent batching.
    
    Implements GitHub's rate limiting best practices with predictive rate limiting
    and automatic request queuing.
    """
    
    def __init__(self):
        self.rate_limits: Dict[str, RateLimitInfo] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.processing_queue = False
        self.backoff_multiplier = 1.0
        
    async def check_rate_limit(self, resource: str = "core") -> bool:
        """Check if request can be made within rate limits."""
        if resource not in self.rate_limits:
            return True
            
        rate_limit = self.rate_limits[resource]
        current_time = datetime.utcnow()
        
        # If reset time has passed, we can proceed
        if current_time >= rate_limit.reset_time:
            return True
            
        # If we have remaining requests, we can proceed
        if rate_limit.remaining > 0:
            return True
            
        return False
    
    async def calculate_wait_time(self, resource: str = "core") -> float:
        """Calculate optimal wait time before next request."""
        if resource not in self.rate_limits:
            return 0.0
            
        rate_limit = self.rate_limits[resource]
        current_time = datetime.utcnow()
        
        if current_time >= rate_limit.reset_time:
            return 0.0
            
        if rate_limit.remaining > 0:
            # Calculate optimal pacing to spread requests evenly
            time_until_reset = (rate_limit.reset_time - current_time).total_seconds()
            optimal_delay = time_until_reset / (rate_limit.remaining + 1)
            return min(optimal_delay, 60.0)  # Max 60 seconds between requests
            
        # Must wait until reset
        return (rate_limit.reset_time - current_time).total_seconds()
    
    def update_rate_limit(self, headers: Dict[str, str], resource: str = "core") -> None:
        """Update rate limit information from response headers."""
        try:
            limit = int(headers.get(f"x-ratelimit-limit", 5000))
            remaining = int(headers.get(f"x-ratelimit-remaining", limit))
            reset_timestamp = int(headers.get(f"x-ratelimit-reset", time.time() + 3600))
            used = int(headers.get(f"x-ratelimit-used", 0))
            
            reset_time = datetime.fromtimestamp(reset_timestamp)
            
            self.rate_limits[resource] = RateLimitInfo(
                limit=limit,
                remaining=remaining,
                reset_time=reset_time,
                used=used,
                resource=resource
            )
            
            logger.debug(f"Updated rate limit for {resource}: {remaining}/{limit} remaining")
            
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse rate limit headers: {e}")
    
    async def wait_if_needed(self, resource: str = "core") -> None:
        """Wait if necessary to respect rate limits."""
        if not await self.check_rate_limit(resource):
            wait_time = await self.calculate_wait_time(resource)
            if wait_time > 0:
                logger.info(f"Rate limit reached for {resource}, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)


class GitHubAPIClient:
    """
    Comprehensive GitHub API client with advanced features.
    
    Provides high-level GitHub operations with automatic retries, rate limiting,
    webhook handling, and comprehensive error management.
    """
    
    def __init__(self, token: str = None, base_url: str = "https://api.github.com"):
        self.token = token or settings.GITHUB_TOKEN
        self.base_url = base_url
        self.rate_limiter = GitHubAPIRateLimiter()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _get_headers(self, additional_headers: Dict[str, str] = None) -> Dict[str, str]:
        """Get standard headers for GitHub API requests."""
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "LeanVibe-Agent-Hive/2.0",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        if additional_headers:
            headers.update(additional_headers)
            
        return headers
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        retries: int = 3,
        resource: str = "core"
    ) -> Dict[str, Any]:
        """
        Make authenticated request to GitHub API with retries and rate limiting.
        """
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        request_headers = self._get_headers(headers)
        
        for attempt in range(retries + 1):
            try:
                # Wait for rate limiting
                await self.rate_limiter.wait_if_needed(resource)
                
                # Make the request
                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers
                )
                
                # Update rate limit information
                self.rate_limiter.update_rate_limit(dict(response.headers), resource)
                
                # Handle successful responses
                if response.status_code < 400:
                    if response.content:
                        return response.json()
                    return {}
                
                # Handle rate limit exceeded
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    continue
                
                # Handle other client errors
                if response.status_code < 500:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    
                    raise GitHubAPIError(
                        f"GitHub API error: {response.status_code} - {error_data.get('message', 'Unknown error')}",
                        status_code=response.status_code,
                        response_data=error_data
                    )
                
                # Handle server errors with exponential backoff
                if attempt < retries:
                    wait_time = (2 ** attempt) * self.rate_limiter.backoff_multiplier
                    logger.warning(f"Server error {response.status_code}, retrying in {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    continue
                
                raise GitHubAPIError(
                    f"GitHub API server error: {response.status_code}",
                    status_code=response.status_code
                )
                
            except httpx.RequestError as e:
                if attempt < retries:
                    wait_time = (2 ** attempt) * self.rate_limiter.backoff_multiplier
                    logger.warning(f"Request error: {e}, retrying in {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    continue
                raise GitHubAPIError(f"Request failed: {str(e)}")
        
        raise GitHubAPIError("Max retries exceeded")
    
    # Repository Operations
    async def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}")
    
    async def list_repository_branches(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """List all branches in repository."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}/branches")
    
    async def get_branch(self, owner: str, repo: str, branch: str) -> Dict[str, Any]:
        """Get specific branch information."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}/branches/{branch}")
    
    async def create_branch(self, owner: str, repo: str, branch_name: str, source_sha: str) -> Dict[str, Any]:
        """Create new branch from source SHA."""
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": source_sha
        }
        return await self._make_request("POST", f"/repos/{owner}/{repo}/git/refs", data=data)
    
    async def delete_branch(self, owner: str, repo: str, branch_name: str) -> bool:
        """Delete a branch."""
        try:
            await self._make_request("DELETE", f"/repos/{owner}/{repo}/git/refs/heads/{branch_name}")
            return True
        except GitHubAPIError:
            return False
    
    # Pull Request Operations
    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = None,
        draft: bool = False
    ) -> Dict[str, Any]:
        """Create a new pull request."""
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body or "",
            "draft": draft
        }
        return await self._make_request("POST", f"/repos/{owner}/{repo}/pulls", data=data)
    
    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        """Get pull request details."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")
    
    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        base: str = None,
        head: str = None
    ) -> List[Dict[str, Any]]:
        """List pull requests with filtering."""
        params = {"state": state}
        if base:
            params["base"] = base
        if head:
            params["head"] = head
            
        return await self._make_request("GET", f"/repos/{owner}/{repo}/pulls", params=params)
    
    async def update_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        title: str = None,
        body: str = None,
        state: str = None
    ) -> Dict[str, Any]:
        """Update pull request."""
        data = {}
        if title:
            data["title"] = title
        if body:
            data["body"] = body
        if state:
            data["state"] = state
            
        return await self._make_request("PATCH", f"/repos/{owner}/{repo}/pulls/{pr_number}", data=data)
    
    async def merge_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_title: str = None,
        commit_message: str = None,
        merge_method: str = "merge"
    ) -> Dict[str, Any]:
        """Merge a pull request."""
        data = {"merge_method": merge_method}
        if commit_title:
            data["commit_title"] = commit_title
        if commit_message:
            data["commit_message"] = commit_message
            
        return await self._make_request("PUT", f"/repos/{owner}/{repo}/pulls/{pr_number}/merge", data=data)
    
    # Issue Operations
    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = None,
        labels: List[str] = None,
        assignees: List[str] = None
    ) -> Dict[str, Any]:
        """Create a new issue."""
        data = {
            "title": title,
            "body": body or "",
            "labels": labels or [],
            "assignees": assignees or []
        }
        return await self._make_request("POST", f"/repos/{owner}/{repo}/issues", data=data)
    
    async def get_issue(self, owner: str, repo: str, issue_number: int) -> Dict[str, Any]:
        """Get issue details."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}")
    
    async def list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        labels: List[str] = None,
        assignee: str = None
    ) -> List[Dict[str, Any]]:
        """List issues with filtering."""
        params = {"state": state}
        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee"] = assignee
            
        return await self._make_request("GET", f"/repos/{owner}/{repo}/issues", params=params)
    
    async def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        title: str = None,
        body: str = None,
        state: str = None,
        assignees: List[str] = None,
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """Update issue."""
        data = {}
        if title:
            data["title"] = title
        if body:
            data["body"] = body
        if state:
            data["state"] = state
        if assignees is not None:
            data["assignees"] = assignees
        if labels is not None:
            data["labels"] = labels
            
        return await self._make_request("PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", data=data)
    
    async def add_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> Dict[str, Any]:
        """Add comment to issue."""
        data = {"body": body}
        return await self._make_request("POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments", data=data)
    
    # Review Operations
    async def create_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        event: str,
        body: str = None,
        comments: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a pull request review."""
        data = {
            "event": event,  # APPROVE, REQUEST_CHANGES, COMMENT
            "body": body or "",
            "comments": comments or []
        }
        return await self._make_request("POST", f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews", data=data)
    
    async def list_reviews(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """List pull request reviews."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews")
    
    # Commit Operations
    async def get_commit(self, owner: str, repo: str, sha: str) -> Dict[str, Any]:
        """Get commit details."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}/commits/{sha}")
    
    async def list_commits(
        self,
        owner: str,
        repo: str,
        sha: str = None,
        path: str = None,
        since: datetime = None,
        until: datetime = None
    ) -> List[Dict[str, Any]]:
        """List commits with filtering."""
        params = {}
        if sha:
            params["sha"] = sha
        if path:
            params["path"] = path
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()
            
        return await self._make_request("GET", f"/repos/{owner}/{repo}/commits", params=params)
    
    async def compare_commits(self, owner: str, repo: str, base: str, head: str) -> Dict[str, Any]:
        """Compare two commits."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}/compare/{base}...{head}")
    
    # Webhook Operations
    async def create_webhook(
        self,
        owner: str,
        repo: str,
        url: str,
        secret: str,
        events: List[str] = None
    ) -> Dict[str, Any]:
        """Create repository webhook."""
        data = {
            "config": {
                "url": url,
                "content_type": "json",
                "secret": secret,
                "insecure_ssl": "0"
            },
            "events": events or ["push", "pull_request", "issues"],
            "active": True
        }
        return await self._make_request("POST", f"/repos/{owner}/{repo}/hooks", data=data)
    
    async def list_webhooks(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """List repository webhooks."""
        return await self._make_request("GET", f"/repos/{owner}/{repo}/hooks")
    
    async def delete_webhook(self, owner: str, repo: str, hook_id: int) -> bool:
        """Delete repository webhook."""
        try:
            await self._make_request("DELETE", f"/repos/{owner}/{repo}/hooks/{hook_id}")
            return True
        except GitHubAPIError:
            return False
    
    def verify_webhook_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verify webhook signature."""
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Remove 'sha256=' prefix from GitHub signature
        if signature.startswith('sha256='):
            signature = signature[7:]
        
        return hmac.compare_digest(expected_signature, signature)
    
    # GraphQL Operations
    async def execute_graphql(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute GraphQL query."""
        data = {
            "query": query,
            "variables": variables or {}
        }
        return await self._make_request("POST", "/graphql", data=data, resource="graphql")
    
    # Utility Methods
    def encrypt_token(self, token: str) -> str:
        """Encrypt access token for storage."""
        return self.fernet.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt access token."""
        return self.fernet.decrypt(encrypted_token.encode()).decode()
    
    async def check_rate_limits(self) -> Dict[str, RateLimitInfo]:
        """Get current rate limit status for all resources."""
        response = await self._make_request("GET", "/rate_limit")
        
        rate_limits = {}
        for resource, data in response.get("resources", {}).items():
            rate_limits[resource] = RateLimitInfo(
                limit=data["limit"],
                remaining=data["remaining"],
                reset_time=datetime.fromtimestamp(data["reset"]),
                used=data["used"],
                resource=resource
            )
        
        return rate_limits
    
    async def get_authenticated_user(self) -> Dict[str, Any]:
        """Get authenticated user information."""
        return await self._make_request("GET", "/user")
    
    async def health_check(self) -> bool:
        """Perform health check on GitHub API connectivity."""
        try:
            await self.get_authenticated_user()
            return True
        except Exception as e:
            logger.error(f"GitHub API health check failed: {e}")
            return False


class GitHubGraphQLQueries:
    """
    Collection of optimized GraphQL queries for efficient GitHub operations.
    """
    
    @staticmethod
    def get_repository_with_prs_and_issues() -> str:
        """Get repository with recent PRs and issues in single query."""
        return """
        query GetRepositoryInfo($owner: String!, $name: String!) {
          repository(owner: $owner, name: $name) {
            id
            name
            nameWithOwner
            url
            defaultBranchRef {
              name
            }
            pullRequests(first: 20, states: [OPEN], orderBy: {field: CREATED_AT, direction: DESC}) {
              nodes {
                id
                number
                title
                body
                headRefName
                baseRefName
                state
                mergeable
                createdAt
                updatedAt
                author {
                  login
                }
                reviews(first: 10) {
                  nodes {
                    state
                    author {
                      login
                    }
                  }
                }
              }
            }
            issues(first: 20, states: [OPEN], orderBy: {field: CREATED_AT, direction: DESC}) {
              nodes {
                id
                number
                title
                body
                state
                labels(first: 10) {
                  nodes {
                    name
                  }
                }
                assignees(first: 5) {
                  nodes {
                    login
                  }
                }
                createdAt
                updatedAt
              }
            }
          }
        }
        """
    
    @staticmethod
    def get_pull_request_files() -> str:
        """Get pull request with file changes."""
        return """
        query GetPullRequestFiles($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              id
              files(first: 100) {
                nodes {
                  path
                  additions
                  deletions
                  changeType
                }
              }
              commits(last: 50) {
                nodes {
                  commit {
                    oid
                    message
                    author {
                      name
                      email
                      date
                    }
                    changedFiles
                    additions
                    deletions
                  }
                }
              }
            }
          }
        }
        """