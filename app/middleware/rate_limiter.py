"""
Rate Limiting Middleware

Middleware để giới hạn số lượng requests từ mỗi IP address trong một khoảng thời gian.
Giúp bảo vệ API khỏi abuse và đảm bảo fair usage.

Requirements: Phase 1 Critical Optimization - Rate Limiting
"""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.utils.logging_utils import get_structured_logger

logger = get_structured_logger(__name__)


class RateLimiter(BaseHTTPMiddleware):
    """
    Rate limiting middleware sử dụng sliding window algorithm.
    
    Giới hạn số requests per IP address trong một time window.
    
    **IMPORTANT - Multi-Worker Limitation**:
    Rate limiting state is stored in-memory per worker process. In production
    deployments with multiple workers (e.g., Gunicorn, Uvicorn with --workers),
    each worker maintains its own independent rate limit counters.
    
    For accurate global rate limiting across all workers, consider using:
    - Redis-based rate limiting (recommended for production)
    - Shared memory solutions (e.g., multiprocessing.Manager)
    - External rate limiting service (e.g., Kong, Nginx rate limiting)
    
    Attributes:
        requests_per_minute: Số requests tối đa mỗi IP trong 1 phút
        cleanup_interval: Interval để cleanup expired entries (seconds)
        trusted_proxies: List of trusted proxy IPs for X-Forwarded-For validation
    
    Example:
        >>> app.add_middleware(RateLimiter, requests_per_minute=10, trusted_proxies=["10.0.0.1"])
    """
    
    def __init__(self, app, requests_per_minute: int = 10, trusted_proxies: list = None):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per IP per minute (default: 10)
            trusted_proxies: List of trusted proxy IPs (default: empty list)
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute in seconds
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.trusted_proxies = set(trusted_proxies or [])
        
        # Storage: IP -> list of request timestamps
        self._requests: Dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.time()
        
        logger.info(
            "Rate limiter initialized",
            requests_per_minute=requests_per_minute,
            window_size_seconds=self.window_size,
            trusted_proxies=list(self.trusted_proxies)
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request with security validation.
        
        Only trusts X-Forwarded-For header when request comes from a trusted proxy.
        This prevents IP spoofing attacks.
        
        Args:
            request: FastAPI request object
        
        Returns:
            Client IP address as string
        
        Security:
            - Only accepts X-Forwarded-For from trusted proxies
            - Falls back to direct client IP if not from trusted proxy
            - Prevents IP spoofing by validating proxy source
        """
        client_host = request.client.host if request.client else "unknown"
        
        # Check X-Forwarded-For header only if from trusted proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded and client_host in self.trusted_proxies:
            # X-Forwarded-For can contain multiple IPs, take the first one (original client)
            original_client = forwarded.split(",")[0].strip()
            logger.info(
                "Using X-Forwarded-For from trusted proxy",
                proxy_ip=client_host,
                original_client=original_client
            )
            return original_client
        
        # If not from trusted proxy or no X-Forwarded-For, use direct client IP
        if forwarded and client_host not in self.trusted_proxies:
            logger.warning(
                "Ignoring X-Forwarded-For from untrusted source",
                client_ip=client_host,
                forwarded_for=forwarded
            )
        
        return client_host
    
    def _cleanup_old_entries(self) -> None:
        """
        Cleanup expired request timestamps to prevent memory leak.
        
        Removes timestamps older than window_size from all IP entries.
        Runs periodically based on cleanup_interval.
        """
        current_time = time.time()
        
        # Only cleanup if interval has passed
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - self.window_size
        cleaned_count = 0
        
        # Remove old timestamps
        for ip in list(self._requests.keys()):
            old_count = len(self._requests[ip])
            self._requests[ip] = [
                ts for ts in self._requests[ip] 
                if ts > cutoff_time
            ]
            
            # Remove IP entry if no recent requests
            if not self._requests[ip]:
                del self._requests[ip]
                cleaned_count += 1
        
        self._last_cleanup = current_time
        
        if cleaned_count > 0:
            logger.info(
                "Rate limiter cleanup completed",
                cleaned_ips=cleaned_count,
                active_ips=len(self._requests)
            )
    
    def _is_rate_limited(self, ip: str) -> Tuple[bool, int]:
        """
        Check if IP is rate limited.
        
        Uses sliding window algorithm:
        1. Remove timestamps older than window_size
        2. Count remaining timestamps
        3. If count >= limit, reject request
        
        Args:
            ip: Client IP address
        
        Returns:
            Tuple of (is_limited: bool, remaining_requests: int)
        """
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        # Remove old timestamps for this IP
        self._requests[ip] = [
            ts for ts in self._requests[ip] 
            if ts > cutoff_time
        ]
        
        # Count recent requests
        request_count = len(self._requests[ip])
        remaining = max(0, self.requests_per_minute - request_count)
        
        # Check if limit exceeded
        is_limited = request_count >= self.requests_per_minute
        
        return is_limited, remaining
    
    def _record_request(self, ip: str) -> None:
        """
        Record a new request timestamp for IP.
        
        Args:
            ip: Client IP address
        """
        self._requests[ip].append(time.time())
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
        
        Returns:
            Response from next handler or 429 error
        
        Raises:
            HTTPException: 429 Too Many Requests if rate limit exceeded
        """
        # Skip rate limiting for health check endpoints
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Periodic cleanup
        self._cleanup_old_entries()
        
        # Check rate limit
        is_limited, remaining = self._is_rate_limited(client_ip)
        
        if is_limited:
            # Log rate limit violation
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path,
                method=request.method,
                limit=self.requests_per_minute
            )
            
            # Return 429 Too Many Requests
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute."
            )
        
        # Record this request
        self._record_request(client_ip)
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)  # -1 for current request
        
        # ⚡ FIX: Calculate reset based on oldest timestamp (accurate sliding window)
        # Reset time = when the oldest request in window will expire
        oldest_timestamp = self._requests[client_ip][0] if self._requests[client_ip] else time.time()
        reset_time = int(oldest_timestamp + self.window_size)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
