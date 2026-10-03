"""
Generic REST / OAuth2 Adapter
Handles standard REST APIs (e.g. Stripe, Twilio, HubSpot) proving the platform's non-LLM applicability.
"""
from typing import Any
from gateway.core.adapter import ProviderAdapter, AdapterRequest
from gateway.core.types import ClassifiedError, ErrorCategory

class GenericRestAdapter(ProviderAdapter):
    """
    A generic adapter for standard REST APIs.
    Maps HTTP semantics to the platform's reliability primitives.
    """
    
    @property
    def provider_name(self) -> str:
        return "generic_rest"
        
    def get_limiting_unit(self) -> str:
        # Standard REST APIs typically limit by token/key unless specified otherwise.
        # Can be overridden by specific subclasses.
        return "key"
        
    def construct_payload(self, request: AdapterRequest) -> dict[str, Any]:
        return request.payload
        
    def parse_rate_limit_headers(self, response: Any) -> dict[str, Any]:
        """
        Attempts to parse standard rate limit headers.
        """
        headers = response.headers if hasattr(response, "headers") else {}
        
        # Standard IETF Draft headers
        limit = headers.get("RateLimit-Limit")
        remaining = headers.get("RateLimit-Remaining")
        reset = headers.get("RateLimit-Reset")
        
        # Fallback to X-RateLimit
        if not limit:
            limit = headers.get("X-RateLimit-Limit")
            remaining = headers.get("X-RateLimit-Remaining")
            reset = headers.get("X-RateLimit-Reset")
            
        parsed = {}
        if limit is not None:
            parsed["limit"] = int(limit)
        if remaining is not None:
            parsed["remaining"] = int(remaining)
        if reset is not None:
            parsed["reset_time"] = int(reset)
            
        return parsed
        
    def classify_error(self, error: Any) -> ClassifiedError:
        """
        Maps standard HTTP status codes to the gateway's ErrorCategory taxonomy.
        """
        status_code = getattr(error, "status_code", 500)
        
        if status_code == 429:
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMITED,
                message="Rate limited"
            )
        elif status_code == 503 or status_code == 502 or status_code == 504:
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message="Transient server error"
            )
        elif status_code == 401 or status_code == 403:
            return ClassifiedError(
                category=ErrorCategory.PERMANENT,
                message="Authentication/Authorization failed"
            )
        elif status_code == 400:
            return ClassifiedError(
                category=ErrorCategory.PERMANENT,
                message="Bad request"
            )
        elif status_code == 402:
            return ClassifiedError(
                category=ErrorCategory.QUOTA_EXHAUSTED,
                message="Payment required / Quota exhausted"
            )
            
        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            message=str(error)
        )
        
    def check_known_bad_patterns(self, response: Any) -> list[Any]:
        # A generic REST adapter has no specific known-bad patterns 
        # (unlike LLMs which have "As an AI..." hallucinations).
        return []

    def quota_bucket(self, request: AdapterRequest) -> str:
        return "default"

    def classify_exception(self, exc: Exception, request: AdapterRequest) -> ClassifiedError:
        return ClassifiedError(
            category=ErrorCategory.TRANSIENT,
            message=str(exc)
        )

    def extract_usage(self, response: Any) -> Any:
        return None

    def extract_request_id(self, response: Any) -> str | None:
        headers = response.headers if hasattr(response, "headers") else {}
        return headers.get("x-request-id") or headers.get("request-id")

    async def send(self, request: AdapterRequest) -> Any:
        # Mock send for generic adapter
        import httpx
        return httpx.Response(200, json={})
