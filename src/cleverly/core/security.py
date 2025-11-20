"""
API Methods Module - Cleverly Data Hub
======================================
Secure API key management with cryptographic generation, rate limiting,
and audit logging following best SWE principles.

Author: Cleverly Engineering Team
Version: 1.0.0

Design Principles:
- O(1) key lookup using hash maps
- Cryptographically secure key generation (256-bit entropy)
- Token bucket algorithm for rate limiting
- Thread-safe operations
- Comprehensive audit logging
"""

import secrets
import hashlib
import time
import threading
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
from pathlib import Path

# Configure logging
logger = logging.getLogger("api_methods")


class KeyPermission(Enum):
    """API key permission levels with hierarchical access control"""
    READ_ONLY = "read_only"      # Can only read data
    STANDARD = "standard"        # Read + trigger pipelines
    ADMIN = "admin"              # Full access including key management


class RateLimitStatus(Enum):
    """Rate limit check result"""
    ALLOWED = "allowed"
    LIMITED = "limited"
    BLOCKED = "blocked"


@dataclass
class APIKey:
    """
    API Key data structure with metadata

    Time Complexity: O(1) for all field access
    Space Complexity: O(1) per key
    """
    key_id: str                          # Unique identifier (first 8 chars of hash)
    key_hash: str                        # SHA-256 hash of the actual key
    name: str                            # Human-readable name
    permission: KeyPermission            # Access level
    created_at: datetime                 # Creation timestamp
    expires_at: Optional[datetime]       # Expiration (None = never)
    created_by: str                      # User who created the key
    last_used: Optional[datetime] = None # Last usage timestamp
    usage_count: int = 0                 # Total API calls
    is_active: bool = True               # Soft delete flag
    description: str = ""                # Optional description
    rate_limit: int = 100                # Requests per minute

    def is_expired(self) -> bool:
        """Check if key has expired - O(1)"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_valid(self) -> bool:
        """Check if key is valid for use - O(1)"""
        return self.is_active and not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "permission": self.permission.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_by": self.created_by,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "is_active": self.is_active,
            "description": self.description,
            "rate_limit": self.rate_limit
        }


@dataclass
class AuditLogEntry:
    """Audit log entry for security tracking"""
    timestamp: datetime
    action: str
    key_id: Optional[str]
    user: str
    ip_address: str
    success: bool
    details: str = ""


@dataclass
class RateLimitBucket:
    """
    Token bucket for rate limiting

    Algorithm: Token Bucket
    - Tokens regenerate at a fixed rate
    - Each request consumes one token
    - When bucket is empty, requests are limited

    Time Complexity: O(1) for all operations
    """
    tokens: float
    last_update: float
    max_tokens: int
    refill_rate: float  # tokens per second


class APIKeyManager:
    """
    Secure API Key Management System

    Features:
    - Cryptographically secure key generation (256-bit)
    - O(1) key validation using hash maps
    - Token bucket rate limiting
    - Thread-safe operations
    - Comprehensive audit logging
    - Persistent storage support

    Time Complexity Analysis:
    - Key generation: O(1)
    - Key validation: O(1) average case
    - Key lookup by ID: O(1)
    - List all keys: O(n)
    - Rate limit check: O(1)

    Space Complexity: O(n) where n = number of keys
    """

    # Constants
    KEY_LENGTH = 32  # 256 bits of entropy
    KEY_PREFIX = "clv_"  # Cleverly API key prefix
    HASH_ALGORITHM = "sha256"

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize API Key Manager

        Args:
            storage_path: Optional path for persistent storage
        """
        # Primary data structures - O(1) lookup
        self._keys_by_hash: Dict[str, APIKey] = {}  # hash -> APIKey
        self._keys_by_id: Dict[str, APIKey] = {}    # key_id -> APIKey

        # Rate limiting buckets - O(1) lookup
        self._rate_limits: Dict[str, RateLimitBucket] = {}

        # Audit log - append-only O(1) insertion
        self._audit_log: List[AuditLogEntry] = []

        # Thread safety
        self._lock = threading.RLock()

        # Storage
        self._storage_path = storage_path

        # Load existing keys if storage exists
        if storage_path and storage_path.exists():
            self._load_from_storage()

        logger.info("APIKeyManager initialized")

    def _generate_key(self) -> Tuple[str, str]:
        """
        Generate cryptographically secure API key

        Uses secrets.token_urlsafe for cryptographic randomness

        Returns:
            Tuple of (plain_key, key_hash)

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        # Generate 32 bytes (256 bits) of random data
        random_bytes = secrets.token_urlsafe(self.KEY_LENGTH)
        plain_key = f"{self.KEY_PREFIX}{random_bytes}"

        # Hash the key for storage (never store plain keys)
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        return plain_key, key_hash

    def _generate_key_id(self, key_hash: str) -> str:
        """
        Generate unique key ID from hash

        Time Complexity: O(1)
        """
        return key_hash[:8]

    def create_key(
        self,
        name: str,
        permission: KeyPermission,
        created_by: str,
        expires_in_days: Optional[int] = None,
        description: str = "",
        rate_limit: int = 100
    ) -> Tuple[str, APIKey]:
        """
        Create a new API key

        Args:
            name: Human-readable name for the key
            permission: Access level
            created_by: Username of creator
            expires_in_days: Days until expiration (None = never)
            description: Optional description
            rate_limit: Requests per minute limit

        Returns:
            Tuple of (plain_key, APIKey object)

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        with self._lock:
            # Generate secure key
            plain_key, key_hash = self._generate_key()
            key_id = self._generate_key_id(key_hash)

            # Handle potential ID collision (extremely rare with SHA-256)
            while key_id in self._keys_by_id:
                plain_key, key_hash = self._generate_key()
                key_id = self._generate_key_id(key_hash)

            # Calculate expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)

            # Create key object
            api_key = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                name=name,
                permission=permission,
                created_at=datetime.now(),
                expires_at=expires_at,
                created_by=created_by,
                description=description,
                rate_limit=rate_limit
            )

            # Store in both indices for O(1) lookup
            self._keys_by_hash[key_hash] = api_key
            self._keys_by_id[key_id] = api_key

            # Initialize rate limit bucket
            self._rate_limits[key_hash] = RateLimitBucket(
                tokens=float(rate_limit),
                last_update=time.time(),
                max_tokens=rate_limit,
                refill_rate=rate_limit / 60.0  # per second
            )

            # Audit log
            self._log_audit("key_created", key_id, created_by, "", True,
                          f"Created key '{name}' with {permission.value} permission")

            # Persist if storage configured
            self._save_to_storage()

            logger.info("Created API key: %s (%s)", key_id, name)

            return plain_key, api_key

    def validate_key(self, plain_key: str, ip_address: str = "") -> Tuple[bool, Optional[APIKey], str]:
        """
        Validate an API key

        Args:
            plain_key: The plain text API key
            ip_address: Client IP for logging

        Returns:
            Tuple of (is_valid, APIKey or None, error_message)

        Time Complexity: O(1) average case
        Space Complexity: O(1)
        """
        with self._lock:
            # Hash the provided key
            key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

            # O(1) lookup
            api_key = self._keys_by_hash.get(key_hash)

            if not api_key:
                self._log_audit("key_validation", None, "unknown", ip_address, False,
                              "Invalid key attempted")
                return False, None, "Invalid API key"

            # Check if key is active
            if not api_key.is_active:
                self._log_audit("key_validation", api_key.key_id, "unknown", ip_address, False,
                              "Revoked key used")
                return False, None, "API key has been revoked"

            # Check expiration
            if api_key.is_expired():
                self._log_audit("key_validation", api_key.key_id, "unknown", ip_address, False,
                              "Expired key used")
                return False, None, "API key has expired"

            # Check rate limit
            rate_status = self._check_rate_limit(key_hash)
            if rate_status != RateLimitStatus.ALLOWED:
                self._log_audit("key_validation", api_key.key_id, "unknown", ip_address, False,
                              f"Rate limited: {rate_status.value}")
                return False, None, "Rate limit exceeded"

            # Update usage statistics
            api_key.last_used = datetime.now()
            api_key.usage_count += 1

            self._log_audit("key_validation", api_key.key_id, "unknown", ip_address, True,
                          "Key validated successfully")

            return True, api_key, ""

    def _check_rate_limit(self, key_hash: str) -> RateLimitStatus:
        """
        Check and update rate limit using token bucket algorithm

        Algorithm: Token Bucket
        1. Calculate tokens to add based on time elapsed
        2. Add tokens (capped at max)
        3. Check if tokens available
        4. Consume one token if available

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        bucket = self._rate_limits.get(key_hash)
        if not bucket:
            return RateLimitStatus.ALLOWED

        current_time = time.time()
        elapsed = current_time - bucket.last_update

        # Refill tokens based on elapsed time
        bucket.tokens = min(
            bucket.max_tokens,
            bucket.tokens + elapsed * bucket.refill_rate
        )
        bucket.last_update = current_time

        # Check if request can be served
        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return RateLimitStatus.ALLOWED
        else:
            return RateLimitStatus.LIMITED

    def revoke_key(self, key_id: str, revoked_by: str, reason: str = "") -> bool:
        """
        Revoke an API key (soft delete)

        Time Complexity: O(1)
        """
        with self._lock:
            api_key = self._keys_by_id.get(key_id)
            if not api_key:
                return False

            api_key.is_active = False

            self._log_audit("key_revoked", key_id, revoked_by, "", True,
                          f"Key revoked: {reason}")
            self._save_to_storage()

            logger.info("Revoked API key: %s", key_id)
            return True

    def get_key_by_id(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID - O(1)"""
        return self._keys_by_id.get(key_id)

    def list_keys(self, include_revoked: bool = False) -> List[APIKey]:
        """
        List all API keys

        Time Complexity: O(n)
        """
        with self._lock:
            if include_revoked:
                return list(self._keys_by_id.values())
            return [k for k in self._keys_by_id.values() if k.is_active]

    def get_key_stats(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a key"""
        api_key = self._keys_by_id.get(key_id)
        if not api_key:
            return None

        bucket = self._rate_limits.get(api_key.key_hash)

        return {
            **api_key.to_dict(),
            "rate_limit_remaining": int(bucket.tokens) if bucket else 0,
            "is_valid": api_key.is_valid()
        }

    def update_key(
        self,
        key_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        rate_limit: Optional[int] = None,
        updated_by: str = ""
    ) -> bool:
        """Update API key properties - O(1)"""
        with self._lock:
            api_key = self._keys_by_id.get(key_id)
            if not api_key:
                return False

            if name:
                api_key.name = name
            if description is not None:
                api_key.description = description
            if rate_limit:
                api_key.rate_limit = rate_limit
                # Update bucket
                bucket = self._rate_limits.get(api_key.key_hash)
                if bucket:
                    bucket.max_tokens = rate_limit
                    bucket.refill_rate = rate_limit / 60.0

            self._log_audit("key_updated", key_id, updated_by, "", True, "Key properties updated")
            self._save_to_storage()

            return True

    def get_audit_log(self, key_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get audit log entries

        Time Complexity: O(n) where n = log entries
        """
        with self._lock:
            entries = self._audit_log

            if key_id:
                entries = [e for e in entries if e.key_id == key_id]

            # Return most recent entries
            return [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "action": e.action,
                    "key_id": e.key_id,
                    "user": e.user,
                    "ip_address": e.ip_address,
                    "success": e.success,
                    "details": e.details
                }
                for e in entries[-limit:]
            ]

    def _log_audit(
        self,
        action: str,
        key_id: Optional[str],
        user: str,
        ip_address: str,
        success: bool,
        details: str = ""
    ):
        """Add entry to audit log - O(1)"""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            action=action,
            key_id=key_id,
            user=user,
            ip_address=ip_address,
            success=success,
            details=details
        )
        self._audit_log.append(entry)

        # Keep log bounded (prevent memory leak)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

    def _save_to_storage(self):
        """Persist keys to storage"""
        if not self._storage_path:
            return

        try:
            data = {
                "keys": [k.to_dict() for k in self._keys_by_id.values()],
                "key_hashes": {k.key_id: k.key_hash for k in self._keys_by_id.values()}
            }
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error("Failed to save keys to storage: %s", e)

    def _load_from_storage(self):
        """Load keys from storage"""
        try:
            data = json.loads(self._storage_path.read_text())

            for key_data in data.get("keys", []):
                key_id = key_data["key_id"]
                key_hash = data.get("key_hashes", {}).get(key_id)

                if not key_hash:
                    continue

                api_key = APIKey(
                    key_id=key_id,
                    key_hash=key_hash,
                    name=key_data["name"],
                    permission=KeyPermission(key_data["permission"]),
                    created_at=datetime.fromisoformat(key_data["created_at"]),
                    expires_at=datetime.fromisoformat(key_data["expires_at"]) if key_data.get("expires_at") else None,
                    created_by=key_data["created_by"],
                    last_used=datetime.fromisoformat(key_data["last_used"]) if key_data.get("last_used") else None,
                    usage_count=key_data.get("usage_count", 0),
                    is_active=key_data.get("is_active", True),
                    description=key_data.get("description", ""),
                    rate_limit=key_data.get("rate_limit", 100)
                )

                self._keys_by_hash[key_hash] = api_key
                self._keys_by_id[key_id] = api_key

                # Initialize rate limit bucket
                self._rate_limits[key_hash] = RateLimitBucket(
                    tokens=float(api_key.rate_limit),
                    last_update=time.time(),
                    max_tokens=api_key.rate_limit,
                    refill_rate=api_key.rate_limit / 60.0
                )

            logger.info("Loaded %d API keys from storage", len(self._keys_by_id))

        except Exception as e:
            logger.error("Failed to load keys from storage: %s", e)

    def cleanup_expired_keys(self) -> int:
        """
        Cleanup expired keys

        Time Complexity: O(n)
        """
        with self._lock:
            expired_count = 0
            for api_key in list(self._keys_by_id.values()):
                if api_key.is_expired() and api_key.is_active:
                    api_key.is_active = False
                    expired_count += 1

            if expired_count > 0:
                self._save_to_storage()
                logger.info("Cleaned up %d expired keys", expired_count)

            return expired_count


# Utility functions for code examples
def generate_curl_example(endpoint: str, api_key: str, method: str = "GET", data: Optional[Dict] = None) -> str:
    """Generate curl command example"""
    cmd = f'curl -X {method}'
    cmd += f' -H "X-API-Key: {api_key}"'
    cmd += ' -H "Content-Type: application/json"'

    if data:
        cmd += f" -d '{json.dumps(data)}'"

    cmd += f' "http://localhost:5000{endpoint}"'
    return cmd


def generate_python_example(endpoint: str, api_key: str, method: str = "GET", data: Optional[Dict] = None) -> str:
    """Generate Python requests example"""
    code = "import requests\n\n"
    code += f'api_key = "{api_key}"\n'
    code += f'url = "http://localhost:5000{endpoint}"\n\n'
    code += "headers = {\n"
    code += '    "X-API-Key": api_key,\n'
    code += '    "Content-Type": "application/json"\n'
    code += "}\n\n"

    if method == "GET":
        code += "response = requests.get(url, headers=headers)\n"
    elif method == "POST":
        if data:
            code += f"data = {json.dumps(data, indent=4)}\n\n"
            code += "response = requests.post(url, headers=headers, json=data)\n"
        else:
            code += "response = requests.post(url, headers=headers)\n"

    code += "\nprint(response.json())"
    return code


# Singleton instance
_manager_instance: Optional[APIKeyManager] = None


def get_api_key_manager(storage_path: Optional[Path] = None) -> APIKeyManager:
    """Get or create singleton API key manager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = APIKeyManager(storage_path)
    return _manager_instance
