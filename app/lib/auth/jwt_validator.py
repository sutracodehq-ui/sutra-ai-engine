"""
JWT Validator — Verifies global identity tokens.

Identity-AI: Validates Sutra-Identity tokens to enable SSO.
"""

import logging
from typing import Any, Dict
import jwt
from app.config import get_settings

logger = logging.getLogger(__name__)


class JwtValidator:
    """Service to decode and verify identity tokens."""

    def __init__(self):
        self.settings = get_settings()
        # In a real system, we'd fetch JWKS or use a shared secret
        self.secret = "shield_secret_change_me" 
        self.algorithm = "HS256"

    def decode_token(self, token: str) -> Dict[str, Any] | None:
        """Decode and verify a JWT token from Sutra-Identity."""
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                # In production, specify audience and issuer
                # audience="sutra-ai-engine",
                # issuer="sutra-identity"
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected JWT error: {e}")
            return None
