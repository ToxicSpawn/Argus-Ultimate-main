"""Push 71 — HMAC-SHA256 request signing for Bybit V5 API.

Bybit V5 signature scheme:
  signature = HMAC-SHA256(api_secret,
      timestamp + api_key + recv_window + payload)

Where payload is:
  - GET:  query string (params sorted alphabetically)
  - POST: raw JSON body string

Ref: https://bybit-exchange.github.io/docs/v5/guide
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional, Tuple


class BybitSigner:
    """Generates and validates Bybit V5 HMAC-SHA256 signatures.

    Args:
        api_key:      Bybit API key
        api_secret:   Bybit API secret
        recv_window:  Request validity window in ms (default 5000)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        recv_window: int = 5_000,
    ):
        self.api_key = api_key
        self._secret = api_secret.encode("utf-8")
        self.recv_window = recv_window

    def sign_get(
        self,
        params: Dict[str, Any],
        timestamp: Optional[int] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Sign a GET request.

        Returns:
            (headers, signed_params) — params with no added fields,
            headers contain X-BAPI-* auth fields.
        """
        ts = timestamp or self._ts()
        payload = "&".join(
            f"{k}={v}" for k, v in sorted(params.items())
        )
        sig = self._hmac(f"{ts}{self.api_key}{self.recv_window}{payload}")
        headers = self._auth_headers(ts, sig)
        return headers, params

    def sign_post(
        self,
        body: Dict[str, Any],
        timestamp: Optional[int] = None,
    ) -> Tuple[Dict[str, str], str]:
        """Sign a POST request.

        Returns:
            (headers, json_body_str)
        """
        ts = timestamp or self._ts()
        body_str = json.dumps(body, separators=(",", ":"))
        sig = self._hmac(f"{ts}{self.api_key}{self.recv_window}{body_str}")
        headers = self._auth_headers(ts, sig)
        headers["Content-Type"] = "application/json"
        return headers, body_str

    def validate_signature(
        self,
        payload: str,
        signature: str,
        timestamp: int,
    ) -> bool:
        """Validate an inbound webhook signature from Bybit."""
        expected = self._hmac(f"{timestamp}{self.api_key}{self.recv_window}{payload}")
        return hmac.compare_digest(expected, signature)

    def _hmac(self, message: str) -> str:
        return hmac.new(
            self._secret,
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, ts: int, signature: str) -> Dict[str, str]:
        return {
            "X-BAPI-API-KEY":     self.api_key,
            "X-BAPI-TIMESTAMP":   str(ts),
            "X-BAPI-SIGN":        signature,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
        }

    @staticmethod
    def _ts() -> int:
        return int(time.time() * 1000)
