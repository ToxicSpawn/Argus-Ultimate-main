"""Gateway wiring — attaches SignalGateway to ArgusBot at startup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.signal_gateway.gateway_config import GatewayConfig
from core.signal_gateway.signal_gateway import SignalGateway
from core.signal_gateway.consensus_engine import ConsensusResult

if TYPE_CHECKING:
    pass  # Avoid circular import; ArgusBot is referenced by string below.

logger = logging.getLogger(__name__)


async def wire_gateway_to_argus_bot(bot: object) -> SignalGateway:  # type: ignore[type-arg]
    """Create and wire a SignalGateway into *bot* (an ArgusBot instance).

    Steps
    -----
    1. Load GatewayConfig from config/signal_gateway.json (or defaults).
    2. Instantiate SignalGateway.
    3. Register consensus callback → bot._on_consensus_signal().
    4. Attach gateway as bot.signal_gateway.
    5. Start the gateway worker.

    Signal sources (VoidBreaker, RL agent, LLM overlay, DeepLOB, OFI/VPIN)
    call ``bot.signal_gateway.ingest(envelope)`` directly.

    Returns
    -------
    The started SignalGateway instance.
    """
    config = GatewayConfig.load_from_json()
    gateway = SignalGateway(config=config)

    async def _on_consensus(result: ConsensusResult) -> None:
        """Route consensus result to ArgusBot's signal handler."""
        handler = getattr(bot, "_on_consensus_signal", None)
        if handler is None:
            logger.warning(
                "ArgusBot has no _on_consensus_signal method — "
                "consensus result dropped (direction=%s)",
                result.winning_direction,
            )
            return
        try:
            await handler(result)
        except Exception:
            logger.exception(
                "ArgusBot._on_consensus_signal raised; direction=%s",
                result.winning_direction,
            )

    gateway.on_consensus(_on_consensus)

    # Attach to bot so signal sources can reach it via bot.signal_gateway.
    setattr(bot, "signal_gateway", gateway)

    await gateway.start()
    logger.info(
        "SignalGateway wired to %s (threshold=%.2f, min_sources=%d)",
        type(bot).__name__,
        config.consensus_threshold,
        config.min_sources,
    )
    return gateway
