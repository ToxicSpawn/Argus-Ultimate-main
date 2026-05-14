"""
ml/model_rolling/state_tracker.py
===================================
Model lifecycle state machine and rollout event log.

Manages the lifecycle of each model version from training through
production, with immutable event sourcing for auditability.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RolloutStage(Enum):
    """Where a model version is in its lifecycle."""
    TRAINED      = "trained"
    SHADOW       = "shadow"       # running in parallel, predictions logged
    CANARY_10Pct = "canary_10"    # 10 % live traffic
    CANARY_25Pct = "canary_25"    # 25 % live traffic
    CANARY_50Pct = "canary_50"    # 50 % live traffic
    PRODUCTION   = "production"
    ROLLED_BACK  = "rolled_back"
    DEPRECATED   = "deprecated"


@dataclass
class RolloutEvent:
    """Immutable record of a lifecycle transition."""
    event_id   : str
    model_name : str
    version    : str
    from_stage : Optional[RolloutStage]
    to_stage   : RolloutStage
    timestamp  : float
    reason     : str
    metadata   : Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id"  : self.event_id,
            "model_name": self.model_name,
            "version"   : self.version,
            "from_stage": self.from_stage.value if self.from_stage else None,
            "to_stage"  : self.to_stage.value,
            "timestamp" : self.timestamp,
            "reason"    : self.reason,
            "metadata"  : self.metadata,
        }


class ModelLifecycleState:
    """
    In-memory + serialisable state for one model name's current version.

    Maintains an ordered event log so the full history is replayable.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name     = model_name
        self.events: List[RolloutEvent] = []
        self._current: Dict[str, str] = {}   # version → RolloutStage.value
        self._version_alias: Dict[str, str] = {
            "champion": "",   # currently serving production traffic
            "challenger": "", # next candidate (shadow or canary)
            "previous": "",   # last champion (for rollback)
        }

    # ------------------------------------------------------------------ API

    def current_champion(self) -> Optional[str]:
        return self._version_alias.get("champion") or None

    def current_champion_stage(self) -> Optional[RolloutStage]:
        v = self.current_champion()
        return RolloutStage(self._current.get(v, "trained")) if v else None

    def challenger(self) -> Optional[str]:
        return self._version_alias.get("challenger") or None

    def previous(self) -> Optional[str]:
        return self._version_alias.get("previous") or None

    def stage_of(self, version: str) -> Optional[RolloutStage]:
        s = self._current.get(version)
        return RolloutStage(s) if s else None

    def all_versions(self) -> List[str]:
        return list(self._current.keys())

    def history(self) -> List[RolloutEvent]:
        return list(self.events)

    # ------------------------------------------------------------------ transitions

    def _add_event(
        self,
        version   : str,
        from_stage: Optional[RolloutStage],
        to_stage  : RolloutStage,
        reason    : str,
        metadata  : Optional[Dict[str, Any]] = None,
    ) -> RolloutEvent:
        ev = RolloutEvent(
            event_id  = uuid.uuid4().hex[:12],
            model_name= self.model_name,
            version   = version,
            from_stage= from_stage,
            to_stage = to_stage,
            timestamp= time.time(),
            reason   = reason,
            metadata = metadata or {},
        )
        self.events.append(ev)
        self._current[version] = to_stage.value
        logger.info(
            "ModelLifecycle [%s] %s %s: %s → %s | %s",
            self.model_name, version,
            from_stage.value if from_stage else "?", to_stage.value, reason,
        )
        return ev

    def promote_to_shadow(self, version: str) -> RolloutEvent:
        prev = self.stage_of(version)
        ev = self._add_event(version, prev, RolloutStage.SHADOW, "shadow_promotion")
        self._version_alias["challenger"] = version
        return ev

    def promote_to_canary(
        self,
        version  : str,
        fraction : float,
    ) -> RolloutEvent:
        stage_map: Dict[float, RolloutStage] = {
            0.10: RolloutStage.CANARY_10Pct,
            0.25: RolloutStage.CANARY_25Pct,
            0.50: RolloutStage.CANARY_50Pct,
        }
        stage = stage_map.get(fraction, RolloutStage.CANARY_10Pct)
        prev = self.stage_of(version)
        ev = self._add_event(version, prev, stage, f"canary_{int(fraction*100)}pct")
        return ev

    def promote_to_production(self, version: str) -> RolloutEvent:
        prev_stage = self.stage_of(version)
        # Demote current champion to previous
        champ = self.current_champion()
        if champ:
            self._add_event(champ, RolloutStage.PRODUCTION, RolloutStage.DEPRECATED,
                            "champion_replaced")
            self._version_alias["previous"] = champ
        ev = self._add_event(version, prev_stage, RolloutStage.PRODUCTION,
                              "promotion")
        self._version_alias["champion"] = version
        self._version_alias["challenger"] = ""
        return ev

    def rollback_to(self, version: str) -> RolloutEvent:
        prev = self.stage_of(version)
        ev = self._add_event(version, prev, RolloutStage.PRODUCTION,
                              "manual_rollback")
        # Demote current champion
        champ = self.current_champion()
        if champ:
            self._add_event(champ, RolloutStage.PRODUCTION, RolloutStage.ROLLED_BACK,
                            "rollback_triggered")
        self._version_alias["champion"]  = version
        self._version_alias["challenger"] = ""
        self._version_alias["previous"] = champ or ""
        return ev

    def mark_rolled_back(self, version: str) -> RolloutEvent:
        prev = self.stage_of(version)
        return self._add_event(version, prev, RolloutStage.ROLLED_BACK,
                               "automatic_rollback")

    def deprecate(self, version: str) -> RolloutEvent:
        prev = self.stage_of(version)
        return self._add_event(version, prev, RolloutStage.DEPRECATED,
                               "deprecated")

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name"     : self.model_name,
            "champion"      : self.current_champion(),
            "challenger"    : self.challenger(),
            "previous"      : self.previous(),
            "versions"      : {v: s.value for v, s in [
                (v, RolloutStage(s_val)) for v, s_val in self._current.items()
            ]},
            "events"         : [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelLifecycleState":
        state = cls(data["model_name"])
        for v, s in data.get("versions", {}).items():
            state._current[v] = s
        state._version_alias["champion"]   = data.get("champion", "") or ""
        state._version_alias["challenger"] = data.get("challenger", "") or ""
        state._version_alias["previous"]   = data.get("previous", "") or ""
        for e_data in data.get("events", []):
            state.events.append(RolloutEvent(
                event_id  = e_data["event_id"],
                model_name= e_data["model_name"],
                version   = e_data["version"],
                from_stage= RolloutStage(e_data["from_stage"]) if e_data["from_stage"] else None,
                to_stage  = RolloutStage(e_data["to_stage"]),
                timestamp = e_data["timestamp"],
                reason    = e_data["reason"],
                metadata  = e_data.get("metadata", {}),
            ))
        return state
