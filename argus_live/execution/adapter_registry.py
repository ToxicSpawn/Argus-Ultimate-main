from __future__ import annotations

from argus_live.execution.venue_adapter import VenueAdapter


class AdapterRegistry:
    def __init__(self, adapters: dict[str, VenueAdapter]) -> None:
        self.adapters = adapters

    def get(self, venue: str) -> VenueAdapter:
        if venue not in self.adapters:
            raise KeyError(f"no adapter registered for venue={venue}")
        return self.adapters[venue]
