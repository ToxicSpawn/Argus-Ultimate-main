from __future__ import annotations

from dataclasses import dataclass, field

from argus_live.portfolio.target_models import TargetProposal


@dataclass
class TargetRegistry:
    proposals: list[TargetProposal] = field(default_factory=list)

    def add(self, proposal: TargetProposal) -> None:
        self.proposals.append(proposal)
