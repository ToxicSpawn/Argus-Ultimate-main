"""
DAO Governance Tracker
======================
Tracks DAO governance for alpha signals:
- Proposal monitoring
- Voting analysis
- Delegate tracking
- Governance token price impact
- Proposal outcome prediction
- Smart money voting patterns

Supports: Snapshot, Tally, Governor Bravo, OpenZeppelin Governor
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ProposalState(Enum):
    """Proposal states."""
    PENDING = "pending"
    ACTIVE = "active"
    PASSED = "passed"
    FAILED = "failed"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    QUEUED = "queued"


class VoteDirection(Enum):
    """Vote directions."""
    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"


class GovernancePlatform(Enum):
    """Governance platforms."""
    SNAPSHOT = "snapshot"
    TALLY = "tally"
    GOVERNOR_BRAVO = "governor_bravo"
    OZ_GOVERNOR = "oz_governor"


@dataclass
class Proposal:
    """DAO proposal."""
    proposal_id: str
    dao_name: str
    title: str
    description: str
    state: ProposalState
    platform: GovernancePlatform
    for_votes: float = 0.0
    against_votes: float = 0.0
    abstain_votes: float = 0.0
    quorum: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time: Optional[float] = None
    proposer: str = ""
    categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Vote:
    """Individual vote."""
    voter: str
    proposal_id: str
    direction: VoteDirection
    weight: float
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    delegated: bool = False


@dataclass
class Delegate:
    """DAO delegate."""
    address: str
    dao_name: str
    voting_power: float
    votes_cast: int = 0
    participation_rate: float = 0.0
    alignment_score: float = 0.0  # How often they vote with majority
    delegators: int = 0
    last_active: float = 0.0


@dataclass
class GovernanceSignal:
    """Governance-based trading signal."""
    dao_name: str
    proposal_id: str
    signal_type: str  # "bullish", "bearish", "neutral"
    confidence: float
    reason: str
    expected_impact_pct: float
    deadline: float


class ProposalTracker:
    """
    Proposal Tracker
    ================
    Tracks DAO proposals and their status.
    """
    
    def __init__(self):
        self.proposals: Dict[str, Proposal] = {}
        self.proposal_history: Dict[str, List[Dict[str, Any]]] = {}
        self.votes: Dict[str, List[Vote]] = {}  # proposal_id -> votes
    
    def add_proposal(self, proposal: Proposal) -> None:
        """Add a proposal to tracking."""
        self.proposals[proposal.proposal_id] = proposal
        logger.info(f"Tracking proposal: {proposal.title[:50]}... ({proposal.dao_name})")
    
    def update_proposal_state(self, proposal_id: str, state: ProposalState) -> None:
        """Update proposal state."""
        if proposal_id in self.proposals:
            self.proposals[proposal_id].state = state
    
    def record_vote(self, vote: Vote) -> None:
        """Record a vote on a proposal."""
        if vote.proposal_id not in self.votes:
            self.votes[vote.proposal_id] = []
        self.votes[vote.proposal_id].append(vote)
    
    def get_active_proposals(self, dao_name: Optional[str] = None) -> List[Proposal]:
        """Get active proposals."""
        active = [
            p for p in self.proposals.values()
            if p.state == ProposalState.ACTIVE
        ]
        
        if dao_name:
            active = [p for p in active if p.dao_name == dao_name]
        
        return active
    
    def get_proposal_sentiment(self, proposal_id: str) -> Dict[str, Any]:
        """Get sentiment for a proposal."""
        if proposal_id not in self.votes:
            return {"sentiment": "unknown", "confidence": 0}
        
        votes = self.votes[proposal_id]
        
        for_votes = sum(v.weight for v in votes if v.direction == VoteDirection.FOR)
        against_votes = sum(v.weight for v in votes if v.direction == VoteDirection.AGAINST)
        abstain_votes = sum(v.weight for v in votes if v.direction == VoteDirection.ABSTAIN)
        
        total = for_votes + against_votes
        
        if total == 0:
            return {"sentiment": "unknown", "confidence": 0, "for_pct": 0, "against_pct": 0}
        
        for_pct = for_votes / total * 100
        against_pct = against_votes / total * 100
        
        if for_pct > 70:
            sentiment = "strongly_for"
        elif for_pct > 55:
            sentiment = "for"
        elif against_pct > 70:
            sentiment = "strongly_against"
        elif against_pct > 55:
            sentiment = "against"
        else:
            sentiment = "contested"
        
        return {
            "sentiment": sentiment,
            "for_votes": for_votes,
            "against_votes": against_votes,
            "abstain_votes": abstain_votes,
            "for_pct": for_pct,
            "against_pct": against_pct,
            "total_voters": len(votes)
        }
    
    def predict_outcome(self, proposal_id: str) -> Dict[str, Any]:
        """Predict proposal outcome."""
        sentiment = self.get_proposal_sentiment(proposal_id)
        
        if proposal_id not in self.proposals:
            return {"prediction": "unknown", "confidence": 0}
        
        proposal = self.proposals[proposal_id]
        
        # Check if quorum met
        total_votes = sentiment.get("for_votes", 0) + sentiment.get("against_votes", 0)
        quorum_met = total_votes >= proposal.quorum if proposal.quorum > 0 else True
        
        if not quorum_met:
            return {
                "prediction": "fail_quorum",
                "confidence": 0.8,
                "reason": "Quorum not met"
            }
        
        for_pct = sentiment.get("for_pct", 0)
        
        if for_pct > 60:
            prediction = "pass"
            confidence = min(for_pct / 100, 0.95)
        elif for_pct > 50:
            prediction = "likely_pass"
            confidence = 0.6
        elif for_pct < 40:
            prediction = "fail"
            confidence = min((100 - for_pct) / 100, 0.95)
        else:
            prediction = "uncertain"
            confidence = 0.5
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "for_pct": for_pct,
            "quorum_met": quorum_met
        }


class DelegateTracker:
    """
    Delegate Tracker
    ================
    Tracks delegate voting patterns.
    """
    
    def __init__(self):
        self.delegates: Dict[str, Delegate] = {}
        self.delegate_votes: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_delegate(self, delegate: Delegate) -> None:
        """Add delegate to tracking."""
        key = f"{delegate.dao_name}:{delegate.address}"
        self.delegates[key] = delegate
    
    def record_delegate_vote(self, delegate_key: str, vote: Dict[str, Any]) -> None:
        """Record delegate vote."""
        if delegate_key not in self.delegate_votes:
            self.delegate_votes[delegate_key] = []
        self.delegate_votes[delegate_key].append(vote)
        
        # Update delegate stats
        if delegate_key in self.delegates:
            delegate = self.delegates[delegate_key]
            delegate.votes_cast += 1
            delegate.last_active = time.time()
    
    def calculate_alignment(self, delegate_key: str, dao_name: str) -> float:
        """Calculate delegate alignment with winning side."""
        if delegate_key not in self.delegate_votes:
            return 0.5
        
        votes = self.delegate_votes[delegate_key]
        if not votes:
            return 0.5
        
        # Count how often delegate voted with majority
        aligned = sum(1 for v in votes if v.get("was_majority", False))
        
        return aligned / len(votes)
    
    def get_top_delegates(self, dao_name: str, n: int = 10) -> List[Delegate]:
        """Get top delegates by voting power."""
        dao_delegates = [
            d for d in self.delegates.values()
            if d.dao_name == dao_name
        ]
        
        return sorted(dao_delegates, key=lambda d: d.voting_power, reverse=True)[:n]


class GovernanceSignalGenerator:
    """
    Governance Signal Generator
    ===========================
    Generates trading signals from governance events.
    """
    
    def __init__(self):
        self.proposal_tracker = ProposalTracker()
        self.delegate_tracker = DelegateTracker()
        self.signals: List[GovernanceSignal] = []
        
        # Categories that typically impact price
        self.high_impact_categories = [
            "treasury", "tokenomics", "emissions", "fee",
            "partnership", "integration", "upgrade", "migration"
        ]
        
        self.low_impact_categories = [
            "governance", "process", "election", "community"
        ]
    
    def analyze_proposal_impact(self, proposal: Proposal) -> Optional[GovernanceSignal]:
        """Analyze potential price impact of a proposal."""
        # Check category impact
        category_impact = 0
        for category in proposal.categories:
            if category.lower() in self.high_impact_categories:
                category_impact += 0.3
            elif category.lower() in self.low_impact_categories:
                category_impact -= 0.1
        
        # Get current sentiment
        sentiment = self.proposal_tracker.get_proposal_sentiment(proposal.proposal_id)
        prediction = self.proposal_tracker.predict_outcome(proposal.proposal_id)
        
        # Determine signal
        for_pct = sentiment.get("for_pct", 50)
        will_pass = prediction.get("prediction", "") in ["pass", "likely_pass"]
        
        # Analyze proposal content for direction
        title_lower = proposal.title.lower()
        desc_lower = proposal.description.lower()
        
        # Bullish keywords
        bullish_keywords = [
            "burn", "buyback", "reduce supply", "increase demand",
            "partnership", "integration", "launch", "upgrade",
            "treasury diversification", "reward", "incentive"
        ]
        
        # Bearish keywords
        bearish_keywords = [
            "mint", "increase supply", "dilution", "sell",
            "unlock", "vesting", "emission increase", "inflation"
        ]
        
        bullish_score = sum(1 for kw in bullish_keywords if kw in title_lower or kw in desc_lower)
        bearish_score = sum(1 for kw in bearish_keywords if kw in title_lower or kw in desc_lower)
        
        # Generate signal
        if will_pass:
            if bullish_score > bearish_score:
                signal_type = "bullish"
                expected_impact = min(bullish_score * 2, 15)
            elif bearish_score > bullish_score:
                signal_type = "bearish"
                expected_impact = -min(bearish_score * 2, 15)
            else:
                signal_type = "neutral"
                expected_impact = 0
        else:
            # Failed proposals often have opposite effect
            if bullish_score > bearish_score:
                signal_type = "bearish"
                expected_impact = -2
            else:
                signal_type = "neutral"
                expected_impact = 0
        
        confidence = prediction.get("confidence", 0.5) * (1 + category_impact)
        
        if signal_type == "neutral" and abs(expected_impact) < 1:
            return None
        
        signal = GovernanceSignal(
            dao_name=proposal.dao_name,
            proposal_id=proposal.proposal_id,
            signal_type=signal_type,
            confidence=min(confidence, 0.9),
            reason=f"Proposal '{proposal.title[:50]}...' predicted to {prediction.get('prediction')}",
            expected_impact_pct=expected_impact,
            deadline=proposal.end_time
        )
        
        self.signals.append(signal)
        return signal
    
    def get_active_signals(self) -> List[GovernanceSignal]:
        """Get active governance signals."""
        return [
            s for s in self.signals
            if s.deadline > time.time()
        ]
    
    def get_dao_summary(self, dao_name: str) -> Dict[str, Any]:
        """Get summary for a DAO."""
        proposals = [
            p for p in self.proposal_tracker.proposals.values()
            if p.dao_name == dao_name
        ]
        
        active = [p for p in proposals if p.state == ProposalState.ACTIVE]
        passed = [p for p in proposals if p.state in [ProposalState.PASSED, ProposalState.EXECUTED]]
        failed = [p for p in proposals if p.state == ProposalState.FAILED]
        
        return {
            "dao_name": dao_name,
            "total_proposals": len(proposals),
            "active_proposals": len(active),
            "passed_proposals": len(passed),
            "failed_proposals": len(failed),
            "pass_rate": len(passed) / max(len(proposals), 1) * 100,
            "active_signals": len([s for s in self.get_active_signals() if s.dao_name == dao_name])
        }


# Export
__all__ = [
    "ProposalState",
    "VoteDirection",
    "GovernancePlatform",
    "Proposal",
    "Vote",
    "Delegate",
    "GovernanceSignal",
    "ProposalTracker",
    "DelegateTracker",
    "GovernanceSignalGenerator"
]
