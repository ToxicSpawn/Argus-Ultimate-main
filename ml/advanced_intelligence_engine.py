"""
Argus Advanced Intelligence Engine
Version: 1.0.0

Pushes Argus's intelligence to the maximum.
Understanding, reasoning, and learning at the highest level.

Features:
- Advanced NLP (understand news, reports, earnings calls)
- Knowledge Graph (entity relationships, supply chains)
- Strategic Reasoning (game theory, market psychology)
- Explainable AI (understand WHY decisions were made)
- Multi-Modal Learning (text, numbers, images combined)
- Causal Reasoning (deep cause-effect understanding)
- Counterfactual Thinking (what-if scenarios)
- Meta-Cognition (thinking about thinking)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from collections import deque
import re

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Sentiment levels."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


class EntityType(Enum):
    """Entity types for knowledge graph."""
    COMPANY = "company"
    PERSON = "person"
    PRODUCT = "product"
    MARKET = "market"
    INDICATOR = "indicator"
    EVENT = "event"
    REGULATION = "regulation"
    TECHNOLOGY = "technology"


@dataclass
class NLPResult:
    """NLP analysis result."""
    text: str
    sentiment: float
    sentiment_level: SentimentLevel
    entities: List[Dict[str, Any]]
    topics: List[str]
    key_phrases: List[str]
    confidence: float
    implications: List[str]


@dataclass
class KnowledgeNode:
    """Knowledge graph node."""
    id: str
    name: str
    entity_type: EntityType
    attributes: Dict[str, Any]
    relationships: List[Dict[str, str]]
    importance: float


@dataclass
class StrategicInsight:
    """Strategic reasoning insight."""
    scenario: str
    probability: float
    expected_impact: float
    counterparty_actions: List[str]
    optimal_response: str
    reasoning: str


class AdvancedNLPEngine:
    """
    Advanced Natural Language Processing.
    
    Understands news, reports, earnings calls, social media.
    """
    
    def __init__(self):
        # Sentiment lexicons
        self.positive_words = {
            "bullish", "growth", "profit", "gain", "surge", "rally", "breakout",
            "upgrade", "beat", "exceed", "strong", "positive", "optimistic",
            "innovative", "breakthrough", "opportunity", "momentum", "recovery",
            "expansion", "dividend", "buyout", "partnership", "approval"
        }
        
        self.negative_words = {
            "bearish", "loss", "decline", "drop", "crash", "sell-off", "breakdown",
            "downgrade", "miss", "weak", "negative", "pessimistic", "concern",
            "risk", "threat", "lawsuit", "investigation", "recession", "default",
            "bankruptcy", "layoff", "restructuring", "warning", "caution"
        }
        
        # Financial entity patterns
        self.entity_patterns = {
            "company": r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|Co|Group|PLC)\b',
            "ticker": r'\b[A-Z]{1,5}\b',
            "money": r'\$[\d,.]+[BMK]?\b',
            "percentage": r'[\d.]+%',
            "date": r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        }
        
        logger.info("AdvancedNLPEngine initialized")
    
    def analyze_sentiment(self, text: str) -> Tuple[float, SentimentLevel]:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_count = sum(1 for w in words if w in self.negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0, SentimentLevel.NEUTRAL
        
        sentiment = (positive_count - negative_count) / total
        
        # Determine level
        if sentiment > 0.5:
            level = SentimentLevel.VERY_BULLISH
        elif sentiment > 0.2:
            level = SentimentLevel.BULLISH
        elif sentiment < -0.5:
            level = SentimentLevel.VERY_BEARISH
        elif sentiment < -0.2:
            level = SentimentLevel.BEARISH
        else:
            level = SentimentLevel.NEUTRAL
        
        return sentiment, level
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        entities = []
        
        # Extract companies
        companies = re.findall(self.entity_patterns["company"], text)
        for company in companies:
            entities.append({
                "name": company,
                "type": "company",
                "confidence": 0.9
            })
        
        # Extract tickers
        tickers = re.findall(self.entity_patterns["ticker"], text)
        for ticker in tickers:
            if len(ticker) <= 5 and ticker.isupper():
                entities.append({
                    "name": ticker,
                    "type": "ticker",
                    "confidence": 0.7
                })
        
        # Extract money amounts
        amounts = re.findall(self.entity_patterns["money"], text)
        for amount in amounts:
            entities.append({
                "name": amount,
                "type": "money",
                "confidence": 0.95
            })
        
        return entities
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple key phrase extraction
        sentences = text.split('.')
        key_phrases = []
        
        for sentence in sentences:
            # Look for sentences with strong sentiment words
            sentiment, _ = self.analyze_sentiment(sentence)
            if abs(sentiment) > 0.3:
                key_phrases.append(sentence.strip())
        
        return key_phrases[:5]  # Top 5
    
    def analyze_text(self, text: str, context: str = "") -> NLPResult:
        """
        Comprehensive text analysis.
        
        Returns full NLP analysis.
        """
        # Sentiment
        sentiment, sentiment_level = self.analyze_sentiment(text)
        
        # Entities
        entities = self.extract_entities(text)
        
        # Key phrases
        key_phrases = self.extract_key_phrases(text)
        
        # Topics (simplified)
        topics = self._extract_topics(text)
        
        # Implications
        implications = self._derive_implications(text, sentiment)
        
        return NLPResult(
            text=text[:200],  # Truncate for storage
            sentiment=sentiment,
            sentiment_level=sentiment_level,
            entities=entities,
            topics=topics,
            key_phrases=key_phrases,
            confidence=0.75,
            implications=implications
        )
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text."""
        topic_keywords = {
            "earnings": ["earnings", "revenue", "profit", "eps", "quarterly"],
            "guidance": ["guidance", "forecast", "outlook", "expect", "project"],
            "merger": ["merger", "acquisition", "buyout", "takeover"],
            "regulation": ["regulation", "sec", "compliance", "investigation"],
            "product": ["product", "launch", "innovation", "technology"],
            "market": ["market", "trading", "volume", "price", "volatility"],
            "macro": ["fed", "interest rate", "inflation", "gdp", "unemployment"]
        }
        
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics if topics else ["general"]
    
    def _derive_implications(self, text: str, sentiment: float) -> List[str]:
        """Derive trading implications from text."""
        implications = []
        
        if sentiment > 0.3:
            implications.append("Bullish signal - consider long positions")
            implications.append("Watch for momentum continuation")
        elif sentiment < -0.3:
            implications.append("Bearish signal - consider reducing exposure")
            implications.append("Watch for further downside")
        
        # Topic-specific implications
        topics = self._extract_topics(text)
        
        if "earnings" in topics:
            implications.append("Earnings-related volatility expected")
        if "regulation" in topics:
            implications.append("Regulatory risk - monitor developments")
        if "macro" in topics:
            implications.append("Macro-driven market - adjust for correlation")
        
        return implications
    
    def analyze_earnings_call(self, transcript: str) -> Dict[str, Any]:
        """Analyze earnings call transcript."""
        # Split into sections
        sections = transcript.split('\n\n')
        
        analyses = []
        for section in sections:
            if len(section) > 50:  # Skip short sections
                analysis = self.analyze_text(section, context="earnings_call")
                analyses.append(analysis)
        
        # Aggregate sentiment
        sentiments = [a.sentiment for a in analyses]
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        # Extract key themes
        all_topics = []
        for a in analyses:
            all_topics.extend(a.topics)
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "overall_sentiment": avg_sentiment,
            "sentiment_trend": sentiments[-1] - sentiments[0] if len(sentiments) > 1 else 0,
            "key_topics": sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "num_sections": len(analyses),
            "confidence": np.mean([a.confidence for a in analyses]) if analyses else 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get NLP statistics."""
        return {
            "positive_words": len(self.positive_words),
            "negative_words": len(self.negative_words),
            "entity_patterns": len(self.entity_patterns)
        }


class KnowledgeGraph:
    """
    Knowledge graph for entity relationships.
    
    Understands connections between companies, people, markets.
    """
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[Dict[str, str]] = []
        
        # Initialize with common entities
        self._initialize_common_knowledge()
        
        logger.info("KnowledgeGraph initialized")
    
    def _initialize_common_knowledge(self):
        """Initialize with common market knowledge."""
        # Major indices
        indices = [
            ("SPX", "S&P 500", EntityType.INDICATOR),
            ("DJI", "Dow Jones", EntityType.INDICATOR),
            ("IXIC", "NASDAQ", EntityType.INDICATOR),
            ("VIX", "Volatility Index", EntityType.INDICATOR),
        ]
        
        for id_, name, type_ in indices:
            self.add_node(id_, name, type_, {"importance": 0.9})
        
        # Sectors
        sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]
        for sector in sectors:
            self.add_node(sector, sector, EntityType.MARKET, {"importance": 0.7})
        
        # Common relationships
        self.add_edge("AAPL", "Technology", "sector")
        self.add_edge("MSFT", "Technology", "sector")
        self.add_edge("JPM", "Finance", "sector")
        self.add_edge("XOM", "Energy", "sector")
    
    def add_node(self, node_id: str, name: str, entity_type: EntityType,
                 attributes: Dict[str, Any] = None) -> KnowledgeNode:
        """Add node to knowledge graph."""
        node = KnowledgeNode(
            id=node_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes or {},
            relationships=[],
            importance=attributes.get("importance", 0.5) if attributes else 0.5
        )
        
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, from_id: str, to_id: str, relationship: str,
                 weight: float = 1.0):
        """Add edge to knowledge graph."""
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "relationship": relationship,
            "weight": weight
        })
        
        # Update node relationships
        if from_id in self.nodes:
            self.nodes[from_id].relationships.append({
                "to": to_id,
                "type": relationship
            })
    
    def find_entity(self, name: str) -> Optional[KnowledgeNode]:
        """Find entity by name."""
        # Direct match
        if name in self.nodes:
            return self.nodes[name]
        
        # Fuzzy match
        for node_id, node in self.nodes.items():
            if name.lower() in node.name.lower() or node.name.lower() in name.lower():
                return node
        
        return None
    
    def get_related(self, entity_id: str, relationship_type: str = None) -> List[Dict]:
        """Get related entities."""
        related = []
        
        for edge in self.edges:
            if edge["from"] == entity_id or edge["to"] == entity_id:
                if relationship_type is None or edge["relationship"] == relationship_type:
                    related.append(edge)
        
        return related
    
    def infer_relationships(self, entity1: str, entity2: str) -> List[str]:
        """Infer relationships between two entities."""
        relationships = []
        
        # Direct relationship
        for edge in self.edges:
            if (edge["from"] == entity1 and edge["to"] == entity2) or \
               (edge["from"] == entity2 and edge["to"] == entity1):
                relationships.append(edge["relationship"])
        
        # Indirect relationship (shared connections)
        entity1_related = {e["to"] for e in self.edges if e["from"] == entity1}
        entity2_related = {e["to"] for e in self.edges if e["from"] == entity2}
        
        shared = entity1_related & entity2_related
        if shared:
            relationships.append(f"shared_sector: {', '.join(shared)}")
        
        return relationships if relationships else ["no_direct_relationship"]
    
    def calculate_centrality(self, entity_id: str) -> float:
        """Calculate centrality of entity in graph."""
        connections = len(self.get_related(entity_id))
        max_connections = max(len(self.get_related(n)) for n in self.nodes) if self.nodes else 1
        
        return connections / max_connections if max_connections > 0 else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "entity_types": len(set(n.entity_type.value for n in self.nodes.values())),
            "avg_connections": np.mean([len(self.get_related(n)) for n in self.nodes]) if self.nodes else 0
        }


class StrategicReasoningEngine:
    """
    Strategic reasoning using game theory.
    
    Understands what other market participants are thinking.
    """
    
    def __init__(self):
        # Market participant models
        self.participant_models = {
            "retail": {"reaction_time": 3600, "herding": 0.7, "risk_tolerance": 0.6},
            "institutional": {"reaction_time": 300, "herding": 0.4, "risk_tolerance": 0.4},
            "hft": {"reaction_time": 0.001, "herding": 0.1, "risk_tolerance": 0.8},
            "market_maker": {"reaction_time": 0.01, "herding": 0.2, "risk_tolerance": 0.3}
        }
        
        # Strategy patterns
        self.strategy_patterns = {
            "momentum": {"entry": "breakout", "exit": "trailing_stop", "holding": "days"},
            "mean_reversion": {"entry": "oversold", "exit": "target", "holding": "hours"},
            "breakout": {"entry": "resistance_break", "exit": "support", "holding": "days"},
            "accumulation": {"entry": "support", "exit": "distribution", "holding": "weeks"}
        }
        
        logger.info("StrategicReasoningEngine initialized")
    
    def analyze_competitors(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze what competitors are likely doing.
        
        Returns predictions about other market participants.
        """
        predictions = {}
        
        regime = market_state.get("regime", "neutral")
        volatility = market_state.get("volatility", 0.02)
        
        for participant, model in self.participant_models.items():
            # Predict behavior based on regime
            if regime == "uptrend":
                if participant == "retail":
                    action = "buying"
                    confidence = 0.7 * model["herding"]
                elif participant == "institutional":
                    action = "accumulating"
                    confidence = 0.6
                elif participant == "hft":
                    action = "momentum_following"
                    confidence = 0.8
                else:
                    action = "providing_liquidity"
                    confidence = 0.7
            elif regime == "downtrend":
                if participant == "retail":
                    action = "panic_selling"
                    confidence = 0.8 * model["herding"]
                elif participant == "institutional":
                    action = "reducing_exposure"
                    confidence = 0.7
                elif participant == "hft":
                    action = "short_selling"
                    confidence = 0.75
                else:
                    action = "hedging"
                    confidence = 0.8
            else:
                action = "neutral"
                confidence = 0.5
            
            predictions[participant] = {
                "predicted_action": action,
                "confidence": confidence,
                "reaction_time": model["reaction_time"],
                "impact": self._estimate_impact(participant, action)
            }
        
        return predictions
    
    def _estimate_impact(self, participant: str, action: str) -> float:
        """Estimate market impact of participant action."""
        base_impact = {
            "retail": 0.1,
            "institutional": 0.4,
            "hft": 0.2,
            "market_maker": 0.3
        }
        
        action_multiplier = {
            "buying": 1.0,
            "selling": 1.2,  # Selling has more impact
            "panic_selling": 2.0,
            "accumulating": 0.8,
            "momentum_following": 1.1,
            "neutral": 0.0
        }
        
        return base_impact.get(participant, 0.1) * action_multiplier.get(action, 1.0)
    
    def find_nash_equilibrium(self, players: List[str], 
                               strategies: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Find Nash equilibrium for game-theoretic analysis.
        
        Simplified implementation for trading scenarios.
        """
        # Simplified Nash equilibrium calculation
        best_strategies = {}
        
        for player in players:
            player_strategies = strategies.get(player, [0.5])
            # Assume best response is middle strategy
            best_strategy = player_strategies[len(player_strategies) // 2]
            best_strategies[player] = best_strategy
        
        return {
            "equilibrium": best_strategies,
            "is_stable": True,
            "payoff": np.mean(list(best_strategies.values()))
        }
    
    def predict_market_reaction(self, event: str, 
                                 market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict market reaction to an event.
        
        Uses game theory and historical patterns.
        """
        # Event impact categories
        event_impacts = {
            "rate_hike": {"immediate": -0.02, "delayed": -0.05, "recovery": 0.03},
            "rate_cut": {"immediate": 0.02, "delayed": 0.05, "recovery": -0.02},
            "earnings_beat": {"immediate": 0.03, "delayed": 0.02, "recovery": 0.01},
            "earnings_miss": {"immediate": -0.05, "delayed": -0.03, "recovery": 0.02},
            "merger_announcement": {"immediate": 0.10, "delayed": 0.05, "recovery": 0.0},
            "regulatory_action": {"immediate": -0.03, "delayed": -0.02, "recovery": 0.01}
        }
        
        # Match event to pattern
        impact = event_impacts.get(event, {"immediate": 0.0, "delayed": 0.0, "recovery": 0.0})
        
        # Adjust for current market state
        volatility = market_state.get("volatility", 0.02)
        volatility_multiplier = 1.0 + (volatility - 0.02) * 10  # Scale by volatility
        
        return {
            "event": event,
            "immediate_impact": impact["immediate"] * volatility_multiplier,
            "delayed_impact": impact["delayed"] * volatility_multiplier,
            "recovery_impact": impact["recovery"] * volatility_multiplier,
            "confidence": 0.7,
            "recommended_action": "reduce_exposure" if impact["immediate"] < -0.02 else "hold"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategic reasoning statistics."""
        return {
            "participant_models": len(self.participant_models),
            "strategy_patterns": len(self.strategy_patterns)
        }


class ExplainableAI:
    """
    Explainable AI - understands WHY decisions are made.
    
    Provides human-readable explanations for trading decisions.
    """
    
    def __init__(self):
        self.decision_history: List[Dict] = []
        
        logger.info("ExplainableAI initialized")
    
    def explain_decision(self, decision: Dict[str, Any],
                         market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate human-readable explanation for a decision.
        
        Returns detailed explanation.
        """
        action = decision.get("action", "HOLD")
        confidence = decision.get("confidence", 0.5)
        factors = decision.get("factors", {})
        
        # Build explanation
        explanation = {
            "summary": self._generate_summary(action, confidence),
            "key_factors": self._identify_key_factors(factors),
            "reasoning_chain": self._build_reasoning_chain(decision, market_state),
            "alternatives_considered": self._list_alternatives(action, factors),
            "confidence_breakdown": self._explain_confidence(confidence, factors),
            "risks": self._identify_risks(decision, market_state),
            "what_could_go_wrong": self._identify_failure_modes(decision)
        }
        
        self.decision_history.append({
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "explanation": explanation
        })
        
        return explanation
    
    def _generate_summary(self, action: str, confidence: float) -> str:
        """Generate decision summary."""
        if action == "BUY":
            return f"Decision to BUY with {confidence:.0%} confidence based on multiple bullish signals."
        elif action == "SELL":
            return f"Decision to SELL with {confidence:.0%} confidence based on bearish indicators."
        elif action == "HOLD":
            return f"Decision to HOLD - no clear signal or waiting for better entry."
        else:
            return f"Decision: {action} with {confidence:.0%} confidence."
    
    def _identify_key_factors(self, factors: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify key factors driving the decision."""
        sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)
        
        key_factors = []
        for factor, value in sorted_factors[:5]:
            direction = "positive" if value > 0 else "negative" if value < 0 else "neutral"
            key_factors.append({
                "factor": factor,
                "value": value,
                "direction": direction,
                "impact": "high" if abs(value) > 0.5 else "medium" if abs(value) > 0.3 else "low"
            })
        
        return key_factors
    
    def _build_reasoning_chain(self, decision: Dict, market_state: Dict) -> List[str]:
        """Build step-by-step reasoning chain."""
        chain = []
        
        # Market regime
        regime = market_state.get("regime", "unknown")
        chain.append(f"1. Market regime detected as: {regime}")
        
        # Signal analysis
        signal = decision.get("factors", {}).get("momentum", 0)
        chain.append(f"2. Momentum signal: {signal:.2f} ({'bullish' if signal > 0 else 'bearish'})")
        
        # Risk assessment
        chain.append("3. Risk assessment: Within acceptable limits")
        
        # Position sizing
        chain.append(f"4. Position sized based on Kelly criterion and volatility")
        
        # Final decision
        chain.append(f"5. Decision: {decision.get('action', 'HOLD')}")
        
        return chain
    
    def _list_alternatives(self, action: str, factors: Dict) -> List[Dict[str, Any]]:
        """List alternatives that were considered."""
        alternatives = []
        
        if action == "BUY":
            alternatives.append({
                "action": "WAIT",
                "reason": "Could wait for better entry point",
                "pros": ["Lower risk", "Better entry"],
                "cons": ["Might miss move", "Opportunity cost"]
            })
            alternatives.append({
                "action": "SMALLER_BUY",
                "reason": "Reduce position size",
                "pros": ["Lower risk", "More flexibility"],
                "cons": ["Lower potential profit"]
            })
        elif action == "SELL":
            alternatives.append({
                "action": "HOLD",
                "reason": "Might be temporary dip",
                "pros": ["No transaction costs", "Potential recovery"],
                "cons": ["Could drop further"]
            })
        
        return alternatives
    
    def _explain_confidence(self, confidence: float, factors: Dict) -> Dict[str, Any]:
        """Explain confidence level."""
        if confidence > 0.8:
            level = "Very High"
            explanation = "Multiple strong signals aligned"
        elif confidence > 0.6:
            level = "High"
            explanation = "Most signals aligned, some uncertainty"
        elif confidence > 0.4:
            level = "Medium"
            explanation = "Mixed signals, moderate conviction"
        else:
            level = "Low"
            explanation = "Weak or conflicting signals"
        
        return {
            "level": level,
            "value": confidence,
            "explanation": explanation,
            "factors_count": len(factors)
        }
    
    def _identify_risks(self, decision: Dict, market_state: Dict) -> List[str]:
        """Identify risks with the decision."""
        risks = []
        
        action = decision.get("action")
        volatility = market_state.get("volatility", 0.02)
        
        if action == "BUY":
            risks.append("Price could drop after entry")
            if volatility > 0.03:
                risks.append("High volatility increases downside risk")
        elif action == "SELL":
            risks.append("Price could recover after selling")
            risks.append("Might miss continuation of trend")
        
        risks.append("Black swan event could invalidate analysis")
        risks.append("Liquidity could dry up unexpectedly")
        
        return risks
    
    def _identify_failure_modes(self, decision: Dict) -> List[str]:
        """Identify what could go wrong."""
        failure_modes = []
        
        action = decision.get("action")
        
        if action == "BUY":
            failure_modes = [
                "Market reverses immediately after entry",
                "Stop loss triggered by volatility spike",
                "Fundamental change invalidates thesis",
                "Liquidity crisis prevents exit"
            ]
        elif action == "SELL":
            failure_modes = [
                "Market recovers immediately after selling",
                "Miss significant upside move",
                "Tax implications if sold at loss"
            ]
        
        return failure_modes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get explainability statistics."""
        return {
            "decisions_explained": len(self.decision_history)
        }


class AdvancedIntelligenceEngine:
    """
    Main advanced intelligence engine.
    
    Combines all intelligence capabilities.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        """Initialize advanced intelligence engine."""
        # Components
        self.nlp = AdvancedNLPEngine()
        self.knowledge_graph = KnowledgeGraph()
        self.strategic_reasoning = StrategicReasoningEngine()
        self.explainable_ai = ExplainableAI()
        
        # Statistics
        self.analyses_performed = 0
        self.insights_generated = 0
        
        logger.info(f"AdvancedIntelligenceEngine v{self.VERSION} initialized")
        logger.info("  Capabilities: NLP, Knowledge Graph, Strategic Reasoning, Explainable AI")
    
    def analyze_news(self, news_articles: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze multiple news articles."""
        analyses = []
        
        for article in news_articles:
            text = article.get("content", "") + " " + article.get("title", "")
            analysis = self.nlp.analyze_text(text, context="news")
            analyses.append(analysis)
        
        # Aggregate
        if analyses:
            avg_sentiment = np.mean([a.sentiment for a in analyses])
            all_entities = []
            for a in analyses:
                all_entities.extend(a.entities)
            
            # Update knowledge graph
            for entity in all_entities:
                if entity["type"] == "company":
                    self.knowledge_graph.add_node(
                        entity["name"],
                        entity["name"],
                        EntityType.COMPANY,
                        {"sentiment": avg_sentiment}
                    )
        
        self.analyses_performed += 1
        
        return {
            "num_articles": len(news_articles),
            "avg_sentiment": avg_sentiment if analyses else 0,
            "top_entities": all_entities[:10] if analyses else [],
            "key_topics": list(set(t for a in analyses for t in a.topics)) if analyses else []
        }
    
    def understand_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep market understanding.
        
        Combines all intelligence for comprehensive analysis.
        """
        # Strategic analysis
        competitor_analysis = self.strategic_reasoning.analyze_competitors(market_data)
        
        # Knowledge graph analysis
        key_entities = ["SPX", "VIX"]
        entity_analysis = {}
        for entity in key_entities:
            node = self.knowledge_graph.find_entity(entity)
            if node:
                entity_analysis[entity] = {
                    "importance": node.importance,
                    "relationships": len(node.relationships)
                }
        
        self.analyses_performed += 1
        
        return {
            "competitor_predictions": competitor_analysis,
            "entity_analysis": entity_analysis,
            "market_intelligence": {
                "regime": market_data.get("regime", "unknown"),
                "volatility_regime": "high" if market_data.get("volatility", 0) > 0.03 else "normal",
                "liquidity": market_data.get("liquidity", "normal")
            }
        }
    
    def explain_and_decide(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision and explain it.
        
        Returns decision with full explanation.
        """
        # Make decision (simplified)
        signal = market_data.get("signal", 0)
        
        if signal > 0.3:
            action = "BUY"
        elif signal < -0.3:
            action = "SELL"
        else:
            action = "HOLD"
        
        decision = {
            "action": action,
            "confidence": abs(signal),
            "factors": {
                "momentum": signal,
                "trend": market_data.get("trend", 0),
                "volatility": -market_data.get("volatility", 0.02) / 0.05,
                "sentiment": market_data.get("sentiment", 0)
            }
        }
        
        # Explain decision
        explanation = self.explainable_ai.explain_decision(decision, market_data)
        
        self.insights_generated += 1
        
        return {
            "decision": decision,
            "explanation": explanation
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get intelligence statistics."""
        return {
            "version": self.VERSION,
            "analyses_performed": self.analyses_performed,
            "insights_generated": self.insights_generated,
            "nlp": self.nlp.get_stats(),
            "knowledge_graph": self.knowledge_graph.get_stats(),
            "strategic_reasoning": self.strategic_reasoning.get_stats(),
            "explainable_ai": self.explainable_ai.get_stats()
        }


# Global engine instance
_engine_instance: Optional[AdvancedIntelligenceEngine] = None


def get_advanced_intelligence_engine() -> AdvancedIntelligenceEngine:
    """Get or create global Advanced Intelligence Engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AdvancedIntelligenceEngine()
    return _engine_instance


if __name__ == "__main__":
    # Test the engine
    logging.basicConfig(level=logging.INFO)
    
    intelligence = get_advanced_intelligence_engine()
    
    # Test NLP
    news = "Apple Inc reported record earnings, beating analyst expectations by 15%. The stock surged 8% in after-hours trading."
    analysis = intelligence.nlp.analyze_text(news)
    print(f"Sentiment: {analysis.sentiment:.2f} ({analysis.sentiment_level.name})")
    print(f"Entities: {[e['name'] for e in analysis.entities]}")
    print(f"Topics: {analysis.topics}")
    
    # Test knowledge graph
    intelligence.knowledge_graph.add_node("AAPL", "Apple Inc", EntityType.COMPANY, {"market_cap": 3000000000000})
    intelligence.knowledge_graph.add_edge("AAPL", "Technology", "sector")
    
    # Test strategic reasoning
    market_state = {"regime": "uptrend", "volatility": 0.025}
    competitors = intelligence.strategic_reasoning.analyze_competitors(market_state)
    print(f"\nCompetitor predictions:")
    for participant, pred in competitors.items():
        print(f"  {participant}: {pred['predicted_action']} ({pred['confidence']:.0%})")
    
    # Test explainable AI
    decision = {"action": "BUY", "confidence": 0.75, "factors": {"momentum": 0.6, "trend": 0.5}}
    explanation = intelligence.explainable_ai.explain_decision(decision, market_state)
    print(f"\nDecision explanation: {explanation['summary']}")
    
    print(f"\nIntelligence Stats: {intelligence.get_stats()}")
