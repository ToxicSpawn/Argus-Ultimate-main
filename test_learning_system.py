"""
Direct test script for the real-time learning system
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.real_time_learning.orchestrator import (
    RealTimeLearningOrchestrator,
    LearningComponent,
    SafetyValidator,
    LearningAudit
)
from core.real_time_learning.example_component import AdaptiveVolatilityCluster


class TestLearningComponent(LearningComponent):
    """Simple test component"""

    def __init__(self, name: str):
        self.name = name
        self.params = {"test_param": 1.0}
        self.update_count = 0

    def learn(self, data: dict) -> dict:
        self.update_count += 1
        self.params["test_param"] += 0.1
        return self.params

    def get_params(self) -> dict:
        return self.params

    def rollback(self) -> None:
        self.params["test_param"] = max(1.0, self.params["test_param"] - 0.1)

    def validate(self, new_params: dict) -> bool:
        return new_params["test_param"] <= 2.0


def test_basic_functionality():
    """Test basic orchestrator functionality"""
    print("Testing Real-Time Learning Orchestrator...")
    
    # Create orchestrator
    orchestrator = RealTimeLearningOrchestrator()
    
    # Register components
    test_component = TestLearningComponent("test1")
    vol_component = AdaptiveVolatilityCluster()
    
    orchestrator.register_component(test_component)
    orchestrator.register_component(vol_component)
    
    print(f"Registered components: {list(orchestrator.components.keys())}")
    
    # Test learning cycle
    test_data = {
        "realized_volatility": 0.025,
        "price_jumps": [0.005, -0.003],
        "volume": 1.2,
        "timestamp": "2023-01-01T12:00:00Z"
    }
    
    print("\nRunning learning cycle...")
    results = orchestrator.on_market_data(test_data)
    print(f"Update results: {results}")
    
    # Test component status
    print("\nComponent status:")
    for name, component in orchestrator.components.items():
        print(f"- {name}: {component.get_params()}")
    
    # Test rollback
    print("\nTesting rollback...")
    orchestrator.rollback_component("test1")
    print(f"Test component after rollback: {orchestrator.components['test1'].get_params()}")
    
    # Test audit trail
    print("\nAudit trail:")
    for entry in orchestrator.audit.get_recent_changes(hours=24):
        print(f"- {entry['component']}: {entry['changes']}")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_basic_functionality()