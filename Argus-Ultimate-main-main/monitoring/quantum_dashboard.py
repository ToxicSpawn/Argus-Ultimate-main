"""
Quantum-Enhanced Adaptive Trading Dashboard

This module provides real-time monitoring of quantum-enhanced adaptive trading components.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class QuantumComponentMetrics:
    """Metrics for a single quantum-enhanced component"""
    component_name: str
    classical_metrics: Dict[str, float] = field(default_factory=dict)
    quantum_metrics: Dict[str, float] = field(default_factory=dict)
    quantum_improvement: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    quantum_execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_classical(self, metrics: Dict[str, float]):
        """Update classical metrics"""
        self.classical_metrics.update(metrics)
        self.last_updated = datetime.now()
        
    def update_quantum(self, metrics: Dict[str, float], improvement: Dict[str, float], 
                      execution_metadata: Dict[str, Any]):
        """Update quantum metrics and improvement"""
        self.quantum_metrics.update(metrics)
        self.quantum_improvement.update(improvement)
        self.quantum_execution_metadata.update(execution_metadata)
        self.last_updated = datetime.now()

@dataclass
class QuantumDashboardState:
    """State of the quantum dashboard"""
    components: Dict[str, QuantumComponentMetrics] = field(default_factory=dict)
    global_quantum_advantage: float = 0.0
    last_system_update: datetime = field(default_factory=datetime.now)
    quantum_execution_stats: Dict[str, Any] = field(default_factory=dict)
    
    def update_component(self, component_name: str, classical_metrics: Dict[str, float], 
                        quantum_metrics: Dict[str, float], improvement: Dict[str, float],
                        execution_metadata: Dict[str, Any]):
        """Update metrics for a component"""
        if component_name not in self.components:
            self.components[component_name] = QuantumComponentMetrics(component_name)
            
        component = self.components[component_name]
        component.update_classical(classical_metrics)
        component.update_quantum(quantum_metrics, improvement, execution_metadata)
        
        # Update global quantum advantage
        avg_improvement = np.mean(list(improvement.values())) if improvement else 0
        self.global_quantum_advantage = max(self.global_quantum_advantage, avg_improvement)
        self.last_system_update = datetime.now()
        
        # Update execution stats
        if 'execution_mode' in execution_metadata:
            mode = execution_metadata['execution_mode']
            self.quantum_execution_stats[mode] = self.quantum_execution_stats.get(mode, 0) + 1

class QuantumDashboard:
    """Quantum-Enhanced Adaptive Trading Dashboard"""
    
    def __init__(self):
        self.state = QuantumDashboardState()
        
    def update_from_validation(self, validation_result: Dict[str, Any]):
        """Update dashboard from validation result"""
        try:
            component_name = validation_result['validation_result'].component
            classical_metrics = {k: v for k, v in validation_result['validation_result'].metrics.items()
                              if not k.startswith('quantum_')}
            
            quantum_metrics = {k: v for k, v in validation_result['validation_result'].metrics.items()
                             if k.startswith('quantum_')}
            
            # Extract quantum improvement metrics
            improvement = {}
            for k, v in validation_result['validation_result'].metrics.items():
                if k.startswith('quantum_') and k.endswith('_improvement'):
                    improvement[k] = v
            
            # Get quantum execution metadata
            execution_metadata = validation_result['validation_result'].quantum_metadata
            
            self.state.update_component(
                component_name=component_name,
                classical_metrics=classical_metrics,
                quantum_metrics=quantum_metrics,
                improvement=improvement,
                execution_metadata=execution_metadata
            )
            
            logger.info(f"Updated dashboard for {component_name} with quantum advantage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update dashboard: {str(e)}")
            return False
    
    def get_component_summary(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific component"""
        if component_name not in self.state.components:
            return None
            
        component = self.state.components[component_name]
        return {
            'component_name': component_name,
            'classical_metrics': component.classical_metrics,
            'quantum_metrics': component.quantum_metrics,
            'quantum_improvement': component.quantum_improvement,
            'last_updated': component.last_updated,
            'quantum_execution_metadata': component.quantum_execution_metadata
        }
    
    def get_global_summary(self) -> Dict[str, Any]:
        """Get global dashboard summary"""
        component_summaries = []
        for component_name, component in self.state.components.items():
            component_summaries.append({
                'component': component_name,
                'last_updated': component.last_updated,
                'avg_improvement': np.mean(list(component.quantum_improvement.values())) 
                                if component.quantum_improvement else 0
            })
            
        return {
            'components': component_summaries,
            'global_quantum_advantage': self.state.global_quantum_advantage,
            'last_system_update': self.state.last_system_update,
            'quantum_execution_stats': self.state.quantum_execution_stats,
            'component_count': len(self.state.components)
        }
    
    def get_quantum_advantage_report(self) -> Dict[str, Any]:
        """Generate quantum advantage report"""
        report = {
            'global_quantum_advantage': self.state.global_quantum_advantage,
            'component_breakdown': {},
            'execution_modes': self.state.quantum_execution_stats
        }
        
        for component_name, component in self.state.components.items():
            report['component_breakdown'][component_name] = {
                'metrics': component.quantum_improvement,
                'avg_improvement': np.mean(list(component.quantum_improvement.values())) 
                                if component.quantum_improvement else 0,
                'execution_metadata': component.quantum_execution_metadata
            }
            
        return report
    
    def reset(self):
        """Reset dashboard state"""
        self.state = QuantumDashboardState()
        logger.info("Quantum dashboard reset")