"""
Quantum Dashboard Visualization Module

Provides visualization capabilities for the quantum-enhanced adaptive trading dashboard.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class QuantumVisualizer:
    """Handles visualization for quantum dashboard metrics"""
    
    def __init__(self):
        self.color_map = {
            'strategy_allocator': '#1f77b4',
            'correlation_matrix': '#ff7f0e',
            'order_router': '#2ca02c',
            'regime_parameters': '#d62728',
            'default': '#9467bd'
        }
    
    def plot_quantum_improvement(self, dashboard_state: Dict[str, Any]) -> Figure:
        """Plot quantum improvement by component"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        components = []
        improvements = []
        colors = []
        
        for component in dashboard_state['components']:
            components.append(component['component'])
            improvements.append(component['avg_improvement'] * 100)  # Convert to percentage
            colors.append(self.color_map.get(component['component'], self.color_map['default']))
        
        bars = ax.bar(components, improvements, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_title('Quantum Improvement by Component (%)')
        ax.set_ylabel('Improvement (%)')
        ax.set_ylim(0, max(improvements) * 1.2)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_metric_comparison(self, component_data: Dict[str, Any], metric_name: str) -> Figure:
        """Plot comparison between classical and quantum metrics"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        classical_value = component_data['classical_metrics'].get(metric_name, 0)
        quantum_value = component_data['quantum_metrics'].get(f'quantum_{metric_name}', 0)
        
        metrics = ['Classical', 'Quantum']
        values = [classical_value, quantum_value]
        
        bars = ax.bar(metrics, values, color=['#1f77b4', '#2ca02c'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        improvement = ((quantum_value - classical_value) / classical_value * 100) \
            if classical_value != 0 else 0
        
        ax.set_title(f'{metric_name} Comparison (Quantum Improvement: {improvement:.1f}%)')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_quantum_execution_stats(self, execution_stats: Dict[str, int]) -> Figure:
        """Plot quantum execution mode statistics"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        modes = list(execution_stats.keys())
        counts = list(execution_stats.values())
        
        bars = ax.bar(modes, counts, color='#9467bd')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        ax.set_title('Quantum Execution Mode Usage')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_quantum_advantage_trend(self, historical_data: List[Dict[str, Any]]) -> Figure:
        """Plot quantum advantage trend over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        timestamps = [data['timestamp'] for data in historical_data]
        advantages = [data['global_quantum_advantage'] * 100 for data in historical_data]  # Convert to percentage
        
        ax.plot(timestamps, advantages, marker='o', color='#2ca02c', linewidth=2)
        
        # Add value labels
        for i, advantage in enumerate(advantages):
            ax.text(timestamps[i], advantage, f'{advantage:.1f}%', ha='center', va='bottom')
        
        ax.set_title('Quantum Advantage Trend Over Time (%)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Quantum Advantage (%)')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        return fig
    
    def generate_component_report(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a detailed report for a component"""
        report = {
            'component': component_data['component_name'],
            'last_updated': component_data['last_updated'],
            'metrics': [],
            'quantum_execution': component_data['quantum_execution_metadata']
        }
        
        for metric, value in component_data['classical_metrics'].items():
            quantum_metric = f'quantum_{metric}'
            quantum_value = component_data['quantum_metrics'].get(quantum_metric, 0)
            improvement = component_data['quantum_improvement'].get(f'{metric}_improvement', 0)
            
            report['metrics'].append({
                'metric': metric,
                'classical': value,
                'quantum': quantum_value,
                'improvement_percent': improvement * 100
            })
            
        return report