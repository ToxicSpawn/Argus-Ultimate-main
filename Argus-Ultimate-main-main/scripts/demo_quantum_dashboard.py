"""
Quantum Dashboard Demo

This script demonstrates the quantum-enhanced adaptive trading dashboard.
"""

import sys
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.quantum_dashboard import QuantumDashboard, QuantumComponentMetrics
from monitoring.quantum_visualization import QuantumVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_quantum_dashboard():
    """Demonstrate quantum dashboard functionality"""
    print("Starting Quantum Dashboard Demo")
    logger.info("Starting Quantum Dashboard Demo")
    
    # Create dashboard and visualizer
    dashboard = QuantumDashboard()
    visualizer = QuantumVisualizer()
    
    # Simulate validation results for different components
    components = [
        {
            'name': 'strategy_allocator',
            'classical': {'sharpe_ratio': 2.0, 'win_rate': 0.6},
            'quantum': {'quantum_sharpe': 2.2, 'quantum_win_rate': 0.65},
            'improvement': {'sharpe_improvement': 0.1, 'win_rate_improvement': 0.083},
            'execution': {'execution_mode': 'qaoa_in_repo_simulator', 'n_qubits': 5}
        },
        {
            'name': 'correlation_matrix',
            'classical': {'diversification_score': 0.85},
            'quantum': {'quantum_diversification': 0.92},
            'improvement': {'diversification_improvement': 0.082},
            'execution': {'execution_mode': 'quantum_annealing', 'n_qubits': 8}
        },
        {
            'name': 'order_router',
            'classical': {'fill_quality': 0.88, 'latency_ms': 12},
            'quantum': {'quantum_fill_quality': 0.92, 'quantum_latency_ms': 10},
            'improvement': {'fill_quality_improvement': 0.045, 'latency_improvement': 0.167},
            'execution': {'execution_mode': 'quantum_search', 'n_qubits': 3}
        },
        {
            'name': 'regime_parameters',
            'classical': {'adaptation_speed': 0.75, 'stability': 0.88},
            'quantum': {'quantum_adaptation_speed': 0.82, 'quantum_stability': 0.93},
            'improvement': {'adaptation_speed_improvement': 0.093, 'stability_improvement': 0.057},
            'execution': {'execution_mode': 'quantum_kernel', 'n_qubits': 4}
        }
    ]
    
    # Update dashboard with component data
    for component in components:
        dashboard.state.update_component(
            component_name=component['name'],
            classical_metrics=component['classical'],
            quantum_metrics=component['quantum'],
            improvement=component['improvement'],
            execution_metadata=component['execution']
        )
        
        print(f"Added {component['name']} to dashboard with quantum improvement")
    
    # Generate and display visualizations
    print("\n=== Quantum Dashboard Visualizations ===")
    
    # 1. Quantum improvement by component
    fig1 = visualizer.plot_quantum_improvement(dashboard.get_global_summary())
    print("\n1. Quantum Improvement by Component:")
    fig1.savefig('quantum_improvement.png')
    print("   Saved to quantum_improvement.png")
    
    # 2. Metric comparison for strategy allocator
    component_data = dashboard.get_component_summary('strategy_allocator')
    fig2 = visualizer.plot_metric_comparison(component_data, 'sharpe_ratio')
    print("\n2. Sharpe Ratio Comparison (Strategy Allocator):")
    fig2.savefig('sharpe_comparison.png')
    print("   Saved to sharpe_comparison.png")
    
    # 3. Quantum execution statistics
    fig3 = visualizer.plot_quantum_execution_stats(dashboard.state.quantum_execution_stats)
    print("\n3. Quantum Execution Mode Usage:")
    fig3.savefig('quantum_execution_stats.png')
    print("   Saved to quantum_execution_stats.png")
    
    # 4. Generate reports
    print("\n=== Quantum Dashboard Reports ===")
    
    # Global summary
    global_summary = dashboard.get_global_summary()
    print(f"\nGlobal Summary:")
    print(f"  Components tracked: {global_summary['component_count']}")
    print(f"  Global quantum advantage: {global_summary['global_quantum_advantage']*100:.1f}%")
    print(f"  Last update: {global_summary['last_system_update']}")
    
    # Quantum advantage report
    advantage_report = dashboard.get_quantum_advantage_report()
    print(f"\nQuantum Advantage Report:")
    print(f"  Global quantum advantage: {advantage_report['global_quantum_advantage']*100:.1f}%")
    print(f"  Execution modes used: {len(advantage_report['execution_modes'])}")
    
    # Component details
    print(f"\nComponent Breakdown:")
    for component_name, component_data in advantage_report['component_breakdown'].items():
        print(f"  {component_name}:")
        print(f"    Avg improvement: {component_data['avg_improvement']*100:.1f}%")
        print(f"    Execution mode: {component_data['execution_metadata']['execution_mode']}")
    
    print("\nQuantum Dashboard Demo completed successfully!")
    logger.info("Quantum Dashboard Demo completed successfully!")

if __name__ == "__main__":
    demo_quantum_dashboard()