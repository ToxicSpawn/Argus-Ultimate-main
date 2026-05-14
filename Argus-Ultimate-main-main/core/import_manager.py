"""
Argus Ultimate - Import Manager
Centralized management of optional dependencies with proper error handling and logging.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ImportStatus:
    """Track import status for optional components"""
    available: bool
    module: Optional[Any] = None
    error: Optional[str] = None
    warning_shown: bool = False

class ImportManager:
    """Centralized import management for optional dependencies"""
    
    def __init__(self):
        self._imports: Dict[str, ImportStatus] = {}
        self._initialized = False
    
    def import_optional(self, module_path: str, component_name: str, critical: bool = False) -> ImportStatus:
        """
        Safely import an optional module with proper error handling.
        
        Args:
            module_path: Full module path (e.g., 'execution.omega_execution')
            component_name: Human-readable component name
            critical: Whether this component is critical for operation
            
        Returns:
            ImportStatus with availability and error information
        """
        if component_name in self._imports:
            return self._imports[component_name]
        
        status = ImportStatus(available=False)
        
        try:
            module = __import__(module_path, fromlist=['*'])
            status.available = True
            status.module = module
            
            # Extract specific classes/functions if needed
            if hasattr(module, '__all__'):
                for name in module.__all__:
                    if hasattr(module, name):
                        setattr(status, name.lower(), getattr(module, name))
            
            logger.info(f"✅ {component_name} loaded successfully")
            
        except ImportError as e:
            status.error = str(e)
            if critical:
                logger.error(f"❌ Critical component {component_name} failed to import: {e}")
            else:
                logger.warning(f"⚠️  Optional component {component_name} not available: {e}")
        except Exception as e:
            status.error = str(e)
            logger.error(f"❌ Unexpected error importing {component_name}: {e}")
        
        self._imports[component_name] = status
        return status
    
    def get_status(self, component_name: str) -> Optional[ImportStatus]:
        """Get import status for a component"""
        return self._imports.get(component_name)
    
    def is_available(self, component_name: str) -> bool:
        """Check if a component is available"""
        status = self.get_status(component_name)
        return status.available if status else False
    
    def get_module(self, component_name: str) -> Optional[Any]:
        """Get the imported module for a component"""
        status = self.get_status(component_name)
        return status.module if status and status.available else None
    
    def initialize_omega_engines(self) -> Dict[str, bool]:
        """Initialize all Omega engines and return availability status"""
        omega_components = {
            'omega_execution': ('execution.omega_execution', 'Omega Execution Engine', True),
            'omega_risk': ('risk.omega_risk', 'Omega Risk Engine', True),
            'omega_strategies': ('strategies.omega_strategies', 'Omega Strategy Engine', True),
            'omega_adaptation': ('adaptive.omega_adaptation', 'Omega Adaptation Engine', False),
            'enhanced_adaptation': ('adaptive.enhanced_adaptation', 'Enhanced Adaptation Engine', False),
            'omega_core': ('core.omega_core', 'Omega Core Engine', True),
            'omega_portfolio': ('portfolio.omega_portfolio', 'Omega Portfolio Engine', True),
            'omega_compliance': ('compliance.omega_compliance', 'Omega Compliance Engine', False),
            'omega_ml': ('ml.omega_ml', 'Omega ML Engine', False),
            'omega_monitoring': ('monitoring.omega_monitoring', 'Omega Monitoring Engine', False),
        }
        
        results = {}
        for key, (module_path, name, critical) in omega_components.items():
            status = self.import_optional(module_path, name, critical)
            results[key] = status.available
        
        self._initialized = True
        return results
    
    def initialize_gpu_engines(self) -> Dict[str, bool]:
        """Initialize GPU-accelerated engines"""
        gpu_components = {
            'gpu_ml': ('ml.gpu_ml_engine', 'GPU ML Engine', False),
            'hft': ('execution.hft_microstructure_engine', 'HFT Microstructure Engine', False),
            'multi_asset': ('portfolio.multi_asset_engine', 'Multi-Asset Engine', False),
            'deep_learning': ('ml.deep_learning_engine', 'Deep Learning Engine', False),
            'gpu_quantum': ('quantum.gpu_quantum_engine', 'GPU Quantum Engine', False),
            'realtime_data': ('core.real_time_data_engine', 'Real-Time Data Engine', False),
        }
        
        results = {}
        for key, (module_path, name, critical) in gpu_components.items():
            status = self.import_optional(module_path, name, critical)
            results[key] = status.available
        
        return results
    
    def initialize_quantum_components(self) -> Dict[str, bool]:
        """Initialize quantum computing components"""
        quantum_components = {
            'quantum_adaptive_risk': ('risk.quantum_adaptive_risk', 'Quantum Adaptive Risk Engine', False),
            'canonical_quantum': ('quantum', 'Canonical Quantum Facade', False),
        }
        
        results = {}
        for key, (module_path, name, critical) in quantum_components.items():
            status = self.import_optional(module_path, name, critical)
            results[key] = status.available
        
        return results
    
    def get_import_summary(self) -> Dict[str, Any]:
        """Get a summary of all import statuses"""
        if not self._initialized:
            self.initialize_omega_engines()
        
        summary = {
            'total_components': len(self._imports),
            'available': sum(1 for status in self._imports.values() if status.available),
            'unavailable': sum(1 for status in self._imports.values() if not status.available),
            'components': {}
        }
        
        for name, status in self._imports.items():
            summary['components'][name] = {
                'available': status.available,
                'error': status.error
            }
        
        return summary

# Global import manager instance
import_manager = ImportManager()
