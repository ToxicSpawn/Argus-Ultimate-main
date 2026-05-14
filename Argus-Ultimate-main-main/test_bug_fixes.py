#!/usr/bin/env python3
"""
Argus Ultimate - Bug Fixes Verification Test
Comprehensive test suite to verify all bug fixes are working correctly.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

class BugFixVerification:
    """Comprehensive bug fix verification"""
    
    def __init__(self):
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "details": {}
        }
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and track results"""
        try:
            logger.info(f"Running test: {test_name}")
            result = test_func()
            if result:
                self.test_results["passed"] += 1
                self.test_results["details"][test_name] = "PASSED"
                logger.info(f"✅ {test_name} - PASSED")
            else:
                self.test_results["failed"] += 1
                self.test_results["details"][test_name] = "FAILED"
                logger.error(f"❌ {test_name} - FAILED")
            return result
        except Exception as e:
            self.test_results["failed"] += 1
            error_msg = f"{test_name} - ERROR: {e}"
            self.test_results["errors"].append(error_msg)
            self.test_results["details"][test_name] = f"ERROR: {e}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            return False
    
    def test_configuration_fixes(self) -> bool:
        """Test configuration conflict fixes"""
        try:
            # Test that config.yaml is properly deprecated
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    content = f.read()
                    if "DEPRECATED" in content and "intentionally left empty" in content:
                        return True
                    else:
                        logger.error("config.yaml not properly deprecated")
                        return False
            return True
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            return False
    
    def test_import_manager(self) -> bool:
        """Test import manager functionality"""
        try:
            from core.import_manager import import_manager
            
            # Test initialization
            omega_status = import_manager.initialize_omega_engines()
            if not isinstance(omega_status, dict):
                return False
            
            # Test import summary
            summary = import_manager.get_import_summary()
            if not isinstance(summary, dict) or "total_components" not in summary:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Import manager test failed: {e}")
            return False
    
    def test_validation_system(self) -> bool:
        """Test data validation system"""
        try:
            from core.validation import validator
            
            # Test numeric validation
            result = validator.validate_numeric(100.0, "test_price", min_val=0, max_val=1000)
            if not result.is_valid or result.sanitized_data != 100.0:
                return False
            
            # Test string validation
            result = validator.validate_string("BTCUSDT", "test_symbol")
            if not result.is_valid or result.sanitized_data != "BTCUSDT":
                return False
            
            # Test price validation
            result = validator.validate_price(50000.0)
            if not result.is_valid or result.sanitized_data != 50000.0:
                return False
            
            # Test symbol validation
            result = validator.validate_symbol("btcusdt")
            if not result.is_valid or result.sanitized_data != "BTCUSDT":
                return False
            
            return True
        except Exception as e:
            logger.error(f"Validation system test failed: {e}")
            return False
    
    def test_async_utils(self) -> bool:
        """Test async utilities"""
        try:
            import asyncio
            from core.async_utils import gather_safe, create_task_safe
            
            async def test_async():
                # Test gather_safe
                async def dummy_task(x):
                    return x * 2
                
                results = await gather_safe(dummy_task(1), dummy_task(2))
                if results != [2, 4]:
                    return False
                
                # Test create_task_safe
                task = await create_task_safe(dummy_task(5), "test_task")
                result = await task
                if result != 5:
                    return False
                
                return True
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(test_async())
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Async utils test failed: {e}")
            return False
    
    def test_resource_manager(self) -> bool:
        """Test resource manager"""
        try:
            from core.resource_manager import resource_manager
            
            # Test metrics collection
            metrics = resource_manager.get_current_metrics()
            if not hasattr(metrics, 'memory_mb'):
                return False
            
            # Test object tracking
            test_obj = {"test": "data"}
            resource_manager.track_object(test_obj, "test_category")
            
            # Test garbage collection
            gc_stats = resource_manager.force_garbage_collection()
            if not isinstance(gc_stats, dict):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Resource manager test failed: {e}")
            return False
    
    def test_main_py_imports(self) -> bool:
        """Test main.py import structure"""
        try:
            # Import main module to test import structure
            import main
            
            # Check that import manager is used
            if not hasattr(main, 'import_manager'):
                return False
            
            # Check that availability flags are set
            required_flags = [
                'OMEGA_EXECUTION_AVAILABLE',
                'OMEGA_RISK_AVAILABLE', 
                'OMEGA_STRATEGIES_AVAILABLE',
                'GPU_ML_AVAILABLE',
                'QUANTUM_ADAPTIVE_RISK_AVAILABLE'
            ]
            
            for flag in required_flags:
                if not hasattr(main, flag):
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Main.py imports test failed: {e}")
            return False
    
    def test_dependency_versions(self) -> bool:
        """Test dependency version consistency"""
        try:
            # Check requirements files exist
            req_files = ["requirements.txt", "requirements-advanced.txt"]
            for req_file in req_files:
                if not Path(req_file).exists():
                    return False
            
            # Check torch version consistency
            with open("requirements.txt", 'r') as f:
                main_req = f.read()
            
            with open("requirements-advanced.txt", 'r') as f:
                adv_req = f.read()
            
            if "torch==2.3.0" in main_req and "torch==2.3.0" in adv_req:
                return True
            else:
                logger.error("Torch version inconsistency detected")
                return False
                
        except Exception as e:
            logger.error(f"Dependency version test failed: {e}")
            return False
    
    def test_error_handling_improvements(self) -> bool:
        """Test error handling improvements"""
        try:
            # Test that unified_trading_system.py has improved error handling
            with open("unified_trading_system.py", 'r') as f:
                content = f.read()
            
            # Check for proper error logging instead of silent exceptions
            if "logger.warning" in content and "logger.error" in content:
                return True
            else:
                logger.error("Error handling improvements not found")
                return False
                
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all bug fix verification tests"""
        logger.info("Starting comprehensive bug fix verification...")
        
        tests = [
            ("Configuration Fixes", self.test_configuration_fixes),
            ("Import Manager", self.test_import_manager),
            ("Validation System", self.test_validation_system),
            ("Async Utils", self.test_async_utils),
            ("Resource Manager", self.test_resource_manager),
            ("Main.py Imports", self.test_main_py_imports),
            ("Dependency Versions", self.test_dependency_versions),
            ("Error Handling", self.test_error_handling_improvements),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Generate summary
        total_tests = self.test_results["passed"] + self.test_results["failed"]
        success_rate = (self.test_results["passed"] / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "passed": self.test_results["passed"],
            "failed": self.test_results["failed"],
            "success_rate": success_rate,
            "errors": self.test_results["errors"],
            "details": self.test_results["details"]
        }
        
        logger.info(f"Bug fix verification completed: {self.test_results['passed']}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        return summary

def main():
    """Main test runner"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Run verification
    verifier = BugFixVerification()
    results = verifier.run_all_tests()
    
    # Print results
    print("\n" + "="*60)
    print("ARGUS ULTIMATE - BUG FIX VERIFICATION RESULTS")
    print("="*60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("\nDetailed Results:")
    for test_name, result in results['details'].items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {test_name}: {result}")
    
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if results['failed'] == 0 else 1)

if __name__ == "__main__":
    main()
