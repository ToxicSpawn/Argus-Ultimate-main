#!/usr/bin/env python3
"""
Argus Cleanup Script
Safely removes unnecessary files from Argus codebase

Usage:
    python cleanup_argus.py --safe      # Safe cleanup only
    python cleanup_argus.py --aggressive # Safe + duplicates
    python cleanup_argus.py --dry-run   # Show what would be deleted
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


class ArgusCleaner:
    """Safe cleanup utility for Argus codebase"""
    
    def __init__(self):
        self.root = Path(".")
        self.deleted_files = []
        self.deleted_dirs = []
        self.errors = []
    
    def safe_cleanup(self) -> Tuple[List[str], List[str]]:
        """Safe cleanup - dead files only"""
        print("🔴 SAFE CLEANUP: Removing dead files only...")
        
        # Tombstoned Python files
        tombstoned_files = [
            "argus_bot.py",
            "argus_max_adaptation.py", 
            "argus_omega_v2.py",
            "argus_ultimate.py",
            "argus_quantum_adaptive.py",
            "argus_ultimate_integration.py",
            "paper_adaptive.py",
            "paper_all_skills.py",
            "paper_enhanced_v2.py",
            "paper_final.py",
            "paper_kraken.py",
            "paper_real_data.py",
            "paper_ultimate_v3.py",
            "paper_unified_learning.py",
            "start_paper.py",
            "run_demo.py",
            "run_evolution.py",
            "run_maximum_evolution.py",
            "run_optimize.py",
            "run_paper.py",
            "run_paper_argus.py",
            "run_paper_test.py",
            "run_pinnacle.py",
            "run_quantum_evolution.py",
            "run_ultimate_evolution.py",
            "run_validation_backtest.py",
            "launch_quantum_maximum.py",
            "launch_quantum_production.py",
            "launch_ultimate.py",
            "train_pinnacle.py",
            "main_adaptive.py",
            "native_language_runner.py",
            "pinnacle_engine.py",
            "quantum_simulator_torch.py",
            "quantum_unified_stubs.py",
            "quantum_walk.py",
            "main_legacy.py"
        ]
        
        # Dead config
        config_files = ["config.yaml.deprecated"]
        
        # Cache/temp files
        cache_files = [
            "__pycache__",
            "quantum_execution_stats.png",
            "quantum_improvement.png", 
            "sharpe_comparison.png",
            "output.txt",
            "test_output.txt"
        ]
        
        # Empty directories
        empty_dirs = ["logs"]
        
        all_files = tombstoned_files + config_files + cache_files
        
        for file in all_files:
            self._safe_delete(file)
        
        for dir_name in empty_dirs:
            self._safe_delete_dir(dir_name)
        
        return self.deleted_files, self.deleted_dirs
    
    def aggressive_cleanup(self) -> Tuple[List[str], List[str]]:
        """Aggressive cleanup - safe + duplicates"""
        print("🔥 AGGRESSIVE CLEANUP: Removing duplicates too...")
        
        # First do safe cleanup
        files, dirs = self.safe_cleanup()
        
        # Duplicate documentation
        duplicate_docs = [
            "ADAPTATION_BEYOND_OMEGA.md",
            "ADAPTATION_CONNECTION_AUDIT.md", 
            "ADAPTATION_MECHANISM_EXPLAINED.md",
            "ALL_ADAPTATION_SYSTEMS.md",
            "ALL_IMPROVEMENTS_COMPLETE.md",
            "ALL_QUANTUM_IMPROVEMENTS_COMPLETE.md",
            "ARCHITECTURE.md",
            "ARCHITECTURE_OPTIMIZATION_ANALYSIS.md",
            "ARGUS_2026_IMPLEMENTATION_COMPLETE.md",
            "ARGUS_2026_IMPROVEMENTS.md",
            "ARGUS_BEYOND_OMEGA.md",
            "ARGUS_CURRENT_STATUS_HONEST.md",
            "ARGUS_FREE_ENHANCEMENTS_COMPLETE.md",
            "ARGUS_HIGHEST_STANDARD_IMPROVEMENTS.md",
            "ARGUS_IMPROVEMENTS_COMPLETE.md",
            "ARGUS_IMPROVEMENTS_ROADMAP.md",
            "ARGUS_REMAINING_OPPORTUNITIES.md",
            "ARGUS_SYDNEY_CONFIG.md",
            "ARGUS_WIRING_CHECKLIST.md",
            "AUSTRALIAN_EXCHANGES_FEES.md",
            "AUSTRALIAN_EXCHANGES_GUIDE.md",
            "COMPLETE_50_SYSTEMS_SUMMARY.md",
            "COMPLETE_ARGUS_OMEGA_62_SYSTEMS.md",
            "COMPLETE_CODEBASE_INVENTORY.md",
            "COMPLETE_IBM_CAPABILITY_MAP.md",
            "COMPLETE_IBM_ENHANCEMENT_MAP.md",
            "COMPLETE_WIRING_SUMMARY.md",
            "CONTINUOUS_EVOLUTION_GUIDE.md",
            "FULL_SYSTEM_WIRING_COMPLETE.md",
            "IMPROVEMENTS.md",
            "IMPROVEMENTS_IMPLEMENTATION_STATUS.md",
            "MAXIMUM_ADVANTAGE_GUIDE.md",
            "META_IMPROVEMENT_GUIDE.md",
            "PERFORMANCE_1K_FULLY_WIRED.md",
            "PERFORMANCE_PROJECTION_1K_REAL.md",
            "PHASE2_IMPLEMENTATION_STATUS.md",
            "PHASE3_IMPLEMENTATION_STATUS.md",
            "PREDICT_ADAPT_LEARN_INVENTORY.md",
            "QUANTUM_ADAPTATION_INTEGRATION_GUIDE.md",
            "QUANTUM_ENHANCED_ADAPTATION_GUIDE.md",
            "QUANTUM_IBM_OPTIMAL_WIRING.md",
            "QUANTUM_IMPROVEMENTS_GUIDE.md",
            "QUANTUM_SIMULATOR_EVOLUTION.md",
            "QUANTUM_TOP3_IMPROVEMENTS_COMPLETE.md",
            "REALITY_CHECK_AND_NEXT_STEPS.md",
            "REAL_MARKET_TEST_SUMMARY.md",
            "ULTRA_ADAPTATION_IMPROVEMENTS.md",
            "ULTRA_ADAPTATION_INTEGRATION_COMPLETE.md",
            "UNIFIED_RUNBOOK.md",
            "WIRING_COMPLETE.md",
            "WIRING_STATUS.md"
        ]
        
        # Duplicate start scripts
        duplicate_scripts = [
            "start_argus_1k.py",
            "start_argus_1k_simple.py", 
            "start_argus_optimal_wiring.py",
            "start_argus_sydney.py"
        ]
        
        # Unused directories
        unused_dirs = ["archive", "checkpoints"]
        
        for doc in duplicate_docs:
            self._safe_delete(doc)
        
        for script in duplicate_scripts:
            self._safe_delete(script)
        
        for dir_name in unused_dirs:
            self._safe_delete_dir(dir_name)
        
        return self.deleted_files, self.deleted_dirs
    
    def _safe_delete(self, file_path: str):
        """Safely delete a file"""
        path = self.root / file_path
        
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    self.deleted_dirs.append(file_path)
                    print(f"   📁 Deleted directory: {file_path}")
                else:
                    path.unlink()
                    self.deleted_files.append(file_path)
                    print(f"   📄 Deleted file: {file_path}")
            except Exception as e:
                self.errors.append(f"Failed to delete {file_path}: {e}")
                print(f"   ❌ Error deleting {file_path}: {e}")
    
    def _safe_delete_dir(self, dir_path: str):
        """Safely delete directory if empty"""
        path = self.root / dir_path
        
        if path.exists() and path.is_dir():
            try:
                # Check if empty
                if not any(path.iterdir()):
                    path.rmdir()
                    self.deleted_dirs.append(dir_path)
                    print(f"   📁 Deleted empty directory: {dir_path}")
                else:
                    print(f"   ⚠️  Directory not empty: {dir_path}")
            except Exception as e:
                self.errors.append(f"Failed to delete dir {dir_path}: {e}")
                print(f"   ❌ Error deleting directory {dir_path}: {e}")
    
    def dry_run(self, aggressive: bool = False) -> Tuple[List[str], List[str]]:
        """Show what would be deleted without actually deleting"""
        print("🔍 DRY RUN: Showing files that would be deleted...")
        
        # Temporarily disable actual deletion
        original_delete = self._safe_delete
        original_delete_dir = self._safe_delete_dir
        
        self._safe_delete = lambda x: print(f"   📄 Would delete: {x}")
        self._safe_delete_dir = lambda x: print(f"   📁 Would delete directory: {x}")
        
        if aggressive:
            self.aggressive_cleanup()
        else:
            self.safe_cleanup()
        
        # Restore original methods
        self._safe_delete = original_delete
        self._safe_delete_dir = original_delete_dir
        
        return [], []
    
    def verify_argus(self) -> bool:
        """Verify Argus still works after cleanup"""
        print("\n🧪 Verifying Argus still works...")
        
        try:
            # Check main entry points exist
            main_files = [
                "main.py",
                "argus_free_enhancements.py", 
                "argus_2026_enhanced.py",
                "start_argus.py"
            ]
            
            for file in main_files:
                if not (self.root / file).exists():
                    print(f"   ❌ Missing critical file: {file}")
                    return False
            
            # Check core directories exist
            core_dirs = [
                "core",
                "wiring", 
                "strategies",
                "data",
                "risk",
                "portfolio",
                "config",
                "quantum"
            ]
            
            for dir_name in core_dirs:
                if not (self.root / dir_name).exists():
                    print(f"   ❌ Missing critical directory: {dir_name}")
                    return False
            
            print("   ✅ All critical files and directories present")
            return True
            
        except Exception as e:
            print(f"   ❌ Verification error: {e}")
            return False
    
    def print_summary(self):
        """Print cleanup summary"""
        print("\n" + "=" * 60)
        print("📊 CLEANUP SUMMARY")
        print("=" * 60)
        
        print(f"\n✅ Files deleted: {len(self.deleted_files)}")
        print(f"✅ Directories deleted: {len(self.deleted_dirs)}")
        
        if self.errors:
            print(f"\n⚠️  Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   {error}")
        
        if self.deleted_files:
            print(f"\n📄 Deleted files:")
            for file in self.deleted_files[:10]:  # Show first 10
                print(f"   {file}")
            if len(self.deleted_files) > 10:
                print(f"   ... and {len(self.deleted_files) - 10} more")
        
        if self.deleted_dirs:
            print(f"\n📁 Deleted directories:")
            for dir_name in self.deleted_dirs:
                print(f"   {dir_name}")
        
        # Verify
        if self.verify_argus():
            print("\n✅ Argus is still functional after cleanup!")
        else:
            print("\n❌ WARNING: Argus may be broken - restore from git")
        
        print("\n💡 To restore: git checkout .")


def main():
    parser = argparse.ArgumentParser(description="Clean up Argus codebase")
    parser.add_argument("--safe", action="store_true", help="Safe cleanup only")
    parser.add_argument("--aggressive", action="store_true", help="Safe + duplicates")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    
    args = parser.parse_args()
    
    if not any([args.safe, args.aggressive, args.dry_run]):
        print("Usage: python cleanup_argus.py [--safe | --aggressive | --dry-run]")
        print("\nOptions:")
        print("  --safe       Remove dead files only (recommended)")
        print("  --aggressive Remove duplicates too")
        print("  --dry-run    Show what would be deleted")
        return
    
    cleaner = ArgusCleaner()
    
    print("🧹 Argus Cleanup Utility")
    print("=" * 60)
    
    if args.dry_run:
        cleaner.dry_run(aggressive=args.aggressive)
    elif args.aggressive:
        cleaner.aggressive_cleanup()
        cleaner.print_summary()
    elif args.safe:
        cleaner.safe_cleanup()
        cleaner.print_summary()


if __name__ == "__main__":
    main()
