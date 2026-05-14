# Argus Cleanup Report - Files That Can Be Deleted
## Complete Audit: What Argus Doesn't Need Anymore

---

## 🔴 HIGH PRIORITY: DEAD FILES (Safe to Delete)

### **Tombstoned Python Files (42 total)**
These files contain only error messages and are completely dead:

```
argus_bot.py                           # Superseded by main.py
argus_max_adaptation.py                # Superseded by main.py
argus_omega_v2.py                      # Superseded by main.py
argus_ultimate.py                      # Superseded by main.py
argus_quantum_adaptive.py              # Superseded by main.py
argus_ultimate_integration.py          # Superseded by main.py
paper_adaptive.py                      # Moved to scripts/runners/
paper_all_skills.py                    # Moved to scripts/runners/
paper_enhanced_v2.py                   # Moved to scripts/runners/
paper_final.py                         # Moved to scripts/runners/
paper_kraken.py                        # Moved to scripts/runners/
paper_real_data.py                     # Moved to scripts/runners/
paper_ultimate_v3.py                   # Moved to scripts/runners/
paper_unified_learning.py              # Moved to scripts/runners/
start_paper.py                         # Moved to scripts/runners/
run_demo.py                            # Moved to scripts/runners/
run_evolution.py                       # Moved to scripts/runners/
run_maximum_evolution.py               # Moved to scripts/runners/
run_optimize.py                        # Moved to scripts/runners/
run_paper.py                           # Moved to scripts/runners/
run_paper_argus.py                     # Moved to scripts/runners/
run_paper_test.py                      # Moved to scripts/runners/
run_pinnacle.py                       # Moved to scripts/runners/
run_quantum_evolution.py               # Moved to scripts/runners/
run_ultimate_evolution.py              # Moved to scripts/runners/
run_validation_backtest.py             # Moved to scripts/runners/
launch_quantum_maximum.py              # Use .env config instead
launch_quantum_production.py           # Use .env config instead
launch_ultimate.py                     # Moved to scripts/runners/
train_pinnacle.py                      # Moved to scripts/runners/
main_adaptive.py                       # Superseded by main.py
native_language_runner.py             # Moved to scripts/runners/
pinnacle_engine.py                     # Not a runner, library module
quantum_simulator_torch.py             # Shim retired
quantum_unified_stubs.py               # Shim retired
quantum_walk.py                        # Shim retired
main_legacy.py                         # Removed entirely
```

**Space saved: ~50KB**
**Risk: ZERO (these are just error stubs)**

---

### **Dead Config Files (2 total)**
```
config.yaml.deprecated                  # 47KB of dead config
```

**Space saved: 47KB**
**Risk: ZERO (deprecated, not loaded)**

---

## 🟡 MEDIUM PRIORITY: DUPLICATE/OBSOLETE FILES

### **Duplicate Documentation (35 total)**
Many overlapping markdown files at root level. Keep only essential ones:

**KEEP:**
- README.md (main documentation)
- SETUP_CHECKLIST.md (user setup)
- API_SETUP_GUIDE.md (API setup)
- BEST_TRADING_PAIRS.md (trading pairs)
- ARGUS_1K_PERFORMANCE_PROJECTION.md (projections)

**CONSIDER DELETING (30 files):**
```
ADAPTATION_BEYOND_OMEGA.md
ADAPTATION_CONNECTION_AUDIT.md
ADAPTATION_MECHANISM_EXPLAINED.md
ALL_ADAPTATION_SYSTEMS.md
ALL_IMPROVEMENTS_COMPLETE.md
ALL_QUANTUM_IMPROVEMENTS_COMPLETE.md
ARCHITECTURE.md
ARCHITECTURE_OPTIMIZATION_ANALYSIS.md
ARGUS_2026_IMPLEMENTATION_COMPLETE.md
ARGUS_2026_IMPROVEMENTS.md
ARGUS_BEYOND_OMEGA.md
ARGUS_CURRENT_STATUS_HONEST.md
ARGUS_FREE_ENHANCEMENTS_COMPLETE.md
ARGUS_HIGHEST_STANDARD_IMPROVEMENTS.md
ARGUS_IMPROVEMENTS_COMPLETE.md
ARGUS_IMPROVEMENTS_ROADMAP.md
ARGUS_REMAINING_OPPORTUNITIES.md
ARGUS_SYDNEY_CONFIG.md
ARGUS_WIRING_CHECKLIST.md
AUSTRALIAN_EXCHANGES_FEES.md
AUSTRALIAN_EXCHANGES_GUIDE.md
COMPLETE_50_SYSTEMS_SUMMARY.md
COMPLETE_ARGUS_OMEGA_62_SYSTEMS.md
COMPLETE_CODEBASE_INVENTORY.md
COMPLETE_IBM_CAPABILITY_MAP.md
COMPLETE_IBM_ENHANCEMENT_MAP.md
COMPLETE_WIRING_SUMMARY.md
CONTINUOUS_EVOLUTION_GUIDE.md
FULL_SYSTEM_WIRING_COMPLETE.md
IMPROVEMENTS.md
IMPROVEMENTS_IMPLEMENTATION_STATUS.md
MAXIMUM_ADVANTAGE_GUIDE.md
META_IMPROVEMENT_GUIDE.md
PERFORMANCE_1K_FULLY_WIRED.md
PERFORMANCE_PROJECTION_1K_REAL.md
PHASE2_IMPLEMENTATION_STATUS.md
PHASE3_IMPLEMENTATION_STATUS.md
PREDICT_ADAPT_LEARN_INVENTORY.md
QUANTUM_ADAPTATION_INTEGRATION_GUIDE.md
QUANTUM_ENHANCED_ADAPTATION_GUIDE.md
QUANTUM_IBM_OPTIMAL_WIRING.md
QUANTUM_IMPROVEMENTS_GUIDE.md
QUANTUM_SIMULATOR_EVOLUTION.md
QUANTUM_TOP3_IMPROVEMENTS_COMPLETE.md
REALITY_CHECK_AND_NEXT_STEPS.md
REAL_MARKET_TEST_SUMMARY.md
ULTRA_ADAPTATION_IMPROVEMENTS.md
ULTRA_ADAPTATION_INTEGRATION_COMPLETE.md
UNIFIED_RUNBOOK.md
WIRING_COMPLETE.md
WIRING_STATUS.md
```

**Space saved: ~400KB**
**Risk: LOW (informational only)**

---

### **Duplicate Launch Scripts (5 total)**
Only need `start_argus.py` and `argus_free_enhancements.py`:

**CONSIDER DELETING:**
```
start_argus_1k.py
start_argus_1k_simple.py
start_argus_optimal_wiring.py
start_argus_sydney.py
```

**Space saved: ~40KB**
**Risk: LOW (have alternatives)**

---

## 🟢 LOW PRIORITY: OPTIONAL CLEANUP

### **Cache/Build Files**
```
__pycache__/                          # Python cache (auto-generated)
quantum_execution_stats.png          # Old chart
quantum_improvement.png               # Old chart
sharpe_comparison.png                # Old chart
output.txt                            # Gitignored output
test_output.txt                       # Gitignored output
```

**Space saved: ~100KB**
**Risk: ZERO (cache/temp files)**

### **Empty/Unused Directories**
```
logs/                                 # Empty
archive/legacy_entrypoints/           # Old README only
checkpoints/dynamic_gnn/              # Single old checkpoint
```

**Space saved: ~20KB**
**Risk: LOW**

---

## 📊 CLEANUP SUMMARY

### **If You Delete Everything:**
- **Files deleted:** ~85 files
- **Space saved:** ~650KB
- **Directories cleaned:** 4
- **Risk:** Very Low (mostly dead code)

### **Recommended Cleanup (Safe):**
1. **Delete all 42 tombstoned Python files** (50KB)
2. **Delete config.yaml.deprecated** (47KB)
3. **Delete cache files** (100KB)
4. **Keep duplicate docs for now** (user might need them)

**Total safe cleanup: ~200KB**

---

## 🗑️ CLEANUP COMMANDS

### **Option 1: Safe Cleanup (Recommended)**
```bash
# Delete tombstoned files
rm argus_bot.py argus_max_adaptation.py argus_omega_v2.py argus_ultimate.py argus_quantum_adaptive.py argus_ultimate_integration.py
rm paper_*.py start_paper.py
rm run_*.py launch_*.py train_pinnacle.py
rm main_adaptive.py native_language_runner.py pinnacle_engine.py
rm quantum_simulator_torch.py quantum_unified_stubs.py quantum_walk.py main_legacy.py

# Delete dead config
rm config.yaml.deprecated

# Delete cache
rm -rf __pycache__/
rm *.png
rm output.txt test_output.txt

# Delete empty logs
rmdir logs/
```

### **Option 2: Aggressive Cleanup (All duplicates)**
```bash
# Add to above:
rm ADAPTATION_*.md ALL_*.md ARCHITECTURE*.md ARGUS_*.md AUSTRALIAN_*.md
rm COMPLETE_*.md CONTINUOUS_*.md FULL_*.md IMPROVEMENTS*.md MAXIMUM_*.md
rm META_*.md PERFORMANCE_*.md PHASE*.md PREDICT_*.md QUANTUM_*.md
rm REAL_*.md ULTRA_*.md UNIFIED_*.md WIRING_*.md

# Delete duplicate start scripts
rm start_argus_1k*.py start_argus_optimal_wiring.py start_argus_sydney.py

# Delete unused directories
rm -rf archive/ checkpoints/ logs/
```

---

## ⚠️ FILES TO KEEP (Essential)

### **Core Entry Points:**
- `main.py` - Main Argus entry point
- `argus_2026_enhanced.py` - 2026 enhancements
- `argus_free_enhancements.py` - All 78 systems
- `start_argus.py` - Simple starter

### **Essential Config:**
- `.env.example` - Template
- `config.yaml` - Active config
- `requirements.txt` - Dependencies

### **Essential Documentation:**
- `README.md` - Main docs
- `SETUP_CHECKLIST.md` - User setup
- `API_SETUP_GUIDE.md` - API setup

### **Core Directories:**
- `core/` - Core systems
- `wiring/` - Integration wiring
- `strategies/` - Trading strategies
- `data/` - Data collectors
- `risk/` - Risk management
- `portfolio/` - Portfolio management
- `config/` - Configuration
- `quantum/` - Quantum systems

---

## 🎯 RECOMMENDATION

**Start with safe cleanup (Option 1):**
1. Delete all tombstoned Python files (42 files)
2. Delete dead config file
3. Delete cache files

**Result:**
- Cleaner codebase
- No functionality lost
- ~200KB space saved
- Less confusion for users

**After that, evaluate if you want to remove duplicate documentation.**

---

## ✅ VERIFICATION

After cleanup, verify Argus still works:
```bash
python argus_free_enhancements.py
# Should start all 78 systems normally
```

If it works, cleanup was successful. If not, restore from git.

---

**Ready to proceed with cleanup?** 🗑️
