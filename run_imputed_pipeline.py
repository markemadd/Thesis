"""
Re-run Phases 3–5 + Differential Analysis on imputed data.

This script patches the pipeline to read from output_imputed/ instead of output/,
then runs Phase 3 (scoring), Phase 5 (reduction), and differential analysis.
"""

import os
import sys
import time

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CRITICAL: Patch config BEFORE any pipeline imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE = os.path.dirname(os.path.abspath(__file__))
IMPUTED_DIR = os.path.join(BASE, "output_imputed")

import pipeline.config as config
config.OUTPUT_DIR = IMPUTED_DIR
os.makedirs(IMPUTED_DIR, exist_ok=True)
os.makedirs(os.path.join(IMPUTED_DIR, "figures"), exist_ok=True)

print("╔══════════════════════════════════════════════╗")
print("║   Re-run Pipeline on IMPUTED Data            ║")
print("╚══════════════════════════════════════════════╝")
print(f"  Output directory: {IMPUTED_DIR}")


def main():
    start = time.time()

    # ── Phase 3: Pathway scoring ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Phase 3 (Imputed) — Pathway Activity Scoring")
    print("─" * 70)
    t0 = time.time()

    # Force reload so module-level paths pick up patched OUTPUT_DIR
    import importlib
    import pipeline.phase3_pathway_scoring as p3
    importlib.reload(p3)
    p3.run_phase3()
    print(f"\n  Phase 3 completed in {time.time() - t0:.1f}s")

    # ── Phase 5: Dimensionality reduction ─────────────────────────────────
    print("\n" + "─" * 70)
    print("Phase 5 (Imputed) — Pathway Dimensionality Reduction")
    print("─" * 70)
    t0 = time.time()

    import pipeline.phase5_dim_reduction as p5
    importlib.reload(p5)
    p5.run_phase5()
    print(f"\n  Phase 5 completed in {time.time() - t0:.1f}s")

    # ── Differential analysis ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Differential Analysis (Imputed)")
    print("─" * 70)
    t0 = time.time()

    import differential_analysis as da
    importlib.reload(da)
    da.main()
    print(f"\n  Differential analysis completed in {time.time() - t0:.1f}s")

    total = time.time() - start
    print(f"\n{'═' * 70}")
    print(f"Full imputed pipeline finished in {total:.1f}s")


if __name__ == "__main__":
    main()
