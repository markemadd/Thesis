"""
Pipeline runner — executes Phase 2 and Phase 3.

Usage:
    python -m pipeline.run_all                     # Run Phase 2 + 3
    python -m pipeline.run_all --phase 2           # Phase 2 only
    python -m pipeline.run_all --phase 3           # Phase 3 only
    python -m pipeline.run_all --phase 2 3         # Phase 2 then 3
"""

import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description="AD Proteomics Pipeline")
    parser.add_argument(
        '--phase', nargs='+', type=int, default=[2, 3],
        help='Which phase(s) to run (default: 2 3)'
    )
    args = parser.parse_args()

    start = time.time()

    print("╔══════════════════════════════════════════════╗")
    print("║   AD Proteomics Pipeline (Rewritten)        ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Phases to run: {args.phase}")

    for phase in args.phase:
        phase_start = time.time()

        if phase == 2:
            from pipeline.phase2_pathway_mapping import run_phase2
            run_phase2()
        elif phase == 3:
            from pipeline.phase3_pathway_scoring import run_phase3
            run_phase3()
        elif phase == 4:
            from pipeline.phase4_wgcna import run_phase4
            run_phase4()
        elif phase == 5:
            from pipeline.phase5_dim_reduction import run_phase5
            run_phase5()
        else:
            print(f"  ⚠ Unknown phase: {phase}")
            continue

        elapsed = time.time() - phase_start
        print(f"\n  Phase {phase} completed in {elapsed:.1f}s")

    total = time.time() - start
    print(f"\n{'─' * 50}")
    print(f"Pipeline finished in {total:.1f}s")


if __name__ == '__main__':
    main()
