"""
LLM-as-judge evaluation for TransfersAI.

For each pair in the held-out test set:
  1. Run predict_transfer() to get the model's top match and confidence.
  2. Ask Claude to independently judge whether the correct pair is equivalent.
  3. Also ask Claude to judge the model's top-1 pick (when it differs from ground truth).
  4. Report: LLM accuracy, model accuracy, agreement rate, failure taxonomy.

Usage:
    python eval/run_llm_eval.py [--all] [--limit N]

    --all    Run on all 67 test pairs (default: hard cases only — where model
             doesn't rank correct match #1)
    --limit  Cap number of pairs evaluated (useful for smoke tests)

Results saved to eval/results/llm_eval_{timestamp}.json
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import predict_transfer, load_artifacts
from eval.llm_judge import judge_pair


RESULTS_DIR = Path(__file__).parent / "results"
TEST_POS_PATH = Path(__file__).parent.parent / "data" / "_test_pos.csv"


def _get_wm_info(artifacts, code: str) -> dict:
    """Pull title and description for a W&M course code from loaded artifacts."""
    inst = artifacts["institutions"].get("wm", {})
    lookup = inst.get("lookup", {})
    info = lookup.get(code, {})
    return {
        "title": info.get("title", ""),
        "description": info.get("description", ""),
    }


def _parse_vccs_dept_and_number(vccs_course_str: str) -> tuple[str, str]:
    """Extract dept and course number from raw VCCS course string."""
    import re
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(vccs_course_str).strip())
    if m:
        return m.group(1), m.group(2)
    return "", ""


def run_eval(run_all: bool = False, limit: int | None = None, delay: float = 0.3):
    print("Loading test set...")
    test_df = pd.read_csv(TEST_POS_PATH)
    print(f"  {len(test_df)} held-out positive pairs")

    print("\nLoading model artifacts (may download BGE model on first run)...")
    artifacts = load_artifacts()
    print("  Model ready.")

    if limit:
        test_df = test_df.head(limit)
        print(f"  Limiting to {limit} pairs (--limit flag)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    n_model_correct = 0
    n_llm_correct = 0
    n_agreement = 0
    n_hard = 0
    errors = 0

    print(f"\nRunning {'all' if run_all else 'hard-case'} eval on {len(test_df)} pairs...\n")

    for i, (_, row) in enumerate(test_df.iterrows()):
        vccs_str = str(row["VCCS Course"])
        vccs_title_raw = vccs_str  # full string; predict_transfer parses internally
        vccs_desc = str(row.get("VCCS Description", "") or "")
        correct_code = str(row["W&M Course Code"]).strip()

        dept, number = _parse_vccs_dept_and_number(vccs_str)

        # ── Model prediction ──────────────────────────────────────────
        try:
            preds = predict_transfer(
                vccs_dept=dept,
                vccs_number=number,
                vccs_title=vccs_title_raw,
                vccs_desc=vccs_desc,
                institutions=["wm"],
                top_k=5,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(test_df)}] PREDICT ERROR: {e}")
            errors += 1
            continue

        wm_preds = preds.get("wm", [])
        model_top1 = wm_preds[0] if wm_preds else None
        model_top1_code = model_top1["code"] if model_top1 else None
        model_top1_conf = model_top1["confidence"] if model_top1 else 0.0

        # Rank of correct answer in model predictions
        model_rank = next(
            (j + 1 for j, p in enumerate(wm_preds) if p["code"] == correct_code),
            None  # not in top-5
        )
        model_correct = model_rank == 1

        if model_correct:
            n_model_correct += 1

        # ── Hard case filter ──────────────────────────────────────────
        is_hard = not model_correct
        if is_hard:
            n_hard += 1

        if not run_all and not is_hard:
            continue

        # ── LLM judge: correct pair ───────────────────────────────────
        correct_wm = _get_wm_info(artifacts, correct_code)
        correct_wm_title = str(row.get("W&M Course Title", "") or correct_wm["title"])
        correct_wm_desc = str(row.get("W&M Description", "") or correct_wm["description"])

        try:
            judge_result = judge_pair(
                source_title=vccs_title_raw,
                source_dept=dept,
                source_desc=vccs_desc,
                target_code=correct_code,
                target_title=correct_wm_title,
                target_desc=correct_wm_desc,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(test_df)}] JUDGE ERROR: {e}")
            errors += 1
            continue

        llm_correct = judge_result["judgment"] == "equivalent"
        if llm_correct:
            n_llm_correct += 1

        # ── LLM judge: model's top-1 (only when different from correct) ──
        model_top1_judge = None
        if model_top1_code and model_top1_code != correct_code:
            top1_wm = _get_wm_info(artifacts, model_top1_code)
            try:
                time.sleep(delay)
                model_top1_judge = judge_pair(
                    source_title=vccs_title_raw,
                    source_dept=dept,
                    source_desc=vccs_desc,
                    target_code=model_top1_code,
                    target_title=top1_wm["title"],
                    target_desc=top1_wm["description"],
                )
            except Exception as e:
                print(f"  [{i+1}/{len(test_df)}] TOP1 JUDGE ERROR: {e}")

        # Agreement: both agree on the correct pair judgment
        if llm_correct == model_correct:
            n_agreement += 1

        # ── Failure mode tagging ──────────────────────────────────────
        failure_mode = None
        if is_hard and llm_correct:
            failure_mode = "model_wrong_llm_right"
        elif is_hard and not llm_correct:
            failure_mode = "both_wrong"
        elif not is_hard and not llm_correct:
            failure_mode = "model_right_llm_wrong"

        record = {
            "idx": i,
            "vccs_course": vccs_str,
            "vccs_dept": dept,
            "vccs_desc_preview": vccs_desc[:120],
            "correct_code": correct_code,
            "correct_title": correct_wm_title,
            "model_top1_code": model_top1_code,
            "model_top1_conf": round(model_top1_conf, 4),
            "model_rank_of_correct": model_rank,
            "model_correct": model_correct,
            "llm_judgment": judge_result["judgment"],
            "llm_confidence": judge_result["confidence"],
            "llm_reasoning": judge_result["reasoning"],
            "llm_correct": llm_correct,
            "llm_error": judge_result.get("error"),
            "model_top1_llm_judgment": model_top1_judge["judgment"] if model_top1_judge else None,
            "model_top1_llm_reasoning": model_top1_judge["reasoning"] if model_top1_judge else None,
            "failure_mode": failure_mode,
            "is_hard": is_hard,
        }
        results.append(record)

        # Progress
        status = []
        if model_correct: status.append("model✓")
        if llm_correct:   status.append("llm✓")
        if failure_mode:  status.append(f"[{failure_mode}]")
        print(f"  [{i+1:>2}/{len(test_df)}] {vccs_str[:45]:<45} → {correct_code:<10} "
              f"{' '.join(status) or 'both_wrong'}")

        time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────
    n_evaluated = len(results)
    if n_evaluated == 0:
        print("\nNo pairs evaluated.")
        return

    n_llm_on_correct = sum(1 for r in results if r["llm_correct"])
    n_both_correct   = sum(1 for r in results if r["model_correct"] and r["llm_correct"])
    n_model_only     = sum(1 for r in results if r["model_correct"] and not r["llm_correct"])
    n_llm_only       = sum(1 for r in results if not r["model_correct"] and r["llm_correct"])
    n_both_wrong     = sum(1 for r in results if not r["model_correct"] and not r["llm_correct"])

    # Failure mode counts
    fm_counts = {}
    for r in results:
        fm = r["failure_mode"] or "model_and_llm_agree_correct"
        fm_counts[fm] = fm_counts.get(fm, 0) + 1

    # LLM reasoning patterns for failure cases
    failure_cases = [r for r in results if r["failure_mode"] in ("model_wrong_llm_right", "both_wrong")]
    model_wrong_llm_right = [r for r in results if r["failure_mode"] == "model_wrong_llm_right"]
    both_wrong = [r for r in results if r["failure_mode"] == "both_wrong"]

    print(f"\n{'='*65}")
    print(f"LLM-AS-JUDGE EVALUATION RESULTS")
    print(f"{'='*65}")
    print(f"  Pairs evaluated:          {n_evaluated}")
    print(f"  Hard cases (model≠rank1): {n_hard} / {len(test_df)} total")
    print()
    print(f"  Model top-1 accuracy:     {n_model_correct}/{len(test_df)} "
          f"({n_model_correct/len(test_df):.1%}) [full test set]")
    print(f"  LLM accuracy (judged):    {n_llm_on_correct}/{n_evaluated} "
          f"({n_llm_on_correct/n_evaluated:.1%}) [on evaluated pairs]")
    print()
    print(f"  Confusion (evaluated pairs):")
    print(f"    Both correct:           {n_both_correct}")
    print(f"    Model only correct:     {n_model_only}")
    print(f"    LLM only correct:       {n_llm_only}  ← model blind spots")
    print(f"    Both wrong:             {n_both_wrong}")
    print()
    print(f"  Failure modes:")
    for fm, count in sorted(fm_counts.items(), key=lambda x: -x[1]):
        print(f"    {fm:<40} {count}")

    if model_wrong_llm_right:
        print(f"\n  Model wrong, LLM right ({len(model_wrong_llm_right)} cases) — model blind spots:")
        for r in model_wrong_llm_right[:5]:
            print(f"    {r['vccs_course'][:40]:<40} → {r['correct_code']}")
            print(f"      Model top1: {r['model_top1_code']} (conf={r['model_top1_conf']})")
            print(f"      LLM: {r['llm_reasoning'][:100]}")

    if both_wrong:
        print(f"\n  Both wrong ({len(both_wrong)} cases) — hard for everyone:")
        for r in both_wrong[:3]:
            print(f"    {r['vccs_course'][:40]:<40} → {r['correct_code']}")
            print(f"      LLM: {r['llm_reasoning'][:100]}")

    # ── Save ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"llm_eval_{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "n_test_total": len(test_df),
        "n_evaluated": n_evaluated,
        "n_hard_cases": n_hard,
        "run_all": run_all,
        "model_top1_accuracy": round(n_model_correct / len(test_df), 4),
        "llm_accuracy_on_evaluated": round(n_llm_on_correct / n_evaluated, 4),
        "confusion": {
            "both_correct": n_both_correct,
            "model_only": n_model_only,
            "llm_only": n_llm_only,
            "both_wrong": n_both_wrong,
        },
        "failure_mode_counts": fm_counts,
        "errors": errors,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all test pairs, not just hard cases")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N pairs (for smoke testing)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between API calls (default 0.3)")
    args = parser.parse_args()
    run_eval(run_all=args.all, limit=args.limit, delay=args.delay)
