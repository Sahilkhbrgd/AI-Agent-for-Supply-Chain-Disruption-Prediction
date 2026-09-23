"""
Rule-Based Scripted Baseline Workflow
======================================
This is a deterministic, rule-based Python script that replicates
the logical steps of a conventional supply chain risk review process.

IMPORTANT: This is NOT a human workflow. It is a scripted comparator
that applies fixed decision rules sequentially, without LLM reasoning,
autonomous tool calling, or data quality validation.

It serves as the comparison benchmark for the AI Agent evaluation.
"""
import pandas as pd, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, OUTPUT_DIR, GROUND_TRUTH, SCENARIOS

# Fixed thresholds for the rule-based baseline
BASELINE_DELAY_THRESHOLD = 7    # days
BASELINE_OTD_THRESHOLD   = 80.0 # %

def baseline_run(scenario_id: str) -> dict:
    """
    Execute the rule-based scripted baseline for one scenario.
    All steps are sequential and deterministic — no LLM, no tool calling.
    """
    t0  = time.time()
    d   = os.path.join(DATA_DIR, scenario_id)
    gt  = GROUND_TRUTH[scenario_id]

    # Step 1 — Load data (no validation layer)
    try:
        deliveries  = pd.read_csv(os.path.join(d, "deliveries.csv"))
        performance = pd.read_csv(os.path.join(d, "performance.csv"))
        disruptions = pd.read_csv(os.path.join(d, "disruption_events.csv"))
    except Exception as e:
        return {"scenario_id": scenario_id, "error": str(e),
                "task_completion_rate": 0.0}

    # Step 2 — Apply delay threshold rule
    delayed = deliveries[deliveries["delay_days"].fillna(0) >= BASELINE_DELAY_THRESHOLD]

    # Step 3 — Count external events (no content analysis)
    ext_events = len(disruptions)

    # Step 4 — Binary risk decision:
    #   any delay >= threshold OR any external event → HIGH
    #   otherwise → LOW
    #   LIMITATION: Cannot distinguish MEDIUM or CRITICAL — structural ceiling
    risk = "HIGH" if (len(delayed) > 0 or ext_events > 0) else "LOW"

    # Step 5 — Alternative supplier lookup (no tool call — static list scan)
    # Baseline reads the alt file but cannot call external DB or validate capacity

    # Step 6 — Compile report (no data quality check, no confidence scoring)
    total_ms = (time.time() - t0) * 1000
    # NOTE: No artificial human review time added — this represents the
    # scripted execution time only, which is the fair comparison point.

    actual_risk  = gt["risk"]
    risk_correct = (risk == actual_risk)
    disr_pred    = risk in ("HIGH", "CRITICAL")
    disr_actual  = gt["disruption"]

    return {
        "scenario_id":            scenario_id,
        "run_number":             1,
        "task_completion_rate":   100.0,
        "predicted_risk":         risk,
        "actual_risk":            actual_risk,
        "risk_correct":           risk_correct,
        "disruption_predicted":   disr_pred,
        "disruption_actual":      disr_actual,
        "processing_time_ms":     round(total_ms, 1),
        "tool_calls_total":       0,
        "tool_calls_successful":  0,
        "tool_success_rate_pct":  0.0,
        "missing_data_detected":  False,   # no validation layer
        "tool_failure_recovered": False,
        "data_quality":           "UNVALIDATED",
        "confidence":             "N/A",
        "human_escalation":       False,
        "react_steps":            0,
        "missing_data_present":   scenario_id == "scenario4_missing",
        "tool_failure_present":   scenario_id == "scenario5_toolfail",
    }

def run_all_baseline():
    results = []
    for s in SCENARIOS:
        r = baseline_run(s)
        results.append(r)
        print(f"  {r['scenario_id']:30s} | {r['predicted_risk']:8s} | "
              f"correct={r['risk_correct']} | time={r['processing_time_ms']:.1f}ms")
    return results

if __name__ == "__main__":
    results = run_all_baseline()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Baseline results saved.")
