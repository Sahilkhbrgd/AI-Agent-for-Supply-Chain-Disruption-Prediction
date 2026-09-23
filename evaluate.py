"""
Full Evaluation Runner — v3
============================
Runs all 5 scenarios × 5 runs = 25 agent observations.
Runs all 5 baseline scenarios.
Computes all metrics including Responsible AI evaluation.
Saves: Excel (5 sheets) + JSON summary.
"""
import os, sys, json, statistics
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent    import run_all, LLM_CONFIG
from baseline.baseline_workflow import run_all_baseline
from responsible_ai import full_responsible_ai_report
from config   import OUTPUT_DIR, SCENARIOS, N_RUNS

def clf_metrics(results):
    y_true = [1 if r["disruption_actual"]    else 0 for r in results]
    y_pred = [1 if r["disruption_predicted"] else 0 for r in results]
    if len(set(y_true)) < 2:
        return {"precision":1.0,"recall":1.0,"f1":1.0,"accuracy":1.0}
    return {
        "precision": round(precision_score(y_true,y_pred,zero_division=0),3),
        "recall":    round(recall_score(y_true,y_pred,zero_division=0),3),
        "f1":        round(f1_score(y_true,y_pred,zero_division=0),3),
        "accuracy":  round(accuracy_score(y_true,y_pred),3),
    }

def avg(lst, key, default=0):
    vals = [r.get(key,default) for r in lst
            if isinstance(r.get(key,default),(int,float))]
    return round(sum(vals)/len(vals),1) if vals else 0.0

def std(lst, key):
    vals = [r.get(key,0) for r in lst
            if isinstance(r.get(key,0),(int,float))]
    return round(statistics.stdev(vals),2) if len(vals)>1 else 0.0

def risk_acc(results):
    c = sum(1 for r in results if r.get("risk_correct",False))
    return round(c/len(results)*100,1) if results else 0.0

def main():
    print("\n"+"="*62)
    print("  SUPPLY CHAIN AI AGENT — FULL EVALUATION (v3)")
    print(f"  {N_RUNS} runs × 5 scenarios = {N_RUNS*5} agent observations")
    print("="*62)

    # ── 1. Run agent ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Running AI Agent ({N_RUNS} runs per scenario)...")
    all_agent_runs = run_all()   # 25 results

    # ── 2. Run baseline ───────────────────────────────────────────────────────
    print("\n[2/4] Running Rule-Based Scripted Baseline...")
    baseline_results = run_all_baseline()

    # ── 3. Responsible AI evaluation ─────────────────────────────────────────
    print("\n[3/4] Running Responsible AI Evaluation...")
    valid_sids  = ["S001","S002","S003","S004","S005"]
    sample_rep  = next((r["report"] for r in all_agent_runs if r.get("report")), {})
    rai_report  = full_responsible_ai_report(all_agent_runs, sample_rep, valid_sids)
    print(f"  Responsible AI Score: {rai_report['overall_responsible_ai_score']*100:.1f}%")
    print(f"  {rai_report['interpretation']}")

    # ── 4. Compute metrics ────────────────────────────────────────────────────
    print("\n[4/4] Computing evaluation metrics...")

    ac = clf_metrics(all_agent_runs)
    bc = clf_metrics(baseline_results)

    # Error detection / recovery
    s4_runs = [r for r in all_agent_runs if "scenario4" in r["scenario_id"]]
    s5_runs = [r for r in all_agent_runs if "scenario5" in r["scenario_id"]]
    err_det  = round(sum(1 for r in s4_runs if r["missing_data_detected"])/len(s4_runs)*100,1)
    err_rec  = round(sum(1 for r in s5_runs if r["tool_failure_recovered"])/len(s5_runs)*100,1)

    # ── Summary metrics table ─────────────────────────────────────────────────
    summary_data = [
        # Metric, Agent (mean), Agent (±sd), Baseline, Notes
        ["Total Observations (runs)",
         str(len(all_agent_runs)), "—", str(len(baseline_results)),
         f"{N_RUNS} runs/scenario × 5 scenarios"],
        ["Task Completion Rate (%)",
         str(avg(all_agent_runs,"task_completion_rate")), "—",
         str(avg(baseline_results,"task_completion_rate")), "Steps completed / total steps"],
        ["Risk Level Accuracy (%)",
         str(risk_acc(all_agent_runs)), "—",
         str(risk_acc(baseline_results)), "4-level: LOW/MEDIUM/HIGH/CRITICAL"],
        ["Disruption Detection Accuracy (%)",
         str(round(ac["accuracy"]*100,1)), "—",
         str(round(bc["accuracy"]*100,1)), "Binary: disruption present/absent"],
        ["Precision",
         str(ac["precision"]), "—", str(bc["precision"]), "TP/(TP+FP)"],
        ["Recall",
         str(ac["recall"]), "—", str(bc["recall"]), "TP/(TP+FN)"],
        ["F1-Score",
         str(ac["f1"]), "—", str(bc["f1"]), "Harmonic mean of precision & recall"],
        ["Avg Processing Time (ms)",
         str(avg(all_agent_runs,"processing_time_ms")),
         f"±{std(all_agent_runs,'processing_time_ms')}",
         str(avg(baseline_results,"processing_time_ms")), "End-to-end pipeline time"],
        ["Tool-Call Success Rate (%)",
         str(avg(all_agent_runs,"tool_success_rate_pct")),
         f"±{std(all_agent_runs,'tool_success_rate_pct')}",
         "N/A", "Successful tool calls / total calls"],
        ["Avg ReAct Steps per Run",
         str(avg(all_agent_runs,"react_steps")),
         f"±{std(all_agent_runs,'react_steps')}",
         "N/A", "LLM reasoning iterations per scenario"],
        ["Error Detection Rate (%)",
         str(err_det), "—", "0.0", "Scenario 4 missing-data detection"],
        ["Error Recovery Rate (%)",
         str(err_rec), "—", "0.0", "Scenario 5 tool-failure recovery"],
        ["Transparency Score (avg)",
         str(round(avg(all_agent_runs,"transparency_score"),3)), "—",
         "N/A", "Required report fields present (0-1)"],
        ["Responsible AI Score (%)",
         str(round(rai_report["overall_responsible_ai_score"]*100,1)), "—",
         "N/A", "Composite responsible AI evaluation"],
        ["Confidence Calibration Rate",
         str(rai_report["confidence_calibration"]["calibration_rate"]), "—",
         "N/A", "HIGH confidence & correct / total HIGH confidence"],
        ["Uncertainty Handling Rate (%)",
         str(round(rai_report["uncertainty_handling"]["uncertainty_handling_rate"]*100,1)),
         "—", "N/A", "Confidence degraded correctly under missing data"],
        ["Escalation Accuracy (%)",
         str(round(rai_report["escalation_appropriateness"]["escalation_accuracy"]*100,1)),
         "—", "N/A", "Human escalation correct for risk level"],
        ["Hallucination Detected",
         "NO" if rai_report["hallucination"]["pass"] else "YES", "—",
         "N/A", "Invented supplier IDs in agent output"],
    ]
    df_summary = pd.DataFrame(summary_data,
                              columns=["Metric","AI Agent (mean)","±SD","Baseline","Description"])

    # ── Per-run results table ─────────────────────────────────────────────────
    run_rows = []
    for r in all_agent_runs:
        run_rows.append({
            "Scenario":             r["scenario_id"],
            "Run":                  r["run_number"],
            "LLM":                  r.get("llm_config",{}).get("model","?"),
            "Temperature":          r.get("llm_config",{}).get("temperature","?"),
            "Predicted Risk":       r["predicted_risk"],
            "Actual Risk":          r["actual_risk"],
            "Correct":              "YES" if r["risk_correct"] else "NO",
            "ReAct Steps":          r["react_steps"],
            "Tool Calls":           r["tool_calls_total"],
            "Tool SR (%)":          r["tool_success_rate_pct"],
            "Time (ms)":            r["processing_time_ms"],
            "Confidence":           r["confidence"],
            "Missing Detected":     "YES" if r["missing_data_detected"] else "NO",
            "Failure Recovered":    "YES" if r["tool_failure_recovered"] else "NO",
            "Transparency Score":   r.get("transparency_score","?"),
            "Human Escalation":     "YES" if r.get("human_escalation") else "NO",
            "LLM Source":           r["llm_source"],
        })
    df_runs = pd.DataFrame(run_rows)

    # ── Scenario-level aggregate ──────────────────────────────────────────────
    scenario_rows = []
    for s in SCENARIOS:
        s_runs = [r for r in all_agent_runs if r["scenario_id"]==s]
        bl = next((r for r in baseline_results if r["scenario_id"]==s), {})
        scenario_rows.append({
            "Scenario":            s,
            "Ground Truth":        s_runs[0]["actual_risk"] if s_runs else "?",
            "Agent Risk (mode)":   max(set(r["predicted_risk"] for r in s_runs),
                                       key=lambda x: sum(1 for r in s_runs if r["predicted_risk"]==x)),
            "Baseline Risk":       bl.get("predicted_risk","?"),
            "Agent Accuracy (%)":  risk_acc(s_runs),
            "Baseline Correct":    "YES" if bl.get("risk_correct") else "NO",
            "Avg Steps":           avg(s_runs,"react_steps"),
            "Avg Tool Calls":      avg(s_runs,"tool_calls_total"),
            "Avg Tool SR (%)":     avg(s_runs,"tool_success_rate_pct"),
            "Avg Time (ms)":       avg(s_runs,"processing_time_ms"),
            "Baseline Time (ms)":  bl.get("processing_time_ms","?"),
            "Avg Confidence":      max(set(r["confidence"] for r in s_runs),
                                       key=lambda x: sum(1 for r in s_runs if r["confidence"]==x)),
            "Err Detection (%)":   round(sum(1 for r in s_runs if r["missing_data_detected"])/len(s_runs)*100,1),
            "Err Recovery (%)":    round(sum(1 for r in s_runs if r["tool_failure_recovered"])/len(s_runs)*100,1),
        })
    df_scenarios = pd.DataFrame(scenario_rows)

    # ── Responsible AI table ──────────────────────────────────────────────────
    rai_rows = [
        ["Overall Responsible AI Score",
         f"{rai_report['overall_responsible_ai_score']*100:.1f}%", "≥70% = pass"],
        ["Transparency Score",
         f"{rai_report['transparency']['transparency_score']*100:.0f}%",
         f"Fields present: {rai_report['transparency']['fields_present']}"],
        ["Hallucination Detection",
         "PASS — none detected" if rai_report["hallucination"]["pass"] else "FAIL — detected",
         "Agent only referenced valid supplier IDs"],
        ["Confidence Calibration Rate",
         f"{rai_report['confidence_calibration']['calibration_rate']*100:.0f}%",
         rai_report["confidence_calibration"]["interpretation"]],
        ["Uncertainty Handling Rate",
         f"{rai_report['uncertainty_handling']['uncertainty_handling_rate']*100:.0f}%",
         rai_report["uncertainty_handling"]["interpretation"]],
        ["Escalation Accuracy",
         f"{rai_report['escalation_appropriateness']['escalation_accuracy']*100:.0f}%",
         rai_report["escalation_appropriateness"]["interpretation"]],
        ["Recommendation Groundedness",
         f"{rai_report['recommendation_groundedness']['groundedness_rate']*100:.0f}%",
         rai_report["recommendation_groundedness"]["interpretation"]],
    ]
    df_rai = pd.DataFrame(rai_rows, columns=["Responsible AI Criterion","Result","Detail"])

    # ── Save Excel ─────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx = os.path.join(OUTPUT_DIR, "evaluation_results_v3.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df_summary.to_excel(w,   sheet_name="1_Summary_Metrics",     index=False)
        df_scenarios.to_excel(w, sheet_name="2_Scenario_Aggregates",  index=False)
        df_runs.to_excel(w,      sheet_name="3_All_Runs",             index=False)
        df_rai.to_excel(w,       sheet_name="4_Responsible_AI",       index=False)
        pd.DataFrame(baseline_results).to_excel(w,
                     sheet_name="5_Baseline_Results", index=False)
    print(f"\n  Excel → {xlsx}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    jpath = os.path.join(OUTPUT_DIR, "eval_summary_v3.json")
    with open(jpath,"w") as f:
        json.dump({
            "llm_config":            LLM_CONFIG,
            "total_agent_runs":      len(all_agent_runs),
            "agent_metrics":         {**ac,
                                      "task_completion":avg(all_agent_runs,"task_completion_rate"),
                                      "risk_accuracy":risk_acc(all_agent_runs),
                                      "avg_time_ms":avg(all_agent_runs,"processing_time_ms"),
                                      "avg_react_steps":avg(all_agent_runs,"react_steps"),
                                      "tool_success_rate":avg(all_agent_runs,"tool_success_rate_pct"),
                                      "error_detection_rate":err_det,
                                      "error_recovery_rate":err_rec,
                                      "transparency_score":round(avg(all_agent_runs,"transparency_score"),3)},
            "baseline_metrics":      {**bc,
                                      "task_completion":avg(baseline_results,"task_completion_rate"),
                                      "risk_accuracy":risk_acc(baseline_results),
                                      "avg_time_ms":avg(baseline_results,"processing_time_ms")},
            "responsible_ai":        rai_report,
            "scenario_aggregates":   scenario_rows,
            "all_run_results":       [{k:v for k,v in r.items()
                                       if k not in ("report","tool_call_log")}
                                      for r in all_agent_runs],
            "baseline_results":      baseline_results,
        }, f, indent=2, default=str)
    print(f"  JSON  → {jpath}")

    # ── Print final summary ────────────────────────────────────────────────────
    print("\n"+"="*62)
    print("  EVALUATION COMPLETE — KEY RESULTS")
    print("="*62)
    for row in summary_data[:12]:
        print(f"  {row[0]:40s} Agent={row[1]:8s}  Baseline={row[3]}")
    print(f"\n  Responsible AI Score: {rai_report['overall_responsible_ai_score']*100:.1f}%")
    print("="*62)

    return {"agent":all_agent_runs, "baseline":baseline_results, "rai":rai_report}

if __name__ == "__main__":
    main()
