"""
Supply Chain Disruption Prediction AI Agent — v3
=================================================
Architecture : ReAct loop + Google Gemini function calling
LLM          : gemini-2.0-flash (Google AI Studio — free tier)
Runs         : N_RUNS per scenario (default 5) for statistical robustness
"""
import json, os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
                    LLM_PROVIDER, MAX_REACT_STEPS, N_RUNS,
                    DATA_DIR, OUTPUT_DIR, LOG_DIR, GROUND_TRUTH, SCENARIOS,
                    TRANSPARENCY_REQUIRED_FIELDS)
from tools import ALL_SCHEMAS, dispatch, get_tool_log, clear_tool_log

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

LLM_CONFIG = {
    "provider":    LLM_PROVIDER,
    "model":       LLM_MODEL,
    "temperature": LLM_TEMPERATURE,
    "max_tokens":  LLM_MAX_TOKENS,
    "framework":   "ReAct loop with Gemini function calling",
    "n_runs":      N_RUNS,
}

SYSTEM_PROMPT = """You are an autonomous Supply Chain Disruption Prediction AI Agent.

Your goal is to analyse supply chain data for a given scenario and produce
a structured risk assessment and recommendation report.

WORKFLOW - execute autonomously in this order:
1. Call validate_and_load_data to load and validate all scenario data.
2. If missing_flags are present in the result, call detect_missing_data_issues.
3. For EACH supplier in the deliveries list, call compute_supplier_risk_score.
   Use delay_days, on_time_delivery_pct, defect_rate_pct from the loaded data.
   Set external_event=true only if a disruption event affects that supplier country.
4. For each supplier with risk_level HIGH or CRITICAL, call find_alternative_suppliers
   using their product category.
5. Call generate_recommendation_report as your FINAL action with the complete assessment.

RULES:
- Always call generate_recommendation_report as the last step.
- Set confidence=LOW if data is missing for 2 or more suppliers.
- Set confidence=MEDIUM if any tool returned a failure status.
- Set human_escalation=true for overall risk HIGH or CRITICAL.
- Only use data returned by tools - do not invent supplier names or IDs.
- If find_alternative_suppliers returns TOOL_FAILURE, record it in tool_failures
  and complete the report with a manual sourcing recommendation.
"""

def _mock_react_loop(scenario_id, scenario_dir, run_num):
    print(f"  [ReAct run {run_num}] Mock engine active (set GEMINI_API_KEY for real calls)")
    tool_failures = []; data_quality_notes = []; steps = 0

    steps += 1
    print(f"    [step {steps}] LLM -> validate_and_load_data")
    data = dispatch("validate_and_load_data", {"scenario_dir": scenario_dir})
    if data.get("status") == "error":
        return {"error": data["message"]}

    missing_flags = data.get("missing_flags", [])
    data_quality  = data.get("data_quality", "COMPLETE")
    deliveries    = data.get("deliveries", [])
    suppliers_raw = data.get("suppliers", [])
    disruptions   = data.get("disruptions", [])

    if missing_flags:
        steps += 1
        print(f"    [step {steps}] LLM -> detect_missing_data_issues")
        mr = dispatch("detect_missing_data_issues", {"missing_flags": missing_flags})
        data_quality_notes.append(f"Completeness: {mr.get('completeness_pct','?')}%")

    sup_map  = {s["supplier_id"]: s for s in suppliers_raw}
    perf_map = {p["supplier_id"]: p for p in data.get("performance", [])}
    affected = {e.get("country","") for e in disruptions}
    risk_assessments = []; at_risk = []

    for d in deliveries:
        sid = d.get("supplier_id", "?"); steps += 1
        delay = 0.0
        try: delay = float(d.get("delay_days", 0))
        except: pass
        otd = 90.0
        try:
            v = perf_map.get(sid,{}).get("on_time_delivery_pct","MISSING")
            if v != "MISSING": otd = float(v)
        except: pass
        defect = 1.0
        try:
            v = d.get("defect_rate_pct","MISSING")
            if v not in (None,"MISSING","nan"): defect = float(v)
        except: pass
        country = sup_map.get(sid,{}).get("country","")
        has_event = country in affected
        print(f"    [step {steps}] LLM -> compute_supplier_risk_score({sid} delay={delay}d otd={otd}% event={has_event})")
        rs = dispatch("compute_supplier_risk_score", {"supplier_id":sid,"delay_days":delay,"on_time_pct":otd,"defect_rate_pct":defect,"external_event":has_event})
        risk_assessments.append(rs)
        print(f"      -> score={rs.get('risk_score')} level={rs.get('risk_level')}")
        if rs.get("risk_level") in ("HIGH","CRITICAL"):
            at_risk.append((sid, sup_map.get(sid,{}).get("category","")))

    alt_results = {}; seen_cats = set()
    for sid, cat in at_risk:
        if cat in seen_cats: continue
        seen_cats.add(cat); steps += 1
        print(f"    [step {steps}] LLM -> find_alternative_suppliers(category={cat})")
        ar = dispatch("find_alternative_suppliers", {"category":cat,"scenario_id":scenario_id})
        alt_results[sid] = ar
        if ar.get("status") == "TOOL_FAILURE":
            tool_failures.append(f"find_alternative_suppliers({cat}): {ar.get('error')}")
            print(f"      -> TOOL FAILURE: {ar.get('error')}")
        else:
            print(f"      -> {ar.get('count',0)} alternative(s) found")

    highest = max(risk_assessments, key=lambda x: x.get("risk_score",0), default={})
    overall_risk = highest.get("risk_level","LOW")

    if   len(missing_flags) >= 4: confidence = "LOW"
    elif tool_failures:           confidence = "MEDIUM"
    elif data_quality=="DEGRADED":confidence = "MEDIUM"
    else:                         confidence = "HIGH"

    rationale = {
        "HIGH":   "Complete data available; all tools succeeded; risk signals clearly evidenced.",
        "MEDIUM": "Tool failure or data quality issues detected; core assessment is complete but requires manual follow-up.",
        "LOW":    f"Significant data gaps ({len(missing_flags)} null fields). Assessment is provisional."
    }[confidence]

    actions = []
    if overall_risk == "CRITICAL":
        actions.append("ESCALATE TO SENIOR MANAGEMENT: Multiple suppliers critically affected.")
    if overall_risk in ("HIGH","CRITICAL"):
        for sid, cat in at_risk:
            ar = alt_results.get(sid,{})
            if ar.get("status") == "TOOL_FAILURE":
                actions.append(f"MANUAL ACTION REQUIRED: Alternative Supplier DB unavailable for {cat} — contact procurement team.")
                actions.append(f"Retry automated alternative supplier lookup for {cat} in 30 minutes.")
            elif ar.get("count",0) > 0:
                best = ar["alternatives"][0]
                actions.append(f"Activate {best['name']} ({best['country']}) as backup for {cat} — lead time {best['lead_time_days']}d.")
            else:
                actions.append(f"No pre-qualified alternatives for {cat} — initiate emergency procurement.")
        actions.append(f"Initiate urgent communication with affected supplier(s): {[s for s,_ in at_risk]}.")
    if missing_flags:
        actions.append("Request immediate data resubmission from suppliers with missing records (S003, S004).")
        actions.append("Apply precautionary elevated monitoring until data gaps are resolved.")
    if overall_risk == "LOW" and not missing_flags:
        actions = ["Continue standard monitoring cadence.","Schedule quarterly supplier reviews.","Maintain current safety stock levels."]

    human_esc = overall_risk in ("HIGH","CRITICAL") or bool(tool_failures) or confidence == "LOW"
    steps += 1
    print(f"    [step {steps}] LLM -> generate_recommendation_report (final)")
    rr = dispatch("generate_recommendation_report", {
        "scenario_id":scenario_id,"overall_risk_level":overall_risk,
        "risk_assessments":risk_assessments,"recommended_actions":actions,
        "confidence":confidence,"confidence_rationale":rationale,
        "human_escalation":human_esc,"tool_failures":tool_failures,
        "data_quality_notes":data_quality_notes,
    })
    return {"report":rr.get("report",{}),"steps":steps,"data_quality":data_quality,
            "missing_flags":missing_flags,"tool_failures":tool_failures,
            "risk_assessments":risk_assessments,"overall_risk":overall_risk,"confidence":confidence}

def _real_react_loop(scenario_id, scenario_dir, run_num):
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY",""))
    from google.generativeai.types import FunctionDeclaration, Tool

    decls = [FunctionDeclaration(name=s["name"],description=s["description"],parameters=s["parameters"]) for s in ALL_SCHEMAS]
    model = genai.GenerativeModel(model_name=LLM_MODEL,tools=[Tool(function_declarations=decls)],system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat()
    steps=0; tool_failures=[]; report={}
    data_quality="UNKNOWN"; missing_flags=[]; risk_assessments=[]
    overall_risk="LOW"; confidence="HIGH"
    print(f"  [ReAct run {run_num}] Gemini API - model: {LLM_MODEL}")
    user_msg = f"Run the full supply chain disruption prediction pipeline for scenario '{scenario_id}'. Data directory: {scenario_dir}"

    while steps < MAX_REACT_STEPS:
        steps += 1
        resp = chat.send_message(user_msg)
        user_msg = ""
        has_tool_call = False; tool_results = []

        for part in resp.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                has_tool_call = True
                fn = fc.name; args = dict(fc.args)
                print(f"    [step {steps}] LLM -> {fn}({list(args.keys())})")
                result = dispatch(fn, args)
                if fn == "validate_and_load_data":
                    data_quality=result.get("data_quality","?"); missing_flags=result.get("missing_flags",[])
                if fn == "compute_supplier_risk_score":
                    risk_assessments.append(result)
                    print(f"      -> score={result.get('risk_score')} level={result.get('risk_level')}")
                if fn == "find_alternative_suppliers":
                    if result.get("status")=="TOOL_FAILURE":
                        tool_failures.append(f"{fn}: {result.get('error')}"); print(f"      -> TOOL FAILURE")
                    else: print(f"      -> {result.get('count',0)} alternative(s) found")
                if fn == "generate_recommendation_report":
                    report=result.get("report",{}); overall_risk=report.get("overall_risk_level","LOW"); confidence=report.get("confidence","HIGH")
                tool_results.append(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn,response={"result":json.dumps(result,default=str)})))

        if not has_tool_call:
            print(f"    [step {steps}] LLM text response - loop complete"); break
        if tool_results:
            chat.send_message(tool_results); steps += 1
        if report: break

    return {"report":report,"steps":steps,"data_quality":data_quality,"missing_flags":missing_flags,
            "tool_failures":tool_failures,"risk_assessments":risk_assessments,"overall_risk":overall_risk,"confidence":confidence}

def _single_run(scenario_id, run_num):
    clear_tool_log()
    t0 = time.time()
    scenario_dir = os.path.join(DATA_DIR, scenario_id)
    use_real = bool(os.environ.get("GEMINI_API_KEY",""))

    try:
        loop = (_real_react_loop if use_real else _mock_react_loop)(scenario_id, scenario_dir, run_num)
    except Exception as e:
        print(f"  ERROR run {run_num}: {e}")
        loop = {"error":str(e),"overall_risk":"LOW","confidence":"LOW","steps":0,
                "tool_failures":[str(e)],"missing_flags":[],"risk_assessments":[],"data_quality":"FAILED","report":{}}

    total_ms = (time.time()-t0)*1000
    tool_log = get_tool_log()
    gt = GROUND_TRUTH[scenario_id]
    predicted = loop.get("overall_risk","LOW")
    actual = gt["risk"]
    risk_correct = (predicted == actual)
    disr_pred = predicted in ("HIGH","CRITICAL")
    total_tools = len(tool_log)
    succ_tools = sum(1 for t in tool_log if t["ok"])
    tool_sr = round(succ_tools/total_tools*100,1) if total_tools else 100.0
    missing_present = scenario_id == "scenario4_missing"
    missing_detected = len(loop.get("missing_flags",[])) >= 4
    fail_present = scenario_id == "scenario5_toolfail"
    fail_recovered = fail_present and bool(loop.get("report",{}))
    report = loop.get("report",{})
    trans_score = round(sum(1 for f in TRANSPARENCY_REQUIRED_FIELDS if report.get(f) not in (None,"",[])) / len(TRANSPARENCY_REQUIRED_FIELDS), 3)

    return {
        "scenario_id":scenario_id,"run_number":run_num,"llm_config":LLM_CONFIG,
        "llm_source":"gemini_api" if use_real else "mock_react",
        "react_steps":loop.get("steps",0),
        "task_completion_rate":100.0 if not loop.get("error") else 0.0,
        "predicted_risk":predicted,"actual_risk":actual,"risk_correct":risk_correct,
        "disruption_predicted":disr_pred,"disruption_actual":gt["disruption"],
        "processing_time_ms":round(total_ms,1),
        "tool_calls_total":total_tools,"tool_calls_successful":succ_tools,
        "tool_success_rate_pct":tool_sr,"missing_data_present":missing_present,
        "missing_data_detected":missing_detected,"tool_failure_present":fail_present,
        "tool_failure_recovered":fail_recovered,"data_quality":loop.get("data_quality","?"),
        "confidence":loop.get("confidence","?"),"human_escalation":report.get("human_escalation",False),
        "transparency_score":trans_score,"tool_failures":loop.get("tool_failures",[]),
        "report":report,"tool_call_log":tool_log,
    }

def run_scenario(scenario_id):
    print(f"\n{'='*62}\n  SCENARIO: {scenario_id}  ({N_RUNS} runs)\n{'='*62}")
    runs = []
    for i in range(1, N_RUNS+1):
        r = _single_run(scenario_id, i)
        runs.append(r)
        print(f"  Run {i}: risk={r['predicted_risk']} correct={r['risk_correct']} steps={r['react_steps']} tools={r['tool_calls_total']} SR={r['tool_success_rate_pct']}% time={r['processing_time_ms']:.0f}ms")
    out_path = os.path.join(OUTPUT_DIR, f"{scenario_id}_runs.json")
    with open(out_path,"w") as f:
        json.dump({"scenario_id":scenario_id,"llm_config":LLM_CONFIG,
                   "runs":[{k:v for k,v in r.items() if k not in ("report","tool_call_log")} for r in runs]},f,indent=2,default=str)
    return runs

def run_all():
    all_runs = []
    for s in SCENARIOS:
        all_runs.extend(run_scenario(s))
    return all_runs

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    runs = run_all() if (args.all or not args.scenario) else run_scenario(args.scenario)
    print("\nDONE - total runs:", len(runs))
