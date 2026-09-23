"""
Tool definitions for the Supply Chain Disruption Prediction AI Agent.

Each tool has:
  - SCHEMA  : the JSON schema passed to the OpenAI API so the LLM can
              autonomously decide to call it
  - IMPL    : the actual Python function that executes when called
"""
import pandas as pd, os, json, time
from config import (DATA_DIR, DELAY_MINOR_DAYS, DELAY_MODERATE_DAYS,
                    DELAY_MAJOR_DAYS, DELAY_CRITICAL_DAYS,
                    OTD_HIGH_RISK, OTD_MEDIUM_RISK,
                    DEFECT_HIGH, DEFECT_MEDIUM)

# ── Tool call log ─────────────────────────────────────────────────────────────
_log = []

def get_tool_log():    return list(_log)
def clear_tool_log():  _log.clear()

def _record(name, args, result, ok, ms, err=None):
    _log.append({"tool": name, "args": args, "ok": ok,
                 "ms": round(ms,1), "error": err,
                 "result_summary": str(result)[:120]})

# ═════════════════════════════════════════════════════════════════════════════
# TOOL 1 — validate_and_load_data
# ═════════════════════════════════════════════════════════════════════════════
SCHEMA_VALIDATE = {
    "name": "validate_and_load_data",
    "description": (
        "Load all supply chain CSV files for the current scenario and perform "
        "data quality validation. Returns supplier records, delivery records, "
        "performance history, disruption events, alternative suppliers, and a "
        "list of any missing or null data fields detected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scenario_dir": {
                "type": "string",
                "description": "Absolute path to the scenario data directory."
            }
        },
        "required": ["scenario_dir"]
    }
}

def validate_and_load_data(scenario_dir: str) -> dict:
    t0 = time.time()
    try:
        deliveries   = pd.read_csv(os.path.join(scenario_dir,"deliveries.csv"))
        performance  = pd.read_csv(os.path.join(scenario_dir,"performance.csv"))
        suppliers    = pd.read_csv(os.path.join(scenario_dir,"suppliers.csv"))
        disruptions  = pd.read_csv(os.path.join(scenario_dir,"disruption_events.csv"))
        alternatives = pd.read_csv(os.path.join(scenario_dir,"alternative_suppliers.csv"))

        missing = []
        for nm, df in [("deliveries",deliveries),("performance",performance)]:
            for col in df.columns:
                n = int(df[col].isnull().sum())
                if n:
                    missing.append(f"{nm}.{col}: {n} null value(s)")

        result = {
            "status":        "ok",
            "data_quality":  "DEGRADED" if missing else "COMPLETE",
            "missing_flags": missing,
            "suppliers":     suppliers.to_dict("records"),
            "deliveries":    deliveries.fillna("MISSING").to_dict("records"),
            "performance":   performance.fillna("MISSING").to_dict("records"),
            "disruptions":   disruptions.to_dict("records"),
            "alternatives":  alternatives.to_dict("records"),
            "record_counts": {
                "suppliers":    len(suppliers),
                "deliveries":   len(deliveries),
                "disruptions":  len(disruptions),
                "alternatives": len(alternatives),
            }
        }
        ms = (time.time()-t0)*1000
        _record("validate_and_load_data",{"scenario_dir":scenario_dir},result,True,ms)
        return result
    except Exception as e:
        ms = (time.time()-t0)*1000
        err = {"status":"error","message":str(e),"data_quality":"FAILED"}
        _record("validate_and_load_data",{"scenario_dir":scenario_dir},err,False,ms,str(e))
        return err

# ═════════════════════════════════════════════════════════════════════════════
# TOOL 2 — compute_supplier_risk_score
# ═════════════════════════════════════════════════════════════════════════════
SCHEMA_RISK = {
    "name": "compute_supplier_risk_score",
    "description": (
        "Compute a quantitative risk score (0–100) and categorical risk level "
        "(LOW / MEDIUM / HIGH / CRITICAL) for a specific supplier based on "
        "delivery delay, on-time delivery percentage, defect rate, and whether "
        "an active external disruption event affects their region. Returns the "
        "score, level, and a list of contributing risk factors."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "supplier_id":        {"type":"string","description":"e.g. S001"},
            "delay_days":         {"type":"number","description":"Current delivery delay in days (0 if none)"},
            "on_time_pct":        {"type":"number","description":"Historical on-time delivery percentage (0-100)"},
            "defect_rate_pct":    {"type":"number","description":"Defect rate percentage"},
            "external_event":     {"type":"boolean","description":"True if an active disruption event affects this supplier's country/region"},
        },
        "required":["supplier_id","delay_days","on_time_pct","defect_rate_pct","external_event"]
    }
}

def compute_supplier_risk_score(supplier_id, delay_days, on_time_pct,
                                 defect_rate_pct, external_event):
    t0 = time.time()
    try:
        score, factors = 0.0, []
        # Delay (40 pts max)
        if   delay_days >= DELAY_CRITICAL_DAYS: score+=40; factors.append(f"Critical delay {delay_days}d (≥{DELAY_CRITICAL_DAYS}d threshold)")
        elif delay_days >= DELAY_MAJOR_DAYS:    score+=28; factors.append(f"Major delay {delay_days}d (≥{DELAY_MAJOR_DAYS}d threshold)")
        elif delay_days >= DELAY_MODERATE_DAYS: score+=16; factors.append(f"Moderate delay {delay_days}d")
        elif delay_days >= DELAY_MINOR_DAYS:    score+=6;  factors.append(f"Minor delay {delay_days}d")
        # OTD (30 pts max)
        if   on_time_pct < OTD_HIGH_RISK:   score+=30; factors.append(f"Very low OTD {on_time_pct}% (threshold {OTD_HIGH_RISK}%)")
        elif on_time_pct < OTD_MEDIUM_RISK: score+=16; factors.append(f"Below-target OTD {on_time_pct}%")
        elif on_time_pct < 90:              score+=6;  factors.append(f"Slightly low OTD {on_time_pct}%")
        # Defect (15 pts max)
        if   defect_rate_pct > DEFECT_HIGH:   score+=15; factors.append(f"High defect rate {defect_rate_pct}%")
        elif defect_rate_pct > DEFECT_MEDIUM: score+=8;  factors.append(f"Elevated defect rate {defect_rate_pct}%")
        # External event (15 pts)
        if external_event: score+=15; factors.append("Active external disruption event in supplier region")
        score = min(score, 100.0)
        if   score >= 70: level="CRITICAL"
        elif score >= 45: level="HIGH"
        elif score >= 18: level="MEDIUM"
        else:             level="LOW"

        result = {"supplier_id":supplier_id,"risk_score":round(score,1),
                  "risk_level":level,"contributing_factors":factors}
        ms=(time.time()-t0)*1000
        _record("compute_supplier_risk_score",
                {"supplier_id":supplier_id,"delay_days":delay_days},result,True,ms)
        return result
    except Exception as e:
        ms=(time.time()-t0)*1000
        err={"error":str(e),"supplier_id":supplier_id}
        _record("compute_supplier_risk_score",{"supplier_id":supplier_id},err,False,ms,str(e))
        return err

# ═════════════════════════════════════════════════════════════════════════════
# TOOL 3 — find_alternative_suppliers
# ═════════════════════════════════════════════════════════════════════════════
SCHEMA_ALTS = {
    "name": "find_alternative_suppliers",
    "description": (
        "Query the alternative supplier database to find backup suppliers for "
        "a given product category. Returns a list of qualified alternatives with "
        "lead times, capacity percentages, and reliability scores. "
        "Raises a tool error if the database is unreachable (simulated in Scenario 5)."
    ),
    "parameters": {
        "type":"object",
        "properties": {
            "category":         {"type":"string","description":"Product category e.g. 'Mechanical Parts'"},
            "scenario_id":      {"type":"string","description":"Current scenario ID — used internally to simulate tool failures"},
        },
        "required":["category","scenario_id"]
    }
}

def find_alternative_suppliers(category, scenario_id, _alternatives_cache=None):
    t0 = time.time()
    # Simulate tool failure for scenario 5
    if scenario_id == "scenario5_toolfail":
        ms=(time.time()-t0)*1000
        err={"status":"TOOL_FAILURE",
             "error":"ConnectionError: Alternative Supplier DB unreachable — connection timed out after 30s",
             "category":category,"alternatives":[]}
        _record("find_alternative_suppliers",{"category":category,"scenario_id":scenario_id},
                err,False,ms,"DB unreachable")
        return err
    try:
        # Load from file
        scenario_dir = os.path.join(DATA_DIR, scenario_id)
        df = pd.read_csv(os.path.join(scenario_dir,"alternative_suppliers.csv"))
        matches = df[df["category"]==category].to_dict("records")
        result = {"status":"SUCCESS","category":category,
                  "count":len(matches),"alternatives":matches}
        ms=(time.time()-t0)*1000
        _record("find_alternative_suppliers",{"category":category},result,True,ms)
        return result
    except Exception as e:
        ms=(time.time()-t0)*1000
        err={"status":"TOOL_FAILURE","error":str(e),"category":category,"alternatives":[]}
        _record("find_alternative_suppliers",{"category":category},err,False,ms,str(e))
        return err

# ═════════════════════════════════════════════════════════════════════════════
# TOOL 4 — detect_missing_data_issues
# ═════════════════════════════════════════════════════════════════════════════
SCHEMA_MISSING = {
    "name": "detect_missing_data_issues",
    "description": (
        "Analyse the loaded dataset for missing, null, or corrupted field values. "
        "Returns a structured report of which suppliers have incomplete data, "
        "which fields are affected, and the overall data completeness percentage. "
        "Call this tool when the initial data load flags missing_flags."
    ),
    "parameters": {
        "type":"object",
        "properties": {
            "missing_flags": {
                "type":"array","items":{"type":"string"},
                "description":"The missing_flags list returned by validate_and_load_data"
            }
        },
        "required":["missing_flags"]
    }
}

def detect_missing_data_issues(missing_flags):
    t0 = time.time()
    try:
        total_possible = 10   # approx key fields per supplier record
        pct_missing = round(len(missing_flags)/total_possible*100, 1)
        severity = ("CRITICAL" if pct_missing>50 else
                    "HIGH"     if pct_missing>30 else
                    "MEDIUM"   if pct_missing>10 else "LOW")
        result = {
            "status":               "ok",
            "total_missing_fields": len(missing_flags),
            "missing_field_details":missing_flags,
            "completeness_pct":     round(100-pct_missing,1),
            "severity":             severity,
            "recommendation":       (
                "Request immediate data resubmission from affected suppliers. "
                "Do not generate high-confidence risk recommendations until "
                "data gaps are resolved. Apply precautionary elevated monitoring."
            )
        }
        ms=(time.time()-t0)*1000
        _record("detect_missing_data_issues",{"flags":len(missing_flags)},result,True,ms)
        return result
    except Exception as e:
        ms=(time.time()-t0)*1000
        err={"error":str(e)}
        _record("detect_missing_data_issues",{},err,False,ms,str(e))
        return err

# ═════════════════════════════════════════════════════════════════════════════
# TOOL 5 — generate_recommendation_report
# ═════════════════════════════════════════════════════════════════════════════
SCHEMA_REPORT = {
    "name": "generate_recommendation_report",
    "description": (
        "Compile all risk assessments, alternative supplier findings, and "
        "data quality information into a structured final recommendation report. "
        "Call this as the LAST tool after all risk scoring and alternative "
        "supplier lookup steps are complete."
    ),
    "parameters": {
        "type":"object",
        "properties": {
            "scenario_id":        {"type":"string"},
            "overall_risk_level": {"type":"string","enum":["LOW","MEDIUM","HIGH","CRITICAL"]},
            "risk_assessments":   {"type":"array","items":{"type":"object"},
                                   "description":"List of per-supplier risk score results"},
            "recommended_actions":{"type":"array","items":{"type":"string"},
                                   "description":"Prioritised list of recommended actions"},
            "confidence":         {"type":"string","enum":["HIGH","MEDIUM","LOW"]},
            "confidence_rationale":{"type":"string"},
            "human_escalation":   {"type":"boolean"},
            "tool_failures":      {"type":"array","items":{"type":"string"},
                                   "description":"Any tool failures encountered during this run"},
            "data_quality_notes": {"type":"array","items":{"type":"string"}},
        },
        "required":["scenario_id","overall_risk_level","risk_assessments",
                    "recommended_actions","confidence","human_escalation"]
    }
}

def generate_recommendation_report(scenario_id, overall_risk_level,
                                    risk_assessments, recommended_actions,
                                    confidence, human_escalation,
                                    confidence_rationale="",
                                    tool_failures=None, data_quality_notes=None):
    t0 = time.time()
    try:
        report = {
            "report_id":            f"RPT-{scenario_id}-{int(time.time())}",
            "scenario_id":          scenario_id,
            "overall_risk_level":   overall_risk_level,
            "risk_assessments":     risk_assessments,
            "recommended_actions":  recommended_actions,
            "confidence":           confidence,
            "confidence_rationale": confidence_rationale,
            "human_escalation":     human_escalation,
            "tool_failures":        tool_failures or [],
            "data_quality_notes":   data_quality_notes or [],
            "workflow_complete":    True,
        }
        ms=(time.time()-t0)*1000
        _record("generate_recommendation_report",{"scenario_id":scenario_id},report,True,ms)
        return {"status":"ok","report":report}
    except Exception as e:
        ms=(time.time()-t0)*1000
        err={"status":"error","error":str(e)}
        _record("generate_recommendation_report",{"scenario_id":scenario_id},err,False,ms,str(e))
        return err

# ── Master schema list (passed to OpenAI API) ─────────────────────────────────
ALL_SCHEMAS = [SCHEMA_VALIDATE, SCHEMA_RISK, SCHEMA_ALTS,
               SCHEMA_MISSING, SCHEMA_REPORT]

# ── Dispatcher ────────────────────────────────────────────────────────────────
TOOL_MAP = {
    "validate_and_load_data":        validate_and_load_data,
    "compute_supplier_risk_score":   compute_supplier_risk_score,
    "find_alternative_suppliers":    find_alternative_suppliers,
    "detect_missing_data_issues":    detect_missing_data_issues,
    "generate_recommendation_report":generate_recommendation_report,
}

def dispatch(tool_name: str, args: dict):
    fn = TOOL_MAP.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return fn(**args)
