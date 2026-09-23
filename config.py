"""
Supply Chain Disruption Prediction AI Agent — Configuration
============================================================
BEFORE RUNNING:
    1. Get free Groq API key: https://console.groq.com
    2. Set key (Windows):   set GROQ_API_KEY=gsk_...
       Set key (Mac/Linux): export GROQ_API_KEY=gsk_...
    3. Run: python evaluate.py

LLM SPECIFICATION (documented for reproducibility):
    Provider    : Groq Cloud API
    Model       : llama-3.3-70b-versatile
    Temperature : 0.1  (low — maximises consistency)
    Max tokens  : 1500
    Tool choice : auto (LLM decides which tools to call)
    Framework   : ReAct loop with OpenAI-compatible function calling
"""
import os

# ── API ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
LLM_MODEL       = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1500
LLM_PROVIDER    = "Google Gemini API"

# ── Experimental design ───────────────────────────────────────────────────────
MAX_REACT_STEPS = 12     # max ReAct iterations per run
N_RUNS          = 5      # repeated runs per scenario (handles LLM stochasticity)
                         # 5 scenarios × 5 runs = 25 total observations

# ── Risk thresholds (documented for reproducibility) ─────────────────────────
DELAY_MINOR_DAYS    = 3
DELAY_MODERATE_DAYS = 7
DELAY_MAJOR_DAYS    = 14
DELAY_CRITICAL_DAYS = 21
OTD_HIGH_RISK       = 65.0   # on-time delivery % below this → HIGH risk
OTD_MEDIUM_RISK     = 80.0   # on-time delivery % below this → MEDIUM risk
DEFECT_HIGH         = 3.0    # defect rate % above this → HIGH contribution
DEFECT_MEDIUM       = 2.0

# ── Responsible AI evaluation thresholds ─────────────────────────────────────
# These operationalise RQ3 (technical & responsible AI challenges)
HALLUCINATION_FLAG_KEYWORDS = [
    # If agent output contains invented supplier names/IDs not in the dataset,
    # these patterns help detect it
    "fabricated", "invented", "assumed"
]
CONFIDENCE_CALIBRATION_THRESHOLD = 0.7   # agent confidence vs correctness ratio
TRANSPARENCY_REQUIRED_FIELDS = [         # fields that MUST appear in every report
    "overall_risk_level", "confidence", "confidence_rationale",
    "human_escalation", "recommended_actions", "risk_assessments"
]

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

# ── Ground truth ──────────────────────────────────────────────────────────────
GROUND_TRUTH = {
    "scenario1_normal":     {"risk": "LOW",      "disruption": False},
    "scenario2_delay":      {"risk": "HIGH",     "disruption": True},
    "scenario3_disruption": {"risk": "CRITICAL", "disruption": True},
    "scenario4_missing":    {"risk": "MEDIUM",   "disruption": False},
    "scenario5_toolfail":   {"risk": "HIGH",     "disruption": True},
}

SCENARIOS     = list(GROUND_TRUTH.keys())
RISK_ORDER    = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
