"""
Generate all 5 experimental scenario datasets.
Run once: python datasets.py
"""
import pandas as pd, os, random
random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

SUPPLIERS = [
    {"supplier_id":"S001","name":"Meridian Components Ltd",  "country":"Germany","category":"Electronic Parts","lead_time_days":14,"reliability":0.95},
    {"supplier_id":"S002","name":"AsiaFlex Manufacturing",   "country":"China",  "category":"Mechanical Parts","lead_time_days":21,"reliability":0.88},
    {"supplier_id":"S003","name":"Nordic Raw Materials",     "country":"Sweden", "category":"Raw Materials",   "lead_time_days":10,"reliability":0.97},
    {"supplier_id":"S004","name":"Balkan Precision Works",   "country":"Serbia", "category":"Electronic Parts","lead_time_days":18,"reliability":0.82},
    {"supplier_id":"S005","name":"Gulf Logistics Partners",  "country":"UAE",    "category":"Logistics",       "lead_time_days":7, "reliability":0.91},
]

ALTERNATIVES = [
    {"supplier_id":"A001","name":"EuroCircuit Backup", "country":"Poland", "category":"Electronic Parts","lead_time_days":16,"capacity_pct":80,"reliability":0.89},
    {"supplier_id":"A002","name":"Pacific Mfg Co",     "country":"Vietnam","category":"Mechanical Parts","lead_time_days":25,"capacity_pct":65,"reliability":0.84},
    {"supplier_id":"A003","name":"Alpine Raw Supplies","country":"Austria","category":"Raw Materials",   "lead_time_days":12,"capacity_pct":90,"reliability":0.93},
]

def mk_deliveries(overrides=None):
    overrides = overrides or {}
    rows = []
    for s in SUPPLIERS:
        sid = s["supplier_id"]
        o = overrides.get(sid, {})
        rows.append({
            "order_id":          f"ORD-2024-{sid}",
            "supplier_id":       sid,
            "supplier_name":     s["name"],
            "expected_delivery": "2024-11-15",
            "actual_delivery":   o.get("actual","2024-11-15"),
            "delay_days":        o.get("delay", 0),
            "quantity_ordered":  o.get("qty",   random.randint(500,2000)),
            "quantity_received": o.get("recv",  None),
            "status":            o.get("status","On Time"),
            "defect_rate_pct":   o.get("defect",round(random.uniform(0.3,1.8),2)),
        })
    return pd.DataFrame(rows)

def mk_performance(overrides=None):
    overrides = overrides or {}
    rows = []
    for s in SUPPLIERS:
        sid = s["supplier_id"]
        o = overrides.get(sid, {})
        rows.append({
            "supplier_id":          sid,
            "on_time_delivery_pct": o.get("otd",  round(s["reliability"]*100,1)),
            "avg_delay_days":       o.get("avgd", round(random.uniform(0.5,2.5),1)),
            "defect_rate_pct":      o.get("defect",round(random.uniform(0.3,1.8),2)),
            "communication_score":  o.get("comm", round(random.uniform(7.5,9.8),1)),
            "risk_flag":            o.get("flag", "Normal"),
        })
    return pd.DataFrame(rows)

def mk_events(events):
    cols=["event_id","event_type","region","country","severity","description","date_reported","source"]
    return pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)

# ── Scenario 1 — Normal ───────────────────────────────────────────────────────
d = os.path.join(DATA,"scenario1_normal"); os.makedirs(d,exist_ok=True)
pd.DataFrame(SUPPLIERS).to_csv(f"{d}/suppliers.csv",index=False)
pd.DataFrame(ALTERNATIVES).to_csv(f"{d}/alternative_suppliers.csv",index=False)
mk_deliveries().to_csv(f"{d}/deliveries.csv",index=False)
mk_performance().to_csv(f"{d}/performance.csv",index=False)
mk_events([]).to_csv(f"{d}/disruption_events.csv",index=False)

# ── Scenario 2 — Supplier delay ───────────────────────────────────────────────
d = os.path.join(DATA,"scenario2_delay"); os.makedirs(d,exist_ok=True)
pd.DataFrame(SUPPLIERS).to_csv(f"{d}/suppliers.csv",index=False)
pd.DataFrame(ALTERNATIVES).to_csv(f"{d}/alternative_suppliers.csv",index=False)
mk_deliveries({"S002":{"actual":"2024-12-03","delay":18,"qty":1200,"recv":0,"status":"Delayed","defect":2.4}}).to_csv(f"{d}/deliveries.csv",index=False)
mk_performance({"S002":{"otd":61.0,"avgd":12.5,"defect":2.4,"comm":5.2,"flag":"HIGH RISK"}}).to_csv(f"{d}/performance.csv",index=False)
mk_events([]).to_csv(f"{d}/disruption_events.csv",index=False)

# ── Scenario 3 — External disruption ─────────────────────────────────────────
d = os.path.join(DATA,"scenario3_disruption"); os.makedirs(d,exist_ok=True)
pd.DataFrame(SUPPLIERS).to_csv(f"{d}/suppliers.csv",index=False)
pd.DataFrame(ALTERNATIVES).to_csv(f"{d}/alternative_suppliers.csv",index=False)
mk_deliveries({
    "S002":{"actual":"2024-12-10","delay":25,"qty":1200,"recv":0,"status":"Disrupted","defect":1.9},
    "S005":{"actual":"2024-12-08","delay":23,"qty":800, "recv":0,"status":"Disrupted","defect":1.1},
}).to_csv(f"{d}/deliveries.csv",index=False)
mk_performance({
    "S002":{"otd":54.0,"avgd":18.0,"defect":1.9,"comm":4.5,"flag":"HIGH RISK"},
    "S005":{"otd":59.0,"avgd":16.0,"defect":1.1,"comm":4.8,"flag":"HIGH RISK"},
}).to_csv(f"{d}/performance.csv",index=False)
mk_events([
    ["EVT-001","Port Closure","East Asia","China","HIGH",
     "Shanghai container port closed — Typhoon Megi. Estimated 3-4 week outbound cargo disruption.",
     "2024-11-01","Reuters/Port Authority"],
    ["EVT-002","Logistics Disruption","Middle East","UAE","MEDIUM",
     "Regional fuel shortage reducing freight carrier capacity by 25-30%.",
     "2024-11-03","Lloyd's List Intelligence"],
]).to_csv(f"{d}/disruption_events.csv",index=False)

# ── Scenario 4 — Missing data ─────────────────────────────────────────────────
d = os.path.join(DATA,"scenario4_missing"); os.makedirs(d,exist_ok=True)
pd.DataFrame(SUPPLIERS).to_csv(f"{d}/suppliers.csv",index=False)
pd.DataFrame(ALTERNATIVES).to_csv(f"{d}/alternative_suppliers.csv",index=False)
del_df = mk_deliveries()
del_df.loc[del_df.supplier_id=="S003","actual_delivery"] = None
del_df.loc[del_df.supplier_id=="S003","status"]          = None
del_df.loc[del_df.supplier_id=="S004","quantity_received"]= None
del_df.loc[del_df.supplier_id=="S004","defect_rate_pct"] = None
del_df.to_csv(f"{d}/deliveries.csv",index=False)
perf_df = mk_performance()
perf_df.loc[perf_df.supplier_id=="S003","on_time_delivery_pct"] = None
perf_df.loc[perf_df.supplier_id=="S004","communication_score"]  = None
perf_df.to_csv(f"{d}/performance.csv",index=False)
mk_events([]).to_csv(f"{d}/disruption_events.csv",index=False)

# ── Scenario 5 — Tool failure ─────────────────────────────────────────────────
d = os.path.join(DATA,"scenario5_toolfail"); os.makedirs(d,exist_ok=True)
pd.DataFrame(SUPPLIERS).to_csv(f"{d}/suppliers.csv",index=False)
pd.DataFrame(ALTERNATIVES).to_csv(f"{d}/alternative_suppliers.csv",index=False)
mk_deliveries({"S002":{"actual":"2024-12-03","delay":18,"qty":1200,"recv":0,"status":"Delayed","defect":2.4}}).to_csv(f"{d}/deliveries.csv",index=False)
mk_performance({"S002":{"otd":61.0,"avgd":12.5,"defect":2.4,"comm":5.2,"flag":"HIGH RISK"}}).to_csv(f"{d}/performance.csv",index=False)
mk_events([]).to_csv(f"{d}/disruption_events.csv",index=False)

print("All 5 scenario datasets generated.")
