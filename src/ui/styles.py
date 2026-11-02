"""
src/ui/styles.py

Dark navy professional theme for GeoFlood AI Phase 4.
Cyan/blue accent palette, compact analytical cards, clear typography.
"""

DARK_CSS = """
<style>
/* ---------- BASE THEME ---------- */
.stApp {
    background-color: #0a1628;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1e35 !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #94afd1 !important;
}

/* Main header */
h1 { color: #00d4ff !important; font-weight: 700; letter-spacing: -0.5px; }
h2 { color: #7dd3fc !important; font-weight: 600; }
h3 { color: #93c5fd !important; font-weight: 600; }
h4 { color: #bfdbfe !important; font-weight: 600; }

/* Tabs */
[data-baseweb="tab-list"] {
    background-color: #0d1e35 !important;
    border-bottom: 1px solid #1e3a5f;
    gap: 2px;
}
[data-baseweb="tab"] {
    color: #7dd3fc !important;
    background: transparent !important;
    border-radius: 6px 6px 0 0;
    font-weight: 500;
    font-size: 13px;
    padding: 6px 14px !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff;
}

/* Buttons */
.stButton > button {
    background-color: #1e3a5f;
    color: #7dd3fc;
    border: 1px solid #2563eb;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.15s;
}
.stButton > button:hover {
    background-color: #2563eb;
    color: #fff;
    border-color: #00d4ff;
}
.stButton > button[kind="primary"] {
    background-color: #0284c7 !important;
    color: #fff !important;
    border-color: #00d4ff !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #0369a1 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #0d1e35;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #7dd3fc !important; font-size: 12px; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 22px; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 12px; }

/* Info/Warning/Error boxes */
[data-testid="stAlert"] {
    border-radius: 6px;
    border-left-width: 3px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: #0d1e35 !important;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
}

/* Number inputs and selects */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div {
    background-color: #0d1e35 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e3a5f !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div {
    background-color: #1e3a5f !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #0d1e35;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
}

/* Code blocks */
pre, code {
    background-color: #0d1e35 !important;
    border: 1px solid #1e3a5f;
    color: #7dd3fc !important;
    border-radius: 4px;
}

/* Dividers */
hr { border-color: #1e3a5f; }

/* ---------- CUSTOM CARDS ---------- */
.intel-card {
    background: #0d1e35;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.intel-card-accent {
    background: #0d1e35;
    border-left: 3px solid #00d4ff;
    border-top: 1px solid #1e3a5f;
    border-right: 1px solid #1e3a5f;
    border-bottom: 1px solid #1e3a5f;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.card-high {
    background: rgba(220, 38, 38, 0.08);
    border-left: 3px solid #dc2626;
    border-top: 1px solid #7f1d1d;
    border-right: 1px solid #7f1d1d;
    border-bottom: 1px solid #7f1d1d;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.card-moderate {
    background: rgba(217, 119, 6, 0.08);
    border-left: 3px solid #d97706;
    border-top: 1px solid #78350f;
    border-right: 1px solid #78350f;
    border-bottom: 1px solid #78350f;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.card-low {
    background: rgba(5, 150, 105, 0.08);
    border-left: 3px solid #059669;
    border-top: 1px solid #064e3b;
    border-right: 1px solid #064e3b;
    border-bottom: 1px solid #064e3b;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.card-water {
    background: rgba(2, 132, 199, 0.08);
    border-left: 3px solid #0284c7;
    border-top: 1px solid #0c4a6e;
    border-right: 1px solid #0c4a6e;
    border-bottom: 1px solid #0c4a6e;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.card-unavail {
    background: rgba(107, 114, 128, 0.08);
    border-left: 3px solid #6b7280;
    border-top: 1px solid #374151;
    border-right: 1px solid #374151;
    border-bottom: 1px solid #374151;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.label-text { color: #7dd3fc; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.value-text { color: #e2e8f0; font-size: 14px; }
.prob-large { font-size: 32px; font-weight: 800; color: #00d4ff; }
.prob-high { color: #f87171; }
.prob-mod { color: #fbbf24; }
.prob-low { color: #34d399; }
.prob-water { color: #38bdf8; }

.section-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin: 16px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e3a5f;
}

.legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    font-size: 12px;
    color: #94afd1;
}
.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    flex-shrink: 0;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.badge-live { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.badge-sample { background: #431407; color: #fb923c; border: 1px solid #c2410c; }
.badge-prediction { background: #0c1a4d; color: #93c5fd; border: 1px solid #1d4ed8; }
.badge-observation { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.badge-synthetic { background: #2d1b69; color: #c4b5fd; border: 1px solid #7c3aed; }
.badge-warning { background: #451a03; color: #fbbf24; border: 1px solid #d97706; }

.step-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e3a5f;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    color: #7dd3fc;
    margin-bottom: 8px;
}
</style>
"""


PLOT_LAYOUT = dict(
    paper_bgcolor="#0a1628",
    plot_bgcolor="#0d1e35",
    font=dict(color="#94afd1", size=12),
    margin=dict(l=40, r=20, t=30, b=30),
)

SUSCEP_COLORS = {
    "High Risk":      "#ef4444",
    "Moderate Risk":  "#f59e0b",
    "Low Risk":       "#10b981",
    "Permanent Water": "#0ea5e9",
    "Unavailable":    "#6b7280",
}
# Backward compatibility alias
SUSC_COLORS = SUSCEP_COLORS

MODEL_COLORS = {
    "neural_network":      "#00d4ff",
    "random_forest":       "#4ade80",
    "logistic_regression": "#f59e0b",
}


def risk_card_class(category: str) -> str:
    mapping = {
        "High Risk": "card-high",
        "Moderate Risk": "card-moderate",
        "Low Risk": "card-low",
        "Permanent Water": "card-water",
    }
    return mapping.get(category, "card-unavail")


def prob_color_class(prob: float) -> str:
    if prob >= 0.70:
        return "prob-high"
    elif prob >= 0.35:
        return "prob-mod"
    else:
        return "prob-low"
