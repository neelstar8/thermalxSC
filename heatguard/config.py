"""
config.py
=========
HEATGUARD AI  -  all tunable presentation and training settings in one place.

Nothing here is a scientific constant. The meteorological rules live in
heatwave.py; these are the knobs a student may want to change without
touching any logic.
"""

# --------------------------------------------------------------------------
# Study area  (used by the map on the Heat Wave page)
# --------------------------------------------------------------------------
# One location is modelled, so the map marks a MODEL STUDY AREA - not a
# spatially resolved city-wide forecast. See the note in app.py.
STUDY_AREA = {
    "name": "Mumbai",
    "label": "Mumbai — Model Study Area",
    "lat": 19.0760,
    "lon": 72.8777,
    "zoom": 8.6,
}
TARGET_MONTH = "May"

# Token-free basemap. MapLibre / Plotly ship these without any API key:
#   "carto-darkmatter", "carto-positron", "open-street-map", "white-bg"
# "white-bg" is the offline fallback (no tiles are downloaded).
MAP_STYLE = "carto-darkmatter"
MAP_STYLE_FALLBACK = "white-bg"
MAP_HEIGHT = 420

# --------------------------------------------------------------------------
# PROJECT RISK DISPLAY THRESHOLDS
# --------------------------------------------------------------------------
# These are UI categories for demonstration only.
# They are NOT official meteorological thresholds and carry no scientific
# authority. Change them freely; every label in the app follows these values.
RISK_LOW_MAX = 0.40        # score  < 0.40           -> LOW
RISK_MODERATE_MAX = 0.70   # 0.40 <= score < 0.70    -> MODERATE
                           # score >= 0.70           -> HIGH

RISK_COLORS = {
    "LOW": "#34d399",       # cool green
    "MODERATE": "#f59e0b",  # amber
    "HIGH": "#ef4444",      # red
}

# The score at which the model reports "HEAT WAVE" rather than "NO HEAT WAVE".
DECISION_THRESHOLD = 0.50

# --------------------------------------------------------------------------
# ANN + training settings
# --------------------------------------------------------------------------
N_HIDDEN = 8
TRAIN_EPOCHS = 2500
TRAIN_LR = 0.8
SEED = 42

# Train/validation/test fractions (the remainder goes to test).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
BG = "#0a0e14"
PANEL = "#111823"
BORDER = "#1e2a3a"
TEXT = "#e6edf3"
MUTED = "#8b98a9"
ACCENT = "#38bdf8"
