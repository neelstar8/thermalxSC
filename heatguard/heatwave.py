"""
heatwave.py
===========
HEATGUARD AI  -  domain (meteorology) logic only.

This file contains NO neural-network code.
It answers one question: "what counts as a heat wave day in this project?"

-------------------------------------------------------------------------
IMPORTANT ACADEMIC DISTINCTION
-------------------------------------------------------------------------
(1) OFFICIAL IMD CRITERIA  -- reference only
    The India Meteorological Department (IMD) declares a heat wave for a
    COASTAL station (Mumbai is a coastal station) when BOTH hold:
        * maximum temperature >= 37 degC, AND
        * departure of maximum temperature from the station normal >= 4.5 degC
    Departure 4.5 - 6.4 degC  -> "Heat Wave"
    Departure  > 6.4 degC     -> "Severe Heat Wave"
    IMD also declares a heat wave for plains when the actual maximum
    temperature reaches 45 degC irrespective of the departure.

(2) PROJECT CLASSIFICATION RULE  -- what this project actually labels with
    We implement the IMD coastal idea above (Condition A and Condition B)
    and we ADD a humid-heat condition (Condition C) built on the NOAA
    heat-index / apparent-temperature formula, because heat stress in
    Mumbai during May is driven by humidity as much as by dry-bulb
    temperature.

    Condition C IS OUR OWN PROJECT RULE. It is NOT part of any official
    IMD definition and must never be presented as one.

    Condition C is also what makes this an interesting ANN problem: the
    decision boundary in (temperature, humidity) space is curved, so a
    single threshold on temperature cannot reproduce the label, but a
    small ANN can learn it.
-------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------
# Rule constants (all tunable, all documented)
# --------------------------------------------------------------------------

# Condition A - IMD coastal criteria
IMD_COASTAL_MIN_TMAX = 37.0      # degC, IMD coastal threshold
IMD_DEPARTURE_THRESHOLD = 4.5    # degC above the May normal

# Condition B - absolute extreme heat (project value for a coastal station)
PROJECT_ABSOLUTE_TMAX = 41.0     # degC

# Condition C - humid heat (PROJECT RULE, not IMD)
PROJECT_HEAT_INDEX_THRESHOLD = 46.0   # degC apparent temperature
PROJECT_HUMID_MIN_TMAX = 35.0         # degC dry-bulb floor for condition C

# Risk bands used for the UI wording (LOW / MODERATE / HIGH).
# These are PROJECT RISK DISPLAY THRESHOLDS defined in config.py - they are a
# presentation choice for the demonstration, not official meteorology.
RISK_LOW_MAX = config.RISK_LOW_MAX
RISK_MODERATE_MAX = config.RISK_MODERATE_MAX


# --------------------------------------------------------------------------
# Heat index (apparent temperature)
# --------------------------------------------------------------------------
def heat_index_celsius(temp_c, relative_humidity):
    """
    NOAA / Rothfusz heat-index ("feels like" temperature).

    The published regression works in degrees Fahrenheit, so we convert in,
    compute, and convert back out.  For cooler conditions the regression is
    not valid, so NOAA's simple averaging form is used instead.

    Accepts scalars or NumPy arrays and always returns the same shape.
    """
    t_f = np.asarray(temp_c, dtype=float) * 9.0 / 5.0 + 32.0
    rh = np.asarray(relative_humidity, dtype=float)

    # Simple form - used when it is not hot enough for the full regression.
    simple_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))

    # Full Rothfusz regression.
    full_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f * t_f
        - 0.05481717 * rh * rh
        + 0.00122874 * t_f * t_f * rh
        + 0.00085282 * t_f * rh * rh
        - 0.00000199 * t_f * t_f * rh * rh
    )

    # NOAA adjustment 1: low humidity in the 80-112 degF range.
    adj_dry = ((13.0 - rh) / 4.0) * np.sqrt(np.clip(17.0 - np.abs(t_f - 95.0), 0, None) / 17.0)
    full_f = np.where((rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0), full_f - adj_dry, full_f)

    # NOAA adjustment 2: high humidity in the 80-87 degF range.
    adj_wet = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    full_f = np.where((rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0), full_f + adj_wet, full_f)

    # The regression is only used once the simple form says it is >= 80 degF.
    hi_f = np.where(simple_f >= 80.0, full_f, simple_f)
    return (hi_f - 32.0) * 5.0 / 9.0


# --------------------------------------------------------------------------
# Station normal
# --------------------------------------------------------------------------
def may_normal_max_temp(df: pd.DataFrame) -> float:
    """
    The "normal" May maximum temperature for the station.

    A true IMD normal is a 30-year climatological average published by IMD.
    Here the normal is computed from the dataset that is loaded, which is an
    approximation.  This is stated on the Methodology page and in the README.
    """
    return float(np.mean(df["max_temp"].astype(float)))


# --------------------------------------------------------------------------
# The project classification rule
# --------------------------------------------------------------------------
def classify_heatwave(max_temp, humidity, normal_max_temp: float):
    """
    Apply the PROJECT classification rule and return a detailed breakdown.

    Returns a dict of NumPy arrays (or scalars) so that the UI can show the
    user exactly which condition fired.
    """
    max_temp = np.asarray(max_temp, dtype=float)
    humidity = np.asarray(humidity, dtype=float)

    departure = max_temp - float(normal_max_temp)
    hi = heat_index_celsius(max_temp, humidity)

    cond_a = (max_temp >= IMD_COASTAL_MIN_TMAX) & (departure >= IMD_DEPARTURE_THRESHOLD)
    cond_b = max_temp >= PROJECT_ABSOLUTE_TMAX
    cond_c = (hi >= PROJECT_HEAT_INDEX_THRESHOLD) & (max_temp >= PROJECT_HUMID_MIN_TMAX)

    label = (cond_a | cond_b | cond_c).astype(int)

    return {
        "departure": departure,
        "heat_index": hi,
        "condition_a_imd_coastal": cond_a,
        "condition_b_absolute": cond_b,
        "condition_c_humid_heat": cond_c,
        "heatwave_label": label,
    }


def add_heatwave_label(df: pd.DataFrame, normal_max_temp: float | None = None) -> pd.DataFrame:
    """
    Add / recompute the `heatwave_label` column using the project rule.

    Used when generating the demo dataset, and available if a real dataset
    is supplied without labels.
    """
    out = df.copy()
    if normal_max_temp is None:
        normal_max_temp = may_normal_max_temp(out)
    result = classify_heatwave(out["max_temp"], out["humidity"], normal_max_temp)
    out["heatwave_label"] = result["heatwave_label"]
    return out


# --------------------------------------------------------------------------
# Risk wording for the UI
# --------------------------------------------------------------------------
def risk_level(probability: float) -> str:
    """Map an ANN score in [0, 1] to LOW / MODERATE / HIGH."""
    if probability < RISK_LOW_MAX:
        return "LOW"
    if probability < RISK_MODERATE_MAX:
        return "MODERATE"
    return "HIGH"


def risk_color(level: str) -> str:
    """Restrained colour per risk band (used by cards, gauge and map marker)."""
    return config.RISK_COLORS.get(level, config.MUTED)


RULE_TEXT = f"""\
**Condition A — IMD coastal criteria**
`max_temp >= {IMD_COASTAL_MIN_TMAX} °C` **and** `max_temp - May_normal >= {IMD_DEPARTURE_THRESHOLD} °C`

**Condition B — absolute extreme (project value)**
`max_temp >= {PROJECT_ABSOLUTE_TMAX} °C`

**Condition C — humid heat (PROJECT RULE, not IMD)**
`heat_index(max_temp, humidity) >= {PROJECT_HEAT_INDEX_THRESHOLD} °C` **and** `max_temp >= {PROJECT_HUMID_MIN_TMAX} °C`

`heatwave_label = 1` if **A or B or C** is true, otherwise `0`.
"""
