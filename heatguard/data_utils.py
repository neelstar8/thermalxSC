"""
data_utils.py
=============
HEATGUARD AI  -  dataset loading, validation, demo-data generation,
normalisation (scaling) and the train / validation / test split.

Nothing here knows about neural networks either; it only prepares numbers.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config
import heatwave as hw

# --------------------------------------------------------------------------
# Dataset contract
# --------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "mumbai_may_weather.csv")

# Marker file. If it exists next to the CSV, the CSV is DEMO (synthetic) data.
# Replace the CSV with genuine observations and delete this marker file, and
# the app will stop showing the DEMO banner. Nothing else has to change.
DEMO_MARKER = os.path.join(DATA_DIR, "DEMO_DATA.marker")

# The six climate inputs fed to the ANN, in a fixed order.
FEATURES = ["max_temp", "min_temp", "humidity", "wind_speed", "rainfall", "cloud_cover"]
TARGET = "heatwave_label"
REQUIRED_COLUMNS = ["date"] + FEATURES + [TARGET]

# Human-friendly labels / units / slider ranges used by the UI.
FEATURE_META = {
    "max_temp":    {"label": "Maximum Temperature", "unit": "°C", "icon": "TMAX",
                    "min": 25.0, "max": 48.0, "step": 0.1, "short": "Tmax"},
    "min_temp":    {"label": "Minimum Temperature", "unit": "°C", "icon": "TMIN",
                    "min": 15.0, "max": 35.0, "step": 0.1, "short": "Tmin"},
    "humidity":    {"label": "Relative Humidity", "unit": "%", "icon": "RH",
                    "min": 10.0, "max": 100.0, "step": 1.0, "short": "Humidity"},
    "wind_speed":  {"label": "Wind Speed", "unit": "km/h", "icon": "WIND",
                    "min": 0.0, "max": 45.0, "step": 0.5, "short": "Wind"},
    "rainfall":    {"label": "Rainfall", "unit": "mm", "icon": "RAIN",
                    "min": 0.0, "max": 120.0, "step": 0.5, "short": "Rainfall"},
    "cloud_cover": {"label": "Cloud Cover", "unit": "%", "icon": "CLOUD",
                    "min": 0.0, "max": 100.0, "step": 1.0, "short": "Cloud"},
}


# Smallest dataset the app will accept before refusing (with a friendly message).
MIN_ROWS = 30
MIN_PER_CLASS = 6


# OPTIONAL geography columns. If - and only if - a supplied dataset carries
# real coordinates, the map switches from a single study-area marker to one
# marker per station. Nothing is ever invented: when these columns are absent
# the app says so and shows the single study area instead.
GEO_COLUMNS = ["station", "latitude", "longitude"]


class DatasetError(Exception):
    """Raised for any problem with the dataset so the UI can show a friendly message."""


# --------------------------------------------------------------------------
# DEMO data generator
# --------------------------------------------------------------------------
def make_demo_dataframe(start_year: int = 2010, end_year: int = 2024, seed: int = 7) -> pd.DataFrame:
    """
    Build a SMALL SYNTHETIC ("demo") May-weather table for Mumbai.

    THIS IS NOT REAL DATA. It exists only so the application can be run and
    tested before a genuine historical dataset is plugged in. The numbers are
    drawn from plausible ranges for Mumbai in May; they are not observations.

    Generation sketch (all of it is made up, deliberately simple):
      * a baseline May maximum temperature with a mild warming trend,
      * a day-to-day correlated (AR(1)) wiggle,
      * occasional 2-4 day hot spells,
      * humidity falling when the temperature anomaly rises (drier air),
      * rainfall as rare pre-monsoon showers, mostly late in the month,
      * cloud cover tied to humidity and rainfall,
      * wind and diurnal range following the same weather regime.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for year in range(start_year, end_year + 1):
        baseline = 33.0 + 0.035 * (year - start_year)     # mild warming trend
        anomaly = 0.0                                      # AR(1) state
        spell_days_left = 0
        spell_strength = 0.0

        for day in range(1, 32):
            # --- day-to-day persistence -----------------------------------
            anomaly = 0.62 * anomaly + rng.normal(0.0, 0.85)

            # --- occasional hot spell -------------------------------------
            if spell_days_left > 0:
                spell_days_left -= 1
            elif rng.random() < 0.045:
                spell_days_left = int(rng.integers(2, 5))
                spell_strength = float(rng.uniform(2.5, 6.5))
            spell = spell_strength if spell_days_left > 0 else 0.0

            # --- rainfall (rare, mostly after the 18th) -------------------
            rain_chance = 0.04 + 0.11 * max(0.0, (day - 18) / 13.0)
            if spell > 0:
                rain_chance *= 0.15
            rainfall = float(rng.gamma(1.4, 9.0)) if rng.random() < rain_chance else 0.0
            rainfall = round(min(rainfall, 115.0), 1)

            # --- maximum temperature --------------------------------------
            max_temp = baseline + anomaly + spell - 0.055 * rainfall
            max_temp = float(np.clip(max_temp, 27.0, 44.0))

            # --- humidity: drier on hot-spell days, wetter when it rains ---
            humidity = 73.0 - 2.6 * (anomaly + spell) + 0.20 * rainfall + rng.normal(0, 3.5)
            humidity = float(np.clip(humidity, 28.0, 95.0))

            # --- cloud cover ----------------------------------------------
            cloud = 0.55 * humidity - 12.0 + 0.35 * rainfall + rng.normal(0, 8.0)
            cloud = float(np.clip(cloud, 0.0, 100.0))

            # --- wind: sea breeze, a bit stronger on cloudy/rainy days ----
            wind = 11.5 + 0.05 * cloud - 0.35 * spell + rng.normal(0, 3.0)
            wind = float(np.clip(wind, 1.0, 42.0))

            # --- minimum temperature: bigger diurnal range on dry days ----
            diurnal = 9.0 - 0.048 * humidity + 0.10 * spell + rng.normal(0, 0.7)
            min_temp = float(np.clip(max_temp - max(2.0, diurnal), 18.0, 34.0))

            rows.append({
                "date": f"{year}-05-{day:02d}",
                "max_temp": round(max_temp, 1),
                "min_temp": round(min_temp, 1),
                "humidity": round(humidity, 1),
                "wind_speed": round(wind, 1),
                "rainfall": rainfall,
                "cloud_cover": round(cloud, 1),
            })

    df = pd.DataFrame(rows)
    # Labels come from the documented project rule in heatwave.py -
    # they are never random.
    df = hw.add_heatwave_label(df)
    return df


def write_demo_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Create data/mumbai_may_weather.csv (demo) plus the DEMO marker file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = make_demo_dataframe()
    df.to_csv(path, index=False)
    with open(DEMO_MARKER, "w", encoding="utf-8") as fh:
        fh.write(
            "This marker means data/mumbai_may_weather.csv is SYNTHETIC DEMO DATA.\n"
            "It was generated by data_utils.make_demo_dataframe() for application\n"
            "testing only. It is NOT a record of real observations.\n\n"
            "Replace the CSV with genuine May observations for Mumbai and delete\n"
            "this file; the app will then stop showing the DEMO banner.\n"
        )
    return df


# --------------------------------------------------------------------------
# Loading + validation
# --------------------------------------------------------------------------
def load_dataset(path: str = DATA_PATH, create_demo_if_missing: bool = True):
    """
    Load the dataset and validate it.

    Returns (dataframe, meta) where meta describes the source.
    Raises DatasetError with a readable message on any problem.
    """
    created = False
    if not os.path.exists(path):
        if not create_demo_if_missing:
            raise DatasetError(f"Dataset not found at `{path}`.")
        write_demo_dataset(path)
        created = True

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise DatasetError(f"`{os.path.basename(path)}` is empty - it contains no rows at all.")
    except Exception as exc:                                # malformed CSV
        raise DatasetError(f"Could not read `{os.path.basename(path)}`: {exc}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            "The dataset is missing required column(s): " + ", ".join(missing)
            + ".\nExpected columns: " + ", ".join(REQUIRED_COLUMNS) + "."
        )

    # Numeric coercion - anything unparseable becomes NaN and is dropped.
    before = len(df)
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    dropped = before - len(df)

    if len(df) == 0:
        raise DatasetError("After removing invalid rows the dataset is empty.")

    df[TARGET] = df[TARGET].astype(int)
    bad_labels = sorted(set(df[TARGET].unique()) - {0, 1})
    if bad_labels:
        raise DatasetError(f"`{TARGET}` must be 0 or 1 only, but found: {bad_labels}.")

    # An ANN needs enough rows - and enough of BOTH classes - to be trainable
    # at all. Fail early with a readable message rather than deep inside NumPy.
    n_pos = int(df[TARGET].sum())
    n_neg = int(len(df) - n_pos)
    if len(df) < MIN_ROWS:
        raise DatasetError(
            f"Only {len(df)} usable row(s) found. At least {MIN_ROWS} are needed to train "
            "and evaluate the network."
        )
    if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        raise DatasetError(
            f"The dataset needs at least {MIN_PER_CLASS} rows of each class, but it has "
            f"{n_pos} heat-wave row(s) and {n_neg} normal row(s)."
        )

    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # Station-level geography is optional and is only reported as available
    # when every one of the three columns is present AND parses to real
    # coordinates. Otherwise the app falls back to the single study area.
    has_geo = all(c in df.columns for c in GEO_COLUMNS)
    stations = None
    if has_geo:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        geo = df.dropna(subset=["latitude", "longitude"])
        has_geo = len(geo) > 0
        if has_geo:
            stations = (geo.groupby("station")[["latitude", "longitude"]]
                        .first().reset_index())

    meta = {
        "has_station_geography": bool(has_geo),
        "stations": stations,
        "path": path,
        "is_demo": os.path.exists(DEMO_MARKER),
        "created_now": created,
        "rows": int(len(df)),
        "dropped_rows": int(dropped),
        "n_heatwave": int(df[TARGET].sum()),
        "n_normal": int(len(df) - df[TARGET].sum()),
        "normal_max_temp": hw.may_normal_max_temp(df),
    }
    return df, meta


# --------------------------------------------------------------------------
# NORMALISATION  (min-max scaling)
# --------------------------------------------------------------------------
class MinMaxScaler:
    """
    Min-max scaling:            x_scaled = (x - min) / (max - min)

    Why normalise?  The six inputs live on completely different scales
    (rainfall 0-115 mm, humidity 28-95 %, temperature 27-44 degC).  Without
    scaling, the large-valued inputs dominate the weighted sum, sigmoid
    saturates, and gradient descent crawls.  After scaling every input sits
    in roughly [0, 1] so all weights start on an equal footing.

    The min and max are computed from the TRAINING DATA ONLY and are then
    reused unchanged for validation, test and live user input.  Fitting on
    all data would leak information from the test set.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        # Guard against a constant column (max == min) -> divide by zero.
        self.range_ = np.where((self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise DatasetError("Scaler used before it was fitted.")
        X = np.asarray(X, dtype=float).reshape(-1, len(self.min_))
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": FEATURES,
            "train_min": np.round(self.min_, 2),
            "train_max": np.round(self.max_, 2),
        })


# --------------------------------------------------------------------------
# Train / validation / test split
# --------------------------------------------------------------------------
def split_dataset(df: pd.DataFrame, train_frac=config.TRAIN_FRAC,
                  val_frac=config.VAL_FRAC, seed: int = config.SEED):
    """
    Stratified random split into train / validation / test.

    Stratified means the heat-wave / no-heat-wave ratio is preserved in each
    part, which matters because heat-wave days are the minority class.
    The split is done on ROW INDICES before any scaling, so the test rows
    never influence the scaler or the weights.
    """
    rng = np.random.default_rng(seed)
    idx_train, idx_val, idx_test = [], [], []

    for label in (0, 1):
        idx = np.array(df.index[df[TARGET] == label].to_numpy(), copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(train_frac * n))
        n_va = int(round(val_frac * n))
        # Make sure every split gets at least one row of this class if possible.
        n_tr = min(n_tr, max(n - 2, 1)) if n >= 3 else n
        n_va = min(n_va, max(n - n_tr - 1, 0)) if n >= 3 else 0
        idx_train += list(idx[:n_tr])
        idx_val += list(idx[n_tr:n_tr + n_va])
        idx_test += list(idx[n_tr + n_va:])

    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)
    # dtype=int matters: an empty np.array([]) would default to float and
    # could not be used as row indices.
    return (np.array(idx_train, dtype=int),
            np.array(idx_val, dtype=int),
            np.array(idx_test, dtype=int))


def prepare_training_data(df: pd.DataFrame, seed: int = config.SEED):
    """
    Full preprocessing pipeline:
        split -> fit scaler on TRAIN ONLY -> transform every split.

    Returns a dictionary that the app and the ANN both use.
    """
    i_tr, i_va, i_te = split_dataset(df, seed=seed)

    X_all = df[FEATURES].to_numpy(dtype=float)
    y_all = df[TARGET].to_numpy(dtype=float).reshape(-1, 1)

    scaler = MinMaxScaler().fit(X_all[i_tr])          # TRAIN ONLY

    return {
        "scaler": scaler,
        "idx": {"train": i_tr, "val": i_va, "test": i_te},
        "X_train": scaler.transform(X_all[i_tr]), "y_train": y_all[i_tr],
        "X_val":   scaler.transform(X_all[i_va]), "y_val":   y_all[i_va],
        "X_test":  scaler.transform(X_all[i_te]), "y_test":  y_all[i_te],
        "X_all_scaled": scaler.transform(X_all), "y_all": y_all,
        "raw_train": X_all[i_tr],
    }


def default_inputs(df: pd.DataFrame) -> dict:
    """Sensible starting slider values: the dataset's average May day."""
    return {f: float(round(df[f].mean(), 1)) for f in FEATURES}
