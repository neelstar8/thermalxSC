# 🔥 HEATGUARD AI
### ANN-Based Heat Wave Prediction for May — Mumbai

A small **Soft Computing** academic project. An artificial neural network, written **from
scratch in NumPy**, predicts whether heat-wave conditions are likely for a May day in Mumbai
from six climate variables — and the application *shows you the mathematics while it happens*.

> No TensorFlow. No PyTorch. No Keras. Every weight, gradient and activation displayed in the
> UI is computed by `ann.py` at run time.

---

## 1. Problem Statement

Given six routinely-measured climate variables for a day in **May** at a **coastal station
(Mumbai)**, decide whether **heat-wave conditions are likely** for that day.

This is a supervised **binary classification** problem:

```
Climate inputs  →  ANN  →  learned weights  →  heat-wave score (0…1)  →  HEAT WAVE / NO HEAT WAVE
```

## 2. Objective

1. Implement a multilayer feed-forward ANN using only NumPy.
2. Make **forward propagation → error → backpropagation → weight update → learning**
   visible, step by step, with real numbers.
3. Apply it to a genuine, clearly-defined heat-wave classification task.
4. Keep everything small enough for a student to explain completely in a viva.

## 3. Technology Stack

| Purpose | Package |
|---|---|
| UI | `streamlit` |
| Mathematics / ANN | `numpy` |
| Data handling | `pandas` |
| Charts | `plotly` |

That is the complete list. No databases, no APIs, no authentication, no deep-learning framework.

---

## 4. ANN Architecture

```
   INPUT LAYER            HIDDEN LAYER             OUTPUT LAYER
   (6 neurons)            (8 neurons, sigmoid)     (1 neuron, sigmoid)

   Tmax      ─┐
   Tmin      ─┤
   Humidity  ─┼──────►   H1 … H8   ──────────►     heat-wave score
   Wind      ─┤                                    (0.0 … 1.0)
   Rainfall  ─┤
   Cloud     ─┘
```

| Item | Value |
|---|---|
| Layers | 6 → 8 → 1 (one hidden layer) |
| Activation | sigmoid (hidden **and** output) |
| Loss | binary cross-entropy |
| Optimiser | full-batch gradient descent |
| Learning rate η | 0.8 |
| Epochs | 2500 |
| Weight init | Xavier/Glorot uniform, biases = 0 |
| Overfitting control | keep the weights from the best validation epoch (early stopping) |
| Random seed | 42 (results are reproducible) |

## 5. Input Features

| # | Column | Meaning | Unit |
|---|---|---|---|
| 1 | `max_temp` | Maximum temperature | °C |
| 2 | `min_temp` | Minimum temperature | °C |
| 3 | `humidity` | Relative humidity | % |
| 4 | `wind_speed` | Wind speed | km/h |
| 5 | `rainfall` | Rainfall | mm |
| 6 | `cloud_cover` | Cloud cover | % |

**Target:** `heatwave_label` — `1` = heat wave, `0` = no heat wave.

---

## 6. Mathematical Equations

**Weighted sum (a neuron)**

```
net_j = Σ (x_i · w_ij) + b_j
```

**Activation — sigmoid**

```
σ(z) = 1 / (1 + e^(−z))          σ'(z) = σ(z) · (1 − σ(z))
```

**Forward propagation**

```
h_j = σ(net_j)                              (hidden layer)
ŷ   = σ( Σ (h_j · w_j) + b )                (output neuron)
```

**Error — binary cross-entropy**

```
E = −(1/m) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]
```

**Backpropagation** — with a sigmoid output *and* cross-entropy loss, the output derivative
simplifies beautifully:

```
δ_out    = ∂E/∂z  = ŷ − y
∂E/∂w_j  = δ_out · h_j
δ_hid_j  = δ_out · w_j · h_j · (1 − h_j)
∂E/∂w_ij = δ_hid_j · x_i
```

**Weight update — gradient descent**

```
w_new = w_old − η · (∂E/∂w)
b_new = b_old − η · (∂E/∂b)
```

**Normalisation — min–max scaling**

```
x_scaled = (x − x_min) / (x_max − x_min)
```

`x_min` / `x_max` come from the **training rows only**, and the same stored values are reused
for validation, test, and every live prediction the user makes.

---

## 7. The Heat-Wave Label — *read this carefully*

The label is **not random** and **not an arbitrary single threshold**. It is produced by a
documented rule in `heatwave.py`.

### Official IMD criteria (reference)

The India Meteorological Department declares a heat wave for a **coastal station** when

* maximum temperature ≥ **37 °C**, **and**
* departure from the station normal ≥ **4.5 °C**

(departure 4.5–6.4 °C → *Heat Wave*; > 6.4 °C → *Severe Heat Wave*). For plains, IMD also
declares a heat wave when the actual maximum temperature reaches 45 °C.

### This project's classification rule (what the dataset is actually labelled with)

| | Condition | Source |
|---|---|---|
| **A** | `max_temp ≥ 37 °C` **and** `max_temp − May_normal ≥ 4.5 °C` | IMD coastal criteria |
| **B** | `max_temp ≥ 41 °C` | project value for a coastal station |
| **C** | `heat_index(max_temp, humidity) ≥ 46 °C` **and** `max_temp ≥ 35 °C` | **project rule — NOT IMD** |

```
heatwave_label = 1  if  A or B or C   else   0
```

`heat_index` is the standard **NOAA / Rothfusz apparent-temperature** regression.

> **Condition C is this project's own addition.** It is not part of any official meteorological
> definition and must never be presented as one. It is included because heat stress in Mumbai
> during May is driven by humidity as much as by dry-bulb temperature — and because it makes
> the decision boundary *curved*, which is precisely what a single threshold cannot represent
> but a small ANN can learn.

The "May normal" used in Condition A is the **mean of the loaded dataset**, not an IMD-published
30-year climatological normal.

---

## 8. Dataset

**Location:** `data/mumbai_may_weather.csv`

```csv
date,max_temp,min_temp,humidity,wind_speed,rainfall,cloud_cover,heatwave_label
2010-05-01,33.0,28.9,69.9,9.7,0.0,22.8,0
```

| Column | Type | Unit |
|---|---|---|
| `date` | `YYYY-MM-DD` (May only) | — |
| `max_temp` | float | °C |
| `min_temp` | float | °C |
| `humidity` | float | % |
| `wind_speed` | float | km/h |
| `rainfall` | float | mm |
| `cloud_cover` | float | % |
| `heatwave_label` | 0 or 1 | — |

### ⚠️ The bundled CSV is DEMO DATA

If `data/mumbai_may_weather.csv` is missing, the app generates a **synthetic demo dataset**
(465 rows, 15 Mays) via `data_utils.make_demo_dataframe()` and writes a marker file
`data/DEMO_DATA.marker`. While that marker exists, the app shows a prominent **DEMO DATA**
banner on every page.

**This synthetic data is not a record of real observations.** It exists only so the application
can be run and tested. It is *not* "30 years of real weather".

**To use a real dataset:**

1. Replace `data/mumbai_may_weather.csv` with genuine May observations for Mumbai
   (same columns, same units).
2. Delete `data/DEMO_DATA.marker`.
3. Restart the app. **Nothing in the ANN implementation has to change.**

If your real data has no `heatwave_label` column, generate it with the documented rule:

```python
import pandas as pd, heatwave as hw
df = pd.read_csv("data/mumbai_may_weather.csv")
df = hw.add_heatwave_label(df)          # applies conditions A / B / C
df.to_csv("data/mumbai_may_weather.csv", index=False)
```

---

## 9. Training Process

1. **Split** the records into train / validation / test (70 / 15 / 15), **stratified** by label
   so each part keeps the same heat-wave ratio.
2. **Fit the min–max scaler on the training rows only**, then apply it everywhere.
   (Fitting on all data would leak test-set information into the model.)
3. **Initialise** weights to small random values (Xavier range); biases to zero.
4. For each of 2500 epochs: forward pass over the training rows → binary cross-entropy →
   backpropagation → gradient-descent update of every weight and bias.
5. **Track validation loss** and restore the weights from the best epoch (early stopping).
6. **Evaluate once** on the untouched test set: accuracy, precision, recall, F1, confusion matrix.

The network is trained **once per session** (`@st.cache_resource`) — moving a slider or changing
page never retrains it.

---

## 10. How to Run

```bash
cd heatguard

# 1. create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the four dependencies
pip install -r requirements.txt

# 3. run
streamlit run app.py
```

The app opens at **http://localhost:8501**.
On first launch it creates the demo CSV automatically if none is present.

The map basemap fetches tiles over the internet (no API key, no account). Everything else —
training, prediction, the Neural Laboratory — works completely offline.

> A working `.venv/` is already present in this folder (~440 MB, git-ignored). Delete it and
> repeat the steps above if you would rather build your own.

---

## 11. File Structure

```
heatguard/
├── app.py            Streamlit UI: navbar + the four pages + the map
├── ann.py            The ANN — sigmoid, forward, BCE, backprop, gradient descent, metrics
├── data_utils.py     Loading, validation, demo-data generation, min–max scaling, splitting
├── heatwave.py       Heat-wave domain rules: heat index, IMD criteria, project rule, risk bands
├── config.py         All tunable settings: study area, map style, risk thresholds, η, epochs
├── requirements.txt  streamlit, numpy, pandas, plotly
├── README.md         this file
└── data/
    ├── mumbai_may_weather.csv   the dataset
    └── DEMO_DATA.marker         present only while the CSV is synthetic demo data
```

| File | Responsibility |
|---|---|
| `heatwave.py` | Meteorology only. Knows nothing about neural networks. |
| `data_utils.py` | Numbers in, clean scaled arrays out. Knows nothing about neural networks. |
| `ann.py` | The neural network. Knows nothing about weather or Streamlit. |
| `config.py` | Every knob in one place: study-area coordinates, map basemap, **project risk display thresholds**, learning rate, epochs, seed, theme colours. No logic. |
| `app.py` | Presentation only. Contains no model mathematics of its own. |

---

## 12. The Four Pages

| Page | What it does |
|---|---|
| 🔥 **Heat Wave** | **Map of the Mumbai study area** beside the ANN risk gauge, six climate sliders, condition cards, *Why this prediction?* walk-through, baseline-vs-scenario What-If simulation, recent May chart, model status. |
| 🧠 **ANN Simulation** | The *Neural Laboratory*: the 6→8→1 topology, real weights per neuron, **Run Forward Pass** (animated, with the full arithmetic), **Explain one neuron** (input × weight = contribution, for any H1–H8 or the output neuron), **Learn One Step**, **Run 50 Steps**, **Train 10 Epochs**, **Reset**, plus an initial-vs-current weight comparison and loss-vs-epoch curves. |
| 📊 **Model** | Data explorer, label rule, split sizes, loss & accuracy curves, test-set accuracy / precision / recall / F1, confusion matrix — plus an honest reading of what those numbers do and do not mean. |
| ℹ️ **Methodology** | A **project flow diagram**, every equation, the preprocessing, the training procedure and the limitations. Built for viva preparation. |

A **Presentation** toggle sits at the right of the navbar. Switched on, the Heat Wave page
hides the working controls and shows only the demonstration essentials: map, heat-wave score,
climate conditions, ANN architecture, key test metrics and model status.

---

## 13. The Map — what it does and does *not* claim

The Heat Wave page shows a real geographic map (Plotly + MapLibre over a **token-free**
CARTO / OpenStreetMap basemap — no API key, no account, no credentials).

**What is drawn:** one marker at the study-area coordinates from `config.py`.

* marker **colour** = the risk band of the live ANN score
* marker **size** ∝ the live ANN score
* the hover popup shows the study-area name, the ANN heat-wave score, the status, and the
  current temperature / humidity / wind / rainfall

Every one of those values is read from the current model state. Nothing is hard-coded, and
pressing **Predict Heat Wave** updates the marker, the gauge, the verdict and the probability
card together (they all read the same `st.session_state.prediction`).

### Academic honesty about the map

> **One location is modelled, so the marker is labelled "MODEL STUDY AREA".**

This project does **not** produce a spatially resolved city-wide forecast. It therefore:

- draws **no** heat-wave boundary or heat polygon over Mumbai neighbourhoods,
- makes **no** per-suburb claims such as "Andheri = 90%, Colaba = 75%, Thane = 82%",
- says plainly, under the map, that the dataset contains no station-level coordinates.

**Station mode (automatic, never fabricated).** If — and only if — the CSV genuinely contains
`station`, `latitude` and `longitude` columns that parse to real coordinates, the map switches
by itself to one marker per station, each showing that station's own ANN prediction for its
most recent record together with its observed label. With the single-location dataset those
columns are absent, so the app stays in study-area mode and says so. No station location or
observation is ever invented.

**Offline note:** the basemap tiles are fetched over the internet. Without a connection the
markers still render on a blank background (`MAP_STYLE_FALLBACK` in `config.py`).

### Risk display thresholds

```python
# config.py
RISK_LOW_MAX      = 0.40     # score  < 0.40        -> LOW
RISK_MODERATE_MAX = 0.70     # 0.40 <= score < 0.70 -> MODERATE
                             # score >= 0.70        -> HIGH
DECISION_THRESHOLD = 0.50    # HEAT WAVE / NO HEAT WAVE
```

These are **project risk display thresholds** — a presentation choice for this demonstration.
They are **not** official meteorological thresholds, and the app says so on screen. Change them
in `config.py` and every label, colour and gauge band in the application follows.


---

## 14. Assumptions

1. Each record is one **May** day at a single coastal station (Mumbai).
2. All six inputs are daily aggregates (daily max, daily min, daily-mean humidity, etc.).
3. The "May normal" for Condition A is the mean `max_temp` of the loaded dataset.
4. The decision threshold is **0.50**. It has not been tuned.
5. The split is **random and stratified**, not chronological.
6. `heatwave_label` in a supplied CSV is trusted as-is.
7. The study-area coordinates in `config.py` represent the location the observations came
   from. Change them if your data is from a different station.
8. The risk bands are display categories, not calibrated probability bands.

## 15. Limitations — state these honestly in the viva

- **The bundled dataset is synthetic demo data.** No conclusion about real Mumbai weather can be
  drawn from it.
- **The label is a deterministic function of two of the six inputs** (max temperature and
  humidity, via the rule above). The ANN can therefore learn it very well. A high test score
  demonstrates that *the learning algorithm works* — **not** that the system can forecast real
  heat waves.
- The sigmoid output is **not a calibrated probability**. No calibration (Platt scaling,
  isotonic regression) has been performed, so the UI always calls it an *estimated score*.
- The test set contains only a handful of heat-wave rows, so precision and recall move in large
  steps and are statistically fragile.
- The model classifies a day from **that same day's** observations. It is a *diagnostic*
  classifier, not a lead-time forecast.
- No formal feature-attribution method (SHAP, permutation importance) is implemented. The
  *"Why this prediction?"* panel shows the actual arithmetic of the forward pass and an exact
  decomposition of the output sum into **hidden-neuron** terms — it makes no claim about how
  much each *input feature* mattered.
- **The map marks one study area, not a spatial forecast.** See section 13.

## 16. What still requires a real dataset

| Item | Status |
|---|---|
| Real historical May observations for Mumbai | **required** — replace the demo CSV |
| An IMD-published 30-year May normal for Condition A | **required** — currently the dataset mean |
| Meaningful accuracy / precision / recall figures | blocked until real data is used |
| Any claim about real-world forecasting skill | blocked until real data is used |
| Probability calibration | not implemented in this project |
| Station-level coordinates (`station`, `latitude`, `longitude`) | optional — add them and the map switches to per-station markers automatically |
| Spatially resolved (multi-station) observations | **required** before any per-neighbourhood claim could ever be made |

---

## 17. Viva Preparation

### The project story in ten steps

1. Historical May climate data is collected.
2. It is validated, cleaned and **min–max normalised**.
3. The **ANN** learns patterns linking the six inputs to heat-wave days.
4. **Weights are adjusted** by backpropagation and gradient descent.
5. The model is **tested** on rows it never saw.
6. The user enters **new climate conditions**.
7. The ANN computes a **heat-wave score**.
8. The user sees the prediction with its risk level.
9. The user runs a **what-if experiment** by changing one variable.
10. The user **inspects the neural computation** neuron by neuron.

### Expected viva questions

**What is a weight?**
A number on the connection between two neurons. It scales how strongly one neuron's output
influences the next neuron's input. Learning *is* the adjustment of these numbers.

**Why do we initialise weights randomly, and why small?**
Randomly, to **break symmetry** — if every weight started at 0, all eight hidden neurons would
compute the same thing and receive the same gradient forever, so they could never specialise.
Small, because a large `net` pushes sigmoid into its flat region where the derivative
`σ(1−σ) ≈ 0`, and learning stalls (**vanishing gradient**).

**What is bias?**
An extra input, always 1, with its own trainable weight `b`. It shifts the activation function
left or right so the neuron can fire even when all inputs are zero. Without it, every decision
boundary would be forced through the origin.

**Why do we normalise data?**
Rainfall spans 0–115 mm while temperature spans 27–44 °C. Fed in raw, the large-valued input
dominates the weighted sum, sigmoid saturates and gradient descent crawls. Min–max scaling puts
every input into roughly [0, 1] so all weights start on equal footing. The scaler is fitted on
**training data only** to avoid leaking test information.

**What is sigmoid and why use it?**
`σ(z) = 1/(1+e^(−z))`. It squashes any real number into (0, 1) — perfect for a score — and it is
smooth, so it has a derivative everywhere, which backpropagation requires. Its derivative has the
convenient form `σ(z)(1−σ(z))`.

**What is forward propagation?**
Pushing the inputs through the network to get a prediction: multiply inputs by weights, add the
bias, apply sigmoid, repeat for the next layer.

**What is backpropagation?**
Applying the **chain rule** backwards through the network to find `∂E/∂w` for every weight — i.e.
how much each weight contributed to the error. The error signal is computed at the output
(`δ_out = ŷ − y`), pushed back through `W2`, multiplied by the hidden layer's sigmoid derivative,
and then used to compute the input-layer gradients.

**How are weights updated?**
`w_new = w_old − η · ∂E/∂w`. We step in the direction that reduces the error. `η`, the learning
rate, is the step size: too small and learning crawls, too large and the updates overshoot and
oscillate.

**Why an ANN instead of a simple threshold rule?**
Because the target is **not** a single threshold. Condition C makes the boundary between heat-wave
and normal days **curved** in (temperature, humidity) space — at 35 °C you need ~61 % humidity to
qualify, at 38 °C only ~42 %. A single `if temp > X` rule cannot express that; a network with a
non-linear hidden layer can learn it from examples. More generally, an ANN learns the relationship
from data rather than having it hand-coded, so it can absorb a real, noisier definition later
without the code changing.

**What is the target variable?**
`heatwave_label` ∈ {0, 1}, produced by the documented project rule in `heatwave.py` (Section 7).

**What is overfitting, and what did you do about it?**
When the network memorises the training rows — including their noise — and then performs badly on
unseen data. Symptom: training loss keeps falling while validation loss starts rising. Defences
used here: a small network (only 8 hidden neurons), a separate validation split, and restoring the
weights from the epoch with the lowest validation loss (early stopping).

**Why binary cross-entropy instead of MSE?**
For a 0/1 target with a sigmoid output, BCE punishes a confident wrong answer far more than MSE
does, and its derivative combines with the sigmoid derivative to give the clean `δ = ŷ − y`. MSE
paired with sigmoid produces an extra `σ'(z)` factor that shrinks toward zero exactly when the
network is most wrong — which slows learning.

**Why 8 hidden neurons?**
Enough capacity to bend the decision boundary, few enough to overfit slowly on a small dataset and
to be displayed and explained completely on one screen.

---

## 18. Exact Demonstration Sequence (for the viva)

Run `streamlit run app.py` and follow this in order — it is one continuous story.

| # | Do this | Say this |
|---|---|---|
| 1 | Open the **🔥 Heat Wave** page | "This is the command centre. Mumbai, May." |
| 2 | Point at the **map** | "One location is modelled, so this marker is the *model study area* — I'm not claiming a per-neighbourhood forecast." |
| 3 | Point at the six **sliders** | "These are the six climate inputs the network sees." |
| 4 | Click **⚡ Predict Heat Wave** | "That runs one forward pass through the trained ANN." |
| 5 | Read the **ANN estimated probability** | "This is the sigmoid output — a score, not a calibrated probability." |
| 6 | Point at the **gauge and the map marker** | "Both are driven by the same number: colour is the risk band, marker size is proportional to the score." |
| 7 | Open **🧠 ANN Simulation** | "Now the network itself." |
| 8 | Point at the **diagram** | "Six inputs, eight hidden neurons, one output. Orange edges are positive weights, blue are negative; thickness is magnitude." |
| 9 | Click **▶ Run Forward Pass** | "Watch the signal flow: inputs, then the weighted sums, then the hidden activations, then the output." |
| 10 | Scroll to the **worked example** | "Here is every term: x times w, summed, plus the bias, through sigmoid." |
| 11 | Change **Select neuron** to H3 (or any) | "I can open any neuron." |
| 12 | Point at its **weight table** | "These are read straight out of `W1[:, 2]` — not typed in by me." |
| 13 | Click **▶ Learn One Step** | "One real training update on one example." |
| 14 | Point at target / prediction / error / ΔW | "Target, prediction, error, and then old weight, gradient, ΔW = −η·gradient, new weight." |
| 15 | Click **⏱ Train 10 Epochs** | "Same three operations over the whole training set, ten times." |
| 16 | Point at **loss vs epoch** and **initial vs current weights** | "The error falls and you can see exactly how far each weight has moved from where it started." |
| 17 | Return to **🔥 Heat Wave** | "Back to the application." |
| 18 | Scroll to **What-If Simulation** | "Same trained network, no retraining." |
| 19 | Drag **Tmax** or **Humidity** | "I am changing one climate variable." |
| 20 | Point at **Baseline vs Scenario** | "And this is how the ANN's score responds." |

Optional finishers: **📊 Model** for the training curves, confusion matrix and the honest
reading of the metrics; **ℹ️ Methodology** for the flow diagram and every equation; the
**Presentation** toggle for a clean screen.

---

## 19. Viva Explanation Script

Roughly two minutes, said in your own words:

> "The problem is to decide whether a given May day in Mumbai has heat-wave conditions, from
> six climate variables: maximum and minimum temperature, humidity, wind speed, rainfall and
> cloud cover.
>
> The label isn't arbitrary. Conditions A and B follow IMD's coastal criteria — 37 °C with a
> 4.5 °C departure from normal, or an absolute extreme. Condition C is my own project rule: I
> use the NOAA heat-index formula, because in Mumbai humidity drives heat stress as much as
> dry-bulb temperature. I keep the project rule and the official definition clearly separate.
>
> That third condition is also *why an ANN*: it makes the boundary curved. At 35 °C you need
> about 61 % humidity to qualify; at 38 °C only about 42 %. A single `if temperature > X` rule
> cannot express that. A network with a non-linear hidden layer learns it from examples.
>
> The network is 6 → 8 → 1, written from scratch in NumPy — no TensorFlow, no PyTorch. Inputs
> are min–max scaled, with the min and max taken from the training rows only so nothing leaks
> from the test set. Each hidden neuron computes net = Σ x·w + b and passes it through
> sigmoid. The output neuron does the same over the eight hidden activations.
>
> Training is gradient descent on binary cross-entropy. Because the output is sigmoid and the
> loss is cross-entropy, the output derivative simplifies to just ŷ − y. That error is pushed
> back through W2, multiplied by the hidden sigmoid derivative h(1−h), and every weight is
> updated as w_new = w_old − η ∂E/∂w. I keep the weights from the epoch with the lowest
> validation loss, which is my defence against overfitting.
>
> The ANN Simulation page shows all of this with real numbers — I can run a single forward
> pass, open any neuron, and perform one learning step and watch the weights change.
>
> Two things I want to be honest about. First, the bundled dataset is synthetic demo data, so
> the accuracy figures say nothing about real-world skill. Second, because the label is a
> deterministic function of temperature and humidity, a high test score means the network
> learned the rule — not that it can forecast real heat waves. With genuine IMD observations,
> the same code runs unchanged."

---

## 20. Replacing the Demo Data with a Real 30-Year May Dataset

Nothing in the ANN changes. In order:

1. **Get the observations.** Daily May records for Mumbai (Santacruz and/or Colaba) with
   `max_temp`, `min_temp`, `humidity`, `wind_speed`, `rainfall`, `cloud_cover`.
   Sources to consider: IMD, the NOAA GHCN-Daily archive, or an ERA5 reanalysis extract.
2. **Match the CSV contract** exactly — column names, units, `date` as `YYYY-MM-DD`, May only.
3. **Generate the label** with the documented rule (Section 8), or supply your own
   `heatwave_label` column if you have official heat-wave declarations.
4. **Replace** `data/mumbai_may_weather.csv`.
5. **Delete** `data/DEMO_DATA.marker` — the DEMO banner disappears and the Model page's
   "the data is synthetic" caveat is dropped automatically.
6. **Restart** the app. The ANN retrains on first load.

Then update these specific claims, which the demo data cannot support:

| Claim | What to change |
|---|---|
| "May normal Tmax" | Replace the dataset mean with the **IMD-published 30-year normal** — set it explicitly instead of calling `may_normal_max_temp()` |
| Accuracy / precision / recall / F1 | They become meaningful; keep the caveat about the deterministic label until you use official heat-wave declarations |
| The split | Consider switching `split_dataset()` to **chronological** (train on earlier years, test on later years) — more honest for a forecasting claim |
| `PROJECT_ABSOLUTE_TMAX = 41.0` | Re-check against the real record maxima for the station |
| `PROJECT_HEAT_INDEX_THRESHOLD = 46.0` | Recalibrate so the positive class is a sensible fraction of real days |
| Map study area | Set `config.STUDY_AREA` to the actual station's coordinates |
| Per-location claims | Only possible after adding real `station` / `latitude` / `longitude` columns |

---

*HEATGUARD AI — Soft Computing academic project. ANN implemented from scratch in NumPy.*
