# F1 Lap Time Prediction — Tire Degradation Modeling

**Track:** Classical Machine Learning / Regression
**Stack:** Python, pandas, scikit-learn, matplotlib
**Notebook:** `F1_Lap_Time_Prediction.ipynb` (Google Colab)

## Objective

Predict in-race lap times from tire age, and compare a baseline model against one that
accounts for tire degradation, on a single Formula 1 race.

## Dataset

Kaggle: [*Formula 1 World Championship (1950–2020)*](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) by Rohan Rao.
Files used: `races.csv`, `results.csv`, `status.csv`, `lap_times.csv`, `pit_stops.csv`, `drivers.csv`.

## Race selection

The raw dataset has no weather or red-flag column, so the race is chosen by year/round and
should be checked against external race reports (e.g. Wikipedia) for dry conditions and no
red flags. The notebook defaults to the **2019 Spanish Grand Prix**, a dry race with no red
flags or safety car. `YEAR`/`ROUND` are exposed as parameters so any other verified dry,
incident-free race can be substituted. From that race, the **5–10 highest-classified
finishers** (`statusId == Finished`) are kept, to control for weather and track-temperature
variation across the field.

## Data cleaning

Two rules are applied to the raw lap times, in order:

1. **Pit laps removed** — the in-lap (the lap a driver pits on) and the immediately
   following out-lap are dropped, since both are mechanically slower/faster than green-flag
   pace and would distort the tire-age signal.
2. **1.5× median outliers removed** — any remaining lap slower than 1.5× that driver's own
   median lap time in the race is dropped. This catches safety-car laps, lock-ups, and other
   anomalies not tied to a specific pit stop.

The notebook prints the lap count removed by each rule and the total, at the point of cleaning.

## Feature engineering

- **`stint`** — increments by 1 each time a driver passes a pit-stop lap.
- **`tire_age`** — laps completed since the start of the current stint; resets to 0 at every
  pit stop. Computed on the full (pre-cleaning) lap sequence per driver so the counters stay
  continuous, then joined back onto the cleaned laps.
- **`grid`** — starting grid position, merged in from `results.csv` for the baseline model.

## Train/test split — by stint, not randomly

For each driver, the **final stint** is held out as test data; **all earlier stints** form
the training set. A random row-level split would place adjacent laps from the same stint on
both sides of the split, letting a model implicitly interpolate between neighboring laps —
a form of leakage that would understate real-world error, since tire wear and track
evolution make consecutive laps highly correlated.

## Models compared

Two feature sets:

| Feature set | Features |
|---|---|
| Baseline | `grid`, `lap` |
| Tire-age enhanced | `grid`, `lap`, `tire_age` |

Two algorithms, each trained on both feature sets (4 models total):

- **Linear Regression**
- **Random Forest Regressor** (`n_estimators=300`, `max_depth=6`)

Each model is scored on the held-out final-stint test data using **RMSE** and **MAE** (in
seconds), reported in a single comparison table produced by the notebook.

## Visualization

Predicted vs. actual lap time is plotted across one driver's complete final stint (the test
set), using the tire-age-enhanced Random Forest model, to visually check whether the model
captures the upward drift in lap time as tires age — something the baseline model, lacking
`tire_age`, cannot represent.

## How to run

1. Open `F1_Lap_Time_Prediction.ipynb` in Google Colab (File → Upload notebook, or drag it
   into https://colab.research.google.com).
2. Run the setup cell, then either:
   - **Kaggle API**: get a `kaggle.json` token from https://www.kaggle.com/settings and
     upload it when prompted, or
   - **Manual upload**: upload the six CSVs into a `f1_data/` folder in the Colab session
     and set `USE_KAGGLE_API = False`.
3. Run all remaining cells top to bottom. Adjust `YEAR`/`ROUND` in section 3 to change the
   selected race.
4. Review the printed cleaning counts, the model comparison table, and the stint plot
   (also saved as `stint_prediction_plot.png`).

## Results

*(Fill in after running the notebook on your chosen race)*

- Total laps removed during cleaning:
- Best-performing model / feature-set combination (lowest RMSE):
- Does `tire_age` reduce error relative to the baseline for both algorithms?
- Does the visualization show the model tracking the late-stint lap-time increase?

## Deliverables

- `F1_Lap_Time_Prediction.ipynb` — full notebook (data loading → cleaning → features →
  stint split → model comparison → visualization)
- Model comparison table — produced inline in the notebook (section 9)
- Stint visualization plot — produced inline and saved as `stint_prediction_plot.png`
  (section 10)
- `README.md` — this file
