# Carbon Payback Atlas

**The right climate investment. In the right place. At the right time.**

An AI-powered decision-support tool for electric-vehicle fleet transitions. It connects three engines into one pipeline:

```
Real grid data (Ember via OWID)
  -> Grid Future AI: Random Forest forecast of electricity carbon intensity, 1-10 years ahead
  -> Carbon Payback Engine: year-by-year lifecycle emissions, petrol vs EV
  -> Deployment Optimizer: mixed-integer program allocating a budget across regions and years
  -> Explainable recommendation + current-data vs forecast-aware comparison + historical backtest
```

Every number in the app comes from the dataset, the trained model, or an editable assumption that is labelled as such. Nothing is hardcoded.

---

## Key results (from the current build)

| | Value |
|---|---|
| Countries in the forecasting universe | 109 (104 with complete features at the forecast origin) |
| Selected model (lowest 2015-2019 validation MAE) | Random Forest |
| Held-out test, target years 2020+ (5,911 country-horizon pairs) | **MAE 32.4 g CO2e/kWh**, RMSE 48.4 |
| Persistence baseline on the same test | MAE 50.5, RMSE 76.0 |
| Linear-trend baseline | MAE 218.2 (fails on volatile grids) |
| 80% interval coverage on test (calibrated on 2015-2019) | 74.4% (slightly below nominal) |
| 1-year-ahead error | Model 19.2 vs persistence 18.6: **persistence is slightly better at 1 year**; the model wins from 2 years onward |
| 2014 decision backtest, default scenario | Forecast-aware plan realized 3,141 t vs 3,167 t for the current-data plan (**-0.8%**) |
| Multi-origin study (900 plans, origins 2010-2015) | Plans identical in 69%; forecast better in 116, worse in 159; mean realized effect **+0.03%**; perfect-foresight ceiling only **+0.43%** |
| Savings estimate bias in the same study | Current-data plans under-promised by **-1.9%**; forecast-aware estimates **+0.3%** (closer in 61% of cases) |

**The honest finding:** in this problem, forecasting rarely changes *where* to invest, because today's gap between grids dominates. Even perfect foresight adds less than half a percent. Its measurable value is in *how much* to promise: forecast-based estimates of savings are nearly unbiased, while frozen-grid estimates consistently under-promise. The app reports this as-is.

## What's in the app

- **Guided demo tour:** one click walks a judge through all six steps with Next/Back buttons.
- **Before/after panel:** what a basic calculator says vs what the Atlas says, computed live for the focus region.
- **Three strategies:** current-data, forecast-aware, and robust (max-min over four grid futures), with a regret table.
- **Human-in-the-loop overrides:** a manager minimum per region; the app shows the price of the overrides in tonnes.
- **Budget per tonne:** gross EV spend divided by net tonnes. This is not a net abatement cost.
- **Sensitivity tornado:** each input moved ±20% on its own, plus a "show the math" trace for every number.
- **Fleet CSV upload** with a downloadable template.
- **Historical backtest** from any origin year 2010-2015, plus the 900-scenario study.

`DEMO_SCRIPT.md` is a timed 3-minute pitch; `DEVPOST.md` is a submission write-up.

---

## Quick start

Requires Python 3.11+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The processed data, forecasts and validation metrics are already included, so the app runs immediately. To rebuild everything from scratch:

```bash
python scripts/build_data.py --force-download   # downloads OWID data, cleans, writes data/processed/
python scripts/train_forecast.py                # trains, validates, writes forecasts + metrics (~20 s)
python scripts/backtest_study.py                # per-origin models + 900-scenario decision study (~90 s)
python -m pytest -q tests                       # lifecycle, optimizer, robust and sensitivity tests (11)
```

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a **public** GitHub repository.
2. Sign in at https://share.streamlit.io with GitHub.
3. Click **Create app**, choose the repository, branch `main` and main file `app.py`, then **Deploy**.
4. You get a permanent URL to share with judges.

Notes: `models/grid_forecast_model.joblib` (34 MB) is under GitHub's 100 MB limit. The app itself only reads the CSV outputs, so it starts quickly. The world-map geometry is served locally from `static/` (`enableStaticServing` in `.streamlit/config.toml`), so the map has no CDN dependency.

---

## Project structure

```
app.py                     Streamlit app (7 pages)
src/data_processing.py     download, clean, filter, manifest
src/forecasting.py         features, models, baselines, direct multi-horizon forecasting
src/validation.py          temporal splits, metrics, interval calibration
src/lifecycle.py           petrol / EV lifecycle emissions, carbon payback
src/scenarios.py           grid-intensity paths (observed + forecast + held-flat tail)
src/optimization.py        MILP (SciPy HiGHS), overrides, max-min robust plan, explanations
src/sensitivity.py         one-at-a-time tornado analysis
src/pipeline.py            strategy comparison + 2014 historical decision backtest
src/report.py              Markdown strategy report
src/ui.py                  theme, validated colour palette, components
scripts/                   build_data.py, train_forecast.py, backtest_study.py
tests/test_engines.py      11 unit tests (incl. MILP vs brute force, robust max-min)
data/raw, data/processed   source data and pipeline outputs
models/, outputs/          trained model, validation metrics, test predictions
static/world_110m.json     Plotly world topology (from plotly/plotly.js, MIT)
```

---

## Data

| Field | Value |
|---|---|
| Source | Our World in Data energy dataset, https://github.com/owid/energy-data |
| Underlying source | Ember Yearly Electricity Data (2026 release) |
| Licence | CC BY 4.0 |
| Target variable | `carbon_intensity_elec`: **lifecycle** g CO2e per kWh of electricity *generated* |
| Coverage used | 2000-2025, 213 countries for the Atlas |
| Forecasting universe | 109 countries with >= 10 TWh generation in their latest year and >= 15 years of data |
| Features | intensity lags, 1- and 3-year changes, coal/gas/oil/nuclear/hydro/wind+solar shares, 3-year change in low-carbon share |
| Missing data | rows with missing features are dropped (1,195 of 18,745 supervised rows) and counted; nothing is interpolated or invented |

Access date and all filtering steps are recorded in `data/processed/data_manifest.json`.

## Grid Future AI (forecasting)

- **Target:** `ln(I[t+h] / I[t])` for horizons h = 1-10. A ratio rather than a level, because tree models cannot extrapolate levels.
- **Direct, not recursive:** one pooled model with h as a feature. A recursive model would need future generation mixes, which we don't have.
- **Candidates:** Ridge, Random Forest, Gradient Boosting (fixed hyperparameters, no tuning on test).
- **Splits by target year (no leakage):**
  1. Selection: train <= 2014, validate 2015-2019, then pick by MAE.
  2. Test: train <= 2019, report on >= 2020.
  3. Backtest: the selection-phase model (trained <= 2014) produces the 2014-origin forecasts used by the decision backtest.
  4. Production: retrain on all rows, then forecast from each country's latest year.
- **Uncertainty:** empirical 10th/90th-percentile residuals per horizon give an 80% interval.
- **Beyond 10 years:** intensity is held flat at the last forecast value. This is an explicit assumption, and the UI flags it.

## Carbon Payback Engine (lifecycle)

`E_petrol = D x (L/100km / 100) x kg/L`, `E_EV = D x kWh/km x (1 + grid loss) x G_y`, `S_y = E_petrol - E_EV,y`, `M = M_EV - M_petrol`, `N_t = sum S_y - M`. Payback is the first year with N_t >= 0, linearly interpolated within that year. If payback never happens within the vehicle's life, the app says so.

| Default | Value | Status / source |
|---|---|---|
| Petrol factor | 2.65 kg CO2e/L well-to-wheel | UK DESNZ 2025 conversion factors: 2.069 tailpipe + 0.581 well-to-tank. Well-to-wheel is used because the Ember grid intensity is lifecycle. |
| Battery manufacturing | 60 kg CO2e/kWh | ICCT (2021) global LCA, Europe/US |
| Non-battery manufacturing | 8 t, equal for both | Argonne GREET range 8-12 t; ICCT (2021) finds it similar across powertrains |
| Battery size | 60 kWh | hypothetical |
| Annual distance, lifetime | 15,000 km, 12 years | hypothetical |
| Fuel / electricity use | 8 L/100 km, 0.18 kWh/km at the wall | hypothetical |
| Cost per EV, budget | $50,000, $10M | hypothetical |

**Boundary:** includes vehicle and battery manufacturing, petrol well-to-wheel, and EV charging at lifecycle grid intensity. It excludes maintenance, end-of-life, recycling credits and infrastructure. **Replacement mode:** "buying new anyway" uses M = M_EV - M_petrol, so the battery is counted once. "Early retirement" uses M = M_EV.

## Deployment Optimizer

The decision variable x[r,y] is the number of whole EVs deployed in region r in year y. The model maximizes the sum of N[r,y] x x[r,y], where N is the per-vehicle net savings counted up to the target year. Constraints: budget, fleet available per region, and an optional per-year capacity. An optional annual cost decline creates a real timing trade-off. It is solved as a MILP with `scipy.optimize.milp` (HiGHS). A unit test checks it against brute-force enumeration.

**Strategy A (current-data)** plans with intensity frozen at the latest observed value. **Strategy B (forecast-aware)** plans with the forecast. Both get identical inputs. Scored under its own forecast, B wins by construction, so the app adds a **historical decision backtest**: plan as of 2014 with a model that only saw data up to 2014, then score both plans on the *actual* 2015-2024 grid data, alongside a perfect-foresight upper bound.

## Limitations

- Intensity is generation-based, so electricity imports and exports are ignored.
- National averages only: no provincial or state grids, and no hourly charging profiles.
- Forecasts extrapolate 2000-2025 patterns and cannot anticipate policy shocks. Intervals slightly under-cover (74% vs 80%).
- Results are estimates for decision support, not causal claims.
- PDF export is not built in. The report downloads as Markdown, and the page can be printed to PDF.

## Credits

Ember and Our World in Data (CC BY 4.0); UK DESNZ GHG conversion factors 2025; ICCT (2021) *A global comparison of the life-cycle greenhouse gas emissions of combustion engine and electric passenger cars*; Argonne National Laboratory GREET; Plotly world topology (MIT). Libraries: Streamlit, Plotly, pandas, NumPy, SciPy (HiGHS), scikit-learn.
