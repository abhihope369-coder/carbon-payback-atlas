# Carbon Payback Atlas

**Tagline:** The right climate investment. In the right place. At the right time.

## Inspiration
Fleet managers who want to switch to EVs face a real dilemma. Every EV starts in carbon debt from its battery, and how fast it pays that back depends on how clean the local grid is and will become. Today's tools are calculators that freeze the grid at today's value, give one number, and never check themselves against reality.

## What it does
- **Global Atlas:** real lifecycle carbon intensity for 213 countries (Ember via Our World in Data).
- **Grid Future AI:** a Random Forest that forecasts each grid 1 to 10 years ahead with an 80% uncertainty band.
- **Carbon Payback Engine:** compares a petrol car (well-to-wheel) with an EV (vehicle plus battery manufacturing, plus the forecast grid) year by year. It finds the payback year, ranks the assumptions with a tornado chart, and has a "show the math" trace for every number.
- **Deployment Optimizer:** a mixed-integer program that allocates a budget across regions and years. It compares three strategies: current-data, forecast-aware, and robust (max-min over four grid futures). Managers can override the AI by setting a minimum number of EVs per region, and the app shows the price of each override in tonnes. You can also upload your own fleet as a CSV.
- **Strategy Report:** a Markdown report plus CSV exports.

## How we built it
Python, pandas, scikit-learn, SciPy (HiGHS MILP), Plotly and Streamlit.
- The model predicts the log ratio of future to current intensity, using pooled direct multi-horizon training.
- Splits are by target year: select the model on 2015-2019, test on 2020 onward, so the test years are never seen in training.
- The colour palette was validated for colour-vision deficiency.
- 11 unit tests, including one that checks the optimizer against brute-force search.

## Results (computed, not hand-picked)
- **Forecast error on unseen 2020+ data:** MAE 32.4 g CO2e/kWh, versus 50.5 for "the grid stays the same" (36% lower).
- **Decision study:** 900 plans made as of 2010 to 2015, each scored on the grids that actually followed. A separate model was trained for each origin year using only data up to that year.
  - Forecasting changed the plan in 31% of cases.
  - The average realized effect was +0.03%.
  - Even perfect foresight was only +0.43% better than the simple plan.
  - However, the simple method under-promised savings by 1.9% on average, while the forecast-based estimate was off by only +0.3%. It was closer in 61% of cases.

## What we learned
We expected forecasting to change *where* to invest. History says it mostly doesn't, because today's gap between grids dominates. Its real value is the *size* of the promise: estimates a budget holder can defend. We report that instead of a flattering result.

## Challenges
- Avoiding data leakage in time-series validation.
- Not double-counting battery emissions.
- Keeping well-to-wheel petrol consistent with lifecycle grid data.
- Making the comparison between strategies fair. Under its own forecast, the forecast-aware plan always wins, so we built the historical study to test it properly.

## What's next
Provincial and state grids, hourly charging, heat pumps and industrial electrification on the same engines, and pilots with real municipal fleet data.

## Data and credits
Ember and Our World in Data (CC BY 4.0), UK DESNZ 2025 conversion factors, ICCT (2021) life-cycle study, Argonne GREET, Plotly world topology.
