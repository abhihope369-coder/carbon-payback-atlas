# Carbon Payback Atlas: 3-minute demo script

Rule from the judging table: show something working within 90 seconds, walk through it as the user, and be candid about what works and what doesn't. Rehearse once out loud and time it.

**Before you start:** open the app on the Overview page, focus region Canada, default settings. Close every other tab.

---

### 0:00 to 0:20 · The problem (Overview)

> "A city has $10 million to replace petrol fleet vehicles with EVs across many regions. Every EV starts in carbon debt from its battery. Today, people judge that with a calculator that assumes the grid never changes. That gives one number, with no uncertainty and no budget decision, and nobody ever checks it against reality."

Point at the two cards: **"A basic carbon calculator says"** next to **"Carbon Payback Atlas says"**.

### 0:20 to 0:40 · Start the guided demo

Click **Start guided demo**. Click **Next** on the banner to move through the pages.

### 0:40 to 1:10 · Real data and the AI (Global Atlas, then AI Forecast)

> "This is real Ember data for 213 countries. Our Random Forest forecasts each grid 10 years ahead with an 80% uncertainty band. We trained on data up to 2019 and tested on 2020 onward, which the model never saw. It has 36% lower error than assuming the grid stays the same. At one year ahead it's slightly worse than that baseline, and we show that too."

### 1:10 to 1:40 · Carbon payback (Payback Engine)

> "Here the EV starts at minus 3.6 tonnes, its battery debt, and pays it back in about 1.3 years in Canada."

Scroll to the tornado chart:

> "This shows which assumption matters most. In Canada it's the petrol car's fuel use, not the grid."

Open **Show the math**:

> "Every number can be verified by hand."

### 1:40 to 2:30 · The decision (Deployment Optimizer)

> "An integer optimizer splits the budget across regions and years."

Point at the heatmap and the "Why this allocation" text. Then type a **Manager minimum** of 30 for Poland:

> "A manager can override the AI, for fairness or local politics. We show exactly what that override costs in tonnes. The human decides; the cost is visible."

Point at the three strategies:

> "We also offer a robust plan that protects the worst case."

### 2:30 to 2:55 · The honest result (study section)

> "We tested our own idea. We made 900 plans as of 2010 to 2015 and scored them on what the grids actually did afterwards. The surprise: forecasting rarely changes *where* to invest. Even perfect foresight only adds 0.4%. What it does fix is *how much* you promise. The simple method under-promised by about 2%, while our forecast estimates were nearly unbiased. Budget holders get a number they can defend."

### 2:55 to 3:00 · Close (Strategy Report)

> "One click gives a decision-ready report and CSVs. The right climate investment, in the right place, at the right time."

---

**Likely judge questions**
- *Why a Random Forest and not an LLM?* The job is a numeric forecast that has to be validated against held-out years and fed straight into the optimizer. A tabular ML model can be tested that way; a chatbot can't.
- *Is the data real?* Grid data is real (Ember via OWID, CC BY 4.0). Fleet size, costs and budget are labelled HYPOTHETICAL and editable. You can upload a real fleet as a CSV.
- *What would you do next?* Provincial and state grids (Ontario vs Alberta differ hugely), hourly charging profiles, and heat pumps using the same engines.
