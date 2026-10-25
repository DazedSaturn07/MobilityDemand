# MobilityDemand

Forecast hourly bike-sharing rentals and explore the tradeoff between shortage
and excess rental capacity. A small, local machine-learning project designed
to be read and run in order.

The project uses the public UCI Bike Sharing data, four notebooks, a SQL query,
and a few shared Python helpers. No cloud account, API key, database server,
or web server is required. Read [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md)
for the business question, intended users, and boundaries.

## Run from start to finish

Reference environment: **Python 3.12.10 on Windows**. Use Python 3.12 to match
the verified setup. Direct dependencies are pinned in `requirements.txt`;
transitive dependencies are resolved by pip rather than fully locked.
Installation requires internet; the analysis runs offline afterward.

Download or clone this repository, open a terminal in its root folder, and
run each command separately. Git is needed for the publication check only.

```bash
git clone https://github.com/DazedSaturn07/MobilityDemand.git
cd MobilityDemand
```

**Windows / PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_notebooks.py 01_data.ipynb
.\.venv\Scripts\python.exe run_notebooks.py 02_model.ipynb
.\.venv\Scripts\python.exe run_notebooks.py 03_capacity.ipynb
.\.venv\Scripts\python.exe run_notebooks.py 04_review.ipynb
.\.venv\Scripts\python.exe check_publication.py
```

These commands use the environment's Python directly, so PowerShell activation
is unnecessary. If the Python launcher is unavailable, use `python -m venv
.venv` after checking that `python --version` selects Python 3.12.

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_notebooks.py
.venv/bin/python check_publication.py
```

The runner with no notebook argument executes all four in order. It starts a
fresh kernel for every notebook, stops at the first failure, and writes executed
copies under `output/executed/`. Source notebooks stay clear of outputs. If
execution fails, the executed copy contains the failing cell and traceback.
Local verification was on Windows; macOS/Linux commands have not been executed.

## Learn interactively

To inspect the cells yourself, run:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

In JupyterLab open `notebooks/01_data.ipynb`. Select the kernel belonging to
`.venv`; check `import sys; print(sys.executable)` in a temporary cell if unsure.
Run cells top to bottom with **Shift+Enter**, reading the explanation above each
cell. Finish one notebook before opening the next. Each starts independently
and shares files with later stages.

| Order | Notebook | Purpose | Main results in `output/` |
| --- | --- | --- | --- |
| 1 | `01_data.ipynb` | Verify data, inspect training patterns with SQL/pandas, build exact-time lags, split dates | `data_report.json`, `hourly.db`, `hourly_summary.csv`, `hourly_demand.png`, three split CSVs |
| 2 | `02_model.ipynb` | Compare a last-week rule and gradient boosting; select on validation; evaluate test | `model.joblib`, `metrics.json`, validation/test predictions |
| 3 | `03_capacity.ipynb` | Compare three hypothetical buffers; select on validation assumptions | `capacity_validation.csv`, `capacity_test.csv`, `capacity.json` |
| 4 | `04_review.ipynb` | Inspect monthly/commute errors and reload a target-free example | `monthly_errors.csv`, `error_review.png`, example input/prediction |

Rerunning an earlier notebook removes its known outputs and downstream results
before rebuilding. Rerun every later notebook afterward. Never keep using an
older model's reports after changing preparation or training. Results are local
and reproducible; the public notebooks intentionally start without outputs.

## How the forecast works

Each row is an observed hour's system-wide rental count, `cnt`. The original
CSV has **17,379 rows** from 2011-2012, no missing cells, and **165 missing clock
hours**. Missing clock hours are left unknown rather than treated as zero rentals.

The eight predictors are:

| Input | Meaning |
| --- | --- |
| `yr` | 0 for 2011, 1 for 2012 |
| `mnth` | Month, 1-12 |
| `hr` | Hour, 0-23 |
| `holiday` | Source holiday flag, 0/1 |
| `weekday` | Sunday = 0 through Saturday = 6 |
| `workingday` | 1 when neither weekend nor holiday |
| `lag_24` | Recorded rentals exactly 24 clock hours before the target hour |
| `lag_168` | Recorded rentals exactly 168 clock hours before the target hour |

`casual + registered = cnt`, so both components would reveal the answer and
are excluded. IDs and observed weather are also excluded. The source contains
observed weather, not archived weather forecasts available before each hour.

A timestamp lookup constructs each lag. Shifting 24 rows would be incorrect
across gaps. Rows lacking either historical count are excluded, without future
averages or invented fallback values: **16,950 eligible rows**, **429 excluded**.

| Partition | Dates | Eligible hours | Role |
| --- | --- | ---: | --- |
| Train | 2011-01-08 to 2012-06-30 | 12,641 | Fit the learned model |
| Validation | 2012-07-01 to 2012-09-30 | 2,208 | Select forecast and buffer |
| Test | 2012-10-01 to 2012-12-31 | 2,101 | Report the frozen choices |

This is a **one-hour-ahead historical replay**: previous recorded rentals can
become available as time advances. It is not a forecast for the entire test
quarter made on October 1. Both lags must have been recorded before the target
hour. The model is fitted once on training data and is not refitted on validation.

The baseline copies the same hour last week. The learned candidate is
`HistGradientBoostingRegressor` with fixed settings visible in notebook 02.
It uses 150 boosting iterations, at most 15 leaves per tree, and no internal
random early-stopping split. The lower validation MAE selects the forecast;
a tie keeps the simpler baseline. The test period does not choose the winner.

## Verified results

A complete Windows run on **2026-10-25**, using the pinned direct dependencies,
selected gradient boosting. These measurements replace the older project's
claims and apply only to the eligible historical hours above.

| Metric | Last-week rule | Gradient boosting |
| --- | ---: | ---: |
| Validation MAE | 56.00 | 54.81 |
| Test MAE | 69.47 | 56.64 |
| Test RMSE | 114.77 | 87.31 |
| Test WAPE | 31.42% | 25.62% |
| Test bias | +7.28% | +1.63% |

Test MAE decreased **18.47%** relative to the last-week rule. MAE is average
absolute error in rentals/hour; RMSE penalizes large mistakes more. WAPE is
total absolute error divided by total observed rentals, not percent accuracy.
Positive bias means total overprediction. Exact values are in `output/metrics.json`.

The capacity exercise rounds forecasts up after adding 5%, 15%, or 30%.
Under assumed costs of **0.5 per excess rental slot** and **3 per shortage slot**,
validation selected the 30% buffer. Its test scenario covered 97.89% of observed
rentals under that mathematical proxy. Costs use arbitrary teaching units.
This is not a measured service improvement, financial saving, or physical fleet
recommendation. Notebook 03 shows how changing cost assumptions changes choices.

## Folder map

```text
README.md                    Public overview and complete run instructions
PROBLEM_STATEMENT.md         Public business question and scope
requirements.txt            Pinned direct dependencies
run_notebooks.py            Optional sequential notebook runner
check_publication.py        Check expected Git files and common privacy issues
data/
  hour.csv                  Original attributed public source
  source.json               Citation, license, and fixed SHA-256 fingerprint
notebooks/                  Four ordered learning notebooks
mobility/
  data.py                   Source validation and stale-output cleanup
  features.py               Exact-time lags, date splits, predictor checks
  evaluation.py             Error formulas and capacity exercise
  scoring.py                Reload the selected forecast
sql/hourly_summary.sql      Training-period hourly summary
tests/test_project.py       Regression checks, no prebuilt model required
guides/                     Local learning guides; ignored by Git
output/                     Generated results and executed notebooks; ignored
```

Training, model selection, and reporting remain visible in the notebooks.
Helpers share only reusable rules. No API framework, cloud deployment,
production monitoring, BI application, or separate database installation is
part of the workflow. The local `guides/` folder, when present in your working
copy, contains detailed section guides and interview preparation. These private
learning documents are deliberately excluded from Git; the public notebooks
and this README contain everything required to reproduce the project.

## Checks and publication

The 11 regression checks cover raw validation, exact clock lags, future-target
perturbations, chronology, scoring inputs, saved-baseline loading, hand-calculated
metrics, capacity arithmetic, and cleared source notebooks. Notebook execution
also checks SQL/pandas agreement and saved-model prediction equivalence.

Only `README.md` and `PROBLEM_STATEMENT.md` are publishable Markdown files.
`.gitignore` also excludes results, environments, credentials, private folders,
local databases, model binaries, and editor state. Source attribution remains
public; it is not private project-owner information.

`.gitattributes` preserves the original CSV bytes across Git clones so automatic
line-ending conversion cannot invalidate the source fingerprint.

Run `check_publication.py` before committing. It works before Git initialization
and checks actual tracked files once a repository exists. It rejects unexpected
files, common private-data patterns, and notebook outputs. If you ran the source
notebooks interactively, use Jupyter's **Restart Kernel and Clear Outputs**, save,
and remove temporary diagnostic cells before publishing. Keep executed copies local.

An ignore rule does not remove already tracked files or Git history. Review
`git status --short` and `git diff --cached` before each push. The checker is a
useful guard, not a guarantee against every possible secret or personal detail.
No Git remote or publication is required to run this project.

## Limitations and next questions

This is one historical system and one final test quarter. Missing-history rows
are excluded, so results do not describe every hour. Holidays and unusual events
can break last-week patterns. Calendar-year coding and a tree model do not
provide reliable future-year growth extrapolation. The historical data has
no timezone offsets to resolve daylight-saving ambiguity.

Real fleet planning needs station inventory, trip durations, bike movements,
and a different operational model. A later project could add archived weather
forecasts or evaluate multiple time windows without reusing this test window
for tuning. This repository makes no deployment, live monitoring, causal-impact,
or real-world savings claim.

## Data attribution

Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository.
[Dataset DOI](https://doi.org/10.24432/C5W894).
The [UCI source page](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The included `hour.csv` was verified byte-for-byte against the official ZIP;
`data/source.json` records its fingerprint. The measured CSV row count is used
here because the source page's displayed instance count differs.

Associated paper: Fanaee-T, H. and Gama, J. (2013), *Event labeling combining
ensemble detectors and background knowledge*,
[paper DOI](https://doi.org/10.1007/s13748-013-0040-3).
