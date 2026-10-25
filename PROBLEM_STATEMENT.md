# Problem statement: MobilityDemand

## The problem

Bike-sharing rental volume changes throughout the day. A quiet hour and a
commute hour can require very different amounts of service. An operations
analyst needs a reasonable estimate of the next hour's rentals before deciding
how much service capacity to plan.

This project asks: **Can a small machine-learning model predict hourly rentals
better than copying the same hour from last week, and what happens when an
analyst adds different buffers to those predictions?**

## Who would use it

A transport analyst or operations planner could use the approach to understand
rental patterns and compare forecast errors. This repository is an educational,
offline demonstration; it is not connected to a real operator's dispatch system.

## Data and prediction

The public UCI Bike Sharing dataset contains system-wide rental counts from
Capital Bikeshare in Washington, D.C., during 2011-2012. The included hourly
CSV contains 17,379 observations. One row represents one observed hour; `cnt`
is the number of recorded rentals during it.

The task is regression: estimate a nonnegative rental count. Inputs are six
calendar fields and the actual recorded rental counts at the same hour one day
and one week earlier. Both historical counts must already be available.
`casual` and `registered` are excluded because they add up to the target.
Observed weather is excluded because archived weather forecasts are unavailable.

## What the project does

1. Verifies and validates the original public file.
2. Summarizes training-period patterns with SQL, pandas, and a chart.
3. Looks up historical counts by exact clock time and excludes missing-history rows.
4. Fits a fixed gradient boosting model and compares it with a last-week rule.
5. Selects using a later validation period and reports errors on a separate test period.
6. Compares hypothetical rental-capacity buffers under explicit cost assumptions.
7. Reviews monthly/commute errors and reloads the saved model for an example.

The forecast is evaluated one hour at a time as past observations become
available. The project does not issue all future hours at once without new
history. Model and policy choices use validation data, not final test results.

## How success is judged

Forecast selection uses validation mean absolute error (MAE). The frozen choice
is evaluated with test MAE, RMSE, WAPE, and signed bias. Results must be
reproducible and include the number of eligible hours, rather than promising
a particular improvement in advance.

The capacity exercise compares shortages and excess rental slots for buffers
of 5%, 15%, and 30%. Its validation-selected policy is evaluated separately on
test. A rental slot represents hypothetical capacity for one trip in an hour;
it does not represent one physical bicycle. Cost assumptions are teaching inputs,
not operator financial records.

## What this does not solve

The source contains recorded rentals rather than unmet customer demand. It has
no station inventory, trip durations, bike transfers, or current weather forecasts.
It cannot determine physical fleet size, station rebalancing, measured stockouts,
financial savings, or causal service improvements. Missing clock hours remain
unknown. Results cover one historical city/system and one final quarter; future
years or other systems require new data and evaluation.

The deliverable is an understandable local workflow with reproducible outputs,
not a deployed API or production monitoring service. See [README.md](README.md)
for setup, execution order, measured results, and dataset attribution.
