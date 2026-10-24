# Cloud Workload Forecasting And Analysis

## Executive Summary

This study constructs and forecasts represented VM population and mean VM CPU utilization for a curated Azure workload trace. The source contains 1,340,000 readings from 10,000 VMs at 134 shared five-minute timestamps. The first 110 timestamps (0 through 32,700 seconds) are used for model development; the final 24 (33,000 through 39,900 seconds) are held out for a fixed-origin two-hour forecast.

The represented population is exactly 10,000 at every step, so the persistence forecast has zero validation and holdout error for `active_vm_count`. For `aggregate_cpu_utilization`, persistence is selected by internal validation (MAE 0.185 percentage points; RMSE 0.236 points over 72 origin-step rows). Its public-holdout MAE is 0.147 points and RMSE is 0.185 points. This is a stable series for this particular fixed sample and window, not evidence that arbitrary VM fleets are equally predictable. The submission file is [`../predictions.csv`](../predictions.csv).

## Problem And Operational Context

The two targets answer different operational questions. `active_vm_count` counts distinct VMs represented by readings. `aggregate_cpu_utilization` is the unweighted arithmetic mean of valid VM-level `avg_cpu` values at each timestamp, expressed as a percentage. It is not total consumption, CPU-core-weighted utilization, or a direct estimate of required cores. Core and memory buckets allow descriptive segments, but per-VM CPU capacity and fleet-level headroom are not fully specified.

The trace timestamp is seconds from the beginning of the workload trace, not a wall-clock datetime. It covers 11.083 hours at 300-second cadence. The final 24 timestamps are public; holdout results are reported only after the predictions were generated and are not used to revise model selection.

## Dataset Understanding And Quality

The repository provides the challenge prompt, [`ASSIGNMENT.md`](../ASSIGNMENT.md), [`SUBMISSION.md`](../SUBMISSION.md), and [`DATASET.md`](../DATASET.md). The dataset documentation defines `timestamp, vm_id, min_cpu, max_cpu, avg_cpu` for readings and 11 metadata fields including workload category, virtual-core bucket, memory bucket, and source lifecycle fields.

| Check | Observed result |
|---|---:|
| CPU reading rows / unique VMs | 1,340,000 / 10,000 |
| Unique timestamps / range | 134 / 0 to 39,900 seconds |
| Timestamp interval | 133 intervals, all 300 seconds |
| Readings per VM | 134 for all 10,000 VMs |
| Rows per timestamp | 10,000 at all 134 timestamps |
| Duplicate VM/timestamp keys | 0 |
| Missing / non-finite / out-of-range `avg_cpu` | 0 / 0 / 0 |
| Metadata rows / duplicate IDs | 10,000 / 0 |
| Reading-to-metadata key mismatches | 0 in either direction |

Input rows are timestamp-sorted; after numeric parsing, timestamp is `int64`, VM ID is string, and the CPU fields are `float64`.
Every VM has an observed trace from timestamp 0 through 39,900. This is consistent with the documented selection of complete traces; it does not establish real infrastructure creation/deletion or prove the broader fleet had no churn. The lifecycle columns are not used because the data card does not define their epoch/units. Duplicate handling is implemented defensively even though none occurred: duplicate readings are reduced to one per-VM/timestamp mean; a VM contributes once to population. A missing or invalid CPU value would be excluded from the utilization mean, never treated as zero, while an existing VM/timestamp row would still represent activity. No target interpolation or forward filling is performed.

## Workload Analysis

Across all VM-level `avg_cpu` readings, the median is 5.11%, the interquartile range is 2.48–9.62%, p90 is 19.60%, p95 is 37.13%, p99 is 87.35%, and the observed maximum is 99.49%. No average-CPU observation is exactly zero; this distinguishes low positive utilization from missing data, and the audit found no missing values. Across per-VM means, the median is 5.48%; the distribution has a long high-utilization tail. The 99th percentile of per-VM standard deviation is 28.58 percentage points, indicating a subset of highly variable VMs alongside consistently low-utilization VMs.

The aggregate mean CPU target averages 9.388% over all timestamps (standard deviation 0.195 percentage points), ranging from 8.984% at second 33,300 to 9.912% at second 16,500. One-step autocorrelation is 0.681; lag 3 is 0.708, while lag 24 is 0.128 and lag 48 is -0.173. Short-run persistence is visible, but the short trace does not support a daily/weekly seasonality claim. The largest adjacent aggregate change is 0.437 percentage points. Past-only trailing-12-step means/standard deviations and first differences are saved in [`../outputs/target_dynamics.csv`](../outputs/target_dynamics.csv). See [`workload_overview.png`](figures/workload_overview.png), [`cpu_distribution.png`](figures/cpu_distribution.png), and [`vm_category_cpu.png`](figures/vm_category_cpu.png).

Metadata segments show workload differences, not causal effects. The 6,180 delay-insensitive VMs have an average per-VM mean of 8.24%; the 3,091 interactive VMs average 11.49%; the 729 unknown-category VMs average 10.23%. The 10-VM `>24` core bucket is too small for a broad conclusion; even the 62-VM 24-core bucket should be interpreted cautiously. All metadata rows matched and no metadata values were missing. Opaque subscription/deployment IDs are not treated as meaningful categories. Per-VM observed zero-share is 0 throughout, and no activity churn is observable in this complete-trace selection.

Operationally, aggregate CPU is useful for detecting movement in average utilization across the sample, while the wide per-VM distribution shows why an average cannot guard against localized hot VMs. Capacity actions should monitor per-VM and capacity-weighted signals in production.

## Targets And Preprocessing

The full chronological 134-row target table is built before splitting. Each target row carries the represented VM count, valid CPU VM count, and source-row count. The challenge contract's 110/24 split is satisfied exactly: training timestamps are 0–32,700 seconds and holdout timestamps are 33,000–39,900 seconds. `active_vm_count` is integer-valued in observations; its forecasts remain numeric and are not rounded. CPU predictions are constrained to [0,100], and VM counts to nonnegative values.

## Forecasting And Temporal Validation

Three predeclared fixed-origin methods are compared independently for each target: persistence, a trailing-12-observation arithmetic mean repeated over the horizon, and a linear trend over the latest 12 observations with 0.85 geometric damping. None uses actual values inside a forecast horizon. No seasonal-naive method is used because the observed trace is too short to justify a recurring period. No learned scaler or imputer is needed for these direct statistical methods.

Expanding origins use the first 62, 74, and 86 observations as histories, followed by 24-step evaluation windows. All are wholly within the first 110 points and all models use identical windows. The windows overlap, so the 72 validation forecast rows (three origins times 24 steps) are correlated, not 72 independent experiments. The 12-step gap between origins offers repeated checks but the short training series limits reliability. RMSE selects the minimum pooled validation error; ties are resolved in favor of the simpler, preordered method. Normalized MAE divides each row's absolute error by the mean absolute target value in that origin's training history, then averages the ratios. Peak periods use each origin's training-only 90th percentile; observed validation ties at or above the threshold count as peaks.

| Target | Method | MAE | RMSE | Normalized MAE | RMSE improvement vs persistence | Peak MAE (n) |
|---|---|---:|---:|---:|---:|---:|
| active VM count (VMs) | persistence | 0.000 | 0.000 | 0.000 | undefined (zero baseline RMSE) | 0.000 (72*) |
| active VM count (VMs) | moving average | 0.000 | 0.000 | 0.000 | undefined (zero baseline RMSE) | 0.000 (72*) |
| active VM count (VMs) | damped trend | 0.000 | 0.000 | 0.000 | undefined (zero baseline RMSE) | 0.000 (72*) |
| mean CPU (%) | persistence | 0.185 pp | 0.236 pp | 0.0195 | 0.0% | 0.073 pp (6) |
| mean CPU (%) | moving average | 0.231 pp | 0.273 pp | 0.0244 | -16.0% | 0.105 pp (6) |
| mean CPU (%) | damped trend | 0.249 pp | 0.347 pp | 0.0264 | -47.3% | 0.145 pp (6) |

`pp` means percentage points, not percent relative error. Relative RMSE improvement is `100 * (persistence RMSE - method RMSE) / persistence RMSE`; it is undefined when persistence RMSE is zero. The `active_vm_count` peak threshold equals 10,000 for every fold, so ties mark every point as a “peak”; these rows are not a meaningful high-load test. CPU peak MAE is based on only six validation rows and should be treated cautiously. Validation errors by horizon are in [`validation_horizon_errors.png`](figures/validation_horizon_errors.png).

Persistence is selected for both targets. Population persistence is exact because the curated sample represents all 10,000 VMs at every step. For mean CPU, persistence has the lowest validation MAE and RMSE; moving average and damped trend degrade RMSE by 16.0% and 47.3%, respectively. The persistence residuals have mean forecast-minus-actual error +0.171 percentage points over validation, showing modest average overprediction in those windows. A pooled p10–p90 residual band had 77.8% in-sample validation coverage for CPU; this is descriptive calibration on a small, overlapping set, not guaranteed coverage. Residual distributions and horizon behavior are shown in [`validation_residuals.png`](figures/validation_residuals.png) and [`validation_horizon_errors.png`](figures/validation_horizon_errors.png).

## Final Forecast And Retrospective Holdout

After selection, each method is fitted on the first 110 observations and forecasts timestamps 33,000–39,900 without accessing intermediate holdout targets. The active population is forecast as 10,000 throughout. The selected CPU forecast repeats the last training target, 9.1646%, across all 24 timestamps.

| Target / method | Holdout MAE | Holdout RMSE | Normalized MAE | Mean forecast error |
|---|---:|---:|---:|---:|
| active VM count / persistence | 0.000 VMs | 0.000 VMs | 0.000 | 0.000 VMs |
| mean CPU / persistence (selected) | 0.147 pp | 0.185 pp | 0.0156 | -0.104 pp |
| mean CPU / moving average (retrospective comparator) | 0.123 pp | 0.161 pp | 0.0131 | -0.053 pp |

Negative mean forecast error is underprediction. The holdout CPU actuals average 9.269%, with a minimum of 8.984% and maximum 9.636%. None reaches the first-110 training p90 threshold (9.668%), so this holdout contains zero CPU peak samples under the predeclared rule. The moving average happens to score 12.7% lower RMSE than persistence on this public holdout, but model selection was based on training-only validation and is not changed after seeing that result. The holdout is now consumed and its score is specific to this trace/window. See [`holdout_forecasts.png`](figures/holdout_forecasts.png).

## Error Analysis And Operational Recommendation

Across the selected CPU forecasts, validation errors are small in absolute units because the aggregate series moves within a narrow band. The forecasts are constant by construction and therefore cannot anticipate abrupt changes. The largest validation adjacent step was 0.437 pp; the final holdout's largest adjacent step was 0.401 pp. A future jump, a change in sample population, or one hot VM could be obscured by the unweighted average. The holdout's absence of threshold-defined peaks means peak-load performance cannot be inferred from its aggregate error.

Use the point forecast as a short-horizon trend signal, not as a standalone scale-to-capacity rule. For scaling, combine forecasted aggregate utilization with per-VM or core-weighted utilization, provisioned cores, latency/SLOs, and a safety margin calibrated on longer rolling production history. Trigger scale-up when capacity-weighted demand plus a calibrated upper residual margin approaches a service-specific operating limit; retain a minimum headroom floor and scale down only after sustained slack to avoid oscillation. The current residual band is too weakly supported to set a numerical production margin. Under-provisioning risks saturation, latency, and failed work; excessive headroom increases cost. This sample's 10,000 represented-VM count is a fixed curated cohort, so it offers no evidence for population autoscaling or churn response.

## Limitations And Reproducibility

The trace is only about 11.1 hours, has no daily/weekly coverage, and represents a deliberately stratified population of complete traces. Observed population constancy reflects this challenge sample, not a live fleet. CPU utilization is unweighted and cannot determine CPU-core requirements. Validation windows overlap; the public holdout is a single window; residual uncertainty is limited and should not be presented as a guarantee. Segment summaries are descriptive and do not establish causal workload effects.

From a clean checkout with Python 3.11+, run `python -m pip install -e ".[dev]"`, then `python -m cloud_forecasting.cli analyze --config configs/default.yaml`, `python -m cloud_forecasting.cli validate --config configs/default.yaml`, and `python -m cloud_forecasting.cli forecast --config configs/default.yaml`. Run tests with `python -m pytest -q`. The generated root submission has exactly 24 rows and the required columns `timestamp,active_vm_count,aggregate_cpu_utilization`; diagnostics and evaluation artifacts are under `outputs/`, figures under `reports/figures/`.