"""Build (and execute) the **retired** electricity-load forecasting tutorial.

Retired on 2026-11-19 and replaced by ``build_solar_forecasting_tutorial.py``, which now
owns ``docs/forecasting_tutorial.ipynb``. This file is kept for its prose, not because
the notebook it builds is wanted: the forecasting task it poses — next-step load from
past load — did not illustrate anything a reader needed.

So it no longer writes ``forecasting_tutorial.ipynb``. It writes
``forecasting_tutorial_electricity.ipynb``, under the guard, and the output is not
checked in. Run it only to read the old tutorial again:

    python docs/build_forecasting_tutorial.py

Needs the ``[timeseries]`` extra — darts — and downloads the electricity dataset the
first time (about a minute; darts caches it afterwards).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook_guard import refuse_if_hand_edited  # noqa: E402

HERE = Path(__file__).resolve().parent
TARGET = HERE / "forecasting_tutorial_electricity.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELLS: list = []


def m(text: str) -> None:
    CELLS.append(md(text.strip("\n")))


def c(text: str) -> None:
    CELLS.append(code(text.strip("\n")))


# ==========================================================================
m(
    """
# Assessing a forecaster end to end

### ElectricityLoadDiagrams · quantile gradient boosting via darts · `themis-assess`

This notebook is the forecasting counterpart of [`tutorial_classification.ipynb`](tutorial_classification.ipynb). It
takes one forecaster through every layer of the library and, at each step, says what the
number means and what it does *not* mean.

The data is the [ElectricityLoadDiagrams2011–2014
dataset](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014): the
electricity consumption of 370 Portuguese clients, recorded every fifteen minutes for
four years. It ships with darts, so the only cost is a one-off download of about a
minute.

**Forecasting is not classification with a different metric.** Four things change, and
each one is a section below:

- The truth is a *slice of the input*. A perturbation that touches the target column
  moves the answer along with the question, so the ground truth has to be captured once
  and held fixed.
- A single error number hides *when* the model is bad and *how far ahead* it stays
  usable. Those are two different curves over the same array.
- A forecast has a time axis inside it, so "which input does it lean on" becomes
  "which input, **at which point in the window**".
- A forecaster that reports uncertainty can be asked whether the uncertainty is honest —
  and under shift, usually it is not.

### Running it

```bash
conda env create -f environment.yml          # or your own environment
conda activate themis-assess
pip install -e ".[timeseries,tutorial]"
jupyter notebook docs/forecasting_tutorial_electricity.ipynb
```

If plotly figures come out blank, that is a Jupyter renderer issue rather than a library
one: `import plotly.io as pio; pio.renderers.default = "notebook"` usually settles it.
"""
)

# ==========================================================================
m(
    """
## 1. The data, the model, and the three objects the library needs

Three wrappers, and none of them asks you to change your model or your data.

**An adapter** presents the data in the library's canonical
`(entities, timesteps, features)` shape. For a forecaster there is a choice to make here
and it is the most consequential decision in the notebook — see below.

**A handler** turns the model into predictions and *declares what it can produce*.
`PROBABILISTIC` says "I can give you intervals and samples", and that declaration is a
promise the library checks and plans against.

**A `ForecastingConfig`** describes the evaluation protocol rather than the model: how
far ahead, how much history, how often a forecast is issued. Repo B passed these four
values separately, which is how its baseline came to be evaluated with `retrain=True`
while the model it was compared against used `retrain=False` — the two were never
scored under the same protocol. Bundling them makes that impossible.
"""
)

c(
    """
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.datasets import ElectricityDataset
from darts.models import SKLearnModel
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import themis_assess as ta

print("themis-assess", ta.__version__)
"""
)

c(
    """
# 370 clients at 15-minute resolution, 2011-2015. Downloaded once, then cached by darts.
full = ElectricityDataset().load()
print("raw:", full.all_values().shape, full.start_time(), "->", full.end_time())

# Eight clients, hourly, one calendar year. Enough for daily and weekly structure to be
# visible, small enough that the whole notebook runs in a couple of minutes.
CLIENTS = [str(c) for c in full.components[:8]]
hourly = full[CLIENTS].resample("h")
year = hourly.slice(pd.Timestamp("2013-01-01"), pd.Timestamp("2013-12-31 23:00"))

values = np.asarray(year.values(), dtype=float)
timestamps = year.time_index
print("subset:", values.shape, year.start_time(), "->", year.end_time())
print("clients:", CLIENTS)
"""
)

m(
    """
### The darts adapter, and where the dependency stops

`DartsAdapter` is the only module in the library that imports darts, and it ships behind
the `[timeseries]` extra (`docs/DECISIONS.md`, D11). Everything downstream — every
perturbation, every metric — sees an array and knows nothing about darts, which is what
lets a perturbation written for the clinical tabular use case act on a time series
unchanged.

It round-trips, so a darts model can go on consuming `TimeSeries` on the other side.
"""
)

c(
    """
from themis_assess.integrations.darts_adapter import DartsAdapter

long_data = DartsAdapter([year], categorical_features=[])
print(long_data)

# round-trip: adapter -> TimeSeries -> values
back = long_data.to_series()[0]
print("round-trips:", np.allclose(np.asarray(back.values(), dtype=float), values))
"""
)

m(
    """
### The split, before anything is fitted

`temporal_split` is survey §3.1's protocol 2: train before a cutoff, test after. The
`gap` is the part worth noticing — a test window reaching back across the boundary reads
training data, so the first `gap` steps of the test side are dropped. Getting this wrong
produces a model that looks better than it is, and nothing in the resulting numbers says
so.
"""
)

c(
    """
LAGS = 168        # one week of hourly history, so the model can see the weekly cycle
HORIZON = 24      # a day ahead

# multi_models=False (below) predicts the whole horizon with one model by shifting its
# input, so it needs LAGS + HORIZON - 1 steps of history rather than LAGS. This is the
# window every later section cuts, so it is defined once here.
HISTORY = LAGS + HORIZON - 1

split = ta.temporal_split(timestamps, test_fraction=0.3, gap=HISTORY)[0]
print(split)
print("cutoff:", split.metadata["cutoff"], "| dropped to gap:", split.metadata["n_dropped_to_gap"])

train_values = values[split.train]
test_values = values[split.test]
test_times = timestamps[split.test]
print("train:", train_values.shape, " test:", test_values.shape)
"""
)

m(
    """
### The model

One client is the target — `MT_001` — and the other seven are past covariates. Seven
quantile models are fitted rather than one point model, so the forecast carries a
distribution and section 5 has something to assess.

The levels are not arbitrary, and the comment on them is worth reading: three separate
constraints pin that grid down, and two of them come from the library rather than from
the model.

The **median imputer is not decoration**. Two of the perturbations in section 6 hand the
model `NaN`, and a bare `HistGradientBoostingRegressor` in a pipeline without one would
refuse them — at which point the honest options are to fix the model or to drop the
perturbation, and dropping it overstates robustness.
"""
)

c(
    """
TARGET = CLIENTS[0]

# Seven levels, 0.15 apart. Three constraints meet here, and this is the smallest grid
# that satisfies all of them:
#   - the outer pair must be 0.05/0.95, because the metric registry declares
#     prediction_interval_coverage best at target=0.9, and a requirement checked against
#     a 90% target has to be measuring a 90% interval;
#   - the spacing must be even, because the CRPS and energy-score estimators read the
#     levels as an equally weighted ensemble, which uneven levels would bias;
#   - 0.5 must be one of them, since the median is the point forecast.
QUANTILES = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]

target_train = TimeSeries.from_values(train_values[:, :1])
covariate_train = TimeSeries.from_values(train_values[:, 1:])


def quantile_model(q):
    return SKLearnModel(
        lags=LAGS,
        lags_past_covariates=LAGS,
        output_chunk_length=HORIZON,
        multi_models=False,
        model=make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=50, random_state=0
            ),
        ),
    )


models = {
    q: quantile_model(q).fit(target_train, past_covariates=covariate_train)
    for q in QUANTILES
}
print(f"fitted {len(models)} quantile models; target = {TARGET}")
"""
)

m(
    """
### One entity per window — the decision that makes the rest work

The library's entity axis is `n_entities`, and the electricity data invites getting this
wrong. Treating each *client* as an entity gives eight rows: a perturbation sweep over
eight samples is noise, and the error model in section 8 would have nothing to learn
from.

The right shape is to cut the test period into fixed windows and let **each window be an
entity**: `(n_windows, HISTORY + HORIZON, n_features)`. Then one forecast is issued per
window, the truth is the window's own tail, the sweep has a few hundred rows, and the
sensitivity map in section 7 runs across positions *inside* the window.

So the notebook uses two adapters, and the difference is worth holding on to: the
long-series adapter above for the rolling evaluation in section 3, and this windowed one
for everything that needs many entities.
"""
)

c(
    """
STRIDE = 12      # a new window every 12 hours

starts = list(range(0, len(test_values) - (HISTORY + HORIZON) + 1, STRIDE))
windows = np.stack([test_values[s : s + HISTORY + HORIZON] for s in starts])
origins = pd.DatetimeIndex([test_times[s + HISTORY] for s in starts])

data = ta.ArrayAdapter(
    windows,
    feature_names=CLIENTS,
    categorical_features=[],            # all eight are continuous
    entity_ids=[str(t) for t in origins],
)
print(data)
print(f"{len(starts)} windows of {HISTORY} history + {HORIZON} horizon")
print("first origin:", origins[0], " last origin:", origins[-1])
"""
)

m(
    """
### The handler

`ForecastingHandler` takes a plain function — `forecast_fn(history, horizon) -> array` —
so the library stays free of any forecasting package. That function is called once per
entity per origin, which is fine for the rolling evaluation and wasteful for a sweep, so
the handler's own docstring points at the alternative: override `predict` where the model
can work on a batch. darts takes a list of series at once, so that is what the subclass
below does, and it is the difference between a sweep that takes seconds and one that
takes minutes.

Declaring `PROBABILISTIC` is a promise. `check_prediction` enforces it: a handler that
claims intervals and returns bare points is refused, because the runner repeats that
claim to you.
"""
)

c(
    """
from themis_assess.handlers.base import Prediction


class DartsQuantileHandler(ta.ForecastingHandler):
    \"\"\"Five darts quantile models behind one handler.

    The median is the point forecast, the outer pair is the 90% interval, and all seven
    levels together are the ensemble the CRPS and energy-score estimators read.
    \"\"\"

    def __init__(self, models, config, **kwargs):
        self.models = models
        self.levels = sorted(models)
        super().__init__(
            self._median_forecast,
            config,
            access=ta.AccessLevel.PROBABILISTIC,
            **kwargs,
        )

    @staticmethod
    def _split_window(block):
        block = np.asarray(block, dtype=float)
        return (
            TimeSeries.from_values(block[:, :1]),      # the target client
            TimeSeries.from_values(block[:, 1:]),      # the other seven, as covariates
        )

    def _median_forecast(self, history, horizon):
        \"\"\"One window, median only. What historical_forecasts() calls.\"\"\"
        target, covariates = self._split_window(history)
        out = self.models[0.5].predict(
            n=horizon, series=target, past_covariates=covariates, show_warnings=False
        )
        return np.asarray(out.values(), dtype=float).ravel()

    def _batch(self, data):
        block = data.values()[:, : self.config.input_chunk_length, :]
        pairs = [self._split_window(block[e]) for e in range(data.n_entities)]
        return [t for t, _ in pairs], [c for _, c in pairs]

    def predict(self, data):
        targets, covariates = self._batch(data)
        levels = []
        for q in self.levels:
            out = self.models[q].predict(
                n=self.config.forecast_horizon,
                series=targets,
                past_covariates=covariates,
                show_warnings=False,
            )
            levels.append(
                np.stack([np.asarray(o.values(), dtype=float).ravel() for o in out])
            )

        # (n_windows, n_levels, horizon). Sorted along the level axis because quantile
        # models fitted independently can cross, and a lower bound above its upper bound
        # would make every interval metric meaningless.
        samples = np.sort(np.stack(levels, axis=1), axis=1)
        median = samples[:, len(self.levels) // 2, :]
        return self.check_prediction(
            Prediction(
                predicted=median,
                lower=samples[:, 0, :],
                upper=samples[:, -1, :],
                samples=samples,
                metadata={
                    "model": self.name,
                    "quantiles": self.levels,
                    "nominal_interval": self.levels[-1] - self.levels[0],
                },
            ),
            data,
        )


config = ta.ForecastingConfig(
    forecast_horizon=HORIZON, input_chunk_length=HISTORY, stride=STRIDE
)
handler = DartsQuantileHandler(models, config, name="quantile_gbm", target=TARGET)

print("access:", handler.access.name, "| task:", handler.task.name)
"""
)

m(
    """
`actual()` sits next to `predict()` on purpose. In a forecasting assessment the truth is
a *slice of the input*, so which slice depends on the same origin arithmetic the forecast
used — and being out by one step is both easy and invisible in the resulting numbers.
Captured once, from clean data, and never recomputed (`docs/DECISIONS.md`, D30).
"""
)

c(
    """
prediction = handler.predict(data)
actual = handler.actual(data)

print("forecasts:", prediction.predicted.shape)
print("truth    :", actual.shape)
print("available:", sorted(prediction.available()))
print("interval :", prediction.metadata["nominal_interval"])
"""
)

# ==========================================================================
m(
    """
## 2. Accuracy — and the thresholds that turn numbers into a verdict

Running the metrics is the easy half. The half that decides anything is the **acceptance
criteria**: what "good enough" means *here*.

The library will not invent them. Survey §8 is explicit that the acceptance bar is a
property of the deployment rather than of the model, so it has to come from whoever owns
the deployment. What the library does is make its absence visible: with no criteria the
verdict is `None`, which means **nothing was checked**, and that is not a pass.
"""
)

c(
    """
criteria = ta.AcceptanceCriteria(
    [
        ta.Requirement(
            "mae",
            acceptable=3.0,
            rationale="a day-ahead error above 3 kW would change the scheduling decision",
            critical=True,
        ),
        ta.Requirement(
            "r2",
            acceptable=0.6,
            rationale="below this the forecast is not beating the seasonal shape",
            critical=False,
        ),
        # tolerance=, not acceptable=. Coverage has no direction: it is best at its
        # nominal level and worse in *either* direction, since over-coverage is bought
        # with width. The registry knows that and refuses an acceptable= on it.
        ta.Requirement(
            "prediction_interval_coverage",
            tolerance=0.07,
            rationale="a 90% interval covering under 83% misleads; over 97% is padding",
            critical=False,
        ),
    ],
    source="tutorial — illustrative, not an operational sign-off",
)

evaluator = ta.AccuracyEvaluator(handler, criteria=criteria)
plan = evaluator.plan(data)
print("will compute:", plan["metrics"])
print()
for name, reason in sorted(plan["skipped"].items()):
    print(f"  skipped {name}: {reason}")
"""
)

m(
    """
`plan()` before `run()` is a habit worth keeping: it says what will be computed, what
will be skipped and *why*, for one model call, before anything expensive starts. Only
`crps_gaussian` is skipped here, and for the honest reason that this handler reports
quantiles rather than a mean and a standard deviation.

Now watch `crps` in the run below. The planner expects to compute it — the handler
returned `samples`, which is all the catalog requires — and then it **fails**, and the
result says `undefined: crps` rather than carrying a number or taking the run down.

That is worth understanding, because it is the distinction the library works hardest to
preserve. `crps` wants `(n_observations, n_members)` against a one-dimensional truth, and
this forecast is a 24-step *trajectory*: 24 dimensions per window. So the metric is
computable in general, inapplicable to this shape, and recorded **by name** as undefined —
not silently dropped, and not quietly returned as a NaN that would then propagate through
every mean and ratio downstream.

The survey separates the two deliberately: CRPS is Eq. 44, and the **energy score** is
Eq. 45, "the natural choice when calibrated joint behaviour across several outputs
matters, for example in multi-horizon forecasting". Section 5 uses the energy score for
that reason, and computes CRPS per horizon step where the univariate question is the one
being asked.
"""
)

c(
    """
accuracy = evaluator.run(data, actual)
print(accuracy.summary())
"""
)

c(
    """
from themis_assess.viz import plot_metrics_table

plot_metrics_table(accuracy.acceptance)
"""
)

m(
    """
The verdict is *in* the table — value, requirement, pass or fail, and by how much — with
a symbol as well as a colour so it survives a black-and-white print and a colour-blind
reader. Rows with no requirement stay neutral rather than being painted as warnings.

**This model fails.** MAE 4.76 against a required 3, R² 0.27 against a required 0.6. It
has not been tuned to pass, and the rest of the notebook deliberately proceeds anyway —
partly because an assessment library has to be legible on a model that is *not* finished,
and partly because section 10 turns out to have something to say about why this one is
weak that none of the intervening sections can see.

`mean_signed_deviation` is the one to read next to `mae`, and here it is doing real work:
at **-3.76** against an MAE of 4.76, almost the entire error is bias. The model runs low
on essentially every window rather than scattering around the truth. It is the only metric
here that keeps its sign, a symmetric error metric cannot show you this, and it is the
difference between a noisy forecast and one that will systematically under-provision.
"""
)

# ==========================================================================
m(
    """
## 3. When is it bad?

A single error number over a rolling evaluation hides the question a forecasting team
actually asks: is the error steady, or concentrated around particular moments?

This is the one section that uses the **long-series adapter** rather than the windowed
one. `historical_forecasts` rolls an origin along a single continuous series, which is
the shape `accuracy_over_time` wants, and it is the generic form of the call that in repo
B could not execute at all — it passed `covariates=` to a handler that takes
`past_covariates`, so it raised `TypeError` before doing anything.
"""
)

c(
    """
from sklearn.metrics import mean_absolute_error

# One entity, the whole test period, origins every 24 hours.
rolling = ta.ArrayAdapter(
    test_values[None, :, :], feature_names=CLIENTS, categorical_features=[]
)
rolling_config = ta.ForecastingConfig(
    forecast_horizon=HORIZON, input_chunk_length=HISTORY, stride=24
)
rolling_handler = DartsQuantileHandler(
    models, rolling_config, name="quantile_gbm_rolling", target=TARGET
)

history = rolling_handler.historical_forecasts(rolling)
print("forecasts:", history["forecasts"].shape, " origins:", history["origins"].shape)

origin_labels = [str(test_times[o])[:16] for o in history["origins"]]
"""
)

c(
    """
from themis_assess.viz import assess_accuracy_over_time

over_time = assess_accuracy_over_time(
    history["forecasts"],
    history["actual"],
    mean_absolute_error,
    origins=origin_labels,
    metric="MAE",
    highlight_worst=3,
    title=f"{TARGET}: day-ahead MAE by forecast origin",
)
print(over_time.summary)
over_time.figure
"""
)

m(
    """
That is the shape of the bundle layer, and the reason it exists. Repo B's version of this
assembled the figure *and* the statistics into a dictionary, stashed it on
`self.results[title]` keyed by the plot's title string, and returned only the figure — so
the caller got the picture and never the numbers. Here `assess_*` returns both
(`docs/DECISIONS.md`, D44), and `viz/figures.py` keeps a pure figure-only form for
anything that needs a rendering with no analysis attached.

`worst_windows` is the output worth acting on. An aggregate says "the model is fine"; this
says which specific moments it was not, labelled with real timestamps, and those are what
somebody can go and look at.
"""
)

c(
    """
for entry in over_time.analysis["worst"]:
    print(f"  {entry['origin']}   MAE {entry['value']:.3f}")
"""
)

# ==========================================================================
m(
    """
## 4. How far ahead is it usable?

The other axis of the same array, and the one repo B never computed — its notebooks
reported a single aggregate over the whole horizon. For a forecast that gets acted on at
a particular lead time this is the more decision-relevant curve: it says how far ahead
the model is still worth trusting.

It is *usually* increasing, and the library's own docstring says that where it is not,
the cause is often a target leak or a misaligned origin. This one is not increasing — and
it is neither of those, which makes it the more useful example.

Look at the shape below rather than the headline. The error falls from one hour ahead to
eighteen and then jumps at twenty-four. Nothing is leaking: every origin here is issued at
the same time of day, because `stride=24`, so horizon step *h* always lands on the same
hour of the clock. The curve is therefore as much **error by hour of day** as error by
lead time, and the daily demand cycle is what is being read.

That is a real limitation of this plot as drawn, not a defect in the model, and the fix is
a stride that is not a divisor of the daily period — which would mix the hours across
origins and let the lead-time effect separate from the diurnal one. Worth knowing before
quoting "the 24-hour-ahead error" at anybody.
"""
)

c(
    """
from themis_assess.metrics.accuracy.temporal import accuracy_by_horizon, aggregate_over_time

by_horizon = accuracy_by_horizon(
    history["forecasts"], history["actual"], mean_absolute_error
)

for h in (0, 5, 11, 17, 23):
    print(f"  {h + 1:2d} hour(s) ahead:  MAE {by_horizon[h]:.3f}")
print()
print(f"  mean over the horizon : {by_horizon.mean():.3f}")
print(f"  worst single horizon  : {by_horizon.max():.3f} at {by_horizon.argmax() + 1}h")
"""
)

m(
    """
`aggregate_over_time` collapses the per-origin curve to one number, and offers `max` and
`p90` as well as `mean` and `median`. That is not padding: survey §8 asks for the worst
case beside the average, and a mean over origins is precisely where a worst case goes
missing.
"""
)

c(
    """
curve = over_time.analysis["curve"]
for method in ("mean", "median", "p90", "max"):
    print(f"  {method:7s} {aggregate_over_time(curve, method=method):.3f}")
"""
)

# ==========================================================================
m(
    """
## 5. Is the uncertainty honest?

The model states a 90% interval. The question is whether the truth falls inside it 90%
of the time — and separately, what that coverage costs in width.

**PICP must never be reported without MPIW.** The survey is blunt about why: "calibration
on its own can be achieved trivially: arbitrarily wide intervals reach any coverage
level". The report layer in section 11 enforces the pairing as one of its ten rules.
"""
)

c(
    """
from themis_assess.metrics.calibration import (
    crps_ensemble,
    energy_score,
    interval_score,
    mean_prediction_interval_width,
    prediction_interval_coverage,
)

nominal = prediction.metadata["nominal_interval"]
picp = prediction_interval_coverage(prediction.lower, prediction.upper, actual)
mpiw = mean_prediction_interval_width(prediction.lower, prediction.upper)

print(f"nominal coverage : {nominal:.2f}")
print(f"PICP             : {picp:.3f}")
print(f"MPIW             : {mpiw:.3f}")
print(f"interval score   : {interval_score(prediction.lower, prediction.upper, actual, alpha=1 - nominal):.3f}")
print()
print(f"mean |error|     : {np.abs(prediction.predicted - actual).mean():.3f}")
"""
)

m(
    """
Read those two together rather than separately. If the coverage is well above nominal,
the intervals are not *better* than advertised — they are wider than they need to be, and
an operator who sizes a reserve from them pays for that width every hour. If it is below,
the stated confidence is a fiction. Either way the number that tells you which is MPIW,
and the interval score is the single proper scoring rule that prices the trade — width
paid for directly, a miss penalised in proportion to how far outside it fell.

### The energy score, and why not CRPS here

The forecast is a 24-step trajectory, so the joint question — is the whole *shape*
plausible? — is the energy score's, Eq. 45. CRPS is its one-dimensional case, which is
the right instrument once the question is about a single lead time, so it is computed
per horizon step below.

A model can be calibrated at every horizon individually while its trajectories are
implausible, and only the first of these two can see that.
"""
)

c(
    """
# samples: (n_windows, n_levels, horizon) -> energy score wants (n, members, dimensions)
print(f"energy score (24-dim trajectory): {energy_score(prediction.samples, actual):.4f}")
print()

# CRPS per horizon step: the univariate question, asked 24 times.
per_horizon_crps = [
    crps_ensemble(prediction.samples[:, :, h], actual[:, h]) for h in range(HORIZON)
]
for h in (0, 5, 11, 17, 23):
    print(f"  {h + 1:2d} hour(s) ahead:  CRPS {per_horizon_crps[h]:.3f}")
print()
print(f"  mean CRPS over the horizon: {np.mean(per_horizon_crps):.3f}")
"""
)

m(
    """
CRPS and MAE sit on the same axis — a single-member ensemble scores exactly its MAE — so
the gap between the mean CRPS above and the mean absolute error in the previous cell is
what the distribution is buying over a bare point forecast.
"""
)

# ==========================================================================
m(
    """
## 6. Robustness — what happens when the sensors misbehave

Everything so far assumed the input arrives intact. Section 6 asks what the model does
when it does not, over an ordered severity axis from 1 to 5.

The perturbations here are the sequence family, and each names a real failure of
instrumentation rather than an abstract corruption: a meter that drifts out of
calibration, one that sticks at a constant reading, a communications outage that removes
a stretch of history, spikes from electrical interference, and a shifted input window —
the one instrument in the set that is about *time* rather than values.

Magnitudes are **relative**: each severity scales the feature's own standard deviation,
so severity 3 means something comparable across clients whose loads differ by an order of
magnitude.
"""
)

c(
    """
PERTURBATIONS = [
    "sensor_drift",        # slow calibration loss
    "flat_sensor",         # a meter stuck at one value
    "missing_segment",     # a communications outage: NaN, which is why the imputer exists
    "outlier_spikes",      # interference
    "input_window_shift",  # the window arrives misaligned in time
]

sweep_evaluator = ta.RobustnessEvaluator(
    handler,
    perturbations=PERTURBATIONS,
    severities=(1, 3, 5),
    n_variants=1,
    seed=0,
)
sweep_plan = sweep_evaluator.plan(data, actual)
print("perturbations:", sweep_plan.perturbations)
print("severities   :", sweep_plan.severities)
print("model calls  :", sweep_plan.n_model_calls)
"""
)

m(
    """
### The tolerance, and the metric it has to name

`RobustnessCriteria` is the §8 counterpart of `AcceptanceCriteria`, and it rests on the
same sentence from the survey: the tolerance is the one element no evaluation instrument
can provide, because it comes from the deployment.

One detail here is easy to get wrong and the library will tell you when you do. `mae` is
an **error** metric, so the scoring layer produces the error family — survey Eqs. 13–18 —
and refuses the accuracy family. A tolerance naming `relative_robustness` comes back
`NOT CHECKED`, not `PASS`, and the summary says so rather than letting an overall verdict
stand over a requirement that was never scored. The one to name over `mae` is
`error_based_robustness`.

The `basis` is part of the requirement rather than a detail: `aggregate` averages away a
single catastrophic perturbation, `worst_cell` is dominated by severity 5 and often by
something the deployment will never see, and `worst_perturbation` sits between them.
"""
)

c(
    """
tolerance = ta.RobustnessCriteria(
    [
        ta.ToleranceRequirement(
            "error_based_robustness",
            acceptable=0.5,
            basis="worst_perturbation",
            rationale="a meter fault must not more than double the day-ahead error",
            critical=True,
        )
    ]
)

robustness = sweep_evaluator.assess(
    data, actual, accuracy_metric="mae", criteria=tolerance
)
print(robustness.summary())
"""
)

c(
    """
from themis_assess.viz import assess_severity

severity = assess_severity(robustness.sweep, "mae", title=f"{TARGET}: MAE under shift")
print(severity.summary)
severity.figure
"""
)

m(
    """
The clean value is drawn as a reference line on purpose. A severity curve without it shows
how much the metric moved and not whether there was anything to lose — which is the
difference between a finding and a shape.
"""
)

c(
    """
from themis_assess.viz import plot_tolerance_table

plot_tolerance_table(robustness.tolerance)
"""
)

m(
    """
### Calibration under shift

Survey §8 rule 7: a robustness report that cites only clean-data calibration can be
misleading, because calibration established on clean data need not survive a shift. The
sweep computed the interval metrics at every cell, so the question needs no extra model
calls — it is already in the result.
"""
)

c(
    """
clean_picp = robustness.sweep.clean["prediction_interval_coverage"]
print(f"clean PICP: {clean_picp:.3f}  (nominal {nominal:.2f})")
print()
for name in PERTURBATIONS:
    cells = robustness.sweep.curves("prediction_interval_coverage").get(name, {})
    widths = robustness.sweep.curves("mean_prediction_interval_width").get(name, {})
    shown = ", ".join(
        f"s{s}: {cells[s]:.3f} (w {widths[s]:.2f})" for s in sorted(cells)
    )
    print(f"  {name:20s} {shown}")
"""
)

m(
    """
Coverage and width have to be read as a pair here too, and this sweep shows why in the
direction that is easiest to miss.

Under `sensor_drift` and `outlier_spikes` the coverage *improves* with severity — 0.85
to 0.91, 0.85 to 0.92 — which looks like the model getting better under stress and is
nothing of the kind. The width grew with it, from 14.1 to 20.2 and 17.8. The intervals
bought their coverage, exactly as the survey says they trivially can.

`missing_segment` is the serious finding and it runs the other way. At severity 5 coverage
falls to **0.45** while the width *shrinks* from 14.1 to **5.9**. The model is not merely
wrong when its history goes missing — it is wrong and has become **more confident** while
being so, which is the single worst failure mode a probabilistic forecaster has. Coverage
alone would have shown half of this. Width alone would have shown none of it.

Neither statement is reachable from clean-data calibration, which is precisely what
survey §8's rule 7 exists to prevent anyone from publishing.
"""
)

# ==========================================================================
m(
    """
## 7. Which input is it leaning on, and when?

For a windowed forecaster the interesting question is not only *which* feature the model
depends on, but **where in the window**. A model that turns out to depend on a value six
days back is a model whose behaviour during a short outage is not what anyone assumed.

Two things to be precise about, because repo B's version of this heatmap was neither.

**Prediction shift is not degradation.** This map perturbs one cell and measures how far
the *forecast* moves, not how much accuracy is lost. That needs no labels, which is why it
can be computed on production data — and sensitivity is not harm: a model can move a lot
and still be right, or barely move and be confidently wrong. `measure="degradation"`
asks the other question, and the result records which was asked.

**It uses the median model alone.** A prediction-shift map needs only the point forecast,
so running one quantile model rather than five costs a fifth as much and measures exactly
the same thing.
"""
)

c(
    """
class MedianOnlyHandler(DartsQuantileHandler):
    \"\"\"The point forecast alone — all a prediction-shift map needs.\"\"\"

    def __init__(self, models, config, **kwargs):
        super().__init__(models, config, **kwargs)
        self.levels = [0.5]
        self.access = ta.AccessLevel.BLACK_BOX

    def predict(self, data):
        targets, covariates = self._batch(data)
        out = self.models[0.5].predict(
            n=self.config.forecast_horizon,
            series=targets,
            past_covariates=covariates,
            show_warnings=False,
        )
        return self.check_prediction(
            Prediction(
                predicted=np.stack(
                    [np.asarray(o.values(), dtype=float).ravel() for o in out]
                )
            ),
            data,
        )


point_handler = MedianOnlyHandler(models, config, name="quantile_gbm_median", target=TARGET)

mapper = ta.SensitivityMapper(
    point_handler,
    perturbation="gaussian_noise",
    measure="prediction_shift",
    n_variants=1,
    seed=0,
)
print(mapper.plan(data, axis="feature_time"))
"""
)

c(
    """
from themis_assess.viz import plot_sensitivity_map

time_map = mapper.feature_time_map(data, severity=3, time_granularity=24, window=HISTORY)
print("hottest cell:", time_map.hottest_cell())
plot_sensitivity_map(time_map, title=f"{TARGET}: prediction shift by client x window position")
"""
)

m(
    """
Each column is a day of the input window, the last column being the most recent 24 hours
before the forecast origin. The target client's own recent history should dominate; where
a *covariate* column lights up, the model has found a cross-client relationship, and
whether that is signal or an artefact of eight neighbouring meters is a question for
somebody who knows the grid.
"""
)

c(
    """
# The severity axis belongs to the mapper, not to the call: one mapper is one
# perturbation over one severity grid, and both maps below read the same grid.
severity_map = mapper.feature_severity_map(data)
print(severity_map.to_frame())
print()
for row in severity_map.most_sensitive_rows(n=3):
    print(f"  {row}")
"""
)

m(
    """
`tipping_point` is the question the severity axis exists to answer: at what severity does
this feature first push the forecast past a threshold you care about? `None` means the
threshold was never reached within the severities swept, which is a finding rather than a
gap — it says the model tolerated everything that was tried.
"""
)

c(
    """
threshold = float(np.median(severity_map.grid)) * 2
print(f"threshold: {threshold:.3f}")
for name in severity_map.row_labels:
    print(f"  {name}: {severity_map.tipping_point(name, threshold)}")
"""
)

# ==========================================================================
m(
    """
## 8. What do the windows it gets wrong have in common?

The sections so far measure *how much* the model degrades. This one asks *why* — a
question the survey raises but does not instrument.

A surrogate model is fitted to predict **this model's error** from summaries of its input
window. If that surrogate learns nothing, its attributions are indistinguishable from
noise and the output looks identical either way — so `is_trustworthy` gates the
attribution methods and overriding it takes `force=True`.

The default feature extractor is twelve interpretable window summaries computed in numpy,
not TSFresh. Interpretability is the substantive argument rather than the dependency:
"the covariate's range over the input window was high" is a finding somebody can act on,
and `cov__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2` is not.
"""
)

c(
    """
from themis_assess.attribution.features import WindowSummaryFeatures

error_model = ta.ErrorModel(extractor=WindowSummaryFeatures(), seed=0)
fit = error_model.fit(data, actual, prediction.predicted)
print(fit.describe())
print()
print("gate:", fit.gate, f"= {fit.gate_value:.3f}", "| threshold:", fit.gate_threshold)
print("trustworthy:", fit.is_trustworthy, "| looks overfitted:", fit.looks_overfitted)
"""
)

c(
    """
if fit.is_trustworthy:
    attribution = error_model.importance(method="permutation", n_repeats=10, on="test")
    for entry in attribution.top(8):
        print(f"  {entry['feature']:40s} {entry['importance']:.4f}")
else:
    print("The surrogate did not learn the error. No attribution is offered, and that")
    print("refusal is the result — an ungated attribution here would be noise with")
    print("feature names attached.")
"""
)

c(
    """
if fit.is_trustworthy:
    from themis_assess.viz import plot_attribution

    display(plot_attribution(attribution, title="What goes with a larger day-ahead error"))
"""
)

m(
    """
### The two answers disagree, and that is not a contradiction

Section 7 said the model leans on `MT_001` above everything else — a prediction shift of
4.07 at severity 5, against 0.27 for `MT_005`. The error model's top feature is
`MT_005__mean`.

Both are right, because they answer different questions. **Sensitivity asks what moves the
forecast**; the model's own recent history moves it most, which is what you would expect
and is not where its mistakes come from. **Attribution asks what the bad windows have in
common**; the mean level of a neighbouring client is a marker for *when* the model
struggles — high-demand periods — without being an input the forecast swings on.

A marker is not a cause, and neither of these is a causal claim. What the pair buys you is
that a single point of failure in the input pipeline and a predictor of when the model is
unreliable are different facts about the same model, and a reader handed only one of them
would draw the wrong conclusion from it.

The check worth doing when a surrogate *does* pass: the raw marginal relationship, with no
model in between. A feature the surrogate ranks highly should show something here — and
the correlation is reported on the held-out rows too, because a marginal effect that
exists only where the surrogate trained is the surrogate's memory rather than a finding.
"""
)

c(
    """
if fit.is_trustworthy:
    top_feature = attribution.top(1)[0]["feature"]
    scatter = error_model.feature_error_scatter(top_feature)
    xs, ys = scatter["feature_value"], scatter["error"]
    held_out = scatter["is_test"]

    # The correlation is the plainest summary of the marginal relationship, and it is
    # reported on the held-out rows as well as on all of them: a marginal effect that
    # exists only on the rows the surrogate trained on is the surrogate's memory.
    print(f"{top_feature}")
    print(f"  correlation with error, all rows  : {np.corrcoef(xs, ys)[0, 1]:+.3f}")
    print(f"  correlation with error, held out  : "
          f"{np.corrcoef(xs[held_out], ys[held_out])[0, 1]:+.3f}")
"""
)

# ==========================================================================
m(
    """
## 9. Does it work equally well all year?

A *group* in this library is any named partition of the data, and for a forecaster the
natural ones are temporal: the season a window falls in, the hour of day it was issued at.
Groups are **masks**, never feature indices — which is what lets the same subgroup
machinery serve a clinical one-hot column, a text theme and a calendar month without
knowing anything about any of them.

The descriptive half is the gap. The inferential half — whether that gap is
distinguishable from sampling variation — is a different question, and the gap alone
invites a reader to get it wrong.
"""
)

c(
    """
season_of = np.where(
    origins.month.isin([12, 1, 2]), "winter",
    np.where(origins.month.isin([3, 4, 5]), "spring",
             np.where(origins.month.isin([6, 7, 8]), "summer", "autumn")),
)
print(pd.Series(season_of).value_counts().to_string())

subgroups = [
    ta.SubgroupSpec(
        name=label,
        mask_fn=(lambda lbl: (lambda _: season_of == lbl))(label),
        description=f"forecast origins falling in {label}",
    )
    for label in sorted(set(season_of))
]

grouped = ta.AccuracyEvaluator(handler, criteria=criteria, subgroups=subgroups).run(
    data, actual
)
for name, result in sorted(grouped.subgroups.items()):
    print(f"  {name:8s} n={result.n_samples:4d}  MAE {result['mae']:.3f}")
print()
print("worst subgroup on mae:", grouped.worst_subgroup("mae", higher_is_better=False))
"""
)

c(
    """
from themis_assess.viz import assess_groups

per_window_error = np.abs(prediction.predicted - actual).mean(axis=1)
groups = assess_groups(
    per_window_error,
    season_of,
    metric="per-window MAE",
    title=f"{TARGET}: day-ahead error by season",
)
print(groups.summary)
groups.figure
"""
)

m(
    """
### Two groups, not four — and the gap is real but not what it looks like

A temporal split puts the test period at the *end* of the year, so there are no spring or
summer windows to compare: autumn and winter are all there is. That is a property of the
protocol rather than an oversight, and it is the kind of thing worth noticing before
writing "performance by season" in a report.

The gap is enormous and the test is emphatic — autumn 6.40 against winter 1.32, Welch
p ≈ 1.5e-36, and both tests agree. **It is still not a seasonal finding.** "Winter" here
is the last days of December, "autumn" is October and November, and the two differ in the
level and volatility of the load as much as in the calendar. A difference that is real,
significant and confounded is the normal case rather than the exception, and the
significance test says only the first two of those three things.

This is where the significance layer earns its place and also where it stops. It tells
you the gap is not sampling noise. Deciding what the gap *is* needs somebody who knows
what happened on the grid in December.

The verdict is annotated in the figure's subtitle, so the gap and whether the gap is
distinguishable from noise cannot be read apart from each other. The sample count under
each tick is repo B's trick and worth keeping: a box drawn from sixty windows and one
drawn from a hundred and twenty-six look identical otherwise.

Welch's ANOVA is used rather than the standard one, and the difference is not cosmetic
here — seasons contribute unequal numbers of windows, so their error estimates have
different precision, which is exactly the assumption the standard ANOVA makes and breaks.
Kruskal-Wallis runs alongside because a bounded, skewed error distribution is not what
either ANOVA assumes, and `tests_agree` surfaces a disagreement rather than hiding it
behind one p-value.
"""
)

c(
    """
comparison = groups.analysis
print(comparison.summary())
print()
print("tests agree:", comparison.tests_agree)
print()
print(comparison.to_frame())

significant = comparison.significant_pairs
if significant:
    print()
    print("pairs that differ after Holm correction:")
    for pair in significant:
        print(" ", pair)
"""
)

# ==========================================================================
m(
    """
## 10. Is it better than doing something trivial?

The first question about any forecaster, and the one that most often goes unasked.
`NaiveLastValueHandler` holds the last observed value flat across the horizon;
`MovingAverageHandler` holds the mean of a recent window. Both are demanding baselines on
anything with autocorrelation, which is most operational data.

Both run under the **same `ForecastingConfig`** as the model, so the protocol cannot
silently differ between them — which is precisely what went wrong in repo B, where the
baseline refitted at every origin and the model did not.
"""
)

c(
    """
naive = ta.NaiveLastValueHandler(config, target=TARGET)
moving = ta.MovingAverageHandler(config, window=24, target=TARGET, name="moving_average_24h")

# The model's own assessment already ran in sections 2 and 6; only the baselines are new.
assessed = {"quantile_gbm": (accuracy, robustness)}
for h in (naive, moving):
    acc = ta.AccuracyEvaluator(h, criteria=criteria).run(data, actual)
    rob = ta.RobustnessEvaluator(
        h, perturbations=PERTURBATIONS, severities=(1, 3, 5), n_variants=1, seed=0
    ).assess(data, actual, accuracy_metric="mae", criteria=tolerance)
    assessed[h.name] = (acc, rob)

rows = []
reports = {}
for name, (acc, rob) in assessed.items():
    reports[name] = ta.build_report(model_name=name, accuracy=acc, robustness=rob)
    rows.append((name, acc.overall["mae"], acc.overall["r2"],
                 rob.scored["error_based_robustness"].aggregate))

print(f"{'model':24s} {'MAE':>8s} {'R2':>8s} {'err-robust':>12s}")
for name, mae, r2, rb in rows:
    print(f"{name:24s} {mae:8.3f} {r2:8.3f} {rb:12.3f}")
"""
)

m(
    """
### The result a tutorial would not have chosen, and the more instructive one

**The gradient-boosted quantile model loses to a 24-hour moving average.** MAE 4.76
against 2.25, R² 0.27 against 0.69. Seven fitted models, 1,528 features, a week of
history, beaten on accuracy by taking the mean of yesterday.

That is not a bug in the notebook and it has not been tuned away, because it is the
single most valuable thing in it. The first question about any forecaster is whether it
beats something trivial, it is the question that most often goes unasked, and here the
answer is no. Everything in sections 2 through 9 — the horizon curve, the sensitivity
map, the error model, the seasonal gap — was computed on a model that should not be
deployed, and **none of it said so**. Only the baseline did.

Now read the third column, where the ordering reverses. `error_based_robustness` is 0.50
for the model and 0.15 for the moving average: the model that is worse on clean data
degrades far less under a meter fault. Those two facts point at different deployment
decisions, and no scoring rule settles the trade — which is exactly why the comparison
layer refuses to collapse them into one number.
"""
)

c(
    """
models_comparison = ta.compare_models(list(reports.values()))
print(models_comparison.summary())
print()
for pair in models_comparison.disagreements():
    print(" ", pair)
"""
)

m(
    """
`disagreements()` is the output most worth reading: the pairs of metrics that prefer
different models. Where every metric agrees you did not need a comparison layer; where
they disagree there is a trade-off, and a trade-off is a decision for the deployment
rather than for a scoring rule.

Anything the comparison **refused** appears in `dropped` with its reason — a metric one
model has and another does not, two robustness scores over different underlying accuracy
metrics, or a sweep aggregate over different perturbation sets. Averaging over "whatever
each model reported" rewards the model that reported fewer of the hard metrics.
"""
)

c(
    """
if models_comparison.dropped:
    for metric, reason in sorted(models_comparison.dropped.items()):
        print(f"  {metric}: {reason}")
else:
    print("nothing dropped: all three models reported the same metrics over the same set")
"""
)

c(
    """
from themis_assess.viz import assess_models

bundle = assess_models(
    list(reports.values()),
    axes=["mae", "rmse", "r2", "error_based_robustness"],
    title="Model comparison (normalised; 1 is always better)",
)
bundle.figure
"""
)

m(
    """
Every value is normalised to [0, 1] with 1 always better, so an error metric and a
goodness-of-fit can share a chart without the worse model looking better. The raw values
stay on the hover, because a normalised score is not a number anyone can act on. Repo D's
ancestor of this chart plotted **raw** values on an axis fixed to [0, 1], which is correct
only while every metric is a bounded accuracy — an error metric drawn there puts the
better model closer to the centre.
"""
)

# ==========================================================================
m(
    """
## 11. The report — everything in one place, and what is wrong with it

One dictionary, with a declared schema and stable section names. Numbers are recorded
rather than rounded, because a rounded record cannot be checked against the run that
produced it, and non-finite values serialise as `null` rather than the `NaN` literal that
most JSON parsers reject.
"""
)

c(
    """
sections = {}
if fit.is_trustworthy:
    sections["attribution"] = {"window_summary": error_model}

report = ta.build_report(
    model_name="quantile_gbm",
    accuracy=grouped,
    robustness=robustness,
    sensitivity=[time_map, severity_map],
    groups={"by_season": comparison},
    **sections,
    context={
        "data": "ElectricityLoadDiagrams, 8 clients, hourly, 2013",
        "target": TARGET,
        "split": f"temporal, cutoff {split.metadata['cutoff']}, gap {HISTORY}",
        "protocol": f"{HORIZON}h horizon, {HISTORY}h history, stride {STRIDE}",
        "owner": "tutorial",
    },
)

payload = report.to_dict()
print("schema :", payload["schema"])
print("sections:", sorted(payload))
"""
)

c(
    """
for finding in report.worst_findings(6):
    print(f"[{finding.severity}] {finding.headline}")
    print(f"    {finding.detail}")
    print(f"    -> {finding.suggestion}")
    print()
"""
)

m(
    """
Findings are mechanical threshold comparisons on purpose. A mechanical finding is
auditable: every one names the numbers it came from, so a reader can disagree with it by
checking them. They are also the structure a summarising language model needs — handing a
model several hundred raw numbers gets a different answer every run, while handing it
findings leaves it the job it is actually good at.

### The report checks itself

Survey §8's ten reporting rules, as executable checks over the serialised dictionary —
because the dictionary is what gets handed to a renderer, a database or a language model,
so it is the thing whose completeness matters.

They **warn and never raise**. A library that refused here would stop an honest
exploratory run, and §8 is about what gets *published*, which this code cannot observe.
The issues join the model's own findings with `area="reporting"`, so ignoring one is a
decision rather than an accident.
"""
)

c(
    """
issues = ta.check_report(payload)
if issues:
    for issue in issues:
        print(f"[{issue.severity}] {issue.rule}")
        print(f"    {issue.headline}")
else:
    print("no reporting issues raised")
"""
)

c(
    """
print(ta.describe_metric("prediction_interval_coverage"))
print()
print(ta.describe_metric("error_based_robustness"))
"""
)

m(
    """
The glossary travels with the numbers, so a reader never has to go and look up whether
lower is better.
"""
)

c(
    """
from pathlib import Path

# The notebook may be run from the repository root or from docs/; both land here.
reports_dir = Path("docs/reports") if Path("docs").is_dir() else Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)
out = reports_dir / "forecasting_quantile_gbm.json"
report.save(out)
print("wrote", out, f"({out.stat().st_size / 1024:.0f} kB)")
"""
)

# ==========================================================================
m(
    """
## 12. A readable summary, for a named reader

Three parts, and only one of them involves a language model.

**(a) State what was measured** — templates, with the numbers substituted from each
finding's evidence. Nothing generated, so nothing can be invented.
**(b) Write it as a narrative** — the one job a small local model does well, and the only
part a model touches. Its output is verified and discarded on failure, so adding a model
can improve the summary or leave it unchanged, never make it wrong.
**(c) Recommend what to do** — retrieved from a YAML file that people write. A model may
select and order entries; it may never author one, because what to do about a weak model
is knowledge about a deployment, and a model asked to invent it produces fluent,
unfounded advice.

The audience shapes the templates and not only the prompt, which is what keeps the
model genuinely optional: a fallback that produced engineer prose for an operator is one
nobody would accept.
"""
)

c(
    """
summary = ta.summarise(report, audience="ml_engineer")
print(summary.to_markdown())
"""
)

c(
    """
print("narrative source:", summary.provenance["narrative_source"])
print()
operator = ta.summarise(report, audience="manager")
print(operator.to_markdown()[:1500])
"""
)

# ==========================================================================
m(
    """
## 13. Reproducing one cell

A sweep records a seed, and the point of recording one is that it reproduces the run. Each
`(perturbation, severity, variant)` cell derives its own generator from the run seed and
the cell's identity, so a cell's draw does not depend on how many cells ran before it —
which means re-running a single severity gives the same noise it had inside the full
sweep.

Repo B seeded numpy globally once per run, so it did not.
"""
)

c(
    """
from themis_assess.core.seeding import derive_generator

a = derive_generator(0, "sensor_drift", 3, 0).normal(size=4)
b = derive_generator(0, "sensor_drift", 3, 0).normal(size=4)
c_ = derive_generator(0, "sensor_drift", 3, 1).normal(size=4)
print("same cell, twice :", np.allclose(a, b))
print("different variant:", np.allclose(a, c_))
"""
)

c(
    """
one_cell = ta.RobustnessEvaluator(
    handler, perturbations=["sensor_drift"], severities=(3,), n_variants=1, seed=0
).run(data, actual)

full_value = robustness.sweep.curves("mae")["sensor_drift"][3]
cell_value = one_cell.curves("mae")["sensor_drift"][3]
print(f"inside the full sweep: {full_value:.6f}")
print(f"re-run on its own    : {cell_value:.6f}")
print("identical:", np.isclose(full_value, cell_value))
"""
)

# ==========================================================================
m(
    """
## What this told us, and what it did not

**What it told us.** That this forecaster should not be deployed, and four separate
reasons why — only one of which a conventional evaluation would have surfaced.

It misses its accuracy requirement, and almost all of that error is **bias** rather than
scatter (`mean_signed_deviation` -3.76 against MAE 4.76). Its 90% interval covers 85%, and
when a stretch of history goes missing that coverage collapses to 45% while the intervals
*narrow* — confidently wrong, which is the worst thing a probabilistic forecaster can be.
It leans on one client's own recent history far harder than on anything else, so the input
pipeline has a single point of failure. And it is **beaten on accuracy by a 24-hour moving
average**, which is the finding that should have come first and which nothing except the
baseline comparison could see.

It is also more robust to a meter fault than that moving average, which is a real argument
and not a consolation — and the comparison layer refuses to average the two into a verdict,
because that trade is a deployment decision.

**What it did not.** Several things, and they are worth stating because an assessment that
does not know its own limits is worse than none.

- **Synthetic robustness is not natural robustness.** Every perturbation here was
  manufactured. A meter that fails in the field fails in ways this list does not contain,
  and survey §6 is clear that passing a synthetic sweep does not transfer.
- **Eight clients is not a population.** The subgroup analysis is over two parts of one
  autumn within one client's forecast. A different client, a different tariff, a different
  year is a different question.
- **One year, one split, and only two seasons in the test period.** `temporal_split` gave
  a single cutoff in September, so the assessment never sees spring or summer at all, and
  the "seasonal" gap it does find is confounded with the level of demand in those
  particular months.
- **The horizon curve is confounded with the hour of day**, because every origin is issued
  at the same clock time. It is not purely a statement about lead time.
- **The thresholds are invented.** Every number in `AcceptanceCriteria` and
  `RobustnessCriteria` was chosen to make the notebook legible, not by anyone who owns a
  grid. That is the one input the library cannot supply and will not guess.
- **Label preservation is assumed, not checked.** The sweep perturbs the input and holds
  the truth fixed, which is correct — but at severity 5 a perturbation large enough to
  change what the right answer *is* would still be scored as though it had not.

### Where to go next

- [`tutorial_classification.ipynb`](tutorial_classification.ipynb) — the same tour for a classifier, with calibration,
  confusion matrices and per-class breakdowns.
- `docs/DECISIONS.md` — every design decision, D1–D68, each with the alternative rejected
  and why. The authority where it disagrees with anything else.
- `examples/assessment.yaml` — the criteria above, as a versioned file rather than a cell.
"""
)


def build() -> int:
    refuse_if_hand_edited(TARGET, CELLS, force="--force" in sys.argv)
    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    target = TARGET

    if "--no-execute" not in sys.argv:
        from nbclient import NotebookClient

        client = NotebookClient(
            notebook,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": str(HERE.parent)}},
        )
        client.execute()

    nbf.write(notebook, target)
    print(f"wrote {target} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
