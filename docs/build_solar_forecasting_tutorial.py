"""Build (and execute) the solar forecasting tutorial.

Generated rather than hand-edited, for the reason D57 gives: every cell is executed on
each rebuild, so the prose cannot drift from what the library actually returns. Run:

    python docs/build_solar_forecasting_tutorial.py

It will **refuse to run** if the notebook has been edited by hand, printing a diff of what
it would have overwritten (``docs/notebook_guard.py``). Prose belongs here, in a tracked
file, not in a generated artefact with no undo. Flags:

    --check        compare against the notebook and stop; write nothing
    --force        overwrite hand edits anyway, saying so
    --no-execute   write the cells without running them

Needs the ``[timeseries]`` extra (darts) plus ``xgboost``, and — unlike the other two
tutorials — one **manual download**: the Spanish weather file is not in the repository,
because ``data/`` is gitignored. See the first cell for where to get it.

This is *the* forecasting tutorial: it writes ``docs/forecasting_tutorial.ipynb``. The
electricity-load tutorial that used to hold that path was retired on 2026-11-19; its
builder, ``build_forecasting_tutorial.py``, is kept for its prose but must not be run
against that path any more.

The cells below were regenerated from the notebook on 2026-11-19, after the prose had
been rewritten by hand and the reporting section removed. They match it exactly —
``--check`` says so — which is what puts D57 back in force: the code here is executed on
every rebuild, so the notebook's outputs cannot drift from what the library returns.

Prose belongs here from now on. Editing the notebook instead is allowed and the guard
protects it, but every hand edit moves the two apart again until this file is brought
back into line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook_guard import refuse_if_hand_edited  # noqa: E402

HERE = Path(__file__).resolve().parent
TARGET = HERE / "forecasting_tutorial.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELLS: list = []


def m(text: str) -> None:
    CELLS.append(md(text.strip("\n")))


def c(text: str) -> None:
    CELLS.append(code(text.strip("\n")))


m(
    """
# Assessing a forecaster end to end

### Spanish solar generation · XGBoost with weather forecasts · `themis-assess`

The task is to forecast seven days of solar generation in Spain, using three days of history, and the
weather expected over those seven days.
"What happens when the weather forecast is wrong" is a robustness question with an answer someone
can act on.

| | |
|---|---|
| target | `generation solar`, hourly, Spain, 2015–2018 |
| forecast horizon | 168 h (7 days) |
| model input | 72 h (3 days) of generation, plus the weather over the whole window |
| model | `XGBModel` using the [Darts](https://unit8co.github.io/darts/generated_api/darts.html)  Python library, point forecast only |
| split | temporal — fit on 2015–2017, assessed on 2018 |

### Getting the data

The generation series can be acessed through the Darts library. The weather data
(`weather_features.csv`) can be downloaded from the [Kaggle energy and weather
dataset](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather). Put it in a `data/` folder. The weather is real history while we treat it as
a perfect forecast. This makes the model performance optimistic, and is worth remembering when reading the
accuracy numbers.

```bash
pip install -e ".[timeseries,tutorial]" xgboost
jupyter notebook "docs/forecasting_tutorial copy.ipynb"
```
"""
)

c(
    """
from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.datasets import EnergyDataset
from darts.models import XGBModel

import themis_assess as ta

import plotly.io as pio
pio.renderers.default = "notebook"

print("themis-assess", ta.__version__)

LAGS, HORIZON, STRIDE = 72, 168, 24      # 3 days in, 7 days out, a new forecast daily
WINDOW = LAGS + HORIZON
WEATHER = ["temp", "clouds", "humidity"]
FEATURES = ["solar", *WEATHER]
"""
)

# ==========================================================================
m(
    """
## 1. The data

Generation data from Darts, weather from the Kaggle file. For simplicity, we averaged the weather data over the five cities that are covered in the weather dataset. Both datasets have a resolution of one hour, and span the same four years, so they line up on one index.
"""
)

c(
    """
# The notebook runs from the repository root or from docs/; both land on the same file.
root = Path.cwd() if Path.cwd().joinpath("data").is_dir() else Path.cwd().parent
weather_file = root / "data" / "weather_features.csv"
if not weather_file.exists():
    raise FileNotFoundError(
        f"{weather_file} is missing. Download weather_features.csv from "
        "https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-"
        "prices-and-weather and place it in data/."
    )

solar = EnergyDataset().load().to_dataframe()["generation solar"].interpolate().bfill()

weather = pd.read_csv(weather_file)
weather["time"] = (
    pd.to_datetime(weather["dt_iso"], utc=True, format="mixed").dt.tz_localize(None)
)
weather["city_name"] = weather["city_name"].str.strip()
weather = weather.drop_duplicates(["time", "city_name"])   # the file has 3076 duplicates
weather["temp"] = weather["temp"] - 273.15                 # kelvin, in the raw file
by_city = weather.pivot(
    index="time", columns="city_name", values=["temp", "clouds_all", "humidity"]
)

observed = pd.DataFrame(
    {
        "solar": solar,
        "temp": by_city["temp"].mean(axis=1),
        "clouds": by_city["clouds_all"].mean(axis=1),
        "humidity": by_city["humidity"].mean(axis=1),
    },
    index=solar.index,
).interpolate().bfill().ffill()

print(observed.describe().T.to_string(float_format=lambda v: f"{v:9.1f}"))
"""
)

m(
    """
The solar generation runs from 0 to 5792 MW with a mean of 1433.
"""
)

m(
    """
### The split, before anything is fitted

`temporal_split`: Train before a cutoff, test after. The
`gap` matters; a test window reaching back across the boundary reads training data, so
it is set to the length of the history the model reads.
"""
)

c(
    """
times = observed.index
split = ta.temporal_split(times, test_fraction=0.25, gap=LAGS)[0]
train_times, test_times = times[split.train], times[split.test]

print(split)
print("train:", train_times[0], "->", train_times[-1], f"({len(train_times)} h)")
print("test :", test_times[0], "->", test_times[-1], f"({len(test_times)} h)")
print("dropped to the gap:", split.metadata["n_dropped_to_gap"])
"""
)

m(
    """
### The model, and what counts as a covariate

The weather is a **future** covariate: known over the forecast horizon. The calendar — hour of day, day of year — is future-known too, but it is *derived*
rather than measured. Historic data of solar generation is used in addition as information to the forecasting model. Note that we do not use historic information of the weather; just a choice made. TODO for an enduser: Improved performance if we include historic weather data?

The output of the forecasting model is set to 24 (`output_chunk_length=24` in Darts), and with autoregression (using previous predictions), we can forecast 7 days ahead (168 time steps). The alternative which is one direct model per horizon step, is 168 outputs over
a 1200-column design matrix. That would be more computer intensive and less user-friendly on this notebook.
"""
)

c(
    '''
def calendar(index: pd.DatetimeIndex) -> np.ndarray:
    """Time of day and time of year, as sine/cosine pairs. Known, so never perturbed."""
    hour, day = index.hour.to_numpy(), index.dayofyear.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25),
    ])


def covariates(index: pd.DatetimeIndex, weather_values: np.ndarray) -> TimeSeries:
    return TimeSeries.from_times_and_values(
        index, np.column_stack([weather_values, calendar(index)])
    )


target_train = TimeSeries.from_times_and_values(
    train_times, observed.loc[train_times, "solar"].to_numpy()
)
covariates_all = covariates(times, observed[WEATHER].to_numpy())

model = XGBModel(
    lags=LAGS,                        # 72 h of generation
    lags_future_covariates=(0, 24),   # the weather over the chunk being predicted
    output_chunk_length=24,
    n_estimators=120,
    max_depth=6,
    learning_rate=0.1,
    random_state=0,
)
model.fit(target_train, future_covariates=covariates_all)
print("fitted on", len(target_train), "hours")
'''
)

# ==========================================================================
m(
    """
## 2. The three objects the library needs

**An adapter** presents the data as `(entities, timesteps, features)`. For a forecaster
the consequential choice is what an *entity* is: In this case we let this be each **input window + forecast horizon** of the test period corresponding to every time interval the model based on three days of history makes 7 days of forecasts. With three days of input (72 steps) and seven days of forecast horizon (168) that makes the window length equal to 240. The forecasts are updated every day (stride = 24), which gives 353 such windows (and corresponding forecasts) in total. There are four features which gives the shape `(353, 240, 4)`.

**A handler** turns the model into predictions and declares what it can produce.
This XGBoost model is a `BLACK_BOX` model — a point forecast, no intervals. The planner will automatically say which metrics that rules out before anything runs.

**A `ForecastingConfig`** describes the evaluation protocol rather than the model: how
far ahead, how much history, how often a forecast is issued. One config, shared by the
model and every baseline model, so the protocol cannot silently differ between them.
"""
)

c(
    '''
from themis_assess.handlers.base import Prediction


class SolarForecaster(ta.ForecastingHandler):
    """A darts model whose covariates reach into the window it is predicting."""

    def __init__(self, forecaster, config, starts, **kwargs):
        self.forecaster = forecaster
        self.starts = pd.DatetimeIndex(starts)   # first timestamp of each entity's row
        super().__init__(
            self._needs_the_future, config, access=ta.AccessLevel.BLACK_BOX, **kwargs
        )

    @staticmethod
    def _needs_the_future(history, horizon):
        raise NotImplementedError(
            "This model reads the weather over the horizon, which (history, horizon) "
            "cannot carry. predict() and forecast_at() see the whole window."
        )

    def _stamps(self, data, entity):
        if data.n_entities != len(self.starts):
            raise ValueError(
                f"This handler holds {len(self.starts)} window start(s) but was given "
                f"{data.n_entities} entit(y/ies). Build a handler for the subset: the "
                f"starts are what the calendar covariates are derived from."
            )
        return pd.date_range(self.starts[entity], periods=data.n_timesteps, freq="h")

    def _pair(self, data, entity, origin):
        window, horizon = self.config.input_chunk_length, self.config.forecast_horizon
        if origin < window:
            raise ValueError(
                f"Entity {entity} offers {origin} step(s) of history where the model "
                f"needs {window}."
            )
        values, stamps = data.values(), self._stamps(data, entity)
        target = TimeSeries.from_times_and_values(
            stamps[origin - window:origin],
            values[entity, origin - window:origin, self._target_index(data)],
        )
        span = slice(origin - window, origin + horizon)
        columns = [data.feature_index(name) for name in WEATHER]
        return target, covariates(stamps[span], values[entity, span, :][:, columns])

    def _run(self, pairs):
        out = self.forecaster.predict(
            n=self.config.forecast_horizon,
            series=[t for t, _ in pairs],
            future_covariates=[c for _, c in pairs],
            show_warnings=False,
        )
        out = out if isinstance(out, list) else [out]
        return np.stack([np.asarray(o.values(), dtype=float).ravel() for o in out])

    def forecast_at(self, data, entity, origin):
        return self._run([self._pair(data, entity, origin)])[0]

    def predict(self, data):
        pairs = [
            self._pair(data, e, self._valid_span(data, e)[1] - self.config.forecast_horizon)
            for e in range(data.n_entities)
        ]
        return self.check_prediction(
            Prediction(
                predicted=self._run(pairs),
                metadata={"model": self.name, "forecast_horizon": HORIZON},
            ),
            data,
        )
'''
)

c(
    """
test_values = observed.loc[test_times, FEATURES].to_numpy()
starts = list(range(0, len(test_values) - WINDOW + 1, STRIDE))
windows = np.stack([test_values[s:s + WINDOW] for s in starts])
origins = pd.DatetimeIndex([test_times[s + LAGS] for s in starts])

data = ta.ArrayAdapter(
    windows,
    feature_names=FEATURES,
    categorical_features=[],
    entity_ids=[str(t) for t in origins],
)
config = ta.ForecastingConfig(
    forecast_horizon=HORIZON, input_chunk_length=LAGS, stride=STRIDE
)
handler = SolarForecaster(
    model, config, [test_times[s] for s in starts], name="xgboost_weather"
)

print(data)
print(f"{len(starts)} windows | first origin {origins[0]} | last {origins[-1]}")
print("access:", handler.access.name, "| task:", handler.task.name)
"""
)

# ==========================================================================
m(
    """
## 3. Accuracy, and the thresholds that turn numbers into a verdict

Running the metrics is the easy half; the half that decides anything is the **acceptance
criteria**. The library will not invent them. With no criteria the verdict is `None`, meaning nothing
was checked, which is not a pass.
"""
)

c(
    """
criteria = ta.AcceptanceCriteria(
    [
        ta.Requirement(
            "mae", acceptable=500.0, critical=True,
            rationale="a week-ahead error above 500 MW would change the reserve booked",
        ),
        ta.Requirement(
            "r2", acceptable=0.5, critical=True,
            rationale="below this the forecast explains less than half of what it tracks",
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
Everything skipped is an interval or a distribution metric, and the reason is the
handler's own declaration: a point forecast cannot be asked whether its 90% interval
covers 90% of the time. `plan()` says so before the run rather than halfway through it.
"""
)

c(
    """
prediction = handler.predict(data)
actual = handler.actual(data)

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
A pass and a fail, which is more useful than either alone. The error is inside the bar
someone set; the share of variation explained is not. `mean_signed_deviation` is the row
to read next to `mae`: it keeps its sign, so it says the model runs 11 MW low on average
— nearly unbiased, which a symmetric error metric cannot tell you.
"""
)

# ==========================================================================
m(
    """
## 4. Watching it forecast

A curve of error per origin says that *some* forecasts were bad. It never shows what bad
looked like. `plot_forecast_animation` plays the stack: one frame per origin, the
forecast against the truth it was scored on, with the three days of history it was issued
from behind it.

The y-range is fixed across frames on purpose — an axis that rescales per frame draws a
forecast that fell apart exactly like one that held.

First the rolling evaluation, which is the one place a **long-series** adapter is used:
`historical_forecasts` slides an origin along one continuous series, which is the shape
the time-axis analyses want.
"""
)

c(
    """
rolling = ta.ArrayAdapter(
    test_values[None, :, :], feature_names=FEATURES, categorical_features=[]
)
rolling_handler = SolarForecaster(
    model,
    ta.ForecastingConfig(
        forecast_horizon=HORIZON, input_chunk_length=LAGS, stride=STRIDE
    ),
    [test_times[0]],
    name="xgboost_rolling",
)

history = rolling_handler.historical_forecasts(rolling)
labels = [str(test_times[o])[:10] for o in history["origins"]]
print("forecasts:", history["forecasts"].shape, "| origins:", len(labels))
"""
)

c(
    """
from sklearn.metrics import mean_absolute_error

from themis_assess.viz import assess_accuracy_over_time, plot_forecast_animation

over_time = assess_accuracy_over_time(
    history["forecasts"], history["actual"], mean_absolute_error,
    origins=labels, metric="MAE", highlight_worst=3,
    title="Week-ahead MAE by forecast origin",
)
curve = over_time.analysis["curve"]

every = 14                                 # one frame a fortnight keeps the file small
frames = slice(None, None, every)
plot_forecast_animation(
    history["forecasts"][0][frames],
    history["actual"][0][frames],
    history=np.stack([test_values[o - LAGS:o, 0] for o in history["origins"]])[frames],
    frame_labels=labels[frames],
    annotations=[f"MAE {value:,.0f} MW" for value in curve[frames]],
    value_name="solar generation (MW)",
    title="A week ahead, fortnight by fortnight",
    height=480,
)
"""
)

m(
    """
The *shape* survives a week out — the forecast tracks the daily cycle closely in almost
every frame. What it misses is the **amplitude**.
"""
)

# ==========================================================================
m(
    """
## 5. When is it bad, and how far ahead is it usable?

Two curves over the same array: error by *origin*, and error by *position in the
horizon*.
"""
)

c(
    """
print(over_time.summary)
print()
for entry in over_time.analysis["worst"]:
    print(f"  {entry['origin']}   MAE {entry['value']:,.0f}")
over_time.figure
"""
)

c(
    """
from themis_assess.metrics.accuracy.temporal import accuracy_by_horizon, aggregate_over_time

by_horizon = accuracy_by_horizon(
    history["forecasts"], history["actual"], mean_absolute_error
)
for day in range(7):
    block = by_horizon[day * 24:(day + 1) * 24]
    print(f"  day {day + 1}:  MAE {block.mean():6.0f}")
print()
for method in ("mean", "median", "p90", "max"):
    print(f"  {method:7s} over origins: {aggregate_over_time(curve, method=method):6.0f}")
"""
)

m(
    """
The error curve showing how MAE changes by time shows a quite stable pattern centered around MAE 450, and as many peaks as valleys.

When looking at the mean MAE per forecast step, we see that the forecast model in general is more inaccurate the further it looks in the future, intutively as the uncertainty increases the further we look ahead. 
"""
)

# ==========================================================================
m(
    """
## 6. Robustness — when the inputs misbehave

Everything so far assumed the inputs arrive intact. Each perturbation below names a real
failure: a noisy weather forecast, a drifting sensor, one stuck at a constant reading, an
outage that removes a stretch of the window, interference spikes. Magnitudes are
**relative** — each severity scales the feature's own standard deviation — so severity 3
means something comparable across generation in MW and cloud cover in per cent.

The sweep perturbs every input column, weather and generation alike. 
"""
)

c(
    """
PERTURBATIONS = [
    "gaussian_noise",     # the weather forecast is wrong, everywhere, a little
    "sensor_drift",       # slow calibration loss
    "flat_sensor",        # a meter stuck at one value
    "missing_segment",    # an outage: NaN, which XGBoost takes natively
    "outlier_spikes",     # interference
]

sweep_evaluator = ta.RobustnessEvaluator(
    handler, perturbations=PERTURBATIONS, severities=(1, 2, 3, 4, 5), n_variants=1, seed=0
)
sweep_plan = sweep_evaluator.plan(data, actual)
print("perturbations:", sweep_plan.perturbations)
print("model calls  :", sweep_plan.n_model_calls)
"""
)

m(
    """
`RobustnessCriteria` is the counterpart of `AcceptanceCriteria`. We use the error-based robustness metric which compare difference in MAE from the clean MAE value across perturbations of different severities. The basis for the evaluation is "worst perturbation" which means we evaluate based on the type of perturbation that gave the worst increase in error (the error-based robustness metric for one perturbation averages over severities). See more details below.
"""
)

c(
    """
tolerance = ta.RobustnessCriteria(
    [
        ta.ToleranceRequirement(
            "error_based_robustness", acceptable=0.5, basis="worst_perturbation",
            rationale="no single fault may more than double the week-ahead error",
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

severity = assess_severity(robustness.sweep, "mae", title="Week-ahead MAE under shift")
print(severity.summary)
severity.figure
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
The clean value is drawn as a reference line (striped horizontal line).

### The basis is part of the requirement

`error_based_robustness` is Eq. 14: for **one perturbation**, the *mean over the severity
grid* of Eq. 13's `E_clean / E_perturbed`. For `missing_segment` those five ratios are
0.85, 0.75, 0.52, 0.44 and 0.33, and their mean is the 0.58 the verdict reports. The 0.33
is severity 5 alone.

Which means the three bases are three different questions, and the same scored result
answers them differently:
"""
)

c(
    """
scored = robustness.scored["error_based_robustness"]
print(f"{'basis':20s}{'value':>8s}   verdict")
for basis in ("aggregate", "worst_perturbation", "worst_cell"):
    verdict = ta.RobustnessCriteria(
        [ta.ToleranceRequirement("error_based_robustness", acceptable=0.5, basis=basis)]
    ).verdict(scored)
    outcome = "PASS" if verdict.passed else "FAIL"
    print(f"{basis:20s}{verdict.value:8.3f}   {outcome}   {verdict.worst_at or ''}")
"""
)

m(
    """
- **aggregate** is Eq. 15's *product* across the five perturbations, so it falls as the
  set grows,
- **worst perturbation** is the minimum of the per-perturbation means. The library's
  default, and the one the tolerance above names.
- **worst cell** is the single worst `(perturbation, severity)` in the grid, as a ratio to
  clean: `445 / 1360 = 0.33`. The strictest, and usually dominated by severity 5 — often a
  severity the deployment will never see, which is why it is not the default.

Three numbers, one of them a pass and two of them failures, from one sweep. A tolerance
that does not say which reading it means has not been stated.
"""
)

# ==========================================================================
m(
    """
## 7. How the perturbations change the forecasts by severity

The same animation builder, with severity on the frame axis instead of time. One window —
the one the model already does worst on — forecast again at each severity of a drift
applied to **cloud cover alone**: the weather forecast degrading, with nothing else
touched.
"""
)

c(
    """
from themis_assess.core.seeding import derive_generator
from themis_assess.core.types import Scope

worst_window = int(np.argmax(np.abs(prediction.predicted - actual).mean(axis=1)))
one = data.take([worst_window])
one_handler = SolarForecaster(
    model, config, [test_times[starts[worst_window]]], name="xgboost_one_window"
)

drift = ta.default_perturbations()["sensor_drift"]
stack, frame_labels = [one_handler.predict(one).predicted[0]], ["clean"]
for level in (1, 2, 3, 4, 5):
    spec = ta.PerturbationSpec(
        name="sensor_drift", severity=level, scope=Scope.NAMED, features=("clouds",)
    )
    drifted = drift.perturb(one, spec, derive_generator(0, "sensor_drift", level, 0))[0]
    stack.append(one_handler.predict(drifted).predicted[0])
    frame_labels.append(str(level))

errors = [np.abs(f - actual[worst_window]).mean() for f in stack]
plot_forecast_animation(
    np.stack(stack), actual[worst_window],
    history=test_values[starts[worst_window]:starts[worst_window] + LAGS, 0],
    frame_labels=frame_labels, frame_name="severity",
    annotations=[f"cloud-cover drift · MAE {e:,.0f} MW" for e in errors],
    value_name="solar generation (MW)",
    title=f"One window ({origins[worst_window]:%Y-%m-%d}) as the weather forecast drifts",
    height=480,
)
"""
)

m(
    """
The daytime peaks barely move — the calendar covariates are untouched, so the diurnal
shape holds. This shows that the model is quite robust in this case with respect to this particular covariate (clouds) and this particular perturbation (sensor drift).
"""
)

# ==========================================================================
m(
    """
## 8. Which input, and where in the window?

Two maps from one `SensitivityMapper`. The first is feature × position-in-window, which
is the question a windowed forecaster raises and a feature ranking cannot answer: *when*
in the three days does a corrupted reading matter?

**Prediction shift is not degradation.** This perturbs one cell and measures how far the
*forecast* moves, not how much accuracy is lost. That needs no labels, so it can be run on
production data — and a model can move a lot and still be right, or barely move and be
confidently wrong. `measure="degradation"` asks the other question, and the result records
which was asked.
"""
)

c(
    """
from themis_assess.viz import plot_sensitivity_map

mapper = ta.SensitivityMapper(
    handler, perturbation="gaussian_noise", measure="prediction_shift",
    n_variants=1, seed=0,
)
time_map = mapper.feature_time_map(data, severity=3, time_granularity=24, window=WINDOW)
print("hottest cell:", time_map.hottest_cell())
plot_sensitivity_map(time_map, title="Prediction shift by feature × day of the window")
"""
)

m(
    """
Columns `t0-23` to `t48-71` are the three days of history; the rest is the horizon. Three
things are legible in one picture and in no feature ranking:

- Generation matters **only in the history**, and overwhelmingly in the most recent day —
  four times the day before it.
- The weather matters **only over the horizon**. The weather in the history columns scores
  exactly zero, because `lags_future_covariates=(0, 24)` tells the model to read covariates
  from the origin forward: three days of weather are sitting in the window unused.
- Over the horizon the influence decays with lead time, and humidity leads temperature and
  cloud cover throughout.

The second of those is a specification made visible. It is also the kind of thing worth
checking before trusting a sensitivity result: a feature that cannot move the model is not
evidence that the feature does not matter.
"""
)

c(
    """
severity_map = mapper.feature_severity_map(data)
print(severity_map.to_frame().round(1).to_string())
print()
threshold = float(np.median(severity_map.grid)) * 2
print(f"tipping point at a shift of {threshold:,.0f} MW:")
for name in severity_map.row_labels:
    print(f"  {name:9s} {severity_map.tipping_point(name, threshold)}")
"""
)

m(
    """
`tipping_point` is what the severity axis is for: the first severity at which a feature
pushes the forecast past a threshold someone cares about. `None` is a finding, not a gap —
it says the model tolerated everything that was tried.
"""
)

# ==========================================================================
m(
    """
## 9. What do the windows it gets wrong have in common?

A surrogate is fitted to predict **this model's error** from summaries of its input
window. If the surrogate learns nothing its attributions are indistinguishable from noise,
and the output looks identical either way — so `is_trustworthy` gates them, and overriding
the gate takes `force=True`.
"""
)

c(
    """
from themis_assess.attribution.features import WindowSummaryFeatures

error_model = ta.ErrorModel(extractor=WindowSummaryFeatures(), seed=0)
fit = error_model.fit(data, actual, prediction.predicted)
print(fit.describe())
print()
print("gate:", fit.gate, f"= {fit.gate_value:.3f}", "| trustworthy:", fit.is_trustworthy)
"""
)

c(
    """
if fit.is_trustworthy:
    from themis_assess.viz import plot_attribution

    attribution = error_model.importance(method="permutation", n_repeats=10, on="test")
    for entry in attribution.top(6):
        print(f"  {entry['feature']:28s} {entry['importance']:.4f}")
    display(plot_attribution(attribution, title="What goes with a larger week-ahead error"))
else:
    print("The surrogate did not learn the error, so no attribution is offered. That")
    print("refusal is the result.")
"""
)

m(
    """
The top feature is `solar__flat_fraction` — how much of the window sits flat, which for
solar means night — followed by the window's variability. Read together: **the error
tracks how much generation there was to get wrong**. Long bright days are where the MW go
missing.
"""
)

# ==========================================================================
m(
    """
## 10. Does it work all year, and does it beat something trivial?

A *group* is any named partition. For a forecaster the natural ones are temporal, and
groups are **masks** — which is what lets the same machinery serve a calendar season here
and a clinical subgroup elsewhere.

The descriptive half is the gap. The inferential half — whether the gap is distinguishable
from sampling variation — is a different question, and the gap alone invites getting it
wrong.
"""
)

c(
    """
from themis_assess.viz import assess_groups

season_of = np.where(
    origins.month.isin([12, 1, 2]), "winter",
    np.where(origins.month.isin([3, 4, 5]), "spring",
             np.where(origins.month.isin([6, 7, 8]), "summer", "autumn")))

subgroups = [
    ta.SubgroupSpec(
        name=label,
        mask_fn=(lambda chosen: (lambda _: season_of == chosen))(label),
        description=f"forecast origins in {label}",
    )
    for label in sorted(set(season_of))
]
grouped = ta.AccuracyEvaluator(handler, criteria=criteria, subgroups=subgroups).run(
    data, actual
)
for name, result in sorted(grouped.subgroups.items()):
    print(f"  {name:7s} n={result.n_samples:4d}  MAE {result['mae']:6.0f}")
print()
print("worst subgroup on mae:", grouped.worst_subgroup("mae", higher_is_better=False))

groups = assess_groups(
    np.abs(prediction.predicted - actual).mean(axis=1), season_of,
    metric="per-window MAE", title="Week-ahead error by season",
)
print()
print("tests agree:", groups.analysis.tests_agree)
groups.figure
"""
)

m(
    """
During Winter, the forecasts are most accurate, while during spring the forecasts are least accurate, by 200 MW. Statisticals tests agree that the gap is significant.
The Kruskall-Wallis test gives a p-value of 5.7e-17 which means the null hypothesis of no group difference can be rejected by any reasonable significance level.

### The baselines

A great question is whether the XGBoost forecast model beats simple baselines. Here we consider three baselines: "naive last value": Predict the same value as the last one the next 7 days, "moving_average_24": Take the average from the last 24 hours, and predict the same for the future, "yesterday_repeated": A model that predicts the same curve as what happened yesterday for seven days in a row. All three baseline models was applied
under the **same `ForecastingConfig`** as the XGBoost model, so the protocol cannot differ between
them.

Two of them need a small subclass, and the reason is worth keeping: `missing_segment` at
severity 5 can take the whole window, and the library refuses to let a model skip a
perturbation it cannot survive — a robustness score over only the perturbations a model
tolerates overstates its robustness. XGBoost takes NaN natively; the library's baselines
raise an error. So they are given the same tolerance rather than the perturbation being dropped.
"""
)

c(
    '''
class _ToleratesMissing:
    """A window the outage emptied is a forecasting problem, not a crash."""

    def _level(self, history):
        finite = history[np.isfinite(history)]
        return 0.0 if finite.size == 0 else super()._level(finite)


class LastValueHandler(_ToleratesMissing, ta.NaiveLastValueHandler):
    pass


class MovingAverageHandler(_ToleratesMissing, ta.MovingAverageHandler):
    pass


class DailyProfileHandler(ta.ForecastingHandler):
    """Repeat the last observed day — the baseline anything diurnal has to beat."""

    def __init__(self, config, **kwargs):
        super().__init__(self._repeat, config, **kwargs)

    @staticmethod
    def _repeat(history, horizon):
        day = np.asarray(history[-24:, 0], dtype=float)
        if not np.isfinite(day).any():
            return np.zeros(horizon)
        return np.resize(np.where(np.isfinite(day), day, np.nanmean(day)), horizon)


baselines = [
    LastValueHandler(config, target="solar"),
    MovingAverageHandler(config, window=24, target="solar", name="moving_average_24h"),
    DailyProfileHandler(config, target="solar", name="yesterday_repeated"),
]

assessed = {"xgboost_weather": (accuracy, robustness)}
for baseline in baselines:
    assessed[baseline.name] = (
        ta.AccuracyEvaluator(baseline, criteria=criteria).run(data, actual),
        ta.RobustnessEvaluator(
            baseline, perturbations=PERTURBATIONS, severities=(1, 2, 3, 4, 5),
            n_variants=1, seed=0,
        ).assess(data, actual, accuracy_metric="mae", criteria=tolerance),
    )

reports = {}
print(f"{'model':22s}{'MAE':>9s}{'R2':>9s}{'err-robust':>12s}")
for name, (acc, rob) in assessed.items():
    reports[name] = ta.build_report(model_name=name, accuracy=acc, robustness=rob)
    print(f"{name:22s}{acc.overall['mae']:9.0f}{acc.overall['r2']:9.2f}"
          f"{rob.scored['error_based_robustness'].aggregate:12.3f}")
'''
)

m(
    """
The XGBoost model is a third better than the best trivial forecast — repeating yesterday — and
roughly four times better than holding the last value flat.

**However, the XGBoost model is on third place with respect to robustness, beaten by naive_last_value and moving_average_24h.** That is not a paradox: a forecast that ignores
its inputs cannot be hurt by corrupting them. The naive baselines score well on
`error_based_robustness` because they were never using the information that the
perturbation destroys. A robustness score is only meaningful beside the accuracy it is
protecting, which is exactly what the comparison layer refuses to collapse into one
number.
"""
)

c(
    """
from themis_assess.viz import assess_models


bundle = assess_models(
    list(reports.values()),
    axes=["mae", "rmse", "r2", "error_based_robustness"],
    scales={"mae": 500.0, "rmse": 800.0},
    title="Model comparison (normalised; 1 is always better)",
)
print(bundle.summary)
print()
print("dropped:", bundle.analysis.dropped or "nothing")
bundle.figure
"""
)

m(
    """
The assess_models() ranks each model based on the seven chosen metrics. The axes argument gives what kind of metrics to show in the polar chart. The scales argument explains for each metric how to normalize the metric to be a number between 0 and 1. `disagreements()` is the output worth reading: every pair of metrics that prefers a
different model. Where every metric agrees you did not need a comparison layer; where they
disagree there is a trade-off, and a trade-off is a decision for the deployment rather than
for a scoring rule.
"""
)

# ==========================================================================
m(
    """
## What this told us, and what it did not

**What it told us.** A week-ahead solar forecaster that is comfortably better than any
trivial alternative, with the rest of the picture attached: the error grows by a quarter
from day 1 to day 7 and then flattens; the worst origins are nearly twice the mean; spring
is served 200 MW worse than winter and the gap is real; the model leans on the most recent
day of generation and on humidity and cloud cover over the horizon; the windows it gets
wrong are the ones with the most generation to get wrong; and a bad weather forecast costs
it more than it costs any of the baselines, because the baselines were not using the
weather at all.

**What it did not.**

- **The weather was perfect.** Real history stood in for a forecast, so every accuracy
  number here is an upper bound. The honest version of this assessment replaces the
  covariates with archived *forecasts* at the matching lead times.
- **Synthetic robustness is not natural robustness.** Every perturbation was manufactured;
  survey §6 is clear that passing a synthetic sweep does not transfer to a real shift.
- **One split, one year of test data.** A model that happened to train through three mild
  springs has not been asked about a cloudy one.
- **Five cities averaged into one weather series.** Spain's solar capacity is not
  distributed like its cities, and the spatial detail that a real operator uses was thrown
  away in the first cell.
- **The thresholds are invented.** Every number in `AcceptanceCriteria` and
  `RobustnessCriteria` was chosen to make the notebook legible. That is the one input the
  library cannot supply and will not guess.
- **Label preservation is assumed, not checked.** The sweep perturbs the input and holds
  the truth fixed, which is correct — but a severity large enough to change what the right
  answer *is* would still be scored as though it had not.

### Where to go next

- [`tutorial.ipynb`](tutorial.ipynb) — the same tour for a classifier, with calibration,
  confusion matrices and per-class breakdowns.
- `docs/DECISIONS.md` — every design decision, each with the alternative rejected and why.
- `examples/assessment.yaml` — the criteria above, as a versioned file rather than a cell.
"""
)


def build() -> int:
    # Before anything else, and before the four minutes of execution: never silently
    # overwrite prose someone improved in the notebook itself.
    refuse_if_hand_edited(TARGET, CELLS, force="--force" in sys.argv)
    if "--check" in sys.argv:
        return 0

    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }

    if "--no-execute" not in sys.argv:
        from nbclient import NotebookClient

        client = NotebookClient(
            notebook,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": str(HERE.parent)}},
        )
        client.execute()

    nbf.write(notebook, TARGET)
    print(f"wrote {TARGET} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
