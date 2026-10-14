"""Re-export of :mod:`themis_assess.metrics.calibration`, kept for its survey provenance.

Calibration moved up a level, to ``metrics/calibration.py``, because it belongs to both
halves of this library and filing it under ``robustness/`` said otherwise. Whether a
model's stated confidence is honest is an accuracy property of its uncertainty, measurable
on a single clean dataset with no shift in sight; only the *change* in that honesty under
shift is a robustness question. The accuracy runner needs ECE and the Brier score as
ordinary metrics, and importing them from a robustness package to do so was a smell.

This module stays because the survey numbers these equations §5.4, within its metrics
chapter, and a reader following the survey's structure will look for them here.
"""

from themis_assess.metrics.calibration import *  # noqa: F403
from themis_assess.metrics.calibration import __all__  # noqa: F401
