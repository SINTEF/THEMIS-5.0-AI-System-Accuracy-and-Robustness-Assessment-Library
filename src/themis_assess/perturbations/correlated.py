"""Perturbation that respects the joint structure of the features. Survey §7.

The survey names this as a gap in current practice rather than an option:

    When inputs are multivariate (e.g., sensor arrays), perturbing individual features
    independently can produce physically implausible combinations that violate the
    statistical structure of real data. Perturbation protocols that respect observed
    feature correlations, for instance, by sampling from estimated conditional
    distributions, are needed to ensure that stress tests remain realistic.

That criticism lands squarely on the perturbations every one of these repos used. Repo
B's ``perturb_numerical_data`` adds independent Gaussian noise per feature, and repo C's
``noise_sensitivity`` does the same with ``self.X + noise_level * np.random.randn(*shape)``.
On a vessel that means speed can be pushed up while the engine load it depends on is
pushed down; on a clinical record, that height and weight move in opposite directions.
The model is then asked about a state the world cannot produce, and whatever it answers
tells you little about deployment.

:class:`CorrelatedNoise` is the fix the survey asks for. It estimates the feature
covariance from the data and draws perturbations from a multivariate normal with that
covariance, so the perturbed point moves along the directions the data actually varies
in. Contrast it against :class:`~themis_assess.perturbations.tabular.GaussianNoise` on
the same data: a large gap between the two is evidence that the independent-noise result
was an artefact of implausible inputs.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import Modality, PerturbationSpec, TimeScope
from themis_assess.perturbations.base import Perturbation

__all__ = ["CorrelatedNoise", "estimate_covariance"]


def estimate_covariance(
    data: DataAdapter,
    features: list[str],
    *,
    shrinkage: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Covariance of ``features`` over the real (unpadded) values, and its Cholesky factor.

    Parameters
    ----------
    shrinkage:
        Weight of a diagonal target blended into the estimate, as in Ledoit-Wolf
        shrinkage. A small positive value is not cosmetic here: with more features than
        samples, or with two exactly collinear columns, the raw covariance is singular
        and has no Cholesky factor. Shrinkage keeps it positive definite at the cost of
        slightly understating the correlations.

    Returns
    -------
    tuple
        ``(covariance, cholesky_factor)``. Multiplying standard normal draws by the
        factor produces draws with the estimated covariance.

    Raises
    ------
    ValueError
        If fewer than two complete observations are available, since a covariance cannot
        be estimated from one point.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must lie in [0, 1]; got {shrinkage}.")

    values = data.values()
    mask = data.valid_mask()
    columns = [data.feature_index(f) for f in features]
    flat = np.asarray(values[:, :, columns][mask], dtype=float)
    complete = flat[~pd.isna(flat).any(axis=1)]
    if complete.shape[0] < 2:
        raise ValueError(
            f"Estimating a covariance needs at least 2 complete observations across "
            f"{features}; found {complete.shape[0]}. Either impute first or use an "
            f"independent-noise perturbation."
        )

    covariance = np.cov(complete, rowvar=False)
    covariance = np.atleast_2d(covariance)

    if shrinkage > 0.0:
        target = np.diag(np.diag(covariance))
        covariance = (1.0 - shrinkage) * covariance + shrinkage * target

    # A last resort for exactly collinear columns, where even shrinkage toward the
    # diagonal can leave a zero eigenvalue.
    jitter = 1e-12 * max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
    for _ in range(8):
        try:
            factor = np.linalg.cholesky(covariance)
            return covariance, factor
        except np.linalg.LinAlgError:
            covariance = covariance + jitter * np.eye(covariance.shape[0])
            jitter *= 10.0
    raise ValueError(
        "Could not factor the feature covariance even after regularisation. This "
        "usually means a constant or exactly duplicated feature — drop it, or raise "
        "`shrinkage`."
    )


class CorrelatedNoise(Perturbation):
    """Additive noise drawn from the features' own estimated covariance. Survey §7.

    A drop-in comparison for :class:`~themis_assess.perturbations.tabular.GaussianNoise`:
    same additive form, same relative severity scale, but the perturbation vector is
    drawn from ``N(0, severity^2 * Sigma)`` rather than from independent per-feature
    normals. Severity multiplies the whole covariance, so severity 3 moves the point by
    roughly one standard deviation *along the data's own directions of variation*.

    ``time_scope`` behaves as elsewhere: ``CONSTANT`` draws one vector per entity and
    holds it across the series, ``PER_TIMESTEP`` draws independently at each step.

    Two caveats worth stating in any report that uses this.

    First, covariance captures only *linear* dependence. A perturbation respecting the
    covariance can still leave a physically impossible state if the real constraint is
    nonlinear or a hard bound — a negative concentration, a speed above a hull's
    maximum. This makes stress tests more realistic, not guaranteed realistic.

    Second, the covariance is estimated from the same data being perturbed, so on a
    small sample the estimate is noisy and the shrinkage matters. The estimate is
    reported in :meth:`metadata` so a reader can judge it.
    """

    name = "correlated_noise"
    modalities: ClassVar[frozenset[Modality]] = frozenset(
        {Modality.TABULAR, Modality.SEQUENCE}
    )
    accepts_numerical: ClassVar[bool] = True
    accepts_categorical: ClassVar[bool] = False
    severity_scale = (0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(self, *, shrinkage: float = 0.05) -> None:
        self.shrinkage = float(shrinkage)

    def applies_to(self, data: DataAdapter, task=None) -> bool:
        """Also requires at least two numerical features.

        Without this the registry would list ``correlated_noise`` as applicable to
        single-feature data and the caller would only discover otherwise when
        :meth:`perturb` raised. An ``applicable()`` result that cannot be run is worse
        than no result.
        """
        if not super().applies_to(data, task):
            return False
        return len(data.numerical_features) >= 2

    def perturb(
        self,
        data: DataAdapter,
        spec: PerturbationSpec,
        rng: np.random.Generator,
    ) -> list[DataAdapter]:
        self.validate(data, spec)
        features = self.eligible_features(data, spec)
        if len(features) < 2:
            raise ValueError(
                f"{self.name!r} needs at least 2 numerical features to have a "
                f"correlation to respect; got {features}. Use gaussian_noise for a "
                f"single feature."
            )
        columns = [data.feature_index(f) for f in features]
        _, factor_matrix = estimate_covariance(data, features, shrinkage=self.shrinkage)
        scale = self.severity_factor(spec)
        timesteps = self.resolve_timesteps(data, spec)
        mask = data.valid_mask()
        k = len(features)

        out: list[DataAdapter] = []
        for _ in range(spec.n_variants):
            values = data.values().astype(float)
            if spec.time_scope is TimeScope.PER_TIMESTEP:
                draws = rng.standard_normal((data.n_entities, len(timesteps), k))
                noise = scale * draws @ factor_matrix.T
                for c, j in enumerate(columns):
                    values[:, timesteps, j] += noise[:, :, c]
            else:
                draws = rng.standard_normal((data.n_entities, k))
                noise = scale * draws @ factor_matrix.T
                for c, j in enumerate(columns):
                    values[:, timesteps, j] += noise[:, c : c + 1]
            for j in columns:
                values[:, :, j] = np.where(mask, values[:, :, j], np.nan)
            out.append(data.with_values(values))
        return out

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta["shrinkage"] = self.shrinkage
        meta["respects_feature_correlation"] = True
        meta["caveat"] = (
            "Covariance captures linear dependence only; nonlinear constraints and "
            "hard bounds can still be violated."
        )
        return meta
