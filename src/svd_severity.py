"""
src/svd_severity.py
===================
Core SVDSeverityAnalyzer class for the Geometric Biosecurity benchmark.

The analyzer fits a spectral threat axis on a reference panel of known
threat and benign protein sequences (embedded with ESM-2) and produces
a continuous severity score in [0, 1] for any new protein sequence.

Usage:
    from src.svd_severity import SVDSeverityAnalyzer
    import numpy as np

    # Fit on reference embeddings
    analyzer = SVDSeverityAnalyzer(n_components=64)
    analyzer.fit(E_ref, labels)   # labels: 1=threat, 0=benign

    # Score new sequences (after embedding with ESM-2)
    scores = analyzer.score(E_new)
    grades = analyzer.grade(scores)
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu
import pickle
from pathlib import Path


# Severity grade thresholds and labels
GRADE_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]
GRADE_LABELS = {
    0: "S0 (Minimal)",
    1: "S1 (Low)",
    2: "S2 (Moderate — review)",
    3: "S3 (High — expert review)",
    4: "S4 (Critical — block)",
}
GRADE_ACTIONS = {
    0: "auto-pass",
    1: "pass with logging",
    2: "soft flag",
    3: "expert review",
    4: "block + report",
}


class SVDSeverityAnalyzer:
    """
    Geometric threat severity analyzer using SVD of protein embeddings.

    Fits a threat axis as the unit vector from the benign class centroid
    to the threat class centroid in SVD-reduced embedding space. Severity
    is the min-max normalised projection onto this axis.

    Parameters
    ----------
    n_components : int
        Number of SVD components. Default 64.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd    = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, E: np.ndarray, labels: np.ndarray) -> "SVDSeverityAnalyzer":
        """
        Fit the analyzer on reference embeddings.

        Parameters
        ----------
        E : np.ndarray, shape (N, D)
            Protein embeddings (e.g. ESM-2 mean-pooled last layer).
        labels : np.ndarray, shape (N,)
            Binary labels: 1 = threat (toxin/virus), 0 = benign.

        Returns
        -------
        self
        """
        assert E.shape[0] == len(labels), "E and labels must have same length"
        assert set(np.unique(labels)) <= {0, 1}, "Labels must be binary (0/1)"

        Ez = self.scaler.fit_transform(E)
        Z  = self.svd.fit_transform(Ez)

        self.threat_centroid_ = Z[labels == 1].mean(axis=0)
        self.benign_centroid_ = Z[labels == 0].mean(axis=0)

        axis = self.threat_centroid_ - self.benign_centroid_
        self.axis_ = axis / (np.linalg.norm(axis) + 1e-8)

        proj = Z @ self.axis_
        self.proj_min_ = float(proj.min())
        self.proj_max_ = float(proj.max())

        # Store reference data for evaluation
        self.E_ref_    = E
        self.labels_   = labels
        self.Z_ref_    = Z
        self._fitted   = True

        return self

    def transform(self, E: np.ndarray) -> np.ndarray:
        """Project embeddings into SVD space."""
        self._check_fitted()
        Ez = self.scaler.transform(E)
        return self.svd.transform(Ez)

    def score(self, E: np.ndarray) -> np.ndarray:
        """
        Compute severity scores for new embeddings.

        Parameters
        ----------
        E : np.ndarray, shape (N, D)
            Protein embeddings.

        Returns
        -------
        np.ndarray, shape (N,)
            Severity scores in [0, 1]. Higher = more threat-proximate.
        """
        Z    = self.transform(E)
        proj = Z @ self.axis_
        return np.clip(
            (proj - self.proj_min_) / (self.proj_max_ - self.proj_min_ + 1e-8),
            0, 1
        )

    def grade(self, scores: np.ndarray) -> np.ndarray:
        """Convert severity scores to integer grades 0–4."""
        return np.digitize(scores, GRADE_THRESHOLDS)

    def grade_label(self, score: float) -> str:
        """Return human-readable grade label for a single score."""
        return GRADE_LABELS[int(np.digitize(score, GRADE_THRESHOLDS))]

    def grade_action(self, score: float) -> str:
        """Return recommended action for a single score."""
        return GRADE_ACTIONS[int(np.digitize(score, GRADE_THRESHOLDS))]

    # ── Evaluation metrics ────────────────────────────────────────

    def auroc(self) -> float:
        """AUROC on the reference panel. Reported for reference only (see AP)."""
        self._check_fitted()
        scores = self.score(self.E_ref_)
        return float(roc_auc_score(self.labels_, scores))

    def average_precision(self) -> float:
        """Average Precision on the reference panel (primary metric)."""
        self._check_fitted()
        scores = self.score(self.E_ref_)
        return float(average_precision_score(self.labels_, scores))

    def spectral_gap(self) -> float:
        """Ratio of first to second singular value (σ1/σ2)."""
        sv = self.svd.singular_values_
        return float(sv[0] / sv[1]) if len(sv) > 1 else float("inf")

    def variance_explained(self, n: int = 5) -> float:
        """Fraction of variance explained by top-n SVD components."""
        return float(self.svd.explained_variance_ratio_[:n].sum())

    def cohens_d(self, n_components: int = 10) -> np.ndarray:
        """
        Cohen's d per SVD component (threat - benign effect size).
        Values > 0.8 indicate large effect.
        """
        self._check_fitted()
        effects = []
        for k in range(min(n_components, self.n_components)):
            t_vals = self.Z_ref_[self.labels_ == 1, k]
            b_vals = self.Z_ref_[self.labels_ == 0, k]
            pooled = np.sqrt((t_vals.std()**2 + b_vals.std()**2) / 2) + 1e-8
            effects.append((t_vals.mean() - b_vals.mean()) / pooled)
        return np.array(effects)

    def mann_whitney_p(self) -> float:
        """Mann-Whitney U p-value (threat > benign severity)."""
        self._check_fitted()
        scores = self.score(self.E_ref_)
        _, p = mannwhitneyu(
            scores[self.labels_ == 1],
            scores[self.labels_ == 0],
            alternative="greater"
        )
        return float(p)

    def summary(self) -> dict:
        """Return a dictionary of all key metrics."""
        self._check_fitted()
        scores = self.score(self.E_ref_)
        t_scores = scores[self.labels_ == 1]
        b_scores = scores[self.labels_ == 0]
        return {
            "n_total":          int(len(self.labels_)),
            "n_threat":         int((self.labels_ == 1).sum()),
            "n_benign":         int((self.labels_ == 0).sum()),
            "n_svd_components": self.n_components,
            "auroc":            self.auroc(),
            "average_precision": self.average_precision(),
            "spectral_gap":     self.spectral_gap(),
            "variance_top5":    self.variance_explained(5),
            "variance_top10":   self.variance_explained(10),
            "threat_score_mean": float(t_scores.mean()),
            "threat_score_std":  float(t_scores.std()),
            "benign_score_mean": float(b_scores.mean()),
            "benign_score_std":  float(b_scores.std()),
            "mann_whitney_p":    self.mann_whitney_p(),
            "cohens_d_top10":   self.cohens_d(10).tolist(),
        }

    # ── Serialization ─────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save fitted analyzer to disk."""
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved analyzer to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SVDSeverityAnalyzer":
        """Load a fitted analyzer from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"Expected SVDSeverityAnalyzer, got {type(obj)}")
        return obj

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return (f"SVDSeverityAnalyzer(n_components={self.n_components}, "
                f"status={status})")
