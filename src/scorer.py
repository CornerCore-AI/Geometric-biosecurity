"""
src/scorer.py
=============
Production scorer — clean API for scoring individual protein sequences.

Usage:
    from src.scorer import GeoBioScorer

    scorer = GeoBioScorer.from_pretrained()
    result = scorer.score("CNCKAPETALCARRCQQH")
    print(result)
    # {
    #   'severity': 0.847,
    #   'grade': 'S4 (Critical)',
    #   'action': 'block + report',
    #   'classification': 'threat',
    #   'confidence': 0.694,
    #   'seq_length': 18
    # }

    # Batch scoring
    results = scorer.score_batch([seq1, seq2, seq3])
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional

from src.svd_severity import SVDSeverityAnalyzer, GRADE_LABELS, GRADE_ACTIONS

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


class GeoBioScorer:
    """
    Production protein threat severity scorer.

    Wraps SVDSeverityAnalyzer and ESM-2 embedder into a single
    callable interface for scoring arbitrary protein sequences.

    Parameters
    ----------
    analyzer : SVDSeverityAnalyzer
        Fitted analyzer.
    embedder : ProteinEmbedder
        Configured embedder (model must match the one used to fit analyzer).
    """

    def __init__(self, analyzer: SVDSeverityAnalyzer, embedder):
        self.analyzer = analyzer
        self.embedder = embedder

    @classmethod
    def from_pretrained(
        cls,
        analyzer_path: Optional[str | Path] = None,
        model_size: str = "150M",
        cache_dir: str | Path = "poc_cache",
    ) -> "GeoBioScorer":
        """
        Load a pre-fitted scorer.

        If analyzer_path is None, fits a new analyzer on the standard
        dataset (requires poc_cache/dataset_full.json and embeddings).

        Parameters
        ----------
        analyzer_path : str or Path, optional
            Path to a saved SVDSeverityAnalyzer (.pkl).
        model_size : str
            ESM-2 model size ("8M", "35M", "150M", "650M").
        cache_dir : str or Path
            Directory containing cached embeddings.
        """
        from src.embedder import ProteinEmbedder

        embedder = ProteinEmbedder(
            model_size=model_size,
            cache_dir=cache_dir,
        )

        if analyzer_path and Path(analyzer_path).exists():
            analyzer = SVDSeverityAnalyzer.load(analyzer_path)
            print(f"Loaded analyzer from {analyzer_path}")
        else:
            print("Fitting analyzer from standard dataset cache...")
            analyzer = cls._fit_from_cache(cache_dir, model_size)

        return cls(analyzer, embedder)

    @staticmethod
    def _fit_from_cache(cache_dir: str | Path, model_size: str) -> SVDSeverityAnalyzer:
        """Fit analyzer from cached embeddings."""
        import json
        cache_dir = Path(cache_dir)

        # Load dataset
        dataset_path = cache_dir / "dataset_full.json"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Run data_fetcher.py to download."
            )
        data = json.loads(dataset_path.read_text())
        THREAT_LABELS = {"toxin", "virus"}
        labels = np.array([1 if r["label"] in THREAT_LABELS else 0 for r in data])

        # Find embedding cache
        model_name = f"esm2_t{'6' if model_size=='8M' else '12' if model_size=='35M' else '30' if model_size=='150M' else '33'}_{model_size}_UR50D"
        candidates = list(cache_dir.glob(f"embeddings_{model_name}_*.npy"))
        if not candidates:
            raise FileNotFoundError(
                f"No embedding cache found for {model_name}. "
                "Run experiments/exp1_benchmark.py to generate."
            )
        emb_path = max(candidates, key=lambda p: p.stat().st_size)
        E = np.load(emb_path)
        assert E.shape[0] == len(data), f"Shape mismatch: {E.shape[0]} vs {len(data)}"

        analyzer = SVDSeverityAnalyzer(n_components=64)
        analyzer.fit(E, labels)
        print(f"Fitted analyzer: AUROC={analyzer.auroc():.4f}, AP={analyzer.average_precision():.4f}")
        return analyzer

    def score(self, sequence: str) -> dict:
        """
        Score a single protein sequence.

        Parameters
        ----------
        sequence : str
            Amino acid sequence string.

        Returns
        -------
        dict with keys:
            severity (float 0-1), grade (str), action (str),
            classification (str), confidence (float), seq_length (int)
        """
        result = self._validate(sequence)
        if result is not None:
            return result

        seq = sequence.upper().strip()
        E   = self.embedder.embed([seq], verbose=False)
        sev = float(self.analyzer.score(E)[0])

        return self._format(sev, len(seq))

    def score_batch(self, sequences: list[str]) -> list[dict]:
        """
        Score a batch of protein sequences efficiently.

        Parameters
        ----------
        sequences : list of str

        Returns
        -------
        list of dict (same format as score())
        """
        results = []
        valid_idx, valid_seqs = [], []

        for i, seq in enumerate(sequences):
            err = self._validate(seq)
            if err:
                results.append((i, err))
            else:
                valid_idx.append(i)
                valid_seqs.append(seq.upper().strip())

        if valid_seqs:
            E    = self.embedder.embed(valid_seqs, verbose=len(valid_seqs) > 50)
            sevs = self.analyzer.score(E)
            for idx, seq, sev in zip(valid_idx, valid_seqs, sevs):
                results.append((idx, self._format(float(sev), len(seq))))

        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def _validate(self, sequence: str) -> Optional[dict]:
        """Return error dict if invalid, else None."""
        seq = sequence.upper().strip() if sequence else ""
        if not seq:
            return {"error": "Empty sequence", "severity": None}
        if not all(c in VALID_AA for c in seq):
            invalid = [c for c in seq if c not in VALID_AA]
            return {"error": f"Invalid characters: {set(invalid)}", "severity": None}
        if len(seq) < 5:
            return {"error": "Sequence too short (<5aa)", "severity": None}
        return None

    @staticmethod
    def _format(sev: float, seq_len: int) -> dict:
        """Format severity score into output dict."""
        import numpy as np
        from src.svd_severity import GRADE_THRESHOLDS
        grade_int = int(np.digitize(sev, GRADE_THRESHOLDS))
        classification = (
            "threat"    if sev >= 0.6 else
            "ambiguous" if sev >= 0.35 else
            "benign"
        )
        return {
            "severity":       round(sev, 4),
            "grade":          GRADE_LABELS[grade_int],
            "action":         GRADE_ACTIONS[grade_int],
            "classification": classification,
            "confidence":     round(abs(sev - 0.5) * 2, 4),
            "seq_length":     seq_len,
        }
