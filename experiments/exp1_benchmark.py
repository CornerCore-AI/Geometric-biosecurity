"""
experiments/exp1_benchmark.py
==============================
Experiment 1: Large-scale SVD severity benchmark on 179K UniProt sequences.

Reproduces the results from the competition submission:
  - AUROC = 0.9094
  - AP = 0.6282 (10.4x baseline)
  - Short-sequence AP = 0.83-0.88 (30-75aa)
  - Biological taxonomy recovery

Usage:
    # Full run (downloads data + embeds — ~3 hours on T4 GPU)
    python experiments/exp1_benchmark.py

    # Use existing cache (fast — minutes)
    python experiments/exp1_benchmark.py --use-cache

    # Custom paths
    python experiments/exp1_benchmark.py \
        --dataset poc_cache/dataset_full.json \
        --embeddings poc_cache/embeddings_esm2_t30_150M_UR50D_179065.npy \
        --output results/
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.svd_severity import SVDSeverityAnalyzer
from src.embedder import ProteinEmbedder
from src.data_fetcher import UniProtFetcher


THREAT_LABELS = {"toxin", "virus"}


def load_or_build_dataset(args) -> tuple[list[dict], np.ndarray, np.ndarray]:
    cache = Path("poc_cache")
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print("Dataset not found. Downloading from UniProt...")
        fetcher = UniProtFetcher(cache_dir=cache)
        data = fetcher.fetch_standard_dataset()
    else:
        data = json.loads(dataset_path.read_text())

    seqs      = [r["seq"] for r in data]
    is_threat = np.array([1 if r["label"] in THREAT_LABELS else 0 for r in data])
    labels    = np.array([r["label"] for r in data])
    lens      = np.array([len(s) for s in seqs])

    print(f"Dataset: {len(data):,} sequences | "
          f"{is_threat.sum():,} threat | {(is_threat==0).sum():,} benign")
    return data, seqs, is_threat, labels, lens


def load_or_build_embeddings(seqs, args) -> np.ndarray:
    emb_path = Path(args.embeddings) if args.embeddings else None

    if emb_path and emb_path.exists():
        E = np.load(emb_path)
        assert E.shape[0] == len(seqs), f"Shape mismatch: {E.shape[0]} vs {len(seqs)}"
        print(f"Loaded embeddings: {E.shape}")
        return E

    print("Embedding sequences with ESM-2...")
    embedder = ProteinEmbedder(model_size="150M", cache_dir="poc_cache")
    return embedder.embed_with_cache(seqs, cache_key=str(len(seqs)))


def run(args):
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────
    data, seqs, is_threat, labels, lens = load_or_build_dataset(args)
    E = load_or_build_embeddings(seqs, args)

    # ── Fit SVD analyzer ──────────────────────────────────────────
    print("\nFitting SVD severity analyzer...")
    analyzer = SVDSeverityAnalyzer(n_components=64)
    analyzer.fit(E, is_threat)

    sev = analyzer.score(E)
    results = analyzer.summary()

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1 RESULTS")
    print(f"{'='*60}")
    print(f"  AUROC:              {results['auroc']:.4f}  (reference only)")
    print(f"  Average Precision:  {results['average_precision']:.4f}  (primary)")
    print(f"  Baseline AP:        {is_threat.mean():.4f}")
    print(f"  AP lift:            {results['average_precision']/is_threat.mean():.1f}x")
    print(f"  Spectral gap:       {results['spectral_gap']:.3f}")
    print(f"  Variance top-5:     {results['variance_top5']:.1%}")

    # ── Per-class severity ────────────────────────────────────────
    print(f"\n  PER-CLASS SEVERITY:")
    print(f"  {'Class':25s}  {'Mean':>6}  {'Std':>6}  {'N':>8}")
    print(f"  {'─'*55}")
    label_arr = np.array(labels)
    for lbl in sorted(set(labels), key=lambda l: -sev[label_arr==l].mean()):
        mask = label_arr == lbl
        print(f"  {lbl:25s}  {sev[mask].mean():>6.3f}  "
              f"{sev[mask].std():>6.3f}  {mask.sum():>8,}")

    # ── Short-sequence AP ─────────────────────────────────────────
    print(f"\n  SHORT-SEQUENCE AP (key competition metric):")
    print(f"  {'Length':10}  {'N threat':>9}  {'N benign':>9}  {'AP':>7}  {'Lift':>6}")
    print(f"  {'─'*50}")
    short_results = []
    for lo, hi in [(30,50),(50,75),(75,100),(100,150),(150,200),(200,300),(300,500)]:
        t_mask = (lens >= lo) & (lens < hi) & (is_threat == 1)
        b_mask = (lens >= lo) & (lens < hi) & (is_threat == 0)
        if t_mask.sum() < 10 or b_mask.sum() < 10: continue
        y_true = np.concatenate([np.ones(t_mask.sum()), np.zeros(b_mask.sum())])
        y_score= np.concatenate([sev[t_mask], sev[b_mask]])
        ap = average_precision_score(y_true, y_score)
        baseline = t_mask.sum() / (t_mask.sum() + b_mask.sum())
        lift = ap / baseline
        short_results.append((lo, hi, t_mask.sum(), b_mask.sum(), ap, lift))
        print(f"  {lo:3d}-{hi:3d}aa:   {t_mask.sum():>9,}  "
              f"{b_mask.sum():>9,}  {ap:>7.4f}  {lift:>5.1f}x")

    # ── Cost-sensitive thresholds ─────────────────────────────────
    print(f"\n  RECALL @ FPR BUDGET:")
    print(f"  {'FPR budget':10}  {'Recall':>8}  {'Flags/1000':>12}")
    print(f"  {'─'*35}")
    fpr_arr, tpr_arr, _ = roc_curve(is_threat, sev)
    for target_fpr in [0.001, 0.01, 0.05, 0.10]:
        idx = np.searchsorted(fpr_arr, target_fpr)
        idx = min(idx, len(tpr_arr)-1)
        print(f"  FPR ≤ {target_fpr:.1%}:    {tpr_arr[idx]:>8.3f}  "
              f"{target_fpr*1000:>12.1f}")

    # ── Save results ──────────────────────────────────────────────
    results["short_sequence"] = [
        {"lo": lo, "hi": hi, "n_threat": int(nt), "n_benign": int(nb),
         "ap": float(ap), "lift": float(lift)}
        for lo, hi, nt, nb, ap, lift in short_results
    ]
    results["per_class"] = {
        lbl: {"mean": float(sev[label_arr==lbl].mean()),
              "std":  float(sev[label_arr==lbl].std()),
              "n":    int((label_arr==lbl).sum())}
        for lbl in set(labels)
    }

    import json as _json
    (out_dir / "exp1_results.json").write_text(_json.dumps(results, indent=2))
    analyzer.save(out_dir / "svd_analyzer.pkl")
    print(f"\n  Saved results to {out_dir}/")

    # ── Save scores for downstream experiments ────────────────────
    np.save(out_dir / "exp1_severity_scores.npy", sev)
    return analyzer, sev, is_threat, lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="poc_cache/dataset_full.json")
    parser.add_argument("--embeddings", default=None,
                        help="Path to .npy embedding cache. Auto-detected if omitted.")
    parser.add_argument("--output",     default="results/")
    parser.add_argument("--use-cache",  action="store_true")
    args = parser.parse_args()

    if not args.embeddings:
        import glob
        candidates = glob.glob("poc_cache/embeddings_esm2_t30_150M_UR50D_*.npy")
        if candidates:
            # Pick the one matching dataset size
            data = json.loads(Path(args.dataset).read_text()) if Path(args.dataset).exists() else []
            n = len(data)
            matched = [c for c in candidates if c.endswith(f"_{n}.npy")]
            args.embeddings = matched[0] if matched else max(candidates, key=lambda c: Path(c).stat().st_size)
            print(f"Auto-detected embeddings: {args.embeddings}")

    run(args)


if __name__ == "__main__":
    main()
