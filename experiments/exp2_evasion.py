"""
experiments/exp2_evasion.py
============================
Experiment 2: Full-scale evasion test on 75,948 sequences.

Generates homologs of all 6,329 reviewed UniProt toxins at 6 mutation
rates (5–85% residue substitution) and compares SVD severity against
identity-to-reference scoring across the full sequence identity range.

Key result:
  At 20-40% identity (AI-redesign evasion zone):
    Identity-score AP = 0.6896
    SVD severity AP   = 0.9076
    SVD improvement   = +31.6%

Usage:
    python experiments/exp2_evasion.py \
        --dataset poc_cache/dataset_full.json \
        --analyzer results/svd_analyzer.pkl \
        --output results/

    # Skip embedding (use cache)
    python experiments/exp2_evasion.py --use-cache
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.svd_severity import SVDSeverityAnalyzer
from src.embedder import ProteinEmbedder


VALID_AA    = list("ACDEFGHIKLMNPQRSTVWY")
THREAT_LABELS = {"toxin", "virus"}
MUTATION_RATES = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85]


def seq_id(s1: str, s2: str) -> float:
    n = min(len(s1), len(s2))
    return sum(a == b for a, b in zip(s1, s2)) / max(len(s1), len(s2))


def mutate(seq: str, rate: float, seed: int = 42) -> str:
    """
    Structured random mutation preserving cysteine positions
    (cysteines maintain disulfide bonds critical for toxin fold).
    """
    rng = random.Random(seed)
    cysteines = {i for i, aa in enumerate(seq) if aa == "C"}
    return "".join(
        aa if i in cysteines or rng.random() >= rate
        else rng.choice([a for a in VALID_AA if a != aa])
        for i, aa in enumerate(seq)
    )


def max_ref_identity(seq: str, ref_panel: list[str]) -> float:
    return max(seq_id(seq, r) for r in ref_panel)


def run(args):
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────
    print("Loading dataset...")
    data_full   = json.loads(Path(args.dataset).read_text())
    toxin_seqs  = [r["seq"] for r in data_full if r["label"] == "toxin"]
    benign_seqs = [r["seq"] for r in data_full if r["label"] not in THREAT_LABELS]
    print(f"  Toxins: {len(toxin_seqs):,}  Benign: {len(benign_seqs):,}")

    # ── Load SVD analyzer ─────────────────────────────────────────
    analyzer_path = Path(args.analyzer)
    if analyzer_path.exists():
        analyzer = SVDSeverityAnalyzer.load(analyzer_path)
        print(f"Loaded analyzer: AUROC={analyzer.auroc():.4f}")
    else:
        raise FileNotFoundError(
            f"Analyzer not found at {analyzer_path}. Run exp1_benchmark.py first."
        )

    # ── Build reference panel for identity scoring ────────────────
    random.seed(42)
    ref_panel = random.sample(toxin_seqs, min(200, len(toxin_seqs)))
    print(f"BSS reference panel: {len(ref_panel)} sequences")

    # ── Generate homologs ─────────────────────────────────────────
    evasion_cache = out_dir / "exp2_records.json"
    if evasion_cache.exists() and not args.force_rebuild:
        print("Loading cached homolog records...")
        records = json.loads(evasion_cache.read_text())
    else:
        print(f"\nGenerating {len(toxin_seqs):,} × {len(MUTATION_RATES)} = "
              f"{len(toxin_seqs)*len(MUTATION_RATES):,} threat homologs...")
        t0 = time.time()
        records = []
        for i, wt in enumerate(toxin_seqs):
            for j, rate in enumerate(MUTATION_RATES):
                mut = mutate(wt, rate, seed=i * 10 + j)
                records.append({
                    "seq": mut, "wt": wt,
                    "mutation_rate": rate,
                    "seq_identity": seq_id(mut, wt),
                    "label_int": 1,
                })
            if (i + 1) % 1000 == 0:
                print(f"  {i+1:,}/{len(toxin_seqs):,}", end="\r", flush=True)
        print(f"\n  Generated {len(records):,} homologs in {time.time()-t0:.1f}s")

        # Add benign negatives
        n_threat = len(records)
        benign_sample = random.sample(benign_seqs, min(n_threat, len(benign_seqs)))
        for seq in benign_sample:
            records.append({"seq": seq, "wt": None, "mutation_rate": None,
                            "seq_identity": None, "label_int": 0})
        print(f"  Added {len(benign_sample):,} benign negatives")
        print(f"  Total: {len(records):,}")

        evasion_cache.write_text(json.dumps(records))

    # ── Compute identity scores ───────────────────────────────────
    print("\nComputing identity-to-reference scores...")
    t0 = time.time()
    ref_id_arr = []
    for i, r in enumerate(records):
        ref_id_arr.append(max_ref_identity(r["seq"], ref_panel))
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(records) - i - 1)
            print(f"  {i+1:,}/{len(records):,}  ETA {eta/60:.1f}min", end="\r", flush=True)
    ref_id_arr = np.array(ref_id_arr)
    print(f"\n  Done in {(time.time()-t0)/60:.1f}min")

    # ── Embed all sequences ───────────────────────────────────────
    emb_cache_path = out_dir / f"exp2_embeddings_{len(records)}.npy"
    if emb_cache_path.exists() and not args.force_rebuild:
        print(f"Loading cached embeddings: {emb_cache_path}")
        E_all = np.load(emb_cache_path)
    else:
        print(f"\nEmbedding {len(records):,} sequences...")
        embedder = ProteinEmbedder(model_size="150M", batch_size=16)
        E_all = embedder.embed([r["seq"] for r in records])
        np.save(emb_cache_path, E_all)

    # ── Score ─────────────────────────────────────────────────────
    print("Scoring with SVD severity...")
    sev_arr    = analyzer.score(E_all)
    labels_all = np.array([r["label_int"] for r in records])
    mut_id_arr = np.array([r["seq_identity"] if r["seq_identity"] else 1.0
                            for r in records])

    # ── Metrics ───────────────────────────────────────────────────
    baseline    = labels_all.mean()
    ap_identity = average_precision_score(labels_all, ref_id_arr)
    ap_svd      = average_precision_score(labels_all, sev_arr)

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT 2 RESULTS")
    print(f"{'='*65}")
    print(f"  Total sequences:          {len(records):,}")
    print(f"  Threat homologs:          {(labels_all==1).sum():,}")
    print(f"  Benign negatives:         {(labels_all==0).sum():,}")
    print(f"  Baseline AP (random):     {baseline:.4f}")
    print(f"  Identity-to-ref AP:       {ap_identity:.4f}  ({ap_identity/baseline:.1f}x)")
    print(f"  SVD severity AP:          {ap_svd:.4f}  ({ap_svd/baseline:.1f}x)")
    print(f"  SVD improvement:          {(ap_svd-ap_identity)/ap_identity:+.1%}")

    print(f"\n  BY SEQUENCE IDENTITY TO WT:")
    print(f"  {'ID range':12}  {'N':>8}  {'ID-AP':>7}  {'SVD-AP':>8}  {'Lift':>7}  {'Winner'}")
    print(f"  {'─'*58}")

    bin_results = []
    for lo, hi in [(0.8,1.0),(0.6,0.8),(0.4,0.6),(0.2,0.4),(0.0,0.2)]:
        t_mask   = (mut_id_arr >= lo) & (mut_id_arr < hi) & (labels_all == 1)
        combined = t_mask | (labels_all == 0)
        if t_mask.sum() < 10: continue
        y_true = labels_all[combined]
        try:
            ap_id  = average_precision_score(y_true, ref_id_arr[combined])
            ap_sv  = average_precision_score(y_true, sev_arr[combined])
            lift   = (ap_sv - ap_id) / (ap_id + 1e-8)
            winner = "SVD ✓" if ap_sv > ap_id else "ID  ✓"
            bin_results.append((lo, hi, int(t_mask.sum()), ap_id, ap_sv, lift))
            marker = " ← EVASION ZONE" if 0.2 <= lo < 0.4 else ""
            print(f"  {lo:.1f}-{hi:.1f} id:   {t_mask.sum():>8,}  "
                  f"{ap_id:>7.4f}  {ap_sv:>8.4f}  {lift:>+7.1%}  {winner}{marker}")
        except Exception as e:
            print(f"  {lo:.1f}-{hi:.1f}: error ({e})")

    # ── Save ──────────────────────────────────────────────────────
    final = {
        "n_total":      len(records),
        "n_threat":     int((labels_all==1).sum()),
        "n_benign":     int((labels_all==0).sum()),
        "baseline_ap":  float(baseline),
        "identity_ap":  float(ap_identity),
        "svd_ap":       float(ap_svd),
        "svd_improvement_pct": float((ap_svd-ap_identity)/ap_identity*100),
        "by_identity_bin": [
            {"lo": lo, "hi": hi, "n_threat": n,
             "identity_ap": float(ap_id), "svd_ap": float(ap_sv),
             "lift_pct": float(lift*100)}
            for lo, hi, n, ap_id, ap_sv, lift in bin_results
        ],
    }
    import json as _json
    (out_dir / "exp2_results.json").write_text(_json.dumps(final, indent=2))
    print(f"\n  Saved results to {out_dir}/exp2_results.json")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       default="poc_cache/dataset_full.json")
    parser.add_argument("--analyzer",      default="results/svd_analyzer.pkl")
    parser.add_argument("--output",        default="results/")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
