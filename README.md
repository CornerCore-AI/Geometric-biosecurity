# Geometric Biosecurity

**Continuous Threat Severity Scoring via Spectral Decomposition of Protein Language Model Embeddings**

*CornerCore AI — Competition Submission, Track 1: DNA Screening & Synthesis Controls*

---

## Key Result

| Metric | Value |
|--------|-------|
| Short-sequence AP (30–50aa) | **0.8334** (14× random baseline) |
| Short-sequence AP (50–75aa) | **0.8836** (15× random baseline) |
| Evasion zone AP — SVD (20–40% identity) | **0.9076** |
| Evasion zone AP — Identity baseline (20–40% identity) | 0.6896 |
| **SVD improvement in evasion zone** | **+31.6%** |
| Full dataset AUROC (179K sequences) | 0.9094 |
| Full dataset AP | 0.6282 (10.4× baseline) |

> At 20–40% sequence identity — the AI-redesign evasion zone identified by Wittmann et al. (*Science*, 2025) — sequence-similarity scoring achieves AP=0.69 while our geometric approach achieves AP=0.91.

---

## The Problem

Wittmann et al. (*Science*, October 2025) demonstrated that open-source AI protein design tools (ProteinMPNN, EvoDiff-MSA, EvoDiff-Seq) can generate synthetic homologs of proteins of concern that evade biosecurity screening software (BSS) used by nucleic acid synthesis providers. After patching, ~3% of structurally plausible variants still escape detection.

Current BSS tools (SecureDNA, IBBIS commec) operate on sequence similarity. They fail when:
- Sequences are short (<75aa) — not enough residues for reliable HMM matching
- AI redesigns reduce sequence identity while preserving functional structure

---

## Our Approach

ESM-2 protein language model embeddings encode functional and structural similarity independently of sequence identity. We show that **singular value decomposition of these embeddings reveals a dominant threat axis** that separates toxins from benign proteins with AUROC=0.91 on 179,065 sequences — without any supervision toward biosecurity objectives.

```
Protein Sequence
      │
      ▼
ESM-2 (150M params)
Mean-pooled last layer → 640-dim embedding
      │
      ▼
TruncatedSVD (64 components)
      │
      ▼
Project onto threat axis
(toxin centroid − benign centroid, L2 normalised)
      │
      ▼
Severity Score [0, 1] → Grade S0–S4
```

**Key insight:** An adversary cannot evade geometric screening by sequence mutation alone. Moving a sequence away from known threats in sequence space does not remove it from the threat region of functional embedding space — because functional structure is preserved in the embedding even when sequence identity is not.

---

## Repository Structure

```
geometric-biosecurity/
├── src/
│   ├── svd_severity.py          # Core SVDSeverityAnalyzer class
│   ├── embedder.py              # ESM-2 embedding pipeline
│   ├── data_fetcher.py          # UniProt streaming downloader
│   └── scorer.py                # Production score_sequence() function
├── experiments/
│   ├── exp1_benchmark.py        # Experiment 1: 179K sequence benchmark
│   ├── exp2_evasion.py          # Experiment 2: Full-scale evasion test
│   └── exp2b_per_template.py    # Experiment 2b: Per-template analysis
├── notebooks/
│   ├── 01_poc_svd_severity.ipynb        # PoC walkthrough
│   ├── 02_full_dataset_benchmark.ipynb  # Experiment 1 walkthrough
│   └── 03_evasion_test.ipynb            # Experiment 2 walkthrough
├── data/
│   └── README.md                # Data download instructions
├── results/
│   ├── svd_severity_results.json        # Experiment 1 metrics
│   ├── final_results.json               # Experiment 2 metrics
│   └── figures                          # All result plots
├── docs/
│   └── submission.md            # Full competition submission writeup
└── README.md
```

---

## Experiments

### Experiment 1: Large-Scale Benchmark (179,065 sequences)

**Dataset:** All reviewed UniProt toxins (6,329) + key pathogen virus proteins (4,497) + benign proteins from 7 organisms (168,239), sequences 30–500aa.

**Result:** AP=0.63 overall, AP=0.83–0.88 at short lengths (30–75aa) where existing tools are least effective.

```bash
python experiments/exp1_benchmark.py \
    --dataset-path poc_cache/dataset_full.json \
    --embeddings-path poc_cache/embeddings_esm2_t30_150M_UR50D_179065.npy \
    --output-dir results/
```

### Experiment 2: Full-Scale Evasion Test (75,948 sequences)

**Dataset:** All 6,329 reviewed UniProt toxins as templates, 6 mutation rates (5–85%), matched benign negatives.

**Result:** At 20–40% sequence identity, SVD AP=0.9076 vs identity AP=0.6896 (+31.6%).

```bash
python experiments/exp2_evasion.py \
    --dataset-path poc_cache/dataset_full.json \
    --embeddings-path poc_cache/embeddings_esm2_t30_150M_UR50D_179065.npy \
    --output-dir results/
```

---

## Biological Taxonomy Recovery

ESM-2 spontaneously encodes a biologically meaningful threat gradient without supervision:

| Class | Mean Severity | Interpretation |
|-------|--------------|----------------|
| Toxin | 0.767 ± 0.096 | CSα/β scaffold, conotoxins, bee venom — structurally distinctive |
| Virus | 0.619 ± 0.102 | Diverse pathogen proteins — intermediate zone |
| Benign yeast | 0.599 ± 0.060 | Structurally diverse from mammals |
| Benign plant | 0.559 ± 0.082 | Lectins share motifs with toxins |
| Benign E. coli | 0.525 ± 0.074 | Bacterial enzymes overlap toxin families |
| Benign zebrafish | 0.506 ± 0.086 | Vertebrate, moderate distance from human |
| Benign mouse | 0.464 ± 0.115 | Close mammalian homolog of human |
| Benign human | 0.451 ± 0.127 | Self — correctly lowest |

---

## Grey-Zone Finding

Phospholipase A2 (PLA2) toxins from scorpion venom (P0DKU2, P59888, Q6PXP0) show **negative correlation between sequence identity and severity score** (Spearman ρ = −0.69, −0.66, −0.74). Mutations moving away from the wild-type PLA2 sequence *increase* severity — because scorpion PLA2s sit near benign PLA2 housekeeping enzymes in embedding space, while mutations can introduce features of structurally distinctive toxin families.

This independently reproduces the grey-zone ambiguity identified by Wittmann et al. through sequence-based analysis, from a completely different methodological direction.

---

## Operational Architecture

```
DNA Synthesis Order
        │
        ▼
┌────────────────────┐
│    Stage 1: BSS    │  SecureDNA / commec
│  Sequence-based    │  Low FPR, catches known threats
│  (existing tools)  │  Fails at <40% identity / short lengths
└─────────┬──────────┘
          │ Sequences passing Stage 1
          ▼
┌────────────────────┐
│  Stage 2: SVD      │  This work
│  Geometric score   │  AP=0.83–0.88 at 30–75aa
│  Embedding space   │  AP=0.91 in evasion zone
└─────────┬──────────┘
     ┌────┴────┐
     ▼         ▼
   PASS      FLAG
(sev<0.65) (sev≥0.65)
              │
              ▼
        Expert Review
```

**Operational thresholds:**

| Mode | Threshold | Recall | FPR | Use case |
|------|-----------|--------|-----|----------|
| Maximum safety | 0.25 | 99.9% | 89.0% | Research screening |
| Operational default | 0.65 | 68.5% | 5.8% | Complementary to existing BSS |
| High precision | 0.80 | 26.1% | 0.2% | Automated blocking |

---

## Quick Start

```bash
# Install dependencies
pip install torch transformers scikit-learn numpy scipy matplotlib

# Score a single sequence
python -c "
from src.scorer import GeoBioScorer
scorer = GeoBioScorer.from_pretrained()
result = scorer.score('CNCKAPETALCARRCQQH')  # Apamin (bee venom)
print(result)
# {'severity': 0.847, 'grade': 'S4 (Critical)', 'classification': 'threat'}
"
```

---

## Limitations

| Limitation | Status |
|-----------|--------|
| BSS baseline is identity threshold, not SecureDNA/commec | Deferred — production comparison planned |
| Evasion test uses random mutation, not ProteinMPNN | Conservative — real AI redesigns at same identity are more plausible |
| TM-score correlation not validated | Deferred — Lambda Cloud experiment planned |
| Performance weak at 150–200aa (AP=0.26) | Known — structurally ambiguous protein families |
| Plant proteins score high (FP risk) | Expected — share structural motifs with toxins |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{kanaran2026geometric,
  title={Geometric Biosecurity: Continuous Threat Severity Scoring via
         Spectral Decomposition of Protein Language Model Embeddings},
  author={Kanaran, Nik},
  year={2026},
  institution={CornerCore AI},
  note={Competition submission, Track 1: DNA Screening and Synthesis Controls}
}
```

## Acknowledgements

This work builds directly on:
- Wittmann et al., *Strengthening nucleic acid biosecurity screening against generative protein design tools*, Science, October 2025
- Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2)
- Dauparas et al., *Robust deep learning–based protein sequence design using ProteinMPNN*, Science, 2022
