# Competition Submission: Track 1 — DNA Screening & Synthesis Controls

**Geometric Biosecurity: Continuous Threat Severity Scoring via Spectral
Decomposition of Protein Language Model Embeddings**

*CornerCore AI — April 2026*

---

## Key Result

At 20–40% sequence identity — the AI-redesign evasion zone identified by
Wittmann et al. (*Science*, 2025) — identity-based scoring achieves AP=0.69
while our geometric approach achieves **AP=0.91 (+31.6% improvement)** on
37,974 threat homologs from all 6,329 reviewed UniProt toxins.

---

## Problem

Wittmann et al. demonstrated that ProteinMPNN, EvoDiff-MSA, and EvoDiff-Seq
can generate synthetic homologs of proteins of concern that evade biosecurity
screening software (BSS). After patching, ~3% of structurally plausible
variants still escape detection. Current BSS tools fail when:

1. **Sequences are short (<75aa)** — conotoxins, scorpion toxins, bee venom
   peptides are below commec's effective range (>150bp)
2. **AI redesigns reduce sequence identity** while preserving functional
   structure — the exact scenario the track targets

---

## Method

ESM-2 (150M parameter protein language model) encodes functional and
structural similarity independently of sequence identity. SVD of these
embeddings reveals a dominant threat axis separating toxins from benign
proteins **without any supervision toward biosecurity objectives**.

```
Sequence → ESM-2 (mean-pool) → 640-dim embedding
→ StandardScaler → TruncatedSVD (64 components)
→ Project onto threat axis → Severity score [0,1] → Grade S0–S4
```

---

## Results

### Experiment 1: 179K Sequence Benchmark

- AP = **0.6282** overall (10.4× random baseline)
- AP = **0.83–0.88** at 30–75aa (14–15× baseline) — commec effective range starts at 150bp
- AUROC = 0.9094 (reference only — AP is primary metric for imbalanced data)
- Biologically meaningful taxonomy recovered without supervision

### Experiment 2: Evasion Test (75,948 sequences)

- All 6,329 UniProt toxins × 6 mutation rates (5–85%)
- At **20–40% identity**: SVD AP=0.9076 vs identity AP=0.6896 → **+31.6%**
- Cysteines preserved during mutation (maintains disulfide scaffold)

### Operational Architecture

Stage 2 complement to existing BSS — at threshold 0.65:
- Recall = 68.5%, FPR = 5.8%
- Catches threats in evasion zone that sequence tools miss

---

## Honest Limitations

- BSS baseline is identity threshold, not SecureDNA/commec
- Evasion test uses random mutation, not ProteinMPNN (conservative)
- TM-score structural validation deferred (Lambda Cloud)
- Performance weak at 150–200aa (AP=0.26)

---

## Code & Reproducibility

All data from public UniProt. All models publicly available on HuggingFace.
See [github.com/cornercore-ai/geometric-biosecurity](https://github.com/cornercore-ai/geometric-biosecurity)
