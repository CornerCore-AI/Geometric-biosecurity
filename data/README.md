# Data

Raw sequence data is **not committed to this repository** (files are large and
publicly available). All data is fetched from UniProt on first run.

## Automatic Download

The fastest way to get the full dataset is:

```python
from src.data_fetcher import UniProtFetcher

fetcher = UniProtFetcher(cache_dir="poc_cache")
data = fetcher.fetch_standard_dataset(target_total=200_000)
# Saves to poc_cache/dataset_full.json
```

Or run the experiment script which handles this automatically:

```bash
python experiments/exp1_benchmark.py
```

## Manual Download

Individual class queries:

```python
fetcher = UniProtFetcher(cache_dir="poc_cache")

# All reviewed toxins (~6,300 sequences)
toxins = fetcher.fetch(
    query="keyword:KW-0800 AND reviewed:true",
    label="toxin", label_int=2
)

# Human proteome reviewed
human = fetcher.fetch(
    query="organism_id:9606 AND reviewed:true",
    label="benign_human", label_int=0
)
```

## Dataset Statistics (as published)

| Class | N sequences | Source |
|-------|------------|--------|
| toxin | 6,329 | UniProt KW-0800, reviewed |
| virus | 4,497 | Key pathogen reviewed entries |
| benign_human | 11,894 | UniProt 9606, reviewed |
| benign_human_unrev | 125,224 | UniProt 9606, unreviewed |
| benign_mouse | 9,840 | UniProt 10090, reviewed |
| benign_plant | 10,907 | UniProt 3702, reviewed |
| benign_yeast | 4,391 | UniProt 559292, reviewed |
| benign_ecoli | 3,771 | UniProt 83333, reviewed |
| benign_zebrafish | 2,212 | UniProt 7955, reviewed |
| **TOTAL** | **179,065** | |

## Embeddings

Pre-computed ESM-2 embeddings are too large for GitHub (~440MB). Regenerate with:

```bash
python experiments/exp1_benchmark.py --dataset poc_cache/dataset_full.json
```

Embedding time: ~175 minutes on a T4 GPU (16GB) at 17 sequences/second.
