"""
src/data_fetcher.py
===================
UniProt sequence downloader for building the Geometric Biosecurity dataset.

Streams sequences directly from the UniProt REST API in FASTA format
without requiring full downloads. Supports caching to JSON.

Usage:
    from src.data_fetcher import UniProtFetcher

    fetcher = UniProtFetcher(cache_dir="poc_cache")

    # All reviewed toxins (KW-0800)
    toxins = fetcher.fetch(
        query="keyword:KW-0800 AND reviewed:true",
        label="toxin", label_int=2,
        min_len=30, max_len=500
    )

    # Human proteome
    human = fetcher.fetch(
        query="organism_id:9606 AND reviewed:true",
        label="benign_human", label_int=0,
        max_seqs=20000
    )
"""

import json
import time
import requests
from pathlib import Path
from typing import Optional


UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Predefined queries for standard dataset classes
STANDARD_QUERIES = {
    # Threat classes
    "toxin": "keyword:KW-0800 AND reviewed:true",
    "virus_key_pathogens": (
        "(organism_id:11234 OR organism_id:11520 OR organism_id:2697049 "
        "OR organism_id:333760 OR organism_id:11103 OR organism_id:12110 "
        "OR organism_id:10359) AND reviewed:true"
    ),
    # Benign classes (keyword-based for broader coverage)
    "virus_all": "(keyword:KW-1138 OR keyword:KW-0244 OR keyword:KW-0945) AND reviewed:true",
    # Benign organisms
    "benign_human":      "organism_id:9606 AND reviewed:true",
    "benign_mouse":      "organism_id:10090 AND reviewed:true",
    "benign_ecoli":      "organism_id:83333 AND reviewed:true",
    "benign_yeast":      "organism_id:559292 AND reviewed:true",
    "benign_plant":      "organism_id:3702 AND reviewed:true",
    "benign_zebrafish":  "organism_id:7955 AND reviewed:true",
    "benign_human_unrev": "organism_id:9606 AND reviewed:false",
}


class UniProtFetcher:
    """
    Streams protein sequences from UniProt REST API.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for JSON caches. Fetches are cached by label name.
    min_len : int
        Minimum sequence length (default 30).
    max_len : int
        Maximum sequence length (default 500).
    """

    def __init__(
        self,
        cache_dir: str | Path = "poc_cache",
        min_len: int = 30,
        max_len: int = 500,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_len = min_len
        self.max_len = max_len

    def fetch(
        self,
        query: str,
        label: str,
        label_int: int,
        max_seqs: Optional[int] = None,
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Fetch sequences from UniProt.

        Parameters
        ----------
        query : str
            UniProt query string.
        label : str
            Class label string (e.g. "toxin", "benign_human").
        label_int : int
            Integer label (0=benign, 1=virus, 2=toxin).
        max_seqs : int, optional
            Maximum sequences to return. None = fetch all.
        force_refresh : bool
            Ignore cache and re-fetch.

        Returns
        -------
        list of dict
            Each dict has keys: id, seq, label, label_int.
        """
        cache_path = self.cache_dir / f"up_{label}.json"

        if cache_path.exists() and not force_refresh:
            data = json.loads(cache_path.read_text())
            n = len(data) if max_seqs is None else min(len(data), max_seqs)
            print(f"  [cache] {label}: {len(data):,} sequences")
            return data[:n] if max_seqs else data

        print(f"  Fetching {label}...", flush=True)
        t0 = time.time()

        try:
            resp = requests.get(
                UNIPROT_STREAM,
                params={"query": query, "format": "fasta", "compressed": "false"},
                stream=True,
                timeout=180,
                headers={"User-Agent": "CornerCore-GeoBio/1.0 (biosecurity-research)"},
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ERROR fetching {label}: {e}")
            return []

        records, acc, seq_lines = [], "", []

        for raw in resp.iter_lines(decode_unicode=True):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if acc and seq_lines:
                    seq = "".join(seq_lines)
                    if self._valid(seq):
                        records.append({
                            "id": acc, "seq": seq,
                            "label": label, "label_int": label_int
                        })
                        if len(records) % 10_000 == 0:
                            print(f"    {label}: {len(records):,}", end="\r", flush=True)
                        if max_seqs and len(records) >= max_seqs:
                            break
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) > 1 else line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)

        # Last record
        if acc and seq_lines and (not max_seqs or len(records) < max_seqs):
            seq = "".join(seq_lines)
            if self._valid(seq):
                records.append({"id": acc, "seq": seq,
                                 "label": label, "label_int": label_int})

        elapsed = time.time() - t0
        print(f"  {label}: {len(records):,} sequences in {elapsed:.1f}s")
        cache_path.write_text(json.dumps(records))
        return records

    def fetch_standard_dataset(
        self,
        target_total: int = 200_000,
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Fetch the standard Geometric Biosecurity training dataset.

        Fetches all reviewed toxins + key pathogen virus proteins +
        benign proteins from 7 organisms. Pads with unreviewed human
        entries if total is below target.

        Returns
        -------
        list of dict, deduplicated and shuffled.
        """
        import random

        all_seqs = []

        # Threat classes
        all_seqs += self.fetch("keyword:KW-0800 AND reviewed:true",
                               "toxin", 2, force_refresh=force_refresh)
        all_seqs += self.fetch(STANDARD_QUERIES["virus_key_pathogens"],
                               "virus", 1, max_seqs=50_000,
                               force_refresh=force_refresh)

        # Benign classes
        for label, organism_id in [
            ("benign_human",     "9606"),
            ("benign_mouse",     "10090"),
            ("benign_plant",     "3702"),
            ("benign_yeast",     "559292"),
            ("benign_ecoli",     "83333"),
            ("benign_zebrafish", "7955"),
        ]:
            all_seqs += self.fetch(
                f"organism_id:{organism_id} AND reviewed:true",
                label, 0, force_refresh=force_refresh
            )

        # Pad with unreviewed human if needed
        if len(all_seqs) < target_total:
            needed = target_total - len(all_seqs)
            print(f"\n  Need {needed:,} more sequences — fetching unreviewed human...")
            all_seqs += self.fetch(
                "organism_id:9606 AND reviewed:false",
                "benign_human_unrev", 0, max_seqs=needed,
                force_refresh=force_refresh
            )

        # Deduplicate
        seen, unique = set(), []
        for r in all_seqs:
            if r["seq"] not in seen:
                seen.add(r["seq"])
                unique.append(r)

        random.shuffle(unique)

        # Print summary
        from collections import Counter
        counts = Counter(r["label"] for r in unique)
        print(f"\n{'='*50}")
        print(f"  DATASET SUMMARY")
        print(f"{'='*50}")
        for label, n in sorted(counts.items()):
            print(f"  {label:25s}: {n:>8,}")
        print(f"{'─'*50}")
        print(f"  {'TOTAL':25s}: {len(unique):>8,}")
        print(f"{'='*50}")

        # Save combined dataset
        out_path = self.cache_dir / "dataset_full.json"
        out_path.write_text(json.dumps(unique))
        print(f"\n  Saved to {out_path}")
        return unique

    def _valid(self, seq: str) -> bool:
        """Check sequence length and alphabet."""
        return (self.min_len <= len(seq) <= self.max_len
                and set(seq) <= VALID_AA)
